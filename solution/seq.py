"""Sequence model: a bidirectional GRU over per-frame features, trained on external games only.

LightGBM sees each frame through fixed +/- 8 s windows. Possessions run ~10 s and the hard cases
(open-play turnovers, long ball gaps) need wider, adaptive context, so this model reads two-minute
stretches at 5 Hz in both directions and labels every step. Same features, same degraded external
games, same leave-one-match-out protocol as the frame model -- its out-of-fold probabilities are
directly comparable and can be averaged with LightGBM's.

    python -m solution.seq cv       # LOMO -> external/dfl/cache/oof_seq/<mid>.npz
    python -m solution.seq final    # fit on all external games -> models/seq_gru.pt
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from solution.match import X  # noqa: E402

# Prediction (load_final / predict_full) must not import the training stack -- in particular
# not the calibrated simulator in solution/degrade.py -- so training-only imports live inside the
# training functions below. These two constants mirror solution/train.py (checked in cmd_cv).
EXCLUDE = {"b_speed_prov", "half_t"}
STEP = 5                   # 5 Hz, = train.TRAIN_STEP

MODEL_DIR = os.path.join(HERE, "models")
WIN = 600                  # steps per training window (120 s)
HIDDEN = int(os.environ.get("SEQ_HIDDEN", 128))
LAYERS = 2
EPOCHS = int(os.environ.get("SEQ_EPOCHS", 14))
BATCH = 32
LR = 2e-3
SEED = int(os.environ.get("SEQ_SEED", 7))
BASE_ONLY = os.environ.get("SEQ_BASE_ONLY", "0") == "1"   # drop pre-windowed columns
OOF_DIR = os.environ.get("SEQ_OOF", "oof_seq")
NAME = os.environ.get("SEQ_NAME", "seq_gru")
WINDOW_SUFFIXES = ("_p1", "_f1", "_p3", "_f3", "_p8", "_f8", "_c2")


def _torch():
    import torch
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return torch


class Norm:
    """Per-feature standardisation fitted on training rows; NaN -> 0 plus a missing flag."""

    def __init__(self, cols):
        self.cols = cols

    def fit(self, df):
        x = df[self.cols].values.astype(np.float64)
        self.mu = np.nanmean(x, axis=0)
        sd = np.nanstd(x, axis=0)
        self.sd = np.where(sd > 1e-6, sd, 1.0)
        self.mask_cols = [i for i in range(len(self.cols)) if np.isnan(x[:, i]).mean() > 0.01]
        return self

    def transform(self, df):
        x = df[self.cols].values.astype(np.float64)
        miss = np.isnan(x[:, self.mask_cols]).astype(np.float32)
        z = (x - self.mu) / self.sd
        z = np.clip(np.nan_to_num(z, nan=0.0), -6, 6).astype(np.float32)
        return np.concatenate([z, miss], axis=1)

    def state(self):
        return dict(cols=self.cols, mu=self.mu.tolist(), sd=self.sd.tolist(),
                    mask_cols=self.mask_cols)

    @classmethod
    def from_state(cls, s):
        n = cls(s["cols"])
        n.mu, n.sd, n.mask_cols = np.array(s["mu"]), np.array(s["sd"]), s["mask_cols"]
        return n


def _sequences(df):
    """Split a 5 Hz table into contiguous per-half sequences (frames are already sorted)."""
    out = []
    for h in (1, 2):
        part = df[df["half"] == h]
        if len(part):
            out.append(part)
    return out


def _train_tables(mids):
    from solution.train import TRAIN_SEEDS, variant_table
    seqs = []
    for mid in mids:
        for swapped in (False, True):
            for seed in TRAIN_SEEDS:
                t = variant_table(mid, swapped, seed, STEP)
                seqs.extend(_sequences(t))
    return seqs


def build_model(torch, n_in):
    nn = torch.nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Sequential(nn.Linear(n_in, HIDDEN), nn.GELU(), nn.Dropout(0.1))
            self.gru = nn.GRU(HIDDEN, HIDDEN, num_layers=LAYERS, batch_first=True,
                              bidirectional=True, dropout=0.1)
            self.out = nn.Sequential(nn.Linear(2 * HIDDEN, HIDDEN), nn.GELU(),
                                     nn.Linear(HIDDEN, 3))

        def forward(self, x):
            h, _ = self.gru(self.inp(x))
            return self.out(h)

    return Net()


def fit(seqs, norm, device="cuda", epochs=EPOCHS, log=True):
    torch = _torch()
    rng = np.random.default_rng(SEED)
    X = [norm.transform(s) for s in seqs]
    Y = [s["y"].values.astype(np.int64) for s in seqs]
    model = build_model(torch, X[0].shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    total = sum(len(x) for x in X)
    steps_per_epoch = max(1, total // (WIN * BATCH) * 2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=epochs *
                                                steps_per_epoch)
    lens = np.array([len(x) for x in X], dtype=np.float64)
    pick_p = lens / lens.sum()
    lossf = torch.nn.CrossEntropyLoss()
    model.train()
    for ep in range(epochs):
        t0, tot = time.time(), 0.0
        for _ in range(steps_per_epoch):
            xb, yb = [], []
            for _ in range(BATCH):
                i = rng.choice(len(X), p=pick_p)
                L = len(X[i])
                s = int(rng.integers(0, max(1, L - WIN)))
                xs, ys = X[i][s:s + WIN], Y[i][s:s + WIN]
                if len(xs) < WIN:
                    pad = WIN - len(xs)
                    xs = np.pad(xs, ((0, pad), (0, 0)))
                    ys = np.pad(ys, (0, pad), constant_values=-100)
                xb.append(xs)
                yb.append(ys)
            xb = torch.from_numpy(np.stack(xb)).to(device)
            yb = torch.from_numpy(np.stack(yb)).to(device)
            logits = model(xb)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 3), yb.reshape(-1),
                                                     ignore_index=-100)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss)
        if log:
            print("  epoch {:>2}  loss {:.4f}  ({:.0f}s)".format(ep + 1, tot / steps_per_epoch,
                                                               time.time() - t0), flush=True)
    model.eval()
    return model


def predict_full(model, norm, F, half, n_total, device="cuda"):
    """Probabilities at 25 fps: run the GRU on each half at 5 Hz (all 5 phase offsets, averaged)
    and interpolate back to every frame."""
    torch = _torch()
    prob = np.full((n_total, 3), np.nan, np.float32)
    idx_all = F.index.values
    for h in (1, 2):
        sel = F["half"].values == h
        if not sel.any():
            continue
        frames = idx_all[sel]
        Fh = F[sel]
        acc = np.zeros((len(frames), 3))
        cnt = np.zeros(len(frames))
        for off in range(STEP):
            sub = np.arange(off, len(frames), STEP)
            x = torch.from_numpy(norm.transform(Fh.iloc[sub])[None]).to(device)
            with torch.no_grad():
                p = torch.softmax(model(x)[0], dim=-1).cpu().numpy()
            for c in range(3):
                acc[:, c] += np.interp(np.arange(len(frames)), sub, p[:, c])
            cnt += 1
        prob[frames] = (acc / cnt[:, None]).astype(np.float32)
    return prob


def feature_cols(df):
    cols = [c for c in df.columns if c not in EXCLUDE | {"y", "frame", "mid"}]
    if BASE_ONLY:
        cols = [c for c in cols if not c.endswith(WINDOW_SUFFIXES)]
    return cols


def cmd_cv():
    from solution.external_dfl import CACHE_DIR, MATCH_IDS
    from solution.train import EXCLUDE as TRAIN_EXCLUDE, TRAIN_STEP, eval_arrays
    assert TRAIN_EXCLUDE == EXCLUDE and TRAIN_STEP == STEP
    out_dir = os.path.join(CACHE_DIR, OOF_DIR)
    os.makedirs(out_dir, exist_ok=True)
    all_seqs = {mid: _train_tables([mid]) for mid in MATCH_IDS}
    cols = feature_cols(all_seqs[MATCH_IDS[0]][0])
    for mid in MATCH_IDS:
        t0 = time.time()
        train = [s for m_ in MATCH_IDS if m_ != mid for s in all_seqs[m_]]
        norm = Norm(cols).fit(pd.concat(train))
        model = fit(train, norm, log=False)
        m, F = eval_arrays(mid)
        prob = predict_full(model, norm, F, m.half, m.n)
        np.savez_compressed(os.path.join(out_dir, "{}.npz".format(mid)), prob=prob,
                            labels=m.labels, half=m.half)
        if os.environ.get("MTAH_STRESS", "0") == "1":
            mh, Fh = eval_arrays(mid, hard=True)
            ph = predict_full(model, norm, Fh, mh.half, mh.n)
            os.makedirs(out_dir + "_hard", exist_ok=True)
            np.savez_compressed(os.path.join(out_dir + "_hard", "{}.npz".format(mid)), prob=ph,
                                labels=mh.labels, half=mh.half)
        for prof in [e for e in os.environ.get("MTAH_EVALS", "").split(",") if e]:
            mr, Fr = eval_arrays(mid, profile=prof)
            pr_ = predict_full(model, norm, Fr, mr.half, mr.n)
            os.makedirs(out_dir + "_" + prof, exist_ok=True)
            np.savez_compressed(os.path.join(out_dir + "_" + prof, "{}.npz".format(mid)),
                                prob=pr_, labels=mr.labels, half=mr.half)
        ok = np.isfinite(prob[:, 0]) & (m.labels != X)
        acc = float(np.mean(np.argmax(prob[ok], axis=1) == m.labels[ok]))
        print("{}  seq oof written  frame acc {:.4f}  ({:.0f}s)".format(mid, acc, time.time() - t0),
              flush=True)


def cmd_final():
    from solution.external_dfl import MATCH_IDS
    torch = _torch()
    seqs = _train_tables(MATCH_IDS)
    cols = feature_cols(seqs[0])
    norm = Norm(cols).fit(pd.concat(seqs))
    model = fit(seqs, norm)
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, NAME + ".pt"))
    with open(os.path.join(MODEL_DIR, NAME + "_norm.json"), "w") as fh:
        json.dump(dict(norm.state(), hidden=HIDDEN, base_only=BASE_ONLY), fh)
    print("saved seq model;", len(cols), "features")


def load_final(device="cuda", name=None):
    torch = _torch()
    name = name or NAME
    state = json.load(open(os.path.join(MODEL_DIR, name + "_norm.json")))
    norm = Norm.from_state(state)
    n_in = len(norm.cols) + len(norm.mask_cols)
    model = build_model(torch, n_in).to(device)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, name + ".pt"), map_location=device))
    model.eval()
    return model, norm


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cv", "final"])
    a = ap.parse_args()
    cmd_cv() if a.cmd == "cv" else cmd_final()
    # torch's CUDA teardown on Windows can return a spurious non-zero code after all work is
    # done and flushed; exit explicitly so shell chains behave.
    sys.stdout.flush()
    os._exit(0)
