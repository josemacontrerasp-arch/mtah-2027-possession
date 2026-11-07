"""NOT PART OF THE SUBMISSION CHAIN -- tried and dropped (second-stage stacking). Kept for transparency; solution/predict.py never
imports it.

Second stage: re-estimate each frame from the first stage's probabilities in context.

The first stage sees +/- 8 s of hand-built context. Possessions last ~10 s on average and their
edges are where the score is decided, so the second stage looks at windowed summaries of the
first-stage probabilities (up to +/- 20 s, past and future separately, and their differences --
which peak at a change of possession). It is trained on the first stage's out-of-fold
probabilities for the external games (classic stacking), so it never sees a probability the first
stage produced for a game it was trained on -- and never sees the provided match.

    python -m solution.stack cv       # LOMO over external games -> cache/oof2/<mid>.npz
    python -m solution.stack final    # fit on all -> models/stack_lgbm.txt
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from solution.external_dfl import CACHE_DIR, MATCH_IDS  # noqa: E402
from solution.features import _nan_window_mean  # noqa: E402
from solution.match import X  # noqa: E402

MODEL_DIR = os.path.join(HERE, "models")
WINDOWS = {"p1": (-25, -1), "f1": (1, 25), "p3": (-75, -1), "f3": (1, 75), "p8": (-200, -1),
           "f8": (1, 200), "p20": (-500, -1), "f20": (1, 500), "c4": (-100, 100)}
STEP = 5
PARAMS = dict(objective="multiclass", num_class=3, learning_rate=0.05, num_leaves=31,
              min_data_in_leaf=300, feature_fraction=0.8, bagging_fraction=0.7, bagging_freq=1,
              lambda_l2=1.0, verbose=-1, num_threads=16, seed=7, deterministic=True,
              force_row_wise=True)
ROUNDS = 300


def blocks_of(half):
    out = []
    for h in (1, 2):
        idx = np.flatnonzero(half == h)
        if len(idx):
            out.append((int(idx[0]), int(idx[-1]) + 1))
    return out


def stack_features(prob, half):
    """(n, k) feature table for frames inside the halves; NaN rows elsewhere."""
    n = len(half)
    cols = {}
    names = ("h", "a", "d")
    for c in range(3):
        cols["p_" + names[c]] = np.full(n, np.nan)
        for w in WINDOWS:
            cols["p_{}_{}".format(names[c], w)] = np.full(n, np.nan)
        for w in ("1", "3", "8"):
            cols["dlt_{}_{}".format(names[c], w)] = np.full(n, np.nan)
    cols["entropy"] = np.full(n, np.nan)
    cols["entropy_c2"] = np.full(n, np.nan)
    for a, b in blocks_of(half):
        p = prob[a:b].astype(np.float64)
        ent = -np.sum(p * np.log(np.clip(p, 1e-6, 1)), axis=1)
        cols["entropy"][a:b] = ent
        cols["entropy_c2"][a:b] = _nan_window_mean(ent, -50, 50)
        for c in range(3):
            x = p[:, c]
            cols["p_" + names[c]][a:b] = x
            win = {}
            for w, (lo, hi) in WINDOWS.items():
                win[w] = _nan_window_mean(x, lo, hi)
                cols["p_{}_{}".format(names[c], w)][a:b] = win[w]
            for w in ("1", "3", "8"):
                cols["dlt_{}_{}".format(names[c], w)][a:b] = win["f" + w] - win["p" + w]
    return pd.DataFrame(cols).astype(np.float32)


def _load_oof(mid):
    z = np.load(os.path.join(CACHE_DIR, "oof", "{}.npz".format(mid)))
    return z["prob"], z["labels"], z["half"]


def _rows(mid):
    prob, labels, half = _load_oof(mid)
    F = stack_features(np.nan_to_num(prob, nan=1.0 / 3), half)
    keep = (half > 0) & (labels != X) & ((np.arange(len(half)) % STEP) == 0)
    F = F[keep].copy()
    F["y"] = labels[keep]
    return F


def fit(df):
    import lightgbm as lgb
    cols = [c for c in df.columns if c != "y"]
    return lgb.train(PARAMS, lgb.Dataset(df[cols], label=df["y"].values),
                     num_boost_round=ROUNDS), cols


def cmd_cv():
    out_dir = os.path.join(CACHE_DIR, "oof2")
    os.makedirs(out_dir, exist_ok=True)
    rows = {mid: _rows(mid) for mid in MATCH_IDS}
    for mid in MATCH_IDS:
        model, cols = fit(pd.concat([rows[m] for m in MATCH_IDS if m != mid]))
        prob, labels, half = _load_oof(mid)
        F = stack_features(np.nan_to_num(prob, nan=1.0 / 3), half)
        p2 = np.full_like(prob, np.nan)
        inside = half > 0
        p2[inside] = model.predict(F[inside][cols])
        np.savez_compressed(os.path.join(out_dir, "{}.npz".format(mid)), prob=p2, labels=labels,
                            half=half)
        print("stage-2 oof written for", mid)


def cmd_final():
    os.makedirs(MODEL_DIR, exist_ok=True)
    model, cols = fit(pd.concat([_rows(m) for m in MATCH_IDS]))
    model.save_model(os.path.join(MODEL_DIR, "stack_lgbm.txt"))
    print("saved stage-2 model with", len(cols), "features")


def apply(prob, half):
    import lightgbm as lgb
    model = lgb.Booster(model_file=os.path.join(MODEL_DIR, "stack_lgbm.txt"))
    F = stack_features(np.nan_to_num(prob, nan=1.0 / 3), half)
    out = np.full_like(prob, np.nan)
    inside = half > 0
    out[inside] = model.predict(F[inside])
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cv", "final"])
    a = ap.parse_args()
    cmd_cv() if a.cmd == "cv" else cmd_final()
