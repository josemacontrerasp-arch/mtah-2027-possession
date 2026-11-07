"""Produce the submission for the provided match from the trained models and tuned decoder.

    python -m solution.predict --name <team_name>                  # the submission
    python -m solution.predict --name test --config other.json     # another pipeline config

Everything it loads was fitted on external games (solution/models/). It prints aggregates only
(label counts, number and mean length of possessions) and never reads the ground truth.
Scoring the public window is a separate, explicit pipeline check -- never a selection step.
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from games import GAME, GAMES  # noqa: E402
from solution.atd import load_atd  # noqa: E402
from solution.decode import decode  # noqa: E402
from solution.features import compute_features  # noqa: E402
from solution.match import LABEL_CHARS, H, A, D, X  # noqa: E402

MODEL_DIR = os.path.join(HERE, "models")
OUT_DIR = os.path.join(ROOT, "submissions")


def lgbm_probs(F, n, model_file="frame_lgbm.txt", features_file="frame_lgbm_features.json"):
    import lightgbm as lgb
    model = lgb.Booster(model_file=os.path.join(MODEL_DIR, model_file))
    cols = json.load(open(os.path.join(MODEL_DIR, features_file)))
    prob = np.full((n, 3), np.nan, np.float32)
    prob[F.index.values] = model.predict(F[cols]).astype(np.float32)
    return prob


def seq_probs(F, m, name, device="cpu"):
    """GRU probabilities on CPU with a single thread, so the arithmetic (and its rounding) does not
    depend on how many cores the machine has."""
    import torch
    torch.set_num_threads(1)
    from solution.seq import load_final, predict_full
    model, norm = load_final(device, name)
    return predict_full(model, norm, F, m.half, m.n, device=device)


def load_pipeline():
    path = os.path.join(MODEL_DIR, "pipeline.json")
    if not os.path.isfile(path):
        raise SystemExit("missing {} -- the pipeline configuration".format(path))
    return json.load(open(path))




def write_submission(path, codes):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    chars = LABEL_CHARS[codes]
    with open(path, "w", newline="\n") as fh:
        fh.write("frame,label\n")
        fh.writelines("{},{}\n".format(i, c) for i, c in enumerate(chars))


def describe(codes, half):
    out = {k: int(np.sum(codes == v)) for k, v in (("H", H), ("A", A), ("D", D), ("X", X))}
    lengths = []
    for h in (1, 2):
        idx = np.flatnonzero(half == h)
        seg = codes[idx[0]:idx[-1] + 1]
        cut = np.flatnonzero(np.diff(seg) != 0) + 1
        bounds = np.concatenate(([0], cut, [len(seg)]))
        lengths += [b - a for a, b in zip(bounds[:-1], bounds[1:]) if seg[a] in (H, A)]
    out["possessions"] = len(lengths)
    out["mean_possession_s"] = round(float(np.mean(lengths)) / 25.0, 1) if lengths else 0.0
    return out


def run(name, cfg=None, validate=True, device="cpu"):
    cfg = cfg or load_pipeline()
    m = load_atd("tactical")
    F = compute_features(m)
    from solution.ensemble import blend
    parts = []
    for mem in cfg["members"]:
        if mem["kind"] == "lgbm":
            p = lgbm_probs(F, m.n, mem.get("file", "frame_lgbm.txt"),
                           mem.get("features", "frame_lgbm_features.json"))
        else:
            p = seq_probs(F, m, mem["name"], device=device)
        parts.append((p.astype(np.float64), float(mem["weight"])))
    prob = blend(parts) if len(parts) > 1 else parts[0][0].astype(np.float32)
    print("members", [(mem.get("name", mem["kind"]), mem["weight"]) for mem in cfg["members"]])
    nd = int(cfg.get("round", 4))
    prob = np.round(prob, nd)        # floating-point noise must not change the file
    if any(cfg.get("fusion", {}).values()):
        raise SystemExit("fusion of broadcast / Smart Tagging evidence is not part of this pipeline")
    dec_name = cfg["decoder"]
    dec = json.load(open(os.path.join(MODEL_DIR, dec_name)))["params"]

    # Each half's kick-off is detected from the tracking (solution/kickoff.py): frames of the play
    # window before it are a stoppage, and the HMM runs from the kick. No frame number of this
    # match is written anywhere in the code.
    codes = np.full(m.n, X, dtype=np.int8)
    blocks = []
    for h, a, b in m.half_blocks():
        k = max(a, int(m.meta["kickoff"][h]))
        codes[a:k] = D
        blocks.append((k, b))
    dec_codes = decode(np.nan_to_num(prob, nan=1.0 / 3), blocks, m.n, **dec)
    for a, b in blocks:
        codes[a:b] = dec_codes[a:b]
    codes[m.half == 0] = X
    if len(codes) != GAMES[GAME]["n_video_frames"]:
        raise SystemExit("row count mismatch")
    path = os.path.join(OUT_DIR, "{}.csv".format(name))
    write_submission(path, codes)
    np.save(os.path.join(OUT_DIR, "{}_prob.npy".format(name)), prob)
    print("decoder", dec_name, dec)
    print("summary", describe(codes, m.half))
    print("wrote", os.path.relpath(path, ROOT))
    if validate:
        subprocess.run([sys.executable, os.path.join(ROOT, "evaluation", "validate_submission.py"),
                        path], check=False)
    return codes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="mtah")
    ap.add_argument("--config", default=None, help="pipeline json (default solution/models/"
                                                    "pipeline.json)")
    ap.add_argument("--device", default="cpu", help="cpu (reproducible) or cuda")
    a = ap.parse_args()
    cfg = json.load(open(a.config)) if a.config else None
    run(a.name, cfg, device=a.device)
    sys.stdout.flush()
    os._exit(0)   # torch teardown on Windows can return a spurious code after all work is done


if __name__ == "__main__":
    main()
