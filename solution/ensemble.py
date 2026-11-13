"""Average out-of-fold probabilities of several models (external games only).

    python -m solution.ensemble oof_ens oof:0.5 oof_seq:0.5

Writes external/dfl/cache/<out>/<mid>.npz, which `solution.tune --oof <out>` then evaluates.
Weights are fixed a priori (equal) or chosen on these external out-of-fold results only.
"""

import os
import sys

import numpy as np

from solution.external_dfl import CACHE_DIR, MATCH_IDS


def blend(parts, pooling="linear", dtype=np.float32):
    """parts: list of (prob array, weight); NaN-aware weighted pool, renormalised per row.
    dtype: float32 for the out-of-fold files, float64 for the submission (solution/predict.py).

    pooling="linear" is a weighted mean of probabilities; "log" a weighted geometric mean
    (log-linear pooling), which rewards agreement between members."""
    if pooling == "log":
        acc, wsum = None, None
        for p, w in parts:
            ok = np.isfinite(p[:, :1])
            v = np.where(ok, np.log(np.clip(p, 1e-6, 1.0)), 0.0) * w
            acc = v if acc is None else acc + v
            ws = ok.astype(np.float64) * w
            wsum = ws if wsum is None else wsum + ws
        lg = np.where(wsum > 0, acc / np.maximum(wsum, 1e-12), np.nan)
        lg = lg - np.nanmax(lg, axis=1, keepdims=True)
        out = np.exp(lg)
        return (out / np.nansum(out, axis=1, keepdims=True)).astype(dtype)
    acc, wsum = None, None
    for p, w in parts:
        ok = np.isfinite(p[:, :1])
        v = np.where(ok, p, 0.0) * w
        acc = v if acc is None else acc + v
        ws = ok.astype(np.float64) * w
        wsum = ws if wsum is None else wsum + ws
    out = np.where(wsum > 0, acc / np.maximum(wsum, 1e-12), np.nan)
    return (out / np.nansum(out, axis=1, keepdims=True)).astype(dtype)


def main(out, specs, pooling="linear"):
    os.makedirs(os.path.join(CACHE_DIR, out), exist_ok=True)
    for mid in MATCH_IDS:
        parts, labels, half = [], None, None
        for spec in specs:
            sub, w = spec.split(":")
            z = np.load(os.path.join(CACHE_DIR, sub, "{}.npz".format(mid)))
            parts.append((z["prob"].astype(np.float64), float(w)))
            labels, half = z["labels"], z["half"]
        np.savez_compressed(os.path.join(CACHE_DIR, out, "{}.npz".format(mid)),
                            prob=blend(parts, pooling), labels=labels, half=half)
    print("wrote", out, "from", specs, "pooling", pooling)


if __name__ == "__main__":
    args = sys.argv[1:]
    pool = "linear"
    if args and args[0].startswith("--pool="):
        pool = args.pop(0).split("=", 1)[1]
    main(args[0], args[1:], pool)
