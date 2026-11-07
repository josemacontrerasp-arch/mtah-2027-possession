"""Average out-of-fold probabilities of several models (external games only).

    python -m solution.ensemble oof_ens oof:0.5 oof_seq:0.5

Writes external/dfl/cache/<out>/<mid>.npz, which `solution.tune --oof <out>` then evaluates.
Weights are fixed a priori (equal) or chosen on these external out-of-fold results only.
"""

import os
import sys

import numpy as np

from solution.external_dfl import CACHE_DIR, MATCH_IDS


def blend(parts):
    """parts: list of (prob array, weight); NaN-aware weighted mean, renormalised per row."""
    acc, wsum = None, None
    for p, w in parts:
        ok = np.isfinite(p[:, :1])
        v = np.where(ok, p, 0.0) * w
        acc = v if acc is None else acc + v
        ws = ok.astype(np.float64) * w
        wsum = ws if wsum is None else wsum + ws
    out = np.where(wsum > 0, acc / np.maximum(wsum, 1e-12), np.nan)
    return (out / np.nansum(out, axis=1, keepdims=True)).astype(np.float32)


def main(out, specs):
    os.makedirs(os.path.join(CACHE_DIR, out), exist_ok=True)
    for mid in MATCH_IDS:
        parts, labels, half = [], None, None
        for spec in specs:
            sub, w = spec.split(":")
            z = np.load(os.path.join(CACHE_DIR, sub, "{}.npz".format(mid)))
            parts.append((z["prob"].astype(np.float64), float(w)))
            labels, half = z["labels"], z["half"]
        np.savez_compressed(os.path.join(CACHE_DIR, out, "{}.npz".format(mid)),
                            prob=blend(parts), labels=labels, half=half)
    print("wrote", out, "from", specs)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
