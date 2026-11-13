"""How close is the submission to changing under floating-point noise? An automatic check.

    python -m solution.noise_check [--trials 200]

A different CPU, instruction set or maths library changes the last bits of the networks' outputs.
The submission chain is built so that this cannot reach the file: the GRUs run in float64, the
blended probabilities are put on a 1e-4 grid and the decoder works in exact integer arithmetic
(solution/decode.py). This script measures the margin. It computes the ensemble members for the
provided match once, then repeats "blend, quantise, decode" with every member probability
multiplied by (1 + eps * gaussian noise), for several noise levels eps, and counts the trials in
which at least one label changes.

Float64 rounding differences between platforms are of the order of 1e-16 to 1e-13; 1e-7 is what
single-precision inference would have produced.

It prints counts only (trials changed, frames changed), reads no ground truth and selects nothing.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from solution.atd import load_atd  # noqa: E402
from solution.features import compute_features  # noqa: E402
from solution.predict import labels_from, load_pipeline, member_probs  # noqa: E402

LEVELS = (1e-13, 1e-10, 1e-7)


def perturbed(parts, eps, rng):
    out = []
    for p, w in parts:
        q = p * (1.0 + eps * rng.standard_normal(p.shape))
        out.append((q / np.nansum(q, axis=1, keepdims=True), w))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=200)
    a = ap.parse_args()
    cfg = load_pipeline()
    m = load_atd("tactical")
    F = compute_features(m)
    parts = member_probs(cfg, m, F)
    ref, ref_prob, _ = labels_from(parts, cfg, m)
    grid_ref = np.rint(np.nan_to_num(ref_prob, nan=0.0) * 10_000)
    print("reference decoded: {} frames".format(len(ref)))
    for eps in LEVELS:
        rng = np.random.default_rng(12345)
        changed, frames, cells = 0, 0, 0
        for _ in range(a.trials):
            codes, prob, _ = labels_from(perturbed(parts, eps, rng), cfg, m)
            d = int((codes != ref).sum())
            changed += d > 0
            frames += d
            cells += int((np.rint(np.nan_to_num(prob, nan=0.0) * 10_000) != grid_ref).sum())
        print("noise {:.0e}: {} of {} trials changed the file; mean frames changed {:.2f}; "
              "mean grid values changed {:.1f} of {}".format(
                  eps, changed, a.trials, frames / a.trials, cells / a.trials, grid_ref.size),
              flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)   # same reason as in solution/predict.py
