"""How close is the submission to changing under floating-point differences? An automatic check.

    python -m solution.noise_check                          # margin + sampled noise trials
    python -m solution.noise_check --trials 0 --save-blend ref     # keep this run's probabilities
    python -m solution.noise_check --trials 0 --against ref        # compare another run with them

What can differ between machines is the last bits of the ensemble members' probabilities (CPU
instruction sets, maths libraries). Whether that reaches the file depends on one thing: after
blending, the probabilities are rounded to a 1e-4 grid, and the decoder works in exact integer
arithmetic (solution/decode.py), so the labels are a pure function of that grid. The file can
change only if a blended probability crosses a rounding boundary. This script reports

1. the margin -- how close the blended probabilities come to a rounding boundary: the smallest
   distance and how many lie within 1e-12 / 1e-10 / 1e-8. A difference between machines smaller
   than the smallest distance cannot move any grid value, and therefore not the file;
2. with --against, the largest difference actually observed between two runs (run the second one
   with instruction sets switched off, see solution/README.md section 9);
3. a sampled sensitivity test: every member probability is multiplied by (1 + eps * gaussian
   noise), blended and decoded again, for several eps, counting the trials in which a grid value
   or a label changes. It samples noise, it is not a bound.

Not covered: differences upstream of the models. The features are built to be bit-identical on
every machine (no maths-library calls, fixed memory layout); if one did differ, a tree split
could flip and move a probability by far more than these levels.

It prints counts and distances only, reads no ground truth and selects nothing. The two optional
files are written to submissions/, which git ignores.
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
from solution.decode import QUANT  # noqa: E402
from solution.features import compute_features  # noqa: E402
from solution.predict import OUT_DIR, blended, labels_from, load_pipeline, member_probs  # noqa: E402

LEVELS = (1e-13, 1e-10, 1e-7)
NEAR = (1e-12, 1e-10, 1e-8)


def grid(prob):
    return np.rint(np.nan_to_num(prob, nan=0.0) * QUANT)


def perturbed(parts, eps, rng):
    out = []
    for p, w in parts:
        q = p * (1.0 + eps * rng.standard_normal(p.shape))
        out.append((q / np.nansum(q, axis=1, keepdims=True), w))
    return out


def _blend_file(name):
    return os.path.join(OUT_DIR, os.path.basename(name) + "_blend.npy")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--save-blend", metavar="NAME", help="keep this run's blended probabilities")
    ap.add_argument("--against", metavar="NAME", help="compare with a run saved by --save-blend")
    a = ap.parse_args()
    cfg = load_pipeline()
    m = load_atd("tactical")
    F = compute_features(m)
    parts = member_probs(cfg, m, F)
    blend = blended(parts)
    ref, ref_prob, _ = labels_from(parts, cfg, m)
    ref_grid = grid(ref_prob)

    p = blend[np.isfinite(blend)]
    v = p * QUANT
    dist = np.abs(v - np.floor(v) - 0.5) / QUANT
    print("blended probabilities: {} finite values on {} frames".format(p.size, len(ref)))
    print("margin: nearest rounding boundary is {:.3e} away; values within {}".format(
        float(dist.min()), ", ".join("{:.0e}: {}".format(t, int((dist < t).sum())) for t in NEAR)))

    if a.save_blend:
        os.makedirs(OUT_DIR, exist_ok=True)
        np.save(_blend_file(a.save_blend), blend)
        print("saved this run's blended probabilities as", os.path.basename(_blend_file(a.save_blend)))
    if a.against:
        other = np.load(_blend_file(a.against))
        same_nan = bool((np.isnan(other) == np.isnan(blend)).all())
        diff = np.abs(np.nan_to_num(other, nan=0.0) - np.nan_to_num(blend, nan=0.0))
        print("against '{}': same missing rows {}; values differing {} of {}; largest difference "
              "{:.3e}; grid values differing {}".format(
                  a.against, same_nan, int((diff > 0).sum()), p.size, float(diff.max()),
                  int((grid(np.round(other, 4)) != ref_grid).sum())))

    for i, eps in enumerate(LEVELS if a.trials > 0 else ()):
        rng = np.random.default_rng(12345 + i)
        file_trials = grid_trials = frames = cells = 0
        for _ in range(a.trials):
            codes, prob, _ = labels_from(perturbed(parts, eps, rng), cfg, m)
            d = int((codes != ref).sum())
            c = int((grid(prob) != ref_grid).sum())
            file_trials += d > 0
            grid_trials += c > 0
            frames += d
            cells += c
        print("noise {:.0e}: file changed in {} of {} trials ({} frames in all); a grid value "
              "changed in {} trials ({} values in all)".format(
                  eps, file_trials, a.trials, frames, grid_trials, cells), flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)   # same reason as in solution/predict.py
