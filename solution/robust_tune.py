"""Choose decoder settings that hold up on BOTH the standard and the stress benchmark.

The standard external benchmark ties ball loss to stoppages more cleanly than a real tracker
does, which flatters dead-ball detection. The stress profile (solution/degrade.py HARD) loosens
that coupling. We score every decoder setting on both sets of out-of-fold predictions (external
games only) and keep the best average -- a hedge against the real tracker being worse than the
simulation.

    python -m solution.robust_tune <standard oof dir> <hard oof dir> [--out decoder_ens.json]
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from solution.tune import MODEL_DIR, search  # noqa: E402

GRID = dict(switch_cost=[3, 4, 6, 8, 12],
            to_dead_cost=[8, 16, 32, 48],
            from_dead_cost=[8, 16, 32, 48],
            min_turnover_s=[2.0, 2.5, 3.0],
            dead_bias=[-0.5, 0.0, 0.5, 1.0])


def key(kw):
    return tuple(sorted(kw.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("std")
    ap.add_argument("hard")
    ap.add_argument("--out", default=None)
    ap.add_argument("--current", default=None, help="decoder json to report for comparison")
    a = ap.parse_args()
    std = {key(kw): (s, m) for s, m, _, kw in search(1, a.std, GRID)}
    hard = {key(kw): (s, m) for s, m, _, kw in search(1, a.hard, GRID)}
    rows = sorted((((std[k][0] + hard[k][0]) / 2, std[k][0], hard[k][0], std[k][1], hard[k][1],
                    dict(k)) for k in std), key=lambda r: -r[0])
    best_std = max(std.items(), key=lambda kv: kv[1][0])
    best_hard = max(hard.items(), key=lambda kv: kv[1][0])
    print("best on standard only: std {:.4f}  hard {:.4f}  {}".format(
        best_std[1][0], hard[best_std[0]][0], dict(best_std[0])))
    print("best on hard only    : std {:.4f}  hard {:.4f}  {}".format(
        std[best_hard[0]][0], best_hard[1][0], dict(best_hard[0])))
    if a.current:
        cur = json.load(open(os.path.join(MODEL_DIR, a.current)))["params"]
        k = key(cur)
        if k in std:
            print("current decoder      : std {:.4f}  hard {:.4f}  {}".format(
                std[k][0], hard[k][0], cur))
    print("top by mean of both:")
    for mean, s, h, ms, mh, kw in rows[:6]:
        print("  mean {:.4f}  std {:.4f} (macro {:.4f})  hard {:.4f} (macro {:.4f})  {}".format(
            mean, s, ms, h, mh, kw))
    if a.out:
        mean, s, h, ms, mh, kw = rows[0]
        with open(os.path.join(MODEL_DIR, a.out), "w") as fh:
            json.dump({"params": kw, "cv_spellF1_standard": s, "cv_spellF1_stress": h,
                       "cv_macroF1_standard": ms, "cv_macroF1_stress": mh,
                       "selection": "best mean spellF1 over standard and stress benchmarks"},
                      fh, indent=1)
        print("saved", a.out, kw)


if __name__ == "__main__":
    main()
