"""Nested leave-one-match-out: how optimistic is a score whose decoder was tuned on the same
out-of-fold predictions it is reported on?

For each external match k: choose decoder parameters on the OTHER six matches only, then score
match k with them. The mean of those scores is an unbiased estimate of the tuned pipeline.

    python -m solution.nested_check oof_b3w
"""

import itertools
import sys

import numpy as np

from solution.tune import GRID, evaluate_params, load_oof


def main(sub):
    data = load_oof(1, sub)
    keys = list(GRID)
    grid = dict(GRID, switch_cost=[3, 4, 6, 8, 12, 16])
    combos = [dict(zip(keys, v)) for v in itertools.product(*(grid[k] for k in keys))]
    # per-combo per-match scores, computed once
    table = np.array([evaluate_params(data, **kw)[3][:, 0] for kw in combos])   # (C, 7)
    mids = list(data)
    nested, chosen = [], []
    for i, mid in enumerate(mids):
        others = [j for j in range(len(mids)) if j != i]
        best = int(np.argmax(table[:, others].mean(axis=1)))
        nested.append(table[best, i])
        chosen.append(combos[best])
    in_sample = table.mean(axis=1).max()
    print("tuned and scored on the same 7 matches : spellF1 {:.4f}".format(in_sample))
    print("nested (tuned on 6, scored on the 7th) : spellF1 {:.4f}  per match {}".format(
        float(np.mean(nested)), np.round(nested, 3).tolist()))
    print("optimism of the in-sample figure       : {:+.4f}".format(in_sample - np.mean(nested)))
    print("chosen switch costs per fold:", [c["switch_cost"] for c in chosen])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "oof_b3w")
