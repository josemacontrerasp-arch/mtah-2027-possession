"""Pre-registered selection (solution/PREREGISTRATION.md): nested leave-one-match-out over the
clean evaluation environments only. External games only; the provided match plays no part.

    python -m solution.clean_select <name> <oof_rnd> <oof_rnd2> <oof_rndh> [--grid JSON] [--save F]

For every decoder setting, spellF1 is computed per environment and per match. For each held-out
match, the setting with the best mean over the other six matches (all environments) is chosen and
scored on the held-out match. The mean of those is the nested clean score used by the rules.
"""

import argparse
import itertools
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from solution.decode import decode  # noqa: E402
from solution.external_dfl import MATCH_IDS  # noqa: E402
from solution.tune import MODEL_DIR, blocks_of, load_oof, quick_score, spaced  # noqa: E402

GRID = dict(switch_cost=[4, 6, 8, 12],
            to_dead_cost=[16, 32, 48],
            from_dead_cost=[16, 32, 48],
            min_turnover_s=[1.5, 2.0, 2.5, 3.0],
            dead_bias=[-0.5, 0.0, 0.5],
            turnover_exit_s=[0.0, 2.0],
            min_dead_s=[0.0, 2.0])

_ENVS = None


def _init(subs):
    global _ENVS
    _ENVS = [load_oof(1, s) for s in subs]


def _job(kw):
    """(n_env, n_match) spellF1 and the pooled predicted/true spell ratio for one setting."""
    out = np.zeros((len(_ENVS), len(MATCH_IDS)))
    npred = ngt = 0
    for e, data in enumerate(_ENVS):
        for j, mid in enumerate(MATCH_IDS):
            prob, labels, half = data[mid]
            pred = decode(np.nan_to_num(prob, nan=1.0 / 3), blocks_of(half), len(half), **kw)
            s, _, n_p, n_g = quick_score(spaced(labels, half), spaced(pred, half))
            out[e, j] = s
            npred += n_p
            ngt += n_g
    return out, npred / max(1, ngt)


def run(subs, grid, workers=16):
    keys = list(grid)
    combos = [dict(zip(keys, v)) for v in itertools.product(*(grid[k] for k in keys))]
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(subs,)) as ex:
        res = list(ex.map(_job, combos, chunksize=2))
    table = np.stack([r[0] for r in res])            # (C, E, M)
    ratio = np.array([r[1] for r in res])
    per_match = table.mean(axis=1)                    # (C, M), averaged over environments
    nested, chosen = [], []
    for i in range(len(MATCH_IDS)):
        others = [j for j in range(len(MATCH_IDS)) if j != i]
        best = int(np.argmax(per_match[:, others].mean(axis=1)))
        nested.append(per_match[best, i])
        chosen.append(best)
    insample = int(np.argmax(per_match.mean(axis=1)))
    return dict(combos=combos, table=table, ratio=ratio, nested=np.array(nested),
                chosen=chosen, insample=insample)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("subs", nargs=3)
    ap.add_argument("--grid", default=None)
    ap.add_argument("--save", default=None, help="write the in-sample best setting to models/")
    a = ap.parse_args()
    grid = json.loads(a.grid) if a.grid else GRID
    r = run(a.subs, grid)
    best = r["combos"][r["insample"]]
    pm = r["table"][r["insample"]]
    print("[{}] {} settings x 3 clean environments x 7 matches".format(a.name, len(r["combos"])))
    print("  nested clean spellF1 {:.4f}   per match {}".format(r["nested"].mean(),
                                                                np.round(r["nested"], 3).tolist()))
    print("  in-sample best {:.4f} (rnd {:.4f}, rnd2 {:.4f}, rndh {:.4f}), spells pred/true "
          "{:.2f}: {}".format(pm.mean(), *pm.mean(axis=1), r["ratio"][r["insample"]], best))
    print("  settings chosen per fold:", [r["combos"][c] == best for c in r["chosen"]].count(True),
          "of 7 folds pick the in-sample best")
    np.savez_compressed(os.path.join(os.path.dirname(HERE), "..", "external", "dfl", "cache",
                                     "select_{}.npz".format(a.name)),
                        table=r["table"], nested=r["nested"], ratio=r["ratio"])
    if a.save:
        with open(os.path.join(MODEL_DIR, a.save), "w") as fh:
            json.dump({"params": best, "nested_clean_spellF1": float(r["nested"].mean()),
                       "insample_clean_spellF1": float(pm.mean()),
                       "environments": a.subs,
                       "selection": "solution/clean_select.py (pre-registered)"}, fh, indent=1)
        print("  saved", a.save)


if __name__ == "__main__":
    main()
