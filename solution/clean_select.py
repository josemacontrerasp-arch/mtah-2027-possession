"""Pre-registered selection (solution/PREREGISTRATION.md): nested leave-one-match-out over the
clean evaluation environments only. External games only; the provided match plays no part.

    python -m solution.clean_select <name> <oof_rnd> <oof_rnd2> <oof_rndh> [--grid JSON] [--save F]
    python -m solution.clean_select --compare <candidate name> <incumbent name>   # paired criteria

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


def run(subs, grid, workers=16, cands=None):
    """cands: optional list of (label, [sub_rnd, sub_rnd2, sub_rndh]) candidate ensembles; the
    candidate becomes one more dimension of the nested choice."""
    keys = list(grid)
    combos = [dict(zip(keys, v)) for v in itertools.product(*(grid[k] for k in keys))]
    cands = cands or [("main", subs)]
    tables, ratios, labels = [], [], []
    for label, csubs in cands:
        with ProcessPoolExecutor(workers, initializer=_init, initargs=(csubs,)) as ex:
            res = list(ex.map(_job, combos, chunksize=2))
        tables.append(np.stack([r[0] for r in res]))
        ratios.append(np.array([r[1] for r in res]))
        labels += [label] * len(combos)
    table = np.concatenate(tables)                    # (cands*C, E, M)
    ratio = np.concatenate(ratios)
    combos = [dict(c, ensemble=l) for l, c in zip(labels, combos * len(cands))]
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


def nested_by_env(table):
    """(E, M) nested scores: for each held-out match, the setting chosen on the other six."""
    per_match = table.mean(axis=1)
    n_match = table.shape[2]
    out = np.zeros(table.shape[1:])
    for i in range(n_match):
        others = [j for j in range(n_match) if j != i]
        best = int(np.argmax(per_match[:, others].mean(axis=1)))
        out[:, i] = table[best, :, i]
    return out


def compare(cand, inc, min_gain=0.005, min_wins=5, max_env_drop=0.003):
    """The paired criteria of PREREGISTRATION.md rule 5 (candidate against incumbent), from the
    tables two earlier runs of this script saved. Prints every number the rule uses."""
    from solution.external_dfl import CACHE_DIR
    a = nested_by_env(np.load(os.path.join(CACHE_DIR, "select_{}.npz".format(cand)))["table"])
    b = nested_by_env(np.load(os.path.join(CACHE_DIR, "select_{}.npz".format(inc)))["table"])
    ma, mb = a.mean(axis=0), b.mean(axis=0)            # per match, averaged over environments
    gain = float(ma.mean() - mb.mean())
    wins = int((ma > mb).sum())
    env_delta = a.mean(axis=1) - b.mean(axis=1)
    print("nested clean spellF1: {} {:.4f}   {} {:.4f}   gain {:+.4f}".format(
        cand, ma.mean(), inc, mb.mean(), gain))
    print("  per match  {}: {}".format(cand, np.round(ma, 3).tolist()))
    print("  per match  {}: {}".format(inc, np.round(mb, 3).tolist()))
    print("  per environment (rnd, rnd2, rndh)  {}: {}   {}: {}   delta {}".format(
        cand, np.round(a.mean(axis=1), 4).tolist(), inc, np.round(b.mean(axis=1), 4).tolist(),
        np.round(env_delta, 4).tolist()))
    c1, c2, c3 = gain >= min_gain, wins >= min_wins, bool(env_delta.min() >= -max_env_drop)
    print("  gain >= {:+.3f}: {}   |   better on >= {} of {} matches: {} ({})   |   no environment "
          "drops by more than {:.3f}: {}".format(min_gain, c1, min_wins, len(ma), c2, wins,
                                                  max_env_drop, c3))
    print("  => {}".format("ADOPT the candidate" if (c1 and c2 and c3) else "KEEP the incumbent"))
    return c1 and c2 and c3


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--compare":
        compare(sys.argv[2], sys.argv[3])
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("subs", nargs="*", help="three OOF dirs (rnd rnd2 rndh) for one ensemble")
    ap.add_argument("--cand", action="append", default=[],
                    help="label=sub_rnd,sub_rnd2,sub_rndh ; repeat for several candidates")
    ap.add_argument("--grid", default=None)
    ap.add_argument("--save", default=None, help="write the in-sample best setting to models/")
    a = ap.parse_args()
    grid = json.loads(a.grid) if a.grid else GRID
    cands = [(c.split("=")[0], c.split("=")[1].split(",")) for c in a.cand] or None
    r = run(a.subs, grid, cands=cands)
    best = r["combos"][r["insample"]]
    pm = r["table"][r["insample"]]
    print("[{}] {} settings x 3 clean environments x 7 matches".format(a.name, len(r["combos"])))
    print("  nested clean spellF1 {:.4f}   per match {}".format(r["nested"].mean(),
                                                                np.round(r["nested"], 3).tolist()))
    print("  in-sample best {:.4f} (rnd {:.4f}, rnd2 {:.4f}, rndh {:.4f}), spells pred/true "
          "{:.2f}: {}".format(pm.mean(), *pm.mean(axis=1), r["ratio"][r["insample"]], best))
    print("  settings chosen per fold:", [r["combos"][c] == best for c in r["chosen"]].count(True),
          "of 7 folds pick the in-sample best")
    from solution.external_dfl import CACHE_DIR
    np.savez_compressed(os.path.join(CACHE_DIR, "select_{}.npz".format(a.name)),
                        table=r["table"], nested=r["nested"], ratio=r["ratio"])
    if a.save:
        with open(os.path.join(MODEL_DIR, a.save), "w") as fh:
            json.dump({"params": {k: v for k, v in best.items() if k != "ensemble"},
                       "ensemble": best.get("ensemble"),
                       "nested_clean_spellF1": float(r["nested"].mean()),
                       "insample_clean_spellF1": float(pm.mean()),
                       "environments": a.subs or a.cand,
                       "selection": "solution/clean_select.py (pre-registered)"}, fh, indent=1)
        print("  saved", a.save)


if __name__ == "__main__":
    main()
