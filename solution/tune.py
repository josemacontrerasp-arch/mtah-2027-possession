"""Tune the decoder on external games only (out-of-fold probabilities from `train cv`).

    python -m solution.tune [stage] [--oof DIR] [--out NAME]

The objective is the official ranking metric (spellF1 from evaluation/possession_metrics.py),
averaged over the held-out DFL matches. The provided match plays no part.
"""

import argparse
import itertools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, ROOT)
import possession_metrics as pm  # noqa: E402

from solution.decode import decode  # noqa: E402
from solution.external_dfl import CACHE_DIR, MATCH_IDS  # noqa: E402
from solution.match import X  # noqa: E402

MODEL_DIR = os.path.join(HERE, "models")

GRID = dict(switch_cost=[4, 6, 8, 12, 16, 24],
            to_dead_cost=[8, 16, 32],
            from_dead_cost=[8, 16, 32],
            min_turnover_s=[0.0, 1.5, 2.0, 2.5],
            dead_bias=[-0.5, 0.0, 0.5])


def spaced(codes, half):
    cut = int(np.flatnonzero(half == 2)[0])
    return np.concatenate([codes[:cut], [X], codes[cut:]]).astype(np.int8)


def quick_score(gt, pred):
    runs = pm.valid_runs(gt)
    filled = pm.fill_abstentions(pred, runs)
    spell = pm.spell_scores(gt, filled, runs)
    macro = pm.macro_f1(pm.scored_confusion(pm.confusion(gt, pred)))
    return spell["mean_f1"], macro, spell["n_pred"], spell["n_gt"]


def blocks_of(half):
    out = []
    for h in (1, 2):
        idx = np.flatnonzero(half == h)
        if len(idx):
            out.append((int(idx[0]), int(idx[-1]) + 1))
    return out


def load_oof(stage=1, sub=None):
    data = {}
    sub = sub or ("oof" if stage == 1 else "oof2")
    for mid in MATCH_IDS:
        z = np.load(os.path.join(CACHE_DIR, sub, "{}.npz".format(mid)))
        data[mid] = (z["prob"], z["labels"], z["half"])
    return data


def evaluate_params(data, **kw):
    rows = []
    for mid, (prob, labels, half) in data.items():
        p = np.nan_to_num(prob, nan=1.0 / 3)
        pred = decode(p, blocks_of(half), len(half), **kw)
        rows.append(quick_score(spaced(labels, half), spaced(pred, half)))
    arr = np.array(rows, dtype=np.float64)
    return arr[:, 0].mean(), arr[:, 1].mean(), arr[:, 2].sum() / arr[:, 3].sum(), arr


_DATA = None


def _init(stage, sub):
    global _DATA
    _DATA = load_oof(stage, sub)


def _job(kw):
    spell, macro, ratio, _ = evaluate_params(_DATA, **kw)
    return spell, macro, ratio, kw


def search(stage=1, sub=None, grid=None, workers=16):
    grid = grid or GRID
    keys = list(grid)
    combos = [dict(zip(keys, v)) for v in itertools.product(*(grid[k] for k in keys))]
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(stage, sub)) as ex:
        results = list(ex.map(_job, combos, chunksize=4))
    results.sort(key=lambda r: -r[0])
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", type=int, default=1)
    ap.add_argument("--oof", default=None, help="sub-directory of OOF probabilities")
    ap.add_argument("--out", default=None, help="decoder json name to write")
    ap.add_argument("--grid", default=None, help="JSON dict overriding the search grid")
    ap.add_argument("--dry", action="store_true", help="report only, do not write the json")
    a = ap.parse_args()
    t0 = time.time()
    results = search(a.stage, a.oof, json.loads(a.grid) if a.grid else None)
    print("grid of {} in {:.0f}s; top 10 by mean spellF1 over {} held-out DFL matches".format(
        len(results), time.time() - t0, len(MATCH_IDS)))
    for spell, macro, ratio, kw in results[:10]:
        print("  spellF1 {:.4f}  macroF1 {:.4f}  pred/gt {:.2f}  {}".format(spell, macro, ratio, kw))
    spell, macro, ratio, kw = results[0]
    data = load_oof(a.stage, a.oof)
    _, _, _, per = evaluate_params(data, **kw)
    for mid, row in zip(data, per):
        print("  {}  spellF1 {:.4f}  macroF1 {:.4f}  spells {:.0f}/{:.0f}".format(mid, *row))
    if a.dry:
        return
    os.makedirs(MODEL_DIR, exist_ok=True)
    name = a.out or ("decoder.json" if a.stage == 1 else "decoder_stack.json")
    with open(os.path.join(MODEL_DIR, name), "w") as fh:
        json.dump({"params": kw, "cv_spellF1": spell, "cv_macroF1": macro}, fh, indent=1)
    print("saved", name, kw)


if __name__ == "__main__":
    main()
