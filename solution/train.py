"""Train the frame classifier on external games and measure it honestly (leave-one-match-out).

    python -m solution.train cv        # 7-fold LOMO on DFL: OOF probabilities -> data cache
    python -m solution.train final     # fit on all external games -> models/frame_lgbm.txt
    python -m solution.train build     # only build the degraded training copies (cached), in parallel

Everything here sees external data only. The provided match is never loaded in this module.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402  (the official metric code)

from solution.degrade import degrade, swap_teams  # noqa: E402
from solution.external_dfl import load_dfl, MATCH_IDS, CACHE_DIR  # noqa: E402
from solution.external_skillcorner import SC_IDS, load_skillcorner  # noqa: E402
from solution.features import compute_features  # noqa: E402
from solution.match import X  # noqa: E402
from solution.relabel import controlled  # noqa: E402

MODEL_DIR = os.path.join(HERE, "models")
EXCLUDE = {"b_speed_prov", "half_t"}   # provider-specific / clock features: do not transfer
TRAIN_SEEDS = tuple(int(x) for x in os.environ.get("MTAH_SEEDS", "11,12,13,14").split(","))
EVAL_SEED = 101
TRAIN_STEP = 5
FEAT_VERSION = 5


def load_ext(mid):
    """External match with labels converted to *controlled* possession (solution/relabel.py)."""
    m = load_skillcorner(mid[2:]) if is_sc(mid) else load_dfl(mid)
    m.labels = controlled(m.labels, m.half)
    return m


# Extra labelled games that are always in training and never evaluated (PREREGISTRATION.md,
# v8): "skillcorner" = SkillCorner's 20 open matches, real broadcast tracking, no simulated
# degradation, both orientations and two 5 Hz frame offsets each.
EXTRA = os.environ.get("MTAH_EXTRA", "")
if EXTRA not in ("", "skillcorner"):
    raise SystemExit("MTAH_EXTRA must be empty or 'skillcorner'")
EXTRA_IDS = ["SC" + i for i in SC_IDS] if EXTRA == "skillcorner" else []
SC_SEEDS = (11, 12)


def is_sc(mid):
    return mid.startswith("SC")


def copy_seeds(mid):
    return SC_SEEDS if is_sc(mid) else TRAIN_SEEDS

LGB_PARAMS = dict(objective="multiclass", num_class=3,
                  learning_rate=float(os.environ.get("MTAH_LR", 0.05)), num_leaves=63,
                  min_data_in_leaf=200, feature_fraction=0.7, bagging_fraction=0.7,
                  bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=16, seed=7,
                  deterministic=True, force_row_wise=True)
NUM_ROUNDS = int(os.environ.get("MTAH_ROUNDS", 400))
OOF_SUB = os.environ.get("MTAH_OOF", "oof")
STRESS = os.environ.get("MTAH_STRESS", "0") == "1"   # also score on the stress profile
# Training degradation: "calibrated" (solution/degrade.py, matched to unlabelled statistics of the
# provided tracking) or "random" (solution/degrade_random.py, first-principles priors only).
DEGRADE = os.environ.get("MTAH_DEGRADE", "random")   # v6 default; "calibrated" = v4 history
# Randomised training trackers: "shared" = one tracker per seed, applied to every external game in
# both orientations (v6: four trackers in all); "percopy" = every (game, orientation, seed) copy
# draws its own tracker (the v7 candidate: 56 trackers; tested and not adopted, selection_log.md).
TRACKERS = os.environ.get("MTAH_TRACKERS", "shared")
# Extra evaluation conditions written next to the standard OOF: "rnd" = a fresh randomised tracker.
EXTRA_EVALS = [e for e in os.environ.get("MTAH_EVALS", "").split(",") if e]


def tracker_seed(mid, swapped, seed):
    """Seed of the simulated tracker for one training copy. Per-copy seeds start at 1,000,000, so
    they cannot coincide with the evaluation trackers (seeds 3101-5107, see eval_arrays)."""
    if TRACKERS == "percopy":
        return 1_000_000 + 10_000 * MATCH_IDS.index(mid) + 1_000 * int(swapped) + seed
    return seed


def _feat_cache(name):
    return os.path.join(CACHE_DIR, "feat_v{}_{}.parquet".format(FEAT_VERSION, name))


def variant_table(mid, swapped, seed, step):
    """Features + label for one degraded variant of an external match (cached)."""
    tag = "{}_{}_s{}_step{}".format(mid, "swap" if swapped else "orig", seed, step)
    if is_sc(mid):
        tag = "sc_" + tag          # a real tracker's own gaps: no simulated degradation
    elif DEGRADE == "random":
        tag = ("rndpc_" if TRACKERS == "percopy" else "rnd_") + tag
    path = _feat_cache(tag)
    if os.path.isfile(path):
        return pd.read_parquet(path)
    m = load_ext(mid)
    if swapped:
        m = swap_teams(m)
    if is_sc(mid):
        pass
    elif DEGRADE == "random":
        from solution.degrade_random import degrade_random
        m = degrade_random(m, seed=tracker_seed(mid, swapped, seed))
    else:
        m = degrade(m, seed=seed)
    F = compute_features(m)
    y = m.labels[F.index.values]
    F["y"] = y
    F["frame"] = F.index.values.astype(np.int64)
    if step > 1:
        F = F[(F["frame"] % step) == (seed % step)]
    F = F[F["y"] != X]
    F.reset_index(drop=True).to_parquet(path)
    return F


def train_rows(mids):
    parts = []
    for mid in mids:
        for swapped in (False, True):
            for seed in copy_seeds(mid):
                t = variant_table(mid, swapped, seed, TRAIN_STEP)
                t["mid"] = mid
                parts.append(t)
    return pd.concat(parts, ignore_index=True)


def _build_one(job):
    variant_table(job[0], job[1], job[2], TRAIN_STEP)
    return job


def cmd_build(args):
    """Build (and cache) every degraded training copy, several at a time. Each copy is seeded on
    its own, so the tables are the same whatever the number of workers."""
    from concurrent.futures import ProcessPoolExecutor
    jobs = [(mid, sw, s) for mid in MATCH_IDS + EXTRA_IDS for sw in (False, True)
            for s in copy_seeds(mid)]
    t0 = time.time()
    with ProcessPoolExecutor(args.jobs) as ex:
        for k, job in enumerate(ex.map(_build_one, jobs), 1):
            print("built {}/{}  {} {} seed {}  ({:.0f}s)".format(
                k, len(jobs), job[0], "swap" if job[1] else "orig", job[2], time.time() - t0),
                flush=True)


def feature_columns(df):
    return [c for c in df.columns if c not in EXCLUDE | {"y", "frame", "mid"}]


def fit(df, cols, rounds=NUM_ROUNDS):
    import lightgbm as lgb
    ds = lgb.Dataset(df[cols], label=df["y"].values)
    return lgb.train(LGB_PARAMS, ds, num_boost_round=rounds)


def eval_arrays(mid, hard=False, profile=None):
    """Full-frame features of an evaluation variant, plus gt codes over the whole match.
    hard=True uses the stress profile (solution/degrade.py HARD); profile="rnd" a randomised
    tracker drawn from solution/degrade_random.py priors (seed fixed per match). Evaluation only."""
    if profile in ("rnd", "rnd2", "rndh"):
        # clean evaluation trackers: two independent draws from the first-principles priors and
        # one from their harsher half; fixed seeds per match, never used for training
        from solution.degrade_random import degrade_random
        base = {"rnd": 3000, "rnd2": 4000, "rndh": 5000}[profile]
        m = degrade_random(load_ext(mid), seed=EVAL_SEED + base + MATCH_IDS.index(mid),
                           harsh=(profile == "rndh"))
    else:
        m = degrade(load_ext(mid), seed=EVAL_SEED + (101 if hard else 0), hard=hard)
    F = compute_features(m)
    return m, F


def spaced(codes, half):
    """Insert an X frame between halves so no spell or run can cross halftime."""
    cut = int(np.flatnonzero(half == 2)[0])
    return np.concatenate([codes[:cut], [X], codes[cut:]]).astype(np.int8)


def score(gt, pred, half):
    r = pm.evaluate(spaced(gt, half), spaced(pred, half))
    return {"spellF1": r["spell_f1_mean"], "macroF1": r["macro_f1"],
            "n_gt": r["n_gt_spells"], "n_pred": r["n_pred_spells"],
            "dur_gt": r["gt_mean_spell_duration_s"], "dur_pred": r["mean_spell_duration_s"],
            "alive_dead_f1": r["alive_dead_f1"], "team_acc": r["team_acc_both_alive"]}


def cmd_cv(args):
    oof_dir = os.path.join(CACHE_DIR, OOF_SUB)
    os.makedirs(oof_dir, exist_ok=True)
    t0 = time.time()
    all_rows = train_rows(MATCH_IDS + EXTRA_IDS)    # extra games: in every fold's training
    cols = feature_columns(all_rows)
    print("train rows {:,}  features {}  ({:.0f}s)".format(len(all_rows), len(cols),
                                                            time.time() - t0))
    for mid in MATCH_IDS:
        t1 = time.time()
        model = fit(all_rows[all_rows["mid"] != mid], cols)
        m, F = eval_arrays(mid)
        prob = np.full((m.n, 3), np.nan, np.float32)
        prob[F.index.values] = model.predict(F[cols]).astype(np.float32)
        np.savez_compressed(os.path.join(oof_dir, "{}.npz".format(mid)), prob=prob,
                            labels=m.labels, half=m.half)
        if STRESS:
            mh, Fh = eval_arrays(mid, hard=True)
            ph = np.full((mh.n, 3), np.nan, np.float32)
            ph[Fh.index.values] = model.predict(Fh[cols]).astype(np.float32)
            os.makedirs(oof_dir + "_hard", exist_ok=True)
            np.savez_compressed(os.path.join(oof_dir + "_hard", "{}.npz".format(mid)), prob=ph,
                                labels=mh.labels, half=mh.half)
        for prof in EXTRA_EVALS:
            mr, Fr = eval_arrays(mid, profile=prof)
            pr_ = np.full((mr.n, 3), np.nan, np.float32)
            pr_[Fr.index.values] = model.predict(Fr[cols]).astype(np.float32)
            os.makedirs(oof_dir + "_" + prof, exist_ok=True)
            np.savez_compressed(os.path.join(oof_dir + "_" + prof, "{}.npz".format(mid)),
                                prob=pr_, labels=mr.labels, half=mr.half)
        frame_pred = np.full(m.n, X, np.int8)
        ok = np.isfinite(prob[:, 0])
        frame_pred[ok] = np.argmax(prob[ok], axis=1)
        s = score(m.labels, frame_pred, m.half)
        print("{}  argmax-only  spellF1 {:.4f}  macroF1 {:.4f}  spells {}/{}  ({:.0f}s)".format(
            mid, s["spellF1"], s["macroF1"], s["n_pred"], s["n_gt"], time.time() - t1))
        if args.importance and mid == MATCH_IDS[0]:
            imp = pd.Series(model.feature_importance("gain"), index=cols).sort_values()[::-1]
            print(imp.head(30).round(0).to_string())


def cmd_final(args):
    os.makedirs(MODEL_DIR, exist_ok=True)
    rows = train_rows(MATCH_IDS + EXTRA_IDS)
    cols = feature_columns(rows)
    model = fit(rows, cols)
    name = os.environ.get("MTAH_MODEL_NAME", "frame_lgbm_v6")
    path = os.path.join(MODEL_DIR, name + ".txt")
    model.save_model(path)
    with open(os.path.join(MODEL_DIR, name + "_features.json"), "w") as fh:
        json.dump(cols, fh, indent=1)
    print("saved", path, "features", len(cols), "rows", len(rows))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cv", "final", "build"])
    ap.add_argument("--importance", action="store_true")
    ap.add_argument("--jobs", type=int, default=4, help="parallel workers for `build`")
    a = ap.parse_args()
    {"cv": cmd_cv, "final": cmd_final, "build": cmd_build}[a.cmd](a)
