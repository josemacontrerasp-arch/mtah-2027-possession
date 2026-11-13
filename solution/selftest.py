"""Does this machine compute exactly what ours did? A check that needs no match data.

    python -m solution.selftest              # compare with solution/models/selftest_reference.json
    python -m solution.selftest --record     # write the reference (on the machine that made the file)

It builds a synthetic match from a fixed seed and runs the whole prediction chain on it: kick-off
detection, features, the five committed models, blending, the 1e-4 grid and the decoder -- the
same functions solution/predict.py uses. The synthetic match is made from the random-number
generator's integers and uniform numbers with exactly rounded arithmetic only (add, multiply,
divide, square root), so it is the same bits on every machine. Each stage is compared with the
values recorded on the machine that produced the submission:

* features, kick-offs, LightGBM raw scores, grid values and labels must be identical;
* the GRUs' double-precision probabilities, sampled at 200 frames, may differ by rounding only
  (expected ~1e-15; anything above 1e-12 fails).

If everything matches, this machine runs the chain the way ours does; with the match bundle,
`python -m solution.repro_check --fresh-cache` then confirms the submission's SHA-256. The
synthetic match is not football: it only has to exercise every code path.
"""

import argparse
import hashlib
import json
import os
import platform
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from solution.decode import QUANT  # noqa: E402
from solution.features import compute_features  # noqa: E402
from solution.kickoff import detect_all  # noqa: E402
from solution.match import Match, H, A, D  # noqa: E402
from solution.predict import MODEL_DIR, blended, labels_from, load_pipeline, member_probs  # noqa

REF = os.path.join(MODEL_DIR, "selftest_reference.json")
SEED = 20271117
N_SAMPLE = 200
GRU_TOL = 1e-12


# ----------------------------------------------------------------------------- synthetic match
def synthetic_match(seed=SEED):
    """Two 8-minute halves: a kick-off formation that breaks into play, players moving with
    momentum, a ball passed between carriers with tracking gaps, stoppages, missing players."""
    rng = np.random.default_rng(seed)
    L, W, fps, P = 105.0, 68.0, 25, 22
    half_x = L / 2.0
    team = np.array([0] * 11 + [1] * 11, dtype=np.int8)
    pre, n_half, gap, post = 300, 12000, 600, 300
    n = pre + n_half + gap + n_half + post
    half = np.zeros(n, np.int8)
    a1, a2 = pre, pre + n_half + gap
    half[a1:a1 + n_half] = 1
    half[a2:a2 + n_half] = 2
    pxy = np.full((n, P, 2), np.nan)
    bxy = np.full((n, 2), np.nan)
    side = np.where(team == 0, -1.0, 1.0)            # home's own half is the low-x side if d = +1

    for h, a, form_len, d in ((1, a1, 200, 1.0), (2, a2, 400, -1.0)):
        start = a - pre if h == 1 else a - gap // 2    # formation is held before the window opens
        kick = a + form_len
        end = a + n_half + (post if h == 2 else gap // 2)
        # kick-off formation: everyone in their own half, two of each team close to the line
        u = 6.0 + rng.random(P) * 34.0
        forwards = [9, 10, 20, 21]
        u[forwards] = 1.6 + rng.random(4) * 0.4
        pos = np.stack([half_x + d * side * u, 5.0 + rng.random(P) * 58.0], axis=1)
        for t in range(start, kick):
            pos = pos + (rng.random((P, 2)) - 0.5) * 0.02
            pxy[t] = pos
            bxy[t] = (half_x, W / 2.0)
        # live play with momentum (m per frame); the kick sends both teams forward
        vel = np.zeros((P, 2))
        vel[:, 0] = -d * side * 0.15
        vel[forwards, 0] = -d * side[forwards] * 0.6    # the forwards cross halfway at once
        carrier = 9                                     # a home forward takes the kick
        next_change = kick + int(rng.integers(50, 400))
        pass_from, pass_to, pass_start, pass_end = None, None, -1, -1
        stop_until = -1
        next_stop = kick + int(rng.integers(1500, 3000))
        for t in range(kick, end):
            stopped = t < stop_until
            calm = 0.85 if stopped else 0.97
            noise = 0.01 if stopped else 0.05
            vel = vel * calm + (rng.random((P, 2)) - 0.5) * noise
            pos = np.clip(pos + vel, (0.5, 0.5), (L - 0.5, W - 0.5))
            pxy[t] = pos
            if t == next_stop:                          # ball goes out: a stoppage of 6-24 s
                stop_until = t + int(rng.integers(150, 600))
                next_stop = stop_until + int(rng.integers(1500, 3000))
            if stopped:
                continue                                # the ball is not tracked in a stoppage
            if t >= next_change and t >= pass_end:      # a pass (70%) or a turnover (30%)
                mates = np.flatnonzero(team == team[carrier])
                others = np.flatnonzero(team != team[carrier])
                pool = mates if rng.random() < 0.7 else others
                new = int(pool[int(rng.integers(0, len(pool)))])
                if new != carrier:
                    pass_from = pos[carrier].copy()
                    pass_to, carrier = new, new
                    pass_start, pass_end = t, t + int(rng.integers(8, 30))
                next_change = t + int(rng.integers(50, 400))
            if t < pass_end:
                f = (t - pass_start) / (pass_end - pass_start)
                bxy[t] = pass_from + (pos[pass_to] - pass_from) * f
            else:
                bxy[t] = pos[carrier] + (0.6, 0.0)

    # provider speeds from positions (exactly rounded operations only)
    def speed(xy):
        dlt = np.zeros(xy.shape[:-1])
        df = xy[1:] - xy[:-1]
        dlt[1:] = np.sqrt(df[..., 0] * df[..., 0] + df[..., 1] * df[..., 1]) * fps
        return dlt

    ps = speed(pxy)
    bs = speed(bxy)
    # tracking gaps: the ball lost in runs, each player lost a few times
    t = 0
    vis = np.zeros(n, bool)
    while t < n:
        on = int(rng.integers(40, 500))
        vis[t:t + on] = True
        t += on + int(rng.integers(5, 120))
    bxy[~vis] = np.nan
    bs[~vis] = np.nan
    for j in range(P):
        for _ in range(int(rng.integers(0, 4))):
            s = int(rng.integers(0, n))
            e = s + int(rng.integers(10, 200))
            pxy[s:e, j] = np.nan
            ps[s:e, j] = np.nan
    f32 = lambda x: np.ascontiguousarray(x, dtype=np.float32)   # the loader's layout and type
    return Match(name="synthetic", fps=fps, pitch=(L, W), ball_xy=f32(bxy), ball_s=f32(bs),
                 players_xy=f32(pxy), players_s=f32(ps), team=team, half=half,
                 attack_dir={1: 1, 2: -1}, meta={"source": "synthetic"})


# ----------------------------------------------------------------------------- the chain
def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def run_chain():
    m = synthetic_match()
    found = detect_all(m)                       # as solution/atd.py does for the real match
    kick = {}
    for h, s, _ in m.half_blocks():
        k, d = found[h]
        if k is not None:
            m.attack_dir[h] = d
        kick[h] = s if k is None else k
    m.meta["kickoff"] = kick
    F = compute_features(m)
    cfg = load_pipeline()
    parts = member_probs(cfg, m, F)
    import lightgbm as lgb
    lg = next(x for x in cfg["members"] if x["kind"] == "lgbm")
    booster = lgb.Booster(model_file=os.path.join(MODEL_DIR, lg["file"]))
    cols = json.load(open(os.path.join(MODEL_DIR, lg["features"])))
    raw = booster.predict(F[cols], raw_score=True)
    blend = blended(parts)
    codes, prob, _ = labels_from(parts, cfg, m)
    grid = np.rint(np.nan_to_num(prob, nan=-1.0) * QUANT).astype(np.int32)
    v = blend[np.isfinite(blend)] * QUANT
    margin = float(np.abs(v - np.floor(v) - 0.5).min() / QUANT)
    rows = F.index.values[::max(1, len(F) // N_SAMPLE)][:N_SAMPLE]
    names = [x.get("name", x["kind"]) for x in cfg["members"]]
    return {
        "seed": SEED,
        "frames": int(m.n),
        "features": {"shape": list(F.shape), "sha256": sha(F.values),
                     "columns_sha256": hashlib.sha256("|".join(F.columns).encode()).hexdigest()},
        "kickoff": {str(h): [None if k is None else int(k), None if d is None else int(d)]
                    for h, (k, d) in found.items()},
        "lightgbm_raw_sha256": sha(raw),
        "sample_frames": [int(r) for r in rows],
        "members": {nm: [[float(x) for x in p[r]] for r in rows] for nm, (p, _) in zip(names, parts)},
        "grid_sha256": sha(grid),
        "labels_sha256": sha(codes),
        "label_counts": {c: int((codes == k).sum()) for c, k in (("H", H), ("A", A), ("D", D))},
        "margin": margin,
    }


def environment():
    import lightgbm
    import numba
    import pandas
    import torch
    return {"python": platform.python_version(), "machine": platform.machine(),
            "system": "{} {}".format(platform.system(), platform.release()),
            "numpy": np.__version__, "pandas": pandas.__version__, "lightgbm": lightgbm.__version__,
            "numba": numba.__version__, "torch": torch.__version__}


def compare(got, ref):
    ok = True

    def line(item, same, detail=""):
        nonlocal ok
        ok &= same
        print("{:<5} {}{}".format("SAME" if same else "DIFF", item, "  -- " + detail if detail else ""),
              flush=True)

    line("synthetic match ({} frames, seed {})".format(got["frames"], got["seed"]),
         got["frames"] == ref["frames"] and got["seed"] == ref["seed"])
    k = got["kickoff"]
    line("kick-off detection", k == ref["kickoff"], ", ".join(
        "half {}: {}".format(h, "frame {} dir {:+d}".format(*v) if v[0] is not None else "not found")
        for h, v in sorted(k.items())))
    fs = got["features"]
    line("features {} x {}, bit for bit".format(*fs["shape"]), fs == ref["features"])
    line("LightGBM raw scores, bit for bit", got["lightgbm_raw_sha256"] == ref["lightgbm_raw_sha256"])
    same_rows = got["sample_frames"] == ref["sample_frames"]
    for nm, vals in got["members"].items():
        r = np.array(ref["members"].get(nm, []), dtype=np.float64)
        g = np.array(vals, dtype=np.float64)
        if not same_rows or r.shape != g.shape:
            line("member {}".format(nm), False, "sampled frames do not match the reference")
            continue
        both = np.isfinite(r) & np.isfinite(g)
        diff = float(np.abs(r[both] - g[both]).max()) if both.any() else 0.0
        line("member {} probabilities ({} sampled frames)".format(nm, len(g)),
             bool((np.isfinite(r) == np.isfinite(g)).all()) and diff <= GRU_TOL,
             "largest difference {:.1e}".format(diff))
    line("blended probabilities on the 1e-4 grid", got["grid_sha256"] == ref["grid_sha256"],
         "nearest rounding boundary {:.1e} away".format(got["margin"]))
    line("decoded labels", got["labels_sha256"] == ref["labels_sha256"],
         "H {H}, A {A}, D {D} frames".format(**got["label_counts"]))
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--record", action="store_true", help="write the reference from this machine")
    a = ap.parse_args()
    env = environment()
    print("environment:", ", ".join("{} {}".format(k, v) for k, v in env.items()), flush=True)
    got = run_chain()
    if a.record:
        with open(REF, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(dict(got, environment=env), fh, indent=1)
            fh.write("\n")
        print("reference written:", os.path.relpath(REF, ROOT))
        return 0
    ref = json.load(open(REF, encoding="utf-8"))
    print("reference made on:", ", ".join("{} {}".format(k, v)
                                          for k, v in ref.get("environment", {}).items()))
    ok = compare(got, ref)
    print()
    print("RESULT: this machine computes the same as the reference machine." if ok else
          "RESULT: differences found (see DIFF lines). Please report them with the lines above.")
    return 0 if ok else 1


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)   # same reason as in solution/predict.py
