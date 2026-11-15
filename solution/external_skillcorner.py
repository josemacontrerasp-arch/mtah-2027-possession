"""Loader for SkillCorner's open data (20 A-League 2024/25 matches; MIT licence) -> Match with labels.

    SkillCorner open data, https://github.com/SkillCorner/opendata (MIT licence). Tracking and
    phases of play by SkillCorner.

Broadcast tracking at 10 fps, produced automatically from the TV feed: a player or the ball is
`is_detected` only while the tracker sees it, and SkillCorner fills the rest with extrapolated
positions. We keep detections only, so the gaps are those of a real automatic tracker -- which is
why this data is useful next to the DFL games, whose tracking is complete and is degraded by a
simulator (solution/degrade_random.py).

Labels come from SkillCorner's phases of play, which "are only defined when the ball is in play"
(SkillCorner's README): a frame inside a phase belongs to the team in possession of that phase,
a frame of a period outside every phase is dead ball, and the frames of a period before its first
phase or after its last are left unlabelled (X), so the approach to a kick-off is never taught as
a stoppage. solution/train.py then applies the same controlled-possession relabel as to the DFL
labels (solution/relabel.py).

Resampling to the 25 fps of `Match`: linear interpolation between consecutive 10 fps samples, and
only where both are detections; labels and halves take the nearest 10 fps frame. Speeds come from
central differences of the 10 fps positions. Coordinates are moved from SkillCorner's centre origin
to the corner origin of `Match`.
"""

import json
import os

import numpy as np
import pandas as pd

from solution.match import Match, H, A, D, X, estimate_attack_dir

EXT_DIR = os.environ.get("MTAH_EXTERNAL_DIR") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "external")
SC_DIR = os.path.join(EXT_DIR, "skillcorner-opendata", "data", "matches")
SC_CACHE = os.path.join(EXT_DIR, "skillcorner-opendata", "cache")
SC_FPS = 10
FPS = 25
SC_IDS = ["1874553", "1886347", "1899585", "1925299", "1927964", "1953632", "1959846", "1986691",
          "1996435", "1996436", "2006229", "2006363", "2007448", "2007721", "2010085", "2011166",
          "2013725", "2015213", "2016236", "2017461"]


def _files(mid):
    d = os.path.join(SC_DIR, mid)
    return (os.path.join(d, mid + "_match.json"), os.path.join(d, mid + "_tracking_extrapolated.jsonl"),
            os.path.join(d, mid + "_phases_of_play.csv"))


def _parse10(mid):
    """Arrays at SkillCorner's own 10 fps, with labels."""
    meta_path, trk_path, ph_path = _files(mid)
    meta = json.load(open(meta_path, encoding="utf-8"))
    L, W = float(meta["pitch_length"]), float(meta["pitch_width"])
    home_id = meta["home_team"]["id"]
    away_id = meta["away_team"]["id"]
    players = [p for p in meta["players"] if p["team_id"] in (home_id, away_id)]
    slot_of = {p["id"]: j for j, p in enumerate(players)}
    team = np.array([0 if p["team_id"] == home_id else 1 for p in players], dtype=np.int8)
    gk = np.array([isinstance(p["player_role"], dict) and p["player_role"].get("name") == "Goalkeeper"
                   for p in players], dtype=bool)

    rows =[json.loads(line) for line in open(trk_path, encoding="utf-8")]
    n = len(rows)
    if [r["frame"] for r in rows] != list(range(n)):
        raise ValueError("{}: tracking frames are not 0..n-1".format(mid))
    ball = np.full((n, 2), np.nan, np.float64)
    pxy = np.full((n, len(players), 2), np.nan, np.float64)
    period = np.zeros(n, np.int8)
    unknown = set()
    for i, r in enumerate(rows):
        if r["period"] in (1, 2):
            period[i] = r["period"]
        b = r["ball_data"]
        if b["is_detected"] and b["x"] is not None and b["y"] is not None:
            ball[i] = (b["x"] + L / 2.0, b["y"] + W / 2.0)
        for p in r["player_data"]:
            j = slot_of.get(p["player_id"])
            if j is None:
                unknown.add(p["player_id"])
                continue
            if p["is_detected"] and p["x"] is not None and p["y"] is not None:
                pxy[i, j] = (p["x"] + L / 2.0, p["y"] + W / 2.0)
    del rows
    if unknown:
        raise ValueError("{}: tracked ids missing from match.json: {}".format(mid, sorted(unknown)))

    lab = np.full(n, X, np.int8)
    ph = pd.read_csv(ph_path)
    for per in (1, 2):
        idx = np.flatnonzero(period == per)
        if not len(idx):
            raise ValueError("{}: no frames in period {}".format(mid, per))
        q = ph[ph["period"] == per].sort_values("frame_start")
        if not len(q):
            raise ValueError("{}: no phases in period {}".format(mid, per))
        first, last = int(q["frame_start"].min()), int(q["frame_end"].max())
        # dead ball inside the period, between its first and last phase ...
        lo, hi = max(int(idx[0]), first), min(int(idx[-1]), last)
        lab[lo:hi + 1] = D
        # ... then every phase, in order, gives its frames to the team in possession
        for _, r in q.iterrows():
            tid = r["team_in_possession_id"]
            k = H if tid == home_id else A if tid == away_id else X
            a, b = max(int(r["frame_start"]), lo), min(int(r["frame_end"]), hi)
            if b >= a:
                lab[a:b + 1] = k
    lab[period == 0] = X

    side = meta["home_team_side"]
    attack_dir = {h: (1 if side[h - 1] == "left_to_right" else -1) for h in (1, 2)}
    return dict(ball=ball, pxy=pxy, period=period, labels=lab, team=team, gk=gk, pitch=(L, W),
                attack_dir=attack_dir)


def _speed10(xy):
    """|central difference| / 0.2 s at 10 fps; NaN unless both neighbours are detections."""
    v = np.full(xy.shape[:-1], np.nan, np.float64)
    d = xy[2:] - xy[:-2]
    v[1:-1] = np.sqrt(d[..., 0] * d[..., 0] + d[..., 1] * d[..., 1]) * (SC_FPS / 2.0)
    return v


def _resample(a, u):
    """Linear interpolation of a 10 fps array at fractional source positions u (NaN-strict)."""
    f0 = np.floor(u).astype(np.int64)
    w = u - f0
    f1 = np.minimum(f0 + 1, len(a) - 1)
    shape = (-1,) + (1,) * (a.ndim - 1)
    w = w.reshape(shape)
    out = a[f0] * (1.0 - w) + a[f1] * w
    exact = (w == 0)
    return np.where(exact, a[f0], out)


def _to25(z10):
    n10 = len(z10["period"])
    n25 = (n10 - 1) * FPS // SC_FPS + 1
    u = np.arange(n25, dtype=np.float64) * (SC_FPS / float(FPS))
    near = np.rint(u).astype(np.int64)
    ball_xy = _resample(z10["ball"], u)
    pxy = _resample(z10["pxy"], u)
    ball_s = _resample(_speed10(z10["ball"]), u)
    ps = _resample(_speed10(z10["pxy"]), u)
    period = z10["period"][near]
    return dict(ball_xy=ball_xy.astype(np.float32), ball_s=ball_s.astype(np.float32),
                pxy=pxy.astype(np.float32), ps=ps.astype(np.float32),
                half=period.astype(np.int8), labels=z10["labels"][near].astype(np.int8),
                team=z10["team"], gk=z10["gk"], pitch=np.array(z10["pitch"], np.float64),
                attack_dir=np.array([z10["attack_dir"][1], z10["attack_dir"][2]], np.int8))


def load_skillcorner(mid, use_cache=True):
    os.makedirs(SC_CACHE, exist_ok=True)
    cache = os.path.join(SC_CACHE, "{}_v1.npz".format(mid))
    if use_cache and os.path.isfile(cache):
        z = dict(np.load(cache))
    else:
        z = _to25(_parse10(mid))
        np.savez_compressed(cache, **z)
    L, W = float(z["pitch"][0]), float(z["pitch"][1])
    m = Match(name="SC-" + mid, fps=FPS, pitch=(L, W), ball_xy=z["ball_xy"], ball_s=z["ball_s"],
              players_xy=z["pxy"], players_s=z["ps"], team=z["team"], half=z["half"],
              labels=z["labels"], ball_z=None,
              meta={"source": "skillcorner", "match_id": mid, "gk": z["gk"]})
    m.attack_dir = {1: int(z["attack_dir"][0]), 2: int(z["attack_dir"][1])}
    return m


def check_attack_dir(m):
    """SkillCorner's home_team_side against where the two starting goalkeepers stand: the home
    team attacks +x in a half when its goalkeeper's mean x is the lower one."""
    gk = m.meta["gk"]
    est = {}
    for h in (1, 2):
        rows = m.half == h
        x = m.players_xy[rows][:, :, 0]
        home = np.nanmean(x[:, gk & (m.team == 0)])
        away = np.nanmean(x[:, gk & (m.team == 1)])
        est[h] = (1 if home < away else -1) if np.isfinite(home) and np.isfinite(away) else 0
    given = dict(m.attack_dir)
    return given == est, given, est


if __name__ == "__main__":
    import sys
    from solution.relabel import controlled, spell_stats
    ids = sys.argv[1:] or SC_IDS
    for mid in ids:
        m = load_skillcorner(mid)
        lab = m.labels
        live = (lab == H) | (lab == A)
        inhalf = m.half > 0
        ok, given, est = check_attack_dir(m)
        n_raw, mean_raw, med_raw = spell_stats(lab, m.half)
        c = controlled(lab, m.half)
        n_c, mean_c, med_c = spell_stats(c, m.half)
        print("SC-{}  frames {:>6}  slots {:>2}  H {:.1%} A {:.1%} D {:.1%} X(in halves) {:.1%}  "
              "ball seen (live) {:.1%}  players seen (live) {:.1f}  spells {} -> {} (median {:.1f} -> "
              "{:.1f} s)  dir {} {}".format(
                  mid, m.n, len(m.team), (lab == H).mean(), (lab == A).mean(), (lab == D).mean(),
                  (lab[inhalf] == X).mean(), np.isfinite(m.ball_xy[live, 0]).mean(),
                  np.isfinite(m.players_xy[live, :, 0]).sum(axis=1).mean(), n_raw, n_c, med_raw,
                  med_c, given, "ok" if ok else "MISMATCH est {}".format(est)), flush=True)
