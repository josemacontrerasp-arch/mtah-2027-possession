"""Loader for the DFL open dataset (Bassek et al., 2025; CC BY 4.0) -> Match with labels.

    Bassek, M., Rein, R., Weber, H., Memmert, D. (2025). An integrated dataset of spatiotemporal
    and event data in elite soccer. Scientific Data 12, 195. doi:10.6084/m9.figshare.28196177

Seven Bundesliga / 2. Bundesliga matches of official 25 Hz position data. Every ball frame carries
BallStatus (1 alive / 0 dead) and BallPossession (1 home / 2 away): per-frame possession labels
produced by the league's data provider -- external data about other games, which is the only
kind of labelled data the rules let us train or tune on.

The raw XML is ~400 MB per match; it is parsed once with regular expressions and cached as .npz.
DFL speeds are km/h and are converted to m/s to match the provided ATD.
"""

import glob
import os
import re

import numpy as np

from solution.match import Match, H, A, D, X, estimate_attack_dir

EXT_DIR = os.environ.get("MTAH_EXTERNAL_DIR") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "external")
DFL_DIR = os.path.join(EXT_DIR, "dfl")
CACHE_DIR = os.path.join(DFL_DIR, "cache")
MATCH_IDS = ["J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9"]

_FS = re.compile(r'<FrameSet GameSection="(\w+)" MatchId="[^"]*" TeamId="([^"]*)" '
                 r'PersonId="([^"]*)">')
_FR = re.compile(r'<Frame N="(\d+)" T="[^"]*" X="([^"]*)" Y="([^"]*)"(?: Z="([^"]*)")? '
                 r'D="[^"]*" S="([^"]*)"[^>]*?(?: BallPossession="(\d)")?(?: BallStatus="(\d)")?/>')


def _paths(match_id):
    def one(pattern):
        hits = glob.glob(os.path.join(DFL_DIR, pattern.format(match_id)))
        if len(hits) != 1:
            raise IOError("DFL file for {} not found: {}".format(match_id, pattern))
        return hits[0]
    return (one("DFL_02_01_matchinformation_*_DFL-MAT-{}.xml"),
            one("DFL_04_03_positions_raw_observed_*_DFL-MAT-{}.xml"))


def _parse(match_id):
    info_path, pos_path = _paths(match_id)
    info = open(info_path, encoding="utf-8").read()
    home_id = re.search(r'HomeTeamId="([^"]*)"', info).group(1)
    guest_id = re.search(r'GuestTeamId="([^"]*)"', info).group(1)
    L = float(re.search(r'PitchX="([\d.]+)"', info).group(1))
    W = float(re.search(r'PitchY="([\d.]+)"', info).group(1))

    text = open(pos_path, encoding="utf-8").read()
    heads = list(_FS.finditer(text))
    sets = []
    for i, h in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        rows = _FR.findall(text, h.end(), end)
        if not rows:
            continue
        arr = np.array([(r[0], r[1], r[2], r[3] or "nan", r[4], r[5] or "-1", r[6] or "-1")
                        for r in rows], dtype=np.float64)
        sets.append((h.group(1), h.group(2), h.group(3), arr))
    del text

    halves = {"firstHalf": 1, "secondHalf": 2}
    persons = []
    for section, team_id, person, _ in sets:
        if team_id in (home_id, guest_id) and person not in [p for p, _ in persons]:
            persons.append((person, 0 if team_id == home_id else 1))
    slot_of = {p: i for i, (p, _) in enumerate(persons)}
    team = np.array([t for _, t in persons], dtype=np.int8)

    blocks = []
    for section in ("firstHalf", "secondHalf"):
        ball = [s for s in sets if s[0] == section and s[1] == "BALL"]
        if len(ball) != 1:
            raise ValueError("{}: expected one ball frameset in {}".format(match_id, section))
        b = ball[0][3]
        n0, n1 = int(b[:, 0].min()), int(b[:, 0].max())
        n = n1 - n0 + 1
        ball_xy = np.full((n, 2), np.nan, np.float32)
        ball_z = np.full(n, np.nan, np.float32)
        ball_s = np.full(n, np.nan, np.float32)
        poss = np.full(n, -1, np.int8)
        status = np.full(n, -1, np.int8)
        k = b[:, 0].astype(np.int64) - n0
        ball_xy[k, 0] = b[:, 1] + L / 2.0
        ball_xy[k, 1] = b[:, 2] + W / 2.0
        ball_z[k] = b[:, 3]
        ball_s[k] = b[:, 4] / 3.6
        poss[k] = b[:, 5]
        status[k] = b[:, 6]
        pxy = np.full((n, len(persons), 2), np.nan, np.float32)
        ps = np.full((n, len(persons)), np.nan, np.float32)
        for sec, team_id, person, arr in sets:
            if sec != section or person not in slot_of:
                continue
            kk = arr[:, 0].astype(np.int64) - n0
            ok = (kk >= 0) & (kk < n)
            j = slot_of[person]
            pxy[kk[ok], j, 0] = arr[ok, 1] + L / 2.0
            pxy[kk[ok], j, 1] = arr[ok, 2] + W / 2.0
            ps[kk[ok], j] = arr[ok, 4] / 3.6
        lab = np.full(n, X, np.int8)
        lab[status == 0] = D
        lab[(status == 1) & (poss == 1)] = H
        lab[(status == 1) & (poss == 2)] = A
        blocks.append(dict(ball_xy=ball_xy, ball_z=ball_z, ball_s=ball_s, pxy=pxy, ps=ps,
                           labels=lab, half=np.full(n, halves[section], np.int8)))
    return dict(team=team, pitch=np.array([L, W]),
                **{k: np.concatenate([bl[k] for bl in blocks]) for k in blocks[0]})


def load_dfl(match_id, use_cache=True):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, "{}_v1.npz".format(match_id))
    if use_cache and os.path.isfile(cache):
        z = dict(np.load(cache))
    else:
        z = _parse(match_id)
        np.savez_compressed(cache, **z)
    L, W = float(z["pitch"][0]), float(z["pitch"][1])
    m = Match(name="DFL-" + match_id, fps=25, pitch=(L, W), ball_xy=z["ball_xy"],
              ball_s=z["ball_s"], players_xy=z["pxy"], players_s=z["ps"], team=z["team"],
              half=z["half"], labels=z["labels"], ball_z=z["ball_z"],
              meta={"source": "dfl", "match_id": match_id})
    starts = {h: int(np.flatnonzero(m.half == h)[0]) for h in (1, 2)}
    estimate_attack_dir(m, starts)
    return m


if __name__ == "__main__":
    import sys
    ids = sys.argv[1:] or MATCH_IDS
    for mid in ids:
        m = load_dfl(mid)
        lab = m.labels
        vis = np.isfinite(m.ball_xy[:, 0])
        tracked = np.isfinite(m.players_xy[:, :, 0]).sum(axis=1)
        print("{}  frames {:>6}  slots {:>2}  H {:.1%} A {:.1%} D {:.1%} X {:.1%}  ball vis {:.1%}"
              "  players/frame {:.1f}  dir {}".format(
                  mid, m.n, len(m.team), (lab == H).mean(), (lab == A).mean(), (lab == D).mean(),
                  (lab == X).mean(), vis.mean(), tracked.mean(), m.attack_dir))
