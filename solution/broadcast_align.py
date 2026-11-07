"""Align the broadcast recording to the tactical timeline, from tracking alone.

No mapping table is provided, so we estimate one. Both recordings track the same players and ball
in the same normalised pitch coordinates, so at the right time offset their positions agree.

1. Coarse search: for every candidate offset c (tactical = broadcast + c), the share of sampled
   frames where both trackers see the ball within 2 m. Two clear peaks appear -- one per half,
   because halftime is cut out of the broadcast.
2. Fine search: per half, the offset (+/- 60 frames around the peak) minimising the median
   distance between the two ball tracks.
3. Per-frame validity: a tactical frame is "mapped" only if the broadcast players at the mapped
   frame overlay the tactical players (median nearest-neighbour distance < 1.5 m). That rejects
   replays, close-ups and crowd shots automatically, without looking at a single image.

Everything here is geometric and unsupervised: no labels, no ground truth, no video.
"""

import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from solution.atd import load_atd  # noqa: E402

CACHE = os.path.join(ROOT, "data", "cache", "broadcast_map_v2.npz")
CHAMFER_OK = 1.5   # m
STEP = 5           # frames between Chamfer evaluations (held in between)


def _ball_agreement(tb, bb, b_idx, offsets, tol=2.0):
    out = np.zeros(len(offsets))
    nt = len(tb)
    bpos = bb[b_idx]
    bok = np.isfinite(bpos[:, 0])
    for i, c in enumerate(offsets):
        t_idx = b_idx + c
        ok = bok & (t_idx >= 0) & (t_idx < nt)
        if ok.sum() < 50:
            continue
        tp = tb[t_idx[ok]]
        both = np.isfinite(tp[:, 0])
        if both.sum() < 50:
            continue
        d = np.hypot(tp[both, 0] - bpos[ok][both, 0], tp[both, 1] - bpos[ok][both, 1])
        out[i] = np.mean(d < tol) * both.mean()
    return out


def _chamfer(tp, bp):
    b = bp[np.isfinite(bp[:, 0])]
    t = tp[np.isfinite(tp[:, 0])]
    if len(b) < 4 or len(t) < 8:
        return np.nan
    d = np.hypot(b[:, None, 0] - t[None, :, 0], b[:, None, 1] - t[None, :, 1]).min(axis=1)
    return float(np.median(d))


def _median_ball_dist(tb, bb, t_idx, b_idx):
    ok = (t_idx >= 0) & (t_idx < len(tb)) & (b_idx >= 0) & (b_idx < len(bb))
    t_idx, b_idx = t_idx[ok], b_idx[ok]
    both = np.isfinite(tb[t_idx, 0]) & np.isfinite(bb[b_idx, 0])
    if both.sum() < 200:
        return np.inf, int(both.sum())
    d = np.hypot(*(tb[t_idx[both]] - bb[b_idx[both]]).T)
    return float(np.median(d)), int(both.sum())


def align(tac=None, bro=None):
    tac = tac if tac is not None else load_atd("tactical")
    bro = bro if bro is not None else load_atd("broadcast")
    tb, bb = tac.ball_xy, bro.ball_xy

    sample = np.arange(0, bro.n, 5)
    offsets = np.arange(-30000, 60000, 25)
    score = _ball_agreement(tb, bb, sample, offsets)
    peaks = []
    for i in np.argsort(score)[::-1]:
        if all(abs(offsets[i] - p) > 2500 for p in peaks):
            peaks.append(int(offsets[i]))
        if len(peaks) == 2:
            break

    half_offset, half_fit = {}, {}
    for h, a, b in tac.half_blocks():
        t_idx = np.arange(a, b)
        best = (np.inf, None)
        for p in peaks:
            for dc in range(-60, 61):
                med, cnt = _median_ball_dist(tb, bb, t_idx, t_idx - (p + dc))
                if med < best[0]:
                    best = (med, p + dc)
        half_offset[h], half_fit[h] = best[1], best[0]

    t2b = np.full(tac.n, -1, dtype=np.int64)
    chamfer = np.full(tac.n, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for h, a, b in tac.half_blocks():
            c = half_offset[h]
            for t in range(a, b, STEP):
                bf = t - c
                if 0 <= bf < bro.n:
                    chamfer[t:t + STEP] = _chamfer(tac.players_xy[t], bro.players_xy[bf])
            t_idx = np.arange(a, b)
            b_idx = t_idx - c
            ok = (b_idx >= 0) & (b_idx < bro.n) & (chamfer[a:b] < CHAMFER_OK)
            t2b[t_idx[ok]] = b_idx[ok]
    return dict(peaks=np.array(peaks), offsets=np.array([half_offset[1], half_offset[2]]),
                fit=np.array([half_fit[1], half_fit[2]]), t2b=t2b, chamfer=chamfer)


def load_map(recompute=False):
    if os.path.isfile(CACHE) and not recompute:
        return dict(np.load(CACHE))
    r = align()
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, **r)
    return r


def broadcast_signals(tac, bro=None, amap=None):
    """Per tactical frame: broadcast ball (where usable), whether the broadcast is showing the
    live wide shot of this instant, and whether the frame falls inside the broadcast's span."""
    bro = bro if bro is not None else load_atd("broadcast")
    amap = amap if amap is not None else load_map()
    t2b = amap["t2b"]
    mapped = t2b >= 0
    ball = np.full((tac.n, 2), np.nan, np.float32)
    ball[mapped] = bro.ball_xy[t2b[mapped]]
    in_span = np.zeros(tac.n, bool)
    for h, a, b in tac.half_blocks():
        c = int(amap["offsets"][h - 1])
        idx = np.arange(a, b)
        in_span[idx] = (idx - c >= 0) & (idx - c < bro.n)
    return dict(ball_xy=ball, live=mapped, in_span=in_span)


if __name__ == "__main__":
    r = load_map(recompute=True)
    print("coarse peaks:", r["peaks"].tolist(), " refined offsets per half:", r["offsets"].tolist(),
          " median ball disagreement per half (m):", np.round(r["fit"], 2).tolist())
    tac = load_atd("tactical")
    s = broadcast_signals(tac)
    graded = tac.half > 0
    tvis = np.isfinite(tac.ball_xy[:, 0])
    bvis = np.isfinite(s["ball_xy"][:, 0])
    print("graded frames inside broadcast span: {:.1%}; showing the live wide shot: {:.1%}".format(
        s["in_span"][graded].mean(), s["live"][graded].mean()))
    print("ball visible on graded frames: tactical {:.1%}, either source {:.1%}".format(
        tvis[graded].mean(), (tvis | bvis)[graded].mean()))
    both = graded & tvis & bvis
    d = np.hypot(*(tac.ball_xy[both] - s["ball_xy"][both]).T)
    print("where both see the ball: median {:.2f} m, p90 {:.2f} m, share > 5 m {:.1%}".format(
        float(np.median(d)), float(np.percentile(d, 90)), float(np.mean(d > 5))))
