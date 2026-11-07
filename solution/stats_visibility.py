"""NOT PART OF THE SUBMISSION CHAIN -- development diagnostic (aggregate visibility statistics; history of the v4 calibration). Kept for transparency; solution/predict.py never
imports it.

Aggregate statistics of ball / player visibility in a Match. Prints quantiles and rates only.

Used to fit the degradation applied to external games so that they look like automatic tracking.
It reads tracking only -- no labels -- and is unsupervised with respect to possession.
"""

import numpy as np


def runs(mask):
    """Lengths of consecutive True runs in a boolean array."""
    m = np.concatenate(([False], mask, [False])).astype(np.int8)
    d = np.diff(m)
    return np.flatnonzero(d == -1) - np.flatnonzero(d == 1)


def summary(m, frames_mask=None):
    sel = np.ones(m.n, bool) if frames_mask is None else frames_mask
    out = {}
    vis = np.isfinite(m.ball_xy[:, 0])
    for h, a, b in m.half_blocks():
        pass
    vis_runs, miss_runs = [], []
    for h, a, b in m.half_blocks():
        s = sel[a:b]
        v = vis[a:b][s]
        vis_runs.append(runs(v))
        miss_runs.append(runs(~v))
    vr = np.concatenate(vis_runs)
    mr = np.concatenate(miss_runs)
    q = [10, 25, 50, 75, 90, 99]
    out["ball_visible_rate"] = float(vis[sel].mean())
    out["visible_run_mean"] = float(vr.mean())
    out["visible_run_q"] = np.percentile(vr, q).round(1).tolist()
    out["missing_run_mean"] = float(mr.mean())
    out["missing_run_q"] = np.percentile(mr, q).round(1).tolist()
    out["missing_frames_in_runs_gt_25"] = float(mr[mr > 25].sum() / max(1, mr.sum()))
    out["missing_frames_in_runs_gt_125"] = float(mr[mr > 125].sum() / max(1, mr.sum()))
    tracked = np.isfinite(m.players_xy[:, :, 0]).sum(axis=1)[sel]
    out["players_tracked_mean"] = float(tracked.mean())
    out["players_tracked_q"] = np.percentile(tracked, q).tolist()
    # Visibility vs where the ball was last seen (distance to the boundary), and vs crowding.
    L, W = m.pitch
    bx, by = m.ball_xy[:, 0], m.ball_xy[:, 1]
    inb = np.minimum(np.minimum(bx, L - bx), np.minimum(by, W - by))
    out["share_seen_outside_pitch"] = float(np.mean(inb[sel & vis] < 0))
    return out


if __name__ == "__main__":
    from solution.atd import load_atd
    m = load_atd("tactical")
    s = summary(m, m.half > 0)
    for k, v in s.items():
        print("{:<32} {}".format(k, v))
