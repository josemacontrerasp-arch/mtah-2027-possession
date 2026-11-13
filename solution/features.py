"""Per-frame features from a Match. Identical code for the provided game and external games.

Design rules
------------
* Collective features only: track slots are not people, so nothing is keyed to a slot index.
* Orientation-normalised: x is expressed in the HOME team's attacking frame (home attacks +x),
  so "ball deep in home's attacking third" means the same thing in both halves.
* NaN-aware: the ball is missing on ~41% of graded frames; LightGBM handles NaN natively, and
  every windowed feature is a NaN-aware mean so a gap shrinks the evidence instead of faking it.
* Computed per half block, so no window ever reaches across halftime.
"""

import warnings

import numpy as np
import pandas as pd

from solution.match import Match

CONTROL_RADIUS = 1.5     # m: nearest player this close to the ball is treated as on the ball
NEAR_RADIUS = 3.0
ZONE_RADIUS = 10.0
RUN_SPEED = 4.0          # m/s
STILL_SPEED = 1.0        # m/s
MAX_GAP_INTERP = 12      # frames: ball gaps up to ~0.5 s are linearly bridged for geometry
LONG_GAP = 375           # frames: v5 "ball path" interpolation across gaps up to 15 s
FEATURE_SET = 5          # 4 = original 152 features; 5 = + turnover / long-gap features


def _nan_window_mean(x, lo, hi):
    """Mean of x[t+lo .. t+hi] (inclusive) ignoring NaN; NaN where the window has no data."""
    n = len(x)
    valid = np.isfinite(x)
    v = np.where(valid, x, 0.0).astype(np.float64)
    cs = np.concatenate(([0.0], np.cumsum(v)))
    cc = np.concatenate(([0], np.cumsum(valid)))
    idx = np.arange(n)
    a = np.clip(idx + lo, 0, n)
    b = np.clip(idx + hi + 1, 0, n)
    cnt = cc[b] - cc[a]
    out = np.full(n, np.nan)
    ok = cnt > 0
    out[ok] = (cs[b] - cs[a])[ok] / cnt[ok]
    return out


def _norm2(a, b):
    """sqrt(a^2 + b^2) from correctly rounded operations only (multiply, add, square root).
    np.hypot goes through the platform's maths library, whose last bit differs between
    operating systems; these three operations give the same bits everywhere."""
    return np.sqrt(a * a + b * b)


def _interp_short_gaps(xy, max_gap):
    """Linear bridge of NaN runs no longer than max_gap that have data on both sides."""
    out = xy.copy()
    vis = np.isfinite(xy[:, 0])
    n = len(vis)
    if vis.sum() < 2:
        return out
    idx = np.flatnonzero(vis)
    gaps = np.flatnonzero(np.diff(idx) > 1)
    for g in gaps:
        a, b = idx[g], idx[g + 1]
        if b - a - 1 <= max_gap:
            t = (np.arange(a + 1, b) - a) / float(b - a)
            out[a + 1:b] = xy[a] + (xy[b] - xy[a]) * t[:, None]
    return out


def _last_next(code, n):
    """For a per-frame code array (-1 = none): last code at/before t, frames since; next code
    at/after t, frames until. Codes are carried; distances are NaN where nothing exists."""
    has = code >= 0
    idx = np.arange(n)
    last_i = np.maximum.accumulate(np.where(has, idx, -1))
    next_i = np.minimum.accumulate(np.where(has, idx, n)[::-1])[::-1]
    last_c = np.where(last_i >= 0, code[np.clip(last_i, 0, n - 1)], -1).astype(np.float64)
    next_c = np.where(next_i < n, code[np.clip(next_i, 0, n - 1)], -1).astype(np.float64)
    since = np.where(last_i >= 0, idx - last_i, np.nan).astype(np.float64)
    until = np.where(next_i < n, next_i - idx, np.nan).astype(np.float64)
    last_c[last_c < 0] = np.nan
    next_c[next_c < 0] = np.nan
    return last_c, since, next_c, until


def _team_sign_nearest(dist, team):
    """+1 if the nearest finite slot is home, -1 if away, NaN if none; and that distance."""
    ok = np.isfinite(dist).any(axis=1)
    slot = np.argmin(np.where(np.isfinite(dist), dist, np.inf), axis=1)
    sign = np.where(team[slot] == 0, 1.0, -1.0)
    sign[~ok] = np.nan
    d = np.where(ok, dist[np.arange(len(slot)), slot], np.nan)
    return sign, d


def _turnover_features(f, bxy_raw, bxy, pxy, team, seen, vx, vy, n, fps):
    """v5: features aimed at open-play turnovers and the long ball gaps.

    * path_*: the ball linearly interpolated across gaps of up to LONG_GAP frames -- which team
      lines the ball's path while the tracker has lost it;
    * ctrl1/ctrl25: control evidence at 1.0 m and 2.5 m (robustness to noise);
    * recv_sign: team of the first player ahead of a moving ball, inside a cone (pass receiver);
    * ttb_diff: who reaches the ball's position 0.6 s ahead first (home minus away, seconds).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        path = _interp_short_gaps(bxy_raw, LONG_GAP)
        dist_p = _norm2(pxy[..., 0] - path[:, None, 0], pxy[..., 1] - path[:, None, 1])
        sign_p, d_p = _team_sign_nearest(dist_p, team)
        f["path_sign"] = np.where(d_p < 3.0, sign_p, np.where(np.isfinite(d_p), 0.0, np.nan))
        dh = np.nanmin(np.where(team[None, :] == 0, dist_p, np.nan), axis=1)
        da = np.nanmin(np.where(team[None, :] == 1, dist_p, np.nan), axis=1)
        f["path_d_diff"] = np.clip(dh - da, -30, 30)
        f["in_long_gap"] = (np.isfinite(path[:, 0]) & ~seen).astype(np.float64)

        dist = _norm2(pxy[..., 0] - bxy[:, None, 0], pxy[..., 1] - bxy[:, None, 1])
        sign_b, d_b = _team_sign_nearest(dist, team)
        for r, key in ((1.0, "ctrl1_sign"), (2.5, "ctrl25_sign")):
            v = np.where(seen, 0.0, np.nan)
            close = seen & (d_b < r)
            v[close] = sign_b[close]
            f[key] = v

        # pass receiver: first player ahead of a moving ball inside a cone
        speed = _norm2(vx, vy)
        moving = seen & (speed > 4.0)
        ux, uy = vx / np.maximum(speed, 1e-6), vy / np.maximum(speed, 1e-6)
        rx = pxy[..., 0] - bxy[:, None, 0]
        ry = pxy[..., 1] - bxy[:, None, 1]
        along = rx * ux[:, None] + ry * uy[:, None]
        perp = np.abs(rx * uy[:, None] - ry * ux[:, None])
        cand = (along > 1.0) & (perp < np.maximum(2.0, 0.3 * along)) & (along < 45.0)
        score = np.where(cand, along, np.inf)
        anyc = np.isfinite(score).any(axis=1) & moving
        slot = np.argmin(score, axis=1)
        recv = np.where(team[slot] == 0, 1.0, -1.0)
        recv[~anyc] = np.nan
        recv[seen & ~moving] = 0.0
        f["recv_sign"] = recv

        # time to the ball's position 0.6 s ahead, players capped at 7 m/s
        fut = bxy + np.stack([vx, vy], axis=1) * 0.6
        dt = _norm2(pxy[..., 0] - fut[:, None, 0], pxy[..., 1] - fut[:, None, 1]) / 7.0
        th = np.nanmin(np.where(team[None, :] == 0, dt, np.nan), axis=1)
        ta = np.nanmin(np.where(team[None, :] == 1, dt, np.nan), axis=1)
        f["ttb_diff"] = np.clip(th - ta, -5, 5)


def _block_features(m: Match, start, stop, direction):
    L, W = m.pitch
    fps = m.fps
    n = stop - start
    bxy_raw = m.ball_xy[start:stop].astype(np.float64)
    pxy = m.players_xy[start:stop].astype(np.float64)
    ps = m.players_s[start:stop].astype(np.float64)
    team = m.team

    # Orientation: home attacks +x.
    if direction < 0:
        bxy_raw = np.stack([L - bxy_raw[:, 0], W - bxy_raw[:, 1]], axis=1)
        pxy = np.stack([L - pxy[..., 0], W - pxy[..., 1]], axis=-1)

    bvis = np.isfinite(bxy_raw[:, 0]).astype(np.float64)
    bxy = _interp_short_gaps(bxy_raw, MAX_GAP_INTERP)       # geometry uses bridged ball
    bok = np.isfinite(bxy[:, 0])

    f = {}
    f["bvis"] = bvis
    f["bvis_bridged"] = bok.astype(np.float64)
    f["bx"] = bxy[:, 0]
    f["by_c"] = np.abs(bxy[:, 1] - W / 2.0)
    inb = np.minimum(np.minimum(bxy[:, 0], L - bxy[:, 0]), np.minimum(bxy[:, 1], W - bxy[:, 1]))
    f["b_inbound"] = inb
    f["b_goal_d"] = np.minimum(bxy[:, 0], L - bxy[:, 0])
    f["b_touch_d"] = np.minimum(bxy[:, 1], W - bxy[:, 1])

    # Ball velocity from bridged positions (centred difference), lightly smoothed.
    v = np.full_like(bxy, np.nan)
    v[1:-1] = (bxy[2:] - bxy[:-2]) * (fps / 2.0)
    vx = _nan_window_mean(v[:, 0], -2, 2)
    vy = _nan_window_mean(v[:, 1], -2, 2)
    bspeed = _norm2(vx, vy)
    bspeed[~bok] = np.nan
    f["b_speed"] = bspeed
    f["b_vx"] = vx
    bs_prov = m.ball_s[start:stop].astype(np.float64)
    f["b_speed_prov"] = bs_prov
    f["b_acc"] = np.abs(_nan_window_mean(bspeed, 1, 5) - _nan_window_mean(bspeed, -5, -1))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # Distances from every slot to the (bridged) ball.
        dist = _norm2(pxy[..., 0] - bxy[:, None, 0], pxy[..., 1] - bxy[:, None, 1])
        dist_h = np.where(team[None, :] == 0, dist, np.nan)
        dist_a = np.where(team[None, :] == 1, dist, np.nan)
        dh = np.nanmin(dist_h, axis=1)
        da = np.nanmin(dist_a, axis=1)
        sh = np.sort(np.where(np.isfinite(dist_h), dist_h, np.inf), axis=1)
        sa = np.sort(np.where(np.isfinite(dist_a), dist_a, np.inf), axis=1)
        dh2 = np.where(np.isfinite(sh[:, 1]), sh[:, 1], np.nan)
        da2 = np.where(np.isfinite(sa[:, 1]), sa[:, 1], np.nan)
        f["d_home"] = dh
        f["d_away"] = da
        f["d_home2"] = dh2
        f["d_away2"] = da2
        f["d_diff"] = dh - da
        f["n_home_near"] = np.where(bok, np.sum(dist_h < NEAR_RADIUS, axis=1), np.nan)
        f["n_away_near"] = np.where(bok, np.sum(dist_a < NEAR_RADIUS, axis=1), np.nan)
        f["n_home_zone"] = np.where(bok, np.sum(dist_h < ZONE_RADIUS, axis=1), np.nan)
        f["n_away_zone"] = np.where(bok, np.sum(dist_a < ZONE_RADIUS, axis=1), np.nan)

        near_d = np.fmin(dh, da)
        near_team = np.where(np.isfinite(near_d), np.where(dh <= da, 0.0, 1.0), np.nan)
        f["near_d"] = near_d
        f["near_team"] = near_team

        # Relative speed of ball and nearest player (a carried ball moves with its carrier).
        valid_any = np.isfinite(dist).any(axis=1)
        slot = np.argmin(np.where(np.isfinite(dist), dist, np.inf), axis=1)
        pv = np.full((n, 2), np.nan)
        dp = pxy[2:] - pxy[:-2]                     # row k <-> frame k + 1
        pv[1:-1] = dp[np.arange(n - 2), slot[1:-1]] * (fps / 2.0)
        rel = _norm2(vx - pv[:, 0], vy - pv[:, 1])
        rel[~valid_any] = np.nan
        f["rel_speed_near"] = np.clip(rel, 0, 40)
        near_s = ps[np.arange(n), slot]
        near_s[~valid_any] = np.nan
        f["near_player_speed"] = near_s

        # Team-level kinematics and shape (always available).
        for t, tag in ((0, "home"), (1, "away")):
            cols = team == t
            x = pxy[:, cols, 0]
            y = pxy[:, cols, 1]
            s = ps[:, cols]
            f["n_" + tag] = np.sum(np.isfinite(x), axis=1).astype(np.float64)
            f["spd_" + tag] = np.nanmean(s, axis=1)
            f["run_" + tag] = np.nanmean(np.where(np.isfinite(s), s > RUN_SPEED, np.nan), axis=1)
            f["still_" + tag] = np.nanmean(np.where(np.isfinite(s), s < STILL_SPEED, np.nan),
                                           axis=1)
            f["cx_" + tag] = np.nanmean(x, axis=1)
            f["sx_" + tag] = np.nanstd(x, axis=1)
            f["sy_" + tag] = np.nanstd(y, axis=1)
            f["maxx_" + tag] = np.nanmax(x, axis=1)
            f["minx_" + tag] = np.nanmin(x, axis=1)
            # Share of the team goal-side of the ball (home defends low x, away high x).
            if t == 0:
                gs = np.nanmean(np.where(np.isfinite(x), x < bxy[:, None, 0], np.nan), axis=1)
            else:
                gs = np.nanmean(np.where(np.isfinite(x), x > bxy[:, None, 0], np.nan), axis=1)
            gs[~bok] = np.nan
            f["goalside_" + tag] = gs

        s_all = ps
        f["spd_all"] = np.nanmean(s_all, axis=1)
        f["still_all"] = np.nanmean(np.where(np.isfinite(s_all), s_all < STILL_SPEED, np.nan),
                                    axis=1)
        f["run_all"] = np.nanmean(np.where(np.isfinite(s_all), s_all > RUN_SPEED, np.nan), axis=1)
        allx = pxy[..., 0]
        ally = pxy[..., 1]
        f["cx_all"] = np.nanmean(allx, axis=1)
        f["cy_all_c"] = np.abs(np.nanmean(ally, axis=1) - W / 2.0)
        f["sx_all"] = np.nanstd(allx, axis=1)
        f["sy_all"] = np.nanstd(ally, axis=1)

    f["cx_diff"] = f["cx_home"] - f["cx_away"]
    f["spd_diff"] = f["spd_home"] - f["spd_away"]
    f["sx_diff"] = f["sx_home"] - f["sx_away"]
    f["sy_diff"] = f["sy_home"] - f["sy_away"]
    f["b_rel_home"] = f["bx"] - f["cx_home"]
    f["b_rel_away"] = f["bx"] - f["cx_away"]
    # Team centroid velocities (oriented, m/s over 1 s).
    for tag in ("home", "away", "all"):
        c = f["cx_" + tag]
        f["vcx_" + tag] = (_nan_window_mean(c, 12, 12) - _nan_window_mean(c, -12, -12)) * fps / 24.0

    # Control evidence: a player of a team within CONTROL_RADIUS of a *seen* ball.
    seen = bvis > 0
    ctrl = np.full(n, -1, dtype=np.int64)
    close = seen & np.isfinite(near_d) & (near_d < CONTROL_RADIUS)
    ctrl[close] = near_team[close].astype(np.int64)
    last_c, since, next_c, until = _last_next(ctrl, n)
    f["ctrl_last_team"] = last_c
    f["ctrl_since"] = np.minimum(since, 750)
    f["ctrl_next_team"] = next_c
    f["ctrl_until"] = np.minimum(until, 750)
    vis_code = np.where(seen, 0, -1)
    _, since_vis, _, until_vis = _last_next(vis_code, n)
    f["vis_since"] = np.minimum(since_vis, 750)
    f["vis_until"] = np.minimum(until_vis, 750)

    # Signed control indicator: +1 home on the ball, -1 away, 0 nobody close, NaN unseen.
    ctrl_sign = np.where(seen, 0.0, np.nan)
    ctrl_sign[ctrl == 0] = 1.0
    ctrl_sign[ctrl == 1] = -1.0
    f["ctrl_sign"] = ctrl_sign
    near_sign = np.where(np.isfinite(near_team), 1.0 - 2.0 * near_team, np.nan)
    near_sign[~seen] = np.nan
    f["near_sign"] = near_sign

    if FEATURE_SET >= 5:
        _turnover_features(f, bxy_raw, bxy, pxy, team, seen, vx, vy, n, fps)

    # Temporal context windows (frames), past / future / centred.
    wins = {"p1": (-25, -1), "f1": (1, 25), "p3": (-75, -1), "f3": (1, 75), "p8": (-200, -1),
            "f8": (1, 200), "c2": (-50, 50)}
    for key in ("ctrl_sign", "near_sign", "bvis", "spd_all", "still_all", "b_inbound",
                "b_speed", "d_diff", "run_all", "cx_diff", "goalside_home", "goalside_away"):
        base = f[key]
        for wname, (lo, hi) in wins.items():
            f["{}_{}".format(key, wname)] = _nan_window_mean(base, lo, hi)
    if FEATURE_SET >= 5:
        for key in ("path_sign", "ctrl1_sign", "ctrl25_sign", "recv_sign", "ttb_diff"):
            for wname in ("p1", "f1", "p3", "f3", "c2"):
                lo, hi = wins[wname]
                f["{}_{}".format(key, wname)] = _nan_window_mean(f[key], lo, hi)

    f["half_t"] = np.arange(n, dtype=np.float64) / fps
    out = pd.DataFrame(f)
    return out.astype(np.float32)


def compute_features(m: Match, frames=None):
    """Feature table indexed by frame (only frames inside a half's play window)."""
    parts = []
    for h, start, stop in m.half_blocks():
        df = _block_features(m, start, stop, m.attack_dir[h])
        df["half"] = np.float32(h)
        df.index = np.arange(start, stop)
        parts.append(df)
    out = pd.concat(parts)
    if frames is not None:
        out = out.loc[frames]
    return out
