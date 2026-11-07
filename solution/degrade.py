"""Make clean external tracking look like the provided automatic tracking.

External games (DFL) have the ball on every frame and all 22 players on every frame. The provided
ATD does not: the ball is missing on ~41% of graded frames, in long blocks, and ~1.5 players are
missing per frame. A model trained on clean data would lean on signals that are absent at test
time, so we degrade the external games before training and tuning.

The target statistics below are AGGREGATE visibility statistics of the provided tactical ATD
over graded frames (solution/stats_visibility.py): run-length quantiles of ball-visible and
ball-missing stretches and the mean number of tracked players. They involve no labels, no ground
truth and no video -- unsupervised domain matching on the input distribution only.
"""

import copy

import numpy as np

from solution.match import Match, H, A, D

QUANTILE_P = np.array([0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99, 1.0])
# Frames. Measured on the provided tactical ATD, graded frames (see module docstring).
VISIBLE_RUN_Q = np.array([5.0, 54.4, 137.0, 345.0, 758.0, 1542.0, 3194.0, 6000.0])
MISSING_RUN_Q = np.array([1.0, 11.8, 152.2, 268.0, 526.5, 845.3, 2619.0, 5000.0])
# The provided ATD loses the ball far more while play is stopped: P(ball visible) falls from 0.83
# when <10% of players stand still to 0.12 when >80% do (stats_visibility / still_all bins), and
# on external games standing still is what dead ball looks like. So visibility is generated per
# game state, with the two on-fractions chosen so that P(visible | share standing) on degraded
# external games reproduces that unlabelled curve. Fitted on inputs only -- no labels of BAR-ATH.
#
# DFL artefact: during every stoppage DFL freezes the ball at the spot where play stopped (100% of
# dead frames have zero ball displacement, <1% of live frames do). A real tracker shows nothing of
# the sort, so the frozen ball is removed: during a stoppage the ball is hidden, except -- with
# probability P_SHOW_SETUP -- for the last SETUP_S seconds before the restart, when it sits on the
# restart spot (a placed free kick, goal kick or corner).
P_VIS_ALIVE = 0.95
P_SHOW_SETUP = 0.8
SETUP_S = (2.0, 10.0)
JITTER = True              # v4: loss / re-acquisition timing is not frame-locked to the stoppage
LOSE_EARLY_S = 1.0
LOSE_LATE_S = 2.0
REACQUIRE_S = 1.5
DEGRADE_VERSION = 4
# Players. v4 (domain check): the ATD usually has all 11 per team (median 11) with occasional
# group losses, so dropout is a light per-player process plus bursty group drop-outs.
PLAYER_MISSING_MEAN_RUN = 50.0      # frames
PLAYER_PRESENT_MEAN_RUN = 1200.0    # frames -> ~4% per player
GROUP_DROP_FRAC = 0.10              # share of frames inside a group drop-out
GROUP_DROP_RUN = 100.0              # frames, mean length of a group drop-out
GROUP_DROP_PLAYERS = (3, 7)         # players lost together
# Positions. v4 (domain check): white noise inflated every velocity (relative ball-player speed
# median 5.1 m/s simulated vs 3.7 real); real tracks are smooth, so the noise is low-pass (AR(1)).
POS_NOISE = 0.20                    # m, stationary std
POS_NOISE_TAU = 25.0                # frames
SPEED_NOISE = 0.25                  # m/s
AERIAL_Z = 3.0                      # m: extra short dropouts while the ball is this high
AERIAL_DROP_P = 0.35


def _sample_runs(rng, q, size):
    u = rng.random(size)
    return np.maximum(1, np.round(np.exp(np.interp(u, QUANTILE_P, np.log(q))))).astype(np.int64)


def _renewal_mask(rng, n, q_on, q_off, p_start_on, scale_on=1.0, scale_off=1.0):
    """Alternating on/off runs with run lengths drawn from the two quantile functions."""
    mask = np.zeros(n, bool)
    t, state = 0, rng.random() < p_start_on
    while t < n:
        q = q_on * scale_on if state else q_off * scale_off
        run = int(_sample_runs(rng, q, 1)[0])
        if state:
            mask[t:t + run] = True
        t += run
        state = not state
    return mask


_MEAN_ON = float(np.mean(np.exp(np.interp(np.linspace(0, 1, 20001), QUANTILE_P,
                                          np.log(VISIBLE_RUN_Q)))))
_MEAN_OFF = float(np.mean(np.exp(np.interp(np.linspace(0, 1, 20001), QUANTILE_P,
                                           np.log(MISSING_RUN_Q)))))


def _mask_with_fraction(rng, n, frac):
    """Renewal mask whose expected on-fraction is `frac`, keeping the ATD run-length shapes:
    the shorter of the two run types is shrunk, the other is left as measured."""
    natural = _MEAN_ON / (_MEAN_ON + _MEAN_OFF)
    if frac >= natural:
        off = _MEAN_ON * (1 - frac) / (frac * _MEAN_OFF)
        return _renewal_mask(rng, n, VISIBLE_RUN_Q, MISSING_RUN_Q, frac, 1.0, off)
    on = frac * _MEAN_OFF / ((1 - frac) * _MEAN_ON)
    return _renewal_mask(rng, n, VISIBLE_RUN_Q, MISSING_RUN_Q, frac, on, 1.0)


def _smooth_noise(rng, shape, sigma, tau):
    """Stationary AR(1) noise along axis 0: std `sigma`, correlation time `tau` frames."""
    from scipy.signal import lfilter
    a = float(np.exp(-1.0 / tau))
    eps = rng.normal(0, 1, size=shape)
    eps[0] *= 1.0 / np.sqrt(1 - a * a)      # start in the stationary distribution
    return (lfilter([sigma * np.sqrt(1 - a * a)], [1.0, -a], eps, axis=0)).astype(np.float32)


def _exp_mask(rng, n, mean_on, mean_off):
    mask = np.zeros(n, bool)
    t, state = 0, rng.random() < mean_on / (mean_on + mean_off)
    while t < n:
        run = max(1, int(rng.exponential(mean_on if state else mean_off)))
        if state:
            mask[t:t + run] = True
        t += run
        state = not state
    return mask


# Stress profile for evaluation only (never for training): looser coupling of ball loss to the
# stoppage than any setting fitted above, to check that conclusions survive a worse tracker.
HARD = dict(p_vis_alive=0.88, lose_early=3.0, lose_late=4.0, reacquire=3.0, p_show=0.95,
            setup=(2.0, 20.0))


def degrade(m: Match, seed: int, hard: bool = False) -> Match:
    rng = np.random.default_rng(seed)
    pr = dict(p_vis_alive=P_VIS_ALIVE, lose_early=LOSE_EARLY_S, lose_late=LOSE_LATE_S,
              reacquire=REACQUIRE_S, p_show=P_SHOW_SETUP, setup=SETUP_S)
    if hard:
        pr.update(HARD)
    out = copy.copy(m)
    n = m.n
    ball_xy = m.ball_xy.copy()
    ball_s = m.ball_s.copy()
    vis = np.zeros(n, bool)
    for h, a, b in m.half_blocks():
        if m.labels is not None:
            vis[a:b] = _mask_with_fraction(rng, b - a, pr["p_vis_alive"])
            dead = np.concatenate(([False], m.labels[a:b] == D, [False]))
            edges = np.diff(dead.astype(np.int8))
            for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
                vis[a + s:a + e] = False
                if JITTER:
                    # A real tracker does not lose the ball on the exact frame play stops: it may
                    # drop it up to LOSE_EARLY_S before, or follow it rolling on for up to
                    # LOSE_LATE_S after (replacing DFL's frozen ball with a decelerating roll).
                    d = int(rng.uniform(-pr["lose_early"], pr["lose_late"]) * m.fps)
                    if d < 0:
                        vis[max(a, a + s + d):a + s] = False
                    elif d > 0 and s >= 2:
                        p0 = m.ball_xy[a + s - 1].astype(np.float64)
                        v0 = (m.ball_xy[a + s - 1] - m.ball_xy[a + s - 2]).astype(np.float64)
                        k = min(d, e - s)
                        tau = 25.0
                        t = np.arange(1, k + 1)
                        ball_xy[a + s:a + s + k] = p0 + v0[None, :] * (tau * (1 - np.exp(-t / tau)))[:, None]
                        vis[a + s:a + s + k] = True
                if a + e < b and rng.random() < pr["p_show"]:
                    k = min(e - s, int(rng.uniform(*pr["setup"]) * m.fps))
                    ball_xy[a + e - k:a + e] = m.ball_xy[a + e]
                    vis[a + e - k:a + e] = True
                elif JITTER and a + e < b:
                    # no placed ball: the tracker re-acquires it shortly after the restart
                    k = int(rng.uniform(0, pr["reacquire"]) * m.fps)
                    vis[a + e:min(b, a + e + k)] = False
        else:
            vis[a:b] = _renewal_mask(rng, b - a, VISIBLE_RUN_Q, MISSING_RUN_Q, 0.6)
    if m.labels is not None:
        alive = m.labels != D
    else:
        alive = np.ones(n, bool)
    if m.ball_z is not None:
        # Whole aerial episodes are lost together (a tracker loses a lofted ball for the flight,
        # not for random single frames).
        high = np.concatenate(([False], np.nan_to_num(m.ball_z, nan=0.0) > AERIAL_Z, [False]))
        edges = np.diff(high.astype(np.int8))
        for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if rng.random() < AERIAL_DROP_P:
                lo, hi = max(0, s - 3), min(n, e + 3)
                vis[lo:hi] &= ~alive[lo:hi]
    ball_xy[~vis] = np.nan
    ball_s[~vis] = np.nan
    ball_xy += _smooth_noise(rng, (n, 2), POS_NOISE, POS_NOISE_TAU)

    pxy = m.players_xy.copy()
    ps = m.players_s.copy()
    P = pxy.shape[1]
    for j in range(P):
        keep = _exp_mask(rng, n, PLAYER_PRESENT_MEAN_RUN, PLAYER_MISSING_MEAN_RUN)
        pxy[~keep, j] = np.nan
        ps[~keep, j] = np.nan
    # bursty group drop-outs (several players lost together, e.g. out of the camera's view)
    group = _exp_mask(rng, n, GROUP_DROP_RUN, GROUP_DROP_RUN * (1 - GROUP_DROP_FRAC)
                      / GROUP_DROP_FRAC)
    g = np.concatenate(([False], group, [False])).astype(np.int8)
    for s, e in zip(np.flatnonzero(np.diff(g) == 1), np.flatnonzero(np.diff(g) == -1)):
        k = int(rng.integers(GROUP_DROP_PLAYERS[0], GROUP_DROP_PLAYERS[1] + 1))
        on = np.flatnonzero(np.isfinite(pxy[s, :, 0]))
        if len(on):
            lost = rng.choice(on, size=min(k, len(on)), replace=False)
            pxy[s:e, lost] = np.nan
            ps[s:e, lost] = np.nan
    pxy += _smooth_noise(rng, pxy.shape, POS_NOISE, POS_NOISE_TAU)
    ps = np.clip(ps + rng.normal(0, SPEED_NOISE, size=ps.shape).astype(np.float32), 0, None)

    out.ball_xy, out.ball_s, out.players_xy, out.players_s = ball_xy, ball_s, pxy, ps
    out.ball_z = None   # the provided ATD has no ball height; never let a feature see it
    out.name = "{}~deg{}".format(m.name, seed)
    return out


def swap_teams(m: Match) -> Match:
    """Same game with home and away exchanged -- doubles the data and enforces symmetry."""
    out = copy.copy(m)
    out.team = (1 - m.team).astype(np.int8)
    out.attack_dir = {h: -d for h, d in m.attack_dir.items()}
    if m.labels is not None:
        lab = m.labels.copy()
        lab[m.labels == H] = A
        lab[m.labels == A] = H
        out.labels = lab
    out.name = m.name + "~swap"
    return out
