"""Domain-randomised degradation: a different plausible automatic tracker for every copy.

The calibrated degradation (solution/degrade.py) matches a few aggregate, unlabelled statistics
of the provided tracking. The organisers' 3-Nov email says parameters "MUST be tuned on external
data", so this module removes that dependence altogether: nothing here was fitted to, measured
on, or chosen by looking at the provided match. Every degraded copy of an external game draws its
own tracker from broad first-principles priors -- from a tracker that keeps the ball almost
always to one that loses it most of the time, ball loss tied loosely or tightly to stoppages,
noisy or smooth positions, few or many missing players -- and the model learns to be right
across that whole family. (For orientation only: the statistics the organisers publish in DATA.md
-- ball visible on ~57-59% of frames, ~20.5 tracked objects -- fall inside these priors.)

    from solution.degrade_random import degrade_random
    m2 = degrade_random(m, seed)
"""

import copy

import numpy as np

from solution.match import Match, D

AERIAL_Z = 3.0      # m: a ball this high is in flight (physics, not fitted)


def _smooth_noise(rng, shape, sigma, tau):
    """Stationary AR(1) noise along axis 0: std `sigma`, correlation time `tau` frames."""
    from scipy.signal import lfilter
    a = float(np.exp(-1.0 / tau))
    eps = rng.normal(0, 1, size=shape)
    eps[0] *= 1.0 / np.sqrt(1 - a * a)
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

# Broad priors (lo, hi). "log" entries are sampled log-uniformly.
PRIORS = dict(
    vis_run_median=(80.0, 1000.0, "log"),     # frames: visible run length in live play
    miss_run_median=(20.0, 500.0, "log"),     # frames: missing run length in live play
    run_sigma=(0.5, 1.3, "lin"),              # lognormal spread of run lengths
    p_dead_hidden=(0.2, 1.0, "lin"),          # a stoppage hides the ball (else it is tracked)
    p_show=(0.2, 1.0, "lin"),                 # hidden stoppage still shows the placed ball
    setup_max=(2.0, 20.0, "lin"),             # s: longest a placed ball is seen before a restart
    lose_early=(0.0, 3.0, "lin"),             # s
    lose_late=(0.0, 4.0, "lin"),              # s
    reacquire=(0.0, 3.0, "lin"),              # s
    aerial_drop=(0.0, 0.8, "lin"),
    player_missing=(0.01, 0.12, "lin"),       # share of frames each player is lost
    player_missing_run=(15.0, 150.0, "log"),  # frames
    group_frac=(0.0, 0.15, "lin"),            # share of frames inside a group drop-out
    group_players=(2.0, 8.0, "lin"),
    pos_noise=(0.05, 0.45, "log"),            # m
    pos_tau=(3.0, 60.0, "log"),               # frames
    speed_noise=(0.05, 0.6, "log"),           # m/s
)


# Clean stress family for EVALUATION only: the harsher half of the same priors (more and longer
# ball gaps, looser coupling of ball loss to stoppages, more noise). No value comes from the
# provided match.
HARSH = dict(PRIORS,
             vis_run_median=(80.0, 300.0, "log"),
             miss_run_median=(100.0, 500.0, "log"),
             p_dead_hidden=(0.2, 0.6, "lin"),
             lose_early=(1.5, 3.0, "lin"),
             lose_late=(2.0, 4.0, "lin"),
             reacquire=(1.5, 3.0, "lin"),
             player_missing=(0.06, 0.12, "lin"),
             pos_noise=(0.2, 0.45, "log"))


def sample_params(rng, priors=None):
    p = {}
    for k, (lo, hi, kind) in (priors or PRIORS).items():
        p[k] = float(np.exp(rng.uniform(np.log(lo), np.log(hi)))) if kind == "log" \
            else float(rng.uniform(lo, hi))
    return p


def _lognormal_mask(rng, n, med_on, med_off, sigma):
    mask = np.zeros(n, bool)
    frac = med_on / (med_on + med_off)
    t, state = 0, rng.random() < frac
    while t < n:
        run = max(1, int(np.exp(rng.normal(np.log(med_on if state else med_off), sigma))))
        if state:
            mask[t:t + run] = True
        t += run
        state = not state
    return mask


def degrade_random(m: Match, seed: int, harsh: bool = False) -> Match:
    rng = np.random.default_rng(seed)
    p = sample_params(rng, HARSH if harsh else None)
    out = copy.copy(m)
    n, fps = m.n, m.fps
    ball_xy = m.ball_xy.copy()
    ball_s = m.ball_s.copy()
    vis = np.zeros(n, bool)
    for h, a, b in m.half_blocks():
        vis[a:b] = _lognormal_mask(rng, b - a, p["vis_run_median"], p["miss_run_median"],
                                   p["run_sigma"])
        if m.labels is None:
            continue
        dead = np.concatenate(([False], m.labels[a:b] == D, [False]))
        edges = np.diff(dead.astype(np.int8))
        for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            S, E = a + s, a + e
            p0 = m.ball_xy[S - 1].astype(np.float64) if S > a else m.ball_xy[S].astype(np.float64)
            p1 = m.ball_xy[E].astype(np.float64) if E < b else m.ball_xy[E - 1].astype(np.float64)
            # DFL freezes the ball during stoppages; replace it with a plausible path: carried
            # from where play stopped to the restart spot, then placed there.
            L = E - S
            k_move = max(1, int(L * rng.uniform(0.5, 0.9)))
            t = np.clip(np.arange(L) / k_move, 0, 1)[:, None]
            ball_xy[S:E] = (p0 + (p1 - p0) * t).astype(np.float32)
            if rng.random() < p["p_dead_hidden"]:
                vis[S:E] = False
                if E < b and rng.random() < p["p_show"]:
                    k = min(L, int(rng.uniform(1.0, p["setup_max"]) * fps))
                    ball_xy[E - k:E] = m.ball_xy[E]
                    vis[E - k:E] = True
            else:
                vis[S:E] = rng.random(L) < 0.9
            d = int(rng.uniform(-p["lose_early"], p["lose_late"]) * fps)
            if d < 0:
                vis[max(a, S + d):S] = False
            elif d > 0 and S >= a + 2:
                v0 = (m.ball_xy[S - 1] - m.ball_xy[S - 2]).astype(np.float64)
                k = min(d, L)
                tt = np.arange(1, k + 1)
                ball_xy[S:S + k] = (p0 + v0[None, :] * (25.0 * (1 - np.exp(-tt / 25.0)))[:, None]
                                    ).astype(np.float32)
                vis[S:S + k] = True
            if E < b:
                k = int(rng.uniform(0, p["reacquire"]) * fps)
                vis[E:min(b, E + k)] = False
    if m.ball_z is not None and m.labels is not None:
        alive = m.labels != D
        high = np.concatenate(([False], np.nan_to_num(m.ball_z, nan=0.0) > AERIAL_Z, [False]))
        ed = np.diff(high.astype(np.int8))
        for s, e in zip(np.flatnonzero(ed == 1), np.flatnonzero(ed == -1)):
            if rng.random() < p["aerial_drop"]:
                lo, hi = max(0, s - 3), min(n, e + 3)
                vis[lo:hi] &= ~alive[lo:hi]
    ball_xy[~vis] = np.nan
    ball_s[~vis] = np.nan
    ball_xy += _smooth_noise(rng, (n, 2), p["pos_noise"], p["pos_tau"])

    pxy = m.players_xy.copy()
    ps = m.players_s.copy()
    run_off = p["player_missing_run"]
    run_on = run_off * (1 - p["player_missing"]) / p["player_missing"]
    for j in range(pxy.shape[1]):
        keep = _exp_mask(rng, n, run_on, run_off)
        pxy[~keep, j] = np.nan
        ps[~keep, j] = np.nan
    if p["group_frac"] > 1e-3:
        gr = 100.0
        group = _exp_mask(rng, n, gr, gr * (1 - p["group_frac"]) / p["group_frac"])
        g = np.concatenate(([False], group, [False])).astype(np.int8)
        for s, e in zip(np.flatnonzero(np.diff(g) == 1), np.flatnonzero(np.diff(g) == -1)):
            k = int(round(p["group_players"]))
            on = np.flatnonzero(np.isfinite(pxy[s, :, 0]))
            if len(on):
                lost = rng.choice(on, size=min(k, len(on)), replace=False)
                pxy[s:e, lost] = np.nan
                ps[s:e, lost] = np.nan
    pxy += _smooth_noise(rng, pxy.shape, p["pos_noise"], p["pos_tau"])
    ps = np.clip(ps + rng.normal(0, p["speed_noise"], size=ps.shape).astype(np.float32), 0, None)

    out.ball_xy, out.ball_s, out.players_xy, out.players_s = ball_xy, ball_s, pxy, ps
    out.ball_z = None
    out.name = "{}~rnd{}".format(m.name, seed)
    out.meta = dict(m.meta, degrade_params=p)
    return out
