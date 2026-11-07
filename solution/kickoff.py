"""Find each half's kick-off from the tracking alone -- no published frame numbers.

At a kick-off every player stands in his own half; the formation holds for several seconds and
breaks the moment the ball is kicked (attackers cross halfway). That is the one time in a half
when both teams are fully separated by the halfway line, so:

1. formation frames: >= SIDE_SHARE of each team's tracked players on their own side of halfway
   (with TOL metres of tolerance for players standing on the line);
2. the kick-off formation is the first run of at least MIN_RUN such frames in the search window;
   its side tells us which goal each team defends (the attacking direction);
3. the kick is the end of that run, moved to the ball's first clear movement if the ball is seen
   within a second of it.

The tracker rarely sees the ball sitting on the spot before the kick, so the ball is not required.
Used for: frames of a half's play window before its kick-off are a stoppage (the second half's
graded window opens ~2 minutes before the kick), and the home attacking direction per half.
"""

import warnings

import numpy as np

# Search range around a half's play window: structural, from the organisers' published grading
# windows (DATA.md / FAQ.md: a window may open at the kick or a couple of minutes before it).
SEARCH_BEFORE = 250        # frames looked at before the window opens
SEARCH_AFTER = 25 * 600    # frames after the window opens (10 minutes)
TOL = 1.5                  # m of tolerance around halfway (external choice)
SIDE_SHARE = 0.85         # chosen on external kick-offs (solution/kickoff_check.py rederive)
MIN_PER_TEAM = 6           # external choice
MIN_RUN = 25               # frames the formation must hold (external choice)
MERGE_GAP = 50             # frames: short breaks do not end it (external choice)
REFINE = 25                # frames around the run end to look for the ball's first movement
KICK_SPEED = 3.0           # m/s
SPEED_WIN = 50             # frames each side of the formation's end for the speed-rise test
SPEED_RISE = 0.2           # m/s: mean player speed must rise when the formation breaks (external)
PLAY_SPEED = 1.2           # m/s: ... into live play (external choice)


def _formation(m, a, b):
    L = m.pitch[0]
    x = m.players_xy[a:b, :, 0]
    home = x[:, m.team == 0]
    away = x[:, m.team == 1]
    nh = np.isfinite(home).sum(axis=1)
    na = np.isfinite(away).sum(axis=1)
    enough = (nh >= MIN_PER_TEAM) & (na >= MIN_PER_TEAM)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        def share(arr, cond):
            return np.nanmean(np.where(np.isfinite(arr), cond, np.nan), axis=1)
        pos = enough & (share(home, home < L / 2 + TOL) >= SIDE_SHARE) & \
            (share(away, away > L / 2 - TOL) >= SIDE_SHARE)
        neg = enough & (share(home, home > L / 2 - TOL) >= SIDE_SHARE) & \
            (share(away, away < L / 2 + TOL) >= SIDE_SHARE)
    return pos, neg


def _runs(mask, min_run, merge_gap=MERGE_GAP):
    """Runs of >= min_run True frames, after merging runs separated by short gaps."""
    m = np.concatenate(([False], mask, [False])).astype(np.int8)
    d = np.diff(m)
    starts, ends = list(np.flatnonzero(d == 1)), list(np.flatnonzero(d == -1))
    merged = []
    for s, e in zip(starts, ends):
        if merged and s - merged[-1][1] <= merge_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(int(s), int(e)) for s, e in merged if e - s >= min_run]   # [s, e)


def detect(m, start, stop):
    """(kick_frame, home_attack_dir) for the half window [start, stop), or (None, None)."""
    a = max(0, start - SEARCH_BEFORE)
    b = min(stop, start + SEARCH_AFTER)
    pos, neg = _formation(m, a, b)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        speed = np.nanmean(m.players_s[a:b], axis=1)
    cands = []
    for mask, direction in ((pos, 1), (neg, -1)):
        for s, e in _runs(mask, MIN_RUN):
            before = np.nanmean(speed[max(0, e - SPEED_WIN):e])
            after = np.nanmean(speed[e:e + SPEED_WIN])
            # the kick-off formation is the one that breaks into play
            if after - before >= SPEED_RISE and after >= PLAY_SPEED:
                cands.append((s, e, direction))
    if not cands:
        return None, None
    s, e, direction = min(cands)
    kick = a + e - 1                      # last frame of the formation
    # refine with the ball: first clear movement within +/- REFINE frames of the run end
    lo, hi = max(0, kick - REFINE), min(m.n - 1, kick + REFINE)
    bx = m.ball_xy[lo:hi + 1]
    sp = np.hypot(*np.diff(bx, axis=0).T) * m.fps
    moving = np.flatnonzero(np.nan_to_num(sp, nan=0.0) > KICK_SPEED)
    if len(moving):
        kick = lo + int(moving[0])
    return int(kick), direction


def detect_all(m):
    return {h: detect(m, s, e) for h, s, e in m.half_blocks()}


if __name__ == "__main__":
    # External validation lives in solution/kickoff_check.py; here we only say whether the
    # detector finds a kick-off in each half of the provided match.
    from solution.atd import load_atd
    m = load_atd("tactical")
    print({h: (k is not None) for h, k in m.meta["kickoff"].items()})
