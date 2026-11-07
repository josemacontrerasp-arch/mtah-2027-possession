"""External validation of the kick-off detector (solution/kickoff.py) -- DFL games only.

DFL halves start at the kick, but every kick-off after a goal is preceded by the lined-up
formation, and DFL's BallStatus gives the exact restart frame. We find those restarts (a stoppage
whose restart is on the centre spot), open a search window a minute before each, run the detector
unchanged on randomly degraded tracking (solution/degrade_random.py), and report the error.

    python -m solution.kickoff_check
"""

import numpy as np

from solution.degrade_random import degrade_random
from solution.external_dfl import MATCH_IDS
from solution.kickoff import detect
from solution.match import D
from solution.train import load_ext

WINDOW_BEFORE = 25 * 60     # the "play window" opens a minute before the kick
WINDOW_AFTER = 25 * 120


def centre_restarts(m):
    """Restart frames (first live frame after a stoppage) located on the centre spot."""
    L, W = m.pitch
    out = []
    for h, a, b in m.half_blocks():
        dead = np.concatenate(([False], m.labels[a:b] == D, [False])).astype(np.int8)
        ends = np.flatnonzero(np.diff(dead) == -1)
        starts = np.flatnonzero(np.diff(dead) == 1)
        for s, e in zip(starts, ends):
            r = a + e
            if r >= b or e - s < 25 * 20:          # kick-offs after goals follow long stoppages
                continue
            bx, by = m.ball_xy[r]
            if np.hypot(bx - L / 2, by - W / 2) < 1.5:
                out.append(int(r))
    return out


def main():
    errs, misses, dir_ok, n = [], 0, 0, 0
    for i, mid in enumerate(MATCH_IDS):
        clean = load_ext(mid)
        kicks = centre_restarts(clean)
        if not kicks:
            continue
        deg = degrade_random(clean, seed=7000 + i)
        deg.attack_dir = clean.attack_dir
        for r in kicks:
            n += 1
            start = max(0, r - WINDOW_BEFORE)
            k, d = detect(deg, start, min(deg.n, r + WINDOW_AFTER))
            if k is None:
                misses += 1
                continue
            errs.append(k - r)
            h = int(clean.half[r])
            dir_ok += int(d == clean.attack_dir[h])
    e = np.array(errs)
    print("after-goal kick-offs in 7 DFL games: {}; detected {}; missed {}".format(
        n, len(e), misses))
    if len(e):
        print("error vs true restart (frames): median {:+.0f}, |median| {:.0f}, max |err| {:.0f}; "
              "within 1 s: {:.0%}; direction correct: {}/{}".format(
                  np.median(e), np.median(np.abs(e)), np.abs(e).max(),
                  np.mean(np.abs(e) <= 25), dir_ok, len(e)))


if __name__ == "__main__":
    main()
