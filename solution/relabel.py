"""Convert per-touch possession labels into *controlled* possession, per the challenge definition.

External per-frame possession (DFL BallPossession) flips on every touch: its spells have a median
of 3.7 s and 30% are under 2 s. The challenge defines possession differently (CHALLENGE.md):

    A change of possession occurs when a player from the opposing team gains CONTROLLED
    possession ... a simple touch without control, a deflection that does not intercept, an
    aerial duel won without subsequent control do not count ... in phases where the ball is
    contested or loose, possession stays with the last team that had it.
    Proxies: a player keeps the ball for a few seconds; a team completes > 2 passes.

So a spell of team Y that interrupts team X's possession and lasts less than CONTROL_S seconds is
not a gain of control: it is relabelled X. Spells that start from a dead ball (restarts) are kept
-- after a stoppage there is always a clear possessor. Shortest spells are resolved first.

CONTROL_S is a definitional constant ("a few seconds"), chosen from the text -- not tuned to any
statistic of the provided match.
"""

import heapq

import numpy as np

from solution.match import H, A, D, X

CONTROL_S = 2.0


def _segments(lab):
    cut = np.flatnonzero(np.diff(lab) != 0) + 1
    bounds = np.concatenate(([0], cut, [len(lab)]))
    return [[int(lab[s]), int(s), int(e)] for s, e in zip(bounds[:-1], bounds[1:])]


def controlled(labels, half, fps=25, control_s=CONTROL_S):
    out = labels.copy()
    min_len = int(round(control_s * fps))
    for h in (1, 2):
        idx = np.flatnonzero(half == h)
        if not len(idx):
            continue
        a, b = int(idx[0]), int(idx[-1]) + 1
        segs = _segments(out[a:b])
        changed = True
        while changed:
            changed = False
            # candidates: short possession spells whose predecessor is the other team, alive
            heap = []
            for i, (k, s, e) in enumerate(segs):
                if k in (H, A) and e - s < min_len and i > 0:
                    pk = segs[i - 1][0]
                    if pk in (H, A) and pk != k:
                        heapq.heappush(heap, (e - s, i))
            if not heap:
                break
            _, i = heapq.heappop(heap)
            segs[i][0] = segs[i - 1][0]
            # merge equal neighbours
            merged = [segs[0]]
            for seg in segs[1:]:
                if seg[0] == merged[-1][0]:
                    merged[-1][2] = seg[2]
                else:
                    merged.append(seg)
            segs = merged
            changed = True
        for k, s, e in segs:
            out[a + s:a + e] = k
    return out


def spell_stats(labels, half, fps=25):
    lens = []
    for h in (1, 2):
        idx = np.flatnonzero(half == h)
        if not len(idx):
            continue
        for k, s, e in _segments(labels[idx[0]:idx[-1] + 1]):
            if k in (H, A):
                lens.append(e - s)
    lens = np.array(lens) / float(fps)
    return len(lens), float(lens.mean()), float(np.median(lens))


if __name__ == "__main__":
    from solution.external_dfl import load_dfl, MATCH_IDS
    ms = [load_dfl(mid) for mid in MATCH_IDS]
    n, mean, med = zip(*[spell_stats(m.labels, m.half) for m in ms])
    print("raw DFL       : spells/match {:.0f}  mean {:.1f} s  median {:.1f} s".format(
        np.mean(n), np.mean(mean), np.mean(med)))
    for c in (1.0, 2.0, 3.0):
        stats = [spell_stats(controlled(m.labels, m.half, control_s=c), m.half) for m in ms]
        n, mean, med = zip(*stats)
        print("control {:.0f} s   : spells/match {:.0f}  mean {:.1f} s  median {:.1f} s".format(
            c, np.mean(n), np.mean(mean), np.mean(med)))
