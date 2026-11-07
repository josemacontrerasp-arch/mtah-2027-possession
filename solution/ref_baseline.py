"""The official baseline's algorithm (nearest player + majority vote), run on degraded external
games, so our model can be compared against it on equal terms -- without touching BAR-ATH.

    python -m solution.ref_baseline
"""

import warnings

import numpy as np

from solution.degrade import degrade
from solution.external_dfl import load_dfl, MATCH_IDS
from solution.match import H, A, D, X
from solution.train import EVAL_SEED, score, load_ext


def baseline_codes(m, window=50):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        d = np.hypot(m.players_xy[..., 0] - m.ball_xy[:, None, 0],
                     m.players_xy[..., 1] - m.ball_xy[:, None, 1])
        ok = np.isfinite(d).any(axis=1)
        slot = np.argmin(np.where(np.isfinite(d), d, np.inf), axis=1)
    code = np.full(m.n, -1, np.int8)
    code[ok] = m.team[slot[ok]]
    out = np.full(m.n, X, np.int8)
    for _, a, b in m.half_blocks():
        c = code[a:b]
        hc = np.concatenate(([0], np.cumsum(c == 0)))
        ac = np.concatenate(([0], np.cumsum(c == 1)))
        i = np.arange(b - a)
        lo, hi = np.maximum(0, i - window), np.minimum(b - a, i + window + 1)
        nh, na = hc[hi] - hc[lo], ac[hi] - ac[lo]
        lab = np.full(b - a, D, np.int8)
        seen = (nh + na) > 0
        lab[seen] = np.where(nh[seen] >= na[seen], H, A)
        out[a:b] = lab
    return out


if __name__ == "__main__":
    rows = []
    for mid in MATCH_IDS:
        m = degrade(load_ext(mid), seed=EVAL_SEED)
        s = score(m.labels, baseline_codes(m), m.half)
        rows.append((s["spellF1"], s["macroF1"]))
        print("{}  baseline_smooth50  spellF1 {:.4f}  macroF1 {:.4f}  spells {}/{}".format(
            mid, s["spellF1"], s["macroF1"], s["n_pred"], s["n_gt"]))
    r = np.array(rows)
    print("MEAN baseline_smooth50 on degraded DFL: spellF1 {:.4f}  macroF1 {:.4f}".format(
        *r.mean(axis=0)))
