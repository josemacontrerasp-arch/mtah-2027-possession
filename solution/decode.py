"""Turn per-frame class probabilities into possession spells.

A three-state HMM (H, A, D) decoded with Viterbi, one half at a time. Emissions are the
classifier's log-posteriors; the only parameters are switching costs, which control how readily
the path changes state -- and therefore how many possessions it reports. They are tuned on
external games only (solution/tune.py), never on the provided match.
"""

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover
    def njit(*a, **k):
        def wrap(f):
            return f
        return wrap if not (a and callable(a[0])) else a[0]

H, A, D, X = 0, 1, 2, 3


@njit(cache=True)
def _viterbi(logp, trans):
    n, k = logp.shape
    score = np.empty(k)
    back = np.empty((n, k), dtype=np.int8)
    for s in range(k):
        score[s] = logp[0, s]
        back[0, s] = s
    new = np.empty(k)
    for t in range(1, n):
        for s in range(k):
            best, arg = -1e300, 0
            for r in range(k):
                v = score[r] + trans[r, s]
                if v > best:
                    best, arg = v, r
            new[s] = best + logp[t, s]
            back[t, s] = arg
        for s in range(k):
            score[s] = new[s]
    path = np.empty(n, dtype=np.int8)
    path[n - 1] = np.argmax(score)
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


@njit(cache=True)
def _viterbi_min_turnover(logp, k, switch_cost, to_dead_cost, from_dead_cost):
    """Viterbi with a minimum duration of k frames for possessions won from the opponent.

    The challenge's possession definition (and our external labels, solution/relabel.py) says a
    team that takes the ball from the other must keep control for a few seconds; a restart
    possession has no such constraint. States: D, free H, free A, and a k-step chain for each
    team that a turnover must traverse before the team can lose the ball again.
    """
    n = logp.shape[0]
    NEG = -1e300
    sD, sFH, sFA = logp[0, 2], logp[0, 0], logp[0, 1]
    cH = np.full(k, NEG)
    cA = np.full(k, NEG)
    backD = np.zeros(n, dtype=np.int8)     # 0 D, 1 FH, 2 FA
    backFH = np.zeros(n, dtype=np.int8)    # 1 FH, 3 chain end, 0 D
    backFA = np.zeros(n, dtype=np.int8)    # 2 FA, 3 chain end, 0 D
    for t in range(1, n):
        eH, eA, eD = logp[t, 0], logp[t, 1], logp[t, 2]
        # D
        best, arg = sD, 0
        v = sFH - to_dead_cost
        if v > best:
            best, arg = v, 1
        v = sFA - to_dead_cost
        if v > best:
            best, arg = v, 2
        nD = best + eD
        backD[t] = arg
        # free H
        best, arg = sFH, 1
        if cH[k - 1] > best:
            best, arg = cH[k - 1], 3
        v = sD - from_dead_cost
        if v > best:
            best, arg = v, 0
        nFH = best + eH
        backFH[t] = arg
        # free A
        best, arg = sFA, 2
        if cA[k - 1] > best:
            best, arg = cA[k - 1], 3
        v = sD - from_dead_cost
        if v > best:
            best, arg = v, 0
        nFA = best + eA
        backFA[t] = arg
        # chains (shift by one; entry from the other team's free state)
        for i in range(k - 1, 0, -1):
            cH[i] = cH[i - 1] + eH
            cA[i] = cA[i - 1] + eA
        cH[0] = sFA - switch_cost + eH
        cA[0] = sFH - switch_cost + eA
        sD, sFH, sFA = nD, nFH, nFA
    # final state: (kind, chain index) kind 0 D, 1 FH, 2 FA, 3 chainH, 4 chainA
    kind, idx, best = 0, 0, sD
    if sFH > best:
        kind, idx, best = 1, 0, sFH
    if sFA > best:
        kind, idx, best = 2, 0, sFA
    for i in range(k):
        if cH[i] > best:
            kind, idx, best = 3, i, cH[i]
        if cA[i] > best:
            kind, idx, best = 4, i, cA[i]
    path = np.empty(n, dtype=np.int8)
    for t in range(n - 1, -1, -1):
        if kind == 0:
            path[t] = 2
            if t > 0:
                kind = backD[t]
        elif kind == 1:
            path[t] = 0
            if t > 0:
                b = backFH[t]
                if b == 3:
                    kind, idx = 3, k - 1
                else:
                    kind = b
        elif kind == 2:
            path[t] = 1
            if t > 0:
                b = backFA[t]
                if b == 3:
                    kind, idx = 4, k - 1
                else:
                    kind = b
        elif kind == 3:
            path[t] = 0
            if idx > 0:
                idx -= 1
            else:
                kind = 2
        else:
            path[t] = 1
            if idx > 0:
                idx -= 1
            else:
                kind = 1
    return path


def transition_matrix(switch_cost, to_dead_cost, from_dead_cost):
    t = np.zeros((3, 3))
    t[H, A] = t[A, H] = -switch_cost
    t[H, D] = t[A, D] = -to_dead_cost
    t[D, H] = t[D, A] = -from_dead_cost
    return t


def decode(prob, blocks, n_total, switch_cost=8.0, to_dead_cost=8.0, from_dead_cost=8.0,
           prior=None, prior_power=0.0, eps=1e-4, min_turnover_s=0.0, fps=25, dead_bias=0.0):
    """prob: (n_total, 3) posteriors (rows outside blocks ignored). Returns codes, X outside.

    min_turnover_s > 0 enforces the definitional minimum length of a possession won from the
    opponent. dead_bias adds a constant to the dead-ball log-emission (calibration of the D/alive
    trade-off; tuned on external games like everything else here).
    """
    out = np.full(n_total, X, dtype=np.int8)
    logp = np.log(np.clip(prob, eps, 1.0))
    if prior is not None and prior_power:
        logp = logp - prior_power * np.log(np.asarray(prior))[None, :]
    if dead_bias:
        logp[:, D] += dead_bias
    k = int(round(min_turnover_s * fps))
    trans = transition_matrix(switch_cost, to_dead_cost, from_dead_cost)
    for a, b in blocks:
        seg = np.ascontiguousarray(logp[a:b])
        if k > 0:
            out[a:b] = _viterbi_min_turnover(seg, k, float(switch_cost), float(to_dead_cost),
                                             float(from_dead_cost))
        else:
            out[a:b] = _viterbi(seg, trans)
    return out
