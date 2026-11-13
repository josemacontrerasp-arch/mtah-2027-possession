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

# Exact arithmetic. Probabilities are quantised to 1/QUANT, and log-probabilities and costs to
# 1/SCALE nat, held as integer-valued float64. Every score in the Viterbi recursions below is then
# an exact integer (far below 2**53), so sums carry no rounding error: two paths either tie exactly
# or differ by a whole unit, and ties are broken by the fixed order of the comparisons. The decoded
# path is therefore a pure function of the quantised probabilities -- it does not depend on the
# platform's log implementation, SIMD width or summation order.
QUANT = 10_000
SCALE = 1_000_000
_TABLE_MARGIN = 1e-6     # units; a platform's log would need an error of ~500 ulp to change the table


def _log_table(eps):
    """Integer log-probability (units of 1/SCALE nat) of every quantised probability k/QUANT."""
    if not eps > 0:
        raise ValueError("eps must be positive")
    x = np.log(np.maximum(np.arange(QUANT + 1) / QUANT, eps)) * SCALE
    if not np.abs(x - np.floor(x) - 0.5).min() >= _TABLE_MARGIN:
        raise RuntimeError("log table entry on a rounding boundary; change SCALE")
    return np.rint(x)


def _units(cost):
    """A cost in nats as an exact integer number of 1/SCALE units (float64)."""
    u = float(np.rint(float(cost) * SCALE))
    if not abs(u) < 2.0 ** 40:          # keeps every path score far inside the exact range 2**53
        raise ValueError("cost {} is outside the range the exact decoder supports".format(cost))
    return u


def _scores(prob, eps, prior, prior_power, dead_bias):
    """Integer-valued float64 log-scores (n, 3) from probabilities; NaN rows count as uniform."""
    p = np.nan_to_num(np.asarray(prob, dtype=np.float64), nan=1.0 / 3)
    q = np.rint(np.clip(p, 0.0, 1.0) * QUANT).astype(np.int64)
    logp = _log_table(eps)[q]
    if prior is not None and prior_power:
        logp = logp - np.rint(prior_power * np.log(np.asarray(prior, np.float64)) * SCALE)[None, :]
    if dead_bias:
        logp[:, D] += _units(dead_bias)
    return logp


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


@njit(cache=True)
def _viterbi_struct(logp, k, kx, kd, sw, to_dead_cost, from_dead_cost):
    """Constrained Viterbi, generalising _viterbi_min_turnover.

    k  : a possession won from the opponent lasts >= k frames (turnover chain), k >= 1;
    kx : 0 = such a possession cannot end in a stoppage before it completes its chain; else it may
         end in a stoppage once it has lasted >= kx frames (kx <= k);
    kd : a stoppage entered from play lasts >= kd frames (0 = no constraint).
    sw : per-frame cost of switching team (array); constant sw with kx = kd = 0 gives exactly
         the result of _viterbi_min_turnover.
    State kinds: 0 D free, 1 H free, 2 A free, 3 H chain, 4 A chain, 5 D chain.
    """
    n = logp.shape[0]
    NEG = -1e300
    sDf, sHf, sAf = logp[0, 2], logp[0, 0], logp[0, 1]
    cH = np.full(k, NEG)
    cA = np.full(k, NEG)
    nd = kd if kd > 0 else 1
    cD = np.full(nd, NEG)
    backDf = np.zeros(n, dtype=np.int8)     # 0 Df, 1 entry (kd == 0), 5 D-chain end
    backHf = np.zeros(n, dtype=np.int8)     # 1 Hf, 3 chain end, 0 Df
    backAf = np.zeros(n, dtype=np.int8)     # 2 Af, 3 chain end, 0 Df
    ent_kind = np.zeros(n, dtype=np.int8)   # predecessor of a D entry: 1 Hf, 2 Af, 3 Hc, 4 Ac
    ent_idx = np.zeros(n, dtype=np.int32)
    for t in range(1, n):
        eH, eA, eD = logp[t, 0], logp[t, 1], logp[t, 2]
        # best entry into a stoppage from play, scored at t-1
        best, kind, idx = sHf, 1, 0
        if sAf > best:
            best, kind, idx = sAf, 2, 0
        if kx > 0:
            for i in range(kx - 1, k):
                if cH[i] > best:
                    best, kind, idx = cH[i], 3, i
                if cA[i] > best:
                    best, kind, idx = cA[i], 4, i
        ent = best - to_dead_cost
        ent_kind[t] = kind
        ent_idx[t] = idx
        # stoppage states
        if kd > 0:
            best, arg = sDf, 0
            if cD[kd - 1] > best:
                best, arg = cD[kd - 1], 5
            nDf = best + eD
            backDf[t] = arg
            for i in range(kd - 1, 0, -1):
                cD[i] = cD[i - 1] + eD
            cD[0] = ent + eD
        else:
            best, arg = sDf, 0
            if ent > best:
                best, arg = ent, 1
            nDf = best + eD
            backDf[t] = arg
        # free possession states
        best, arg = sHf, 1
        if cH[k - 1] > best:
            best, arg = cH[k - 1], 3
        v = sDf - from_dead_cost
        if v > best:
            best, arg = v, 0
        nHf = best + eH
        backHf[t] = arg
        best, arg = sAf, 2
        if cA[k - 1] > best:
            best, arg = cA[k - 1], 3
        v = sDf - from_dead_cost
        if v > best:
            best, arg = v, 0
        nAf = best + eA
        backAf[t] = arg
        # turnover chains
        for i in range(k - 1, 0, -1):
            cH[i] = cH[i - 1] + eH
            cA[i] = cA[i - 1] + eA
        cH[0] = sAf - sw[t] + eH
        cA[0] = sHf - sw[t] + eA
        sDf, sHf, sAf = nDf, nHf, nAf
    kind, idx, best = 0, 0, sDf
    if sHf > best:
        kind, idx, best = 1, 0, sHf
    if sAf > best:
        kind, idx, best = 2, 0, sAf
    for i in range(k):
        if cH[i] > best:
            kind, idx, best = 3, i, cH[i]
        if cA[i] > best:
            kind, idx, best = 4, i, cA[i]
    if kd > 0:
        for i in range(kd):
            if cD[i] > best:
                kind, idx, best = 5, i, cD[i]
    path = np.empty(n, dtype=np.int8)
    for t in range(n - 1, -1, -1):
        if kind == 0 or kind == 5:
            path[t] = 2
        elif kind == 1 or kind == 3:
            path[t] = 0
        else:
            path[t] = 1
        if t == 0:
            break
        if kind == 0:
            b = backDf[t]
            if b == 0:
                kind = 0
            elif b == 5:
                kind, idx = 5, kd - 1
            else:
                kind, idx = ent_kind[t], ent_idx[t]
        elif kind == 5:
            if idx > 0:
                idx -= 1
            else:
                kind, idx = ent_kind[t], ent_idx[t]
        elif kind == 1:
            b = backHf[t]
            if b == 3:
                kind, idx = 3, k - 1
            else:
                kind = b
        elif kind == 2:
            b = backAf[t]
            if b == 3:
                kind, idx = 4, k - 1
            else:
                kind = b
        elif kind == 3:
            if idx > 0:
                idx -= 1
            else:
                kind = 2
        else:
            if idx > 0:
                idx -= 1
            else:
                kind = 1
    return path


def posterior_jump(prob, half_window=10):
    """|change| of the home-vs-away log-odds across +/- half_window frames (0.4 s at 25 fps):
    large where the ensemble's belief about who has the ball flips -- a likely turnover."""
    ph = prob[:, 0] / np.clip(prob[:, 0] + prob[:, 1], 1e-6, None)
    q = np.log(np.clip(ph, 1e-4, 1 - 1e-4) / np.clip(1 - ph, 1e-4, 1 - 1e-4))
    qp = np.concatenate([np.full(half_window, q[0]), q, np.full(half_window, q[-1])])
    return np.abs(qp[2 * half_window:] - qp[:-2 * half_window])


def transition_matrix(switch_cost, to_dead_cost, from_dead_cost):
    t = np.zeros((3, 3))
    t[H, A] = t[A, H] = -switch_cost
    t[H, D] = t[A, D] = -to_dead_cost
    t[D, H] = t[D, A] = -from_dead_cost
    return t


def decode(prob, blocks, n_total, switch_cost=8.0, to_dead_cost=8.0, from_dead_cost=8.0,
           prior=None, prior_power=0.0, eps=1e-4, min_turnover_s=0.0, fps=25, dead_bias=0.0,
           turnover_exit_s=0.0, min_dead_s=0.0, boundary_beta=0.0, boundary_cap=4.0):
    """prob: (n_total, 3) posteriors (rows outside blocks ignored). Returns codes, X outside.

    min_turnover_s > 0 enforces the definitional minimum length of a possession won from the
    opponent. turnover_exit_s > 0 lets such a possession end in a stoppage once it has lasted that
    long. min_dead_s > 0 gives stoppages entered from play a minimum length. dead_bias adds a
    constant to the dead-ball log-emission. All tuned on external games only.

    Probabilities are quantised to 4 decimals and all scores are exact integers (see QUANT / SCALE
    at the top of this module), so the result is identical on every platform. (`prior` and
    `boundary_beta`, which the pipeline does not use, go through the platform's log before they
    are rounded and are outside that guarantee.)
    """
    kx = int(round(turnover_exit_s * fps))
    kd = int(round(min_dead_s * fps))
    if kx > 0 or kd > 0 or boundary_beta > 0:
        k = max(1, int(round(min_turnover_s * fps)))
        kx = min(kx, k)
        out = np.full(n_total, X, dtype=np.int8)
        logp = _scores(prob, eps, prior, prior_power, dead_bias)
        for a, b in blocks:
            if b <= a:
                continue
            sw = np.full(b - a, float(switch_cost))
            if boundary_beta > 0:
                z = posterior_jump(prob[a:b])
                z = np.clip(z - np.median(z), 0.0, boundary_cap)
                sw = np.maximum(0.5, sw - boundary_beta * z)
            out[a:b] = _viterbi_struct(np.ascontiguousarray(logp[a:b]), k, kx, kd,
                                       np.rint(sw * SCALE), _units(to_dead_cost),
                                       _units(from_dead_cost))
        return out
    out = np.full(n_total, X, dtype=np.int8)
    logp = _scores(prob, eps, prior, prior_power, dead_bias)
    k = int(round(min_turnover_s * fps))
    trans = np.rint(transition_matrix(switch_cost, to_dead_cost, from_dead_cost) * SCALE)
    for a, b in blocks:
        if b <= a:
            continue
        seg = np.ascontiguousarray(logp[a:b])
        if k > 0:
            out[a:b] = _viterbi_min_turnover(seg, k, _units(switch_cost), _units(to_dead_cost),
                                             _units(from_dead_cost))
        else:
            out[a:b] = _viterbi(seg, trans)
    return out
