"""Optional fusion of match-specific automatic signals that no external game has.

Two signals exist only for the provided match, so the frame models (trained on external games)
cannot learn them:

* broadcast cut-away: the aligned broadcast is not showing the live wide shot of this instant
  (replay, close-up, crowd) -- TV directors cut away mostly while play is stopped;
* Smart Tagging team hints (solution/tagging.py).

Each becomes a per-frame log-likelihood term for the HMM. Its strength is estimated
SELF-SUPERVISED, never from labels: the likelihood ratio of the signal between frames where the
models are confidently in one state vs the other. Evidence is strongly autocorrelated, so the
log-ratio is spread over TAU frames (one independent observation per TAU frames).

Only POSITIVE evidence is used for the broadcast: a cut-away argues for a stoppage, but a live
shot does not argue for play -- short stoppages (throw-ins) are rarely cut away from, and the
confident frames the ratio is measured on are the long stoppages.
"""

import numpy as np

from solution.match import H, A, D

TAU = 125   # frames: one independent observation per 5 s


def _lr(signal, pos_mask, neg_mask, floor=1e-3):
    p = float(np.mean(signal[pos_mask])) if pos_mask.any() else np.nan
    q = float(np.mean(signal[neg_mask])) if neg_mask.any() else np.nan
    return max(p, floor) / max(q, floor), p, q


def broadcast_dead_evidence(prob, live, in_span, conf=0.9, tau=TAU):
    pd_ = prob[:, D]
    ok = in_span & np.isfinite(pd_)
    cut = (ok & ~live).astype(np.float64)
    dead_conf = ok & (pd_ >= conf)
    alive_conf = ok & (pd_ <= 1 - conf)
    lr, p, q = _lr(cut, dead_conf, alive_conf)
    adj = np.zeros((len(pd_), 3))
    if lr > 1:
        adj[cut > 0, D] = np.log(lr) / tau
    return adj, dict(lr_cut_away=lr, p_cut_given_dead=p, p_cut_given_alive=q,
                     n_dead_conf=int(dead_conf.sum()), n_alive_conf=int(alive_conf.sum()))


def tag_team_evidence(prob, ev, families=("in", "out"), conf=0.9, tau=TAU):
    alive = prob[:, H] + prob[:, A]
    p_home = prob[:, H] / np.clip(alive, 1e-6, None)
    sure = alive >= 0.8
    home_conf = sure & (p_home >= conf)
    away_conf = sure & (p_home <= 1 - conf)
    adj = np.zeros((len(alive), 3))
    stats = {}
    for fam in families:
        e = ev[fam]
        lr_h, _, _ = _lr((e > 0).astype(np.float64), home_conf, away_conf)
        lr_a, _, _ = _lr((e < 0).astype(np.float64), away_conf, home_conf)
        stats[fam] = dict(lr_home=lr_h, lr_away=lr_a)
        if lr_h > 1:
            adj[e > 0, H] += np.log(lr_h) / tau
        if lr_a > 1:
            adj[e < 0, A] += np.log(lr_a) / tau
    return adj, stats


def apply_adjustment(prob, adj):
    """Posteriors x likelihood adjustments, renormalised per frame."""
    logp = np.log(np.clip(prob, 1e-6, 1.0)) + adj
    logp -= logp.max(axis=1, keepdims=True)
    p = np.exp(logp)
    return p / p.sum(axis=1, keepdims=True)
