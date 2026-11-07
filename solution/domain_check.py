"""Unsupervised domain-gap diagnostic: can a classifier tell degraded external frames from
provided-match frames? Features that make this easy will not transfer, whatever their CV score.

No labels of the provided match are involved. Prints AUCs, feature names and quantiles only.

    python -m solution.domain_check
"""

import numpy as np
import pandas as pd

from solution.atd import load_atd
from solution.degrade import degrade
from solution.features import compute_features
from solution.train import EXCLUDE, EVAL_SEED, load_ext
from solution.external_dfl import MATCH_IDS


def _auc(x, y):
    """Rank AUC of a single feature for separating y=1 from y=0 (NaN-aware, symmetric)."""
    ok = np.isfinite(x)
    if ok.sum() < 100 or len(np.unique(y[ok])) < 2:
        return np.nan
    r = pd.Series(x[ok]).rank().values
    yy = y[ok]
    n1, n0 = yy.sum(), (1 - yy).sum()
    auc = (r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return max(auc, 1 - auc)


def main(n_per_domain=120000, seed=0):
    import lightgbm as lgb
    rng = np.random.default_rng(seed)
    ext = []
    for mid in MATCH_IDS[:4]:
        F = compute_features(degrade(load_ext(mid), EVAL_SEED))
        ext.append(F)
    E = pd.concat(ext)
    T = compute_features(load_atd("tactical"))
    cols = [c for c in T.columns if c not in EXCLUDE]
    E = E.iloc[rng.choice(len(E), min(n_per_domain, len(E)), replace=False)][cols]
    T = T.iloc[rng.choice(len(T), min(n_per_domain, len(T)), replace=False)][cols]
    X = pd.concat([E, T], ignore_index=True)
    y = np.concatenate([np.zeros(len(E)), np.ones(len(T))])

    uni = {c: _auc(X[c].values.astype(np.float64), y) for c in cols}
    nan_gap = {c: abs(E[c].isna().mean() - T[c].isna().mean()) for c in cols}

    idx = rng.permutation(len(X))
    half = len(idx) // 2
    params = dict(objective="binary", learning_rate=0.1, num_leaves=31, verbose=-1,
                  num_threads=16, seed=1)
    tr, te = idx[:half], idx[half:]
    model = lgb.train(params, lgb.Dataset(X.iloc[tr], label=y[tr]), num_boost_round=150)
    p = model.predict(X.iloc[te])
    print("domain classifier AUC (0.5 = indistinguishable): {:.3f}".format(_auc(p, y[te])))
    imp = pd.Series(model.feature_importance("gain"), index=cols)
    imp = (imp / imp.sum()).sort_values(ascending=False)
    print("\nfeatures the domain classifier relies on (gain share), with univariate AUC and "
          "missing-rate gap:")
    for c in imp.index[:20]:
        print("  {:<24} gain {:.3f}   uniAUC {:.3f}   |nan gap| {:.3f}".format(
            c, imp[c], uni[c], nan_gap[c]))
    worst = sorted(uni.items(), key=lambda kv: -np.nan_to_num(kv[1]))[:12]
    print("\nmost separable single features (univariate AUC), with p10/p50/p90 ext vs provided:")
    for c, a in worst:
        qe = np.nanpercentile(E[c].values.astype(float), [10, 50, 90]).round(2).tolist()
        qt = np.nanpercentile(T[c].values.astype(float), [10, 50, 90]).round(2).tolist()
        print("  {:<24} AUC {:.3f}   ext {}   provided {}".format(c, a, qe, qt))


if __name__ == "__main__":
    main()
