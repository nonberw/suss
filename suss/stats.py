"""Hierarchical bootstrap and Holm-Bonferroni (pre-registered analysis)."""
from __future__ import annotations

import numpy as np


def hier_boot(per_seed, stat=np.mean, n_boot=10_000, rng=0):
    """per_seed: list of 1-D arrays (one per training seed) of per-episode values.
    Resample seeds with replacement, then episodes within each seed; return the bootstrap
    distribution of stat(pooled values) and the point estimate."""
    rng = np.random.default_rng(rng)
    per_seed = [np.asarray(x, dtype=float) for x in per_seed]
    point = stat(np.concatenate(per_seed))
    S = len(per_seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        seeds = rng.integers(0, S, S)
        vals = [per_seed[s][rng.integers(0, len(per_seed[s]), len(per_seed[s]))] for s in seeds]
        out[b] = stat(np.concatenate(vals))
    return point, out


def hier_boot_ratio(num, den, n_boot=10_000, rng=0):
    """Ratio of means num/den with paired episodes (num[s][i] and den[s][i] from the same episode)."""
    rng = np.random.default_rng(rng)
    num = [np.asarray(x, float) for x in num]
    den = [np.asarray(x, float) for x in den]
    point = np.concatenate(num).mean() / np.concatenate(den).mean()
    S = len(num)
    out = np.empty(n_boot)
    for b in range(n_boot):
        seeds = rng.integers(0, S, S)
        nn_, dd = [], []
        for s in seeds:
            idx = rng.integers(0, len(num[s]), len(num[s]))
            nn_.append(num[s][idx])
            dd.append(den[s][idx])
        out[b] = np.concatenate(nn_).mean() / np.concatenate(dd).mean()
    return point, out


def ci(dist, level=0.95):
    a = (1 - level) / 2
    return float(np.quantile(dist, a)), float(np.quantile(dist, 1 - a))


def p_greater(dist, thr=0.0):
    """One-sided bootstrap p-value for H: stat > thr."""
    return (1 + np.sum(dist <= thr)) / (1 + len(dist))


def p_less(dist, thr=0.0):
    return (1 + np.sum(dist >= thr)) / (1 + len(dist))


def holm(pvals: dict, alpha=0.05):
    """Return {name: (p, p_adj, reject)} using Holm-Bonferroni."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out = {}
    running = 0.0
    stop = False
    for i, (k, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)
        rej = (not stop) and running <= alpha
        if not rej:
            stop = True
        out[k] = (p, running, rej)
    return out
