"""Ridge probes with K-fold CV, scored by held-out R^2 with per-episode sums for bootstrapping."""
from __future__ import annotations

import numpy as np
import torch

ALPHAS = [1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2]


def _fit(X, Y, alpha):
    # X: (n,d) standardized, Y centered; closed-form ridge (sklearn convention)
    d = X.shape[1]
    A = X.T @ X + alpha * torch.eye(d, device=X.device, dtype=X.dtype)
    return torch.linalg.solve(A, X.T @ Y)


class RidgeProbe:
    def __init__(self, alphas=ALPHAS, folds=5, seed=0):
        self.alphas, self.folds, self.seed = alphas, folds, seed

    def fit(self, X, Y, groups):
        """X: (n,d) torch, Y: (n,k) torch, groups: (n,) episode ids for grouped CV."""
        X = X.double()
        Y = Y.double()
        self.mx, self.sx = X.mean(0), X.std(0) + 1e-6
        self.my = Y.mean(0)
        Xs = (X - self.mx) / self.sx
        Yc = Y - self.my
        ug = np.unique(groups)
        rng = np.random.default_rng(self.seed)
        rng.shuffle(ug)
        fold_of = {g: i % self.folds for i, g in enumerate(ug)}
        f = torch.tensor([fold_of[g] for g in groups], device=X.device)
        best, best_err = None, np.inf
        for a in self.alphas:
            err = 0.0
            for k in range(self.folds):
                tr, te = f != k, f == k
                W = _fit(Xs[tr], Yc[tr], a)
                err += float(((Xs[te] @ W - Yc[te]) ** 2).sum())
            if err < best_err:
                best, best_err = a, err
        self.alpha = best
        self.W = _fit(Xs, Yc, best)
        return self

    def predict(self, X):
        return ((X.double() - self.mx) / self.sx) @ self.W + self.my

    def direction_matrix(self):
        """Linear map residual -> readout: y = (h - mx)/sx @ W + my  =>  y = h @ (W / sx[:,None]) + c."""
        return (self.W / self.sx[:, None]).float()  # (d, k)


def r2_parts(Yhat, Y, groups):
    """Per-episode sufficient pieces for bootstrapping R^2 per target: SSE, sum y, sum y^2, count."""
    Yhat, Y = Yhat.double().cpu().numpy(), Y.double().cpu().numpy()
    ug, inv = np.unique(groups, return_inverse=True)
    k = Y.shape[1]
    sse = np.zeros((len(ug), k)); sy = np.zeros((len(ug), k)); syy = np.zeros((len(ug), k)); cnt = np.zeros(len(ug))
    np.add.at(sse, inv, (Yhat - Y) ** 2)
    np.add.at(sy, inv, Y)
    np.add.at(syy, inv, Y ** 2)
    np.add.at(cnt, inv, 1)
    return dict(sse=sse, sy=sy, syy=syy, cnt=cnt)


def r2_from_parts(parts, idx=None):
    sse, sy, syy, cnt = parts["sse"], parts["sy"], parts["syy"], parts["cnt"]
    if idx is not None:
        sse, sy, syy, cnt = sse[idx], sy[idx], syy[idx], cnt[idx]
    n = cnt.sum()
    sst = syy.sum(0) - sy.sum(0) ** 2 / n
    return float(np.mean(1 - sse.sum(0) / sst))


def boot_r2(parts_per_seed, n_boot=10_000, rng=0):
    """Hierarchical bootstrap of mean-over-targets R^2 (seeds then episodes)."""
    rng = np.random.default_rng(rng)
    S = len(parts_per_seed)
    point = np.mean([r2_from_parts(p) for p in parts_per_seed])
    out = np.empty(n_boot)
    for b in range(n_boot):
        vals = []
        for s in rng.integers(0, S, S):
            p = parts_per_seed[s]
            idx = rng.integers(0, len(p["cnt"]), len(p["cnt"]))
            vals.append(r2_from_parts(p, idx))
        out[b] = np.mean(vals)
    return point, out
