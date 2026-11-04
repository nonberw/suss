"""Linear-Gaussian systems with unknown parameters and their exact in-context estimators.

x_{t+1} = A x_t + B u_t + w_t,  z_t = (x_t, u_t) in R^p,  theta = [A B] in R^{n x p}.
Each row of theta has prior N(0, diag(prior_sd^2)); the joint prior is truncated to rho(A) < rho_max.
"""
from __future__ import annotations

import torch

N, M = 2, 1
P = N + M
SIGMA = 0.1
SD_A, SD_B = 0.4, 0.5
RHO_MAX = 0.95


def prior_sd(device="cpu"):
    return torch.tensor([SD_A] * N + [SD_B] * M, device=device)


def spectral_radius_2x2(A):
    """A: (..., 2, 2). Exact spectral radius."""
    tr = A[..., 0, 0] + A[..., 1, 1]
    det = A[..., 0, 0] * A[..., 1, 1] - A[..., 0, 1] * A[..., 1, 0]
    disc = tr * tr - 4 * det
    sq = torch.sqrt(disc.clamp(min=0))
    real_r = torch.maximum((tr + sq).abs(), (tr - sq).abs()) / 2
    cplx_r = torch.sqrt(det.clamp(min=0))
    return torch.where(disc >= 0, real_r, cplx_r)


def sample_theta(n, gen, device="cpu", rho_max=RHO_MAX):
    out = []
    got = 0
    sd = prior_sd(device)
    while got < n:
        th = torch.randn(2 * n, N, P, generator=gen, device=device) * sd
        ok = spectral_radius_2x2(th[:, :, :N]) < rho_max
        th = th[ok]
        out.append(th)
        got += th.shape[0]
    return torch.cat(out)[:n]


def rollout(theta, T, gen, device="cpu", u=None):
    """Open-loop rollout with u ~ N(0,1). Returns tokens (n,T,P) and next states (n,T,N)."""
    n = theta.shape[0]
    x = torch.randn(n, N, generator=gen, device=device)
    if u is None:
        u = torch.randn(n, T, M, generator=gen, device=device)
    w = torch.randn(n, T, N, generator=gen, device=device) * SIGMA
    xs, nxt = [], []
    for t in range(T):
        z = torch.cat([x, u[:, t]], -1)
        xn = torch.einsum("bij,bj->bi", theta, z) + w[:, t]
        xs.append(x)
        nxt.append(xn)
        x = xn
    X = torch.stack(xs, 1)
    tokens = torch.cat([X, u], -1)
    return tokens, torch.stack(nxt, 1)


def make_batch(n, T, gen, device="cpu"):
    theta = sample_theta(n, gen, device)
    tokens, nxt = rollout(theta, T, gen, device)
    return theta, tokens, nxt


def suff_stats(tokens, nxt):
    """Statistics of the context available when predicting at position t (pairs s=0..t-1).
    S[:,t] = sum_{s<t} z_s z_s^T (p x p),  C[:,t] = sum_{s<t} z_s x_{s+1}^T (p x n)."""
    zz = tokens.unsqueeze(-1) * tokens.unsqueeze(-2)
    zx = tokens.unsqueeze(-1) * nxt.unsqueeze(-2)
    S = torch.cumsum(zz, 1) - zz
    C = torch.cumsum(zx, 1) - zx
    return S, C


def ridge_theta(S, C, sigma=SIGMA):
    """Posterior mean under the untruncated Gaussian prior. Returns theta (..., n, p)."""
    lam = (sigma / prior_sd(S.device)) ** 2
    Pm = S + torch.diag_embed(lam.expand(S.shape[:-1]))
    return torch.linalg.solve(Pm, C).transpose(-1, -2)


def ols_theta(S, C):
    return (torch.linalg.pinv(S) @ C).transpose(-1, -2)


def gd1_theta(S, C, eta):
    t = torch.arange(S.shape[1], device=S.device).clamp(min=1).view(1, -1, 1, 1)
    return eta * C.transpose(-1, -2) / t


def predict(theta_hat, tokens):
    return torch.einsum("btij,btj->bti", theta_hat, tokens)


def bayes_theta(S, C, gen, K=2048, sigma=SIGMA, rho_max=RHO_MAX, chunk=16):
    """Posterior mean of theta under the truncated prior by Monte Carlo.
    S: (b,T,p,p), C: (b,T,p,n). Returns theta (b,T,n,p) and acceptance rate (b,T)."""
    b, T = S.shape[:2]
    lam = 1.0 / prior_sd(S.device) ** 2  # prior precision
    out = torch.empty(b, T, N, P, device=S.device)
    acc = torch.empty(b, T, device=S.device)
    for i in range(0, b, chunk):
        Si, Ci = S[i:i + chunk], C[i:i + chunk]
        prec = torch.diag_embed(lam.expand(Si.shape[:-1])) + Si / sigma ** 2
        cov = torch.linalg.inv(prec)
        cov = 0.5 * (cov + cov.transpose(-1, -2))
        mean = (cov @ (Ci / sigma ** 2)).transpose(-1, -2)  # (c,T,n,p)
        L = torch.linalg.cholesky(cov)  # (c,T,p,p)
        eps = torch.randn(*mean.shape[:2], K, N, P, generator=gen, device=S.device)
        th = mean.unsqueeze(2) + torch.einsum("ctij,ctknj->ctkni", L, eps)
        ok = (spectral_radius_2x2(th[..., :N]) < rho_max).float()  # (c,T,K)
        cnt = ok.sum(2)
        m = (th * ok[..., None, None]).sum(2) / cnt.clamp(min=1)[..., None, None]
        m = torch.where((cnt > 0)[..., None, None], m, mean)
        out[i:i + chunk] = m
        acc[i:i + chunk] = cnt / K
    return out, acc


def posterior_logtrace(S, sigma=SIGMA):
    lam = 1.0 / prior_sd(S.device) ** 2
    prec = torch.diag_embed(lam.expand(S.shape[:-1])) + S / sigma ** 2
    return torch.log(torch.linalg.inv(prec).diagonal(dim1=-2, dim2=-1).sum(-1))


def ridge_excess_theory(S, tokens, sigma=SIGMA):
    """Exact Bayes excess risk under the untruncated Gaussian prior, conditional on the data:
    n * sigma^2 * z_t^T (sigma^2 Lambda0 + S_t)^{-1} z_t  (Theorem 1)."""
    lam = (sigma / prior_sd(S.device)) ** 2
    Pm = S + torch.diag_embed(lam.expand(S.shape[:-1]))
    sol = torch.linalg.solve(Pm, tokens.unsqueeze(-1)).squeeze(-1)
    return N * sigma ** 2 * (sol * tokens).sum(-1)
