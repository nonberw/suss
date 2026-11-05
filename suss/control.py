"""Experiment C1: in-context adaptive regulation of linear plants with unknown (A, B)."""
from __future__ import annotations

import torch

from .linear import spectral_radius_2x2

N, M = 2, 1
SIGMA = 0.1
H = 48
R_U = 0.1
U_MAX = 5.0
A0 = 0.9
X_CLIP = 1e3


def sample_plants(n, gen, device="cpu"):
    out = []
    got = 0
    while got < n:
        A = A0 * torch.eye(2, device=device) + 0.25 * torch.randn(2 * n, 2, 2, generator=gen, device=device)
        A = A[spectral_radius_2x2(A) < 1.1]
        out.append(A)
        got += A.shape[0]
    A = torch.cat(out)[:n]
    r = 0.3 + 0.7 * torch.rand(n, generator=gen, device=device)
    phi = 2 * torch.pi * torch.rand(n, generator=gen, device=device)
    B = torch.stack([r * torch.cos(phi), r * torch.sin(phi)], -1).unsqueeze(-1)  # (n,2,1)
    return A, B


def sample_noise(n, gen, device="cpu"):
    x0 = torch.randn(n, N, generator=gen, device=device)
    w = SIGMA * torch.randn(n, H, N, generator=gen, device=device)
    return x0, w


def stage_cost(x, u):
    return (x ** 2).sum(-1) + R_U * (u ** 2).sum(-1)


def lqr_gains(A, B, horizon, Q=1.0, R=R_U, Qf=1.0):
    """Finite-horizon LQR, batched. Returns K (n, horizon, 1, 2) with u_t = -K_t x_t."""
    n = A.shape[0]
    I = torch.eye(N, device=A.device).expand(n, N, N)
    P = Qf * I
    Ks = []
    Bt = B.transpose(-1, -2)
    for _ in range(horizon):
        S = R + Bt @ P @ B  # (n,1,1)
        K = torch.linalg.solve(S, Bt @ P @ A)
        P = Q * I + A.transpose(-1, -2) @ P @ (A - B @ K)
        P = 0.5 * (P + P.transpose(-1, -2))
        Ks.append(K)
    return torch.stack(Ks[::-1], 1)


def simulate(A, B, x0, w, policy, tbptt=0):
    """policy(t, xs, us) -> u_t (n,1), where xs: list of past states incl. x_t, us: list of past actions.
    Returns J (n,), X (n,H+1,2), U (n,H,1)."""
    x = x0
    xs, us = [x], []
    J = torch.zeros(x0.shape[0], device=x0.device)
    for t in range(H):
        u = policy(t, xs, us).clamp(-U_MAX, U_MAX)
        J = J + stage_cost(x, u)
        x = (torch.einsum("bij,bj->bi", A, x) + torch.einsum("bij,bj->bi", B, u) + w[:, t]).clamp(-X_CLIP, X_CLIP)
        if tbptt and (t + 1) % tbptt == 0:
            x = x.detach()  # truncated BPTT through the plant (the policy still sees the full history)
        xs.append(x)
        us.append(u)
    J = J + (x ** 2).sum(-1)
    return J, torch.stack(xs, 1), torch.stack(us, 1)


def oracle_policy(A, B):
    K = lqr_gains(A, B, H)
    return lambda t, xs, us: -(K[:, t] @ xs[-1].unsqueeze(-1)).squeeze(-1)


def fixed_policy(Kc):
    return lambda t, xs, us: -(xs[-1] @ Kc.T)


def ridge_estimate(xs, us, lam):
    """Ridge estimate of theta=[A B] toward the prior mean [A0 I, 0] from pairs s<t."""
    X = torch.stack(xs[:-1], 1)  # (n,t,2)
    U = torch.stack(us, 1)  # (n,t,1)
    Xn = torch.stack(xs[1:], 1)
    Z = torch.cat([X, U], -1)
    n = Z.shape[0]
    th0 = torch.zeros(n, N, N + M, device=Z.device)
    th0[:, 0, 0] = A0; th0[:, 1, 1] = A0
    S = Z.transpose(1, 2) @ Z + lam * torch.eye(N + M, device=Z.device)
    C = Z.transpose(1, 2) @ Xn + lam * th0.transpose(1, 2)
    return torch.linalg.solve(S, C).transpose(1, 2)  # (n,2,3)


def ce_policy(lam, s0=0.0, gen=None, explore_steps=0, s_e=0.0):
    """Certainty equivalence with optional decaying dither s0/sqrt(t+1), or explore-then-commit."""
    def pol(t, xs, us):
        n = xs[-1].shape[0]
        dev = xs[-1].device
        if t < explore_steps:
            return s_e * torch.randn(n, 1, generator=gen, device=dev)
        if t == 0:
            Ah = A0 * torch.eye(2, device=dev).expand(n, 2, 2)
            Bh = torch.zeros(n, 2, 1, device=dev)
        else:
            th = ridge_estimate(xs, us, lam)
            Ah, Bh = th[:, :, :N], th[:, :, N:]
        K = lqr_gains(Ah, Bh, H - t)[:, 0]
        u = -(K @ xs[-1].unsqueeze(-1)).squeeze(-1)
        if s0 > 0:
            u = u + s0 / (t + 1) ** 0.5 * torch.randn(n, 1, generator=gen, device=dev)
        return u
    return pol


def ts_policy(lam, gen, inflate=1.0):
    """Thompson sampling: sample (A,B) from the Gaussian ridge posterior each step, act with its LQR gain.
    A classical Bayesian dual-control heuristic (exploration comes from posterior sampling)."""
    def pol(t, xs, us):
        n = xs[-1].shape[0]
        dev = xs[-1].device
        th0 = torch.zeros(n, N, N + M, device=dev)
        th0[:, 0, 0] = A0; th0[:, 1, 1] = A0
        if t == 0:
            mean = th0
            cov = torch.eye(N + M, device=dev).expand(n, -1, -1) * SIGMA ** 2 / lam
        else:
            X = torch.stack(xs[:-1], 1); U = torch.stack(us, 1); Xn = torch.stack(xs[1:], 1)
            Z = torch.cat([X, U], -1)
            S = Z.transpose(1, 2) @ Z + lam * torch.eye(N + M, device=dev)
            mean = torch.linalg.solve(S, Z.transpose(1, 2) @ Xn + lam * th0.transpose(1, 2)).transpose(1, 2)
            cov = SIGMA ** 2 * torch.linalg.inv(S)
        Lc = torch.linalg.cholesky(0.5 * (cov + cov.transpose(1, 2))) * inflate
        eps = torch.randn(n, N, N + M, generator=gen, device=dev)
        th = mean + torch.einsum("bij,bkj->bki", Lc, eps)
        K = lqr_gains(th[:, :, :N], th[:, :, N:], H - t)[:, 0]
        return -(K @ xs[-1].unsqueeze(-1)).squeeze(-1)
    return pol


def squash(x):
    """Soft input compression for the controller's tokens (states can reach 1e3 on unstable plants)."""
    return torch.sign(x) * torch.log1p(x.abs())


def tf_policy(model, squash_in=False):
    """Transformer controller: token t = (x_t, u_{t-1}), output u_t = U_MAX * tanh(head)."""
    cache = model.new_cache()

    def pol(t, xs, us):
        n = xs[-1].shape[0]
        uprev = us[-1] if us else torch.zeros(n, 1, device=xs[-1].device)
        tok = torch.cat([xs[-1], uprev], -1)
        if squash_in:
            tok = squash(tok)
        out = model.step(tok, t, cache)
        return U_MAX * torch.tanh(out)
    return pol


def tokens_from_traj(X, U, squash_in=False):
    """Tokens (x_t, u_{t-1}) for t=0..H-1 from a finished trajectory."""
    n = X.shape[0]
    uprev = torch.cat([torch.zeros(n, 1, 1, device=X.device), U[:, :-1]], 1)
    tok = torch.cat([X[:, :H], uprev], -1)
    return squash(tok) if squash_in else tok


def info_gain(X, U, k=8, prior_prec=1.0):
    """log det(Lambda0 + sigma^-2 S_k), S_k = sum_{s<k} z_s z_s^T, z=(x_s,u_s)."""
    Z = torch.cat([X[:, :k], U[:, :k]], -1)
    S = Z.transpose(1, 2) @ Z
    return torch.logdet(prior_prec * torch.eye(N + M, device=X.device) + S / SIGMA ** 2)
