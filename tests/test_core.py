"""Tests for the core pieces: estimators, theory formulas, model hooks, control, MuJoCo systems, stats."""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from suss import control as C
from suss import linear as L
from suss import mj
from suss.model import GPT, GPTConfig
from suss.probes import RidgeProbe, r2_from_parts, r2_parts
from suss.stats import hier_boot, holm


def g(seed=0):
    return torch.Generator().manual_seed(seed)


def test_spectral_radius_matches_numpy():
    A = torch.randn(500, 2, 2, generator=g())
    ref = np.abs(np.linalg.eigvals(A.numpy())).max(-1)
    assert np.allclose(L.spectral_radius_2x2(A).numpy(), ref, atol=1e-5)


def test_prior_truncation():
    th = L.sample_theta(2000, g())
    assert (L.spectral_radius_2x2(th[:, :, :2]) < L.RHO_MAX).all()


def test_suff_stats_exclude_current_pair():
    th, tok, nxt = L.make_batch(3, 10, g())
    S, C_ = L.suff_stats(tok, nxt)
    t = 6
    Sref = sum(torch.outer(tok[0, s], tok[0, s]) for s in range(t))
    Cref = sum(torch.outer(tok[0, s], nxt[0, s]) for s in range(t))
    assert torch.allclose(S[0, t], Sref, atol=1e-5) and torch.allclose(C_[0, t], Cref, atol=1e-5)
    assert torch.all(S[:, 0] == 0)


def test_ridge_is_gaussian_posterior_mean():
    th, tok, nxt = L.make_batch(2, 20, g(1))
    S, C_ = L.suff_stats(tok, nxt)
    t = 15
    Z, Y = tok[0, :t].double(), nxt[0, :t].double()
    prec = torch.diag(1 / L.prior_sd().double() ** 2) + Z.T @ Z / L.SIGMA ** 2
    mean = torch.linalg.solve(prec, Z.T @ Y / L.SIGMA ** 2).T
    assert torch.allclose(L.ridge_theta(S, C_)[0, t].double(), mean, atol=1e-4)


def test_ridge_converges_to_truth():
    th, tok, nxt = L.make_batch(64, 400, g(2))
    S, C_ = L.suff_stats(tok, nxt)
    err = (L.ridge_theta(S, C_)[:, -1] - th).abs().mean()
    assert err < 0.02


def test_bayes_mc_equals_ridge_without_truncation():
    th, tok, nxt = L.make_batch(4, 12, g(3))
    S, C_ = L.suff_stats(tok, nxt)
    tb, acc = L.bayes_theta(S, C_, g(4), K=20000, rho_max=1e9)
    assert torch.allclose(tb, L.ridge_theta(S, C_), atol=0.03)
    assert torch.all(acc == 1)


def test_bayes_risk_formula_monte_carlo():
    """Theorem 1: E[(theta z - theta_hat z)^2 | D] = n sigma^2 z^T (sigma^2 Lambda0 + S)^{-1} z
    under the untruncated prior; check by sampling theta from the prior without truncation."""
    gen = g(5)
    n, T = 20000, 6
    th = torch.randn(n, L.N, L.P, generator=gen) * L.prior_sd()
    th[:, :, :2] *= 0.5  # keep rollouts tame; prior used by the estimator is what matters below
    sd = L.prior_sd().clone(); sd[:2] *= 0.5
    tok, nxt = L.rollout(th, T, gen)
    S, C_ = L.suff_stats(tok, nxt)
    lam = (L.SIGMA / sd) ** 2
    Pm = S + torch.diag_embed(lam.expand(S.shape[:-1]))
    thh = torch.linalg.solve(Pm, C_).transpose(-1, -2)
    ex = ((L.predict(thh, tok) - L.predict(th.unsqueeze(1).expand(-1, T, -1, -1), tok)) ** 2).sum(-1)
    sol = torch.linalg.solve(Pm, tok.unsqueeze(-1)).squeeze(-1)
    theory = L.N * L.SIGMA ** 2 * (sol * tok).sum(-1)
    assert torch.allclose(ex.mean(0), theory.mean(0), rtol=0.05)


def test_model_causal_and_hooks():
    torch.manual_seed(0)
    m = GPT(GPTConfig(d_in=3, d_out=2, T=16)).eval()
    x = torch.randn(2, 16, 3)
    y = m(x)
    x2 = x.clone(); x2[:, 10:] += 5
    assert torch.allclose(m(x2)[:, :10], y[:, :10], atol=1e-5)
    y_e, res = m(x, return_resid=True, edit=lambda l, h: h)
    assert torch.allclose(y_e, y, atol=1e-6) and len(res) == 5
    for l in range(5):
        assert torch.allclose(m.forward_from(res[l], l), y, atol=1e-5)
    y_h = m(x, head_edit=lambda l, o: o)
    assert torch.allclose(y_h, y, atol=1e-5)
    cache = m.new_cache()
    inc = torch.stack([m.step(x[:, t], t, cache) for t in range(16)], 1)
    assert torch.allclose(inc, y, atol=1e-5)


def test_probe_recovers_linear_target():
    X = torch.randn(3000, 20, dtype=torch.float64)
    W = torch.randn(20, 3, dtype=torch.float64)
    Y = X @ W + 0.01 * torch.randn(3000, 3, dtype=torch.float64)
    grp = np.arange(3000) // 10
    pr = RidgeProbe().fit(X[:2000], Y[:2000], grp[:2000])
    assert r2_from_parts(r2_parts(pr.predict(X[2000:]), Y[2000:], grp[2000:])) > 0.99
    D = pr.direction_matrix().double()
    h = torch.randn(1, 20, dtype=torch.float64)
    dh = torch.linalg.pinv(D)[1]
    assert torch.allclose(pr.predict(h + dh) - pr.predict(h), torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64), atol=1e-3)


def test_lqr_scalar_riccati():
    A = torch.tensor([[[1.1, 0.0], [0.0, 0.5]]])
    B = torch.tensor([[[1.0], [0.0]]])
    K = C.lqr_gains(A, B, 200)
    p = 1.0
    for _ in range(500):  # scalar Riccati for the controllable mode
        k = 1.1 * p / (C.R_U + p)
        p = 1 + 1.1 * p * (1.1 - k)
    assert abs(K[0, 0, 0, 0].item() - 1.1 * p / (C.R_U + p)) < 1e-4
    assert abs(K[0, 0, 0, 1].item()) < 1e-6


def test_closed_loop_nonidentifiability():
    """Proposition 2: under u = Kx the regressor matrix has rank n, the direction (K^T, -1) is unexcited."""
    gen = g(7)
    A, B = C.sample_plants(4, gen)
    x0, w = C.sample_noise(4, gen)
    Kc = torch.tensor([[0.3, -0.2]])
    _, X, U = C.simulate(A, B, x0, w, C.fixed_policy(Kc))
    Z = torch.cat([X[:, :C.H], U], -1)
    S = Z.transpose(1, 2) @ Z
    v = torch.cat([Kc[0], torch.ones(1)])  # u_s + K x_s = 0 for all s, so S v = 0
    assert torch.allclose(S @ v, torch.zeros(4, 3), atol=1e-3 * S.abs().max())


def test_oracle_beats_zero_control():
    gen = g(8)
    A, B = C.sample_plants(256, gen)
    x0, w = C.sample_noise(256, gen)
    Jo, _, _ = C.simulate(A, B, x0, w, C.oracle_policy(A, B))
    Jz, _, _ = C.simulate(A, B, x0, w, C.fixed_policy(torch.zeros(1, 2)))
    assert (Jo <= Jz + 1e-4).float().mean() > 0.99


def test_mujoco_params_change_dynamics():
    for name in ["pendulum", "block"]:
        D = mj.gen_episodes(name, 20, 0)
        assert D["obs"].shape[1] == mj.T + 1
        nxt = D["obs"][:, 1:]
        err_or = ((D["oracle"] - nxt) ** 2).mean()
        err_nom = ((D["nominal"] - nxt) ** 2).mean()
        assert err_or < err_nom  # true parameters predict better than nominal ones


def test_feature_rls_recovers_pendulum_inverse_inertia():
    D = mj.gen_episodes("pendulum", 10, 1)
    _, beta = mj.feature_rls("pendulum", D["obs"], D["u"], lam=1e-6)
    m = D["params"][:, 0]
    est = beta[:, -1, 0] / mj.DT  # coefficient on u ~ dt / (m l^2)
    assert np.allclose(est, 1 / (m * mj.L_POLE ** 2), rtol=0.1)


def test_holm_and_bootstrap():
    out = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert out["a"][2] and not out["b"][2] and not out["c"][2]
    pt, dist = hier_boot([np.ones(50), 2 * np.ones(50)], n_boot=200)
    assert pt == 1.5 and dist.min() >= 1 and dist.max() <= 2
