"""Experiment C1 evaluation: tune baselines on training plants, evaluate all controllers on a paired
test set, measure excitation/information gain, and probe the controller for implicit system ID."""
import argparse
import itertools
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from suss import control as C
from suss.model import GPT, GPTConfig
from suss.probes import RidgeProbe, r2_from_parts, r2_parts

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--out", default="results/c")
ap.add_argument("--n_test", type=int, default=4096)
a = ap.parse_args()
dev = "cuda"
os.makedirs(a.out, exist_ok=True)


def plants(n, seed):
    gen = torch.Generator(device=dev).manual_seed(seed)
    A, B = C.sample_plants(n, gen, dev)
    x0, w = C.sample_noise(n, gen, dev)
    return A, B, x0, w


def run(pol, P, chunk=1024):
    A, B, x0, w = P
    Js, Xs, Us = [], [], []
    for i in range(0, A.shape[0], chunk):
        sl = slice(i, i + chunk)
        with torch.no_grad():
            J, X, U = C.simulate(A[sl], B[sl], x0[sl], w[sl], pol(sl))
        Js.append(J); Xs.append(X); Us.append(U)
    return torch.cat(Js), torch.cat(Xs), torch.cat(Us)


def oracle(P):
    A, B = P[0], P[1]
    return lambda sl: C.oracle_policy(A[sl], B[sl])


def dgen(seed):
    return torch.Generator(device=dev).manual_seed(seed)


# ---- tuning on 1024 training plants (seed 30001)
tuneP = plants(1024, 30001)
Jo_t, _, _ = run(oracle(tuneP), tuneP)
cost = lambda J: float((J / Jo_t).clamp(max=1e3).mean())
best_ce = min([0.01, 0.1, 1.0], key=lambda lam: cost(run(lambda sl: C.ce_policy(lam), tuneP)[0]))
grid = list(itertools.product([0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6], [0.01, 0.1, 1.0]))
scores = {g: cost(run(lambda sl, g=g: C.ce_policy(g[1], g[0], dgen(1)), tuneP)[0]) for g in grid}
best_d = min(scores, key=scores.get)
grid2 = list(itertools.product([2, 4, 8], [0.5, 1.0, 2.0]))
scores2 = {g: cost(run(lambda sl, g=g: C.ce_policy(best_ce, 0.0, dgen(2), g[0], g[1]), tuneP)[0]) for g in grid2}
best_e = min(scores2, key=scores2.get)
grid3 = list(itertools.product([0.01, 0.1, 1.0], [0.5, 1.0, 2.0]))
scores3 = {g: cost(run(lambda sl, g=g: C.ts_policy(g[0], dgen(5), g[1]), tuneP)[0]) for g in grid3}
best_ts = min(scores3, key=scores3.get)
print("tuned TS (lam, inflate)", best_ts)
print("tuned: CE lam", best_ce, "CE+dither (s0, lam)", best_d, "ETC (k, s_e)", best_e)

# ---- test
P = plants(a.n_test, 10003)
Jo, Xo, Uo = run(oracle(P), P)
res = {"tuning": dict(ce_lam=best_ce, dither=best_d, etc=best_e, ts=best_ts, dither_scores=scores, etc_scores=scores2, ts_scores=scores3)}
ctrl = {
    "ce": lambda sl: C.ce_policy(best_ce),
    "ce_dither": lambda sl: C.ce_policy(best_d[1], best_d[0], dgen(3)),
    "etc": lambda sl: C.ce_policy(best_ce, 0.0, dgen(4), best_e[0], best_e[1]),
    "ts": lambda sl: C.ts_policy(best_ts[0], dgen(6), best_ts[1]),  # exploratory baseline (added post hoc)
}
traj = {}
for k, pol in ctrl.items():
    J, X, U = run(pol, P)
    traj[k] = (X, U)
    res[k] = dict(norm=(J / Jo).cpu().numpy(), info8=C.info_gain(X, U).cpu().numpy())
res["oracle"] = dict(J=Jo.cpu().numpy(), info8=C.info_gain(Xo, Uo).cpu().numpy())
for s in a.seeds:
    Kc = torch.load(f"runs/c/fixed_seed{s}.pt")["K"].to(dev)
    J, X, U = run(lambda sl: C.fixed_policy(Kc), P)
    res[f"fixed_seed{s}"] = dict(norm=(J / Jo).cpu().numpy(), K=Kc.cpu().numpy())
    if os.path.exists(f"runs/c/bc_seed{s}.pt"):  # stage-1 (imitation only) controller, for the ablation
        ckb = torch.load(f"runs/c/bc_seed{s}.pt")
        mb = GPT(GPTConfig(**ckb["cfg"])).to(dev).eval()
        mb.load_state_dict(ckb["state"])
        Jb, Xb, Ub = run(lambda sl: C.tf_policy(mb), P)
        res[f"bc_seed{s}"] = dict(norm=(Jb / Jo).cpu().numpy(), info8=C.info_gain(Xb, Ub).cpu().numpy())
    ck = torch.load(f"runs/c/seed{s}.pt")
    model = GPT(GPTConfig(**ck["cfg"])).to(dev).eval()
    model.load_state_dict(ck["state"])
    sq = bool(ck.get("args", {}).get("squash", 0))
    J, X, U = run(lambda sl: C.tf_policy(model, sq), P)
    # deviation of TF action from the CE action computed on TF's own history (excitation beyond CE)
    dev_ce = []
    with torch.no_grad():
        xs = [X[:, t] for t in range(C.H + 1)]
        us = [U[:, t] for t in range(C.H)]
        ce = C.ce_policy(best_ce)
        for t in range(C.H):
            uce = ce(t, xs[:t + 1], us[:t]).clamp(-C.U_MAX, C.U_MAX)
            dev_ce.append(((U[:, t] - uce) ** 2).sum(-1))
    res[f"tf_seed{s}"] = dict(norm=(J / Jo).cpu().numpy(), info8=C.info_gain(X, U).cpu().numpy(),
                              dev_ce=torch.stack(dev_ce, 1).cpu().numpy(), u_abs=U.abs()[..., 0].cpu().numpy())
    # implicit system identification probes (H5e): residual stream on own trajectories
    tok = C.tokens_from_traj(X, U, sq)
    with torch.no_grad():
        _, rs = model(tok, return_resid=True)
    E = tok.shape[0]; half = E // 2; t0 = 4
    Bt = P[1][..., 0]  # (E,2)
    Bh = torch.stack([C.ridge_estimate(xs[:t + 1], us[:t], best_ce)[:, :, 2] for t in range(t0, C.H)], 1)  # (E,T-t0,2)
    tgt = {"B_true": Bt.unsqueeze(1).expand(-1, C.H - t0, -1), "B_hat": Bh}
    g_tr = np.repeat(np.arange(half), C.H - t0); g_te = np.repeat(np.arange(half, E), C.H - t0)
    pr_res = {}
    for l, r in enumerate(rs):
        h = r[:, t0:]
        for tn, Y in tgt.items():
            pr = RidgeProbe().fit(h[:half].reshape(-1, 128), Y[:half].reshape(-1, 2), g_tr)
            parts = r2_parts(pr.predict(h[half:].reshape(-1, 128)), Y[half:].reshape(-1, 2), g_te)
            pr_res[(tn, l)] = (r2_from_parts(parts), parts, pr)
    res[f"tf_seed{s}"]["probe"] = {k: (v[0], v[1]) for k, v in pr_res.items()}
    # exploratory steering: flip the probed B_hat at t=20 and check whether u_20 flips sign
    lb = max(range(len(rs)), key=lambda l: pr_res[("B_hat", l)][0])
    pr = pr_res[("B_hat", lb)][2]
    Dm = pr.direction_matrix().to(dev)
    t_s = 20
    with torch.no_grad():
        h = rs[lb][half:, :t_s + 1].clone()
        b_read = pr.predict(h[:, -1]).float()  # (E/2, 2)
        h[:, -1] = h[:, -1] + (-2 * b_read) @ torch.linalg.pinv(Dm)
        u0 = C.U_MAX * torch.tanh(model.forward_from(rs[lb][half:, :t_s + 1], lb)[:, -1, 0])
        u1 = C.U_MAX * torch.tanh(model.forward_from(h, lb)[:, -1, 0])
    res[f"tf_seed{s}"]["steer_flip"] = dict(layer=lb, u0=u0.cpu().numpy(), u1=u1.cpu().numpy())
    r = res[f"tf_seed{s}"]
    print(f"seed {s}: TF norm mean {r['norm'].mean():.3f} median {np.median(r['norm']):.3f} fail {(r['norm'] > 10).mean():.3f} | "
          f"fixed {res[f'fixed_seed{s}']['norm'].mean():.1f} | info8 TF {r['info8'].mean():.2f} CE {res['ce']['info8'].mean():.2f} | "
          f"probe B_true {max(v[0] for k, v in pr_res.items() if k[0] == 'B_true'):.3f} B_hat {pr_res[('B_hat', lb)][0]:.3f} | "
          f"flip frac {float((np.sign(u0.cpu()) != np.sign(u1.cpu())).float().mean()):.3f}")
for k in ctrl:
    v = res[k]["norm"]
    print(k, f"mean {v.mean():.3f} median {np.median(v):.3f} fail {(v > 10).mean():.3f} info8 {res[k]['info8'].mean():.2f}")
torch.save(res, f"{a.out}/eval.pt")
