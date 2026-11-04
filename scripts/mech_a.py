"""Experiment A mechanistic analysis: probes, steering, patching, head roles.
Uses results/a/references.pt (from eval_a.py). Writes results/a/mech_seed{s}.pt."""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from suss import linear as L
from suss.model import GPT, GPTConfig
from suss.probes import RidgeProbe, r2_parts

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--runs", default="runs/a")
ap.add_argument("--out", default="results/a")
a = ap.parse_args()
dev = "cuda"
R = torch.load(f"{a.out}/references.pt")
tok, nxt, theta = R["tok"], R["nxt"], R["theta"]
E, T, _ = tok.shape
half = E // 2
t0 = 4
pos = torch.arange(t0, T, device=dev)
iu = torch.triu_indices(L.P, L.P)

# ---- probe targets, (E, T, k)
norm = (torch.arange(T, device=dev) + 1.0).view(1, T, 1)
targets = {
    "theta_true": theta.reshape(E, 1, -1).expand(E, T, -1),
    "theta_bayes": R["th_bayes"].reshape(E, T, -1),
    "theta_ols": L.ols_theta(R["S"], R["C"]).reshape(E, T, -1).clamp(-5, 5),
    "moments_S": R["S"][:, :, iu[0], iu[1]] / norm,
    "moments_C": R["C"].reshape(E, T, -1) / norm,
    "post_logtrace": L.posterior_logtrace(R["S"]).unsqueeze(-1),
}
targets["moments"] = torch.cat([targets["moments_S"], targets["moments_C"]], -1)


def flat(x, idx):  # (E,T,k) -> rows for episodes idx and positions t0..T-1
    return x[idx][:, t0:].reshape(-1, x.shape[-1])


ep_tr = torch.arange(half, device=dev)
ep_te = torch.arange(half, E, device=dev)
g_tr = np.repeat(np.arange(half), T - t0)
g_te = np.repeat(np.arange(half, E), T - t0)


def run_resid(model, x, bs=512):
    out = None
    with torch.no_grad():
        for i in range(0, x.shape[0], bs):
            _, res = model(x[i:i + bs], return_resid=True)
            res = [r.float() for r in res]
            if out is None:
                out = [[] for _ in res]
            for l, r in enumerate(res):
                out[l].append(r)
    return [torch.cat(o) for o in out]


for s in a.seeds:
    ck = torch.load(f"{a.runs}/seed{s}.pt")
    model = GPT(GPTConfig(**ck["cfg"])).to(dev).eval()
    model.load_state_dict(ck["state"])
    nL = model.cfg.n_layer
    resid = run_resid(model, tok)
    res = {"probe": {}, "probe_by_t": {}}
    probes = {}
    # ---- probes per layer and target
    for l in range(nL + 1):
        Xtr, Xte = flat(resid[l], ep_tr), flat(resid[l], ep_te)
        for name, Y in targets.items():
            pr = RidgeProbe().fit(Xtr, flat(Y, ep_tr), g_tr)
            parts = r2_parts(pr.predict(Xte), flat(Y, ep_te), g_te)
            res["probe"][(name, l)] = parts
            probes[(name, l)] = pr
    from suss.probes import r2_from_parts
    r2 = {k: r2_from_parts(v) for k, v in res["probe"].items()}
    l_star = max(range(nL + 1), key=lambda l: r2[("theta_bayes", l)])
    l_mom = max(range(nL + 1), key=lambda l: r2[("moments", l)])
    res["l_star"], res["l_mom"], res["r2"] = l_star, l_mom, r2
    print(f"seed {s}: l*={l_star} l_mom={l_mom}", {k: round(v, 3) for k, v in r2.items() if k[0] in ("theta_bayes", "theta_true", "moments")})

    # R^2 of theta_bayes probe at l* as a function of t (held-out)
    pr = probes[("theta_bayes", l_star)]
    yh = pr.predict(resid[l_star][ep_te].reshape(-1, 128)).view(len(ep_te), T, -1)
    Yb = targets["theta_bayes"][ep_te]
    res["r2_by_t"] = [float(1 - ((yh[:, t] - Yb[:, t]) ** 2).sum(0).div(((Yb[:, t] - Yb[:, t].mean(0)) ** 2).sum(0)).mean()) for t in range(T)]

    # ---- steering at l*, last position (H3a)
    D = pr.direction_matrix().to(dev)  # (128, 6): readout = h @ D + c
    Dp = torch.linalg.pinv(D)  # (6, 128): minimal-norm residual change per unit readout change
    h_te = resid[l_star][ep_te]
    z = tok[ep_te, -1]  # query token (x_47, u_47)
    with torch.no_grad():
        y0 = model.forward_from(h_te, l_star)[:, -1]
        obs, pred, epi = [], [], []
        for k in range(L.N * L.P):
            i, j = divmod(k, L.P)
            for delta in (0.25, -0.25):
                h = h_te.clone()
                h[:, -1] += delta * Dp[k]
                dy = model.forward_from(h, l_star)[:, -1] - y0
                p = torch.zeros_like(dy)
                p[:, i] = delta * z[:, j]
                obs.append(dy); pred.append(p); epi.append(ep_te)
        res["steer"] = dict(obs=torch.stack(obs).cpu().numpy(), pred=torch.stack(pred).cpu().numpy())

    # ---- patching (H3b + exploratory), pairs (A=i+1, B=i) within the evaluation half
    idxB = ep_te
    idxA = torch.roll(ep_te, -1)
    tokB = tok[idxB]
    tokA2 = torch.cat([tok[idxA, :-1], tok[idxB, -1:]], 1)  # A' = A context + B query
    with torch.no_grad():
        rB = run_resid(model, tokB)
        rA = run_resid(model, tokA2)
        yB = model.readout(rB[-1])[:, -1]
        yA = model.readout(rA[-1])[:, -1]
        dAB = yA - yB
        den = (dAB ** 2).sum(-1).clamp(min=1e-12)

        def tau(y):
            return (((y - yB) * dAB).sum(-1) / den).cpu().numpy()

        Pi = D @ torch.linalg.solve(D.T @ D, D.T)  # projector onto readout subspace
        h = rB[l_star].clone()
        h[:, -1] = h[:, -1] + (rA[l_star][:, -1] - rB[l_star][:, -1]) @ Pi
        res["tau_sub"] = tau(model.forward_from(h, l_star)[:, -1])
        # random-subspace control of the same dimension
        Q = torch.linalg.qr(torch.randn(128, D.shape[1], device=dev, generator=torch.Generator(device=dev).manual_seed(s)))[0]
        h = rB[l_star].clone()
        h[:, -1] = h[:, -1] + (rA[l_star][:, -1] - rB[l_star][:, -1]) @ (Q @ Q.T)
        res["tau_rand"] = tau(model.forward_from(h, l_star)[:, -1])
        res["tau_ctx"], res["tau_last"] = [], []
        for l in range(nL + 1):
            h = rB[l].clone(); h[:, :-1] = rA[l][:, :-1]
            res["tau_ctx"].append(tau(model.forward_from(h, l)[:, -1]))
            h = rB[l].clone(); h[:, -1] = rA[l][:, -1]
            res["tau_last"].append(tau(model.forward_from(h, l)[:, -1]))
        # per-head patching at the last position
        capA = {}
        model(tokA2, capture=capA)
        tau_head = np.zeros((nL, model.cfg.n_head))
        for l in range(nL):
            for hh in range(model.cfg.n_head):
                def he(layer, o, l=l, hh=hh):
                    if layer == l:
                        o = o.clone(); o[:, hh, -1] = capA["heads"][l][:, hh, -1]
                    return o
                tau_head[l, hh] = tau(model(tokB, head_edit=he)[:, -1]).mean()
        res["tau_head"] = tau_head
        # head roles: attention entropy (normalised), previous-token mass, mean-ablation loss increase
        cap = {}
        model(tok[ep_te], capture=cap)
        ent = np.zeros((nL, model.cfg.n_head)); prev = np.zeros_like(ent); abl = np.zeros_like(ent); self_ = np.zeros_like(ent)
        tt = torch.arange(T, device=dev)
        for l in range(nL):
            att = cap["attn"][l]  # (E,H,T,T)
            e = -(att.clamp(min=1e-12).log() * att).sum(-1)  # (E,H,T)
            e = e[:, :, t0:] / torch.log(tt[t0:].float() + 1)
            ent[l] = e.mean((0, 2)).cpu().numpy()
            prev[l] = att[:, :, tt[1:], tt[:-1]][:, :, t0 - 1:].mean((0, 2)).cpu().numpy()
            self_[l] = att[:, :, tt, tt][:, :, t0:].mean((0, 2)).cpu().numpy()
        base_loss = ((model(tok[ep_te]) - R["mean"][ep_te]) ** 2).sum(-1).mean().item()
        for l in range(nL):
            for hh in range(model.cfg.n_head):
                mu = cap["heads"][l][:, hh].mean(0, keepdim=True)
                def he(layer, o, l=l, hh=hh, mu=mu):
                    if layer == l:
                        o = o.clone(); o[:, hh] = mu
                    return o
                abl[l, hh] = ((model(tok[ep_te], head_edit=he) - R["mean"][ep_te]) ** 2).sum(-1).mean().item() - base_loss
        res.update(entropy=ent, prev_token=prev, self_attn=self_, ablation=abl, base_excess=base_loss)
    torch.save(res, f"{a.out}/mech_seed{s}.pt")
    print(f"seed {s}: tau_sub {res['tau_sub'].mean():.3f} tau_rand {res['tau_rand'].mean():.3f} "
          f"tau_last {[round(float(x.mean()), 2) for x in res['tau_last']]} tau_ctx {[round(float(x.mean()), 2) for x in res['tau_ctx']]}")
    print("tau_head", np.round(tau_head, 2).tolist())
    print("entropy", np.round(ent, 2).tolist(), "prev", np.round(prev, 2).tolist(), "ablation", np.round(abl, 4).tolist())
