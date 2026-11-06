"""Experiment A, EXPLORATORY controls, round 2 (after external review). CPU.
 (1) shuffled-label baseline for the fast-weight probe (targets permuted across episodes);
 (2) mean-pooled residual over the context and the last context token (position 46) as probe features;
 (3) ceiling: how well theta_bayes_47 is a linear / MLP function of the true normalised statistics (S,C)_47;
 (4) numerical check of the first-order attention picture (Remark): per head, R^2 of
       o_head(t=47) ~ vbar + (1/(t+1)) sum_s (v_s - vbar)(k_s - kbar)^T q / sqrt(d_h)   against the actual head output,
     plus the normalised attention entropy;
 (5) joint K,V patching of the top-k heads (by single-head transfer) vs k random heads, k in {2,3}, blocks 3-4.
Writes results/a/mech6.pt."""
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from suss import linear as L
from suss.model import GPT, GPTConfig
from scripts.mech_a3 import fast_weights
from scripts.mech_a4 import run as kv_run
from scripts.mech_a5 import dual_ridge_r2

dev = os.environ.get("SUSS_DEV", "cpu")
torch.set_num_threads(int(os.environ.get("SUSS_THREADS", "6")))
SEEDS = [int(x) for x in os.environ.get("SUSS_SEEDS", "0,1,2").split(",")]
OUT = os.environ.get("SUSS_OUT", "results/a/mech6.pt")


def mlp_r2(Xtr, Ytr, Xte, Yte, seed=0, steps=3000):
    torch.manual_seed(seed)
    Xtr, Ytr, Xte, Yte = [a.float() for a in (Xtr, Ytr, Xte, Yte)]
    mx, sx, my, sy = Xtr.mean(0), Xtr.std(0) + 1e-6, Ytr.mean(0), Ytr.std(0) + 1e-6
    net = nn.Sequential(nn.Linear(Xtr.shape[1], 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, Ytr.shape[1]))
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    Xs, Ys = (Xtr - mx) / sx, (Ytr - my) / sy
    for _ in range(steps):
        idx = torch.randint(0, Xs.shape[0], (512,))
        loss = ((net(Xs[idx]) - Ys[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        P = net((Xte - mx) / sx) * sy + my
    return float((1 - ((P - Yte) ** 2).sum(0) / ((Yte - Yte.mean(0)) ** 2).sum(0)).mean())


def head_io(model, tok, l, t):
    """Per-head actual output and first-order approximation at position t for block l. Returns (o, approx, ent): (E,H,dh),(E,H,dh),(E,H)."""
    with torch.no_grad():
        _, res = model(tok[:, :t + 1], return_resid=True)
        blk = model.blocks[l]
        x = blk.ln1(res[l])
        D = model.cfg.d_model; H = model.cfg.n_head; dh = D // H
        q, k, v = blk.qkv(x).split(D, -1)
        q = q.view(-1, t + 1, H, dh)[:, t]  # (E,H,dh)
        k = k.view(-1, t + 1, H, dh); v = v.view(-1, t + 1, H, dh)
        att = torch.einsum("ehd,eshd->ehs", q, k) / math.sqrt(dh)
        p = att.softmax(-1)
        o = torch.einsum("ehs,eshd->ehd", p, v)
        vbar, kbar = v.mean(1), k.mean(1)
        cov_q = torch.einsum("eshd,eshj,ehj->ehd", v - vbar[:, None], k - kbar[:, None], q) / (t + 1)
        approx = vbar + cov_q / math.sqrt(dh)
        ent = -(p.clamp(min=1e-12).log() * p).sum(-1) / math.log(t + 1)
    return o, approx, ent


if __name__ == "__main__":
    R = torch.load("results/a/references.pt", map_location=dev)
    tok, theta = R["tok"], R["theta"]
    E, T, _ = tok.shape
    half, t = E // 2, T - 1
    iu = torch.triu_indices(L.P, L.P)
    thb = R["th_bayes"][:, t].reshape(E, -1)
    mom = torch.cat([R["S"][:, t][:, iu[0], iu[1]], R["C"][:, t].reshape(E, -1)], 1) / (t + 1)
    out = {"ceiling_linear": dual_ridge_r2(mom[:half], thb[:half], mom[half:], thb[half:])[0],
           "ceiling_mlp": mlp_r2(mom[:half], thb[:half], mom[half:], thb[half:])}
    print("ceiling (S,C)->theta: linear %.3f  mlp %.3f" % (out["ceiling_linear"], out["ceiling_mlp"]), flush=True)
    idxB = torch.arange(half, E); idxA = torch.roll(idxB, -1)
    tokB = tok[idxB]; tokA2 = torch.cat([tok[idxA, :-1], tok[idxB, -1:]], 1)
    for s in SEEDS:
        ck = torch.load(f"runs/a/seed{s}.pt", map_location=dev)
        model = GPT(GPTConfig(**ck["cfg"])).to(dev).eval()
        model.load_state_dict(ck["state"])
        nL, H = model.cfg.n_layer, model.cfg.n_head
        r = {}
        with torch.no_grad():
            _, res = model(tok, return_resid=True)
        g = torch.Generator().manual_seed(1000 + s)
        for l in range(nL):
            F = fast_weights(model, tok, l, t).float()
            perm = torch.randperm(half, generator=g)
            r[("shuffled", l)] = dual_ridge_r2(F[:half], thb[:half][perm], F[half:], thb[half:])[0]
            h = res[l + 1].float()
            for fname, X in (("resid_mean", h.mean(1)), ("resid_last_ctx", h[:, t - 1])):
                r[(fname, l, "theta_bayes")] = dual_ridge_r2(X[:half], thb[:half], X[half:], thb[half:])[0]
                r[(fname, l, "moments")] = dual_ridge_r2(X[:half], mom[:half], X[half:], mom[half:])[0]
            o, ap, ent = head_io(model, tok[:1024], l, t)
            num = ((o - ap) ** 2).sum((0, 2)); den = ((o - o.mean(0)) ** 2).sum((0, 2))
            r[("remark_r2", l)] = (1 - num / den).tolist()
            r[("entropy", l)] = ent.mean(0).tolist()
        print(f"seed {s}: shuffled {[round(r[('shuffled', l)], 3) for l in range(nL)]} | resid_mean->theta {[round(r[('resid_mean', l, 'theta_bayes')], 3) for l in range(nL)]} "
              f"| last_ctx->theta {[round(r[('resid_last_ctx', l, 'theta_bayes')], 3) for l in range(nL)]} | remark R2 {[[round(x, 2) for x in r[('remark_r2', l)]] for l in range(nL)]} "
              f"| entropy {[[round(x, 2) for x in r[('entropy', l)]] for l in range(nL)]}", flush=True)
        # (5) top-k vs random-k heads, joint over blocks 3-4
        mech = torch.load(f"results/a/mech_seed{s}.pt", map_location="cpu", weights_only=False)
        tau_head = np.array(mech["tau_head"])  # (nL, H) single-head K,V... (head-output patch); use as ranking
        with torch.no_grad():
            yB, _ = kv_run(model, tokB); yA, kvA = kv_run(model, tokA2)
            yB, yA = yB[:, -1], yA[:, -1]
            d = yA - yB; den = (d ** 2).sum(-1).clamp(min=1e-12)
            tau = lambda y: float((((y[:, -1] - yB) * d).sum(-1) / den).mean())
            for kk in (2, 3):
                top = {l: list(np.argsort(-tau_head[l])[:kk]) for l in (nL - 2, nL - 1)}
                r[("top", kk)] = tau(kv_run(model, tokB, kvA, (nL - 2, nL - 1), "kv", heads=top)[0])
                rs = []
                for rep_ in range(5):
                    rnd = {l: list(torch.randperm(H, generator=g)[:kk].numpy()) for l in (nL - 2, nL - 1)}
                    rs.append(tau(kv_run(model, tokB, kvA, (nL - 2, nL - 1), "kv", heads=rnd)[0]))
                r[("random", kk)] = float(np.mean(rs))
                r[("random_sd", kk)] = float(np.std(rs))
        print(f"seed {s}: top-2 {r[('top', 2)]:.3f} vs random-2 {r[('random', 2)]:.3f}+-{r[('random_sd', 2)]:.3f}; top-3 {r[('top', 3)]:.3f} vs random-3 {r[('random', 3)]:.3f}", flush=True)
        out[s] = r
    torch.save(out, OUT)
