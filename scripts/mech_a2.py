"""Experiment A, EXPLORATORY follow-up (added after the pre-registered probes for theta_hat failed):
what *is* represented, if not the parameter estimate?
 (1) linear probes for the Bayes prediction  yhat_Bayes(t) = theta_hat_t z_t  (query-specific answer);
 (2) MLP probes for theta_hat (is the information there but not linearly readable?);
 (3) position-specific linear probes for theta_hat at t = 47 only;
 (4) logit lens: excess risk of readout(resid_l);
 (5) causal: patch / steer along the yhat-probe subspace at each layer."""
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from suss import linear as L
from suss.model import GPT, GPTConfig
from suss.probes import RidgeProbe, r2_from_parts, r2_parts

dev = "cuda"
R = torch.load("results/a/references.pt")
tok, theta = R["tok"], R["theta"]
E, T, _ = tok.shape
half, t0 = E // 2, 4
yb = L.predict(R["th_bayes"], tok)  # (E,T,2)
thb = R["th_bayes"].reshape(E, T, -1)
g_tr = np.repeat(np.arange(half), T - t0)
g_te = np.repeat(np.arange(half, E), T - t0)


def flat(x, a, b):
    return x[a:b, t0:].reshape(-1, x.shape[-1])


def mlp_probe_r2(Xtr, Ytr, Xte, Yte, seed=0, steps=3000):
    torch.manual_seed(seed)
    mx, sx, my, sy = Xtr.mean(0), Xtr.std(0) + 1e-6, Ytr.mean(0), Ytr.std(0) + 1e-6
    net = nn.Sequential(nn.Linear(Xtr.shape[1], 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, Ytr.shape[1])).to(dev)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    Xs, Ys = (Xtr - mx) / sx, (Ytr - my) / sy
    for i in range(steps):
        idx = torch.randint(0, Xs.shape[0], (1024,), device=dev)
        loss = ((net(Xs[idx]) - Ys[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        P = net((Xte - mx) / sx) * sy + my
    return float((1 - ((P - Yte) ** 2).sum(0) / ((Yte - Yte.mean(0)) ** 2).sum(0)).mean())


out = {}
for s in [0, 1, 2]:
    ck = torch.load(f"runs/a/seed{s}.pt")
    model = GPT(GPTConfig(**ck["cfg"])).to(dev).eval()
    model.load_state_dict(ck["state"])
    with torch.no_grad():
        _, res = model(tok, return_resid=True)
    r = {"lin_yhat": [], "lin_ytrue_mean": [], "mlp_theta": [], "lin_theta_t47": [], "mlp_theta_t47": [], "lens": [],
         "tau_yhat_sub": [], "steer_yhat": []}
    mean = R["mean"]
    probes = []
    for l in range(5):
        Xtr, Xte = flat(res[l], 0, half), flat(res[l], half, E)
        pr = RidgeProbe().fit(Xtr, flat(yb, 0, half), g_tr)
        r["lin_yhat"].append(r2_from_parts(r2_parts(pr.predict(Xte), flat(yb, half, E), g_te)))
        probes.append(pr)
        pr2 = RidgeProbe().fit(Xtr, flat(mean, 0, half), g_tr)
        r["lin_ytrue_mean"].append(r2_from_parts(r2_parts(pr2.predict(Xte), flat(mean, half, E), g_te)))
        r["mlp_theta"].append(mlp_probe_r2(Xtr, flat(thb, 0, half), Xte, flat(thb, half, E), seed=s))
        h47 = res[l][:, -1]
        pr3 = RidgeProbe().fit(h47[:half], thb[:half, -1], np.arange(half))
        r["lin_theta_t47"].append(r2_from_parts(r2_parts(pr3.predict(h47[half:]), thb[half:, -1], np.arange(half, E))))
        r["mlp_theta_t47"].append(mlp_probe_r2(h47[:half], thb[:half, -1], h47[half:], thb[half:, -1], seed=s, steps=2000))
        with torch.no_grad():
            r["lens"].append(float(((model.readout(res[l][half:, t0:]) - mean[half:, t0:]) ** 2).sum(-1).mean()))
    # causal: patch the 2-d yhat-probe subspace of the query token at each layer (A' -> B), and steer along it
    idxB = torch.arange(half, E, device=dev); idxA = torch.roll(idxB, -1)
    tokA2 = torch.cat([tok[idxA, :-1], tok[idxB, -1:]], 1)
    with torch.no_grad():
        _, rB = model(tok[idxB], return_resid=True)
        _, rA = model(tokA2, return_resid=True)
        yB, yA = model.readout(rB[-1])[:, -1], model.readout(rA[-1])[:, -1]
        d = yA - yB; den = (d ** 2).sum(-1).clamp(min=1e-12)
        for l in range(5):
            D = probes[l].direction_matrix().to(dev)
            Pi = D @ torch.linalg.solve(D.T @ D, D.T)
            h = rB[l].clone(); h[:, -1] = h[:, -1] + (rA[l][:, -1] - rB[l][:, -1]) @ Pi
            yp = model.forward_from(h, l)[:, -1]
            r["tau_yhat_sub"].append(float((((yp - yB) * d).sum(-1) / den).mean()))
            Dp = torch.linalg.pinv(D)
            obs, pred = [], []
            for k in range(2):
                for delta in (0.1, -0.1):
                    h = rB[l].clone(); h[:, -1] += delta * Dp[k]
                    dy = model.forward_from(h, l)[:, -1] - yB
                    p = torch.zeros_like(dy); p[:, k] = delta
                    obs.append(dy.ravel()); pred.append(p.ravel())
            o, p = torch.cat(obs).cpu().numpy(), torch.cat(pred).cpu().numpy()
            c = np.cov(p, o)
            r["steer_yhat"].append((float(c[0, 1] / c[0, 0]), float(c[0, 1] / np.sqrt(c[0, 0] * c[1, 1]))))
    out[s] = r
    print(f"seed {s}:", {k: np.round(v, 3).tolist() for k, v in r.items()}, flush=True)
torch.save(out, "results/a/mech2.pt")
