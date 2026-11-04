"""Experiment B evaluation + mechanistic analysis (probes for physical parameters, steering)."""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from suss import mj
from suss.model import GPT, GPTConfig
from suss.probes import RidgeProbe, r2_from_parts, r2_parts
from scripts.train_b import load

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--out", default="results/b")
a = ap.parse_args()
dev = "cuda"
os.makedirs(a.out, exist_ok=True)
VEL = {"pendulum": 2, "block": 1}


def load_model(path, d_in, d_out):
    ck = torch.load(path)
    if ck["cfg"] is None:
        m = nn.Sequential(nn.Linear(d_in, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, d_out))
    else:
        m = GPT(GPTConfig(**ck["cfg"]))
    m.load_state_dict(ck["state"])
    return m.to(dev).eval(), {k: v.to(dev) for k, v in ck["norm"].items()}


for name in ["pendulum", "block"]:
    D, tok, nxt = load(name, "test")
    obs = D["obs"]
    Y = nxt.cpu().numpy()
    var = Y.reshape(-1, Y.shape[-1]).var(0)

    def nerr(p):  # per-episode, per-t normalised squared error averaged over dims; and velocity only
        e = (p - Y) ** 2 / var
        return e.mean(-1), e[..., VEL[name]]

    rls, beta = mj.feature_rls(name, obs, D["u"])
    base = {"oracle": D["oracle"], "nominal": D["nominal"], "feature_rls": rls, "persistence": obs[:, :-1]}
    out = {}
    for k, p in base.items():
        out[f"{k}_err"], out[f"{k}_vel"] = nerr(p)
    keys = list(D["keys"])
    prm = D["params"]
    if name == "pendulum":
        targets = {"g": prm[:, 2], "inv_m": 1 / prm[:, 0], "c_over_m": prm[:, 1] / prm[:, 0], "log_m": np.log(prm[:, 0]), "c": prm[:, 1]}
    else:
        targets = {"mu": prm[:, 1], "inv_m": 1 / prm[:, 0], "log_m": np.log(prm[:, 0])}
    E = tok.shape[0]; half = E // 2
    mech = {"r2": {}, "parts": {}}
    for s in a.seeds:
        mlp, st = load_model(f"runs/b/{name}_mlp_seed{s}.pt", tok.shape[-1], nxt.shape[-1])
        tf, st = load_model(f"runs/b/{name}_tf_seed{s}.pt", tok.shape[-1], nxt.shape[-1])
        X = (tok - st["tm"]) / st["ts"]
        with torch.no_grad():
            for tag, m in (("mlp", mlp), ("tf", tf)):
                yh = torch.cat([m(X[i:i + 512]) for i in range(0, E, 512)]) * st["ys"] + st["ym"]
                out[f"{tag}_err_seed{s}"], out[f"{tag}_vel_seed{s}"] = nerr(yh.cpu().numpy())
            _, res = tf(X, return_resid=True)
        # probes on the final-position residual, train on first half, score on second half
        grp_tr, grp_te = np.arange(half), np.arange(half, E)
        best = {}
        for tname, yv in targets.items():
            Yt = torch.tensor(yv, dtype=torch.float64, device=dev).view(-1, 1)
            for l, r in enumerate(res):
                h = r[:, -1].double()
                pr = RidgeProbe().fit(h[:half], Yt[:half], grp_tr)
                parts = r2_parts(pr.predict(h[half:]), Yt[half:], grp_te)
                mech["parts"][(tname, l, s)] = parts
                r2v = r2_from_parts(parts)
                mech["r2"][(tname, l, s)] = r2v
                if tname not in best or r2v > best[tname][0]:
                    best[tname] = (r2v, l, pr)
        mech[f"best_seed{s}"] = {k: (v[0], v[1]) for k, v in best.items()}
        # R^2 of inv_m and the identifiable targets as a function of t at the best layer (fixed probe)
        for tname in targets:
            r2v, l, pr = best[tname]
            yt = torch.tensor(targets[tname][half:], device=dev, dtype=torch.float64)
            curve = []
            for t in range(tok.shape[1]):
                p = pr.predict(res[l][half:, t].double())[:, 0]
                curve.append(float(1 - ((p - yt) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()))
            mech[(tname, "by_t", s)] = curve
        # H4d: steer along the inv_m direction (+1 target SD), measure d y_vel / d u at the last token
        r2v, l, pr = best["inv_m"]
        Dm = pr.direction_matrix().to(dev)  # (128,1)
        dh = torch.linalg.pinv(Dm)[0] * float(np.std(targets["inv_m"]))
        Xg = X[half:].clone().requires_grad_(True)

        def sens(edit):
            Xg.grad = None
            yv = tf(Xg, edit=edit)[:, -1, VEL[name]]
            gx, = torch.autograd.grad(yv.sum(), Xg)
            return (gx[:, -1, -1] * st["ys"][VEL[name]] / st["ts"][-1]).detach()  # physical units

        def edit(li, h):
            if li == l:
                h = h.clone(); h[:, -1] = h[:, -1] + dh
            return h
        s0 = sens(None)
        s1 = sens(edit)
        mech[f"steer_seed{s}"] = dict(base=s0.cpu().numpy(), steered=s1.cpu().numpy(), layer=l)
        print(name, "seed", s, "best", {k: (round(v[0], 3), v[1]) for k, v in best.items()},
              "steer d_sens mean", float((s1 - s0).mean()), "frac>0", float(((s1 - s0) > 0).float().mean()))
    np.savez(f"{a.out}/{name}_eval.npz", **out)
    torch.save(mech, f"{a.out}/{name}_mech.pt")
    sl = slice(24, 48)
    print(name, {k: float(v[:, sl].mean()) for k, v in out.items() if "_err" in k})
