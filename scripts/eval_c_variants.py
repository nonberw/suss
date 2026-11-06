"""Evaluate controller training variants (imitation only, + log-cost, + raw cost, + CVaR) on the same 4096 test plants
and noise as eval_c.py (same generator seed), paired. Writes results/c/variants.pt."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from suss import control as C
from suss.model import GPT, GPTConfig

dev = "cuda"
N_TEST = 4096
gen = torch.Generator(device=dev).manual_seed(10003)
A, B = C.sample_plants(N_TEST, gen, dev)
x0, w = C.sample_noise(N_TEST, gen, dev)
rho = torch.linalg.eigvals(A).abs().amax(-1).cpu().numpy()


def run(pol, chunk=1024):
    Js, Xs, Us = [], [], []
    for i in range(0, N_TEST, chunk):
        sl = slice(i, i + chunk)
        with torch.no_grad():
            J, X, U = C.simulate(A[sl], B[sl], x0[sl], w[sl], pol(sl))
        Js.append(J); Xs.append(X); Us.append(U)
    return torch.cat(Js), torch.cat(Xs), torch.cat(Us)


Jo, _, _ = run(lambda sl: C.oracle_policy(A[sl], B[sl]))
variants = {"imitation": "runs/c/bc_seed{s}.pt", "log-cost": "runs/c/seed{s}.pt", "raw-cost": "runs/c/seed{s}_raw.pt", "cvar": "runs/c/seed{s}_cvar.pt"}
res = {"rho": rho}
for name, pat in variants.items():
    for s in [0, 1, 2]:
        path = pat.format(s=s)
        if not os.path.exists(path):
            continue
        ck = torch.load(path, map_location=dev)
        m = GPT(GPTConfig(**ck["cfg"])).to(dev).eval()
        m.load_state_dict(ck["state"])
        J, X, U = run(lambda sl: C.tf_policy(m, bool(ck.get("args", {}).get("squash", 0))))
        r = (J / Jo).cpu().numpy()
        res[(name, s)] = dict(norm=r, info8=C.info_gain(X, U).cpu().numpy(), selected=ck.get("args", {}))
        print(f"{name:10s} seed {s}: median {np.median(r):.3f} mean {r.mean():.1f} p90 {np.percentile(r, 90):.2f} fail {(r > 10).mean() * 100:.1f}% "
              f"fail_unstable {(r[rho > 1] > 10).mean() * 100:.1f}% info8 {res[(name, s)]['info8'].mean():.2f}", flush=True)
os.makedirs("results/c", exist_ok=True)
torch.save(res, "results/c/variants.pt")
