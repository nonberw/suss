"""Pre-registration 2, part B (PREREGISTRATION_2.md, tag prereg2-94b6cf8): under a shifted prior (A entries
~ N(0, 0.1^2) instead of N(0, 0.4^2)), does the network's effective estimate (query Jacobian) follow ridge with the
TRAINING prior, the Bayes estimate for the NEW prior, or OLS? D_X = mean over episodes and t = 8..47 of
||J_t[:, A] - theta_X,t[:, A]||_F^2, for the 10 training seeds of the main model."""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from suss import linear as L
from suss.model import GPT, GPTConfig

dev = os.environ.get("SUSS_DEV", "cuda" if torch.cuda.is_available() else "cpu")
E, T, T0 = 1024, 48, 8
SD_NEW = torch.tensor([0.1, 0.1, L.SD_B])
SEEDS = list(range(10))
OUT = "results/confirm"


def sample_theta(n, sd, gen):
    out, got = [], 0
    while got < n:
        th = torch.randn(2 * n, L.N, L.P, generator=gen) * sd
        th = th[L.spectral_radius_2x2(th[:, :, :L.N]) < L.RHO_MAX]
        out.append(th)
        got += th.shape[0]
    return torch.cat(out)[:n]


def ridge(S, C, lam):
    return torch.linalg.solve(S + torch.diag_embed(lam.expand(S.shape[:-1])), C).transpose(-1, -2)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    gen = torch.Generator().manual_seed(20261107)
    theta = sample_theta(E, SD_NEW, gen)
    tok, nxt = L.rollout(theta, T, gen)
    S, C = L.suff_stats(tok, nxt)
    S, C = S.double(), C.double()
    lam_train = (L.SIGMA / L.prior_sd()) ** 2
    lam_new = (L.SIGMA / SD_NEW) ** 2
    est = {"ridge_train": ridge(S, C, lam_train.double()), "bayes_new": ridge(S, C, lam_new.double()),
           "ols": (torch.linalg.pinv(S) @ C).transpose(-1, -2)}
    tok = tok.to(dev)
    per_seed, curves = {}, {}
    for s in SEEDS:
        ck = torch.load(f"runs/a/seed{s}.pt", map_location=dev)
        model = GPT(GPTConfig(**ck["cfg"])).to(dev).eval()
        model.load_state_dict(ck["state"])
        D = {k: [] for k in est}
        for t in range(T0, T):
            f = lambda q: model(torch.cat([tok[:, :t], q.unsqueeze(1)], 1))[:, t].sum(0)
            J = torch.autograd.functional.jacobian(f, tok[:, t].clone()).permute(1, 0, 2).double().cpu()  # (E, 2, 3)
            for k, th in est.items():
                D[k].append(float(((J[..., :L.N] - th[:, t, :, :L.N]) ** 2).sum((-1, -2)).mean()))
        per_seed[s] = {k: float(np.mean(v)) for k, v in D.items()}
        curves[s] = D
        print(f"seed {s}: " + ", ".join(f"D_{k}={v:.4f}" for k, v in per_seed[s].items()), flush=True)
    # per-seed means and per-t curves; the statistics are in scripts/confirm_stats.py
    json.dump({"E": E, "t_range": [T0, T - 1], "sd_new": SD_NEW.tolist(), "lam_train": lam_train.tolist(),
               "lam_new": lam_new.tolist(), "per_seed": per_seed, "curves": curves}, open(f"{OUT}/b_summary.json", "w"), indent=1)
