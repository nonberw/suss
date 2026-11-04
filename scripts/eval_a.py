"""Experiment A evaluation: transformer vs exact estimators on a fixed test set.
Writes results/a/eval_seed{s}.npz with per-episode, per-position excess risk and algorithmic distances."""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from suss import linear as L
from suss.model import GPT, GPTConfig

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--runs", default="runs/a")
ap.add_argument("--out", default="results/a")
ap.add_argument("--n_test", type=int, default=4096)
a = ap.parse_args()
dev = "cuda"
os.makedirs(a.out, exist_ok=True)
T = 48


def references(n, seed):
    gen = torch.Generator(device=dev).manual_seed(seed)
    theta, tok, nxt = L.make_batch(n, T, gen, dev)
    S, C = L.suff_stats(tok, nxt)
    mean = L.predict(theta.unsqueeze(1).expand(-1, T, -1, -1), tok)  # noiseless target
    th_b, acc = L.bayes_theta(S, C, torch.Generator(device=dev).manual_seed(seed + 1))
    preds = {
        "bayes": L.predict(th_b, tok),
        "ridge": L.predict(L.ridge_theta(S, C), tok),
        "ols": L.predict(L.ols_theta(S, C), tok),
        "zero": torch.zeros_like(mean),
    }
    g = L.predict(L.gd1_theta(S, C, 1.0), tok)
    return dict(theta=theta, tok=tok, nxt=nxt, S=S, C=C, mean=mean, th_bayes=th_b, acc=acc, preds=preds, g=g)


ref_path = f"{a.out}/references.pt"
if os.path.exists(ref_path):
    R = torch.load(ref_path)
else:
    # GD-1 step size tuned on 512 separate episodes (closed form: prediction is linear in eta)
    tune = references(512, 20001)
    eta = float((tune["g"] * tune["mean"]).sum() / (tune["g"] ** 2).sum())
    R = references(a.n_test, 10001)
    R["preds"]["gd1"] = eta * R["g"]
    R["eta"] = eta
    R["theory_ridge_excess"] = L.ridge_excess_theory(R["S"], R["tok"])
    torch.save(R, ref_path)
print("eta_gd1", R["eta"], "bayes acceptance (t=0, t=47):", R["acc"][:, 0].mean().item(), R["acc"][:, -1].mean().item())

mean = R["mean"]
base = {k: ((v - mean) ** 2).sum(-1).cpu().numpy() for k, v in R["preds"].items()}
np.savez(f"{a.out}/references_excess.npz", **base, theory_ridge=R["theory_ridge_excess"].cpu().numpy())

for s in a.seeds:
    ck = torch.load(f"{a.runs}/seed{s}.pt")
    model = GPT(GPTConfig(**ck["cfg"])).to(dev)
    model.load_state_dict(ck["state"])
    model.eval()
    with torch.no_grad():
        yt = torch.cat([model(R["tok"][i:i + 512]) for i in range(0, a.n_test, 512)])
    out = {"tf_excess": ((yt - mean) ** 2).sum(-1).cpu().numpy(),
           "tf_mse": ((yt - R["nxt"]) ** 2).sum(-1).cpu().numpy()}
    for k, v in R["preds"].items():
        out[f"dist_{k}"] = ((yt - v) ** 2).sum(-1).cpu().numpy()
    np.savez(f"{a.out}/eval_seed{s}.npz", **out)
    print(f"seed {s}: TF excess (mean t>=4) {out['tf_excess'][:, 4:].mean():.5f}  "
          f"Bayes {base['bayes'][:, 4:].mean():.5f}  OLS {base['ols'][:, 4:].mean():.5f}  "
          f"d(TF,Bayes) {out['dist_bayes'][:, 1:].mean():.5f} d(TF,OLS) {out['dist_ols'][:, 1:].mean():.5f} "
          f"d(TF,GD1) {out['dist_gd1'][:, 1:].mean():.5f}")
