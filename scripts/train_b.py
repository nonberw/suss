"""Experiment B: train the in-context predictor (and the no-context MLP) on a fixed MuJoCo dataset."""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from suss.model import GPT, GPTConfig
from suss.train import fit


def load(name, split, dev="cuda"):
    D = np.load(f"data/{name}_{split}.npz")
    obs, u = D["obs"], D["u"]
    tok = np.concatenate([obs[:, :-1], u[..., None]], -1)
    return D, torch.tensor(tok, dtype=torch.float32, device=dev), torch.tensor(obs[:, 1:], dtype=torch.float32, device=dev)


def norm_stats(name):
    _, tok, nxt = load(name, "train")
    return dict(tm=tok.mean((0, 1)), ts=tok.std((0, 1)), ym=nxt.mean((0, 1)), ys=nxt.std((0, 1)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--mlp", action="store_true", help="train the no-context MLP baseline instead")
    a = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(a.seed)
    _, tok, nxt = load(a.system, "train")
    st = norm_stats(a.system)
    X = (tok - st["tm"]) / st["ts"]
    Y = (nxt - st["ym"]) / st["ys"]
    n = X.shape[0]
    g = torch.Generator(device=dev).manual_seed(a.seed)
    if a.mlp:
        d = X.shape[-1]
        model = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, Y.shape[-1])).to(dev)
    else:
        cfg = GPTConfig(d_in=X.shape[-1], d_out=Y.shape[-1], T=X.shape[1])
        model = GPT(cfg).to(dev)

    def loss_fn(step):
        idx = torch.randint(0, n, (256,), generator=g, device=dev)
        return ((model(X[idx]) - Y[idx]) ** 2).sum(-1).mean()

    hist = fit(model, loss_fn, a.steps)
    os.makedirs("runs/b", exist_ok=True)
    tag = "mlp" if a.mlp else "tf"
    torch.save({"cfg": None if a.mlp else cfg.__dict__, "state": model.state_dict(), "norm": {k: v.cpu() for k, v in st.items()}},
               f"runs/b/{a.system}_{tag}_seed{a.seed}.pt")
    json.dump(hist, open(f"runs/b/{a.system}_{tag}_seed{a.seed}_hist.json", "w"))
