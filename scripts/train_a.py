"""Experiment A: train the in-context predictor on random linear systems (fresh batches)."""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from suss import linear as L
from suss.model import GPT, GPTConfig, n_params
from suss.train import fit

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--steps", type=int, default=20000)
ap.add_argument("--out", default="runs/a")
a = ap.parse_args()

dev = "cuda"
torch.manual_seed(a.seed)
gen = torch.Generator(device=dev).manual_seed(a.seed)
cfg = GPTConfig(d_in=L.P, d_out=L.N, T=48)
model = GPT(cfg).to(dev)
print("params", n_params(model))


BUF = 64  # generate 64 batches at once (same distribution, fewer kernel launches)
buf = {}


def loss_fn(step):
    if step % BUF == 0:
        _, buf["tok"], buf["nxt"] = L.make_batch(256 * BUF, cfg.T, gen, dev)
    i = (step % BUF) * 256
    tok, nxt = buf["tok"][i:i + 256], buf["nxt"][i:i + 256]
    return ((model(tok) - nxt) ** 2).sum(-1).mean()


hist = fit(model, loss_fn, a.steps)
os.makedirs(a.out, exist_ok=True)
torch.save({"cfg": cfg.__dict__, "state": model.state_dict()}, f"{a.out}/seed{a.seed}.pt")
json.dump(hist, open(f"{a.out}/seed{a.seed}_hist.json", "w"))
