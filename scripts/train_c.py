"""Experiment C1: train the transformer controller by back-propagation through the plant,
and the best constant-gain (fixed) controller."""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from suss import control as C
from suss.model import GPT, GPTConfig
from suss.train import fit

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--steps", type=int, default=6000)
ap.add_argument("--fixed", action="store_true")
ap.add_argument("--out", default="runs/c")
a = ap.parse_args()
dev = "cuda"
torch.manual_seed(a.seed)
gen = torch.Generator(device=dev).manual_seed(a.seed)
os.makedirs(a.out, exist_ok=True)

if a.fixed:
    Kc = torch.zeros(1, 2, device=dev, requires_grad=True)
    opt = torch.optim.Adam([Kc], lr=0.02)
    for step in range(1000):
        A, B = C.sample_plants(1024, gen, dev)
        x0, w = C.sample_noise(1024, gen, dev)
        J, _, _ = C.simulate(A, B, x0, w, C.fixed_policy(Kc))
        loss = J.clamp(max=1e4).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 200 == 0:
            print(step, float(loss), Kc.detach().cpu().numpy())
    torch.save({"K": Kc.detach().cpu()}, f"{a.out}/fixed_seed{a.seed}.pt")
    sys.exit()

cfg = GPTConfig(d_in=C.N + C.M, d_out=C.M, T=C.H)
model = GPT(cfg).to(dev)


def loss_fn(step):
    A, B = C.sample_plants(256, gen, dev)
    x0, w = C.sample_noise(256, gen, dev)
    J, _, _ = C.simulate(A, B, x0, w, C.tf_policy(model))
    return (J / C.H).mean()


hist = fit(model, loss_fn, a.steps, log_every=250)
torch.save({"cfg": cfg.__dict__, "state": model.state_dict()}, f"{a.out}/seed{a.seed}.pt")
json.dump(hist, open(f"{a.out}/seed{a.seed}_hist.json", "w"))
