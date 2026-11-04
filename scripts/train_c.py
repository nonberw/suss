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
ap.add_argument("--device", default="cuda")
a = ap.parse_args()
dev = a.device
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


if dev == "cuda":
    # Same algorithm, but the 48-step closed-loop rollout + backward is captured once as a CUDA graph
    # (static shapes) and replayed; plants/noise are sampled outside the graph into static buffers.
    import math
    import time
    from suss.train import lr_at
    sA, sB = C.sample_plants(256, gen, dev)
    sx0, sw = C.sample_noise(256, gen, dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01, betas=(0.9, 0.95))

    def fwd_bwd():
        J, _, _ = C.simulate(sA, sB, sx0, sw, C.tf_policy(model))
        loss = (J / C.H).mean()
        loss.backward()
        return loss

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):  # warm-up (grads discarded)
            opt.zero_grad(set_to_none=True)
            fwd_bwd()
    torch.cuda.current_stream().wait_stream(s)
    opt.zero_grad(set_to_none=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_loss = fwd_bwd()
    hist, t0 = [], time.time()
    for step in range(a.steps):
        A, B = C.sample_plants(256, gen, dev)
        x0, w = C.sample_noise(256, gen, dev)
        sA.copy_(A); sB.copy_(B); sx0.copy_(x0); sw.copy_(w)
        graph.replay()
        for gp in opt.param_groups:
            gp["lr"] = lr_at(step, a.steps)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 250 == 0 or step == a.steps - 1:
            lv = float(static_loss)
            hist.append((step, lv))
            print(f"step {step:6d} loss {lv:.5f} ({time.time() - t0:.0f}s)", flush=True)
else:
    hist = fit(model, loss_fn, a.steps, log_every=250)
torch.save({"cfg": cfg.__dict__, "state": model.state_dict()}, f"{a.out}/seed{a.seed}.pt")
json.dump(hist, open(f"{a.out}/seed{a.seed}_hist.json", "w"))
