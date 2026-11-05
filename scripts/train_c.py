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
ap.add_argument("--loss", default="norm", choices=["raw", "norm", "lognorm"],
                help="raw: mean J/H (pre-registered, unstable); norm: mean J/J_oracle (= evaluation metric); lognorm: mean log(J/J_oracle)")
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--tbptt", type=int, default=0, help="if >0, stop state gradients every k steps (truncated BPTT)")
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--squash", type=int, default=0, help="1: soft log-compression of the controller's input tokens")
ap.add_argument("--init", default="", help="checkpoint to start from (stage-2 fine-tuning after the DAgger warm start)")
ap.add_argument("--tag", default="", help="checkpoint name suffix")
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
if a.init:
    model.load_state_dict(torch.load(a.init, map_location=dev)["state"])


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
    NB_ = a.batch
    sA, sB = C.sample_plants(NB_, gen, dev)
    sx0, sw = C.sample_noise(NB_, gen, dev)
    sJo = torch.ones(NB_, device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01, betas=(0.9, 0.95))

    def oracle_cost(A, B, x0, w):
        with torch.no_grad():
            return C.simulate(A, B, x0, w, C.oracle_policy(A, B))[0]

    def fwd_bwd():
        J, _, _ = C.simulate(sA, sB, sx0, sw, C.tf_policy(model, bool(a.squash)), tbptt=a.tbptt)
        if a.loss == "raw":
            loss = (J / C.H).mean()
        elif a.loss == "norm":
            loss = (J / sJo).mean()
        else:
            loss = torch.log(J / sJo).mean()
        loss.backward()
        return loss

    # held-out validation plants for monitoring only (normalised cost, never used for training)
    vgen = torch.Generator(device=dev).manual_seed(777)
    vA, vB = C.sample_plants(1024, vgen, dev)
    vx0, vw = C.sample_noise(1024, vgen, dev)
    vJo = oracle_cost(vA, vB, vx0, vw)

    def validate():
        with torch.no_grad():
            J, _, _ = C.simulate(vA, vB, vx0, vw, C.tf_policy(model, bool(a.squash)))
            r = J / vJo
        return float(r.mean()), float(r.median()), float((r > 10).float().mean())

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
        A, B = C.sample_plants(NB_, gen, dev)
        x0, w = C.sample_noise(NB_, gen, dev)
        sA.copy_(A); sB.copy_(B); sx0.copy_(x0); sw.copy_(w)
        if a.loss != "raw":
            sJo.copy_(oracle_cost(A, B, x0, w))
        graph.replay()
        for gp in opt.param_groups:
            gp["lr"] = lr_at(step, a.steps)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 250 == 0 or step == a.steps - 1:
            lv = float(static_loss)
            val = validate() if (step % 1000 == 0 or step == a.steps - 1) else None
            hist.append((step, lv, val))
            print(f"step {step:6d} loss {lv:.5f} val(mean,median,fail) {val} ({time.time() - t0:.0f}s)", flush=True)
else:
    hist = fit(model, loss_fn, a.steps, log_every=250)
torch.save({"cfg": cfg.__dict__, "state": model.state_dict(), "args": vars(a)}, f"{a.out}/seed{a.seed}{a.tag}.pt")
json.dump(hist, open(f"{a.out}/seed{a.seed}{a.tag}_hist.json", "w"))
