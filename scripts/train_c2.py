"""Experiment C1, stage 1 (deviation, see DEVIATIONS.md): DAgger warm start for the transformer controller.

Pure closed-loop BPTT on the cost was unstable in pilots (exploding gradients through open-loop-unstable plants).
Stage 1 trains the controller to imitate the oracle LQR action u*_t = -K_t(A,B) x_t (which the controller cannot see:
it must infer (A,B) from its own history), on states visited by its own noisy policy (DAgger with beta-mixing).
Training is teacher-forced over the realised history, so one parallel forward/backward per batch.
Stage 2 (train_c.py --init) then fine-tunes on the closed-loop cost by BPTT."""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from suss import control as C
from suss.model import GPT, GPTConfig
from suss.train import lr_at

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--steps", type=int, default=4000)
ap.add_argument("--noise", type=float, default=0.3)
ap.add_argument("--out", default="runs/c")
a = ap.parse_args()
dev = "cuda"
torch.manual_seed(a.seed)
gen = torch.Generator(device=dev).manual_seed(1000 + a.seed)
cfg = GPTConfig(d_in=C.N + C.M, d_out=C.M, T=C.H)
model = GPT(cfg).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01, betas=(0.9, 0.95), fused=True)


def rollout(beta, n=256):
    """Executed action: oracle w.p. beta else the model's action, plus Gaussian exploration noise.
    Returns tokens (x_t, u_{t-1}) and oracle labels u*_t."""
    A, B = C.sample_plants(n, gen, dev)
    x0, w = C.sample_noise(n, gen, dev)
    K = C.lqr_gains(A, B, C.H)
    pol = C.tf_policy(model)
    x = x0; xs, us, lab = [x], [], []
    with torch.no_grad():
        for t in range(C.H):
            u_star = -(K[:, t] @ x.unsqueeze(-1)).squeeze(-1).clamp(-C.U_MAX, C.U_MAX)
            u_tf = pol(t, xs, us)
            mix = (torch.rand(n, 1, generator=gen, device=dev) < beta).float()
            u = (mix * u_star + (1 - mix) * u_tf + a.noise * torch.randn(n, 1, generator=gen, device=dev)).clamp(-C.U_MAX, C.U_MAX)
            lab.append(u_star)
            x = (torch.einsum("bij,bj->bi", A, x) + torch.einsum("bij,bj->bi", B, u) + w[:, t]).clamp(-C.X_CLIP, C.X_CLIP)
            xs.append(x); us.append(u)
    X, U = torch.stack(xs, 1), torch.stack(us, 1)
    return C.tokens_from_traj(X, U), torch.stack(lab, 1)


hist, t0 = [], time.time()
for step in range(a.steps):
    for g in opt.param_groups:
        g["lr"] = lr_at(step, a.steps)
    beta = max(0.0, 1.0 - step / (0.5 * a.steps))
    tok, lab = rollout(beta)
    pred = C.U_MAX * torch.tanh(model(tok))
    loss = ((pred - lab) ** 2).mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if step % 250 == 0 or step == a.steps - 1:
        hist.append((step, float(loss)))
        print(f"step {step:6d} bc-loss {float(loss):.5f} beta {beta:.2f} ({time.time() - t0:.0f}s)", flush=True)
os.makedirs(a.out, exist_ok=True)
torch.save({"cfg": cfg.__dict__, "state": model.state_dict(), "args": vars(a)}, f"{a.out}/bc_seed{a.seed}.pt")
json.dump(hist, open(f"{a.out}/bc_seed{a.seed}_hist.json", "w"))
