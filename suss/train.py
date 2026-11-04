"""Shared training loop (AdamW + warm-up + cosine, as pre-registered)."""
from __future__ import annotations

import math
import time

import torch


def make_opt(model, lr=3e-4, wd=0.01):
    fused = next(model.parameters()).is_cuda
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.95), fused=fused)


def lr_at(step, total, base=3e-4, warmup=500, floor=0.1):
    if step < warmup:
        return base * (step + 1) / warmup
    prog = (step - warmup) / max(1, total - warmup)
    return base * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * prog)))


def fit(model, loss_fn, steps, log_every=500, lr=3e-4, log=print):
    """loss_fn(step) -> scalar loss tensor (samples its own batch)."""
    opt = make_opt(model, lr)
    hist = []
    t0 = time.time()
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps, lr)
        loss = loss_fn(step)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % log_every == 0 or step == steps - 1:
            lv = float(loss.detach())
            hist.append((step, lv))
            log(f"step {step:6d} loss {lv:.5f} ({time.time() - t0:.0f}s)", flush=True) if log is print else log(f"step {step:6d} loss {lv:.5f}")
    return hist
