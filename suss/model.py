"""A small GPT-style causal decoder with hooks for probing, steering and patching.

Residual stream index l in {0..L}: l=0 is the embedding output (input to block 1),
l=L is the output of the last block (before the final LayerNorm).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    d_in: int
    d_out: int
    T: int = 48
    d_model: int = 128
    n_layer: int = 4
    n_head: int = 4
    d_mlp: int = 512


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_mlp), nn.GELU(), nn.Linear(cfg.d_mlp, cfg.d_model))

    def attn(self, x, capture=None, head_edit=None, layer=0):
        B, T, D = x.shape
        H = self.cfg.n_head
        dh = D // H
        q, k, v = self.qkv(x).split(D, dim=-1)
        q = q.view(B, T, H, dh).transpose(1, 2)
        k = k.view(B, T, H, dh).transpose(1, 2)
        v = v.view(B, T, H, dh).transpose(1, 2)
        if capture is None and head_edit is None:
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            att = (q @ k.transpose(-1, -2)) / math.sqrt(dh)
            mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
            att = att.masked_fill(~mask, float("-inf")).softmax(-1)
            o = att @ v  # B,H,T,dh
            if capture is not None:
                capture.setdefault("attn", []).append(att.detach())
                capture.setdefault("heads", []).append(o.detach())
            if head_edit is not None:
                o = head_edit(layer, o)
        o = o.transpose(1, 2).reshape(B, T, D)
        return self.proj(o)

    def forward(self, x, capture=None, head_edit=None, layer=0):
        x = x + self.attn(self.ln1(x), capture, head_edit, layer)
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Linear(cfg.d_in, cfg.d_model)
        self.pos = nn.Parameter(torch.zeros(1, cfg.T, cfg.d_model))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.d_out)

    def forward(self, tokens, return_resid=False, edit=None, head_edit=None, capture=None):
        """tokens: (B, T, d_in). edit(l, h) -> h is applied to residual l in {0..L}.
        head_edit(layer, o) -> o edits per-head attention outputs (B,H,T,dh) of block `layer` (0-based).
        """
        T = tokens.shape[1]
        h = self.embed(tokens) + self.pos[:, :T]
        resid = []
        for l, blk in enumerate(self.blocks):
            if edit is not None:
                h = edit(l, h)
            if return_resid:
                resid.append(h)
            h = blk(h, capture, head_edit, l)
        L = len(self.blocks)
        if edit is not None:
            h = edit(L, h)
        if return_resid:
            resid.append(h)
        out = self.head(self.ln_f(h))
        if return_resid:
            return out, resid
        return out

    def step(self, tok_t, t, cache):
        """Incremental forward for one new token (B, d_in) at position t, with a differentiable KV cache
        (list per layer of (k, v) lists). Equivalent to forward(tokens[:, :t+1])[:, -1]."""
        h = self.embed(tok_t) + self.pos[:, t]
        H = self.cfg.n_head
        D = self.cfg.d_model
        dh = D // H
        for l, blk in enumerate(self.blocks):
            x = blk.ln1(h)
            q, k, v = blk.qkv(x).split(D, dim=-1)
            cache[l][0].append(k.view(-1, H, 1, dh))
            cache[l][1].append(v.view(-1, H, 1, dh))
            K = torch.cat(cache[l][0], 2)
            V = torch.cat(cache[l][1], 2)
            att = (q.view(-1, H, 1, dh) @ K.transpose(-1, -2)) / math.sqrt(dh)
            o = (att.softmax(-1) @ V).reshape(-1, D)
            h = h + blk.proj(o)
            h = h + blk.mlp(blk.ln2(h))
        return self.head(self.ln_f(h))

    def new_cache(self):
        return [([], []) for _ in self.blocks]

    def readout(self, h):
        return self.head(self.ln_f(h))

    def forward_from(self, h, l, edit=None):
        """Continue the forward pass from residual h at index l."""
        for j in range(l, len(self.blocks)):
            if edit is not None:
                h = edit(j, h)
            h = self.blocks[j](h)
        if edit is not None:
            h = edit(len(self.blocks), h)
        return self.readout(h)


def n_params(model):
    return sum(p.numel() for p in model.parameters())
