"""Experiment A, EXPLORATORY causal test of the fast-weight hypothesis.

Run episode B, but in chosen blocks replace the keys and values of the CONTEXT positions (s < T-1) by those
computed in the clean run of A' (A's context followed by B's query token). The query's own residual stream,
its queries q and every MLP stay those of B. If the parameter estimate is held in the attention key-value
memory, KV patching of the right blocks alone should transfer A's behaviour (tau -> 1).
Also: patching only the values, or only the keys, and per-head KV patching in the best block."""
import itertools
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from suss.model import GPT, GPTConfig

dev = os.environ.get("SUSS_DEV", "cuda")


def run(model, tok, kv_src=None, blocks=(), mode="kv", heads=None):
    """Manual forward. Returns output and per-block (k, v) (B,H,T,dh). kv_src: dict block -> (k, v) to inject
    at context positions (all but the last) for blocks in `blocks`."""
    cfg = model.cfg
    H, D = cfg.n_head, cfg.d_model
    dh = D // H
    Bn, T, _ = tok.shape
    h = model.embed(tok) + model.pos[:, :T]
    mask = torch.ones(T, T, dtype=torch.bool, device=tok.device).tril()
    kvs = {}
    for l, blk in enumerate(model.blocks):
        x = blk.ln1(h)
        q, k, v = blk.qkv(x).split(D, -1)
        q = q.view(Bn, T, H, dh).transpose(1, 2)
        k = k.view(Bn, T, H, dh).transpose(1, 2)
        v = v.view(Bn, T, H, dh).transpose(1, 2)
        kvs[l] = (k, v)
        if kv_src is not None and l in blocks:
            ks, vs = kv_src[l]
            hs = heads[l] if isinstance(heads, dict) else (list(range(H)) if heads is None else heads)
            k = k.clone(); v = v.clone()
            if "k" in mode:
                k[:, hs, :-1] = ks[:, hs, :-1]
            if "v" in mode:
                v[:, hs, :-1] = vs[:, hs, :-1]
        att = (q @ k.transpose(-1, -2)) / math.sqrt(dh)
        att = att.masked_fill(~mask, float("-inf")).softmax(-1)
        o = (att @ v).transpose(1, 2).reshape(Bn, T, D)
        h = h + blk.proj(o)
        h = h + blk.mlp(blk.ln2(h))
    return model.head(model.ln_f(h)), kvs


if __name__ == "__main__":
    R = torch.load("results/a/references.pt", map_location=dev)
    tok = R["tok"]
    E = tok.shape[0]; half = E // 2
    idxB = torch.arange(half, E, device=dev); idxA = torch.roll(idxB, -1)
    tokB = tok[idxB]
    tokA2 = torch.cat([tok[idxA, :-1], tok[idxB, -1:]], 1)
    out = {}
    for s in [int(x) for x in os.environ.get("SUSS_SEEDS", "0,1,2").split(",")]:
        ck = torch.load(f"runs/a/seed{s}.pt", map_location=dev)
        model = GPT(GPTConfig(**ck["cfg"])).to(dev).eval()
        model.load_state_dict(ck["state"])
        nL, H = model.cfg.n_layer, model.cfg.n_head
        r = {}
        with torch.no_grad():
            yB, _ = run(model, tokB)
            yA, kvA = run(model, tokA2)
            yB, yA = yB[:, -1], yA[:, -1]
            d = yA - yB; den = (d ** 2).sum(-1).clamp(min=1e-12)
            tau = lambda y: float((((y[:, -1] - yB) * d).sum(-1) / den).mean())
            for nb in range(1, nL + 1):
                for bl in itertools.combinations(range(nL), nb):
                    r[("kv",) + bl] = tau(run(model, tokB, kvA, bl, "kv")[0])
            for bl in [(nL - 2,), (nL - 1,), (nL - 2, nL - 1)]:
                r[("k",) + bl] = tau(run(model, tokB, kvA, bl, "k")[0])
                r[("v",) + bl] = tau(run(model, tokB, kvA, bl, "v")[0])
            for l in range(nL):
                for hh in range(H):
                    r[("head", l, hh)] = tau(run(model, tokB, kvA, (l,), "kv", heads=[hh])[0])
        out[s] = r
        best = sorted(((v, k) for k, v in r.items() if k[0] == "kv"), reverse=True)[:6]
        print(f"seed {s}: top KV-patch sets {[(k, round(v, 3)) for v, k in best]}", flush=True)
        print("  single blocks", {k: round(v, 3) for k, v in r.items() if k[0] == "kv" and len(k) == 2},
              "k/v only", {k: round(v, 3) for k, v in r.items() if k[0] in ("k", "v")}, flush=True)
        print("  heads", np.round(np.array([[r[("head", l, hh)] for hh in range(H)] for l in range(nL)]), 2).tolist(), flush=True)
    torch.save(out, os.environ.get("SUSS_OUT", "results/a/mech4.pt"))
