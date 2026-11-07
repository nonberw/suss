"""Pre-registration 2, part A (PREREGISTRATION_2.md, tag prereg2-94b6cf8): the key-value account on models it was
not derived from. KV-only patching of the late blocks and of block 1, and dimension-matched probes of the Bayes
estimate from the fast weights vs the residual stream, on long60k, deep8 and wide256 (small64: descriptive).
Same data and procedure as scripts/mech_a4.py and scripts/mech_a5.py. Saves per-pair transfers and held-out
probe predictions for the bootstrap in scripts/confirm_stats.py."""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from suss.model import GPT, GPTConfig
from scripts.mech_a3 import fast_weights
from scripts.mech_a4 import run

dev = os.environ.get("SUSS_DEV", "cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(int(os.environ.get("SUSS_THREADS", "6")))
MODELS = ["long60k", "deep8", "wide256", "small64"]
ALPHAS = [1e-2, 1e-1, 1, 10, 100, 1e3, 1e4]
OUT = "results/confirm"


def dual_ridge(Xtr, Ytr, Xte, folds=5, seed=0):
    """As mech_a5.dual_ridge_r2 (standardise on train, centre Y, alpha by K-fold CV), returning test predictions."""
    Xtr, Ytr, Xte = [a.double().cpu() for a in (Xtr, Ytr, Xte)]
    mx, sx = Xtr.mean(0), Xtr.std(0) + 1e-6
    my = Ytr.mean(0)
    Xs, Xt = (Xtr - mx) / sx, (Xte - mx) / sx
    Yc = Ytr - my
    n = Xs.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    fold = torch.zeros(n, dtype=torch.long)
    fold[perm] = torch.arange(n) % folds
    K = Xs @ Xs.T
    best, best_err = None, float("inf")
    for a in ALPHAS:
        err = 0.0
        for k in range(folds):
            tr, te = fold != k, fold == k
            coef = torch.linalg.solve(K[tr][:, tr] + a * torch.eye(int(tr.sum()), dtype=K.dtype), Yc[tr])
            err += float(((K[te][:, tr] @ coef - Yc[te]) ** 2).sum())
        if err < best_err:
            best, best_err = a, err
    coef = torch.linalg.solve(K + best * torch.eye(n, dtype=K.dtype), Yc)
    return ((Xt @ Xs.T) @ coef + my).numpy(), best


def r2(pred, y):
    return float(np.mean(1 - ((pred - y) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)))


def project(X, d_out, seed):
    g = torch.Generator().manual_seed(seed)
    P = torch.randn(X.shape[1], d_out, generator=g, dtype=torch.float64) / np.sqrt(d_out)
    return X.double().cpu() @ P


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    R = torch.load("results/a/references.pt", map_location=dev)
    tok = R["tok"]
    E, T, _ = tok.shape
    half, t = E // 2, T - 1
    y = R["th_bayes"][:, t].reshape(E, -1).double().cpu()
    ytr, yte = y[:half], y[half:].numpy()
    idxB = torch.arange(half, E, device=dev)
    idxA = torch.roll(idxB, -1)
    tokB = tok[idxB]
    tokA2 = torch.cat([tok[idxA, :-1], tok[idxB, -1:]], 1)
    summary = {}
    for name in MODELS:
        ck = torch.load(f"runs/scale/{name}.pt", map_location=dev)
        model = GPT(GPTConfig(**ck["cfg"])).to(dev).eval()
        model.load_state_dict(ck["state"])
        nL, D = model.cfg.n_layer, model.cfg.d_model
        late = tuple(range(nL // 2, nL))
        with torch.no_grad():
            yB, _ = run(model, tokB)
            yA, kvA = run(model, tokA2)
            yB, yA = yB[:, -1], yA[:, -1]
            dd = yA - yB
            den = (dd ** 2).sum(-1).clamp(min=1e-12)
            tau = lambda blocks: (((run(model, tokB, kvA, blocks, "kv")[0][:, -1] - yB) * dd).sum(-1) / den).cpu().numpy()
            tau_late, tau_b1 = tau(late), tau((0,))
            tau_single = {l: float(tau((l,)).mean()) for l in range(nL)}
            _, res = model(tok, return_resid=True)
        readouts = {}
        for l in range(nL):
            h = res[l + 1].double().cpu()
            readouts[("resid_query", l)] = h[:, t]
            readouts[("resid_mean", l)] = h.mean(1)
        for l in late:
            F = fast_weights(model, tok, l, t)
            readouts[("kv_rp", l)] = project(F, D, 200 + l)
        r2s, preds = {}, {}
        for key, X in readouts.items():
            p, a = dual_ridge(X[:half], ytr, X[half:])
            r2s[key] = r2(p, yte)
            preds[key] = p
        best_res = max((k for k in r2s if k[0] != "kv_rp"), key=lambda k: r2s[k])
        best_kv = max((k for k in r2s if k[0] == "kv_rp"), key=lambda k: r2s[k])
        np.savez(f"{OUT}/a_{name}.npz", tau_late=tau_late, tau_b1=tau_b1, y_test=yte,
                 pred_kv=preds[best_kv], pred_res=preds[best_res])
        summary[name] = {
            "n_layer": nL, "d_model": D, "late_blocks": [l + 1 for l in late],
            "tau_late": float(tau_late.mean()), "tau_block1": float(tau_b1.mean()),
            "tau_single_block": {str(l + 1): v for l, v in tau_single.items()},
            "r2": {f"{k[0]}|block{k[1] + 1}": v for k, v in r2s.items()},
            "best_kv": f"{best_kv[0]}|block{best_kv[1] + 1}", "best_resid": f"{best_res[0]}|block{best_res[1] + 1}",
            "r2_best_kv": r2s[best_kv], "r2_best_resid": r2s[best_res], "gap": r2s[best_kv] - r2s[best_res],
        }
        s = summary[name]
        print(f"{name}: tau_late={s['tau_late']:.3f} tau_b1={s['tau_block1']:.3f} kv={s['r2_best_kv']:.3f} "
              f"({s['best_kv']}) resid={s['r2_best_resid']:.3f} ({s['best_resid']}) gap={s['gap']:.3f}", flush=True)
        json.dump(summary, open(f"{OUT}/a_summary.json", "w"), indent=1)  # after every model, so a crash keeps the rest
        del model, res, readouts, kvA
