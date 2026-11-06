"""Experiment A, EXPLORATORY controls (added after external review): dimension-matched probes, and
"does the key-value memory hold the estimate or the sufficient statistics?"

At t = 47 on the 4096 test episodes (2048 probe-train / 2048 held-out), for each seed and block l:
  feature sets
    resid_q      residual stream at the query position after block l           (128-d)
    resid_ctx    residual stream at all 48 positions after block l              (6144-d)
    ctx_rp128    random Gaussian projection of resid_ctx                        (128-d)   [control]
    fw           per-head key-value fast weights Cov(v,k) of block l            (4224-d)
    fw_rp128     random Gaussian projection of fw                               (128-d)   [control]
    fw_pca128    top-128 principal components of fw (fit on the train half)     (128-d)   [control]
  targets
    theta_bayes  Bayes estimate at t=47 (6 numbers)
    moments      sufficient statistics (S_47, C_47)/48 (12 numbers)
    theta_true   the true theta (6)
Ridge probes in dual form (n < d), alpha by 5-fold CV. Writes results/a/mech5.pt."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from suss import linear as L
from suss.model import GPT, GPTConfig
from scripts.mech_a3 import fast_weights

dev = os.environ.get("SUSS_DEV", "cpu")
torch.set_num_threads(int(os.environ.get("SUSS_THREADS", "6")))
ALPHAS = [1e-2, 1e-1, 1, 10, 100, 1e3, 1e4]


def dual_ridge_r2(Xtr, Ytr, Xte, Yte, folds=5, seed=0):
    """Standardise X on train, centre Y, choose alpha by K-fold CV, return held-out mean R^2 over targets."""
    Xtr, Ytr, Xte, Yte = [a.double() for a in (Xtr, Ytr, Xte, Yte)]
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
            Ktr = K[tr][:, tr]
            coef = torch.linalg.solve(Ktr + a * torch.eye(int(tr.sum()), dtype=K.dtype), Yc[tr])
            pred = K[te][:, tr] @ coef
            err += float(((pred - Yc[te]) ** 2).sum())
        if err < best_err:
            best, best_err = a, err
    coef = torch.linalg.solve(K + best * torch.eye(n, dtype=K.dtype), Yc)
    pred = (Xt @ Xs.T) @ coef + my
    r2 = 1 - ((pred - Yte) ** 2).sum(0) / ((Yte - Yte.mean(0)) ** 2).sum(0).clamp(min=1e-12)
    return float(r2.mean()), best


def rp(X, d_out, seed):
    g = torch.Generator().manual_seed(seed)
    P = torch.randn(X.shape[1], d_out, generator=g, dtype=X.dtype) / np.sqrt(d_out)
    return X @ P


def pca(Xtr, Xte, d_out):
    mu = Xtr.mean(0)
    U, S, V = torch.linalg.svd(Xtr - mu, full_matrices=False)
    P = V[:d_out].T
    return (Xtr - mu) @ P, (Xte - mu) @ P


if __name__ == "__main__":
    R = torch.load("results/a/references.pt", map_location=dev)
    tok, theta = R["tok"], R["theta"]
    E, T, _ = tok.shape
    half, t = E // 2, T - 1
    iu = torch.triu_indices(L.P, L.P)
    targets = {
        "theta_bayes": R["th_bayes"][:, t].reshape(E, -1),
        "moments": torch.cat([R["S"][:, t][:, iu[0], iu[1]], R["C"][:, t].reshape(E, -1)], 1) / (t + 1),
        "theta_true": theta.reshape(E, -1),
    }
    out = {}
    for s in [int(x) for x in os.environ.get("SUSS_SEEDS", "0,1,2").split(",")]:
        ck = torch.load(f"runs/a/seed{s}.pt", map_location=dev)
        model = GPT(GPTConfig(**ck["cfg"])).to(dev).eval()
        model.load_state_dict(ck["state"])
        with torch.no_grad():
            _, res = model(tok, return_resid=True)
        r = {}
        for l in range(model.cfg.n_layer):
            feats = {}
            h = res[l + 1].float()  # residual after block l
            feats["resid_q"] = h[:, t]
            feats["resid_ctx"] = h.reshape(E, -1)
            feats["ctx_rp128"] = rp(feats["resid_ctx"], 128, 100 + l)
            F = fast_weights(model, tok, l, t).float()
            feats["fw"] = F
            feats["fw_rp128"] = rp(F, 128, 200 + l)
            feats["fw_pca128"] = None  # filled per split below
            for fname, X in feats.items():
                if fname == "fw_pca128":
                    Xtr, Xte = pca(F[:half].double(), F[half:].double(), 128)
                else:
                    Xtr, Xte = X[:half], X[half:]
                for tname, Y in targets.items():
                    r2, a = dual_ridge_r2(Xtr, Y[:half], Xte, Y[half:])
                    r[(fname, l, tname)] = r2
                    r[("dim", fname, l)] = int(Xtr.shape[1])
            print(f"seed {s} block {l}: " + ", ".join(f"{fn}/{tn}={r[(fn, l, tn)]:.3f}" for fn in feats for tn in targets), flush=True)
        out[s] = r
    os.makedirs("results/a", exist_ok=True)
    torch.save(out, os.environ.get("SUSS_OUT", "results/a/mech5.pt"))
