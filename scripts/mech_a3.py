"""Experiment A, EXPLORATORY: where does the parameter estimate live, if not in the residual stream?

Hypothesis (post hoc, motivated by the failed pre-registered probes): the model implements a function of the
query that is (approximately) linear, ŷ_t ≈ J_t z_t, and the *Jacobian* J_t = dŷ_t/dz_t is its implicit
parameter estimate. Linear-attention algebra suggests J_t is assembled from the heads' key-value "fast weights"
F = mean_s v_s k_s^T, i.e. stored across context positions rather than in any single residual vector.

Outputs results/a/mech3.pt:
 - jac: R^2 of J_t against theta_Bayes / theta_true / theta_OLS (pooled t=4..47), and by t
 - lin: linearity of the prediction in the query (counterfactual queries)
 - fw:  probe R^2 of theta_Bayes(t=47) from per-layer fast weights vs from the residual at t=47
"""
import os
import sys

import numpy as np
import torch
from torch.func import jvp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from suss import linear as L
from suss.model import GPT, GPTConfig
from suss.probes import RidgeProbe, r2_from_parts, r2_parts

dev = os.environ.get("SUSS_DEV", "cuda")


def query_jacobian(model, tok, t0=0):
    """J[b,t] = d yhat[b,t] / d tok[b,t]  (n_out x d_in), exact via forward-mode AD, one position at a time."""
    E, T, d = tok.shape
    n_out = model.cfg.d_out
    J = torch.zeros(E, T, n_out, d, device=tok.device)
    f = lambda x: model(x, head_edit=lambda l, o: o)  # manual attention path (forward-mode AD friendly)
    for t in range(t0, T):
        for j in range(d):
            tan = torch.zeros_like(tok)
            tan[:, t, j] = 1.0
            _, dy = jvp(f, (tok,), (tan,))
            J[:, t, :, j] = dy[:, t]
    return J


def r2(pred, y):
    pred, y = pred.reshape(-1, pred.shape[-1]), y.reshape(-1, y.shape[-1])
    return float((1 - ((pred - y) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)).mean())


def fast_weights(model, tok, l, t):
    """Per-head mean_s v_s k_s^T over s<=t at block l, flattened: (E, H*dh*dh), plus mean value (E, H*dh)."""
    with torch.no_grad():
        _, res = model(tok[:, :t + 1], return_resid=True)
        blk = model.blocks[l]
        x = blk.ln1(res[l])
        D = model.cfg.d_model; H = model.cfg.n_head; dh = D // H
        q, k, v = blk.qkv(x).split(D, -1)
        k = k.view(*k.shape[:2], H, dh); v = v.view(*v.shape[:2], H, dh)
        F = torch.einsum("bshi,bshj->bhij", v, k) / (t + 1)
        return torch.cat([F.flatten(1), v.mean(1).flatten(1)], 1)


if __name__ == "__main__":
    R = torch.load("results/a/references.pt", map_location=dev)
    tok, theta = R["tok"], R["theta"]
    E, T, _ = tok.shape
    half, t0 = E // 2, 4
    thb = R["th_bayes"]
    tho = L.ols_theta(R["S"], R["C"]).clamp(-5, 5)
    ev = slice(half, E)
    out = {}
    for s in [0, 1, 2]:
        ck = torch.load(f"runs/a/seed{s}.pt", map_location=dev)
        model = GPT(GPTConfig(**ck["cfg"])).to(dev).eval()
        model.load_state_dict(ck["state"])
        for p in model.parameters():
            p.requires_grad_(False)
        r = {}
        J = query_jacobian(model, tok[ev], t0)  # (E/2, T, 2, 3)
        Jp, Bp = J[:, t0:].flatten(2), thb[ev, t0:].flatten(2)
        r["jac_r2_bayes"] = r2(Jp, Bp)
        r["jac_r2_true"] = r2(Jp, theta[ev].flatten(1).unsqueeze(1).expand_as(Jp))
        r["jac_r2_ols"] = r2(Jp, tho[ev, t0:].flatten(2))
        r["jac_r2_bayes_by_t"] = [r2(J[:, t].flatten(1), thb[ev, t].flatten(1)) for t in range(T)]
        r["jac_err_bayes_by_t"] = [float(((J[:, t] - thb[ev, t]) ** 2).sum((1, 2)).mean()) for t in range(T)]
        r["bayes_err_true_by_t"] = [float(((thb[ev, t] - theta[ev]) ** 2).sum((1, 2)).mean()) for t in range(T)]
        r["jac_err_true_by_t"] = [float(((J[:, t] - theta[ev]) ** 2).sum((1, 2)).mean()) for t in range(T)]
        # linearity in the query: replace the last token by random counterfactual queries
        with torch.no_grad():
            g = torch.Generator(device=dev).manual_seed(s)
            base = tok[ev].clone()
            y0 = model(base)[:, -1]
            obs, lin, bay = [], [], []
            for k in range(8):
                dz = torch.randn(base.shape[0], 3, generator=g, device=dev)
                tk = base.clone(); tk[:, -1] = tk[:, -1] + dz
                obs.append(model(tk)[:, -1] - y0)
                lin.append(torch.einsum("bij,bj->bi", J[:, -1], dz))
                bay.append(torch.einsum("bij,bj->bi", thb[ev, -1], dz))
            o, li, ba = torch.cat(obs), torch.cat(lin), torch.cat(bay)
            r["cf_r2_linear_jac"] = r2(li, o)  # how linear is the model in its query
            r["cf_r2_bayes"] = r2(ba, o)  # counterfactual response matches the Bayes linear map
            c = np.cov(ba.ravel().cpu().numpy(), o.ravel().cpu().numpy())
            r["cf_slope_bayes"] = float(c[0, 1] / c[0, 0])
        # fast-weight probes at t = 47 (pre-registered-style ridge probe, 50/50 episode split of the eval half)
        hh = (E - half) // 2
        grp_tr, grp_te = np.arange(hh), np.arange(hh, E - half)
        Y = thb[ev, -1].flatten(1)
        r["fw_r2"], r["resid_r2"] = [], []
        for l in range(model.cfg.n_layer):
            X = fast_weights(model, tok[ev], l, T - 1)
            pr = RidgeProbe(alphas=[1e-1, 1, 10, 100, 1e3, 1e4]).fit(X[:hh], Y[:hh], grp_tr)
            r["fw_r2"].append(r2_from_parts(r2_parts(pr.predict(X[hh:]), Y[hh:], grp_te)))
            with torch.no_grad():
                _, res = model(tok[ev], return_resid=True)
            Xr = res[l][:, -1]
            pr = RidgeProbe().fit(Xr[:hh], Y[:hh], grp_tr)
            r["resid_r2"].append(r2_from_parts(r2_parts(pr.predict(Xr[hh:]), Y[hh:], grp_te)))
        out[s] = r
        print(f"seed {s}:", {k: (np.round(v, 3).tolist() if isinstance(v, list) and len(v) < 10 else (round(v, 4) if isinstance(v, float) else '...')) for k, v in r.items()}, flush=True)
    torch.save(out, "results/a/mech3.pt")
