"""Experiment C, EXPLORATORY mechanism of the in-context controller (same recipe as for the predictor).

(1) Functional readout: the controller's feedback Jacobian K^TF_t = -du_t/dx_t (1x2) on its own closed-loop
    trajectories, compared with the oracle finite-horizon LQR gain K*(A,B,t) and with the certainty-equivalent gain
    computed from the ridge estimate of its own history.
(2) Where is B? Ridge probes for the true input matrix B at t in {8, 16, 32, 47}: from the residual stream vs. from the
    per-block key-value fast weights.
Writes results/c/mech.pt."""
import os
import sys

import numpy as np
import torch
from torch.func import jvp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from suss import control as C
from suss.model import GPT, GPTConfig
from suss.probes import RidgeProbe, r2_from_parts, r2_parts
from scripts.mech_a3 import fast_weights

dev = os.environ.get("SUSS_DEV", "cuda")
N_EP = int(os.environ.get("SUSS_NEP", 2048))
PATTERN = os.environ.get("SUSS_CKPT", "runs/c/seed{s}.pt")  # e.g. runs/c/bc_seed{s}.pt for the warm-start ablation
OUT = os.environ.get("SUSS_OUT", "results/c/mech.pt")
gen = torch.Generator(device=dev).manual_seed(10_004)
A, B = C.sample_plants(N_EP, gen, dev)
x0, w = C.sample_noise(N_EP, gen, dev)
Kstar = C.lqr_gains(A, B, C.H)  # (E,H,1,2)


def r2(pred, y):
    pred, y = pred.reshape(len(pred), -1), y.reshape(len(y), -1)
    return float((1 - ((pred - y) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0).clamp(min=1e-12)).mean())


out = {}
for s in [0, 1, 2]:
    path = PATTERN.format(s=s)
    if not os.path.exists(path):
        continue
    ck = torch.load(path, map_location=dev)
    model = GPT(GPTConfig(**ck["cfg"])).to(dev).eval()
    model.load_state_dict(ck["state"])
    for p in model.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        J, X, U = C.simulate(A, B, x0, w, C.tf_policy(model))
    tok = C.tokens_from_traj(X, U)  # (E,H,3): (x_t, u_{t-1})
    f = lambda z: C.U_MAX * torch.tanh(model(z, head_edit=lambda l, o: o))
    ts = [8, 16, 32, 47]
    r = {"t": ts, "gain_r2_oracle": [], "gain_r2_ce": [], "gain_err_oracle": [], "ce_err_oracle": [], "sat_frac": []}
    xs = [X[:, t] for t in range(C.H + 1)]
    us = [U[:, t] for t in range(C.H)]
    for t in ts:
        Kt = torch.zeros(N_EP, 1, 2, device=dev)
        for j in range(2):
            tan = torch.zeros_like(tok); tan[:, t, j] = 1.0
            _, du = jvp(f, (tok,), (tan,))
            Kt[:, 0, j] = -du[:, t, 0]
        th = C.ridge_estimate(xs[:t + 1], us[:t], 0.1)
        Kce = C.lqr_gains(th[:, :, :2], th[:, :, 2:], C.H - t)[:, 0]
        unsat = (U[:, t, 0].abs() < 0.95 * C.U_MAX)  # the Jacobian is only meaningful where tanh is not saturated
        r["sat_frac"].append(float(1 - unsat.float().mean()))
        m = unsat
        r["gain_r2_oracle"].append(r2(Kt[m], Kstar[m, t]))
        r["gain_r2_ce"].append(r2(Kt[m], Kce[m]))
        r["gain_err_oracle"].append(float(((Kt[m] - Kstar[m, t]) ** 2).sum((1, 2)).mean()))
        r["ce_err_oracle"].append(float(((Kce[m] - Kstar[m, t]) ** 2).sum((1, 2)).mean()))
    # where is B?
    half = N_EP // 2
    Yb = B[..., 0].double()
    with torch.no_grad():
        _, res = model(tok, return_resid=True)
    r["probe_B_resid"], r["probe_B_fw"] = {}, {}
    for t in ts:
        best_res = max(r2_from_parts(r2_parts(RidgeProbe().fit(res[l][:half, t], Yb[:half], np.arange(half)).predict(res[l][half:, t]), Yb[half:], np.arange(half, N_EP)))
                       for l in range(1, len(res)))
        best_fw = -1.0
        for l in range(model.cfg.n_layer):
            F = fast_weights(model, tok, l, t)
            pr = RidgeProbe(alphas=[1e-1, 1, 10, 100, 1e3, 1e4]).fit(F[:half], Yb[:half], np.arange(half))
            best_fw = max(best_fw, r2_from_parts(r2_parts(pr.predict(F[half:]), Yb[half:], np.arange(half, N_EP))))
        r["probe_B_resid"][t] = best_res
        r["probe_B_fw"][t] = best_fw
    out[s] = r
    print(f"seed {s}:", {k: (np.round(v, 3).tolist() if isinstance(v, list) else {kk: round(vv, 3) for kk, vv in v.items()}) for k, v in r.items() if k != "t"}, flush=True)
os.makedirs("results/c", exist_ok=True)
torch.save(out, OUT)
