"""Experiment B, EXPLORATORY: are the physical parameters stored in the attention key-value memory
("fast weights") rather than in the residual stream? Probe each parameter at t=47 from (i) the residual
stream and (ii) the per-block fast weights mean_s v_s k_s^T, with the same ridge probe and episode split."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from suss.model import GPT, GPTConfig
from suss.probes import RidgeProbe, r2_from_parts, r2_parts
from scripts.mech_a3 import fast_weights

dev = os.environ.get("SUSS_DEV", "cuda")
out = {}
for name in ["pendulum", "block", "bouncer"]:
    if not os.path.exists(f"runs/b/{name}_tf_seed2.pt"):
        continue
    D = np.load(f"data/{name}_test.npz")
    obs, u, prm = D["obs"], D["u"], D["params"]
    tok = torch.tensor(np.concatenate([obs[:, :-1], u[..., None]], -1), dtype=torch.float32, device=dev)
    if name == "pendulum":
        targets = {"g": prm[:, 2], "inv_m": 1 / prm[:, 0], "c_over_m": prm[:, 1] / prm[:, 0]}
    elif name == "block":
        targets = {"mu": prm[:, 1], "inv_m": 1 / prm[:, 0]}
    else:
        targets = {"g": prm[:, 1], "inv_m": 1 / prm[:, 0], "zeta": prm[:, 2]}
    E = tok.shape[0]; half = E // 2
    for s in [0, 1, 2]:
        ck = torch.load(f"runs/b/{name}_tf_seed{s}.pt", map_location=dev)
        model = GPT(GPTConfig(**ck["cfg"])).to(dev).eval()
        model.load_state_dict(ck["state"])
        st = {k: v.to(dev) for k, v in ck["norm"].items()}
        X = (tok - st["tm"]) / st["ts"]
        with torch.no_grad():
            _, res = model(X, return_resid=True)
        T = X.shape[1]
        for l in range(model.cfg.n_layer):
            F = fast_weights(model, X, l, T - 1)
            for tn, yv in targets.items():
                Y = torch.tensor(yv, dtype=torch.float64, device=dev).view(-1, 1)
                pr = RidgeProbe(alphas=[1e-1, 1, 10, 100, 1e3, 1e4]).fit(F[:half], Y[:half], np.arange(half))
                fw = r2_from_parts(r2_parts(pr.predict(F[half:]), Y[half:], np.arange(half, E)))
                h = res[l + 1][:, -1]  # residual after block l (same depth as the block's output)
                pr = RidgeProbe().fit(h[:half], Y[:half], np.arange(half))
                rs = r2_from_parts(r2_parts(pr.predict(h[half:]), Y[half:], np.arange(half, E)))
                out[(name, s, l, tn)] = (fw, rs)
        print(name, s, {f"{tn}@{l}": tuple(round(x, 3) for x in out[(name, s, l, tn)]) for l in range(model.cfg.n_layer) for tn in targets}, flush=True)
torch.save(out, "results/b/mech2.pt")
