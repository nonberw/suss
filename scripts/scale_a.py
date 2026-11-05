"""EXPLORATORY scale study for Experiment A (added after the pre-registered analysis, see DEVIATIONS.md):
does the approximation floor shrink with longer training / bigger models, and do the findings hold for a
larger system (n=4, m=2: 24 unknown parameters)?

Reference: ridge = exact Bayes posterior mean under the untruncated Gaussian prior (the truncation to stable A
only matters at very short contexts, see Experiment A). Also reports the query-Jacobian vs ridge estimate."""
import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.func import jvp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from suss.model import GPT, GPTConfig, n_params
from suss.train import fit

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=2)
ap.add_argument("--m", type=int, default=1)
ap.add_argument("--d_model", type=int, default=128)
ap.add_argument("--n_layer", type=int, default=4)
ap.add_argument("--steps", type=int, default=20000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--tag", required=True)
a = ap.parse_args()
dev = "cuda"
T, SIG = 48, 0.1
n, m = a.n, a.m
p = n + m
sdA = 0.4 if n == 2 else 0.8 / np.sqrt(n)
sd = torch.tensor([sdA] * n + [0.5] * m, device=dev)


def sample_theta(k, gen):
    out, got = [], 0
    while got < k:
        th = torch.randn(2 * k, n, p, generator=gen, device=dev) * sd
        ok = torch.linalg.eigvals(th[:, :, :n]).abs().amax(-1) < 0.95
        out.append(th[ok]); got += int(ok.sum())
    return torch.cat(out)[:k]


def batch(k, gen):
    th = sample_theta(k, gen)
    x = torch.randn(k, n, generator=gen, device=dev)
    u = torch.randn(k, T, m, generator=gen, device=dev)
    w = torch.randn(k, T, n, generator=gen, device=dev) * SIG
    xs, nx = [], []
    for t in range(T):
        z = torch.cat([x, u[:, t]], -1)
        xn = torch.einsum("bij,bj->bi", th, z) + w[:, t]
        xs.append(x); nx.append(xn); x = xn
    return th, torch.cat([torch.stack(xs, 1), u], -1), torch.stack(nx, 1)


torch.manual_seed(a.seed)
gen = torch.Generator(device=dev).manual_seed(a.seed)
cfg = GPTConfig(d_in=p, d_out=n, T=T, d_model=a.d_model, n_layer=a.n_layer, n_head=4, d_mlp=4 * a.d_model)
model = GPT(cfg).to(dev)
buf = {}


def loss_fn(step):
    if step % 64 == 0:
        _, buf["tok"], buf["nxt"] = batch(256 * 64, gen)
    i = (step % 64) * 256
    return ((model(buf["tok"][i:i + 256]) - buf["nxt"][i:i + 256]) ** 2).sum(-1).mean()


hist = fit(model, loss_fn, a.steps)
os.makedirs("runs/scale", exist_ok=True)
torch.save({"cfg": cfg.__dict__, "state": model.state_dict(), "args": vars(a)}, f"runs/scale/{a.tag}.pt")

# ---- evaluation on 4096 fixed test episodes
model.eval()
tg = torch.Generator(device=dev).manual_seed(10_000 + 7)
th, tok, nxt = batch(4096, tg)
zz = tok.unsqueeze(-1) * tok.unsqueeze(-2); zx = tok.unsqueeze(-1) * nxt.unsqueeze(-2)
S = torch.cumsum(zz, 1) - zz; C = torch.cumsum(zx, 1) - zx
lam = (SIG / sd) ** 2
thr = torch.linalg.solve(S + torch.diag_embed(lam.expand(S.shape[:-1])), C).transpose(-1, -2)
mean = torch.einsum("bij,btj->bti", th, tok)
yr = torch.einsum("btij,btj->bti", thr, tok)
with torch.no_grad():
    yt = torch.cat([model(tok[i:i + 512]) for i in range(0, 4096, 512)])
ex_tf = ((yt - mean) ** 2).sum(-1).mean(0).cpu().numpy()
ex_r = ((yr - mean) ** 2).sum(-1).mean(0).cpu().numpy()
# query Jacobian at selected positions on 1024 episodes
for q in model.parameters():
    q.requires_grad_(False)
f = lambda x: model(x, head_edit=lambda l, o: o)
jr2 = {}
sub = tok[:1024]
for t in [4, 8, 16, 32, 47]:
    J = torch.zeros(1024, n, p, device=dev)
    for j in range(p):
        tan = torch.zeros_like(sub); tan[:, t, j] = 1
        _, dy = jvp(f, (sub,), (tan,))
        J[:, :, j] = dy[:, t]
    Y = thr[:1024, t].flatten(1); P = J.flatten(1)
    jr2[t] = float((1 - ((P - Y) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)).mean())
res = dict(args=vars(a), params=n_params(model), excess_tf=ex_tf.tolist(), excess_ridge=ex_r.tolist(),
           ratio_mean_t4=float((ex_tf[4:] / ex_r[4:]).mean()), ratio_t47=float(ex_tf[47] / ex_r[47]),
           npsigma2=n * p * SIG ** 2, t_excess_ridge_47=float(47 * ex_r[47]), jac_r2=jr2, hist=hist)
os.makedirs("results/scale", exist_ok=True)
json.dump(res, open(f"results/scale/{a.tag}.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k not in ("excess_tf", "excess_ridge", "hist")}))
