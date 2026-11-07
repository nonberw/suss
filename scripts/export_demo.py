"""Export the experiment-A model (seed 0) for the in-browser demo (docs/demo/): raw float32 weights plus a
manifest, read by docs/demo/model.js, and the reference curves the demo draws (average over 4096 fresh episodes).
tests/test_demo.py checks that the JavaScript forward pass matches PyTorch."""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from suss import linear as L
from suss.model import GPT, GPTConfig

ck = torch.load("runs/a/seed0.pt", map_location="cpu")
model = GPT(GPTConfig(**ck["cfg"])).eval()
model.load_state_dict(ck["state"])
os.makedirs("docs/demo", exist_ok=True)

tensors, offset, chunks = [], 0, []
for name, p in model.state_dict().items():
    a = p.detach().float().contiguous().numpy().astype("<f4")
    tensors.append({"name": name, "shape": list(a.shape), "offset": offset})
    offset += a.size
    chunks.append(a.ravel())
np.concatenate(chunks).tofile("docs/demo/weights.bin")
manifest = {"cfg": ck["cfg"], "dtype": "float32", "tensors": tensors,
            "source": "runs/a/seed0.pt (experiment A, seed 0)", "n_params": int(offset)}
json.dump(manifest, open("docs/demo/weights.json", "w"), indent=1)
print(f"weights.bin: {offset} floats, {os.path.getsize('docs/demo/weights.bin') / 1e6:.2f} MB")

gen = torch.Generator().manual_seed(7)
E, T = 4096, 48
theta, tok, nxt = L.make_batch(E, T, gen)
S, C = L.suff_stats(tok, nxt)
mean = L.predict(theta.unsqueeze(1).expand(-1, T, -1, -1), tok)
with torch.no_grad():
    y = model(tok)
th_r = L.ridge_theta(S, C)
r4 = lambda v: [round(float(x), 6) for x in v]
ref = {
    "episodes": E,
    "excess_tf": r4(((y - mean) ** 2).sum(-1).mean(0)),
    "excess_bayes": r4(((L.predict(th_r, tok) - mean) ** 2).sum(-1).mean(0)),
    "param_err_bayes": r4(((th_r - theta.unsqueeze(1)) ** 2).sum((-1, -2)).sqrt().mean(0)),
    "prior": {"sigma": L.SIGMA, "sd_a": L.SD_A, "sd_b": L.SD_B, "rho_max": L.RHO_MAX},
}
json.dump(ref, open("docs/demo/reference.json", "w"), separators=(",", ":"))
print("reference curves written; excess ratio, mean over t >= 4:",
      round(float(np.mean(np.array(ref["excess_tf"][4:]) / np.array(ref["excess_bayes"][4:]))), 3))
