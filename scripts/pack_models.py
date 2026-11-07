"""Pack every trained checkpoint into submission/suss_models.zip with a manifest (MODELS.md) for the GitHub release."""
import glob
import json
import os
import zipfile

import torch

DESCR = [
    ("runs/a/seed*.pt", "A: in-context predictor on linear systems (n=2, m=1), 20k steps; seeds 0-2 pre-registered, 3-9 replication"),
    ("runs/b/*_tf_seed*.pt", "B: in-context predictor on MuJoCo pendulum / block / bouncer (includes token normalisation stats)"),
    ("runs/b/*_mlp_seed*.pt", "B: no-context MLP baseline"),
    ("runs/c/bc_seed*.pt", "C: controller after the DAgger warm start (imitation only)"),
    ("runs/c/seed?.pt", "C: final controller (warm start + log-cost fine-tuning, validation-selected checkpoint)"),
    ("runs/c/seed?_last.pt", "C: last fine-tuning checkpoint (not selected)"),
    ("runs/c/seed0_raw*.pt", "C: pilot, warm start + raw cost (diverged; selected = warm start)"),
    ("runs/c/seed0_cvar*.pt", "C: pilot, warm start + CVaR (diverged; selected = warm start)"),
    ("runs/c/fixed_seed*.pt", "C: best constant-gain baseline (a 1x2 gain matrix)"),
    ("runs/scale/*.pt", "Scale study: long60k, deep8, wide256, small64, n4m2_s{0,1,2}"),
]
lines = ["# SUSS trained models", "",
         "All checkpoints used in the paper. Load with `torch.load(path, map_location='cpu')`; GPT checkpoints hold",
         "`cfg` (a `suss.model.GPTConfig` as a dict) and `state` (the state dict). See `notebooks/quickstart.ipynb`.", "",
         "| file | what | parameters | MB |", "|---|---|---|---|"]
files = []
for pat, what in DESCR:
    for f in sorted(glob.glob(pat)):
        if f in [x for x, _ in files]:
            continue
        ck = torch.load(f, map_location="cpu", weights_only=False)
        n = sum(v.numel() for v in ck["state"].values()) if "state" in ck else (ck["K"].numel() if "K" in ck else 0)
        files.append((f, what))
        lines.append(f"| `{f.replace(os.sep, '/')}` | {what} | {n:,} | {os.path.getsize(f) / 1e6:.1f} |")
os.makedirs("submission", exist_ok=True)
manifest = "\n".join(lines) + "\n"
with zipfile.ZipFile("submission/suss_models.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f, _ in files:
        z.write(f, f.replace(os.sep, "/"))
    z.writestr("MODELS.md", manifest)
open("docs/MODELS.md", "w", encoding="utf-8").write(manifest)
# the one model notebooks/quickstart.ipynb needs, as a small separate download
with zipfile.ZipFile("submission/suss_quickstart_models.zip", "w", zipfile.ZIP_DEFLATED) as z:
    z.write("runs/a/seed0.pt", "runs/a/seed0.pt")
print(f"{len(files)} checkpoints, zip {os.path.getsize('submission/suss_models.zip') / 1e6:.1f} MB; "
      f"quickstart zip {os.path.getsize('submission/suss_quickstart_models.zip') / 1e6:.1f} MB")
