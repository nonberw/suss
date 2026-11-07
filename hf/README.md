---
license: mit
library_name: pytorch
tags:
- in-context-learning
- system-identification
- adaptive-control
- mechanistic-interpretability
- robotics
- mujoco
---

# SUSS models

Every checkpoint behind the paper *What a Transformer Learns When It Adapts to Unknown Physics In-Context: Probing, Patching and Controlling* (Anuar Zhambakin, preprint 2026). Code, paper and the pre-registration: https://github.com/nonberw/suss. Live demo of the main model in the browser: https://nonberw.github.io/suss/demo/

Small GPT-style transformers (0.1M to 3.2M parameters, plus the baselines) trained from scratch to adapt **in-context** to dynamics they have never seen: linear systems with unknown (A, B), MuJoCo systems with contact, and linear adaptive control.

## What the paper finds

- On linear systems the predictor tracks the exact Bayes predictor while data are scarce, then settles at 1.59x the Bayes excess risk.
- The pre-registered mechanism, a parameter estimate in the residual stream, is rejected (probe R^2 = 0.19).
- Exploratory, with dimension-matched controls: the sufficient statistics live in the attention key-value memory of the last two blocks. Swapping only those keys and values between episodes transfers 91-94% of the behaviour; replicated over 10 seeds.
- A second pre-registration (tag `prereg2-94b6cf8`), pushed before the analyses ran, confirms this on three other architectures (deeper, wider, longer-trained): 11 of 11 tests hold, including a prior-shift test where the network keeps applying its training prior.

## Files

`runs/a/seed*.pt` (experiment A, seeds 0-9), `runs/b/*` (MuJoCo pendulum, pushed block, bouncing ball, and no-context MLP baselines), `runs/c/*` (in-context controllers and constant-gain baselines), `runs/scale/*` (scale study). The full table with parameter counts is in `MODELS.md`.

## Load

```python
import torch
from model import GPT, GPTConfig   # model.py in this repo (same as suss/model.py on GitHub)

ck = torch.load("runs/a/seed0.pt", map_location="cpu")
model = GPT(GPTConfig(**ck["cfg"])).eval()
model.load_state_dict(ck["state"])
# tokens: (batch, time, 3) = (x1, x2, u) per step; output: (batch, time, 2) = predicted next state
```

The quickstart notebook reproduces the headline numbers on a CPU in a few minutes: https://github.com/nonberw/suss/blob/main/notebooks/quickstart.ipynb

## Cite

```bibtex
@misc{zhambakin2026suss,
  author       = {Zhambakin, Anuar},
  title        = {What a Transformer Learns When It Adapts to Unknown Physics In-Context:
                  Probing, Patching and Controlling},
  year         = {2026},
  howpublished = {Preprint},
  doi          = {10.5281/zenodo.24031898},
  url          = {https://github.com/nonberw/suss}
}
```

MIT license. Not peer reviewed.
