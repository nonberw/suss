"""Generate the fixed MuJoCo datasets for Experiment B (train seed 1, test seed 10002)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from suss import mj

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    for name in ["pendulum", "block"]:
        for split, n, seed in [("train", 60_000, 1), ("test", 4096, 10002)]:
            path = f"data/{name}_{split}.npz"
            if os.path.exists(path):
                continue
            D = mj.gen_parallel(name, n, seed)
            np.savez_compressed(path, **D)
            print(name, split, {k: v.shape for k, v in D.items()})
