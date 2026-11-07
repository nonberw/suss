"""The in-browser demo (docs/demo/model.js) must compute what the PyTorch model computes."""
import json
import os
import shutil
import subprocess

import numpy as np
import pytest
import torch

from suss.model import GPT, GPTConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO = os.path.join(ROOT, "docs", "demo")


def load_demo_model():
    man = json.load(open(os.path.join(DEMO, "weights.json")))
    flat = np.fromfile(os.path.join(DEMO, "weights.bin"), dtype="<f4")
    sd = {t["name"]: torch.from_numpy(flat[t["offset"]:t["offset"] + int(np.prod(t["shape"]))].reshape(t["shape"]).copy())
          for t in man["tensors"]}
    model = GPT(GPTConfig(**man["cfg"])).eval()
    model.load_state_dict(sd)
    return man, model


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_js_model_matches_pytorch(tmp_path):
    man, model = load_demo_model()
    T, d_in = man["cfg"]["T"], man["cfg"]["d_in"]
    tok = torch.randn(1, T, d_in, generator=torch.Generator().manual_seed(3))
    jac_at = [0, 5, T - 1]
    inp = tmp_path / "input.json"
    inp.write_text(json.dumps({"tokens": tok.flatten().tolist(), "T": T, "jac_at": jac_at}))
    run = subprocess.run(["node", os.path.join(ROOT, "tests", "demo_check.js"), os.path.join(DEMO, "weights.json"),
                          os.path.join(DEMO, "weights.bin"), str(inp)], capture_output=True, text=True, check=True)
    js = json.loads(run.stdout)
    with torch.no_grad():
        ref = model(tok)[0].numpy()
    assert np.abs(np.array(js["preds"]) - ref).max() < 1e-4
    for k, t in enumerate(jac_at):
        f = lambda q: model(torch.cat([tok[:, :t], q.view(1, 1, d_in)], 1))[0, t]
        J = torch.autograd.functional.jacobian(f, tok[0, t].clone())
        assert np.abs(np.array(js["jac"][k]) - J.numpy()).max() < 1e-3
