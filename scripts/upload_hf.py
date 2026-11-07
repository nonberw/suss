"""Upload every trained checkpoint, the model card (hf/README.md) and the manifest to the Hugging Face Hub.

    pip install huggingface_hub
    hf auth login                      # paste a write token from https://huggingface.co/settings/tokens
    python scripts/upload_hf.py --repo <your-hf-username>/suss-models

Run from the repo root after unzipping suss_models.zip (from the GitHub release) there, or after training."""
import argparse
import glob
import os
import shutil
import tempfile

from huggingface_hub import HfApi

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True, help="e.g. nonberw/suss-models")
ap.add_argument("--private", action="store_true")
a = ap.parse_args()

files = sorted(glob.glob("runs/a/seed*.pt") + glob.glob("runs/b/*.pt") + glob.glob("runs/c/*.pt") + glob.glob("runs/scale/*.pt"))
if not files:
    raise SystemExit("No checkpoints under runs/. Unzip suss_models.zip from the v0.1.3 release at the repo root first.")
api = HfApi()
api.create_repo(a.repo, repo_type="model", private=a.private, exist_ok=True)
with tempfile.TemporaryDirectory() as tmp:
    for f in files:
        dst = os.path.join(tmp, f)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(f, dst)
    shutil.copy2("hf/README.md", os.path.join(tmp, "README.md"))
    shutil.copy2("docs/MODELS.md", os.path.join(tmp, "MODELS.md"))
    shutil.copy2("suss/model.py", os.path.join(tmp, "model.py"))
    api.upload_folder(folder_path=tmp, repo_id=a.repo, repo_type="model",
                      commit_message="SUSS checkpoints (v0.1.3), model card and manifest")
print(f"uploaded {len(files)} checkpoints to https://huggingface.co/{a.repo}")
