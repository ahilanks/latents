"""Upload runs/<name> folders to the HF model repo and verify each final.pt landed.
usage: python scripts/hf_upload_runs.py <name> [<name> ...]   (HF_TOKEN from the environment)"""
import sys
from pathlib import Path

from huggingface_hub import HfApi

REPO = "a5ilank/hnet-latents"
ROOT = Path(__file__).resolve().parents[1]
api = HfApi()
names = sys.argv[1:]
for n in names:
    d = ROOT / "runs" / n
    if not (d / "final.pt").exists():
        print("MISSING locally", n, flush=True)
        continue
    api.upload_folder(repo_id=REPO, folder_path=str(d), path_in_repo=f"runs/{n}", commit_message=f"think run {n}")
    print("uploaded", n, flush=True)
files = set(api.list_repo_files(REPO))
bad = [n for n in names if f"runs/{n}/final.pt" not in files]
print("VERIFY", "OK" if not bad else f"MISSING on HF: {bad}", flush=True)
sys.exit(1 if bad else 0)
