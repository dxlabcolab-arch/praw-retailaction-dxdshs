#!/bin/bash
# Downloads the fine-tuned weights (about 13 GB) into weights/ from our Hugging Face model repo and checks them.
# usage: bash download_weights.sh    (HF_REPO can be overridden: HF_REPO=user/name bash download_weights.sh)
set -e; cd "$(dirname "$0")"
HF_REPO=${HF_REPO:-juuuncha/praw-retailaction-dxdshs-weights}
python - <<PY
from huggingface_hub import snapshot_download
snapshot_download("$HF_REPO", local_dir="weights", allow_patterns=["*.pt", "*.csv", "MD5SUMS"])
PY
cd weights && md5sum -c MD5SUMS && echo "weights OK"   # MD5SUMS comes with the repo (git), the files from HF
