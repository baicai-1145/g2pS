#!/bin/bash
# Fetch numpy weights from HF (fallback: ModelScope) into models/<tier>/
set -e
cd "$(dirname "$0")/.."
for tier in small tiny nano; do
  for f in "$tier.npz" "meta.pkl"; do
    hf_url="https://huggingface.co/baicai1145/g2pS/resolve/main/models/$tier/$f"
    ms_url="https://modelscope.cn/models/baicai1145/g2pS/resolve/master/models/$tier/$f"
    curl -fL --retry 2 -o "models/$tier/$f" "$hf_url" || curl -fL --retry 2 -o "models/$tier/$f" "$ms_url"
  done
done
echo "done: $(du -sh models/*/ | tr '\n' ' ')"
