#!/usr/bin/env bash
set -euo pipefail
W=/root/experiments/embeddinggemma2-ort-cuda
CUDA=/opt/host-cuda/targets/x86_64-linux/lib
NV=/root/workspace/.venv/lib/python3.15/site-packages/nvidia
export LD_LIBRARY_PATH="$CUDA:$NV/cu13/lib:$NV/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
for format in fp16 q8 q4 q4f16; do
  for run in 1 2 3; do
    "$W/.ort-venv/bin/python" "$W/scripts/benchmark_ort.py" "$format" "$run" --iters 20 \
      2>&1 | tee "$W/logs/benchmark-${format}-run${run}.log"
  done
done
