#!/usr/bin/env bash
set -euo pipefail
source /root/experiments/embeddinggemma2-sm120-compile/scripts/env.sh
OUT_DIR=${OUT_DIR:-"$EXP/results/verify-rerun-$(date +%Y%m%d-%H%M%S)"}
[[ ! -e "$OUT_DIR" ]] || { echo "Refusing to overwrite $OUT_DIR" >&2; exit 2; }
mkdir -p "$OUT_DIR"
BIN="$EXP/worktree/target/release/egemma2"
for target in compute_80 compute_120 sm_120; do
  export EGEMMA2_NVRTC_TARGET="$target"
  export CUDA_CACHE_PATH="$EXP/cache/verify-rerun/$target"
  export EGEMMA2_NVRTC_ARTIFACT_DIR="$EXP/artifacts/verify-rerun/$target"
  export EGEMMA2_NVRTC_METADATA_PATH="$OUT_DIR/runtime-$target.json"
  export EGEMMA2_KERNEL_ATTRIBUTES_PATH="$OUT_DIR/kernel-attributes-$target.json"
  mkdir -p "$CUDA_CACHE_PATH" "$EGEMMA2_NVRTC_ARTIFACT_DIR"
  nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.gr,clocks.mem,power.draw,pstate,utilization.gpu,memory.used --format=csv,noheader > "$OUT_DIR/gpu-before-$target.csv"
  "$BIN" --device cuda --bf16 --model-dir "$BASE/model" verify --reference-dir "$BASE/reference" 2>&1 | tee "$OUT_DIR/verify-$target.log"
  nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.gr,clocks.mem,power.draw,pstate,utilization.gpu,memory.used --format=csv,noheader > "$OUT_DIR/gpu-after-$target.csv"
done
printf 'Verification outputs: %s\n' "$OUT_DIR"
