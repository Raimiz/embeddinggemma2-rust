#!/usr/bin/env bash
set -euo pipefail
source /root/experiments/embeddinggemma2-sm120-compile/scripts/env.sh
OUT_DIR=${OUT_DIR:-"$EXP/results/profile-rerun-$(date +%Y%m%d-%H%M%S)"}
[[ ! -e "$OUT_DIR" ]] || { echo "Refusing to overwrite $OUT_DIR" >&2; exit 2; }
mkdir -p "$OUT_DIR"
BIN="$EXP/worktree/target/release/examples/profile_passes"
for target in compute_80 compute_120 sm_120; do
  export EGEMMA2_NVRTC_TARGET="$target"
  export CUDA_CACHE_PATH="$EXP/cache/profile-rerun/$target"
  export EGEMMA2_NVRTC_ARTIFACT_DIR="$EXP/artifacts/profile-rerun/$target"
  export EGEMMA2_NVRTC_METADATA_PATH="$OUT_DIR/runtime-$target.json"
  export EGEMMA2_KERNEL_ATTRIBUTES_PATH="$OUT_DIR/kernel-attributes-$target.json"
  export EGEMMA2_PROFILE=1
  export EGEMMA2_MODEL_DIR="$BASE/model"
  export EGEMMA2_DATA_DIR="$EXP/data"
  mkdir -p "$CUDA_CACHE_PATH" "$EGEMMA2_NVRTC_ARTIFACT_DIR"
  "$BIN" bf16 2>&1 | tee "$OUT_DIR/profile-$target.log"
done
printf 'Profile outputs: %s\n' "$OUT_DIR"
