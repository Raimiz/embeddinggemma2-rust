#!/usr/bin/env bash
set -euo pipefail
source /root/experiments/embeddinggemma2-sm120-compile/scripts/env.sh
RUN_ID=${RUN_ID:-$(date +%Y%m%d-%H%M%S)}
OUT=${OUT_DIR:-"$EXP/results/run-$RUN_ID"}
ART=${ART_DIR:-"$EXP/artifacts/runtime-bench/$RUN_ID"}
CACHE_ROOT=${CACHE_ROOT:-"$EXP/cache/$RUN_ID"}
[[ ! -e "$OUT" ]] || { echo "Refusing to overwrite $OUT" >&2; exit 2; }
BIN="$EXP/worktree/target/release/egemma2"
mkdir -p "$OUT" "$ART" "$CACHE_ROOT"
# Rotated orders reduce ordering bias; each case uses one warmup and 20 timed calls.
orders=("compute_80 compute_120 sm_120" "compute_120 sm_120 compute_80" "sm_120 compute_80 compute_120")
for r in 1 2 3; do
  for target in ${orders[$((r-1))]}; do
    export EGEMMA2_NVRTC_TARGET="$target"
    export CUDA_CACHE_PATH="$CACHE_ROOT/$target"
    export EGEMMA2_NVRTC_ARTIFACT_DIR="$ART/$target"
    export EGEMMA2_NVRTC_METADATA_PATH="$OUT/runtime-${target}-run${r}.json"
    export EGEMMA2_KERNEL_ATTRIBUTES_PATH="$OUT/kernel-attributes-${target}-run${r}.json"
    mkdir -p "$CUDA_CACHE_PATH" "$EGEMMA2_NVRTC_ARTIFACT_DIR"
    echo "START run=$r target=$target at $(date --iso-8601=seconds)" | tee "$OUT/benchmark-${target}-run${r}.log"
    nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.gr,clocks.mem,power.draw,pstate,utilization.gpu,memory.used --format=csv,noheader -lms 500 >> "$OUT/gpu-${target}-run${r}.csv" &
    sampler=$!
    set +e
    "$BIN" --device cuda --bf16 --model-dir "$BASE/model" bench --data-dir "$EXP/data" --iters 20 --out "$OUT/benchmark-${target}-run${r}.json" 2>&1 | tee -a "$OUT/benchmark-${target}-run${r}.log"
    rc=${PIPESTATUS[0]}
    set -e
    kill "$sampler" 2>/dev/null || true
    wait "$sampler" 2>/dev/null || true
    if [ "$rc" -ne 0 ]; then echo "FAILED run=$r target=$target rc=$rc" | tee -a "$OUT/benchmark-${target}-run${r}.log"; exit "$rc"; fi
    echo "END run=$r target=$target at $(date --iso-8601=seconds)" | tee -a "$OUT/benchmark-${target}-run${r}.log"
    nvidia-smi --query-gpu=temperature.gpu,clocks.gr,clocks.mem,power.draw,pstate,utilization.gpu,memory.used --format=csv,noheader > "$OUT/gpu-after-${target}-run${r}.csv"
    sleep 2
  done
done
printf 'Benchmark outputs: %s\n' "$OUT"
