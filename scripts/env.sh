#!/usr/bin/env bash
set -euo pipefail
export EXP=/root/experiments/embeddinggemma2-sm120-compile
export BASE=/root/experiments/embeddinggemma2-blackwell
export RUSTUP_HOME="$BASE/.rustup"
export CARGO_HOME="$BASE/.cargo"
export RUSTUP_TOOLCHAIN=stable
export PATH="$CARGO_HOME/bin:/opt/host-cuda/bin:$BASE/tools/ffmpeg-static:$PATH"
export CUDA_HOME=/opt/host-cuda
export LD_LIBRARY_PATH="/opt/host-cuda/targets/x86_64-linux/lib:/opt/host-cuda/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_CACHE_PATH="$EXP/cache/default"
mkdir -p "$CUDA_CACHE_PATH"
cd "$EXP/worktree"
