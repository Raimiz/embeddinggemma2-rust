#!/usr/bin/env bash
set -euo pipefail
source /root/experiments/embeddinggemma2-sm120-compile/scripts/env.sh
D="$EXP/artifacts/disassembly"
mkdir -p "$D"
A="$EXP/artifacts/cache-compute_80-f503c48c4bd568-0.elf"
B="$EXP/artifacts/cache-compute_120-8c2cdd9cf9fd28-0.elf"
C="$EXP/artifacts/runtime_sm_120_verify/kernels.cubin"
for spec in "compute_80_jit:$A" "compute_120_jit:$B" "sm_120_cubin:$C"; do
  name=${spec%%:*}; image=${spec#*:}
  [[ -r "$image" ]] || { echo "missing retained image: $image" >&2; exit 2; }
  cuobjdump --dump-sass "$image" > "$D/$name.sass"
  cuobjdump --dump-resource-usage "$image" > "$D/$name.resources.txt"
  cuobjdump --dump-elf "$image" > "$D/$name.elf.txt"
done
python3 "$EXP/worktree/scripts/compare_sass.py"
