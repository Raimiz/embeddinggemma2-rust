# Experiment worktree notes

This branch adds a BF16-only `EGEMMA2_NVRTC_TARGET` switch and measurement hooks for comparing the original `compute_80` PTX, `compute_120` PTX, and NVRTC-native `sm_120` CUBIN. Unset target defaults to `compute_80`, preserving the baseline path. Kernel sources and runtime math settings are unchanged.

The measured report and raw samples are under `results/REPORT.md`; environment details and scripts are deliberately tied to the `pytorch-dev` Incus environment and the separate experiment directory at `/root/experiments/embeddinggemma2-sm120-compile`.

## Reproduce

1. Use Incus container `pytorch-dev` and the recorded baseline/model/reference data under `/root/experiments/embeddinggemma2-blackwell`.
2. `source /root/experiments/embeddinggemma2-sm120-compile/scripts/env.sh`.
3. Build/check with `cargo check --release --examples`.
4. Run `scripts/run_verify.sh` first. It uses a unique output directory by default and refuses to overwrite it.
5. Run `scripts/run_benchmarks.sh` for three rotated rounds (one warmup plus 20 measured calls per fixture); set `RUN_ID` to choose a unique output/cache/artifact set. `scripts/run_profiles.sh` is separate and disables CUDA Graphs as required by the existing profile hook.
6. Run `python3 scripts/summarize_results.py` for the canonical experiment paths. `scripts/extract_sass.sh` regenerates raw `cuobjdump` disassembly/resource reports and `results/sass-comparison.json` from the retained extracted ELF images.

Do not clear global CUDA caches or reuse output paths if you want to preserve the published raw run. Cold and warm cache behavior are recorded separately in the report.
