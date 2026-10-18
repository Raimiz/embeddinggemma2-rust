# NVRTC compile-target comparison on Blackwell SM120

**Status:** complete, reproducible experiment; no PR.
**Repository baseline:** `blackwell-sm120` at `f15131dc109b4b541f3585f15d7643cd65984d41`.
**Experimental branch:** `exp/nvrtc-sm120-targets` (isolated worktree).
**Hardware/software:** RTX 5060 Ti (SM120, 16 GB), driver 615.71.09, CUDA Toolkit/NVRTC 13.4.92, cuBLAS 13.8.0.4; Incus container `pytorch-dev`.

## Executive summary

All three BF16 modes passed the original **90-check** verification (0 failures). Across the five saved benchmark fixtures, all captured embeddings were **bitwise identical** between `compute_80`, `compute_120`, and native `sm_120` CUBIN. No kernel source, launch geometry, precision, cuBLAS setting, tolerance, model, tokenizer, or fixture was changed.

`compute_120` PTX and native `sm_120` CUBIN produce exactly the same disassembled SASS instruction/operand sequence in all 74 functions inspected. Their warm end-to-end results are also indistinguishable within this experiment. Native CUBIN therefore shows **no steady-state inference speedup over `compute_120` PTX**. It does avoid driver PTX JIT at module load: about 0.73 ms versus 2.09 ms with a warm JIT cache. That saving is offset on a cold run by slower NVRTC compilation, so first model-load times were effectively the same.

Changing from `compute_80` to `compute_120` does change generated machine code and resources, but the backend changed too: the PTX header identifies NVVM 7.0.1 for `compute_80` and NVVM 23.0.0 for `compute_120`. Thus this is an NVRTC target-path comparison, not a controlled comparison with a frozen NVVM backend. The long-text workload improved consistently by about 1.2%; other cases were mixed and small. There is no general win to justify changing the default. The existing `compute_80` default remains in place.

## Modes and artifact identity

All modes use the same BF16 source: `kernels.cu` + `kernels_sm80.cu`, `EG_BF16`, and `FLASH_D512_BK=32`. The source SHA-256 is recorded in `compiler-modes.json`. The sole effective NVRTC option is the architecture option; fast math remains false.

| Mode | NVRTC option / output | PTX target or ELF identity | Artifact size | Stand-alone NVRTC compile call |
|---|---|---|---:|---:|
| A | `--gpu-architecture=compute_80` → PTX | `.target sm_80`; PTX ISA 9.4; NVVM 7.0.1 | 1,759,289 B | 2,715.5 ms |
| B | `--gpu-architecture=compute_120` → PTX | `.target sm_120`; PTX ISA 9.4; NVVM 23.0.0 | 1,580,161 B | 3,185.0 ms |
| C | `--gpu-architecture=sm_120` → CUBIN | `cuobjdump`: ELF `sm=120`, toolkit 13.4 | 1,298,160 B | 3,445.6 ms |

Each runtime-loaded artifact SHA-256 matches its corresponding separate NVRTC probe artifact. NVRTC logs were empty. The CUBIN path uses the installed `cudarc 0.19.10` low-level NVRTC bindings: `nvrtcCompileProgram`, `nvrtcGetCUBINSize()`, and `nvrtcGetCUBIN()`; it is loaded through `Ptx::from_binary`/`cuModuleLoadData`, not parsed as PTX. `cuobjdump` positively identifies it as an SM120 ELF CUBIN; `nvcc` was not used.

The driver JIT cache images for both PTX modes identify as `sm_120a`. In particular, B's driver-JIT CUBIN has the same 74-function SASS instruction/operand sequence as C's native `sm_120` CUBIN. The architecture/header flags differ, but the inspected function code does not.

### Compile and module-load timing

Cold values below are the first runtime load into a separate per-mode cache directory. The first A attempt used an incorrect reference-fixture path: compilation and module load completed, then verification stopped on the fixture error. A's 90-check run was repeated with the correct fixtures. Its compile/load measurements remain valid startup observations, not a verification result.

| Mode | First compile wall | First module load | First `model loaded` log | Warm compile wall median | Warm module-load median |
|---|---:|---:|---:|---:|---:|
| A `compute_80` | 2,700 ms | 2,919 ms | 7,058 ms | 118.6 ms | 2.23 ms |
| B `compute_120` | 2,966 ms | 2,972 ms | 7,297 ms | 128.0 ms | 2.09 ms |
| C `sm_120` CUBIN | 5,966 ms | 0.73 ms | 7,288 ms | 129.1 ms | 0.73 ms |

C avoids the driver-JIT module-load cost, but cold NVRTC CUBIN compilation takes longer. The measured first model-load time therefore did not improve. Warm process startup is dominated by model loading (median 1.41–1.45 s); C's roughly 1.36 ms module-load saving is immaterial there.

## Correctness

| Mode | Checks | Failed | Group counts |
|---|---:|---:|---|
| `compute_80` | 90 | 0 | text 26, image 25, audio 20, video 13, extra 6 |
| `compute_120` | 90 | 0 | text 26, image 25, audio 20, video 13, extra 6 |
| `sm_120` CUBIN | 90 | 0 | text 26, image 25, audio 20, video 13, extra 6 |

The original BF16 tolerance profile was unchanged: embedding max-absolute error ≤ `4e-3` and cosine ≥ `0.99991`; soft-token error ≤ `3.5` and cosine ≥ `0.9996`; patches ≤ `1e-6`; log-mel ≤ `1e-5`. For the five saved inference outputs, B and C were bitwise equal to A: max absolute difference 0.0 for short text, long text, image, audio, and all 64 batch rows. The direct-output JSON comparison is in `verify.json`.

## Warm end-to-end benchmarks

Each series used one warmup plus 20 timed calls per case, with CUDA Graphs enabled. There were three independent processes per mode; variant order was rotated across series. Numbers below are the median of the three per-run medians. `Δ` is relative to A. Full individual samples and per-run summaries are in `benchmark-*.json` and `benchmark-summary.json`.

| Fixture | A `compute_80` | B `compute_120` | C `sm_120` CUBIN |
|---|---:|---:|---:|
| Short text, 17 tokens | 1.617 ms | 1.641 ms (**+1.50%**) | 1.640 ms (**+1.47%**) |
| Long text, 2,208 tokens | 33.999 ms | 33.584 ms (**−1.22%**) | 33.630 ms (**−1.09%**) |
| Batch, 64 short texts | 30.270 ms | 30.299 ms (**+0.10%**) | 30.326 ms (**+0.19%**) |
| Image, 641×479 | 43.803 ms | 44.088 ms (**+0.65%**) | 43.659 ms (**−0.33%**) |
| Audio, 9.3 s | 17.389 ms | 17.092 ms (**−1.71%**) | 17.152 ms (**−1.36%**) |

Long-text run medians were A `[33.756, 34.118, 33.999]`, B `[33.341, 33.622, 33.584]`, and C `[33.424, 33.630, 33.631]` ms: the ~0.4 ms gain was consistent across the three rotated series. Per-operation profiling supports this result: for long text, the `d256` flash-attention pass changed from 5.51 ms to 5.71 ms, while `d512` changed from 5.35 ms to 4.72–4.73 ms; the net device-time change was about −0.4 ms. Profiling disables CUDA Graphs, so these pass timings are diagnostic and are not directly interchangeable with end-to-end benchmark numbers.

The short-text delta is only about 24 μs, batch results are effectively tied, image results are mixed, and the audio comparison includes a high A run (18.477 ms). With three series and these mixed results, do not claim a universal speedup. B and C have nearly identical timings as expected from their identical SASS.

## Generated code and kernel resources

`cuobjdump --dump-sass` found 74 functions in each driver-JIT/native image. Normalized instruction mnemonics and operands match for **all 74 functions** between B's PTX-JIT CUBIN and C's native CUBIN (52,624 instruction lines each). A's `compute_80` JIT image differs in 70 of 74 functions from B; its disassembly has 54,112 instruction lines. All three have 512 `HMMA.16816.F32.BF16` and 73 `LDGSTS.E.BYPASS.128` instructions in the extracted image. No `WGMMA` or `TCGEN05` instructions were found. Thus, in this code path, SM120-targeted compilation changes scheduling/register allocation and some instruction sequences, but did not introduce those newer instruction families.

Selected actual loaded-function attributes (registers/thread; local bytes/thread; static shared bytes/block; occupancy at the recorded launch configuration):

| Kernel | A `compute_80` | B `compute_120` | C `sm_120` |
|---|---|---|---|
| `flash_attn_d64_bb` | 128; 0; 16,384; 4 blocks/SM | same | same |
| `flash_attn_d256_bb` | 206; 0; 49,152; 2 blocks/SM | 242; 0; 49,152; 2 blocks/SM | same as B |
| `flash_attn_d512_bb` | 255; 168; 1,024 static; up to 81,920 dynamic; 1 block/SM | 255; 48; 1,024 static; up to 81,920 dynamic; 1 block/SM | same as B |

For `d512`, the lower-target PTX JIT reports a larger stack/local allocation (168 versus 48 bytes per thread); occupancy remains one block/SM because the kernel requests 80 KiB dynamic shared memory. The `d256` target-120 path uses more registers, but the 49,152-byte shared-memory footprint keeps occupancy at two blocks/SM. Full attributes and disassembler resource reports are saved separately.

## Environment and methodology notes

- The same pinned model/revision, tokenizer, reference fixtures and `bench_inputs.json` were used. Input-file SHA-256 is in `environment.json`.
- GPU idle before runs: 40 °C, 180 MHz core, 405 MHz memory, 4.87 W, P8, 0% utilization, 2 MiB used; no compute process. During series, the sampled core clock reached 2,842–2,857 MHz, memory clock 13,801–14,001 MHz, maximum temperature 67 °C, and utilization 91–92%. Raw 500 ms telemetry CSVs are retained.
- CUDA Graphs stayed enabled for E2E benchmarks; only the separate per-op profile disabled them. All warmup, thread-pool, fixture, and timing policy remained constant.
- Each mode used an isolated, retained `CUDA_CACHE_PATH`; the global CUDA JIT cache was not cleared. No system CUDA/driver/library settings, PyTorch, ExLlama, services, or baseline worktree were changed.
- `cargo check --release --examples` passed before runtime tests. Verification ran before the benchmark series.

## Files

- `compiler-modes.json` — options, artifact identity/hash/size, compile and load timings.
- `verify.json`, `verify-*.log` — correctness results and original check logs.
- `benchmark-summary.json`, `benchmark-*-run*.json`, `gpu-*-run*.csv` — summaries, raw samples, and GPU telemetry.
- `profile-*.json` and `profile-*.log` — separate per-operation profile runs.
- `kernel-attributes.json`, `sass-comparison.json` — loaded kernel attributes and SASS comparison.
- Stand-alone NVRTC probe logs and exact PTX/CUBIN files are under `../artifacts/probe_*`; extracted driver-JIT/native SASS, ELF, and resource reports are under `../artifacts/disassembly`.
- Scripts for the experiment and analysis are under `../scripts` and `worktree/scripts`.

## Decision

Keep `compute_80` as the default. `compute_120` may be a useful optional path for long-text-heavy workloads, but its roughly 1.2% gain is narrow and comes with a small short-text regression; more series would be needed before changing the default. Native CUBIN is valuable only if reducing module-load/JIT latency is independently important: here it gives no warm inference gain over `compute_120` PTX, and its cold compile time offsets the JIT saving.
