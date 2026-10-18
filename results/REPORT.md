# Targeted SM120 GEMM and d512 tile experiment

## Decision

No change is recommended. The cuBLASLt candidates did not produce a material GEMM speedup, and the only requested attention change, `FLASH_D512_BK=16`, made the d512 pass slower. No runtime change was integrated; no full verification or E2E run was warranted.

## Setup and reproducibility

- Isolated worktree/branch: `/root/experiments/embeddinggemma2-gemm-d512/worktree`, `exp/sm120-gemm-d512-targeted`, based on `f15131dc109b4b541f3585f15d7643cd65984d41` (`blackwell-sm120`). That branch/worktree was not modified.
- GPU: RTX 5060 Ti 16 GB (SM120), driver 615.71.09; CUDA Toolkit/NVRTC 13.4.92; cuBLAS 13.8.0.4; Incus container `pytorch-dev`.
- Reused model, fixtures, and references from `/root/experiments/embeddinggemma2-blackwell/model` and `/root/experiments/embeddinggemma2-sm120-compile/data`.
- The experimental environment variables default to the original settings (`compute_80`, BK=32). Both BK measurements explicitly used `EGEMMA2_NVRTC_TARGET=compute_120`; the cuBLASLt benchmark used the current `CUBLAS_GEMM_DFALT` backend as its baseline.
- Raw Rust GEMM-call traces, cuBLASLt timing/correctness results, d512 per-pass samples, and the two output embeddings are in this directory. The standalone benchmark source is `../benchmarks/cublaslt_bench.cpp`.

Re-run the cuBLASLt benchmark with CUDA 13.4/cuBLAS on `pytorch-dev`:

```sh
g++ -O3 -std=c++20 -I/opt/host-cuda/include ../benchmarks/cublaslt_bench.cpp \
  -L/opt/host-cuda/lib64 -Wl,-rpath,/opt/host-cuda/lib64 \
  -lcublasLt -lcublas -lcudart -o /tmp/cublaslt_bench
/tmp/cublaslt_bench > cublaslt-raw.jsonl
```

The benchmark uses the real production dimensions/layouts/dtypes and deterministic BF16 A/B buffers, reusing the *same* buffers for `cublasGemmEx` and every cuBLASLt candidate. The baseline handle uses `CUBLAS_TF32_TENSOR_OP_MATH`, matching `Gpu::new` in BF16 mode. It performs five warmups, then five interleaved CUDA-event samples of 20 calls each; allocation, handle creation, heuristic search, and initialization are outside measured regions. Output values are checked before timing. The arrays are synthetic shape-matched values, not captured model activations.

Re-run the BK comparison from the worktree root (replace `32` with `16`):

```sh
EGEMMA2_PROFILE=1 EGEMMA2_NVRTC_TARGET=compute_120 EGEMMA2_D512_BK=32 \
  cargo run --release --example d512_profile -- \
  /root/experiments/embeddinggemma2-blackwell/model \
  /root/experiments/embeddinggemma2-sm120-compile/data 20
```

`EGEMMA2_PROFILE=1` disables CUDA graphs. The example does two warmups and collects 20 per-inference device-event sums for `attn: flash (d512)`; it also reports runtime function attributes and occupancy. `gemm_trace` and its `EGEMMA2_GEMM_TRACE_PATH` switch were used only to extract actual production call signatures, outside timing.

## Production GEMM calls

The trace confirms row-major `Y[M,N] = X[M,K] W[N,K]^T` is passed to cuBLAS with row-major weights viewed as column-major `[K,N]` and transposed (`opA=T`), row-major activations viewed as column-major `[K,M]` (`opB=N`), and output viewed as column-major `[N,M]`. Thus `lda=K`, `ldb=K`, `ldc=N`, `alpha=1`, `beta=0`. For the four text/vision GEMM families measured below, A and B are BF16, C is BF16, and accumulation is `CUBLAS_COMPUTE_32F`; the algorithm is `CUBLAS_GEMM_DFALT`. (Other traced calls, such as final projection/embedding calls, use FP32 C and were not included in these candidate families.)

Batch-64 contains two packed passes with `M=815` and `M=2042`. Counts below are per inference **for each M value**; cuBLAS's actual `(m,n,k)` is `(logical N, logical M, logical K)`.

| Family | Logical `(M,N,K)` | Calls per inference, each M | cuBLAS `(m,n,k)` |
|---|---:|---:|---:|
| Text local QKV | `(815 or 2042, 2048, 512)` | 20 | `(2048, M, 512)` |
| Text global QKV | `(815 or 2042, 3072, 512)` | 4 | `(3072, M, 512)` |
| Text gate/up | `(815 or 2042, 4096, 512)` | 24 | `(4096, M, 512)` |
| Text down | `(815 or 2042, 512, 2048)` | 24 | `(512, M, 2048)` |
| Vision gate/up | `(2394, 6144, 768)` | 16 | `(6144, 2394, 768)` |

The trace files retain the other GEMM signatures and counts as well. These dimensions/counts came from the optional trace inside the production `gemm_bf16`, not from profiler label names.

## cuBLASLt results

The Lt benchmark requested up to eight heuristic results and measured the first three valid candidates (64 MiB workspace limit). No candidate reached 1% median improvement over the current call; the best was +0.55%, below the noise/engineering threshold and far short of the ~5% go/no-go. Several candidates were 3–40% slower. All measured Lt outputs matched `cublasGemmEx` exactly for the tested BF16 output buffers (max absolute error 0, cosine 1.0).

| Shape | `GemmEx` median (ms) | Fastest Lt median (ms) | Best Lt config | Median change |
|---|---:|---:|---|---:|
| QKV local, M=815 | 0.039810 | 0.039589 | ID 21 / tile 15 | +0.55% |
| QKV local, M=2042 | 0.092712 | 0.092486 | ID 21 / tile 15 | +0.24% |
| QKV global, M=815 | 0.058534 | 0.058365 | ID 21 / tile 15 | +0.29% |
| QKV global, M=2042 | 0.136632 | 0.136317 | ID 21 / tile 15 | +0.23% |
| Gate/up, M=815 | 0.076734 | 0.076560 | ID 21 / tile 15 | +0.23% |
| Gate/up, M=2042 | 0.180600 | 0.180086 | ID 21 / tile 15 | +0.28% |
| Down, M=815 | 0.041485 | 0.041390 | ID 67 / tile 331 | +0.23% |
| Down, M=2042 | 0.094499 | 0.094197 | ID 67 / tile 400 | +0.32% |
| Vision gate/up | 0.471346 | 0.471147 | ID 21 / tile 15 | +0.04% |

An optimistic sum choosing the fastest heuristic independently for every shape saves about **0.040 ms per batch-64 inference**; the earlier E2E baseline is ~30.3 ms. Vision gate/up saves ~0.0048 ms against a ~44.1 ms image E2E baseline. These projections exclude any runtime algorithm-selection/cache overhead. They do not justify adding cuBLASLt.

## Flash d512: BK=32 vs BK=16

Both builds used `compute_120`, the same model and unprompted long benchmark fixture (2202 rows), and the same profiled path. Each reported sample is the d512 device-event time summed over the global-attention layers in one inference; 20 profiled passes follow two warmups.

| BK | Median d512 pass (ms) | Range (ms) | Registers/thread | Local bytes/thread | Dynamic shared/block | Occupancy API max blocks/SM |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 4.7215 | 4.7156–4.7699 | 255 | 48 | 81,920 | 1 |
| 16 | 5.1253 | 5.1183–5.1834 | 255 | 16 | 40,960 | 1 |

BK=16 is **8.55% slower** by median; the sample ranges do not overlap. Its output embedding relative to BK=32 has cosine **0.9999931** and max absolute error **5.41e-4**. Thus the smaller tile reduces spill/local storage and halves dynamic shared memory, but leaves register use and occupancy unchanged and doubles key-tile iterations/barrier phases. The source's persistent per-thread state is dominated by the 128-float output accumulator and 64 Q-fragment words; `sacc`/`pa` shrink with BK but are not enough to offset the extra loop/synchronization work. The prior `compute_120` disassembly/resource report already showed the BK=32 register/stack pressure (255 registers, 48 bytes local); no broad SASS sweep was repeated.

BK=16 failed the requested performance gate, so no further tile sweep or additional kernel modification was attempted.

## Outcome

No promising candidate was found; no inference backend change was integrated, no 90-check suite or full E2E benchmark was run, and no PR was opened. The result is a targeted no-go, not a claim that cuBLASLt or a different attention design could never help.
