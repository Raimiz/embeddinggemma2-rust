# EmbeddingGemma 2 ONNX Runtime CUDA — report

## Summary

At pinned Hugging Face revision `daa72c51243991dfcaf9f9137d2c573d8f7790c0`, all 12 published FP16, Q8, Q4, and Q4F16 text/vision/audio graphs loaded and executed on the RTX 5060 Ti (SM120) using ONNX Runtime CUDA EP 1.27.0. All four formats pass the GPU execution gate: the neural-network GEMM/MatMulNBits, attention, and convolution work was assigned to CUDA. The small CPU-assigned node groups are shape/dimension and metadata plumbing, not encoder computation.

The practical results favor **FP16** for the simplest reliable choice and **Q4F16** where lower model/VRAM footprint matters. **Q8 and Q4 are substantially slower** than FP16 on these graphs despite smaller weight files. Q4/Q4F16 also show a meaningful accuracy/retrieval-quality drop. CUDA Graph capture was not possible with the published graphs as-is; full-pipeline I/O Binding produced no material speed improvement.

## Environment and reproducibility

- Hardware: RTX 5060 Ti, SM120, 16,311 MiB VRAM; driver 615.71.09.
- Isolated runtime: Python 3.14.4, `onnxruntime-gpu==1.27.0`, ONNX 1.23.2, NumPy 2.5.4, Tokenizers 0.23.3. CUDA Toolkit 13.4, cuBLAS 13.8.0.4, cuDNN 9.26.0, NVRTC 13.4.92.
- Workspace and venv: `/root/experiments/embeddinggemma2-ort-cuda/` and `/root/experiments/embeddinggemma2-ort-cuda/.ort-venv/`. CUDA/cuDNN libraries were exposed through a per-command `LD_LIBRARY_PATH`; no system CUDA, driver, or existing PyTorch/ExLlama environment was changed.
- Only the 12 requested published ONNX graphs and external weights were downloaded. No FP32 graph was downloaded or benchmarked; no graph, weight, or quantization was modified.
- Reference: Rust CUDA BF16 at commit `f15131dc109b4b541f3585f15d7643cd65984d41`.
- Text token IDs for short, long, batch, image, and audio matched the Rust reference exactly. Image patches (`[1,2394,768]`) and audio mel features (`[1,929,128]`) matched the author fixtures bit-for-bit. Long text contains 2,202 tokenizer tokens.

## GPU placement and graph compatibility

All 12 sessions exposed `CUDAExecutionProvider` and completed actual inference. The fixed-shape profiling probes observed these node-event counts per graph:

| Tower | CUDA EP events | CPU EP events | CPU work observed |
|---|---:|---:|---|
| Text | 1,049–1,057 | 16 | Shape/mask/dimension bookkeeping |
| Vision | 698–709 | 9 | Shape/size and metadata Concat/Reshape operations |
| Audio | 1,086–1,197 | 23 | Shape and metadata arithmetic/reshape/concat operations |

All text `MatMul`/`MatMulNBits`, vision `MatMul`/`MatMulNBits` and `MultiHeadAttention`, and audio `MatMul`/`MatMulNBits` and `Conv` events were on CUDA. The quantized graphs use CUDA `com.microsoft::MatMulNBits` (146 text, 113 vision, 123 audio events in the probes). No main GEMM, attention, or convolution node fell back to CPU. Full per-node assignments and traces are retained in `results/provider-assignment.json` and `logs/provider-probes-fixed/`.

The key format distinction is the compute type encoded in the graphs: Q8/Q4 `MatMulNBits` use FP32 activations/scales and outputs, while Q4F16 uses FP16 activations/scales and outputs. Q4F16 is therefore much closer to the FP16 runtime than the nominally lower-bit Q4 graph. FP16 and Q4F16 matrix operations are Tensor-Core-capable by type and CUDA placement; **active Tensor Core use was not directly measured** because Nsight Compute counter access returned `ERR_NVGPUCTRPERM`. No counter permissions or system settings were changed.

## Accuracy against Rust CUDA BF16

Cosines compare normalized embeddings from the same exact token IDs/preprocessing and raw inputs. Batch values below are per-document cosine means/minima; neighbor agreement compares each batch item's nearest other item against Rust.

| Format | Minimum cosine across short/long/batch/image/audio | Batch cosine mean (min) | Batch top-1 neighbor agreement | Largest absolute component error |
|---|---:|---:|---:|---:|
| FP16 | 0.999946 | 0.999989 (0.999980) | 98.4% | 0.00124 |
| Q8 | 0.999452 | 0.999934 (0.999922) | 100% | 0.00421 |
| Q4 | 0.970112 | 0.981660 (0.978533) | 76.6% | 0.03135 |
| Q4F16 | 0.970168 | 0.981717 (0.978559) | 75.0% | 0.03130 |

All emitted embeddings remain unit normalized. FP16/Q8 closely reproduce BF16 embedding geometry; Q4/Q4F16 are materially noisier and alter many batch nearest-neighbor relationships. The small FP16 top-1 discrepancy occurs despite very high embedding cosine and reflects near-ties in this fixture's neighbor scores.

## Latency

Three separate runs, 20 timed samples per case per run, one warm-up per case; values are the median of the three run medians. E2E includes preprocessing, encoder(s), text graph, and the output copy used by the benchmark. Rust numbers are the already completed reference run-1 medians.

| Backend/format | Short text | Long text | Batch 64 | Image E2E | Audio E2E |
|---|---:|---:|---:|---:|---:|
| Rust CUDA BF16 | 1.614 ms | 33.920 ms | 30.138 ms | 43.786 ms | 18.665 ms |
| ORT FP16 | 4.793 ms | 51.928 ms | 34.990 ms | 201.949 ms | 30.275 ms |
| ORT Q8 | 5.375 ms | 100.618 ms | 71.383 ms | 250.271 ms | 37.363 ms |
| ORT Q4 | 6.016 ms | 102.790 ms | 73.354 ms | 253.048 ms | 42.461 ms |
| ORT Q4F16 | 5.411 ms | 52.784 ms | 35.955 ms | 203.895 ms | 32.890 ms |

Image E2E includes about 121 ms of exact, reference-matching Python image preprocessing; the measured FP16 vision graph itself is about 75 ms, followed by about 5 ms for the text graph. Audio preprocessing is about 10 ms. This matters when interpreting the image/audio totals. The Rust baseline is still faster in every case; image E2E is especially dominated by the Python preprocessing implementation.

Q8/Q4 do not yield a speed win: long-text graph medians are about 100–103 ms, roughly twice FP16. Q4F16 returns to near-FP16 latency while reducing total external weight files from 1.464 GB to 0.425 GB. Weight-file totals are: FP16 1.464 GB, Q8 0.849 GB, Q4 0.472 GB, Q4F16 0.425 GB (decimal units). Raw samples, per-run medians, p95s, preprocessing/encoder/text splits, and session initialization are in `results/benchmark-*-run*.json` and `results/benchmark-summary.json`.

## Session initialization

Median fresh-session initialization times (across the three benchmark launches) were:

| Format | Text | Vision | Audio | All three |
|---|---:|---:|---:|---:|
| FP16 | 447 ms | 141 ms | 231 ms | 821 ms |
| Q8 | 422 ms | 135 ms | 197 ms | 754 ms |
| Q4 | 391 ms | 117 ms | 168 ms | 677 ms |
| Q4F16 | 399 ms | 114 ms | 175 ms | 688 ms |

## Memory

These are isolated-process captures; `global` is the total used GPU memory and process VRAM is the corresponding `nvidia-smi` compute-app peak. RSS is process host memory. GPU snapshots were sampled at approximately 200 ms, so short transients may be missed.

| Format | After text session | After first vision | After first audio | Peak during five-case pass | Peak process VRAM | Peak RSS |
|---|---:|---:|---:|---:|---:|---:|
| FP16 | 830 MiB | 2,516 MiB | 3,684 MiB | 3,684 MiB | 3,674 MiB | 1,678 MiB |
| Q8 | 798 MiB | 3,138 MiB | 4,676 MiB | 4,676 MiB | 4,666 MiB | 1,585 MiB |
| Q4 | 572 MiB | 2,398 MiB | 3,168 MiB | 3,424 MiB | 3,414 MiB | 1,547 MiB |
| Q4F16 | 556 MiB | 1,872 MiB | 2,640 MiB | 2,640 MiB | 2,630 MiB | 1,648 MiB |

Rust BF16's recorded peak global GPU use was about 1,426 MiB. Smaller ONNX weight files do not guarantee lower peak VRAM: Q8 peaked highest, with the CUDA allocator/temporary workspace contributing beyond model weights. Q4F16 had the lowest observed peak. Memory snapshots and phase samples are in `results/memory.json` and `results/memory-*.json`.

## I/O Binding and CUDA Graphs

- **I/O Binding:** tested across all formats and all five cases, with CPU preprocessing/input binding, GPU-resident encoder outputs between modality and text sessions, and the final embedding copied back to CPU. Full-pipeline medians were effectively unchanged from ordinary `session.run` (typically within about 1%). It did not justify replacing the baseline path.
- **CUDA Graphs:** tested for all four text formats with fixed-shape CUDA I/O Binding. ORT rejected capture during session initialization: `all compute graph nodes have not been partitioned to the CUDAExecutionProvider`. The published graphs' CPU shape/metadata nodes prevent capture without changing/re-exporting the graphs, which was out of scope.

## Artifacts

- `results/REPORT.md` — this report.
- `results/environment.json`, `graph-analysis.json`, `provider-assignment.json`, `provider-assignment-raw.json`.
- `results/accuracy.json`, `results/benchmark-summary.json`, 12 raw per-run benchmark JSON files, I/O Binding summaries, optimization probes, memory captures, and Rust reference vectors.
- `logs/provider-probes-fixed/` — 12 ORT profiles and corrected provider probes; `logs/` also contains benchmark, memory, and optimization logs.
- `scripts/` — isolated probe/benchmark/preprocessing and reporting scripts.
