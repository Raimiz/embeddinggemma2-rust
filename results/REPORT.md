# SM120 long/packed CUDA Graph PoC

## Verdict

**No-go for integration.** On the RTX 5060 Ti (SM120), fixed-shape CUDA Graph replay removed nearly all host dispatch time, but improved median warm end-to-end latency by only **0.51% (long)** and **0.57% (packed Batch 64)** versus production eager. Persistent eager itself improved median by only about **0.14%** in both cases. These effects are well below the pre-agreed 3–5% gate; Batch 64 graph replay also had a worse p95 and a 32.70 ms outlier. Per the stop rule, no third independent series or full verification was run. Nothing was integrated into the baseline.

## Scope and setup

- Repository baseline: `f15131dc109b4b541f3585f15d7643cd65984d41`; isolated branch `exp/sm120-large-cuda-graphs`.
- GPU: NVIDIA RTX 5060 Ti 16 GB, SM120; CUDA Toolkit/NVRTC 13.4.92, cuBLAS 13.8.0.4.
- BF16 inference; existing model and fixtures. Graphs were captured for exact fixture shapes, after RoPE preparation. Each graph replays with the same `Work` buffers as persistent eager; embeddings, row metadata and flash tile list are uploaded before replay. No kernels were changed.
- Three modes: **A** production eager (new `Work` per pass), **B** persistent eager (cached `Work`), **C** CUDA Graph captured over that same cached `Work`. Graph capture/instantiation is excluded from warm timing.
- 20 warm E2E samples and 20 GPU-component samples per mode and case. Modes were rotated; the report uses the conventional median (mean of the two middle values for 20 samples) and nearest-rank p95. Ranges are shown to communicate sample variability. One series per case was enough to apply the stop rule.
- The run also checked changed input text with unchanged token lengths. Outputs from graph and eager matched bit-for-bit, including all 64 ordered Batch 64 embeddings.

### Exact captured shapes

- **Long:** one sequence, 2,202 token rows; non-fused, ordinary per-sequence attention path, no packed flash tile list.
- **Batch 64:** 2,857 rows greedily packed at the existing 2,048-token pass budget into:
  - sequences `[0,46)`: 46 sequences, 2,042 rows, lengths and boundaries in `batch64-shape-manifest.json`, 46 flash tiles;
  - sequences `[46,64)`: 18 sequences, 815 rows, lengths and boundaries in the manifest, 18 flash tiles.
  Both passes use packed fused attention and the existing 64-row flash tiles. The benchmark rewrites row metadata and tile lists before each replay. These are fixture-specific graphs, not a general-purpose shape cache.

## Warm latency

All values in ms. E2E includes input preparation, host dispatch, GPU work and output handling; GPU event components below are separately instrumented.

| Case | Mode | E2E median | p95 | min–max |
|---|---|---:|---:|---:|
| Long | A production eager | 34.097 | 34.317 | 33.901–34.327 |
| Long | B persistent eager | 34.048 | 34.388 | 33.619–35.023 |
| Long | C CUDA Graph | **33.924** | **34.179** | 33.590–34.640 |
| Batch 64 | A production eager | 30.363 | 30.553 | 30.186–30.824 |
| Batch 64 | B persistent eager | 30.322 | **30.408** | 30.065–31.009 |
| Batch 64 | C CUDA Graph | **30.190** | 31.257 | 29.962–32.697 |

- A → B (buffer reuse): median E2E fell **0.14% long / 0.14% Batch 64**. The measured `new_work()` cost after warm-up was 0.027 ms per long pass and 0.053 ms across the two Batch 64 passes; B has no per-call `Work` allocation. This is a small part of total latency.
- B → C (graph replay): median E2E fell **0.36% long / 0.43% Batch 64**. A → C fell **0.51% / 0.57%**, respectively. Distribution overlap and the Batch 64 tail do not support a material win.
- Graphs cut median host dispatch from **1.256 to 0.012 ms** (long) and **2.715 to 0.021 ms** (Batch 64). This saved host submission time is largely hidden by device work and the remaining CPU/input/output work.

## GPU events and host submission

Medians across the 20 component samples, ms. “Enqueue/replay stream span” is the CUDA-event interval around enqueue/replay; it includes any stream idle gaps caused by host submission and is not a sum of kernel-active times.

| Case | Mode | Host enqueue/replay | H2D event | Enqueue/replay stream span | D2H event | Host H2D-submit |
|---|---|---:|---:|---:|---:|---:|
| Long | A | 1.256 | 0.705 | 29.444 | 1.022 | 0.523 |
| Long | B | 1.268 | 0.706 | 29.413 | 1.019 | 0.526 |
| Long | C | 0.012 | 0.704 | 29.323 | 1.018 | 0.526 |
| Batch 64 | A | 2.715 | 0.954 | 25.070 | 1.434 | 0.488 |
| Batch 64 | B | 2.756 | 0.959 | 25.062 | 1.436 | 0.495 |
| Batch 64 | C | 0.021 | 0.956 | 24.974 | 1.433 | 0.496 |

H2D and D2H event times are effectively unchanged. The E2E minus the disjoint H2D, enqueue/replay-span and D2H event spans is about 2.9 ms in both cases; that remainder includes CPU tokenization/input embedding/pooling, synchronization and other overhead. It is **not** a launch-overhead estimate. Host dispatch is measured separately and overlaps with GPU execution. `new_work()` timings cover the main `Work` buffers; temporary allocations in the existing enqueue path remain common to B and C.

## Capture and memory

- Capture/instantiation: **1.771 ms** for long; **3.587 ms total** for the two Batch 64 graphs. This cost is excluded from the warm E2E table.
- Device-used snapshots after each mode’s warm setup (decimal MB; modes share one process and allocator pool):

| Case | A eager | B persistent eager | C graph | C − B |
|---|---:|---:|---:|---:|
| Long | 720.4 | 720.4 | 756.1 | **+35.7 MB** |
| Batch 64 | 686.9 | 754.0 | 789.6 | **+35.7 MB** |

B and C share the same persistent `Work`; the graph therefore adds about **34 MiB** of retained device memory in these measurements, rather than a second copy of the large work buffers. Pool counters were unchanged from B to C. The instrumented process ended with about 957 MB used for long and 991 MB for Batch 64 after the allocator retained its high-water pool; these run-end totals are cumulative, not isolated per-mode peaks.

## Correctness and reproducibility

- Original and changed-content tests compared eager against graph at unchanged token lengths. Graph outputs were bit-identical: cosine 1.0 and max absolute error 0.0 for the long output and all 64 Batch 64 outputs; Batch 64 order was preserved.
- The changed-content outputs differed from the original eager outputs (max absolute delta 0.0179 long, 0.0135 Batch 64), showing the replay consumed updated input rather than stale captured data.
- Checks passed: `cargo check --release --examples`, `cargo check --no-default-features`, and `git diff --check`.
- Benchmark sources: `examples/graph_poc.rs` and `examples/graph_manifest.rs`. Raw JSONL measurements, fixture-shape manifests, logs and GPU-before/after snapshots are alongside this report under `results/` and `logs/`.
- Reproduction (inside the `pytorch-dev` CUDA environment; substitute the existing paths if needed):

```sh
EGEMMA2_NO_GRAPH=1 cargo run --release --example graph_poc -- \
  long /root/experiments/embeddinggemma2-blackwell/model \
  /root/experiments/embeddinggemma2-sm120-compile/data 20
EGEMMA2_NO_GRAPH=1 cargo run --release --example graph_poc -- \
  batch64 /root/experiments/embeddinggemma2-blackwell/model \
  /root/experiments/embeddinggemma2-sm120-compile/data 20
```

## Decision

Stop at the PoC. CUDA Graphs are correct and substantially reduce host dispatch, but the measured E2E benefit is under 1%, below the adoption threshold, and costs ~34 MiB retained VRAM plus capture complexity. Keep production eager and leave the baseline unchanged; no PR or full verification is warranted.
