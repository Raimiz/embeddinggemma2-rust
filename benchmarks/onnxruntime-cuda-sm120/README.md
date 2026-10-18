# EmbeddingGemma 2 ONNX Runtime CUDA experiment (SM120)

This is the result archive for the isolated ONNX Runtime CUDA experiment. Start with [`results/REPORT.md`](results/REPORT.md).

It includes accuracy vectors/metrics, all three 20-sample benchmark runs, I/O Binding and CUDA Graph probes, per-phase memory captures, graph/operator analysis, ONNX Runtime provider profiles, environment/dependency details, original benchmark/media fixtures, and the scripts used for the runs.

The experiment used the public `onnx-community/embeddinggemma-2-ONNX` repository at revision `daa72c51243991dfcaf9f9137d2c573d8f7790c0`. The 12 ONNX graph and external-weight files are deliberately not included (about 1.46 GB even for FP16); no FP32 graph was downloaded or benchmarked. Re-download the published files at that exact revision to reproduce inference. The benchmark inputs and media are included here; the same author fixtures are also present under the Rust project's `testdata/`.

The scripts record the original controlled workspace and library paths (`/root/experiments/embeddinggemma2-ort-cuda` and the CUDA/cuDNN paths in `results/environment.json`). They are archived as experimental provenance, not a portable one-command installer. Do not use them without first reviewing/adapting paths and installing the locked dependencies in an isolated environment.
