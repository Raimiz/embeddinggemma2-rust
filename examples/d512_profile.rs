use anyhow::{Context, Result};
use embeddinggemma2::{Device, EmbeddingGemma2, Input, Precision};
use serde_json::{Value, json};
use std::{env, fs, path::PathBuf};

fn main() -> Result<()> {
    anyhow::ensure!(env::var_os("EGEMMA2_PROFILE").is_some(), "set EGEMMA2_PROFILE=1 to enable device event profiling");
    let mut args = env::args().skip(1);
    let model_dir = PathBuf::from(args.next().context("usage: d512_profile <model-dir> <data-dir> <samples>")?);
    let data_dir = PathBuf::from(args.next().context("missing data directory")?);
    let samples: usize = args.next().context("missing sample count")?.parse()?;
    let inputs: Value = serde_json::from_slice(&fs::read(data_dir.join("bench_inputs.json"))?)?;
    let text = inputs["long"].as_str().context("missing long fixture")?.to_owned();
    let model = EmbeddingGemma2::load_on(&model_dir, Device::Cuda { ordinal: 0, precision: Precision::Bf16 })?;
    let gpu = model.gpu().context("CUDA model unavailable")?;
    let (bk, dynamic_smem) = gpu.flash_d512_runtime_config();
    let (registers, local_bytes, _) = gpu.flash_d512_info()?;
    let attrs = json!({
        "bk": bk,
        "registers_per_thread": registers,
        "local_bytes_per_thread": local_bytes,
        "static_shared_bytes": gpu.flash_d512_static_shared_bytes()?,
        "dynamic_shared_bytes": dynamic_smem,
        "active_blocks_per_sm": gpu.flash_d512_active_blocks_per_sm()?,
    });
    let run = || model.embed(&Input::text(text.clone())).map(|_| ());
    for _ in 0..2 { run()?; let _ = gpu.take_profile(); }
    let mut samples_ms = Vec::with_capacity(samples);
    for _ in 0..samples {
        run()?;
        let profile = gpu.take_profile();
        let total = profile.iter().filter(|(label, _)| *label == "attn: flash (d512)").map(|(_, ms)| *ms).sum::<f64>();
        anyhow::ensure!(total > 0.0, "no d512 attention marks in profile");
        samples_ms.push(total);
    }
    let mut sorted = samples_ms.clone(); sorted.sort_by(f64::total_cmp);
    let report = json!({
        "target": env::var("EGEMMA2_NVRTC_TARGET").unwrap_or_else(|_| "compute_80".into()),
        "precision": "bf16",
        "input_fixture": "long benchmark fixture (unprompted; 2202 rows per shape trace)",
        "cuda_graphs_enabled": false,
        "warmups": 2,
        "profiled_passes": samples,
        "timing_scope": "sum of device-event intervals labeled attn: flash (d512), across all global layers in one long-text inference",
        "attributes": attrs,
        "samples_ms": samples_ms,
        "median_ms": sorted[sorted.len()/2],
        "min_ms": sorted[0],
        "max_ms": sorted[sorted.len()-1],
    });
    println!("{}", serde_json::to_string(&report)?);
    Ok(())
}
