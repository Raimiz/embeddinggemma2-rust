use anyhow::{Context, Result};
use embeddinggemma2::{Device, EmbeddingGemma2, Input, Precision, media};
use serde_json::{Value, json};
use std::{env, fs, path::PathBuf, time::Instant};

fn main() -> Result<()> {
    let mut args = env::args().skip(1);
    let model_dir = PathBuf::from(args.next().context("usage: compare_nvrtc_modes <model-dir> <data-dir> <output-json>")?);
    let data_dir = PathBuf::from(args.next().context("missing data-dir")?);
    let out_path = PathBuf::from(args.next().context("missing output-json")?);
    let started = Instant::now();
    let model = EmbeddingGemma2::load_on(&model_dir, Device::Cuda { ordinal: 0, precision: Precision::Bf16 })?;
    let load_ms = started.elapsed().as_secs_f64() * 1000.0;
    let input_path = data_dir.join("bench_inputs.json");
    let inputs: Value = serde_json::from_slice(&fs::read(&input_path)?)?;
    let short = inputs["short"].as_str().context("missing short")?;
    let long = inputs["long"].as_str().context("missing long")?;
    let batch: Vec<String> = inputs["batch64"].as_array().context("missing batch64")?.iter()
        .map(|v| v.as_str().map(str::to_owned).context("batch item is not a string"))
        .collect::<Result<_>>()?;
    let img = media::load_image(&data_dir.join("image/img_a_641x479.png"))?;
    let wav = media::load_audio(&data_dir.join("audio/c_9p3s.wav"))?;
    let result = json!({
        "target": env::var("EGEMMA2_NVRTC_TARGET").unwrap_or_else(|_| "compute_80".into()),
        "model_load_ms": load_ms,
        "short": model.embed(&Input::text(short))?,
        "long": model.embed(&Input::text(long))?,
        "batch64": model.embed_texts(&batch)?,
        "image": model.embed(&Input::image(img))?,
        "audio": model.embed(&Input::audio(wav))?,
    });
    if let Some(parent) = out_path.parent() { fs::create_dir_all(parent)?; }
    fs::write(&out_path, serde_json::to_vec_pretty(&result)?)?;
    println!("wrote embeddings for {} to {}", result["target"], out_path.display());
    Ok(())
}
