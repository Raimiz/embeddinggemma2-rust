use anyhow::{Context, Result, bail};
use embeddinggemma2::{Device, EmbeddingGemma2, Input, Precision, media};
use serde_json::{Value, json};
use std::{env, fs, path::PathBuf};

fn main() -> Result<()> {
    let mut args = env::args().skip(1);
    let case = args.next().context("usage: gemm_trace <long|batch64|image> <model-dir> <data-dir>")?;
    let model_dir = PathBuf::from(args.next().context("missing model directory")?);
    let data_dir = PathBuf::from(args.next().context("missing data directory")?);
    let inputs: Value = serde_json::from_slice(&fs::read(data_dir.join("bench_inputs.json"))?)?;
    let model = EmbeddingGemma2::load_on(&model_dir, Device::Cuda { ordinal: 0, precision: Precision::Bf16 })?;
    if let Some(gpu) = model.gpu() {
        let (bk, dynamic_smem) = gpu.flash_d512_runtime_config();
        let (registers, local_bytes, _) = gpu.flash_d512_info()?;
        println!("D512_INFO {}", json!({"bk":bk,"registers_per_thread":registers,"local_bytes_per_thread":local_bytes,"static_shared_bytes":gpu.flash_d512_static_shared_bytes()? ,"dynamic_shared_bytes":dynamic_smem,"active_blocks_per_sm":gpu.flash_d512_active_blocks_per_sm()?}));
    }
    match case.as_str() {
        "long" => {
            let text = inputs["long"].as_str().context("missing long fixture")?;
            let embedding = model.embed(&Input::text(text.to_owned()))?;
            if let Some(path) = env::var_os("EGEMMA2_TRACE_EMBEDDING_PATH") {
                fs::write(path, serde_json::to_vec(&embedding)?)?;
            }
        }
        "batch64" => {
            let texts: Vec<String> = inputs["batch64"].as_array().context("missing batch64 fixture")?.iter()
                .map(|v| v.as_str().map(str::to_owned).context("batch fixture value is not a string"))
                .collect::<Result<_>>()?;
            anyhow::ensure!(texts.len() == 64, "expected 64 batch fixtures, got {}", texts.len());
            let _ = model.embed_texts(&texts)?;
        }
        "image" => {
            let image = media::load_image(&data_dir.join("image/img_a_641x479.png"))?;
            let _ = model.embed(&Input::image(image))?;
        }
        _ => bail!("usage: gemm_trace <long|batch64|image> <model-dir> <data-dir>"),
    }
    println!("traced one {case} inference");
    Ok(())
}
