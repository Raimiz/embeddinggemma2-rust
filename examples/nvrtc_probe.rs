use anyhow::{Context, Result, bail};
use embeddinggemma2::gpu::nvrtc_mode::{Target, compile_probe_artifact};
use serde_json::json;
use std::{env, fs, path::PathBuf};

fn main() -> Result<()> {
    let mut args = env::args().skip(1);
    let target = match args.next().as_deref() {
        Some("compute_80") => Target::Compute80,
        Some("compute_120") => Target::Compute120,
        Some("sm_120") => Target::Sm120,
        _ => bail!("usage: nvrtc_probe <compute_80|compute_120|sm_120> <output-dir>"),
    };
    let dir = PathBuf::from(args.next().context("missing output directory")?);
    fs::create_dir_all(&dir)?;
    let source = format!(
        "#define EG_BF16\n#define FLASH_D512_BK 32\n{}\n{}",
        include_str!("../src/gpu/kernels.cu"),
        include_str!("../src/gpu/kernels_sm80.cu"),
    );
    let (artifact, log, compile_call_ms) = compile_probe_artifact(&source, target)?;
    let artifact_path = dir.join(format!("nvrtc-probe.{}", target.artifact_ext()));
    fs::write(&artifact_path, &artifact)?;
    fs::write(dir.join("nvrtc-compilation.log"), &log)?;
    let metadata = json!({
        "target": target.as_str(),
        "artifact_kind": target.artifact_kind(),
        "artifact_bytes": artifact.len(),
        "artifact_path": artifact_path.display().to_string(),
        "options": target.options(),
        "use_fast_math": false,
        "compilation_call_ms": compile_call_ms,
        "nvrtc_log_path": dir.join("nvrtc-compilation.log").display().to_string(),
        "nvrtc_log": log,
        "source_description": "kernels.cu + kernels_sm80.cu with EG_BF16 and FLASH_D512_BK=32, matching Gpu::new",
        "role": "diagnostic artifact/log capture; runtime timing is recorded separately by Gpu::new",
    });
    fs::write(dir.join("nvrtc-probe.json"), serde_json::to_vec_pretty(&metadata)?)?;
    println!("{} {} bytes, compile API call {:.3} ms", target.as_str(), artifact.len(), compile_call_ms);
    Ok(())
}
