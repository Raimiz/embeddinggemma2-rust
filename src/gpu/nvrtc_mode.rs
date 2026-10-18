use anyhow::{Context, Result, anyhow, ensure};
use cudarc::nvrtc::{Ptx, result, sys};
use std::{ffi::CString, time::Instant};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Target {
    Compute80,
    Compute120,
    Sm120,
}

impl Target {
    pub fn from_env() -> Result<Self> {
        let value = std::env::var("EGEMMA2_NVRTC_TARGET").unwrap_or_else(|_| "compute_80".into());
        match value.as_str() {
            "compute_80" => Ok(Self::Compute80),
            "compute_120" => Ok(Self::Compute120),
            "sm_120" => Ok(Self::Sm120),
            _ => Err(anyhow!("EGEMMA2_NVRTC_TARGET must be compute_80, compute_120, or sm_120 (got {value:?})")),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self { Self::Compute80 => "compute_80", Self::Compute120 => "compute_120", Self::Sm120 => "sm_120" }
    }

    pub fn artifact_ext(self) -> &'static str {
        match self { Self::Compute80 | Self::Compute120 => "ptx", Self::Sm120 => "cubin" }
    }

    pub fn artifact_kind(self) -> &'static str {
        match self { Self::Compute80 | Self::Compute120 => "PTX", Self::Sm120 => "CUBIN" }
    }

    pub fn options(self) -> Vec<String> {
        vec![format!("--gpu-architecture={}", self.as_str())]
    }
}

struct Program(sys::nvrtcProgram);

impl Drop for Program {
    fn drop(&mut self) {
        // Best-effort cleanup; the original NVRTC error is more useful than a destroy error.
        let _ = unsafe { result::destroy_program(self.0) };
    }
}

fn log_text(program: sys::nvrtcProgram) -> Result<String> {
    let bytes = unsafe { result::get_program_log(program) }.context("nvrtcGetProgramLog")?;
    let raw: Vec<u8> = bytes.into_iter().map(|x| x as u8).collect();
    let end = raw.iter().position(|b| *b == 0).unwrap_or(raw.len());
    Ok(String::from_utf8_lossy(&raw[..end]).into_owned())
}

/// NVRTC's CUBIN API is available through cudarc 0.19.10's public low-level bindings.
/// `Ptx::from_binary` deliberately marks the image as a binary so `load_module` passes it
/// directly to cuModuleLoadData instead of asking the driver to JIT PTX.
pub fn compile_cubin(source: &str) -> Result<(Ptx, Vec<u8>, String, f64)> {
    let src = CString::new(source).context("CUDA source contains NUL")?;
    let program = Program(result::create_program(src.as_c_str(), None).context("nvrtcCreateProgram")?);
    let options = ["--gpu-architecture=sm_120"];
    let started = Instant::now();
    let compiled = unsafe { result::compile_program(program.0, &options) };
    let compile_ms = started.elapsed().as_secs_f64() * 1000.0;
    let log = log_text(program.0)?;
    compiled.map_err(|e| anyhow!("NVRTC CUBIN compilation failed: {e:?}\n{log}"))?;

    let mut size = 0usize;
    unsafe { sys::nvrtcGetCUBINSize(program.0, &mut size) }
        .result()
        .map_err(|e| anyhow!("nvrtcGetCUBINSize failed: {e:?}\n{log}"))?;
    ensure!(size > 0, "NVRTC returned an empty CUBIN (NVRTC log: {log})");
    let mut cubin = vec![0u8; size];
    unsafe { sys::nvrtcGetCUBIN(program.0, cubin.as_mut_ptr().cast()) }
        .result()
        .map_err(|e| anyhow!("nvrtcGetCUBIN failed: {e:?}\n{log}"))?;
    Ok((Ptx::from_binary(cubin.clone()), cubin, log, compile_ms))
}

/// Compile an artifact through cudarc's public NVRTC result wrappers. Used by the
/// reproducible metadata probe; runtime PTX modes use compile_ptx_with_opts directly.
pub fn compile_probe_artifact(source: &str, target: Target) -> Result<(Vec<u8>, String, f64)> {
    let src = CString::new(source).context("CUDA source contains NUL")?;
    let program = Program(result::create_program(src.as_c_str(), None).context("nvrtcCreateProgram")?);
    let options = target.options();
    let started = Instant::now();
    let compiled = unsafe { result::compile_program(program.0, &options) };
    let compile_ms = started.elapsed().as_secs_f64() * 1000.0;
    let log = log_text(program.0)?;
    compiled.map_err(|e| anyhow!("NVRTC {} compilation failed: {e:?}\n{log}", target.as_str()))?;

    if target == Target::Sm120 {
        let mut size = 0usize;
        unsafe { sys::nvrtcGetCUBINSize(program.0, &mut size) }
            .result()
            .map_err(|e| anyhow!("nvrtcGetCUBINSize failed: {e:?}\n{log}"))?;
        ensure!(size > 0, "NVRTC returned empty CUBIN");
        let mut bytes = vec![0u8; size];
        unsafe { sys::nvrtcGetCUBIN(program.0, bytes.as_mut_ptr().cast()) }
            .result()
            .map_err(|e| anyhow!("nvrtcGetCUBIN failed: {e:?}\n{log}"))?;
        Ok((bytes, log, compile_ms))
    } else {
        let bytes = unsafe { result::get_ptx(program.0) }.context("nvrtcGetPTX")?;
        let raw: Vec<u8> = bytes.into_iter().map(|x| x as u8).collect();
        let end = raw.iter().position(|b| *b == 0).unwrap_or(raw.len());
        Ok((raw[..end].to_vec(), log, compile_ms))
    }
}
