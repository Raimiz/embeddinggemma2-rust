use anyhow::{Context, Result, bail, ensure};
use embeddinggemma2::{Device, EmbeddingGemma2, Input, Precision};
use embeddinggemma2::gpu::text::{GraphPocMode, GraphPocPassMetrics};
use serde_json::{Value, json};
use std::{env, fs, path::PathBuf, time::Instant};

#[derive(Clone, Copy, Debug)]
struct Mode { name: &'static str, value: GraphPocMode }
const MODES: [Mode; 3] = [
    Mode { name: "A_production_eager", value: GraphPocMode::Eager },
    Mode { name: "B_persistent_eager", value: GraphPocMode::PersistentEager },
    Mode { name: "C_cuda_graph", value: GraphPocMode::CudaGraph },
];

fn percentile(xs: &[f64], p: f64) -> f64 {
    let mut sorted = xs.to_vec(); sorted.sort_by(f64::total_cmp);
    sorted[((p * sorted.len() as f64).ceil() as usize).saturating_sub(1).min(sorted.len()-1)]
}
fn median(xs: &[f64]) -> f64 { percentile(xs, 0.5) }
fn sum_metric(ms: &[GraphPocPassMetrics], f: impl Fn(&GraphPocPassMetrics) -> f64) -> f64 { ms.iter().map(f).sum() }

fn run(model: &EmbeddingGemma2, scenario: &str, long: &str, batch: &[String]) -> Result<Vec<Vec<f32>>> {
    match scenario {
        "long" => Ok(vec![model.embed(&Input::text(long.to_owned()))?]),
        "batch64" => model.embed_texts(batch),
        _ => bail!("scenario must be long or batch64"),
    }
}
fn metrics_json(ms: &[GraphPocPassMetrics]) -> Value {
    let passes: Vec<Value> = ms.iter().map(|x| json!({
        "rows":x.rows,"allocation_ms":x.allocation_ms,"capture_ms":x.capture_ms,
        "h2d_submit_ms":x.h2d_submit_ms,"host_dispatch_ms":x.host_dispatch_ms,
        "gpu_h2d_ms":x.gpu_h2d_ms,"gpu_compute_ms":x.gpu_compute_ms,
        "d2h_wall_ms":x.d2h_wall_ms,"gpu_d2h_ms":x.gpu_d2h_ms,
    })).collect();
    json!({
        "passes": passes,
        "allocation_ms": sum_metric(ms, |x| x.allocation_ms),
        "capture_ms": sum_metric(ms, |x| x.capture_ms),
        "h2d_submit_ms": sum_metric(ms, |x| x.h2d_submit_ms),
        "host_dispatch_ms": sum_metric(ms, |x| x.host_dispatch_ms),
        "gpu_h2d_ms": sum_metric(ms, |x| x.gpu_h2d_ms),
        "gpu_compute_ms": sum_metric(ms, |x| x.gpu_compute_ms),
        "d2h_wall_ms": sum_metric(ms, |x| x.d2h_wall_ms),
        "gpu_d2h_ms": sum_metric(ms, |x| x.gpu_d2h_ms),
    })
}
fn compare(a: &[Vec<f32>], b: &[Vec<f32>]) -> Result<Value> {
    ensure!(a.len() == b.len(), "embedding count differs: {} vs {}", a.len(), b.len());
    let mut per_item = Vec::with_capacity(a.len());
    for (i, (x,y)) in a.iter().zip(b).enumerate() {
        ensure!(x.len() == y.len(), "embedding dimension differs at item {i}");
        let mut dot=0.0f64; let mut nx=0.0f64; let mut ny=0.0f64; let mut max_abs=0.0f64;
        for (&u,&v) in x.iter().zip(y) { let (u,v)=(u as f64,v as f64); dot+=u*v; nx+=u*u; ny+=v*v; max_abs=max_abs.max((u-v).abs()); }
        let cosine=dot/(nx*ny).sqrt();
        per_item.push(json!({"item":i,"dim":x.len(),"cosine":cosine,"max_abs":max_abs}));
    }
    let min_cosine=per_item.iter().map(|x| x["cosine"].as_f64().unwrap()).fold(f64::INFINITY,f64::min);
    let max_abs=per_item.iter().map(|x| x["max_abs"].as_f64().unwrap()).fold(0.0f64,f64::max);
    Ok(json!({"count":a.len(),"dim":a.first().map_or(0,Vec::len),"min_cosine":min_cosine,"max_abs":max_abs,"per_item":per_item}))
}
fn mutate_same_lengths(model: &EmbeddingGemma2, text: &str) -> Result<String> {
    let n=model.tokenize(text)?.len();
    for (i,ch) in text.char_indices() {
        if ch.is_ascii_lowercase() || ch.is_ascii_uppercase() {
            let flipped=if ch.is_ascii_lowercase(){ch.to_ascii_uppercase()}else{ch.to_ascii_lowercase()};
            let mut candidate=text.to_owned(); candidate.replace_range(i..i+ch.len_utf8(),&flipped.to_string());
            if candidate!=text && model.tokenize(&candidate)?.len()==n { return Ok(candidate); }
        }
    }
    bail!("could not mutate fixture while preserving token count")
}
fn device_memory(model:&EmbeddingGemma2)->Result<Value>{
    let gpu=model.gpu().context("CUDA model unavailable")?;
    let (free,total)=gpu.ctx.mem_get_info().map_err(|e|anyhow::anyhow!("{e:?}"))?;
    let (pool_used,pool_reserved)=gpu.memory_pool_bytes()?;
    Ok(json!({"total_bytes":total,"free_bytes":free,"used_bytes":total-free,"pool_used_bytes":pool_used,"pool_reserved_bytes":pool_reserved}))
}

fn main()->Result<()> {
    let mut args=env::args().skip(1);
    let scenario=args.next().context("usage: graph_poc <long|batch64> <model-dir> <data-dir> [iters]")?;
    let model_dir=PathBuf::from(args.next().context("missing model directory")?);
    let data_dir=PathBuf::from(args.next().context("missing data directory")?);
    let iters:usize=args.next().unwrap_or_else(||"20".into()).parse()?;
    ensure!(iters>=20,"use at least 20 samples per mode");
    let fixture:Value=serde_json::from_slice(&fs::read(data_dir.join("bench_inputs.json"))?)?;
    let long=fixture["long"].as_str().context("fixture has no long text")?.to_owned();
    let batch:Vec<String>=fixture["batch64"].as_array().context("fixture has no batch64")?.iter().map(|x|x.as_str().map(str::to_owned).context("batch item is not a string")).collect::<Result<_>>()?;
    ensure!(batch.len()==64,"expected 64 fixture documents, got {}",batch.len());
    let model=EmbeddingGemma2::load_on(model_dir,Device::Cuda{ordinal:0,precision:Precision::Bf16})?;
    let gpu=model.gpu().context("CUDA model unavailable")?;
    model.set_graph_poc_gpu_timing(false)?;
    println!("{}",json!({"record":"environment","scenario":scenario,"precision":"bf16","device":model.device.to_string(),"iters_per_mode":iters,"baseline_commit":"f15131dc109b4b541f3585f15d7643cd65984d41","input_rows":if scenario=="long" {model.tokenize(&long)?.len()} else {batch.iter().map(|s|model.tokenize(s).map(|x|x.len())).collect::<Result<Vec<_>>>()?.iter().sum()},"memory_after_model_load":device_memory(&model)?}));

    let mut first_output:Option<Vec<Vec<f32>>>=None;
    let mut setup_memory=serde_json::Map::new();
    // Two warmups per mode, interleaved. This creates the persistent Work and/or graph outside the warm timer.
    for rep in 0..2 {
        for j in 0..3 {
            let m=MODES[(rep+j)%3];
            model.set_graph_poc_mode(m.value)?;
            let _=run(&model,&scenario,&long,&batch)?;
            let ms=model.take_graph_poc_metrics();
            if m.value==GraphPocMode::Eager && first_output.is_none() { first_output=Some(run(&model,&scenario,&long,&batch)?); let _=model.take_graph_poc_metrics(); }
            if rep==0 { setup_memory.insert(m.name.into(),json!({"metrics":metrics_json(&ms),"memory":device_memory(&model)?})); }
        }
    }
    let reference=first_output.context("eager reference output missing")?;
    println!("{}",json!({"record":"warm_setup","scenario":scenario,"modes":setup_memory}));

    let mut e2e:std::collections::BTreeMap<&str,Vec<f64>>=MODES.iter().map(|m|(m.name,Vec::with_capacity(iters))).collect();
    let mut host_dispatch:std::collections::BTreeMap<&str,Vec<f64>>=MODES.iter().map(|m|(m.name,Vec::with_capacity(iters))).collect();
    let mut outputs_correct=true;
    for rep in 0..iters {
        for j in 0..3 {
            let m=MODES[(rep+j)%3];
            model.set_graph_poc_mode(m.value)?;
            let t=Instant::now();
            let out=run(&model,&scenario,&long,&batch)?;
            let wall_ms=t.elapsed().as_secs_f64()*1e3;
            let pass_metrics=model.take_graph_poc_metrics();
            let accuracy=compare(&reference,&out)?;
            outputs_correct &= accuracy["min_cosine"].as_f64().unwrap()>=0.9999 && accuracy["max_abs"].as_f64().unwrap()<=4e-3;
            e2e.get_mut(m.name).unwrap().push(wall_ms);
            let dispatch=sum_metric(&pass_metrics,|x|x.host_dispatch_ms);
            host_dispatch.get_mut(m.name).unwrap().push(dispatch);
            println!("{}",json!({"record":"sample","phase":"e2e","scenario":scenario,"rep":rep,"mode":m.name,"wall_ms":wall_ms,"host_dispatch_ms":dispatch,"accuracy_vs_eager":accuracy,"passes":pass_metrics.len()}));
        }
    }
    ensure!(outputs_correct,"warm output differs from the eager reference beyond the existing BF16 tolerance");

    // Separate event-timed device components from the uninstrumented E2E runs above.
    model.set_graph_poc_gpu_timing(true)?;
    let mut components:std::collections::BTreeMap<&str,Vec<Value>>=MODES.iter().map(|m|(m.name,Vec::with_capacity(iters))).collect();
    for rep in 0..iters {
        for j in 0..3 {
            let m=MODES[(rep+2*j)%3];
            model.set_graph_poc_mode(m.value)?;
            let _=run(&model,&scenario,&long,&batch)?;
            let pass_metrics=model.take_graph_poc_metrics();
            components.get_mut(m.name).unwrap().push(metrics_json(&pass_metrics));
            println!("{}",json!({"record":"sample","phase":"gpu_components","scenario":scenario,"rep":rep,"mode":m.name,"metrics":metrics_json(&pass_metrics)}));
        }
    }
    model.set_graph_poc_gpu_timing(false)?;

    // Mutate contents without changing any token lengths; replay must consume the new embeddings and metadata.
    let changed_long=mutate_same_lengths(&model,&long)?;
    let changed_batch:Vec<String>=batch.iter().map(|s|mutate_same_lengths(&model,s)).collect::<Result<_>>()?;
    let eager_mode=MODES[0]; let graph_mode=MODES[2];
    model.set_graph_poc_mode(eager_mode.value)?;
    let changed_reference=run(&model,&scenario,&changed_long,&changed_batch)?;
    let _=model.take_graph_poc_metrics();
    model.set_graph_poc_mode(graph_mode.value)?;
    let changed_graph=run(&model,&scenario,&changed_long,&changed_batch)?;
    let _=model.take_graph_poc_metrics();
    let changed_accuracy=compare(&changed_reference,&changed_graph)?;
    let input_changed=compare(&reference,&changed_reference)?;
    ensure!(changed_accuracy["min_cosine"].as_f64().unwrap()>=0.9999 && changed_accuracy["max_abs"].as_f64().unwrap()<=4e-3,"graph replay failed after changed same-shape inputs");
    ensure!(input_changed["max_abs"].as_f64().unwrap()>0.0,"mutated fixture did not change the eager output");
    println!("{}",json!({"record":"correctness_changed_input","scenario":scenario,"same_token_lengths":true,"items_in_order":changed_accuracy["count"],"graph_vs_eager":changed_accuracy,"changed_vs_original_eager":input_changed}));

    let summary:serde_json::Map<String,Value>=MODES.iter().map(|m|{
        let e=e2e.get(m.name).unwrap(); let d=host_dispatch.get(m.name).unwrap();
        (m.name.into(),json!({"e2e_median_ms":median(e),"e2e_p95_ms":percentile(e,0.95),"e2e_min_ms":e.iter().cloned().fold(f64::INFINITY,f64::min),"e2e_max_ms":e.iter().cloned().fold(0.0f64,f64::max),"host_dispatch_median_ms":median(d),"host_dispatch_p95_ms":percentile(d,0.95)}))
    }).collect();
    println!("{}",json!({"record":"summary","scenario":scenario,"samples_per_mode":iters,"e2e":summary,"memory_after_all_modes":device_memory(&model)?}));
    let _=gpu.synchronize();
    Ok(())
}
