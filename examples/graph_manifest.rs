use anyhow::{Context, Result};
use serde_json::{Value, json};
use std::{env, fs, path::PathBuf};
use tokenizers::Tokenizer;

fn ids(tokenizer:&Tokenizer, text:&str)->Result<Vec<u32>> {
    let mut ids=tokenizer.encode(text,true).map_err(|e|anyhow::anyhow!("tokenizing: {e}"))?.get_ids().to_vec();
    if ids.len()>8192 { let eos=*ids.last().unwrap(); ids.truncate(8191); ids.push(eos); }
    Ok(ids)
}
fn main()->Result<()> {
    let mut a=env::args().skip(1);
    let model=PathBuf::from(a.next().context("usage: graph_manifest <model-dir> <data-dir> <long|batch64>")?);
    let data=PathBuf::from(a.next().context("missing data directory")?);
    let scenario=a.next().context("missing scenario")?;
    let tokenizer=Tokenizer::from_file(model.join("tokenizer.json")).map_err(|e|anyhow::anyhow!("loading tokenizer: {e}"))?;
    let inputs:Value=serde_json::from_slice(&fs::read(data.join("bench_inputs.json"))?)?;
    if scenario=="long" {
        let text=inputs["long"].as_str().context("no long fixture")?;
        let l=ids(&tokenizer,text)?.len();
        println!("{}",json!({"scenario":"long","rows":l,"lens":[l],"fused":false,"attention":"ordinary per-sequence path; no packed tile list","graph_shape":"exact rows and one sequence"}));
        return Ok(());
    }
    anyhow::ensure!(scenario=="batch64","scenario must be long or batch64");
    let batch=inputs["batch64"].as_array().context("no batch64 fixture")?;
    let lens:Vec<usize>=batch.iter().map(|x|ids(&tokenizer,x.as_str().context("batch entry is not text")?).map(|v|v.len())).collect::<Result<_>>()?;
    let mut packs=Vec::new(); let mut start=0;
    while start<lens.len() {
        let mut end=start+1; let mut rows=lens[start];
        while end<lens.len() && rows+lens[end]<=2048 { rows+=lens[end]; end+=1; }
        let pl=&lens[start..end];
        let tiles:usize=pl.iter().map(|l|l.div_ceil(64)).sum();
        packs.push(json!({"sequence_indices":[start,end],"sequence_count":pl.len(),"lens":pl,"rows":rows,"packed_fused":true,"mean_len":rows as f64/pl.len() as f64,"flash_packed":rows>=24*pl.len(),"flash_tile_count":tiles,"tile_rows":64}));
        start=end;
    }
    println!("{}",json!({"scenario":"batch64","pack_token_budget":2048,"total_sequences":lens.len(),"total_rows":lens.iter().sum::<usize>(),"all_sequence_token_lengths":lens,"greedy_packs":packs}));
    Ok(())
}
