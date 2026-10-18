use embeddinggemma2::{Device, EmbeddingGemma2, Input, Precision, media};
use serde_json::{json, Value};
use std::{fs, path::PathBuf};
fn main() -> anyhow::Result<()> {
 let args: Vec<String> = std::env::args().collect();
 let model_dir=PathBuf::from(&args[1]); let data=PathBuf::from(&args[2]); let out=PathBuf::from(&args[3]);
 fs::create_dir_all(&out)?;
 let model=EmbeddingGemma2::load_on(model_dir,Device::Cuda{ordinal:0,precision:Precision::Bf16})?;
 let b:Value=serde_json::from_slice(&fs::read(data.join("bench_inputs.json"))?)?;
 let short=b["short"].as_str().unwrap().to_string(); let long=b["long"].as_str().unwrap().to_string();
 let batch:Vec<String>=b["batch64"].as_array().unwrap().iter().map(|x|x.as_str().unwrap().to_string()).collect();
 let short_ids=model.tokenize(&short)?; let long_ids=model.tokenize(&long)?;
 let batch_ids:Vec<Vec<u32>>=batch.iter().map(|x|model.tokenize(x)).collect::<anyhow::Result<_>>()?;
 let short_e=model.embed(&Input::text(short.clone()))?; let long_e=model.embed(&Input::text(long.clone()))?;
 let batch_e=model.embed_texts(&batch)?;
 let img=media::load_image(&data.join("image/img_a_641x479.png"))?;
 let (image_ids,img_e)=model.embed_with_ids(&Input::image(img))?;
 let wav=media::load_audio(&data.join("audio/c_9p3s.wav"))?;
 let (audio_ids,aud_e)=model.embed_with_ids(&Input::audio(wav))?;
 let outputs=json!({"short":{"text":short,"input_ids":short_ids,"embedding":short_e},"long":{"text":long,"input_ids":long_ids,"embedding":long_e},"batch64":{"texts":batch,"input_ids":batch_ids,"embeddings":batch_e},"image":{"path":"image/img_a_641x479.png","input_ids":image_ids,"embedding":img_e},"audio":{"path":"audio/c_9p3s.wav","input_ids":audio_ids,"embedding":aud_e}});
 fs::write(out.join("rust-bf16-vectors.json"),serde_json::to_vec(&outputs)?)?;
 Ok(())
}
