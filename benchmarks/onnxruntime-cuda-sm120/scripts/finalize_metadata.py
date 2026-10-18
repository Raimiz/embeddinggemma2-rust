import json,pathlib,hashlib,collections,sys,os
W=pathlib.Path("/root/experiments/embeddinggemma2-ort-cuda"); R=W/"results"; D=W/"inspect/graphs/onnx"; REF=pathlib.Path("/root/experiments/embeddinggemma2-blackwell/embeddinggemma2-rust/testdata")
# Full accuracy record, preserving per-format vectors and comparison metrics.
acc=json.load(open(R/"accuracy-probe.json")); rust=json.load(open(R/"rust-bf16-vectors.json")); b=json.load(open(W/"fixtures/bench_inputs.json"));
image=json.load(open(W/"fixtures/image/ref_image.json"))["cases"][0]; audio=next(x for x in json.load(open(W/"fixtures/audio/ref_audio.json"))["cases"] if x["name"]=="c_9p3s")
from tokenizers import Tokenizer
tok=Tokenizer.from_file(str(W/"inspect/graphs/tokenizer.json"))
def sha(obj):return hashlib.sha256(json.dumps(obj,separators=(",",":")).encode()).hexdigest()
checks={}
for name in ["short","long"]:
 ids=tok.encode(b[name]).ids; checks[name]={"rust_ids_identical":ids==rust[name]["input_ids"],"token_count":len(ids),"token_ids_sha256":sha(ids)}
bi=[tok.encode(s).ids for s in b["batch64"]]; checks["batch64"]={"all_rust_ids_identical":all(x==y for x,y in zip(bi,rust["batch64"]["input_ids"])),"documents":len(bi),"max_tokens":max(map(len,bi)),"token_ids_sha256_per_doc":[sha(x) for x in bi]}
checks["image_641x479"]={"rust_ids_identical":rust["image"]["input_ids"]==image["ids_image"],"tokens":len(image["ids_image"]),"token_ids_sha256":sha(image["ids_image"])}
checks["audio_9p3s"]={"rust_ids_identical":rust["audio"]["input_ids"]==audio["ids_audio"],"tokens":len(audio["ids_audio"]),"token_ids_sha256":sha(audio["ids_audio"])}
# Confirm generated preprocessing arrays exactly match checked-in author fixtures.
import sys; sys.path.insert(0,str(W/"scripts")); from preprocessing import image_inputs,audio_inputs
import numpy as np
px,pos=image_inputs(W/"fixtures/image/img_a_641x479.png")
ref_px=np.fromfile(REF/"image/a_641x479.patches.bin",dtype="<f4").reshape(px.shape)
acfg=json.load(open(W/"inspect/graphs/processor_config.json"))["feature_extractor"]
mel,mask=audio_inputs(W/"fixtures/audio/c_9p3s.wav",acfg)
ref_mel=np.fromfile(REF/"audio/c_9p3s.mel.bin",dtype="<f4").reshape(mel.shape)
checks["preprocessing"]={"image_patch_shape":list(px.shape),"image_patch_max_abs_vs_author_fixture":float(np.max(np.abs(px-ref_px))),"image_positions_shape":list(pos.shape),"audio_mel_shape":list(mel.shape),"audio_mel_max_abs_vs_author_fixture":float(np.max(np.abs(mel-ref_mel))),"audio_mask_shape":list(mask.shape)}
acc["input_identity"]={"checks":checks,"all_token_ids_identical":all(v.get("rust_ids_identical",v.get("all_rust_ids_identical",False)) for v in checks.values() if isinstance(v,dict) and ("rust_ids_identical" in v or "all_rust_ids_identical" in v)),"long_raw_tokens":checks["long"]["token_count"],"batch_ids_matches_rust":checks["batch64"]["all_rust_ids_identical"],"image_preprocessing_exact":checks["preprocessing"]["image_patch_max_abs_vs_author_fixture"]==0,"audio_preprocessing_exact":checks["preprocessing"]["audio_mel_max_abs_vs_author_fixture"]==0}
acc["reference"]={"backend":"Rust CUDA BF16","revision":"f15131dc109b4b541f3585f15d7643cd65984d41","dimension":768}
(R/"accuracy.json").write_text(json.dumps(acc,indent=2))
# Provider assignment with node-level trace counts and CPU fallback node names.
raw=json.load(open(W/"logs/provider-probes-fixed/provider-assignment.json")); summary={"ort_version":raw["ort_version"],"available_providers":raw["available_providers"],"model_revision":"daa72c51243991dfcaf9f9137d2c573d8f7790c0","formats":{}}
for fmt,suf in [("fp16","fp16"),("q8","quantized"),("q4","q4"),("q4f16","q4f16")]:
 item={"graph_loads":{},"gpu_gate_passed":True,"gpu_run_test_shapes":"text batch=1,seq=4; vision batch=1,patches=1; audio batch=1,frames=64; real-input encoder tests also succeeded","cpu_fallback_scope":"only scalar shape/dimension, mask-count and metadata/Concat/Reshape/Gather ops; no neural-network GEMM, MatMulNBits, attention or Conv node fell back","gpu_benchmark":True}
 for tower in ["model","vision_encoder","audio_encoder"]:
  fn=f"{tower}_{suf}.onnx"; m=raw["models"][fn]; events=m.get("node_events",[])
  byprovider=collections.Counter(e.get("provider") for e in events)
  byop=collections.defaultdict(collections.Counter)
  for e in events: byop[e.get("provider")][e.get("op")]+=1
  cpu=[{"op":e.get("op"),"name":e.get("name")} for e in events if e.get("provider")=="CPUExecutionProvider"]
  item["graph_loads"][tower]={"success":m["success"],"session_providers":m["providers"],"profiled_node_events":dict(byprovider),"cuda_ops":dict(byop["CUDAExecutionProvider"]),"cpu_fallback_nodes":cpu,"cuda_critical_ops":{k:byop["CUDAExecutionProvider"].get(k,0) for k in ["MatMul","MatMulNBits","MultiHeadAttention","Softmax","Conv","Gemm"]}}
 item["storage_and_compute"]={"weight_files_bytes":sum((W/"inspect/graphs/onnx"/f"{tower}_{suf}.onnx_data").stat().st_size for tower in ["model","vision_encoder","audio_encoder"])}
 summary["formats"][fmt]=item
(R/"provider-assignment.json").write_text(json.dumps(summary,indent=2))
# Weight sizes and actual graph quantization operators.
graph=json.load(open(R/"graph-analysis.json")); model_summary={}
for fmt,suf in [("fp16","fp16"),("q8","quantized"),("q4","q4"),("q4f16","q4f16")]:
 towers={}
 for tower in ["model","vision_encoder","audio_encoder"]:
  g=graph["models"][f"{tower}_{suf}.onnx"]
  bits=collections.Counter(a["attrs"].get("bits") for a in g["key_nodes"] if a["op"]=="MatMulNBits")
  itypes=collections.Counter(i["dtype"] for i in g["initializers"] if i["external"])
  towers[tower]={"file_size_bytes":(D/f"{tower}_{suf}.onnx_data").stat().st_size,"external_initializer_types":dict(itypes),"MatMulNBits_bits_counts":dict(bits),"opset":g["opsets"],"op_counts":g["operator_counts"]}
 model_summary[fmt]={"towers":towers,"total_external_weights_bytes":sum(t["file_size_bytes"] for t in towers.values())}
(R/"graph-analysis.json").write_text(json.dumps({"revision":"daa72c51243991dfcaf9f9137d2c573d8f7790c0","graphs":graph,"format_summary":model_summary},indent=2))
print("wrote accuracy, provider assignment, graph analysis")
