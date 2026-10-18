import argparse,json,pathlib,time,statistics
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from preprocessing import image_inputs,audio_inputs

ap=argparse.ArgumentParser(); ap.add_argument("format",choices=["fp16","q8","q4","q4f16"]); ap.add_argument("run",type=int); ap.add_argument("--iters",type=int,default=20); a=ap.parse_args()
W=pathlib.Path("/root/experiments/embeddinggemma2-ort-cuda"); F=W/"fixtures"; M=W/"inspect/graphs/onnx"; OUT=W/"results"
suffix={"fp16":"fp16","q8":"quantized","q4":"q4","q4f16":"q4f16"}[a.format]
bench=json.load(open(F/"bench_inputs.json")); imeta=json.load(open(F/"image/ref_image.json"))["cases"][0]; ameta=next(x for x in json.load(open(F/"audio/ref_audio.json"))["cases"] if x["name"]=="c_9p3s")
acfg=json.load(open(W/"inspect/graphs/processor_config.json"))["feature_extractor"]
tok=Tokenizer.from_file(str(W/"inspect/graphs/tokenizer.json"))
so=ort.SessionOptions(); so.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL; so.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL; so.intra_op_num_threads=1; so.inter_op_num_threads=1; so.log_severity_level=3
providers=[("CUDAExecutionProvider",{"device_id":0}),"CPUExecutionProvider"]
sessions={}; init={}; t_all=time.perf_counter()
for tower in ["model","vision_encoder","audio_encoder"]:
 t=time.perf_counter(); sessions[tower]=ort.InferenceSession(str(M/f"{tower}_{suffix}.onnx"),sess_options=so,providers=providers); init[tower]=(time.perf_counter()-t)*1000
init_total=(time.perf_counter()-t_all)*1000
empty=np.empty((0,512),dtype=np.float32)

def pad(texts):
 ids=[np.asarray(tok.encode(x).ids,dtype=np.int64) for x in texts]
 n=max(map(len,ids)); x=np.zeros((len(ids),n),dtype=np.int64); mask=np.zeros_like(x)
 for i,v in enumerate(ids): x[i,:len(v)]=v; mask[i,:len(v)]=1
 return x,mask

def text_feed(ids,mask,image=empty,audio=empty): return {"input_ids":ids,"attention_mask":mask,"image_features":image,"video_features":empty,"audio_features":audio}
def run_text(text):
 t0=time.perf_counter(); ids,mask=pad([text]); prep=(time.perf_counter()-t0)*1000
 t1=time.perf_counter(); y=sessions["model"].run(["sentence_embedding"],text_feed(ids,mask))[0]; rt=(time.perf_counter()-t1)*1000
 return y,prep,rt

def run_batch():
 t0=time.perf_counter(); ids,mask=pad(bench["batch64"]); prep=(time.perf_counter()-t0)*1000
 t1=time.perf_counter(); y=sessions["model"].run(["sentence_embedding"],text_feed(ids,mask))[0]; rt=(time.perf_counter()-t1)*1000
 return y,prep,rt

def run_image():
 t0=time.perf_counter(); px,pos=image_inputs(F/"image/img_a_641x479.png"); ids=np.asarray(imeta["ids_image"],dtype=np.int64)[None,:]; mask=np.ones_like(ids); prep=(time.perf_counter()-t0)*1000
 t1=time.perf_counter(); feat=sessions["vision_encoder"].run(None,{"pixel_values":px,"pixel_position_ids":pos})[0]; vis=(time.perf_counter()-t1)*1000
 t2=time.perf_counter(); y=sessions["model"].run(["sentence_embedding"],text_feed(ids,mask,image=feat))[0]; txt=(time.perf_counter()-t2)*1000
 return y,prep,vis,txt

def run_audio():
 t0=time.perf_counter(); feats,maskf=audio_inputs(F/"audio/c_9p3s.wav",acfg); ids=np.asarray(ameta["ids_audio"],dtype=np.int64)[None,:]; mask=np.ones_like(ids); prep=(time.perf_counter()-t0)*1000
 t1=time.perf_counter(); feat=sessions["audio_encoder"].run(None,{"input_features":feats,"input_features_mask":maskf})[0]; aud=(time.perf_counter()-t1)*1000
 t2=time.perf_counter(); y=sessions["model"].run(["sentence_embedding"],text_feed(ids,mask,audio=feat))[0]; txt=(time.perf_counter()-t2)*1000
 return y,prep,aud,txt

cases={"short":{"fn":lambda:run_text(bench["short"]),"keys":["preprocess_ms","text_session_run_ms"]},"long":{"fn":lambda:run_text(bench["long"]),"keys":["preprocess_ms","text_session_run_ms"]},"batch64":{"fn":run_batch,"keys":["preprocess_ms","text_session_run_ms"]},"image":{"fn":run_image,"keys":["preprocess_ms","vision_session_run_ms","text_session_run_ms"]},"audio":{"fn":run_audio,"keys":["preprocess_ms","audio_session_run_ms","text_session_run_ms"]}}
res={"format":a.format,"run":a.run,"iterations":a.iters,"ort_version":ort.__version__,"session_init_ms":init_total,"session_init_by_model_ms":init,"providers":{k:s.get_providers() for k,s in sessions.items()},"provider_options":{k:s.get_provider_options() for k,s in sessions.items()},"cases":{}}
for name,c in cases.items():
 c["fn"]() # warm-up
 samples=[]; output_shape=None; output_l2=None
 for _ in range(a.iters):
  start=time.perf_counter(); val=c["fn"](); end=time.perf_counter()
  y,*parts=val; output_shape=list(np.asarray(y).shape); output_l2=float(np.linalg.norm(y))
  samples.append({"e2e_ms":(end-start)*1000,**dict(zip(c["keys"],parts))})
 e2e=[x["e2e_ms"] for x in samples]; srt=sorted(e2e)
 res["cases"][name]={"samples":samples,"summary":{"median_ms":statistics.median(e2e),"min_ms":min(e2e),"p95_ms":float(np.percentile(e2e,95)),"max_ms":max(e2e)},"output_shape":output_shape,"output_l2_norm_last_sample":output_l2}
 print(a.format,name,"median",round(statistics.median(e2e),3),"min",round(min(e2e),3),"p95",round(float(np.percentile(e2e,95)),3),flush=True)
path=OUT/f"benchmark-{a.format}-run{a.run}.json"; path.write_text(json.dumps(res,indent=2))
print("saved",path,"session init ms",round(init_total,1))
