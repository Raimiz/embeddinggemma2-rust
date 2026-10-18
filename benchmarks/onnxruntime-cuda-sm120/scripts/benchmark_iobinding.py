import json,pathlib,time,statistics,sys
import numpy as np, onnxruntime as ort
from tokenizers import Tokenizer
from preprocessing import image_inputs,audio_inputs
W=pathlib.Path("/root/experiments/embeddinggemma2-ort-cuda"); F=W/"fixtures"; M=W/"inspect/graphs/onnx"; OUT=W/"results"; fmt=sys.argv[1]
suffix={"fp16":"fp16","q8":"quantized","q4":"q4","q4f16":"q4f16"}[fmt]
bench=json.load(open(F/"bench_inputs.json")); imeta=json.load(open(F/"image/ref_image.json"))["cases"][0]; ameta=next(x for x in json.load(open(F/"audio/ref_audio.json"))["cases"] if x["name"]=="c_9p3s"); acfg=json.load(open(W/"inspect/graphs/processor_config.json"))["feature_extractor"]
tok=Tokenizer.from_file(str(W/"inspect/graphs/tokenizer.json")); so=ort.SessionOptions(); so.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL; so.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL; so.intra_op_num_threads=1; so.inter_op_num_threads=1; so.log_severity_level=3; ep=[("CUDAExecutionProvider",{"device_id":0}),"CPUExecutionProvider"]
t0=time.perf_counter(); sessions={k:ort.InferenceSession(str(M/f"{k}_{suffix}.onnx"),sess_options=so,providers=ep) for k in ["model","vision_encoder","audio_encoder"]}; init=(time.perf_counter()-t0)*1000
empty=np.empty((0,512),np.float32)
def gpu_buf(shape,dtype): return ort.OrtValue.ortvalue_from_shape_and_type(shape,dtype,"cuda",0)
def run_bound(session,feeds,out_name,outbuf):
 b=session.io_binding()
 for k,v in feeds.items():
  if isinstance(v,ort.OrtValue): b.bind_ortvalue_input(k,v)
  else: b.bind_cpu_input(k,v)
 b.bind_ortvalue_output(out_name,outbuf)
 session.run_with_iobinding(b); b.synchronize_outputs(); return b.copy_outputs_to_cpu()[0]
def tokbatch(texts):
 ids=[np.asarray(tok.encode(x).ids,np.int64) for x in texts]; n=max(map(len,ids)); x=np.zeros((len(ids),n),np.int64); m=np.zeros_like(x)
 for i,v in enumerate(ids): x[i,:len(v)]=v; m[i,:len(v)]=1
 return x,m
def txtfeeds(ids,mask,img=empty,aud=empty):return {"input_ids":ids,"attention_mask":mask,"image_features":img,"video_features":empty,"audio_features":aud}
# Allocate fixed output buffers once so bindings can reuse them.
outvec=gpu_buf([1,768],np.float32); imgout=gpu_buf([266,512],np.float32); audout=gpu_buf([233,512],np.float32)
def text(text):
 ids,m=tokbatch([text]); return run_bound(sessions["model"],txtfeeds(ids,m),"sentence_embedding",outvec)
def batch():
 ids,m=tokbatch(bench["batch64"]); return run_bound(sessions["model"],txtfeeds(ids,m),"sentence_embedding",gpu_buf([64,768],np.float32))
def image():
 px,pos=image_inputs(F/"image/img_a_641x479.png")
 feat=run_bound(sessions["vision_encoder"],{"pixel_values":px,"pixel_position_ids":pos},"image_features",imgout)
 # output OrtValue persists in the preallocated GPU output; bind directly, no intermediate D2H/H2D.
 featval=imgout
 ids=np.asarray(imeta["ids_image"],np.int64)[None,:]; m=np.ones_like(ids)
 feeds=txtfeeds(ids,m,img=featval)
 return run_bound(sessions["model"],feeds,"sentence_embedding",outvec)
def audio():
 af,am=audio_inputs(F/"audio/c_9p3s.wav",acfg); feat=run_bound(sessions["audio_encoder"],{"input_features":af,"input_features_mask":am},"audio_features",audout)
 ids=np.asarray(ameta["ids_audio"],np.int64)[None,:]; m=np.ones_like(ids)
 return run_bound(sessions["model"],txtfeeds(ids,m,aud=audout),"sentence_embedding",outvec)
cases={"short":lambda:text(bench["short"]),"long":lambda:text(bench["long"]),"batch64":batch,"image":image,"audio":audio}
result={"format":fmt,"ort_version":ort.__version__,"session_init_ms":init,"notes":"I/O Binding, CPU preprocessing each repetition; host inputs bound as CPU and copied by ORT; vision/audio encoder outputs stay in GPU OrtValues between sessions; final embedding copied to CPU.","cases":{}}
for name,fn in cases.items():
 fn(); vals=[]
 for i in range(20):
  t=time.perf_counter(); y=fn(); vals.append((time.perf_counter()-t)*1000)
 result["cases"][name]={"samples_ms":vals,"median_ms":statistics.median(vals),"min_ms":min(vals),"p95_ms":float(np.percentile(vals,95)),"output_shape":list(y.shape),"l2_norm":float(np.linalg.norm(y))}
 print(fmt,name,"I/O Binding med",round(statistics.median(vals),3),"ms",flush=True)
(OUT/f"benchmark-iobinding-{fmt}.json").write_text(json.dumps(result,indent=2))
