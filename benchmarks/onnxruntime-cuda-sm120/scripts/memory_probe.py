import json,pathlib,time,threading,subprocess,os,resource,gc
import numpy as np, onnxruntime as ort
from tokenizers import Tokenizer
from preprocessing import image_inputs,audio_inputs
W=pathlib.Path("/root/experiments/embeddinggemma2-ort-cuda"); F=W/"fixtures"; M=W/"inspect/graphs/onnx"; fmt=__import__("sys").argv[1]
suffix={"fp16":"fp16","q8":"quantized","q4":"q4","q4f16":"q4f16"}[fmt]
bench=json.load(open(F/"bench_inputs.json")); imeta=json.load(open(F/"image/ref_image.json"))["cases"][0]; ameta=next(x for x in json.load(open(F/"audio/ref_audio.json"))["cases"] if x["name"]=="c_9p3s")
acfg=json.load(open(W/"inspect/graphs/processor_config.json"))["feature_extractor"]; tok=Tokenizer.from_file(str(W/"inspect/graphs/tokenizer.json"))
so=ort.SessionOptions(); so.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL; so.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL; so.intra_op_num_threads=1; so.inter_op_num_threads=1; so.log_severity_level=3; ep=[("CUDAExecutionProvider",{"device_id":0}),"CPUExecutionProvider"]
rows=[]; stop=False; pid=os.getpid()
def sample():
 try: gpu=int(subprocess.check_output(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],text=True).splitlines()[0])
 except: gpu=None
 try:
  st=pathlib.Path(f"/proc/{pid}/status").read_text(); rss=int(next(x for x in st.splitlines() if x.startswith("VmRSS:")).split()[1])
 except: rss=None
 try:
  apps=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,used_memory","--format=csv,noheader,nounits"],text=True).strip().splitlines()
  proc_gpu=next((int(x.split(",")[1].strip()) for x in apps if int(x.split(",")[0].strip())==pid),None)
 except Exception: proc_gpu=None
 return {"t":time.time(),"gpu_used_mib":gpu,"process_gpu_used_mib":proc_gpu,"process_rss_kib":rss}
def monitor():
 while not stop:
  r=sample(); r["phase"]=phase[0]; rows.append(r); time.sleep(.2)
phase=["import/init"]
thread=threading.Thread(target=monitor,daemon=True); thread.start()
def mark(k):
 phase[0]=k; time.sleep(.25); print(k,sample(),flush=True)
def text_run(text):
 ids=np.asarray(tok.encode(text).ids,np.int64)[None,:]; mask=np.ones_like(ids); empty=np.empty((0,512),np.float32)
 return sessions["model"].run(["sentence_embedding"],{"input_ids":ids,"attention_mask":mask,"image_features":empty,"video_features":empty,"audio_features":empty})
start=time.perf_counter(); sessions={}
for tower in ["model"]:
 sessions[tower]=ort.InferenceSession(str(M/f"{tower}_{suffix}.onnx"),sess_options=so,providers=ep)
print("text session ms",(time.perf_counter()-start)*1000); mark("text-loaded-and-run"); text_run(bench["short"]); mark("after-text-run")
start=time.perf_counter(); sessions["vision_encoder"]=ort.InferenceSession(str(M/f"vision_encoder_{suffix}.onnx"),sess_options=so,providers=ep); print("vision session ms",(time.perf_counter()-start)*1000)
px,pos=image_inputs(F/"image/img_a_641x479.png"); imout=sessions["vision_encoder"].run(None,{"pixel_values":px,"pixel_position_ids":pos})[0]
ids=np.asarray(imeta["ids_image"],np.int64)[None,:]; empty=np.empty((0,512),np.float32); sessions["model"].run(["sentence_embedding"],{"input_ids":ids,"attention_mask":np.ones_like(ids),"image_features":imout,"video_features":empty,"audio_features":empty}); mark("after-first-vision")
start=time.perf_counter(); sessions["audio_encoder"]=ort.InferenceSession(str(M/f"audio_encoder_{suffix}.onnx"),sess_options=so,providers=ep); print("audio session ms",(time.perf_counter()-start)*1000)
af,am=audio_inputs(F/"audio/c_9p3s.wav",acfg); au=sessions["audio_encoder"].run(None,{"input_features":af,"input_features_mask":am})[0]
ids=np.asarray(ameta["ids_audio"],np.int64)[None,:]; sessions["model"].run(["sentence_embedding"],{"input_ids":ids,"attention_mask":np.ones_like(ids),"image_features":empty,"video_features":empty,"audio_features":au}); mark("after-first-audio")
# One complete pass of all five scenarios with all sessions retained.
text_run(bench["short"]); text_run(bench["long"])
seq=[np.asarray(tok.encode(x).ids,np.int64) for x in bench["batch64"]]; mx=max(map(len,seq)); x=np.zeros((64,mx),np.int64); mk=np.zeros_like(x)
for i,v in enumerate(seq): x[i,:len(v)]=v; mk[i,:len(v)]=1
sessions["model"].run(["sentence_embedding"],{"input_ids":x,"attention_mask":mk,"image_features":empty,"video_features":empty,"audio_features":empty})
px,pos=image_inputs(F/"image/img_a_641x479.png"); fi=sessions["vision_encoder"].run(None,{"pixel_values":px,"pixel_position_ids":pos})[0]; sessions["model"].run(["sentence_embedding"],{"input_ids":ids,"attention_mask":np.ones_like(ids),"image_features":fi,"video_features":empty,"audio_features":empty})
af,am=audio_inputs(F/"audio/c_9p3s.wav",acfg); fa=sessions["audio_encoder"].run(None,{"input_features":af,"input_features_mask":am})[0]; sessions["model"].run(["sentence_embedding"],{"input_ids":ids,"attention_mask":np.ones_like(ids),"image_features":empty,"video_features":empty,"audio_features":fa}); mark("full-five-scenario-pass")
stop=True; thread.join(timeout=1)
result={"format":fmt,"ort_version":ort.__version__,"session_providers":{k:v.get_providers() for k,v in sessions.items()},"phase_samples":rows,"peak_global_gpu_used_mib":max((x["gpu_used_mib"] or 0) for x in rows),"peak_process_gpu_used_mib":max((x["process_gpu_used_mib"] or 0) for x in rows),"peak_process_rss_mib":max((x["process_rss_kib"] or 0) for x in rows)/1024,"ru_maxrss_mib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
(W/"results"/f"memory-{fmt}.json").write_text(json.dumps(result,indent=2)); print("PEAK",result["peak_global_gpu_used_mib"],"MiB GPU;",result["peak_process_rss_mib"],"MiB RSS",flush=True)
