import pathlib,json,time,sys,traceback
import numpy as np, onnxruntime as ort
from tokenizers import Tokenizer
W=pathlib.Path("/root/experiments/embeddinggemma2-ort-cuda"); M=W/"inspect/graphs/onnx"; F=W/"fixtures"
fmt=sys.argv[1]; suffix={"fp16":"fp16","q8":"quantized","q4":"q4","q4f16":"q4f16"}[fmt]
bench=json.load(open(F/"bench_inputs.json")); tok=Tokenizer.from_file(str(W/"inspect/graphs/tokenizer.json")); ids=np.asarray(tok.encode(bench["short"]).ids,np.int64)[None,:]; mask=np.ones_like(ids); empty=np.empty((0,512),np.float32)
result={"format":fmt,"ort_version":ort.__version__}
for graph in [False,True]:
 key="cuda_graph" if graph else "io_binding"
 try:
  so=ort.SessionOptions(); so.log_severity_level=3; so.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
  opts={"device_id":0}
  if graph: opts["enable_cuda_graph"]="1"
  s=ort.InferenceSession(str(M/f"model_{suffix}.onnx"),sess_options=so,providers=[("CUDAExecutionProvider",opts),"CPUExecutionProvider"])
  vals={"input_ids":ids,"attention_mask":mask,"image_features":empty,"video_features":empty,"audio_features":empty}
  gpu={k:ort.OrtValue.ortvalue_from_numpy(v,"cuda",0) for k,v in vals.items()}
  out=ort.OrtValue.ortvalue_from_shape_and_type([1,768],np.float32,"cuda",0)
  b=s.io_binding()
  for k,v in gpu.items(): b.bind_ortvalue_input(k,v)
  b.bind_ortvalue_output("sentence_embedding",out)
  samples=[]
  for i in range(12):
   t=time.perf_counter(); s.run_with_iobinding(b); b.synchronize_outputs(); arr=b.copy_outputs_to_cpu()[0]; samples.append((time.perf_counter()-t)*1000)
  result[key]={"success":True,"providers":s.get_providers(),"samples_ms":samples,"median_ms":float(np.median(samples[2:])) ,"embedding_norm":float(np.linalg.norm(arr))}
  print(fmt,key,"OK",result[key]["median_ms"],flush=True)
 except Exception as e:
  result[key]={"success":False,"error":repr(e),"traceback":traceback.format_exc()}
  print(fmt,key,"FAIL",repr(e),flush=True)
(W/"results"/f"optimization-probe-{fmt}.json").write_text(json.dumps(result,indent=2))
