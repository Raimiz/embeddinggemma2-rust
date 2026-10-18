import json,pathlib,collections
import numpy as np, onnxruntime as ort
W=pathlib.Path("/root/experiments/embeddinggemma2-ort-cuda")
REF=pathlib.Path("/root/experiments/embeddinggemma2-blackwell/embeddinggemma2-rust/testdata")
MODEL=W/"inspect/graphs/onnx"
R={"ort":ort.__version__,"providers":{}}
formats={"fp16":"fp16","q8":"quantized","q4":"q4","q4f16":"q4f16"}

def cos(a,b): return float(np.dot(a.ravel(),b.ravel())/(np.linalg.norm(a)*np.linalg.norm(b)))
for mode,suffix in formats.items():
 for tower in ["vision_encoder","audio_encoder"]:
  p=MODEL/f"{tower}_{suffix}.onnx"
  so=ort.SessionOptions(); so.log_severity_level=3; so.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
  s=ort.InferenceSession(str(p),sess_options=so,providers=["CUDAExecutionProvider","CPUExecutionProvider"])
  if tower=="vision_encoder":
   meta=json.load(open(REF/"image/ref_image.json"))["cases"][0]
   raw=np.fromfile(REF/"image/a_641x479.patches.bin",dtype="<f4").reshape(1,-1,768)
   gw,gh=meta["grid_w"],meta["grid_h"]
   pos=np.stack(np.meshgrid(np.arange(gw,dtype=np.int64),np.arange(gh,dtype=np.int64)),axis=-1).reshape(-1,2)
   if len(pos)!=raw.shape[1]: raise ValueError((len(pos),raw.shape))
   out=s.run(None,{"pixel_values":raw,"pixel_position_ids":pos[None]})[0]
   expected=np.fromfile(REF/"image/a_641x479.soft.bin",dtype="<f4").reshape(-1,512)
  else:
   frames=929
   raw=np.fromfile(REF/"audio/c_9p3s.mel.bin",dtype="<f4").reshape(1,frames,128)
   mask=np.ones((1,frames),dtype=np.bool_)
   out=s.run(None,{"input_features":raw,"input_features_mask":mask})[0]
   expected=np.fromfile(REF/"audio/c_9p3s.soft.bin",dtype="<f4").reshape(-1,512)
  a=np.asarray(out).squeeze().astype(np.float32)
  result={"shape":list(np.asarray(out).shape),"expected_shape":list(expected.shape),"max_abs":float(np.max(np.abs(a-expected))),"cosine":cos(a,expected),"cuda":s.get_providers()}
  R["providers"][f"{mode}_{tower}"]=result
  print(mode,tower,result,flush=True)
  del s
(W/"results/encoder-input-checks.json").write_text(json.dumps(R,indent=2))
