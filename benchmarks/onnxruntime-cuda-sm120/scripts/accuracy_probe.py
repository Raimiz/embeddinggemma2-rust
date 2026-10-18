import json,pathlib,collections,math,time
import numpy as np, onnxruntime as ort
from tokenizers import Tokenizer
W=pathlib.Path("/root/experiments/embeddinggemma2-ort-cuda")
DATA=pathlib.Path("/root/experiments/embeddinggemma2-blackwell/embeddinggemma2-rust/testdata")
MODELS=W/"inspect/graphs/onnx"
REF=json.load(open(W/"results/rust-bf16-vectors.json"))
tok=Tokenizer.from_file(str(W/"inspect/graphs/tokenizer.json"))
bench=json.load(open(DATA/"bench_inputs.json"))
imgmeta=json.load(open(DATA/"image/ref_image.json"))["cases"][0]
audmeta=next(x for x in json.load(open(DATA/"audio/ref_audio.json"))["cases"] if x["name"]=="c_9p3s")
patch=np.fromfile(DATA/"image/a_641x479.patches.bin",dtype="<f4").reshape(1,-1,768)
gw,gh=imgmeta["grid_w"],imgmeta["grid_h"]
pos=np.stack(np.meshgrid(np.arange(gw,dtype=np.int64),np.arange(gh,dtype=np.int64)),axis=-1).reshape(1,-1,2)
mel_frames=audmeta["frames"]
mel=np.fromfile(DATA/"audio/c_9p3s.mel.bin",dtype="<f4").reshape(1,mel_frames,128)
melmask=np.ones((1,mel_frames),dtype=np.bool_)
empty=np.empty((0,512),dtype=np.float32)
formats={"fp16":"fp16","q8":"quantized","q4":"q4","q4f16":"q4f16"}

def tokenize(text): return np.array(tok.encode(text).ids,dtype=np.int64)
def pad_batch(texts):
 ids=[tokenize(x) for x in texts]; m=max(map(len,ids)); a=np.zeros((len(ids),m),np.int64); mask=np.zeros_like(a)
 for i,x in enumerate(ids): a[i,:len(x)]=x; mask[i,:len(x)]=1
 return a,mask

def feed_text(ids,mask,im=empty,au=empty): return {"input_ids":ids,"attention_mask":mask,"image_features":im,"video_features":empty,"audio_features":au}
def norm(v): return float(np.linalg.norm(v))
def compare(a,b):
 a=np.asarray(a,dtype=np.float64); b=np.asarray(b,dtype=np.float64)
 return {"shape":list(a.shape),"max_abs":float(np.max(np.abs(a-b))),"rmse":float(np.sqrt(np.mean((a-b)**2))),"cosine":float(np.sum(a*b)/(np.linalg.norm(a)*np.linalg.norm(b))),"ort_l2_norm":norm(a),"rust_l2_norm":norm(b)}

def run(model,feed): return model.run(["sentence_embedding"],feed)[0]

out={"ort_version":ort.__version__,"revision":"daa72c51243991dfcaf9f9137d2c573d8f7790c0","formats":{}}
for fmt,suffix in formats.items():
 so=ort.SessionOptions(); so.log_severity_level=3
 ep=[("CUDAExecutionProvider",{"device_id":0}),"CPUExecutionProvider"]
 sess={tower:ort.InferenceSession(str(MODELS/f"{tower}_{suffix}.onnx"),sess_options=so,providers=ep) for tower in ["model","vision_encoder","audio_encoder"]}
 # independent text inputs and packed batch
 singles={}
 for label in ["short","long"]:
  ids=tokenize(bench[label])[None,:]; mask=np.ones_like(ids)
  singles[label]=run(sess["model"],feed_text(ids,mask))[0]
 ids,mask=pad_batch(bench["batch64"])
 batch_emb=run(sess["model"],feed_text(ids,mask))
 # Multimodal full path; input tensors are author preprocessing fixtures, run encoder then text graph.
 image_feat=sess["vision_encoder"].run(None,{"pixel_values":patch,"pixel_position_ids":pos})[0]
 image_ids=np.asarray(imgmeta["ids_image"],dtype=np.int64)[None,:]; image_mask=np.ones_like(image_ids)
 image_emb=run(sess["model"],feed_text(image_ids,image_mask,im=image_feat))[0]
 audio_feat=sess["audio_encoder"].run(None,{"input_features":mel,"input_features_mask":melmask})[0]
 audio_ids=np.asarray(audmeta["ids_audio"],dtype=np.int64)[None,:]; audio_mask=np.ones_like(audio_ids)
 audio_emb=run(sess["model"],feed_text(audio_ids,audio_mask,au=audio_feat))[0]
 pred={"short":singles["short"],"long":singles["long"],"batch64":batch_emb,"image":image_emb,"audio":audio_emb}
 refs={"short":np.array(REF["short"]["embedding"]),"long":np.array(REF["long"]["embedding"]),"batch64":np.array(REF["batch64"]["embeddings"]),"image":np.array(REF["image"]["embedding"]),"audio":np.array(REF["audio"]["embedding"])}
 metrics={k:compare(pred[k],refs[k]) for k in pred}
 metrics["input_tokens"]={"short":int(len(tokenize(bench["short"]))),"long":int(len(tokenize(bench["long"]))),"batch64_first":int(len(tokenize(bench["batch64"][0]))),"batch64_shape":list(ids.shape),"image_ids":len(imgmeta["ids_image"]),"audio_ids":len(audmeta["ids_audio"])}
 # Compare all pairwise cosine scores, then nearest-neighbor top-1 agreement.
 a=pred["batch64"]; b=refs["batch64"]
 an=a/np.linalg.norm(a,axis=1,keepdims=True); bn=b/np.linalg.norm(b,axis=1,keepdims=True)
 ac=an@an.T; bc=bn@bn.T; np.fill_diagonal(ac,-np.inf); np.fill_diagonal(bc,-np.inf)
 top_a=np.argmax(ac,axis=1); top_b=np.argmax(bc,axis=1)
 metrics["batch64_similarity"]={"max_pairwise_cosine_delta":float(np.max(np.abs(ac[np.isfinite(ac)]-bc[np.isfinite(bc)]))),"mean_pairwise_cosine_delta":float(np.mean(np.abs(ac[np.isfinite(ac)]-bc[np.isfinite(bc)]))),"top1_neighbor_agreement":float(np.mean(top_a==top_b)),"top1_indices_ort":top_a.tolist(),"top1_indices_rust":top_b.tolist()}
 out["formats"][fmt]={"metrics":metrics,"embeddings":{k:np.asarray(v).tolist() for k,v in pred.items()},"execution_providers":{k:v.get_providers() for k,v in sess.items()}}
 print(fmt,{k:(round(v["cosine"],7),round(v["max_abs"],6)) for k,v in metrics.items() if isinstance(v,dict) and "cosine" in v},"batch-top1",metrics["batch64_similarity"]["top1_neighbor_agreement"],flush=True)
 del sess
path=W/"results/accuracy-probe.json"; path.write_text(json.dumps(out,indent=2))
