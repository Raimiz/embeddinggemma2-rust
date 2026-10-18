import collections,gc,json,pathlib,sys,traceback
import numpy as np
import onnxruntime as ort
root=pathlib.Path(sys.argv[1]); out=pathlib.Path(sys.argv[2]); out.mkdir(parents=True,exist_ok=True)
result={"ort_version":ort.__version__,"available_providers":ort.get_available_providers(),"models":{}}
def feeds_for(s):
 f={}
 for x in s.get_inputs():
  n=x.name
  if n=="input_ids": f[n]=np.array([[2,100,101,1]],dtype=np.int64)
  elif n=="attention_mask": f[n]=np.ones((1,4),dtype=np.int64)
  elif n in ("image_features","video_features","audio_features"): f[n]=np.empty((0,512),dtype=np.float32)
  elif n=="pixel_values": f[n]=np.zeros((1,1,768),dtype=np.float32)
  elif n=="pixel_position_ids": f[n]=np.zeros((1,1,2),dtype=np.int64)
  elif n=="input_features": f[n]=np.zeros((1,64,128),dtype=np.float32)
  elif n=="input_features_mask": f[n]=np.ones((1,64),dtype=np.bool_)
  else: raise ValueError("unknown input "+repr((n,x.type,x.shape)))
 return f
for p in sorted(root.glob("onnx/*.onnx")):
 name=p.name; logpath=out/(name+".ort.log"); prof=out/(name+".profile.json")
 rec={"file":str(p),"success":False}
 try:
  so=ort.SessionOptions(); so.enable_profiling=True; so.profile_file_prefix=str(out/(name+".trace")); so.log_severity_level=3; so.log_verbosity_level=0
  so.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
  # Verbose session creation is captured per model for operator placement diagnostics.
  import contextlib,io
  buf=io.StringIO()
  with contextlib.redirect_stderr(buf),contextlib.redirect_stdout(buf):
   s=ort.InferenceSession(str(p),sess_options=so,providers=[("CUDAExecutionProvider",{"device_id":0}),"CPUExecutionProvider"])
   inp=feeds_for(s); outs=s.run(None,inp)
   actual=s.get_providers(); options=s.get_provider_options()
   end=s.end_profiling()
  rec.update({"success":True,"providers":actual,"provider_options":options,"inputs":[{"name":x.name,"type":x.type,"shape":x.shape} for x in s.get_inputs()],"outputs":[{"name":x.name,"shape":list(a.shape),"dtype":str(a.dtype)} for x,a in zip(s.get_outputs(),outs)],"profile":end})
  events=json.load(open(end)); histogram=collections.Counter(e.get("args",{}).get("provider","?" ) for e in events if e.get("cat")=="Node")
  rec["node_events_by_provider"]=dict(histogram)
  rec["node_events"]=[{"name":e.get("name"),"provider":e.get("args",{}).get("provider"),"op":e.get("args",{}).get("op_name"),"dur_us":e.get("dur")} for e in events if e.get("cat")=="Node"]
  logpath.write_text(buf.getvalue())
  del s,outs,inp; gc.collect()
 except Exception as e:
  rec["error"]=repr(e); rec["traceback"]=traceback.format_exc(); logpath.write_text((buf.getvalue() if "buf" in locals() else "")+"\nERROR: "+traceback.format_exc())
 result["models"][name]=rec
 out.joinpath("provider-assignment.partial.json").write_text(json.dumps(result,indent=2))
 print(name,"OK" if rec["success"] else "FAIL",rec.get("node_events_by_provider"),rec.get("error",""),flush=True)
result["complete"]=True
(out/"provider-assignment.json").write_text(json.dumps(result,indent=2))
