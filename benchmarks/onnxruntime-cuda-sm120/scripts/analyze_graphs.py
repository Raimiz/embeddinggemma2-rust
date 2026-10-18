import collections, json, pathlib, sys
import onnx
from onnx import TensorProto
root=pathlib.Path(sys.argv[1]); out=pathlib.Path(sys.argv[2]); result={"models":{}}

def dims(v):
 t=v.type.tensor_type
 return {"elem_type":TensorProto.DataType.Name(t.elem_type),"shape":[d.dim_value if d.HasField("dim_value") else (d.dim_param or "?") for d in t.shape.dim]}
def attr_value(a):
 try: return onnx.helper.get_attribute_value(a)
 except Exception: return str(a)
for p in sorted(root.glob("onnx/*.onnx")):
 m=onnx.load(str(p),load_external_data=False)
 g=m.graph
 counts=collections.Counter(n.op_type for n in g.node)
 initializers=[]
 for t in g.initializer:
  item={"name":t.name,"dtype":TensorProto.DataType.Name(t.data_type),"shape":list(t.dims),"external":t.data_location==TensorProto.EXTERNAL}
  if t.external_data: item["external_data"]={e.key:e.value for e in t.external_data}
  initializers.append(item)
 nodes=[]
 for n in g.node:
  if n.op_type in ("MatMulNBits","MatMulInteger","DynamicQuantizeLinear","QLinearMatMul","DequantizeLinear","Cast","MatMul","Gemm","Attention","LayerNormalization","Conv","ConvTranspose","GroupQueryAttention","RotaryEmbedding"):
   nodes.append({"name":n.name,"op":n.op_type,"domain":n.domain,"inputs":list(n.input),"outputs":list(n.output),"attrs":{a.name:attr_value(a) for a in n.attribute}})
 result["models"][p.name]={"ir_version":m.ir_version,"opsets":[{"domain":x.domain,"version":x.version} for x in m.opset_import],"node_count":len(g.node),"operator_counts":dict(sorted(counts.items())),"inputs":[{"name":x.name,**dims(x)} for x in g.input],"outputs":[{"name":x.name,**dims(x)} for x in g.output],"value_info":[{"name":x.name,**dims(x)} for x in g.value_info],"initializers":initializers,"key_nodes":nodes}
out.write_text(json.dumps(result,indent=2,default=lambda x:x.tolist() if hasattr(x,"tolist") else str(x)))
print("wrote",out,"models",len(result["models"]))
for name,m in result["models"].items():
 print(name,"ops",m["operator_counts"],"inputs",m["inputs"],"outputs",m["outputs"])
