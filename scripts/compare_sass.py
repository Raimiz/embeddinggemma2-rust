from pathlib import Path
import json, re, hashlib
root=Path('/root/experiments/embeddinggemma2-sm120-compile/artifacts/disassembly')
files={k:root/f'{k}.sass' for k in ['compute_80_jit','compute_120_jit','sm_120_cubin']}
def parse(p):
 d={}; cur=None
 for line in p.read_text(errors='replace').splitlines():
  if 'Function : ' in line:
   cur=line.split('Function : ',1)[1].strip(); d[cur]=[]; continue
  if cur and '*/' in line and ';' in line:
   x=line.split('*/',1)[1].split(';',1)[0].strip()
   if x: d[cur].append(x)
 return d
F={k:parse(v) for k,v in files.items()}
def compare(a,b):
 x,y=F[a],F[b]; names=sorted(set(x)|set(y)); dif=[]
 for n in names:
  xx=x.get(n,[]); yy=y.get(n,[]); L=max(len(xx),len(yy))
  dd=sum((xx[i] if i<len(xx) else None)!=(yy[i] if i<len(yy) else None) for i in range(L))
  if dd: dif.append({'function':n,'instructions_a':len(xx),'instructions_b':len(yy),'differing_instruction_positions':dd})
 return {'mode_a':a,'mode_b':b,'function_count_a':len(x),'function_count_b':len(y),'different_function_count':len(dif),'differences':dif}
patterns={'HMMA':r'\bHMMA\.', 'LDGSTS':r'\bLDGSTS\.', 'TCGEN05':r'\bTCGEN05', 'WGMMA':r'\bWGMMA', 'CP_ASYNC':r'\bCP\.ASYNC'}
counts={}
for mode,d in F.items():
 s='\n'.join(i for seq in d.values() for i in seq)
 counts[mode]={name:len(re.findall(pattern,s)) for name,pattern in patterns.items()}
obj={'method':'cuobjdump --dump-sass; extract instruction mnemonics+operands by function, excluding addresses, binary encodings, target/header metadata','targets':{k:{'function_count':len(F[k]),'sass_instruction_lines':sum(map(len,F[k].values())),'disassembly_sha256':hashlib.sha256(files[k].read_bytes()).hexdigest(),'resource_report':str(root/f'{k}.resources.txt'),'elf_report':str(root/f'{k}.elf.txt')} for k in F},'opcode_counts':counts,'comparisons':[compare('compute_120_jit','sm_120_cubin'),compare('compute_80_jit','compute_120_jit')],'important_kernels':{n:{k:len(v.get(n,[])) for k,v in F.items()} for n in ['flash_attn_d64_bb','flash_attn_d256_bb','flash_attn_d256_packed_bb','flash_attn_d512_bb']}}
Path('/root/experiments/embeddinggemma2-sm120-compile/results/sass-comparison.json').write_text(json.dumps(obj,indent=2))
print(json.dumps(obj,indent=2))
