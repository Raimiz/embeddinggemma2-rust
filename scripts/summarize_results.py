#!/usr/bin/env python3
import csv, hashlib, json, math, os, pathlib, re, statistics, subprocess, time
EXP=pathlib.Path(os.environ.get('EXP','/root/experiments/embeddinggemma2-sm120-compile'))
BASE=pathlib.Path('/root/experiments/embeddinggemma2-blackwell')
R=EXP/'results'; A=EXP/'artifacts'; W=EXP/'worktree'
MODES=['compute_80','compute_120','sm_120']
CASES={'short (17 tok)':'short','long (2208 tok)':'long','batch of 64 short docs':'batch64','image 641x479':'image','audio 9.3 s':'audio'}
def jload(p): return json.load(open(p))
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
def p95(xs):
 z=sorted(xs); return z[max(0,math.ceil(.95*len(z))-1)]
def cos(a,b):
 na=math.sqrt(sum(x*x for x in a)); nb=math.sqrt(sum(x*x for x in b))
 return sum(x*y for x,y in zip(a,b))/(na*nb)
def vector_metric(a,b): return {'exact_equal':a==b,'dimensions':len(a),'cosine':cos(a,b),'max_abs':max((abs(x-y) for x,y in zip(a,b)),default=0.0)}
# Full 90-check verify gates and exact pairwise output comparisons.
verify={}
verify_logs={
 'compute_80':R/'verify-compute_80.log',
 'compute_120':R/'verify-compute_120.log',
 'sm_120':R/'verify-sm_120.log',
}
for mode,p in verify_logs.items():
 s=p.read_text(errors='replace'); counts=re.findall(r'(\d+) checks, (\d+) failed',s)
 check_groups={}
 for k,v in re.findall(r'^\[(text|image|audio|video|extra)\] (\d+) checks',s,re.M): check_groups[k]=check_groups.get(k,0)+int(v)
 tolerance=re.search(r'tolerance profile (.+)',s)
 verify[mode]={'checks':int(counts[-1][0]) if counts else 0,'failed':int(counts[-1][1]) if counts else None,'passed':bool(counts and counts[-1][1]=='0' and counts[-1][0]=='90'),'groups':check_groups,'tolerance_profile':tolerance.group(1) if tolerance else None,'log':str(p.relative_to(EXP))}
# Direct same-process-output dumps, one run per mode, all values compared to A.
outputs={m:jload(A/f'embeddings-{m}.json') for m in MODES}
direct={}
for mode in ['compute_120','sm_120']:
 cm={case:vector_metric(outputs['compute_80'][case],outputs[mode][case]) for case in ['short','long','image','audio']}
 rows=[vector_metric(x,y) for x,y in zip(outputs['compute_80']['batch64'],outputs[mode]['batch64'])]
 cm['batch64']={'documents':len(rows),'exact_documents':sum(x['exact_equal'] for x in rows),'cosine_mean':statistics.mean(x['cosine'] for x in rows),'cosine_min':min(x['cosine'] for x in rows),'max_abs_max':max(x['max_abs'] for x in rows)}
 direct[mode]={'vs_compute_80':cm,'all_cases_exact':all(x.get('exact_equal',x.get('exact_documents')==64) for x in cm.values())}
verify['direct_embedding_comparison']=direct
(R/'verify.json').write_text(json.dumps(verify,indent=2))
# Benchmark distributions: per-run summaries, all 60 raw samples per case, deltas to A.
bench={}
for mode in MODES:
 bench[mode]={}
 for label,key in CASES.items():
  runs=[jload(R/f'benchmark-{mode}-run{i}.json')[label] for i in (1,2,3)]
  samples=[t for x in runs for t in x['samples_ms']]
  medians=[x['median_ms'] for x in runs]
  bench[mode][key]={'runs':3,'samples_per_run':[len(x['samples_ms']) for x in runs],'run_medians_ms':medians,'median_of_run_medians_ms':statistics.median(medians),'run_median_range_ms':[min(medians),max(medians)],'min_all_samples_ms':min(samples),'max_all_samples_ms':max(samples),'p95_all_60_samples_ms':p95(samples),'run_p95_ms':[x['p95_ms'] for x in runs],'run_min_ms':[x['min_ms'] for x in runs]}
for mode in ['compute_120','sm_120']:
 for key in CASES.values():
  v=bench[mode][key]['median_of_run_medians_ms']; a=bench['compute_80'][key]['median_of_run_medians_ms']
  bench[mode][key]['delta_vs_compute_80_ms']=v-a
  bench[mode][key]['delta_vs_compute_80_percent']=100*(v-a)/a
# Per-run model-load times from benchmark process logs.
load_re=re.compile(r'model loaded on cuda:0 \(bf16\) in ([0-9.]+) ms')
startup={}
for mode in MODES:
 vals=[]
 for i in (1,2,3):
  txt=(R/f'benchmark-{mode}-run{i}.log').read_text(errors='replace'); m=load_re.search(txt); vals.append(float(m.group(1)) if m else None)
 startup[mode]={'model_load_ms_per_run':vals,'median_ms':statistics.median(x for x in vals if x is not None)}
# Sensor samples from nvidia-smi during each run.
telemetry={}
for mode in MODES:
 telemetry[mode]={}
 for i in (1,2,3):
  p=R/f'gpu-{mode}-run{i}.csv'; rows=[]
  for row in csv.reader(open(p)):
   if len(row)<8: continue
   try: rows.append({'timestamp':row[0].strip(),'temperature_c':float(row[1].strip()),'sm_clock_mhz':float(row[2].strip().split()[0]),'memory_clock_mhz':float(row[3].strip().split()[0]),'power_w':float(row[4].strip().split()[0]),'pstate':row[5].strip(),'utilization_percent':float(row[6].strip().split()[0]),'global_vram_mib':float(row[7].strip().split()[0])})
   except ValueError: pass
  telemetry[mode][f'run{i}']={'samples':rows,'sample_count':len(rows),'temperature_range_c':[min(x['temperature_c'] for x in rows),max(x['temperature_c'] for x in rows)] if rows else None,'sm_clock_range_mhz':[min(x['sm_clock_mhz'] for x in rows),max(x['sm_clock_mhz'] for x in rows)] if rows else None,'memory_clock_range_mhz':[min(x['memory_clock_mhz'] for x in rows),max(x['memory_clock_mhz'] for x in rows)] if rows else None}
summary={'benchmark':'bf16, CUDA Graphs enabled, 1 warmup + 20 timed calls/case, 3 independent processes/mode; variant order rotated per series','case_key_map':CASES,'target_order_by_run':{'run1':['compute_80','compute_120','sm_120'],'run2':['compute_120','sm_120','compute_80'],'run3':['sm_120','compute_80','compute_120']},'modes':bench,'model_load':startup,'gpu_telemetry':telemetry,'gpu_idle_before_bench':'40 C, 180 MHz core, 405 MHz memory, 4.87 W, P8, 0% util, 2 MiB used; no compute processes'}
(R/'benchmark-summary.json').write_text(json.dumps(summary,indent=2))
# Per-op profiles, profiling mode disables CUDA Graphs by design.
profile_re=re.compile(r'^(.+): ([0-9.]+) ms of device time per pass$')
op_re=re.compile(r'^\s{2}(.+?)\s+([0-9.]+) ms\s+([0-9.]+)%$')
profiles={}
for mode in MODES:
 lines=(R/f'profile-{mode}.log').read_text(errors='replace').splitlines(); cases=[]; cur=None
 for line in lines:
  m=profile_re.match(line)
  if m:
   cur={'case':m.group(1),'device_ms_per_pass':float(m.group(2)),'ops':[]};cases.append(cur);continue
  m=op_re.match(line)
  if m and cur is not None: cur['ops'].append({'operation':m.group(1).strip(),'device_ms_per_pass':float(m.group(2)),'percent':float(m.group(3))})
 profiles[mode]={'profiling_enabled':True,'cuda_graphs_enabled':False,'warmups_per_case':2,'profiled_passes':5,'cases':cases,'log':str((R/f'profile-{mode}.log').relative_to(EXP))}
 (R/f'profile-{mode}.json').write_text(json.dumps(profiles[mode],indent=2))
# Runtime function resource metadata. These attrs are from actual loaded modules/JIT images.
attrs={m:jload(R/f'kernel-attributes-{m}.json') for m in MODES}
(R/'kernel-attributes.json').write_text(json.dumps({'source':'CudaFunction Driver API attributes on the loaded module; occupancy probe uses launch-bound block sizes (d64/d256=128, d512=256) and FLASH_D512_SMEM=81920','modes':attrs},indent=2))
# NVRTC mode/artifact matrix. SHA compares the runtime-loaded byte image with the separate raw-NVRTC probe.
source='#define EG_BF16\n#define FLASH_D512_BK 32\n'+(W/'src/gpu/kernels.cu').read_text()+'\n'+(W/'src/gpu/kernels_sm80.cu').read_text()
source_sha=hashlib.sha256(source.encode()).hexdigest()
probe_info={}
runtime_path={
 'compute_80':R/'runtime_compute_80_verify_full.json',
 'compute_120':R/'runtime_compute_120_verify.json',
 'sm_120':R/'runtime-sm_120-verify.json',
}
cold_meta={'compute_80':R/'runtime_compute_80_smoke.json','compute_120':R/'runtime_compute_120_verify.json','sm_120':R/'runtime-sm_120-verify.json'}
cold_log={'compute_80':EXP/'logs/verify-compute_80-smoke.log','compute_120':EXP/'logs/verify-compute_120.log','sm_120':EXP/'logs/verify-sm_120.log'}
for mode in MODES:
 d=A/f'probe_{mode}'; p=jload(d/'nvrtc-probe.json'); rt=jload(runtime_path[mode]); art=pathlib.Path(rt['artifact_path']); probe_art=pathlib.Path(p['artifact_path'])
 target_text=''
 if p['artifact_kind']=='PTX':
  probe_text=probe_art.read_text(errors='replace')
  target_text=re.search(r'^\.target (.+)$',probe_text,re.M).group(1)
  nvvm_version=re.search(r'Based on NVVM ([^\r\n]+)',probe_text).group(1)
 else:
  elf_meta=next(x.strip() for x in subprocess.check_output(['/opt/host-cuda/bin/cuobjdump','--dump-elf',str(art)],text=True).splitlines() if 'sm=' in x)
  target_text=re.search(r'sm=(\d+)',elf_meta).group(0)
  nvvm_version='23.0.0 (NVRTC 13.4.92; CUBIN has no printable NVVM-version header; same NVRTC backend as compute_120)'
 cold=jload(cold_meta[mode]); cold_text=cold_log[mode].read_text(errors='replace'); lm=load_re.search(cold_text)
 # Per-process actual warm-start compiler and module-load instrumentation.
 warm=[jload(R/f'runtime-{mode}-run{i}.json') for i in (1,2,3)]
 probe_info[mode]={'nvrtc_target':mode,'artifact_kind':p['artifact_kind'],'artifact_target':target_text,'options':p['options'],'use_fast_math':False,'effective_fast_math_option_emitted':False,'compiler_version':'NVRTC 13.4.92; NVVM noted in generated output','ptx_version':'9.4' if p['artifact_kind']=='PTX' else None,'nvvm_version':nvvm_version,'source_sha256':source_sha,'probe_compile_call_ms':p['compilation_call_ms'],'probe_artifact':{'path':str(probe_art),'size_bytes':probe_art.stat().st_size,'sha256':sha(probe_art),'nvrtc_log_path':str(d/'nvrtc-compilation.log'),'nvrtc_log_bytes':(d/'nvrtc-compilation.log').stat().st_size},'runtime_loaded_artifact':{'path':str(art),'size_bytes':art.stat().st_size,'sha256':sha(art),'hash_matches_probe':sha(art)==sha(probe_art),'load_path':rt['loader'],'module_load_includes_driver_ptx_jit':rt['module_load_includes_driver_ptx_jit']},'cold_first_runtime':{'compile_wall_ms':cold['compile_wall_ms'],'nvrtc_compile_call_ms':cold.get('nvrtc_compile_call_ms'),'module_load_ms':cold['module_load_ms'],'model_load_ms':float(lm.group(1)) if lm else None,'cuda_cache_path':str(EXP/'cache'/mode),'cache_note':'isolated per target, retained; global CUDA JIT cache was not cleared'},'warm_runtime_compile_wall_ms':[x['compile_wall_ms'] for x in warm],'warm_nvrtc_compile_call_ms':[x.get('nvrtc_compile_call_ms') for x in warm],'warm_module_load_ms':[x['module_load_ms'] for x in warm],'warm_compile_wall_median_ms':statistics.median(x['compile_wall_ms'] for x in warm),'warm_module_load_median_ms':statistics.median(x['module_load_ms'] for x in warm),'cold_runtime_log_note':'compute_80 first module load used a failed verify attempt due to wrong fixture root; compilation and module load completed, then full 90-check verify was rerun with corrected fixtures.' if mode=='compute_80' else None}
(R/'compiler-modes.json').write_text(json.dumps({'source_sha256':source_sha,'runtime_path':'A/B call cudarc compile_ptx_with_opts with only arch changed; C calls nvrtcCompileProgram + nvrtcGetCUBINSize/GetCUBIN and passes Ptx::from_binary into CudaContext::load_module. Full NVRTC options, logs, bytes, and timings below.','modes':probe_info},indent=2))
# Environment record: command outputs and immutable baseline identifiers.
def cmd(args):
 try:return subprocess.check_output(args,text=True,stderr=subprocess.STDOUT).strip()
 except Exception as e:return str(e)
base_repo=BASE/'embeddinggemma2-rust'; repo=W
commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
base_status=subprocess.check_output(['git','-C',str(base_repo),'status','--short','--branch'],text=True).strip()
env={'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'experiment_root':str(EXP),'worktree':str(W),'worktree_branch':subprocess.check_output(['git','-C',str(repo),'branch','--show-current'],text=True).strip(),'baseline_commit':'f15131dc109b4b541f3585f15d7643cd65984d41','worktree_commit_before_experiment':'f15131dc109b4b541f3585f15d7643cd65984d41','baseline_branch_status':base_status,'source_repo':'https://github.com/Raimiz/embeddinggemma2-rust','cuda_container':'pytorch-dev (Incus)','gpu':cmd(['nvidia-smi','--query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader']),'driver':'615.71.09','cuda_toolkit':'13.4.92','nvrtc':'13.4.92','cublas':'13.8.0.4','nvvm_compute_80_ptx':'7.0.1','nvvm_compute_120_ptx':'23.0.0','nvvm_sm_120_cubin':'23.0.0 (same NVRTC 13.4.92 backend; CUBIN does not expose a printable NVVM version string)','cuobjdump':cmd(['/opt/host-cuda/bin/cuobjdump','--version']),'nvdisasm':cmd(['/opt/host-cuda/bin/nvdisasm','--version']),'rustc':cmd([str(BASE/'.cargo/bin/rustc'),'--version']),'cargo':cmd([str(BASE/'.cargo/bin/cargo'),'--version']),'cudarc':'0.19.10; dynamic-loading, nvrtc, driver, cublas, cuda-13020','model_repo':'google/embeddinggemma-2','model_revision':'914f7f89142e33e77833254d9c9b90c3cef7303b','model_directory':str(BASE/'model'),'benchmark_input_file':str(EXP/'data/bench_inputs.json'),'benchmark_inputs_sha256':sha(EXP/'data/bench_inputs.json'),'reference_directory':str(BASE/'reference'),'kernel_source_files':{'kernels.cu':sha(W/'src/gpu/kernels.cu'),'kernels_sm80.cu':sha(W/'src/gpu/kernels_sm80.cu')},'benchmark':{'precision':'BF16','warmups_per_case':1,'timed_calls_per_case':20,'independent_series_per_mode':3,'cuda_graphs_enabled':True,'per_op_profiling_enabled':False,'thread_pool_threads':16,'mode_order':summary['target_order_by_run'],'data_preprocessing':'unchanged baseline CLI benchmark path; fixture load outside timer, inference and model-side image/audio preprocessing within model.embed timer'},'per_op_profile':{'precision':'BF16','warmups_per_case':2,'profiled_passes':5,'cuda_graphs_enabled':False,'reason':'EGEMMA2_PROFILE=1 disables CUDA Graphs by design; do not compare per-op device-time directly with E2E short path'},'cuda_cache':{'per_target_paths':{m:str(EXP/'cache'/m) for m in MODES},'global_cache_cleared':False,'isolated_paths_retained':True,'CUDA_CACHE_DISABLE':'not set'},'gpu_before_benchmark':'RTX 5060 Ti, 40 C, core 180 MHz, memory 405 MHz, power 4.87 W, P8, 0% GPU, 2 MiB used; no compute process','system_changes':{'driver_or_toolkit_updated':False,'pytorch_or_exllama_modified':False,'baseline_branch_modified':False,'kernel_sources_modified':False,'cuBLAS_configuration_changed':False,'system_cuda_settings_changed':False}}
(R/'environment.json').write_text(json.dumps(env,indent=2))
# Compact profile values for printing/auditing.
print('Wrote compiler-modes.json, verify.json, benchmark-summary.json, kernel-attributes.json, profile-*.json, environment.json')
print('benchmark medians and deltas vs compute_80 (ms):')
for key in CASES.values():
 print(key,*[round(bench[m][key]['median_of_run_medians_ms'],3) for m in MODES], 'delta%',*[round(bench[m][key]['delta_vs_compute_80_percent'],2) for m in ['compute_120','sm_120']])
print('verify:',{m:(verify[m]['checks'],verify[m]['failed']) for m in MODES})
print('direct exact:',{m:direct[m]['all_cases_exact'] for m in direct})
