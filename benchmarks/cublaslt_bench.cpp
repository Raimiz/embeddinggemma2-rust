#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#define CUDA_OK(x) do { cudaError_t _status_=(x); if(_status_!=cudaSuccess){std::cerr<<"CUDA "<<cudaGetErrorString(_status_)<<" at "<<__LINE__<<"\n"; std::exit(2);} } while(0)
#define BLAS_OK(x) do { cublasStatus_t _status_=(x); if(_status_!=CUBLAS_STATUS_SUCCESS){std::cerr<<"cuBLAS status "<<int(_status_)<<" at "<<__LINE__<<"\n"; std::exit(2);} } while(0)
#define LT_OK(x) do { cublasStatus_t _status_=(x); if(_status_!=CUBLAS_STATUS_SUCCESS){std::cerr<<"cuBLASLt status "<<int(_status_)<<" at "<<__LINE__<<"\n"; std::exit(2);} } while(0)

struct Shape { const char* name; int m,n,k; int calls; };
struct Algo { cublasLtMatmulHeuristicResult_t h; int id=-1; uint32_t tile=0, stages=0; std::vector<double> samples; double max_abs=0, cosine=0; };
uint16_t f32bf16(float x) {
  uint32_t u=std::bit_cast<uint32_t>(x);
  u += 0x7fffu + ((u >> 16) & 1u);
  return uint16_t(u >> 16);
}
float bf16f32(uint16_t x) { return std::bit_cast<float>(uint32_t(x) << 16); }
void get_attr(const cublasLtMatmulAlgo_t* a, cublasLtMatmulAlgoConfigAttributes_t attr, void* p, size_t n) {
  size_t written=0;
  cublasLtMatmulAlgoConfigGetAttribute(a,attr,p,n,&written);
}
int main() {
  int device=0; CUDA_OK(cudaSetDevice(device));
  cublasHandle_t blas; BLAS_OK(cublasCreate(&blas));
  BLAS_OK(cublasSetMathMode(blas,CUBLAS_TF32_TENSOR_OP_MATH));
  cublasLtHandle_t lt; LT_OK(cublasLtCreate(&lt));
  constexpr size_t WORKSPACE_BYTES=64ull<<20;
  constexpr int HEURISTIC_REQUEST=8, MAX_CANDIDATES=3, WARM=5, REPS=5, ITERS=20;
  void* workspace=nullptr; CUDA_OK(cudaMalloc(&workspace,WORKSPACE_BYTES));
  const std::vector<Shape> shapes={
    {"batch64_qkv_local_m815",815,2048,512,20}, {"batch64_qkv_local_m2042",2042,2048,512,20},
    {"batch64_qkv_global_m815",815,3072,512,4}, {"batch64_qkv_global_m2042",2042,3072,512,4},
    {"batch64_gate_up_m815",815,4096,512,24}, {"batch64_gate_up_m2042",2042,4096,512,24},
    {"batch64_down_m815",815,512,2048,24}, {"batch64_down_m2042",2042,512,2048,24},
    {"image_vision_gate_up",2394,6144,768,16},
  };
  const float alpha=1.0f,beta=0.0f;
  std::mt19937 rng(0xE66E2u);
  std::uniform_real_distribution<float> dist(-0.08f,0.08f);
  for(const Shape& s:shapes) {
    const size_t a_count=size_t(s.n)*s.k, b_count=size_t(s.m)*s.k, c_count=size_t(s.m)*s.n;
    std::vector<uint16_t> hA(a_count),hB(b_count),hRef(c_count),hTmp(c_count);
    for(auto& x:hA)x=f32bf16(dist(rng)); for(auto& x:hB)x=f32bf16(dist(rng));
    uint16_t *A=nullptr,*B=nullptr,*Cex=nullptr,*Clt=nullptr;
    std::vector<double> baseline_samples;
    CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&A),a_count*sizeof(uint16_t))); CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&B),b_count*sizeof(uint16_t)));
    CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&Cex),c_count*sizeof(uint16_t))); CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&Clt),c_count*sizeof(uint16_t)));
    CUDA_OK(cudaMemcpy(A,hA.data(),a_count*sizeof(uint16_t),cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(B,hB.data(),b_count*sizeof(uint16_t),cudaMemcpyHostToDevice));
    cublasLtMatmulDesc_t op; LT_OK(cublasLtMatmulDescCreate(&op,CUBLAS_COMPUTE_32F,CUDA_R_32F));
    cublasOperation_t transA=CUBLAS_OP_T,transB=CUBLAS_OP_N;
    LT_OK(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&transA,sizeof(transA)));
    LT_OK(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSB,&transB,sizeof(transB)));
    cublasLtMatrixLayout_t ad,bd,cd;
    LT_OK(cublasLtMatrixLayoutCreate(&ad,CUDA_R_16BF,s.k,s.n,s.k));
    LT_OK(cublasLtMatrixLayoutCreate(&bd,CUDA_R_16BF,s.k,s.m,s.k));
    LT_OK(cublasLtMatrixLayoutCreate(&cd,CUDA_R_16BF,s.n,s.m,s.n));
    cublasLtOrder_t col_order=CUBLASLT_ORDER_COL;
    LT_OK(cublasLtMatrixLayoutSetAttribute(ad,CUBLASLT_MATRIX_LAYOUT_ORDER,&col_order,sizeof(col_order)));
    LT_OK(cublasLtMatrixLayoutSetAttribute(bd,CUBLASLT_MATRIX_LAYOUT_ORDER,&col_order,sizeof(col_order)));
    LT_OK(cublasLtMatrixLayoutSetAttribute(cd,CUBLASLT_MATRIX_LAYOUT_ORDER,&col_order,sizeof(col_order)));
    cublasLtMatmulPreference_t pref; LT_OK(cublasLtMatmulPreferenceCreate(&pref));
    uint64_t workspace_limit=WORKSPACE_BYTES;
    LT_OK(cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&workspace_limit,sizeof(workspace_limit)));
    cublasLtMatmulHeuristicResult_t found[HEURISTIC_REQUEST]; int found_count=0;
    LT_OK(cublasLtMatmulAlgoGetHeuristic(lt,op,ad,bd,cd,cd,pref,HEURISTIC_REQUEST,found,&found_count));
    if(found_count>MAX_CANDIDATES)found_count=MAX_CANDIDATES;
    std::vector<Algo> algos;
    for(int i=0;i<found_count;i++)if(found[i].state==CUBLAS_STATUS_SUCCESS) {
      Algo a{};a.h=found[i];get_attr(&a.h.algo,CUBLASLT_ALGO_CONFIG_ID,&a.id,sizeof(a.id));
      get_attr(&a.h.algo,CUBLASLT_ALGO_CONFIG_TILE_ID,&a.tile,sizeof(a.tile));
      get_attr(&a.h.algo,CUBLASLT_ALGO_CONFIG_STAGES_ID,&a.stages,sizeof(a.stages));algos.push_back(a);
    }
    // Warm every implementation; the same device A/B tensors and BF16 output layout are reused.
    for(int i=0;i<WARM;i++) BLAS_OK(cublasGemmEx(blas,CUBLAS_OP_T,CUBLAS_OP_N,s.n,s.m,s.k,&alpha,A,CUDA_R_16BF,s.k,B,CUDA_R_16BF,s.k,&beta,Cex,CUDA_R_16BF,s.n,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DFALT));
    for(auto& a:algos)for(int i=0;i<WARM;i++)LT_OK(cublasLtMatmul(lt,op,&alpha,A,ad,B,bd,&beta,Clt,cd,Clt,cd,&a.h.algo,workspace,WORKSPACE_BYTES,0));
    CUDA_OK(cudaDeviceSynchronize());
    // Correctness against current cublasGemmEx on identical A/B values.
    BLAS_OK(cublasGemmEx(blas,CUBLAS_OP_T,CUBLAS_OP_N,s.n,s.m,s.k,&alpha,A,CUDA_R_16BF,s.k,B,CUDA_R_16BF,s.k,&beta,Cex,CUDA_R_16BF,s.n,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DFALT));
    CUDA_OK(cudaDeviceSynchronize()); CUDA_OK(cudaMemcpy(hRef.data(),Cex,c_count*sizeof(uint16_t),cudaMemcpyDeviceToHost));
    for(auto& a:algos) {
      LT_OK(cublasLtMatmul(lt,op,&alpha,A,ad,B,bd,&beta,Clt,cd,Clt,cd,&a.h.algo,workspace,WORKSPACE_BYTES,0));
      CUDA_OK(cudaDeviceSynchronize()); CUDA_OK(cudaMemcpy(hTmp.data(),Clt,c_count*sizeof(uint16_t),cudaMemcpyDeviceToHost));
      double dot=0,aa=0,bb=0,mx=0;
      for(size_t j=0;j<c_count;j++){float x=bf16f32(hRef[j]),y=bf16f32(hTmp[j]);dot+=double(x)*y;aa+=double(x)*x;bb+=double(y)*y;mx=std::max(mx,std::abs(double(x)-y));}
      a.max_abs=mx;a.cosine=dot/std::sqrt(aa*bb);
    }
    // Interleave benchmark blocks; each sample is a CUDA-event mean of ITERS calls, allocations excluded.
    const int variant_count=int(algos.size())+1;
    for(int rep=0;rep<REPS;rep++)for(int pos=0;pos<variant_count;pos++) {
      const int slot=(pos+rep)%variant_count;
      cudaEvent_t beg,end;CUDA_OK(cudaEventCreate(&beg));CUDA_OK(cudaEventCreate(&end));CUDA_OK(cudaEventRecord(beg,0));
      if(slot==0) {
        for(int i=0;i<ITERS;i++)BLAS_OK(cublasGemmEx(blas,CUBLAS_OP_T,CUBLAS_OP_N,s.n,s.m,s.k,&alpha,A,CUDA_R_16BF,s.k,B,CUDA_R_16BF,s.k,&beta,Cex,CUDA_R_16BF,s.n,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DFALT));
      } else {
        Algo& a=algos[slot-1];
        for(int i=0;i<ITERS;i++)LT_OK(cublasLtMatmul(lt,op,&alpha,A,ad,B,bd,&beta,Clt,cd,Clt,cd,&a.h.algo,workspace,WORKSPACE_BYTES,0));
      }
      CUDA_OK(cudaEventRecord(end,0));CUDA_OK(cudaEventSynchronize(end));float f;CUDA_OK(cudaEventElapsedTime(&f,beg,end));
      const double ms=double(f)/ITERS;CUDA_OK(cudaEventDestroy(beg));CUDA_OK(cudaEventDestroy(end));
      if(slot==0)baseline_samples.push_back(ms);else algos[slot-1].samples.push_back(ms);
    }
    auto print_samples=[](const std::vector<double>& xs){std::cout<<"[";for(size_t i=0;i<xs.size();i++){if(i)std::cout<<",";std::cout<<std::setprecision(8)<<xs[i];}std::cout<<"]";};
    std::cout<<"{\"shape\":\""<<s.name<<"\",\"m\":"<<s.m<<",\"n\":"<<s.n<<",\"k\":"<<s.k<<",\"calls_per_inference\":"<<s.calls<<",\"a_layout\":\"bf16 col-major view of row-major weights [n,k], op=T, lda=k\",\"b_layout\":\"bf16 col-major view of row-major activations [m,k], op=N, ldb=k\",\"output\":\"bf16, ldc=n\",\"compute\":\"CUBLAS_COMPUTE_32F\",\"baseline_samples_ms\":";
    print_samples(baseline_samples);
    std::cout<<",\"lt_algorithms\":[";
    for(size_t i=0;i<algos.size();i++){if(i)std::cout<<",";auto&a=algos[i];std::cout<<"{\"id\":"<<a.id<<",\"tile\":"<<a.tile<<",\"stages\":"<<a.stages<<",\"heuristic_workspace_bytes\":"<<a.h.workspaceSize<<",\"heuristic_waves\":"<<a.h.wavesCount<<",\"max_abs_vs_ex\":"<<a.max_abs<<",\"cosine_vs_ex\":"<<a.cosine<<",\"samples_ms\":";print_samples(a.samples);std::cout<<"}";}
    std::cout<<"]}\n";
    LT_OK(cublasLtMatmulPreferenceDestroy(pref));LT_OK(cublasLtMatrixLayoutDestroy(ad));LT_OK(cublasLtMatrixLayoutDestroy(bd));LT_OK(cublasLtMatrixLayoutDestroy(cd));LT_OK(cublasLtMatmulDescDestroy(op));
    CUDA_OK(cudaFree(A));CUDA_OK(cudaFree(B));CUDA_OK(cudaFree(Cex));CUDA_OK(cudaFree(Clt));
  }
  CUDA_OK(cudaFree(workspace)); LT_OK(cublasLtDestroy(lt)); BLAS_OK(cublasDestroy(blas));
}
