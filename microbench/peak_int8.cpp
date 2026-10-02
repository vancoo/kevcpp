// peak_int8.cpp - raw AVX512-VNNI vpdpbusd INT8 MAC throughput ceiling.
// Operands come from an L1-resident array so the loops are NOT constant-folded,
// giving a clean measure of the int8 dot-compute ceiling of this CPU (Zen5).
// If the ceiling is high (multi-TMAC/s) a batch weight-reuse GEMM (candidate X)
// could be compute-fast enough to be memory-bound; if low, reuse alone won't help.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <immintrin.h>

static double now_ms(){return (double)std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count()/1e3;}

struct Cfg{ long iters=200000000; int threads=32; };
int main(int argc,char**argv){
    Cfg c; for(int i=1;i<argc;++i){std::string a=argv[i];auto nx=[&](){return i+1<argc?std::string(argv[++i]):std::string();};
        if(a=="--iters")c.iters=atol(nx().c_str()); else if(a=="--threads")c.threads=atoi(nx().c_str());}
    // L1 operand pool (32 zmm = 2KB), read repeatedly, not stored-to so no folding of loads.
    alignas(64) int8_t op[256];
    for(int i=0;i<256;++i) op[i]=(int8_t)(i*3+1);
    std::vector<double> per((size_t)c.threads,1e18);
    int nthreads=c.threads;
    for(int rep=0;rep<3;++rep){
        std::vector<std::thread> pool;
        for(int t=0;t<nthreads;++t)pool.emplace_back([&,t](){
            const __m512i* m=(const __m512i*)op;
            __m512i a=m[0], b=m[1], cc=m[2];           // 3 operands in regs
            __m512i acc[8]; for(int i=0;i<8;++i)acc[i]=_mm512_setzero_si512();
            long iters=c.iters;
            double t0=now_ms();
            for(long i=0;i<iters;++i){
                acc[0]=_mm512_dpbusd_epi32(acc[0],a,b);
                acc[1]=_mm512_dpbusd_epi32(acc[1],b,cc);
                acc[2]=_mm512_dpbusd_epi32(acc[2],cc,a);
                acc[3]=_mm512_dpbusd_epi32(acc[3],a,cc);
                acc[4]=_mm512_dpbusd_epi32(acc[4],b,a);
                acc[5]=_mm512_dpbusd_epi32(acc[5],cc,b);
                acc[6]=_mm512_dpbusd_epi32(acc[6],a,b);
                acc[7]=_mm512_dpbusd_epi32(acc[7],b,cc);
                // periodic reload from L1 keeps operand from being constant-folded
                if((i&0xffff)==0){ __m512i r=m[(i>>16)&3]; a=r; }
            }
            double t1=now_ms();
            __m512i z=_mm512_setzero_si512();
            for(int i=0;i<8;++i)z=_mm512_add_epi32(z,acc[i]);
            alignas(64) int32_t tmp[16]; _mm512_storeu_si512((__m512i*)tmp,z);
            volatile float s=0; for(int q=0;q<16;++q)s+=(float)tmp[q];
            per[t]=t1-t0; (void)s;
        });
        for(auto&th:pool)th.join();
    }
    double best=1e18; for(double p:per) if(p<best) best=p;
    double macs=(double)c.iters*8.0*64.0*c.threads;   // 8 dpbusd/iter, 64 MAC/dpbusd
    double sec=best/1e3;
    double perthread_MACs=(double)c.iters*8.0*64.0;
    std::printf("peak_int8(%d threads): best_trim=%.1f ms  aggregate=%.0f GMAC/s  per-thread=%.0f GMAC/s\n",
                c.threads,best,macs/1e9/sec,perthread_MACs/1e9/sec);
    return 0;
}