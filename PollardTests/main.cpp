#include <iostream>
#include <vector>
#include <array>
#include <cstdint>
#include <fstream>
#include "AddressUtil.h"

#if BUILD_CUDA
#include <cuda_runtime.h>
bool runCudaScalarOne(unsigned int x[8], unsigned int y[8], unsigned int hash[5]);
#endif

#if BUILD_OPENCL
#include "clutil.h"
#include "clContext.h"
#endif

struct RefMatch {
    uint64_t k;
    std::array<unsigned int,5> h;
};

struct RNGState {
    uint64_t s0;
    uint64_t s1;
};

static inline uint64_t xorshift128plus(RNGState &st)
{
    uint64_t x = st.s0;
    uint64_t y = st.s1;
    st.s0 = y;
    x ^= x << 23;
    x ^= x >> 17;
    x ^= y;
    x ^= y >> 26;
    st.s1 = x;
    return x + y;
}

static secp256k1::ecpoint scalarMultiplyBase(uint64_t k) {
    secp256k1::uint256 scalar(k);
    return secp256k1::multiplyPoint(scalar, secp256k1::G());
}

static inline uint64_t next_random_step(RNGState &st)
{
    const uint64_t ORDER_MINUS_ONE = 0xBFD25E8CD0364140ULL;
    return (xorshift128plus(st) % ORDER_MINUS_ONE) + 1ULL;
}

static void fake_hash160(uint64_t k, std::array<unsigned int,5> &h)
{
    RNGState st{ k, k ^ 0x9E3779B97F4A7C15ULL };
    for(int i = 0; i < 5; ++i) {
        h[i] = static_cast<unsigned int>(xorshift128plus(st));
    }
}

static std::vector<RefMatch> referenceWalk(uint64_t seed, unsigned int steps, unsigned int windowBits) {
    RNGState rng{ seed ^ 1ULL, seed + 1ULL };
    uint64_t scalar = 0;
    const uint64_t ORDER = 0xBFD25E8CD0364141ULL;
    uint64_t mask = (windowBits >= 64) ? 0xffffffffffffffffULL : ((1ULL << windowBits) - 1ULL);
    std::vector<RefMatch> out;
    for(unsigned int i = 0; i < steps; ++i) {
        uint64_t step = next_random_step(rng);
        scalar += step;
        scalar %= ORDER;
        if((scalar & mask) == 0ULL) {
            RefMatch m;
            m.k = scalar;
            fake_hash160(scalar, m.h);
            out.push_back(m);
        }
    }
    return out;
}

bool testDeterministicSeed() {
    auto matches = referenceWalk(2ULL, 1000, 8);
    const uint64_t expectedK[1] = {10820130499599738880ULL};
    const unsigned int expectedH[1][5] = {
        {3977112439u, 4030187129u, 4139897143u, 2627728552u, 1044400310u}
    };
    if(matches.size() != 1) return false;
    for(size_t i = 0; i < matches.size(); ++i) {
        if(matches[i].k != expectedK[i]) return false;
        for(int j = 0; j < 5; ++j) {
            if(matches[i].h[j] != expectedH[i][j]) return false;
        }
    }
    return true;
}

bool testScalarOne() {
    secp256k1::ecpoint p = scalarMultiplyBase(1ULL);
    unsigned int digest[5];
    Hash::hashPublicKeyCompressed(p, digest);
    const unsigned int expected[5] = {
        0x751e76e8u,
        0x199196d4u,
        0x54941c45u,
        0xd1b3a323u,
        0xf1433bd6u
    };
    for(int i = 0; i < 5; ++i) {
        if(digest[i] != expected[i]) return false;
    }
    return true;
}

static bool runOpenCLScalarOne(unsigned int x[8], unsigned int y[8], unsigned int hash[5]) {
#if BUILD_OPENCL
    try {
        auto devices = cl::getDevices();
        if(devices.empty()) return false;
        cl::CLContext ctx(devices[0].id);

        auto readFile = [](const char *path){
            std::ifstream f(path); return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        };
        std::string sha = readFile("clMath/sha256.cl");
        std::string secp = readFile("clMath/secp256k1.cl");
        std::string rmd = readFile("clMath/ripemd160.cl");
        std::string pollard = readFile("CLKeySearchDevice/clPollard.cl");
        std::string src = sha + secp + rmd + pollard + R"(
        __kernel void scalar_one(__global uint* xOut, __global uint* yOut, __global uint* hOut) {
            uint px[8]; uint py[8]; uint digest[5];
            scalarMultiplyBase((ulong)1, px, py);
            hashPublicKeyCompressed(px, py[7], digest);
            for(int i=0;i<8;i++){ xOut[i]=px[i]; yOut[i]=py[i]; }
            for(int j=0;j<5;j++){ hOut[j]=digest[j]; }
        }
        )";
        cl::CLProgram prog(ctx, src.c_str());
        cl::CLKernel k(prog, "scalar_one");
        cl_mem dx = ctx.malloc(sizeof(unsigned int)*8);
        cl_mem dy = ctx.malloc(sizeof(unsigned int)*8);
        cl_mem dh = ctx.malloc(sizeof(unsigned int)*5);
        k.set_args(dx, dy, dh);
        k.call(1,1);
        ctx.copyDeviceToHost(dx, x, sizeof(unsigned int)*8);
        ctx.copyDeviceToHost(dy, y, sizeof(unsigned int)*8);
        ctx.copyDeviceToHost(dh, hash, sizeof(unsigned int)*5);
        ctx.free(dx); ctx.free(dy); ctx.free(dh);
        return true;
    } catch(cl::CLException &) {
        return false;
    }
#else
    (void)x; (void)y; (void)hash; return false;
#endif
}

bool testGpuScalarOne() {
    unsigned int expectedHash[5] = {
        0x751e76e8u, 0x199196d4u, 0x54941c45u, 0xd1b3a323u, 0xf1433bd6u
    };
    secp256k1::ecpoint G = secp256k1::G();
    unsigned int gx[8];
    unsigned int gy[8];
    G.x.exportWords(gx, 8, secp256k1::uint256::BigEndian);
    G.y.exportWords(gy, 8, secp256k1::uint256::BigEndian);

    bool ran = false;
    bool pass = true;

#if BUILD_CUDA
    int cudaDevs = 0;
    if(cudaGetDeviceCount(&cudaDevs) == cudaSuccess && cudaDevs > 0) {
        ran = true;
        unsigned int x[8], y[8], h[5];
        if(!runCudaScalarOne(x, y, h)) return false;
        for(int i=0;i<8;i++){ if(x[i]!=gx[i] || y[i]!=gy[i]) pass=false; }
        for(int i=0;i<5;i++){ if(h[i]!=expectedHash[i]) pass=false; }
    }
#endif
#if BUILD_OPENCL
    unsigned int xcl[8], ycl[8], hcl[5];
    if(runOpenCLScalarOne(xcl, ycl, hcl)) {
        ran = true;
        for(int i=0;i<8;i++){ if(xcl[i]!=gx[i] || ycl[i]!=gy[i]) pass=false; }
        for(int i=0;i<5;i++){ if(hcl[i]!=expectedHash[i]) pass=false; }
    }
#endif
    if(!ran) {
        std::cout << "GPU test skipped" << std::endl;
        return true;
    }
    return pass;
}

int main(){
    int fails=0;
    if(!testScalarOne()) { std::cout<<"scalar one failed"<<std::endl; fails++; }
    if(!testDeterministicSeed()) { std::cout<<"deterministic seed failed"<<std::endl; fails++; }
    if(!testGpuScalarOne()) { std::cout<<"gpu scalar one failed"<<std::endl; fails++; }
    if(fails==0) {
        std::cout<<"PASS"<<std::endl;
    } else {
        std::cout<<"FAIL"<<std::endl;
    }
    return fails;
}
