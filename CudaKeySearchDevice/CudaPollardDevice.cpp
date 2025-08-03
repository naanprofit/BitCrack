#include "CudaPollardDevice.h"
#include <cuda_runtime.h>
#include <vector>
#include <cstring>

using namespace secp256k1;

struct CudaPollardMatch {
    unsigned long long k[4];
    unsigned int hash[5];
};

extern "C" __global__ void pollardRandomWalk(CudaPollardMatch *out,
                                             unsigned int *outCount,
                                             unsigned int maxOut,
                                             unsigned long long seed,
                                             unsigned int steps,
                                             unsigned int windowBits);

CudaPollardDevice::CudaPollardDevice(PollardEngine &engine,
                                     unsigned int windowBits,
                                     const std::vector<unsigned int> &offsets,
                                     const std::vector<std::array<unsigned int,5>> &targets)
    : _engine(engine), _windowBits(windowBits), _offsets(offsets), _targets(targets) {}

uint256 CudaPollardDevice::maskBits(unsigned int bits) {
    uint256 m(0);
    for(unsigned int i = 0; i < bits; ++i) {
        m.v[i/32] |= (1u << (i % 32));
    }
    return m;
}

uint64_t CudaPollardDevice::hashWindowLE(const unsigned int h[5], unsigned int offset, unsigned int bits) {
    unsigned int word = offset / 32;
    unsigned int bit = offset % 32;
    uint64_t val = 0;
    if(word < 5) {
        val = ((uint64_t)h[word]) >> bit;
        if(bit + bits > 32 && word + 1 < 5) {
            val |= ((uint64_t)h[word + 1]) << (32 - bit);
        }
    }
    if(bit + bits > 64 && word + 2 < 5) {
        val |= ((uint64_t)h[word + 2]) << (64 - bit);
    }
    if(bits < 64) {
        uint64_t mask = (bits == 64) ? 0xffffffffffffffffULL : ((1ULL << bits) - 1ULL);
        val &= mask;
    }
    return val;
}

void CudaPollardDevice::startTameWalk(const uint256 &start, uint64_t steps, uint64_t seed) {
    (void)start;
    CudaPollardMatch *d_out = nullptr;
    unsigned int *d_count = nullptr;
    unsigned int maxOut = static_cast<unsigned int>(steps);
    cudaMalloc(&d_out, sizeof(CudaPollardMatch) * steps);
    cudaMalloc(&d_count, sizeof(unsigned int));
    cudaMemset(d_count, 0, sizeof(unsigned int));

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    pollardRandomWalk<<<1,1,0,stream>>>(d_out, d_count, maxOut, (unsigned long long)seed, (unsigned int)steps, _windowBits);

    std::vector<CudaPollardMatch> h_out(steps);
    unsigned int h_count = 0;
    cudaMemcpyAsync(&h_count, d_count, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_out.data(), d_out, sizeof(CudaPollardMatch) * steps, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    for(unsigned int i = 0; i < h_count; ++i) {
        PollardMatch m;
        for(int j = 0; j < 4; ++j) {
            m.scalar.v[j*2] = (unsigned int)(h_out[i].k[j] & 0xffffffffULL);
            m.scalar.v[j*2+1] = (unsigned int)(h_out[i].k[j] >> 32);
        }
        std::memcpy(m.hash, h_out[i].hash, sizeof(m.hash));
        for(size_t t = 0; t < _targets.size(); ++t) {
            for(unsigned int off : _offsets) {
                if(off + _windowBits > 160) continue;
                uint64_t want = hashWindowLE(_targets[t].data(), off, _windowBits);
                uint64_t got  = hashWindowLE(m.hash, off, _windowBits);
                if(got == want) {
                    unsigned int modBits = off + _windowBits;
                    if(modBits > 256) continue;
                    uint256 mask = maskBits(modBits);
                    uint256 frag;
                    for(int w = 0; w < 8; ++w) {
                        frag.v[w] = m.scalar.v[w] & mask.v[w];
                    }
                    PollardWindow w{static_cast<unsigned int>(t), off, _windowBits, frag};
                    _engine.processWindow(w);
                }
            }
        }
    }

    cudaFree(d_out);
    cudaFree(d_count);
    cudaStreamDestroy(stream);
}

void CudaPollardDevice::startWildWalk(const ecpoint &start, uint64_t steps, uint64_t seed) {
    (void)start;
    startTameWalk(uint256(), steps, seed);
}
