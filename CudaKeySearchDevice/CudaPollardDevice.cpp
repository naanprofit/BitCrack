#include "CudaPollardDevice.h"
#include <cuda_runtime.h>
#include <vector>
#include <cstring>

using namespace secp256k1;

struct CudaPollardMatch {
    unsigned long long k[4];
    unsigned int hash[5];
};

// Each thread runs an independent walk using a unique seed and starting
// scalar.  Optional starting points can be supplied for wild walks.
extern "C" __global__ void pollardRandomWalk(CudaPollardMatch *out,
                                             unsigned int *outCount,
                                             unsigned int maxOut,
                                             const unsigned long long *seeds,
                                             const unsigned long long *starts,
                                             const unsigned int *startX,
                                             const unsigned int *startY,
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
    // Determine launch configuration based on device capabilities
    cudaDeviceProp prop;
    int dev = 0;
    cudaGetDevice(&dev);
    cudaGetDeviceProperties(&prop, dev);
    unsigned int threadsPerBlock = prop.warpSize * 4;
    if(threadsPerBlock > prop.maxThreadsPerBlock) {
        threadsPerBlock = prop.maxThreadsPerBlock;
    }
    unsigned int blocks = prop.multiProcessorCount;
    unsigned int totalThreads = threadsPerBlock * blocks;

    // Build per-thread seeds and starting scalars using the ``start`` value
    std::vector<unsigned long long> h_seeds(totalThreads);
    std::vector<unsigned long long> h_starts(totalThreads);
    uint64_t base = ((uint64_t)start.v[1] << 32) | start.v[0];
    for(unsigned int i = 0; i < totalThreads; ++i) {
        h_seeds[i]  = seed + i;
        h_starts[i] = base + i;
    }

    unsigned long long *d_seeds = nullptr;
    unsigned long long *d_starts = nullptr;
    cudaMalloc(&d_seeds, sizeof(unsigned long long) * totalThreads);
    cudaMalloc(&d_starts, sizeof(unsigned long long) * totalThreads);
    cudaMemcpy(d_seeds, h_seeds.data(), sizeof(unsigned long long) * totalThreads, cudaMemcpyHostToDevice);
    cudaMemcpy(d_starts, h_starts.data(), sizeof(unsigned long long) * totalThreads, cudaMemcpyHostToDevice);

    CudaPollardMatch *d_out = nullptr;
    unsigned int *d_count = nullptr;
    unsigned int maxOut = static_cast<unsigned int>(steps * totalThreads);
    cudaMalloc(&d_out, sizeof(CudaPollardMatch) * maxOut);
    cudaMalloc(&d_count, sizeof(unsigned int));
    cudaMemset(d_count, 0, sizeof(unsigned int));

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    pollardRandomWalk<<<blocks, threadsPerBlock, 0, stream>>>(d_out, d_count, maxOut,
                                                             d_seeds, d_starts,
                                                             nullptr, nullptr,
                                                             static_cast<unsigned int>(steps),
                                                             _windowBits);

    std::vector<CudaPollardMatch> h_out(maxOut);
    unsigned int h_count = 0;
    cudaMemcpyAsync(&h_count, d_count, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_out.data(), d_out, sizeof(CudaPollardMatch) * maxOut, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    unsigned int count = (h_count > maxOut) ? maxOut : h_count;
    for(unsigned int i = 0; i < count; ++i) {
        PollardMatch m;
        for(int j = 0; j < 4; ++j) {
            m.scalar.v[j*2]     = static_cast<unsigned int>(h_out[i].k[j] & 0xffffffffULL);
            m.scalar.v[j*2 + 1] = static_cast<unsigned int>(h_out[i].k[j] >> 32);
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
    cudaFree(d_seeds);
    cudaFree(d_starts);
    cudaStreamDestroy(stream);
}

void CudaPollardDevice::startWildWalk(const ecpoint &start, uint64_t steps, uint64_t seed) {
    // Determine launch configuration similar to tame walk
    cudaDeviceProp prop;
    int dev = 0;
    cudaGetDevice(&dev);
    cudaGetDeviceProperties(&prop, dev);
    unsigned int threadsPerBlock = prop.warpSize * 4;
    if(threadsPerBlock > prop.maxThreadsPerBlock) {
        threadsPerBlock = prop.maxThreadsPerBlock;
    }
    unsigned int blocks = prop.multiProcessorCount;
    unsigned int totalThreads = threadsPerBlock * blocks;

    std::vector<unsigned long long> h_seeds(totalThreads);
    std::vector<unsigned long long> h_starts(totalThreads, 0ULL);
    std::vector<unsigned int> h_startX(totalThreads * 8);
    std::vector<unsigned int> h_startY(totalThreads * 8);

    for(unsigned int i = 0; i < totalThreads; ++i) {
        h_seeds[i] = seed + i;
        uint256 idx(i);
        ecpoint p = addPoints(start, multiplyPoint(idx, G()));
        for(int w = 0; w < 8; ++w) {
            h_startX[i*8 + w] = p.x.v[w];
            h_startY[i*8 + w] = p.y.v[w];
        }
    }

    unsigned long long *d_seeds = nullptr;
    unsigned long long *d_starts = nullptr;
    unsigned int *d_startX = nullptr;
    unsigned int *d_startY = nullptr;
    cudaMalloc(&d_seeds, sizeof(unsigned long long) * totalThreads);
    cudaMalloc(&d_starts, sizeof(unsigned long long) * totalThreads);
    cudaMalloc(&d_startX, sizeof(unsigned int) * totalThreads * 8);
    cudaMalloc(&d_startY, sizeof(unsigned int) * totalThreads * 8);
    cudaMemcpy(d_seeds, h_seeds.data(), sizeof(unsigned long long) * totalThreads, cudaMemcpyHostToDevice);
    cudaMemcpy(d_starts, h_starts.data(), sizeof(unsigned long long) * totalThreads, cudaMemcpyHostToDevice);
    cudaMemcpy(d_startX, h_startX.data(), sizeof(unsigned int) * totalThreads * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(d_startY, h_startY.data(), sizeof(unsigned int) * totalThreads * 8, cudaMemcpyHostToDevice);

    CudaPollardMatch *d_out = nullptr;
    unsigned int *d_count = nullptr;
    unsigned int maxOut = static_cast<unsigned int>(steps * totalThreads);
    cudaMalloc(&d_out, sizeof(CudaPollardMatch) * maxOut);
    cudaMalloc(&d_count, sizeof(unsigned int));
    cudaMemset(d_count, 0, sizeof(unsigned int));

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    pollardRandomWalk<<<blocks, threadsPerBlock, 0, stream>>>(d_out, d_count, maxOut,
                                                             d_seeds, d_starts,
                                                             d_startX, d_startY,
                                                             static_cast<unsigned int>(steps),
                                                             _windowBits);

    std::vector<CudaPollardMatch> h_out(maxOut);
    unsigned int h_count = 0;
    cudaMemcpyAsync(&h_count, d_count, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_out.data(), d_out, sizeof(CudaPollardMatch) * maxOut, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    unsigned int count = (h_count > maxOut) ? maxOut : h_count;
    for(unsigned int i = 0; i < count; ++i) {
        PollardMatch m;
        for(int j = 0; j < 4; ++j) {
            m.scalar.v[j*2]     = static_cast<unsigned int>(h_out[i].k[j] & 0xffffffffULL);
            m.scalar.v[j*2 + 1] = static_cast<unsigned int>(h_out[i].k[j] >> 32);
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
    cudaFree(d_seeds);
    cudaFree(d_starts);
    cudaFree(d_startX);
    cudaFree(d_startY);
    cudaStreamDestroy(stream);
}
