#include "CudaPollardDevice.h"
#include <cuda_runtime.h>
#include <vector>
#include <cstring>

using namespace secp256k1;

struct GpuPollardWindow {
    unsigned int targetIdx;
    unsigned int offset;
    unsigned int bits;
    unsigned int k[8];
};

struct GpuTargetWindow {
    unsigned int targetIdx;
    unsigned int offset;
    unsigned int bits;
    unsigned long long target;
};

static uint64_t hashWindowLE(const unsigned int h[5], unsigned int offset, unsigned int bits) {
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

// Each thread runs an independent walk using a unique seed and starting
// scalar.  Optional starting points can be supplied for wild walks.
extern "C" __global__ void pollardRandomWalk(GpuPollardWindow *out,
                                             unsigned int *outCount,
                                             unsigned int maxOut,
                                             const unsigned long long *seeds,
                                             const unsigned long long *starts,
                                             const unsigned int *startX,
                                             const unsigned int *startY,
                                             unsigned int steps,
                                             const GpuTargetWindow *windows,
                                             unsigned int windowCount);

CudaPollardDevice::CudaPollardDevice(PollardEngine &engine,
                                     unsigned int windowBits,
                                     const std::vector<unsigned int> &offsets,
                                     const std::vector<std::array<unsigned int,5>> &targets)
    : _engine(engine), _windowBits(windowBits), _offsets(offsets), _targets(targets) {}

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

    // Prepare target windows
    std::vector<GpuTargetWindow> h_windows;
    for(size_t t = 0; t < _targets.size(); ++t) {
        for(unsigned int off : _offsets) {
            if(off + _windowBits > 160) continue;
            GpuTargetWindow tw;
            tw.targetIdx = static_cast<unsigned int>(t);
            tw.offset = off;
            tw.bits = _windowBits;
            tw.target = hashWindowLE(_targets[t].data(), off, _windowBits);
            h_windows.push_back(tw);
        }
    }
    unsigned int windowCount = static_cast<unsigned int>(h_windows.size());
    GpuTargetWindow *d_windows = nullptr;
    if(windowCount > 0) {
        cudaMalloc(&d_windows, sizeof(GpuTargetWindow) * windowCount);
        cudaMemcpy(d_windows, h_windows.data(), sizeof(GpuTargetWindow) * windowCount, cudaMemcpyHostToDevice);
    }

    GpuPollardWindow *d_out = nullptr;
    unsigned int *d_count = nullptr;
    unsigned int maxOut = static_cast<unsigned int>(steps * totalThreads);
    cudaMalloc(&d_out, sizeof(GpuPollardWindow) * maxOut);
    cudaMalloc(&d_count, sizeof(unsigned int));
    cudaMemset(d_count, 0, sizeof(unsigned int));

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    pollardRandomWalk<<<blocks, threadsPerBlock, 0, stream>>>(d_out, d_count, maxOut,
                                                             d_seeds, d_starts,
                                                             nullptr, nullptr,
                                                             static_cast<unsigned int>(steps),
                                                             d_windows, windowCount);

    std::vector<GpuPollardWindow> h_out(maxOut);
    unsigned int h_count = 0;
    cudaMemcpyAsync(&h_count, d_count, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_out.data(), d_out, sizeof(GpuPollardWindow) * maxOut, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    unsigned int count = (h_count > maxOut) ? maxOut : h_count;
    for(unsigned int i = 0; i < count; ++i) {
        PollardWindow w;
        w.targetIdx = h_out[i].targetIdx;
        w.offset = h_out[i].offset;
        w.bits = h_out[i].bits;
        for(int j = 0; j < 8; ++j) {
            w.scalarFragment.v[j] = h_out[i].k[j];
        }
        _engine.processWindow(w);
    }

    cudaFree(d_out);
    cudaFree(d_count);
    cudaFree(d_seeds);
    cudaFree(d_starts);
    if(d_windows) cudaFree(d_windows);
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

    // Prepare target windows
    std::vector<GpuTargetWindow> h_windows;
    for(size_t t = 0; t < _targets.size(); ++t) {
        for(unsigned int off : _offsets) {
            if(off + _windowBits > 160) continue;
            GpuTargetWindow tw;
            tw.targetIdx = static_cast<unsigned int>(t);
            tw.offset = off;
            tw.bits = _windowBits;
            tw.target = hashWindowLE(_targets[t].data(), off, _windowBits);
            h_windows.push_back(tw);
        }
    }
    unsigned int windowCount = static_cast<unsigned int>(h_windows.size());
    GpuTargetWindow *d_windows = nullptr;
    if(windowCount > 0) {
        cudaMalloc(&d_windows, sizeof(GpuTargetWindow) * windowCount);
        cudaMemcpy(d_windows, h_windows.data(), sizeof(GpuTargetWindow) * windowCount, cudaMemcpyHostToDevice);
    }

    GpuPollardWindow *d_out = nullptr;
    unsigned int *d_count = nullptr;
    unsigned int maxOut = static_cast<unsigned int>(steps * totalThreads);
    cudaMalloc(&d_out, sizeof(GpuPollardWindow) * maxOut);
    cudaMalloc(&d_count, sizeof(unsigned int));
    cudaMemset(d_count, 0, sizeof(unsigned int));

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    pollardRandomWalk<<<blocks, threadsPerBlock, 0, stream>>>(d_out, d_count, maxOut,
                                                             d_seeds, d_starts,
                                                             d_startX, d_startY,
                                                             static_cast<unsigned int>(steps),
                                                             d_windows, windowCount);

    std::vector<GpuPollardWindow> h_out(maxOut);
    unsigned int h_count = 0;
    cudaMemcpyAsync(&h_count, d_count, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_out.data(), d_out, sizeof(GpuPollardWindow) * maxOut, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    unsigned int count = (h_count > maxOut) ? maxOut : h_count;
    for(unsigned int i = 0; i < count; ++i) {
        PollardWindow w;
        w.targetIdx = h_out[i].targetIdx;
        w.offset = h_out[i].offset;
        w.bits = h_out[i].bits;
        for(int j = 0; j < 8; ++j) {
            w.scalarFragment.v[j] = h_out[i].k[j];
        }
        _engine.processWindow(w);
    }

    cudaFree(d_out);
    cudaFree(d_count);
    cudaFree(d_seeds);
    cudaFree(d_starts);
    cudaFree(d_startX);
    cudaFree(d_startY);
    if(d_windows) cudaFree(d_windows);
    cudaStreamDestroy(stream);
}
