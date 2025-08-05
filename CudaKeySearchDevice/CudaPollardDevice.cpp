#include "CudaPollardDevice.h"
#include <cuda_runtime.h>
#include <vector>
#include <cstring>
#include <unordered_set>
#include <algorithm>
#include <stdexcept>


using namespace secp256k1;

extern "C" void launchWindowKernel(dim3 gridDim,
                                    dim3 blockDim,
                                    uint64_t start_k,
                                    uint64_t range_length,
                                    uint32_t ws,
                                    const uint32_t *offsets,
                                    uint32_t offsets_count,
                                    uint32_t mask,
                                    const uint32_t target_fragments[][MAX_OFFSETS],
                                    MatchRecord *out_buffer,
                                    uint32_t *out_count);

struct GpuPollardWindow {
    uint32_t targetIdx;
    uint32_t offset;
    uint32_t bits;
    // Returned scalar fragment (full 256-bit value in little-endian order)
    uint32_t k[8];
};

struct GpuTargetWindow {
    uint32_t targetIdx;
    uint32_t offset;
    uint32_t bits;
    uint32_t target[5];
};

// Extract ``bits`` bits starting at ``offset`` from the 160-bit hash ``h``.
// ``h`` must be provided in little-endian word order so bit offsets match
// the expectations of the device kernels.
static uint256 hashWindowLE(const uint32_t h[5], uint32_t offset, uint32_t bits) {
    uint256 out(0);
    uint32_t word = offset / 32;
    uint32_t bit = offset % 32;
    uint32_t words = (bits + 31) / 32;
    for(uint32_t i = 0; i < words && word + i < 5; ++i) {
        uint64_t val = ((uint64_t)h[word + i]) >> bit;
        if(bit && word + i + 1 < 5) {
            val |= ((uint64_t)h[word + i + 1]) << (32 - bit);
        }
        out.v[i] = static_cast<uint32_t>(val & 0xffffffffULL);
    }
    uint32_t maskBits = bits % 32;
    if(maskBits) {
        uint32_t mask = (1u << maskBits) - 1u;
        out.v[words - 1] &= mask;
    }
    for(uint32_t i = words; i < 8; ++i) {
        out.v[i] = 0u;
    }
    return out;
}

// Each thread runs an independent walk using a unique seed and starting
// scalar.  Optional starting points can be supplied for wild walks.  When
// ``stride`` is non-zero, a deterministic sequential walk is performed where
// each thread increments by ``stride`` instead of a random step.
extern "C" __global__ void pollardWalk(GpuPollardWindow *out,
                                       uint32_t *outCount,
                                       uint32_t maxOut,
                                       const uint32_t *seeds,
                                       const uint32_t *starts,
                                       const uint32_t *startX,
                                       const uint32_t *startY,
                                       uint32_t steps,
                                       const GpuTargetWindow *windows,
                                       uint32_t windowCount,
                                       const uint32_t *stride);

CudaPollardDevice::CudaPollardDevice(PollardEngine &engine,
                                     unsigned int windowBits,
                                     const std::vector<unsigned int> &offsets,
                                     const std::vector<std::array<uint32_t,5>> &targets,
                                     bool debug,
                                     unsigned int gridDim,
                                     unsigned int blockDim)
    : _engine(engine), _windowBits(windowBits), _offsets(offsets),
      _targets(targets), _debug(debug), _gridDim(gridDim), _blockDim(blockDim) {}

void CudaPollardDevice::startTameWalk(const uint256 &start, uint64_t steps,
                                      const uint256 &seed, bool sequential) {
    // Determine launch configuration based on device capabilities
    cudaDeviceProp prop;
    int dev = 0;
    cudaGetDevice(&dev);
    cudaGetDeviceProperties(&prop, dev);

    unsigned int threadsPerBlock;
    if(_blockDim) {
        if(_blockDim % prop.warpSize != 0 || _blockDim > (unsigned)prop.maxThreadsPerBlock) {
            throw std::runtime_error("invalid blockDim");
        }
        threadsPerBlock = _blockDim;
    } else {
        threadsPerBlock = prop.warpSize * 4;
        if(threadsPerBlock > (unsigned)prop.maxThreadsPerBlock) {
            threadsPerBlock = prop.maxThreadsPerBlock;
        }
    }

    unsigned int blocks;
    if(_gridDim) {
        if(_gridDim > (unsigned)prop.maxGridSize[0]) {
            throw std::runtime_error("invalid gridDim");
        }
        blocks = _gridDim;
    } else {
        blocks = prop.multiProcessorCount;
    }

    unsigned int totalThreads = threadsPerBlock * blocks;

    // Build per-thread 256-bit seeds and starting scalars using the ``start`` value
    std::vector<uint32_t> h_seeds(totalThreads * 8);
    std::vector<uint32_t> h_starts(totalThreads * 8);
    std::vector<uint32_t> h_stride(totalThreads * 8);
    uint256 strideVal = sequential ? uint256(totalThreads) : uint256(0);
    for(unsigned int i = 0; i < totalThreads; ++i) {
        uint256 sSeed = seed + uint256(i);
        sSeed.exportWords(&h_seeds[i*8], 8);
        uint256 sStart = addModN(start, uint256(i));
        sStart.exportWords(&h_starts[i*8], 8);
        strideVal.exportWords(&h_stride[i*8], 8);
    }

    uint32_t *d_seeds = nullptr;
    uint32_t *d_starts = nullptr;
    uint32_t *d_stride = nullptr;
    cudaMalloc(&d_seeds, sizeof(uint32_t) * totalThreads * 8);
    cudaMalloc(&d_starts, sizeof(uint32_t) * totalThreads * 8);
    cudaMalloc(&d_stride, sizeof(uint32_t) * totalThreads * 8);
    cudaMemcpy(d_seeds, h_seeds.data(), sizeof(uint32_t) * totalThreads * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(d_starts, h_starts.data(), sizeof(uint32_t) * totalThreads * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(d_stride, h_stride.data(), sizeof(uint32_t) * totalThreads * 8, cudaMemcpyHostToDevice);

    // Prepare target windows
    std::vector<GpuTargetWindow> h_windows;
    for(size_t t = 0; t < _targets.size(); ++t) {
        for(unsigned int off : _offsets) {
            if(off + _windowBits > 160) continue;
            GpuTargetWindow tw;
            tw.targetIdx = static_cast<uint32_t>(t);
            tw.offset = off;
            tw.bits = _windowBits;
            uint256 hv = hashWindowLE(_targets[t].data(), off, _windowBits);
            hv.exportWords(tw.target, 5);
            h_windows.push_back(tw);
        }
    }
    uint32_t windowCount = static_cast<uint32_t>(h_windows.size());
    GpuTargetWindow *d_windows = nullptr;
    if(windowCount > 0) {
        cudaMalloc(&d_windows, sizeof(GpuTargetWindow) * windowCount);
        cudaMemcpy(d_windows, h_windows.data(), sizeof(GpuTargetWindow) * windowCount, cudaMemcpyHostToDevice);
    }

    GpuPollardWindow *d_out = nullptr;
    uint32_t *d_count = nullptr;
    uint32_t maxOut = static_cast<uint32_t>(steps * totalThreads);
    cudaMalloc(&d_out, sizeof(GpuPollardWindow) * maxOut);
    cudaMalloc(&d_count, sizeof(uint32_t));
    cudaMemset(d_count, 0, sizeof(uint32_t));

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    pollardWalk<<<blocks, threadsPerBlock, 0, stream>>>(d_out, d_count, maxOut,
                                                       d_seeds, d_starts,
                                                       nullptr, nullptr,
                                                       static_cast<uint32_t>(steps),
                                                       d_windows, windowCount,
                                                       d_stride);

    std::vector<GpuPollardWindow> h_out(maxOut);
    uint32_t h_count = 0;
    cudaMemcpyAsync(&h_count, d_count, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_out.data(), d_out, sizeof(GpuPollardWindow) * maxOut, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    uint32_t count = (h_count > maxOut) ? maxOut : h_count;
    for(uint32_t i = 0; i < count; ++i) {
        unsigned int offset = h_out[i].offset;
        unsigned int modBits = offset + h_out[i].bits;
        secp256k1::uint256 rem;
        for(int j = 0; j < 8; ++j) {
            rem.v[j] = h_out[i].k[j];
        }
        if(modBits < 256) {
            unsigned int word = modBits / 32;
            unsigned int bit = modBits % 32;
            if(bit) {
                unsigned int mask = (1u << bit) - 1u;
                rem.v[word] &= mask;
                for(unsigned int k = word + 1; k < 8; ++k) {
                    rem.v[k] = 0u;
                }
            } else {
                for(unsigned int k = word; k < 8; ++k) {
                    rem.v[k] = 0u;
                }
            }
        }
        secp256k1::uint256 mod(0);
        if(modBits < 256) {
            mod.v[modBits / 32] = (1u << (modBits % 32));
        }
        PollardEngine::Constraint c{mod, rem};
        _engine.processWindow(h_out[i].targetIdx, offset, c);
    }

    cudaFree(d_out);
    cudaFree(d_count);
    cudaFree(d_seeds);
    cudaFree(d_starts);
    cudaFree(d_stride);
    if(d_windows) cudaFree(d_windows);
    cudaStreamDestroy(stream);
}

void CudaPollardDevice::startWildWalk(const uint256 &start, uint64_t steps,
                                      const uint256 &seed, bool sequential) {
    // Determine launch configuration similar to tame walk
    cudaDeviceProp prop;
    int dev = 0;
    cudaGetDevice(&dev);
    cudaGetDeviceProperties(&prop, dev);

    unsigned int threadsPerBlock;
    if(_blockDim) {
        if(_blockDim % prop.warpSize != 0 || _blockDim > (unsigned)prop.maxThreadsPerBlock) {
            throw std::runtime_error("invalid blockDim");
        }
        threadsPerBlock = _blockDim;
    } else {
        threadsPerBlock = prop.warpSize * 4;
        if(threadsPerBlock > (unsigned)prop.maxThreadsPerBlock) {
            threadsPerBlock = prop.maxThreadsPerBlock;
        }
    }

    unsigned int blocks;
    if(_gridDim) {
        if(_gridDim > (unsigned)prop.maxGridSize[0]) {
            throw std::runtime_error("invalid gridDim");
        }
        blocks = _gridDim;
    } else {
        blocks = prop.multiProcessorCount;
    }

    unsigned int totalThreads = threadsPerBlock * blocks;

    // Prepare per-thread 256-bit seeds, starting scalars and points
    std::vector<uint32_t> h_seeds(totalThreads * 8);
    std::vector<uint32_t> h_starts(totalThreads * 8);
    std::vector<uint32_t> h_startX(totalThreads * 8);
    std::vector<uint32_t> h_startY(totalThreads * 8);
    std::vector<uint32_t> h_stride(totalThreads * 8);

    uint256 base = start;
    uint256 strideVal = sequential ? uint256(totalThreads) : uint256(0);
    uint256 startBase = base;
    if(sequential) {
        uint256 offset = multiplyModN(strideVal, uint256(steps - 1));
        startBase = subModN(base, offset);
    }
    ecpoint startPoint = multiplyPoint(start, G());

    for(unsigned int i = 0; i < totalThreads; ++i) {
        uint256 sSeed = seed + uint256(i);
        sSeed.exportWords(&h_seeds[i*8], 8);
        uint256 s = sequential ? subModN(startBase, uint256(i)) : uint256(0);
        s.exportWords(&h_starts[i*8], 8);
        strideVal.exportWords(&h_stride[i*8], 8);
        ecpoint p;
        if(sequential) {
            p = multiplyPoint(s, G());
        } else {
            uint256 idx(i);
            p = addPoints(startPoint, multiplyPoint(idx, G()));
        }
        for(int w = 0; w < 8; ++w) {
            h_startX[i*8 + w] = p.x.v[w];
            h_startY[i*8 + w] = p.y.v[w];
        }
    }

    uint32_t *d_seeds = nullptr;
    uint32_t *d_starts = nullptr;
    uint32_t *d_startX = nullptr;
    uint32_t *d_startY = nullptr;
    uint32_t *d_stride = nullptr;
    cudaMalloc(&d_seeds, sizeof(uint32_t) * totalThreads * 8);
    cudaMalloc(&d_starts, sizeof(uint32_t) * totalThreads * 8);
    cudaMalloc(&d_startX, sizeof(uint32_t) * totalThreads * 8);
    cudaMalloc(&d_startY, sizeof(uint32_t) * totalThreads * 8);
    cudaMalloc(&d_stride, sizeof(uint32_t) * totalThreads * 8);
    cudaMemcpy(d_seeds, h_seeds.data(), sizeof(uint32_t) * totalThreads * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(d_starts, h_starts.data(), sizeof(uint32_t) * totalThreads * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(d_startX, h_startX.data(), sizeof(uint32_t) * totalThreads * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(d_startY, h_startY.data(), sizeof(uint32_t) * totalThreads * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(d_stride, h_stride.data(), sizeof(uint32_t) * totalThreads * 8, cudaMemcpyHostToDevice);

    // Prepare target windows
    std::vector<GpuTargetWindow> h_windows;
    for(size_t t = 0; t < _targets.size(); ++t) {
        for(unsigned int off : _offsets) {
            if(off + _windowBits > 160) continue;
            GpuTargetWindow tw;
            tw.targetIdx = static_cast<uint32_t>(t);
            tw.offset = off;
            tw.bits = _windowBits;
            uint256 hv = hashWindowLE(_targets[t].data(), off, _windowBits);
            hv.exportWords(tw.target, 5);
            h_windows.push_back(tw);
        }
    }
    uint32_t windowCount = static_cast<uint32_t>(h_windows.size());
    GpuTargetWindow *d_windows = nullptr;
    if(windowCount > 0) {
        cudaMalloc(&d_windows, sizeof(GpuTargetWindow) * windowCount);
        cudaMemcpy(d_windows, h_windows.data(), sizeof(GpuTargetWindow) * windowCount, cudaMemcpyHostToDevice);
    }

    GpuPollardWindow *d_out = nullptr;
    uint32_t *d_count = nullptr;
    uint32_t maxOut = static_cast<uint32_t>(steps * totalThreads);
    cudaMalloc(&d_out, sizeof(GpuPollardWindow) * maxOut);
    cudaMalloc(&d_count, sizeof(uint32_t));
    cudaMemset(d_count, 0, sizeof(uint32_t));

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    pollardWalk<<<blocks, threadsPerBlock, 0, stream>>>(d_out, d_count, maxOut,
                                                       d_seeds, d_starts,
                                                       d_startX, d_startY,
                                                       static_cast<uint32_t>(steps),
                                                       d_windows, windowCount,
                                                       d_stride);

    std::vector<GpuPollardWindow> h_out(maxOut);
    uint32_t h_count = 0;
    cudaMemcpyAsync(&h_count, d_count, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_out.data(), d_out, sizeof(GpuPollardWindow) * maxOut, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    uint32_t count = (h_count > maxOut) ? maxOut : h_count;
    for(uint32_t i = 0; i < count; ++i) {
        unsigned int offset = h_out[i].offset;
        unsigned int modBits = offset + h_out[i].bits;
        secp256k1::uint256 rem;
        for(int j = 0; j < 8; ++j) {
            rem.v[j] = h_out[i].k[j];
        }
        if(modBits < 256) {
            unsigned int word = modBits / 32;
            unsigned int bit = modBits % 32;
            if(bit) {
                unsigned int mask = (1u << bit) - 1u;
                rem.v[word] &= mask;
                for(unsigned int k = word + 1; k < 8; ++k) {
                    rem.v[k] = 0u;
                }
            } else {
                for(unsigned int k = word; k < 8; ++k) {
                    rem.v[k] = 0u;
                }
            }
        }
        secp256k1::uint256 mod(0);
        if(modBits < 256) {
            mod.v[modBits / 32] = (1u << (modBits % 32));
        }
        PollardEngine::Constraint c{mod, rem};
        _engine.processWindow(h_out[i].targetIdx, offset, c);
    }

    cudaFree(d_out);
    cudaFree(d_count);
    cudaFree(d_seeds);
    cudaFree(d_starts);
    cudaFree(d_startX);
    cudaFree(d_startY);
    cudaFree(d_stride);
    if(d_windows) cudaFree(d_windows);
    cudaStreamDestroy(stream);
}

void CudaPollardDevice::scanKeyRange(uint64_t start_k,
                                     uint64_t end_k,
                                     uint32_t windowBits,
                                     const uint32_t targetFragments[][MAX_OFFSETS],
                                     std::vector<PollardEngine::Constraint> &outConstraints) {
    uint32_t offsetsCount = static_cast<uint32_t>(_offsets.size());
    if(offsetsCount == 0 || windowBits == 0) {
        return;
    }
    uint32_t mask = (windowBits >= 32) ? 0xffffffffu : ((1u << windowBits) - 1u);

    uint32_t *d_offsets = nullptr;
    uint32_t (*d_targets)[MAX_OFFSETS] = nullptr;
    MatchRecord *d_out = nullptr;
    uint32_t *d_count = nullptr;

    cudaMalloc(&d_offsets, offsetsCount * sizeof(uint32_t));
    cudaMalloc(&d_targets, windowBits * MAX_OFFSETS * sizeof(uint32_t));
    cudaMalloc(&d_out, sizeof(MatchRecord) * 1024);
    cudaMalloc(&d_count, sizeof(uint32_t));

    cudaMemcpy(d_offsets, _offsets.data(), offsetsCount * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_targets, targetFragments, windowBits * MAX_OFFSETS * sizeof(uint32_t), cudaMemcpyHostToDevice);

    std::vector<MatchRecord> hostOut(1024);
    std::unordered_set<uint32_t> seen;

    cudaDeviceProp prop;
    int dev = 0;
    cudaGetDevice(&dev);
    cudaGetDeviceProperties(&prop, dev);

    unsigned int blockSize;
    if(_blockDim) {
        if(_blockDim % prop.warpSize != 0 || _blockDim > (unsigned)prop.maxThreadsPerBlock) {
            throw std::runtime_error("invalid blockDim");
        }
        blockSize = _blockDim;
    } else {
        blockSize = prop.warpSize * 8;
        if(blockSize > (unsigned)prop.maxThreadsPerBlock) {
            blockSize = prop.maxThreadsPerBlock;
        }
    }

    uint64_t chunk = (1ULL << 32);
    for(uint64_t chunkStart = start_k; chunkStart < end_k && seen.size() < offsetsCount; chunkStart += chunk) {
        uint64_t range = std::min(chunk, end_k - chunkStart);
        dim3 block(blockSize);
        unsigned int gridSize;
        if(_gridDim) {
            if(_gridDim > (unsigned)prop.maxGridSize[0]) {
                throw std::runtime_error("invalid gridDim");
            }
            gridSize = _gridDim;
        } else {
            gridSize = (range + block.x - 1) / block.x;
            if(gridSize > (unsigned)prop.maxGridSize[0]) {
                gridSize = prop.maxGridSize[0];
            }
        }
        dim3 grid(gridSize);

        cudaMemset(d_count, 0, sizeof(uint32_t));
        launchWindowKernel(grid, block, chunkStart, range, windowBits,
                           d_offsets, offsetsCount, mask, d_targets, d_out, d_count);

        uint32_t hCount = 0;
        cudaMemcpy(&hCount, d_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        if(hCount > hostOut.size()) hCount = hostOut.size();
        cudaMemcpy(hostOut.data(), d_out, hCount * sizeof(MatchRecord), cudaMemcpyDeviceToHost);

        for(uint32_t i = 0; i < hCount; ++i) {
            const MatchRecord &r = hostOut[i];
            if(!seen.insert(r.offset).second) {
                continue;
            }

            uint32_t modBits = r.offset + windowBits;
            secp256k1::uint256 rem(0);
            uint64_t frag = static_cast<uint64_t>(r.fragment & mask);
            uint32_t word = r.offset / 32;
            uint32_t bit = r.offset % 32;
            if(word < 8) {
                rem.v[word] |= static_cast<uint32_t>(frag << bit);
                if(bit && word + 1 < 8) {
                    rem.v[word + 1] |= static_cast<uint32_t>(frag >> (32 - bit));
                }
            }

            secp256k1::uint256 mod(0);
            if(modBits < 256) {
                mod.v[modBits / 32] = (1u << (modBits % 32));
            }
            PollardEngine::Constraint c{mod, rem};
            outConstraints.push_back(c);

            _engine.processWindow(0, r.offset, c);
        }
    }

    cudaFree(d_offsets);
    cudaFree(d_targets);
    cudaFree(d_out);
    cudaFree(d_count);
}

extern "C" bool runCudaHashWindowLE(const unsigned int h[5], unsigned int offset,
                                    unsigned int bits, unsigned int out[5]) {
    // Lightweight wrapper used by unit tests to validate the CUDA window
    // extraction logic.  Guard against invalid ranges so tests can detect
    // misuse.
    if(offset + bits > 160u) {
        return false;
    }
    uint256 v = hashWindowLE(h, offset, bits);
    v.exportWords(out, 5);
    return true;
}
