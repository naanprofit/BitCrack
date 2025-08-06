#ifndef WINDOW_KERNEL_H
#define WINDOW_KERNEL_H

#include <cstdint>
#include <cuda_runtime.h>

// Minimal record emitted by ``windowKernel`` describing a matching window.
struct MatchRecord {
    uint32_t offset;      // bit offset of the window
    uint32_t fragment;    // extracted fragment of the x-coordinate
    uint64_t k;           // scalar where the match occurred
};

#ifdef __CUDACC__
extern "C" __global__ void windowKernel(uint64_t start_k,
                                         uint64_t range_len,
                                         uint32_t ws,
                                         const uint32_t *offsets,
                                         uint32_t offsets_count,
                                         uint32_t mask,
                                         const uint32_t *target_frags,
                                         MatchRecord *out_buf,
                                         uint32_t *out_count);
#endif

// Host-side wrapper used to launch ``windowKernel`` from C++ code.
extern "C" void launchWindowKernel(dim3 gridDim,
                                   dim3 blockDim,
                                   uint64_t start_k,
                                   uint64_t range_len,
                                   uint32_t ws,
                                   const uint32_t *offsets,
                                   uint32_t offsets_count,
                                   uint32_t mask,
                                   const uint32_t *target_frags,
                                   MatchRecord *out_buf,
                                   uint32_t *out_count);

#endif // WINDOW_KERNEL_H
