#ifndef WINDOW_KERNEL_H
#define WINDOW_KERNEL_H

#include <cstdint>

// When compiling without CUDA headers (e.g. unit tests on the host), provide a
// minimal ``dim3`` replacement to keep the interface compatible.  The
// definition is skipped when the real CUDA ``dim3`` is available to avoid
// multiple-definition errors.
#ifndef __CUDACC__
struct dim3 { unsigned int x, y, z; dim3(unsigned int a=1, unsigned int b=1, unsigned int c=1) : x(a), y(b), z(c) {} };
#else
#include <cuda_runtime.h>
#endif

// Minimal record emitted by ``windowKernel`` describing a matching window.
struct MatchRecord {
    uint32_t offset;    // bit offset of the window
    uint32_t fragment;  // extracted fragment of the x-coordinate
    uint64_t k;         // scalar where the match occurred
};

// Host-side wrapper used to launch ``windowKernel`` from C++ code.  The grid
// and block dimensions are passed in so callers can tune occupancy as
// required.
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
