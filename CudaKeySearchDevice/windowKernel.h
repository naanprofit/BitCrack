#ifndef WINDOW_KERNEL_H
#define WINDOW_KERNEL_H

#include <cstdint>

// ``cuda_runtime.h`` is only available when compiling with NVCC.  Provide a
// lightweight definition of ``dim3`` for host-only builds so files including
// this header do not need the CUDA SDK.
#ifndef __CUDACC__
struct dim3 {
    unsigned int x, y, z;
    dim3(unsigned int a = 1, unsigned int b = 1, unsigned int c = 1)
        : x(a), y(b), z(c) {}
};
#endif

// Minimal record emitted by ``windowKernel`` describing a matching window.
struct MatchRecord {
    uint32_t offset;      // bit offset of the window
    uint32_t fragment;    // extracted fragment of the x-coordinate
    uint64_t k;           // scalar where the match occurred
};

// Host-side wrapper used to launch ``windowKernel`` from C++ code.  The
// implementation is provided in ``windowKernel.cu`` which is always compiled
// with NVCC when ``BUILD_CUDA=1``.
extern "C" void launchWindowKernel(uint64_t start_k,
                                   uint64_t range_len,
                                   uint32_t ws,
                                   const uint32_t *offsets,
                                   uint32_t offsets_count,
                                   uint32_t mask,
                                   const uint32_t *target_frags,
                                   MatchRecord *out_buf,
                                   uint32_t *out_count);

#endif // WINDOW_KERNEL_H
