#ifndef WINDOW_KERNEL_H
#define WINDOW_KERNEL_H

#include <cstdint>


// ``cuda_runtime.h`` defines ``dim3``.  When building without CUDA headers we
// provide a small stand-in to keep interfaces compatible.  Extra guards prevent
// conflicts if this header is included before the CUDA headers.
#if !defined(__CUDACC__) && !defined(__CUDA_RUNTIME_H__) && !defined(__VECTOR_TYPES_H__)

struct dim3 {
    unsigned int x, y, z;
    dim3(unsigned int a = 1u, unsigned int b = 1u, unsigned int c = 1u)
        : x(a), y(b), z(c) {}
};
#endif

// Minimal record emitted by ``windowKernel`` describing a matching window.
struct MatchRecord {
    uint32_t offset;   // bit offset of the window
    uint32_t fragment; // extracted fragment of the x-coordinate
    uint64_t k;        // scalar where the match occurred
};

// Host-side wrapper used to launch ``windowKernel`` from C++ code.
extern "C" void launchWindowKernel(dim3 grid,
                                   dim3 block,
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
