#ifndef WINDOW_KERNEL_H
#define WINDOW_KERNEL_H

#include <cstdint>

// Minimal record describing a fragment match for ``k``.  Each entry records
// the bit ``offset`` within the x-coordinate, the extracted ``fragment`` and
// the corresponding scalar ``k`` where the fragment was observed.
struct MatchRecord {
    uint32_t offset;
    uint32_t fragment;
    uint64_t k;
};

// When compiling without NVCC we still need a definition of ``dim3`` so that
// host code including this header can declare grid and block dimensions.  NVCC
// provides a proper definition via <cuda_runtime.h>.
#ifndef __CUDACC__
struct dim3 {
    unsigned int x, y, z;
    dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1)
        : x(vx), y(vy), z(vz) {}
};
#endif

// Host-side wrapper used to launch ``windowKernel``.  ``gridDim`` and
// ``blockDim`` allow callers to customise the launch configuration.
extern "C" void launchWindowKernel(uint64_t start_k,
                                   uint64_t range_len,
                                   uint32_t ws,
                                   const uint32_t *offsets,
                                   uint32_t offsets_count,
                                   uint32_t mask,
                                   const uint32_t *target_frags,
                                   MatchRecord *out_buf,
                                   uint32_t *out_count,
                                   dim3 gridDim,
                                   dim3 blockDim);

#endif // WINDOW_KERNEL_H
