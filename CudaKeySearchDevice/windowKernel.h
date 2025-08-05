#ifndef WINDOW_KERNEL_H
#define WINDOW_KERNEL_H

#include <cstdint>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#else
struct dim3 { unsigned int x, y, z; dim3(unsigned int vx=1, unsigned int vy=1, unsigned int vz=1) : x(vx), y(vy), z(vz) {} };
#endif
#include "../KeyFinder/PollardTypes.h"

#ifdef __CUDACC__
extern "C" __global__ void windowKernel(uint64_t start_k,
                                         uint64_t range_len,
                                         int ws,
                                         const uint32_t *offsets,
                                         uint32_t mask,
                                         const uint32_t *target_frags,
                                         MatchRecord *out_buf,
                                         unsigned int *out_count);
#endif

extern "C" void launchWindowKernel(dim3 gridDim,
                                   dim3 blockDim,
                                   uint64_t start_k,
                                   uint64_t range_len,
                                   int ws,
                                   const uint32_t *offsets,
                                   uint32_t mask,
                                   const uint32_t *target_frags,
                                   MatchRecord *out_buf,
                                   unsigned int *out_count);

#endif // WINDOW_KERNEL_H
