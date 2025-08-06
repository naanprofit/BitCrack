// SPDX-License-Identifier: MIT
#ifndef WINDOW_KERNEL_H
#define WINDOW_KERNEL_H

#include <cstdint>

// Simple POD describing a fragment match produced by ``windowKernel``.
struct MatchRecord { uint32_t offset, fragment; uint64_t k; };

// Provide a lightweight ``dim3`` definition for non-CUDA compilation units so
// that host code can still declare grid dimensions when testing.  When
// compiled with NVCC we pull in ``cuda_runtime.h`` for the real definition.
#ifdef __CUDACC__
#include <cuda_runtime.h>
#else
struct dim3 { unsigned int x, y, z; dim3(unsigned int a=1,unsigned int b=1,unsigned int c=1):x(a),y(b),z(c){} };
#endif

// Host wrapper launching the GPU kernel. The caller specifies the grid and
// block configuration explicitly in order to mirror standard CUDA launches.
extern "C" void launchWindowKernel(dim3 grid, dim3 block,
                                   uint64_t start_k, uint64_t range_len,
                                   uint32_t ws, const uint32_t* offsets,
                                   uint32_t offsets_count, uint32_t mask,
                                   const uint32_t* target_frags,
                                   MatchRecord* out_buf,
                                   uint32_t* out_count);

#endif // WINDOW_KERNEL_H
