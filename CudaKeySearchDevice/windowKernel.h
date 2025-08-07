// SPDX-License-Identifier: MIT
#ifndef WINDOW_KERNEL_H
#define WINDOW_KERNEL_H

#include <cstdint>

// Simple POD describing a fragment match produced by ``windowKernel``.
struct MatchRecord { uint32_t offset, fragment; uint64_t k; };

// Provide a lightweight ``dim3`` definition for builds that don't pull in the
// CUDA runtime headers. When ``cuda_runtime.h`` is available (either because
// ``__CUDACC__`` is defined or the header has already been included) we rely on
// the official definition to avoid clashes.
#if !defined(__CUDACC__) && !defined(__CUDA_RUNTIME_H__)
struct dim3 { unsigned int x, y, z; dim3(unsigned int a=1,unsigned int b=1,unsigned int c=1):x(a),y(b),z(c){} };
#endif

// Host wrapper launching the GPU kernel. Grid configuration is chosen
// internally; callers only specify the range and window parameters.
extern "C" void launchWindowKernel(uint64_t start_k,
                                   uint64_t range_len,
                                   uint32_t ws,
                                   const uint32_t *d_offsets,
                                   uint32_t offsetsCount,
                                   uint32_t mask,
                                   const uint32_t *d_target_frags,
                                   MatchRecord *d_out_buf,
                                   uint32_t *d_out_count,
                                   dim3 grid = dim3(0),
                                   dim3 block = dim3(0));

#endif // WINDOW_KERNEL_H
