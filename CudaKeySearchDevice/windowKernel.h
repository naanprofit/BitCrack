// SPDX-License-Identifier: MIT
#ifndef WINDOW_KERNEL_H
#define WINDOW_KERNEL_H

#include <cstdint>

// Simple POD describing a fragment match produced by ``windowKernel``.
struct MatchRecord { uint32_t offset, fragment; uint64_t k; };

// Provide a lightweight ``dim3`` definition for non-CUDA compilation units so
// that host code can still declare grid dimensions when testing.
#ifndef __CUDACC__
struct dim3 { unsigned int x, y, z; dim3(unsigned int a=1,unsigned int b=1,unsigned int c=1):x(a),y(b),z(c){} };
#endif

// Host wrapper launching the GPU kernel. Grid configuration is chosen
// internally; callers only specify the range and window parameters.
extern "C" void launchWindowKernel(uint64_t start_k, uint64_t range_len,
                                   uint32_t ws, const uint32_t* offsets,
                                   uint32_t offsets_count, uint32_t mask,
                                   const uint32_t* target_frags,
                                   MatchRecord* out_buf,
                                   uint32_t* out_count);

#endif // WINDOW_KERNEL_H
