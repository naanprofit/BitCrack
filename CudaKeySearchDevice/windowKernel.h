// SPDX-License-Identifier: MIT
#ifndef WINDOW_KERNEL_H
#define WINDOW_KERNEL_H

#include <cstdint>
#include <cuda_runtime.h>

// Simple POD describing a fragment match produced by ``windowKernel``.
struct MatchRecord { uint32_t offset, fragment; uint64_t k; };

// Host wrapper launching the GPU kernel. ``grid`` and ``block`` control the
// launch configuration while the remaining parameters describe the range and
// window extraction settings.
extern "C" void launchWindowKernel(dim3 grid,
                                   dim3 block,
                                   uint64_t start_k,
                                   uint64_t range_len,
                                   uint32_t window_bits,
                                   const uint32_t *d_offsets,
                                   uint32_t offsets_count,
                                   const uint32_t *d_targets,
                                   MatchRecord *d_out,
                                   uint32_t *d_out_count,
                                   unsigned int max_out,
                                   cudaStream_t stream = 0);

#endif // WINDOW_KERNEL_H
