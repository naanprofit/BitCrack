#ifndef WINDOW_KERNEL_H
#define WINDOW_KERNEL_H

#include <cstdint>
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

#endif // WINDOW_KERNEL_H
