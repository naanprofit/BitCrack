#ifndef WINDOW_KERNEL_H
#define WINDOW_KERNEL_H

#include <cstdint>

#ifndef __CUDACC__
struct dim3 {
    unsigned int x, y, z;
    dim3(unsigned int a = 1u, unsigned int b = 1u, unsigned int c = 1u)
        : x(a), y(b), z(c) {}
};
#endif

struct MatchRecord {
    uint32_t offset;
    uint32_t fragment;
    uint64_t k;
};

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
