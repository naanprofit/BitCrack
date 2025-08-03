#include <stdint.h>
#include "../KeyFinder/PollardTypes.h"

struct CudaPollardMatch {
    unsigned long long k[4];
    unsigned int hash[5];
};

struct RNGState {
    unsigned long long s0;
    unsigned long long s1;
};

__device__ static inline unsigned long long xorshift128plus(RNGState &state)
{
    unsigned long long x = state.s0;
    unsigned long long y = state.s1;
    state.s0 = y;
    x ^= x << 23;
    x ^= x >> 17;
    x ^= y;
    x ^= y >> 26;
    state.s1 = x;
    return x + y;
}

__device__ static inline unsigned long long next_random_step(RNGState &state)
{
    const unsigned long long ORDER_MINUS_ONE = 0xBFD25E8CD0364140ULL;
    return (xorshift128plus(state) % ORDER_MINUS_ONE) + 1ULL;
}

__device__ static void fakeHash160(unsigned long long k, unsigned int hash[5])
{
    RNGState st{ k, k ^ 0x9E3779B97F4A7C15ULL };
    for(int i = 0; i < 5; i++) {
        hash[i] = static_cast<unsigned int>(xorshift128plus(st));
    }
}

extern "C" __global__ void pollardRandomWalk(CudaPollardMatch *out,
                                              unsigned int *outCount,
                                              unsigned int maxOut,
                                              unsigned long long seed,
                                              unsigned int steps,
                                              unsigned int windowBits)
{
    if(threadIdx.x == 0 && blockIdx.x == 0) {
        RNGState rng{ seed ^ 1ULL, seed + 1ULL };
        unsigned long long scalar = 0ULL;
        const unsigned long long ORDER = 0xBFD25E8CD0364141ULL;
        unsigned long long mask = (windowBits >= 64) ? 0xffffffffffffffffULL : ((1ULL << windowBits) - 1ULL);
        unsigned int count = 0;
        for(unsigned int i = 0; i < steps && count < maxOut; i++) {
            unsigned long long step = next_random_step(rng);
            scalar += step;
            scalar %= ORDER;
            if((scalar & mask) == 0ULL) {
                fakeHash160(scalar, out[count].hash);
                out[count].k[0] = scalar & 0xffffffffULL;
                out[count].k[1] = scalar >> 32;
                out[count].k[2] = 0ULL;
                out[count].k[3] = 0ULL;
                count++;
            }
        }
        *outCount = count;
    }
}
