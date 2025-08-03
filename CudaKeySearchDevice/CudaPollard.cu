#include <stdint.h>
#include "../KeyFinder/PollardTypes.h"

struct CudaPollardMatch {
    unsigned long long k[4];
    unsigned int hash[5];
};

__device__ static unsigned long long xorshift64(unsigned long long &state)
{
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
}

__device__ static void fakeHash160(unsigned long long k, unsigned int hash[5])
{
    unsigned long long x = k;
    for(int i = 0; i < 5; i++) {
        xorshift64(x);
        hash[i] = static_cast<unsigned int>(x);
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
        unsigned long long state = seed;
        unsigned long long scalar = 0ULL;
        unsigned long long mask = (windowBits >= 64) ? 0xffffffffffffffffULL : ((1ULL << windowBits) - 1ULL);
        unsigned int count = 0;
        for(unsigned int i = 0; i < steps && count < maxOut; i++) {
            unsigned long long step = (xorshift64(state) & mask) + 1ULL;
            scalar += step;
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
