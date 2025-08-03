#include <stdint.h>
#include "../KeyFinder/PollardTypes.h"

struct CudaPollardMatch {
    unsigned long long k[4];
    unsigned int hash[5];
};

extern "C" __global__ void pollardRandomWalk(CudaPollardMatch *out, unsigned long long seed)
{
    if(threadIdx.x == 0 && blockIdx.x == 0) {
        for(int i=0;i<4;i++) out[0].k[i] = seed;
        for(int i=0;i<5;i++) out[0].hash[i] = 0;
    }
}
