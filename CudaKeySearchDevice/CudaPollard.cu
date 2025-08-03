#include <stdint.h>
#include "../KeyFinder/PollardTypes.h"
#include "sha256.cuh"
#include "ripemd160.cuh"
#include "secp256k1.cuh"

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

__device__ void doRMD160FinalRound(const unsigned int hIn[5], unsigned int hOut[5])
{
    const unsigned int iv[5] = {
        0x67452301,
        0xefcdab89,
        0x98badcfe,
        0x10325476,
        0xc3d2e1f0
    };

    for(int i = 0; i < 5; i++) {
        hOut[i] = endian(hIn[i] + iv[(i + 1) % 5]);
    }
}

__device__ static void setPointInfinity(unsigned int x[8], unsigned int y[8])
{
    for(int i = 0; i < 8; i++) {
        x[i] = 0xffffffff;
        y[i] = 0xffffffff;
    }
}

__device__ static void pointDouble(const unsigned int x[8], const unsigned int y[8], unsigned int rx[8], unsigned int ry[8])
{
    if(isInfinity(x)) {
        setPointInfinity(rx, ry);
        return;
    }

    unsigned int x2[8];
    unsigned int three_x2[8];
    unsigned int two_y[8];
    unsigned int inv[8];
    unsigned int lambda[8];
    unsigned int lambda2[8];
    unsigned int k[8];

    mulModP(x, x, x2);
    addModP(x2, x2, three_x2);
    addModP(three_x2, x2, three_x2);

    addModP(y, y, two_y);
    invModP(two_y, inv);
    mulModP(three_x2, inv, lambda);

    mulModP(lambda, lambda, lambda2);
    subModP(lambda2, x, rx);
    subModP(rx, x, rx);

    subModP(x, rx, k);
    mulModP(lambda, k, ry);
    subModP(ry, y, ry);
}

__device__ static void pointAdd(const unsigned int ax[8], const unsigned int ay[8],
                                const unsigned int bx[8], const unsigned int by[8],
                                unsigned int rx[8], unsigned int ry[8])
{
    if(isInfinity(ax)) {
        copyBigInt(bx, rx);
        copyBigInt(by, ry);
        return;
    }
    if(isInfinity(bx)) {
        copyBigInt(ax, rx);
        copyBigInt(ay, ry);
        return;
    }
    if(equal(ax, bx) && equal(ay, by)) {
        pointDouble(ax, ay, rx, ry);
        return;
    }

    unsigned int rise[8];
    unsigned int run[8];
    unsigned int inv[8];
    unsigned int lambda[8];
    unsigned int lambda2[8];
    unsigned int k[8];

    subModP(by, ay, rise);
    subModP(bx, ax, run);
    invModP(run, inv);
    mulModP(rise, inv, lambda);

    mulModP(lambda, lambda, lambda2);
    subModP(lambda2, ax, rx);
    subModP(rx, bx, rx);

    subModP(ax, rx, k);
    mulModP(lambda, k, ry);
    subModP(ry, ay, ry);
}

__device__ static void scalarMultiplyBase(unsigned long long k, unsigned int rx[8], unsigned int ry[8])
{
    setPointInfinity(rx, ry);
    if(k == 0ULL) {
        return;
    }

    unsigned int qx[8];
    unsigned int qy[8];
    copyBigInt(_GX, qx);
    copyBigInt(_GY, qy);

    while(k) {
        if(k & 1ULL) {
            unsigned int tx[8];
            unsigned int ty[8];
            pointAdd(rx, ry, qx, qy, tx, ty);
            copyBigInt(tx, rx);
            copyBigInt(ty, ry);
        }
        k >>= 1ULL;
        if(k) {
            unsigned int tx[8];
            unsigned int ty[8];
            pointDouble(qx, qy, tx, ty);
            copyBigInt(tx, qx);
            copyBigInt(ty, qy);
        }
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
        unsigned int px[8];
        unsigned int py[8];
        setPointInfinity(px, py);
        for(unsigned int i = 0; i < steps && count < maxOut; i++) {
            unsigned long long step = next_random_step(rng);
            scalar += step;
            scalar %= ORDER;

            unsigned int sx[8];
            unsigned int sy[8];
            scalarMultiplyBase(step, sx, sy);
            if(isInfinity(px)) {
                copyBigInt(sx, px);
                copyBigInt(sy, py);
            } else {
                unsigned int tx[8];
                unsigned int ty[8];
                pointAdd(px, py, sx, sy, tx, ty);
                copyBigInt(tx, px);
                copyBigInt(ty, py);
            }

            if((scalar & mask) == 0ULL) {
                unsigned int digest[5];
                unsigned int finalHash[5];
                hashPublicKeyCompressed(px, py[7], digest);
                doRMD160FinalRound(digest, finalHash);
                for(int w = 0; w < 5; w++) {
                    out[count].hash[w] = finalHash[w];
                }
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
