#include <stdint.h>
#include "../KeyFinder/PollardTypes.h"
#include "sha256.cuh"   // SHA256 hashing for public keys
#include "ripemd160.cuh" // RIPEMD160 finalisation
#include "secp256k1.cuh" // EC point operations

__device__ void hashPublicKeyCompressed(const unsigned int*, unsigned int, unsigned int*);

// Result written by the kernel when a hash window matches a target.
struct GpuPollardWindow {
    unsigned int targetIdx;
    unsigned int offset;
    unsigned int bits;
    unsigned int k[8];
};

// Description of a window to test for each step.
struct TargetWindow {
    unsigned int targetIdx;
    unsigned int offset;
    unsigned int bits;
    unsigned long long target;
};

struct RNGState {
    unsigned long long s0;
    unsigned long long s1;
};

// Output structure used by the legacy random walk kernel. Each entry
// stores the 64-bit scalar of a distinguished point along with the
// corresponding hash160 digest so the host can perform window matching.
struct CudaPollardMatch {
    unsigned long long k[4];
    unsigned int hash[5];
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

static __device__ __forceinline__ void doRMD160FinalRound(const unsigned int hIn[5], unsigned int hOut[5])
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

// Extract ``bits`` bits starting at ``offset`` from the RIPEMD160 digest
// ``h``.  Bits are interpreted in little-endian order.
__device__ unsigned long long hashWindow(const unsigned int h[5],
                                         unsigned int offset,
                                         unsigned int bits)
{
    unsigned int word = offset / 32;
    unsigned int bit  = offset % 32;
    unsigned long long val = 0ULL;
    if(word < 5) {
        val = ((unsigned long long)h[word]) >> bit;
        if(bit + bits > 32 && word + 1 < 5) {
            val |= ((unsigned long long)h[word + 1]) << (32 - bit);
        }
    }
    if(bit + bits > 64 && word + 2 < 5) {
        val |= ((unsigned long long)h[word + 2]) << (64 - bit);
    }
    if(bits < 64) {
        unsigned long long mask = (bits == 64) ? 0xffffffffffffffffULL : ((1ULL << bits) - 1ULL);
        val &= mask;
    }
    return val;
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

// Legacy kernel retained for backwards compatibility. It performs a
// random walk and records distinguished points (those where the low
// ``windowBits`` bits of the scalar are zero), outputting the scalar and
// its hash160 digest.
extern "C" __global__ void pollardRandomWalk(CudaPollardMatch *out,
                                             unsigned int *outCount,
                                             unsigned int maxOut,
                                             const unsigned long long *seeds,
                                             const unsigned long long *starts,
                                             const unsigned int *startX,
                                             const unsigned int *startY,
                                             unsigned int steps,
                                             unsigned int windowBits)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long scalar = starts[tid];
    const unsigned long long ORDER = 0xBFD25E8CD0364141ULL;
    unsigned int px[8];
    unsigned int py[8];

    if(startX && startY) {
        for(int i = 0; i < 8; ++i) {
            px[i] = startX[tid * 8 + i];
            py[i] = startY[tid * 8 + i];
        }
    } else {
        scalarMultiplyBase(scalar, px, py);
    }

    RNGState rng{ seeds[tid] ^ 1ULL, seeds[tid] + 1ULL };
    unsigned long long mask = (windowBits >= 64) ? 0xffffffffffffffffULL : ((1ULL << windowBits) - 1ULL);

    for(unsigned int i = 0; i < steps; ++i) {
        unsigned long long step = next_random_step(rng);
        scalar += step;
        scalar %= ORDER;

        unsigned int sx[8];
        unsigned int sy[8];
        scalarMultiplyBase(step, sx, sy);
        unsigned int tx[8];
        unsigned int ty[8];
        pointAdd(px, py, sx, sy, tx, ty);
        copyBigInt(tx, px);
        copyBigInt(ty, py);

        if((scalar & mask) == 0ULL) {
            unsigned int digest[5];
            unsigned int finalHash[5];
            // py[0] holds the least significant word; py[0] & 1 yields the parity bit
            hashPublicKeyCompressed(px, py[0] & 1, digest);
            doRMD160FinalRound(digest, finalHash);

            unsigned int idx = atomicAdd(outCount, 1u);
            if(idx < maxOut) {
                out[idx].k[0] = scalar;
                out[idx].k[1] = 0ULL;
                out[idx].k[2] = 0ULL;
                out[idx].k[3] = 0ULL;
                for(int j = 0; j < 5; ++j) {
                    out[idx].hash[j] = finalHash[j];
                }
            }
        }
    }
}

extern "C" __global__ void pollardWalk(GpuPollardWindow *out,
                                       unsigned int *outCount,
                                       unsigned int maxOut,
                                       const unsigned long long *seeds,
                                       const unsigned long long *starts,
                                       const unsigned int *startX,
                                       const unsigned int *startY,
                                       unsigned int steps,
                                       const TargetWindow *windows,
                                       unsigned int windowCount,
                                       unsigned long long stride)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long scalar = starts[tid];
    const unsigned long long ORDER = 0xBFD25E8CD0364141ULL;
    unsigned int px[8];
    unsigned int py[8];

    if(startX && startY) {
        for(int i = 0; i < 8; ++i) {
            px[i] = startX[tid * 8 + i];
            py[i] = startY[tid * 8 + i];
        }
    } else {
        scalarMultiplyBase(scalar, px, py);
    }

    if(stride == 0ULL) {
        RNGState rng{ seeds[tid] ^ 1ULL, seeds[tid] + 1ULL };
        for(unsigned int i = 0; i < steps; ++i) {
            unsigned long long step = next_random_step(rng);
            scalar += step;
            scalar %= ORDER;

            unsigned int sx[8];
            unsigned int sy[8];
            scalarMultiplyBase(step, sx, sy);
            unsigned int tx[8];
            unsigned int ty[8];
            pointAdd(px, py, sx, sy, tx, ty);
            copyBigInt(tx, px);
            copyBigInt(ty, py);

            unsigned int digest[5];
            unsigned int finalHash[5];
            // py[0] holds the least significant word; py[0] & 1 yields the parity bit
            hashPublicKeyCompressed(px, py[0] & 1, digest);
            doRMD160FinalRound(digest, finalHash);

            for(unsigned int w = 0; w < windowCount; ++w) {
                TargetWindow tw = windows[w];
                unsigned long long hv = hashWindow(finalHash, tw.offset, tw.bits);
                if(hv == tw.target) {
                    unsigned int idx = atomicAdd(outCount, 1u);
                    if(idx < maxOut) {
                        out[idx].targetIdx = tw.targetIdx;
                        out[idx].offset    = tw.offset;
                        out[idx].bits      = tw.bits;
                        unsigned int modBits = tw.offset + tw.bits;
                        unsigned long long mask = (modBits >= 64) ? 0xffffffffffffffffULL : ((1ULL << modBits) - 1ULL);
                        unsigned long long frag = scalar & mask;
                        out[idx].k[0] = (unsigned int)(frag & 0xffffffffULL);
                        out[idx].k[1] = (unsigned int)(frag >> 32);
                        for(int j = 2; j < 8; ++j) {
                            out[idx].k[j] = 0U;
                        }
                    }
                }
            }
        }
    } else {
        unsigned int sx[8];
        unsigned int sy[8];
        scalarMultiplyBase(stride, sx, sy);
        for(unsigned int i = 0; i < steps; ++i) {
            unsigned int digest[5];
            unsigned int finalHash[5];
            // py[0] holds the least significant word; py[0] & 1 yields the parity bit
            hashPublicKeyCompressed(px, py[0] & 1, digest);
            doRMD160FinalRound(digest, finalHash);

            for(unsigned int w = 0; w < windowCount; ++w) {
                TargetWindow tw = windows[w];
                unsigned long long hv = hashWindow(finalHash, tw.offset, tw.bits);
                if(hv == tw.target) {
                    unsigned int idx = atomicAdd(outCount, 1u);
                    if(idx < maxOut) {
                        out[idx].targetIdx = tw.targetIdx;
                        out[idx].offset    = tw.offset;
                        out[idx].bits      = tw.bits;
                        unsigned int modBits = tw.offset + tw.bits;
                        unsigned long long mask = (modBits >= 64) ? 0xffffffffffffffffULL : ((1ULL << modBits) - 1ULL);
                        unsigned long long frag = scalar & mask;
                        out[idx].k[0] = (unsigned int)(frag & 0xffffffffULL);
                        out[idx].k[1] = (unsigned int)(frag >> 32);
                        for(int j = 2; j < 8; ++j) {
                            out[idx].k[j] = 0U;
                        }
                    }
                }
            }

            scalar += stride;
            scalar %= ORDER;
            unsigned int tx[8];
            unsigned int ty[8];
            pointAdd(px, py, sx, sy, tx, ty);
            copyBigInt(tx, px);
            copyBigInt(ty, py);
        }
    }
}
