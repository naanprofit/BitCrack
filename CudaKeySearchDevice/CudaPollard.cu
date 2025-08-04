#include <stdint.h>
#include "../KeyFinder/PollardTypes.h"
#include "sha256.cuh"   // SHA256 hashing for public keys
#include "ripemd160.cuh" // RIPEMD160 finalisation
#include "secp256k1.cuh" // EC point operations
#include "ptx.cuh"       // byte order helpers

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
    unsigned int target[5];
};

struct RNGState {
    unsigned long long s0;
    unsigned long long s1;
};

// Output structure used by the legacy random walk kernel. Each entry
// stores the full 256-bit scalar of a distinguished point along with the
// corresponding hash160 digest so the host can perform window matching.
struct CudaPollardMatch {
    unsigned int k[8];
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

// Full secp256k1 group order expressed in little-endian words for scalar
// arithmetic.  Scalars within this file are represented in little-endian
// form where index 0 holds the least significant 32 bits.
static __device__ __constant__ unsigned int ORDER[8] = {
    0xD0364141U, 0xBFD25E8CU, 0xAF48A03BU, 0xBAAEDCE6U,
    0xFFFFFFFEU, 0xFFFFFFFFU, 0xFFFFFFFFU, 0xFFFFFFFFU
};

__device__ static inline bool isZero256(const unsigned int a[8]) {
    for(int i = 0; i < 8; ++i) {
        if(a[i] != 0U) return false;
    }
    return true;
}

__device__ static inline bool ge256(const unsigned int a[8], const unsigned int b[8]) {
    for(int i = 7; i >= 0; --i) {
        if(a[i] > b[i]) return true;
        if(a[i] < b[i]) return false;
    }
    return true; // equal
}

__device__ static inline void sub256(const unsigned int a[8], const unsigned int b[8], unsigned int r[8]) {
    unsigned int borrow = 0U;
    for(int i = 0; i < 8; ++i) {
        unsigned int ai = a[i];
        unsigned int bi = b[i];
        unsigned int t = ai - bi;
        unsigned int ri = t - borrow;
        borrow = (ai < bi) | (t < borrow);
        r[i] = ri;
    }
}

__device__ static inline void addModN(const unsigned int a[8], const unsigned int b[8], unsigned int r[8]) {
    unsigned int carry = 0U;
    for(int i = 0; i < 8; ++i) {
        unsigned int ai = a[i];
        unsigned int bi = b[i];
        unsigned int s = ai + bi;
        unsigned int ri = s + carry;
        carry = (s < ai) | (ri < s);
        r[i] = ri;
    }
    if(carry || ge256(r, ORDER)) {
        sub256(r, ORDER, r);
    }
}

// Generate a random step in the range [1, ORDER-1]
__device__ static inline void next_random_step(RNGState &state, unsigned int step[8]) {
    do {
        for(int i = 0; i < 4; ++i) {
            unsigned long long v = xorshift128plus(state);
            step[i * 2]     = (unsigned int)(v & 0xffffffffULL);
            step[i * 2 + 1] = (unsigned int)(v >> 32);
        }
    } while(isZero256(step) || ge256(step, ORDER));
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
        hOut[i] = hIn[i] + iv[i];
    }
}

// Extract ``bits`` bits starting at ``offset`` from the RIPEMD160 digest
// ``h``.  Bits are interpreted in little-endian order and returned in the
// lower bits of the 160-bit structure.
struct Hash160 {
    unsigned int v[5];
};

__device__ Hash160 hashWindow(const unsigned int h[5], unsigned int offset, unsigned int bits)
{
    Hash160 out;
    for(int i = 0; i < 5; ++i) {
        out.v[i] = 0u;
    }
    unsigned int word = offset / 32;
    unsigned int bit  = offset % 32;
    unsigned int span = bit + bits;
    unsigned int words = (span + 31) / 32;
    for(unsigned int i = 0; i < words && word + i < 5; ++i) {
        unsigned long long val = ((unsigned long long)h[word + i]) >> bit;
        if(bit && word + i + 1 < 5) {
            val |= ((unsigned long long)h[word + i + 1]) << (32 - bit);
        }
        out.v[i] = (unsigned int)(val & 0xffffffffULL);
    }
    if(span % 32) {
        unsigned int mask = (1u << (span % 32)) - 1u;
        out.v[words - 1] &= mask;
    }
    for(unsigned int i = words; i < 5; ++i) {
        out.v[i] = 0u;
    }
    return out;
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

__device__ static void scalarMultiplySmall(const unsigned int bx[8], const unsigned int by[8],
                                           const unsigned int k[8], unsigned int rx[8], unsigned int ry[8])
{
    setPointInfinity(rx, ry);
    unsigned int qx[8];
    unsigned int qy[8];
    copyBigInt(bx, qx);
    copyBigInt(by, qy);
    for(int i = 0; i < 4; ++i) {
        unsigned int word = k[i];
        for(int bit = 0; bit < 32; ++bit) {
            if(word & 1U) {
                unsigned int tx[8];
                unsigned int ty[8];
                pointAdd(rx, ry, qx, qy, tx, ty);
                copyBigInt(tx, rx);
                copyBigInt(ty, ry);
            }
            word >>= 1U;
            unsigned int tx[8];
            unsigned int ty[8];
            pointDouble(qx, qy, tx, ty);
            copyBigInt(tx, qx);
            copyBigInt(ty, qy);
        }
    }
}

// Multiply the secp256k1 base point by a 256-bit scalar ``k`` (little endian)
__device__ static void scalarMultiplyBase(const unsigned int k[8], unsigned int rx[8], unsigned int ry[8])
{
    GLVScalarSplit split;
    splitScalar(k, split);

    unsigned int r1x[8];
    unsigned int r1y[8];
    scalarMultiplySmall(_GX, _GY, split.k1, r1x, r1y);
    if(split.k1Neg) {
        unsigned int ny[8];
        negModP(r1y, ny);
        copyBigInt(ny, r1y);
    }

    if(isZero256(split.k2)) {
        copyBigInt(r1x, rx);
        copyBigInt(r1y, ry);
        return;
    }

    unsigned int base2x[8];
    unsigned int base2y[8];
    mulModP(_GX, _BETA, base2x);
    copyBigInt(_GY, base2y);
    unsigned int r2x[8];
    unsigned int r2y[8];
    scalarMultiplySmall(base2x, base2y, split.k2, r2x, r2y);
    if(split.k2Neg) {
        unsigned int ny[8];
        negModP(r2y, ny);
        copyBigInt(ny, r2y);
    }

    pointAdd(r1x, r1y, r2x, r2y, rx, ry);
}

// Legacy kernel retained for backwards compatibility. It performs a
// random walk and records distinguished points (those where the low
// ``windowBits`` bits of the scalar are zero), outputting the scalar and
// its hash160 digest.
extern "C" __global__ void pollardRandomWalk(CudaPollardMatch *out,
                                             unsigned int *outCount,
                                             unsigned int maxOut,
                                             const unsigned int *seeds,
                                             const unsigned int *starts,
                                             const unsigned int *startX,
                                             const unsigned int *startY,
                                             unsigned int steps,
                                             unsigned int windowBits)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int scalar[8];
    for(int i = 0; i < 8; ++i) {
        scalar[i] = starts[tid * 8 + i];
    }
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

    unsigned long long s0 = ((unsigned long long)seeds[tid*8 + 1] << 32) | seeds[tid*8 + 0];
    unsigned long long s1 = ((unsigned long long)seeds[tid*8 + 3] << 32) | seeds[tid*8 + 2];
    RNGState rng{ s0 ^ 1ULL, s1 + 1ULL };

    unsigned int mask[8];
    for(int i = 0; i < 8; ++i) mask[i] = 0U;
    unsigned int fullWords = windowBits / 32;
    for(unsigned int i = 0; i < fullWords && i < 8; ++i) mask[i] = 0xffffffffU;
    if(windowBits % 32 && fullWords < 8) {
        mask[fullWords] = (1U << (windowBits % 32)) - 1U;
    }

    for(unsigned int i = 0; i < steps; ++i) {
        unsigned int step[8];
        next_random_step(rng, step);
        addModN(scalar, step, scalar);

        unsigned int sx[8];
        unsigned int sy[8];
        scalarMultiplyBase(step, sx, sy);
        unsigned int tx[8];
        unsigned int ty[8];
        pointAdd(px, py, sx, sy, tx, ty);
        copyBigInt(tx, px);
        copyBigInt(ty, py);

        bool distinguished = true;
        for(int j = 0; j < 8; ++j) {
            if((scalar[j] & mask[j]) != 0U) { distinguished = false; break; }
        }
        if(distinguished) {
            unsigned int digest[5];
            unsigned int finalHash[5];
            hashPublicKeyCompressed(px, py[7] & 1, digest);
            doRMD160FinalRound(digest, finalHash);
            for(int j = 0; j < 5; ++j) {
                finalHash[j] = endian(finalHash[j]);
            }

            unsigned int idx = atomicAdd(outCount, 1u);
            if(idx < maxOut) {
                for(int j = 0; j < 8; ++j) {
                    out[idx].k[j] = scalar[j];
                }
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
                                       const unsigned int *seeds,
                                       const unsigned int *starts,
                                       const unsigned int *startX,
                                       const unsigned int *startY,
                                       unsigned int steps,
                                       const TargetWindow *windows,
                                       unsigned int windowCount,
                                       const unsigned int *strides)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int scalar[8];
    for(int i = 0; i < 8; ++i) {
        scalar[i] = starts[tid * 8 + i];
    }
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

    unsigned int stride[8];
    for(int i = 0; i < 8; ++i) {
        stride[i] = strides ? strides[tid * 8 + i] : 0U;
    }

    if(isZero256(stride)) {
        unsigned long long s0 = ((unsigned long long)seeds[tid*8 + 1] << 32) | seeds[tid*8 + 0];
        unsigned long long s1 = ((unsigned long long)seeds[tid*8 + 3] << 32) | seeds[tid*8 + 2];
        RNGState rng{ s0 ^ 1ULL, s1 + 1ULL };
        for(unsigned int i = 0; i < steps; ++i) {
            unsigned int step[8];
            next_random_step(rng, step);
            addModN(scalar, step, scalar);

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
            hashPublicKeyCompressed(px, py[7] & 1, digest);
            doRMD160FinalRound(digest, finalHash);
            for(int j = 0; j < 5; ++j) {
                finalHash[j] = endian(finalHash[j]);
            }

            for(unsigned int w = 0; w < windowCount; ++w) {
                TargetWindow tw = windows[w];
                Hash160 hv = hashWindow(finalHash, tw.offset, tw.bits);
                unsigned int words = (tw.bits + 31) / 32;
                bool match = true;
                for(unsigned int j = 0; j < words; ++j) {
                    if(hv.v[j] != tw.target[j]) { match = false; break; }
                }
                if(match) {
                    unsigned int idx = atomicAdd(outCount, 1u);
                    if(idx < maxOut) {
                        out[idx].targetIdx = tw.targetIdx;
                        out[idx].offset    = tw.offset;
                        out[idx].bits      = tw.bits;
                        for(int j = 0; j < 8; ++j) {
                            out[idx].k[j] = scalar[j];
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
            hashPublicKeyCompressed(px, py[7] & 1, digest);
            doRMD160FinalRound(digest, finalHash);
            for(int j = 0; j < 5; ++j) {
                finalHash[j] = endian(finalHash[j]);
            }

            for(unsigned int w = 0; w < windowCount; ++w) {
                TargetWindow tw = windows[w];
                Hash160 hv = hashWindow(finalHash, tw.offset, tw.bits);
                unsigned int words = (tw.bits + 31) / 32;
                bool match = true;
                for(unsigned int j = 0; j < words; ++j) {
                    if(hv.v[j] != tw.target[j]) { match = false; break; }
                }
                if(match) {
                    unsigned int idx = atomicAdd(outCount, 1u);
                    if(idx < maxOut) {
                        out[idx].targetIdx = tw.targetIdx;
                        out[idx].offset    = tw.offset;
                        out[idx].bits      = tw.bits;
                        for(int j = 0; j < 8; ++j) {
                            out[idx].k[j] = scalar[j];
                        }
                    }
                }
            }

            addModN(scalar, stride, scalar);
            unsigned int tx[8];
            unsigned int ty[8];
            pointAdd(px, py, sx, sy, tx, ty);
            copyBigInt(tx, px);
            copyBigInt(ty, py);
        }
    }
}
