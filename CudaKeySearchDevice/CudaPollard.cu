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
    unsigned int target[5];
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
    unsigned long long borrow = 0ULL;
    for(int i = 0; i < 8; ++i) {
        unsigned long long diff = (unsigned long long)a[i] - b[i] - borrow;
        r[i] = (unsigned int)diff;
        borrow = (diff >> 63) & 1ULL;
    }
}

__device__ static inline void addModN(const unsigned int a[8], const unsigned int b[8], unsigned int r[8]) {
    unsigned long long carry = 0ULL;
    for(int i = 0; i < 8; ++i) {
        unsigned long long sum = (unsigned long long)a[i] + b[i] + carry;
        r[i] = (unsigned int)sum;
        carry = sum >> 32;
    }
    if(carry || ge256(r, ORDER)) {
        sub256(r, ORDER, r);
    }
}

__device__ static inline void addModNU64(unsigned int a[8], unsigned long long b) {
    unsigned int t[8];
    t[0] = (unsigned int)b;
    t[1] = (unsigned int)(b >> 32);
    for(int i = 2; i < 8; ++i) t[i] = 0U;
    addModN(a, t, a);
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
        hOut[i] = endian(hIn[i] + iv[(i + 1) % 5]);
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
    unsigned int words = (bits + 31) / 32;
    for(unsigned int i = 0; i < words && word + i < 5; ++i) {
        unsigned long long val = ((unsigned long long)h[word + i]) >> bit;
        if(bit && word + i + 1 < 5) {
            val |= ((unsigned long long)h[word + i + 1]) << (32 - bit);
        }
        out.v[i] = (unsigned int)(val & 0xffffffffULL);
    }
    if(bits % 32) {
        unsigned int mask = (1u << (bits % 32)) - 1u;
        out.v[words - 1] &= mask;
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

__device__ static void scalarMultiplyBase64(unsigned long long k, unsigned int rx[8], unsigned int ry[8])
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

// Multiply the secp256k1 base point by a 256-bit scalar ``k`` (little endian)
__device__ static void scalarMultiplyBase(const unsigned int k[8], unsigned int rx[8], unsigned int ry[8])
{
    setPointInfinity(rx, ry);
    unsigned int qx[8];
    unsigned int qy[8];
    copyBigInt(_GX, qx);
    copyBigInt(_GY, qy);

    for(int i = 0; i < 8; ++i) {
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
        scalarMultiplyBase64(scalar, px, py);
    }

    RNGState rng{ seeds[tid] ^ 1ULL, seeds[tid] + 1ULL };
    unsigned long long mask = (windowBits >= 64) ? 0xffffffffffffffffULL : ((1ULL << windowBits) - 1ULL);

    for(unsigned int i = 0; i < steps; ++i) {
        unsigned long long step = next_random_step(rng);
        scalar += step;
        scalar %= ORDER;

        unsigned int sx[8];
        unsigned int sy[8];
        scalarMultiplyBase64(step, sx, sy);
        unsigned int tx[8];
        unsigned int ty[8];
        pointAdd(px, py, sx, sy, tx, ty);
        copyBigInt(tx, px);
        copyBigInt(ty, py);

        if((scalar & mask) == 0ULL) {
            unsigned int digest[5];
            unsigned int finalHash[5];
            // py[7] holds the least significant word; py[7] & 1 yields the parity bit
            hashPublicKeyCompressed(px, py[7] & 1, digest);
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
                                       const unsigned int *seeds,
                                       const unsigned int *starts,
                                       const unsigned int *startX,
                                       const unsigned int *startY,
                                       unsigned int steps,
                                       const TargetWindow *windows,
                                       unsigned int windowCount,
                                       unsigned long long stride)
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

    if(stride == 0ULL) {
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
                        unsigned int modBits = tw.offset + tw.bits;
                        unsigned int wordsK = (modBits + 31) / 32;
                        for(unsigned int j = 0; j < wordsK; ++j) {
                            out[idx].k[j] = scalar[j];
                        }
                        if(modBits % 32) {
                            out[idx].k[wordsK - 1] &= ((1U << (modBits % 32)) - 1U);
                        }
                        for(unsigned int j = wordsK; j < 8; ++j) {
                            out[idx].k[j] = 0U;
                        }
                    }
                }
            }
        }
    } else {
        unsigned int strideArr[8] = { (unsigned int)stride, (unsigned int)(stride >> 32), 0,0,0,0,0,0 };
        unsigned int sx[8];
        unsigned int sy[8];
        scalarMultiplyBase(strideArr, sx, sy);
        for(unsigned int i = 0; i < steps; ++i) {
            unsigned int digest[5];
            unsigned int finalHash[5];
            hashPublicKeyCompressed(px, py[7] & 1, digest);
            doRMD160FinalRound(digest, finalHash);

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
                        unsigned int modBits = tw.offset + tw.bits;
                        unsigned int wordsK = (modBits + 31) / 32;
                        for(unsigned int j = 0; j < wordsK; ++j) {
                            out[idx].k[j] = scalar[j];
                        }
                        if(modBits % 32) {
                            out[idx].k[wordsK - 1] &= ((1U << (modBits % 32)) - 1U);
                        }
                        for(unsigned int j = wordsK; j < 8; ++j) {
                            out[idx].k[j] = 0U;
                        }
                    }
                }
            }

            addModNU64(scalar, stride);
            unsigned int tx[8];
            unsigned int ty[8];
            pointAdd(px, py, sx, sy, tx, ty);
            copyBigInt(tx, px);
            copyBigInt(ty, py);
        }
    }
}
