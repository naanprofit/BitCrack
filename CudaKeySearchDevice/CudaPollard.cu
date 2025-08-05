#include <stdint.h>
#include <cuda_runtime.h>
#include <cstdio>
#include "../KeyFinder/PollardTypes.h"
#include "sha256.cuh"   // SHA256 hashing for public keys
#include "ripemd160.cuh" // RIPEMD160 finalisation
#include "secp256k1.cuh" // EC point operations
#include "ptx.cuh"       // byte order helpers

__device__ void hashPublicKeyCompressed(const uint32_t*, uint32_t, uint32_t*);

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if(err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    } \
} while(0)

// Result written by the kernel when a hash window matches a target.
struct GpuPollardWindow {
    uint32_t targetIdx;
    uint32_t offset;
    uint32_t bits;
    // Full 256-bit scalar of the distinguished point in little-endian word order
    uint32_t k[8];
};

// Description of a window to test for each step.
struct TargetWindow {
    uint32_t targetIdx;
    uint32_t offset;
    uint32_t bits;
    uint32_t target[5];
};

struct RNGState {
    uint64_t s0;
    uint64_t s1;
};

// Output structure used by the legacy random walk kernel. Each entry
// stores the full 256-bit scalar of a distinguished point along with the
// corresponding hash160 digest so the host can perform window matching.
struct CudaPollardMatch {
    uint32_t k[8];
    uint32_t hash[5];
};

#define MAX_OFFSETS 32
__device__ static inline uint64_t xorshift128plus(RNGState &state)
{
    uint64_t x = state.s0;
    uint64_t y = state.s1;
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
static __device__ __constant__ uint32_t ORDER[8] = {
    0xD0364141U, 0xBFD25E8CU, 0xAF48A03BU, 0xBAAEDCE6U,
    0xFFFFFFFEU, 0xFFFFFFFFU, 0xFFFFFFFFU, 0xFFFFFFFFU
};

__device__ static inline bool isZero256(const uint32_t a[8]) {
    for(int i = 0; i < 8; ++i) {
        if(a[i] != 0U) return false;
    }
    return true;
}

__device__ static inline bool ge256(const uint32_t a[8], const uint32_t b[8]) {
    for(int i = 7; i >= 0; --i) {
        if(a[i] > b[i]) return true;
        if(a[i] < b[i]) return false;
    }
    return true; // equal
}

__device__ static inline void sub256(const uint32_t a[8], const uint32_t b[8], uint32_t r[8]) {
    uint32_t borrow = 0U;
    for(int i = 0; i < 8; ++i) {
        uint32_t ai = a[i];
        uint32_t bi = b[i];
        uint32_t t = ai - bi;
        uint32_t ri = t - borrow;
        borrow = (ai < bi) | (t < borrow);
        r[i] = ri;
    }
}

__device__ static inline void addModN(const uint32_t a[8], const uint32_t b[8], uint32_t r[8]) {
    uint32_t carry = 0U;
    for(int i = 0; i < 8; ++i) {
        uint32_t ai = a[i];
        uint32_t bi = b[i];
        uint32_t s = ai + bi;
        uint32_t ri = s + carry;
        carry = (s < ai) | (ri < s);
        r[i] = ri;
    }
    if(carry || ge256(r, ORDER)) {
        sub256(r, ORDER, r);
    }
}

// Generate a random step in the range [1, ORDER-1]
__device__ static inline void next_random_step(RNGState &state, uint32_t step[8]) {
    do {
        for(int i = 0; i < 4; ++i) {
            uint64_t v = xorshift128plus(state);
            step[i * 2]     = (uint32_t)(v & 0xffffffffULL);
            step[i * 2 + 1] = (uint32_t)(v >> 32);
        }
    } while(isZero256(step) || ge256(step, ORDER));
}

static __device__ __forceinline__ void doRMD160FinalRound(const uint32_t hIn[5], uint32_t hOut[5])
{
    const uint32_t iv[5] = {
        0x67452301,
        0xefcdab89,
        0x98badcfe,
        0x10325476,
        0xc3d2e1f0
    };

    for(int i = 0; i < 5; ++i) {
        // Add the corresponding RIPEMD-160 IV directly
        hOut[i] = hIn[i] + iv[i];
    }
}

// Extract ``bits`` bits starting at ``offset`` from the 160-bit RIPEMD160
// digest ``h``. Bits are interpreted in little-endian order and the result is
// returned as five 32-bit words with higher words cleared.
struct Hash160 {
    uint32_t v[5];
};

// Return ``bits`` bits of ``h`` starting at ``offset`` as a little-endian
// 160-bit value. Any combination of offset and size that stays within 160
// bits is supported.
__device__ Hash160 hashWindow(const uint32_t h[5], uint32_t offset, uint32_t bits)
{
    Hash160 out;
    for(int i = 0; i < 5; ++i) {
        out.v[i] = 0u;
    }
    uint32_t word  = offset / 32;
    uint32_t bit   = offset % 32;
    uint32_t words = (bits + 31) / 32;
    for(uint32_t i = 0; i < words && word + i < 5; ++i) {
        uint64_t val = ((uint64_t)h[word + i]) >> bit;
        if(bit && word + i + 1 < 5) {
            val |= ((uint64_t)h[word + i + 1]) << (32 - bit);
        }
        out.v[i] = (uint32_t)(val & 0xffffffffULL);
    }
    uint32_t maskBits = bits % 32;
    if(maskBits) {
        uint32_t mask = (1u << maskBits) - 1u;
        out.v[words - 1] &= mask;
    }
    for(uint32_t i = words; i < 5; ++i) {
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
    copyBigInt(_GX, base2x);
    copyBigInt(_GY, base2y);
    mulModP(base2x, _BETA, base2x);

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

__device__ static inline void point_mul_G(const uint32_t k[8], uint32_t X[8], uint32_t Y[8]) {
    scalarMultiplyBase(k, X, Y);
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

    uint64_t s0 = ((uint64_t)seeds[tid*8 + 1] << 32) | seeds[tid*8 + 0];
    uint64_t s1 = ((uint64_t)seeds[tid*8 + 3] << 32) | seeds[tid*8 + 2];
    s0 ^= ((uint64_t)seeds[tid*8 + 5] << 32) | seeds[tid*8 + 4];
    s1 ^= ((uint64_t)seeds[tid*8 + 7] << 32) | seeds[tid*8 + 6];
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
            unsigned int finalHashBE[5];
            unsigned int finalHash[5];
            hashPublicKeyCompressed(px, py[7] & 1, digest);
            doRMD160FinalRound(digest, finalHashBE);
            // Convert to little-endian word order
            for(int j = 0; j < 5; ++j) {
                finalHash[j] = endian(finalHashBE[4 - j]);
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
                                       uint32_t *outCount,
                                       uint32_t maxOut,
                                       const uint32_t *seeds,
                                       const uint32_t *starts,
                                       const uint32_t *startX,
                                       const uint32_t *startY,
                                       uint32_t steps,
                                       const TargetWindow *windows,
                                       uint32_t windowCount,
                                       const uint32_t *strides)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t scalar[8];
    for(int i = 0; i < 8; ++i) {
        scalar[i] = starts[tid * 8 + i];
    }
    uint32_t px[8];
    uint32_t py[8];

    if(startX && startY) {
        for(int i = 0; i < 8; ++i) {
            px[i] = startX[tid * 8 + i];
            py[i] = startY[tid * 8 + i];
        }
    } else {
        scalarMultiplyBase(scalar, px, py);
    }

    uint32_t stride[8];
    for(int i = 0; i < 8; ++i) {
        stride[i] = strides ? strides[tid * 8 + i] : 0U;
    }

    if(isZero256(stride)) {
        uint64_t s0 = ((uint64_t)seeds[tid*8 + 1] << 32) | seeds[tid*8 + 0];
        uint64_t s1 = ((uint64_t)seeds[tid*8 + 3] << 32) | seeds[tid*8 + 2];
        s0 ^= ((uint64_t)seeds[tid*8 + 5] << 32) | seeds[tid*8 + 4];
        s1 ^= ((uint64_t)seeds[tid*8 + 7] << 32) | seeds[tid*8 + 6];
        RNGState rng{ s0 ^ 1ULL, s1 + 1ULL };
        for(uint32_t i = 0; i < steps; ++i) {
            uint32_t step[8];
            next_random_step(rng, step);
            addModN(scalar, step, scalar);

            uint32_t sx[8];
            uint32_t sy[8];
            scalarMultiplyBase(step, sx, sy);
            uint32_t tx[8];
            uint32_t ty[8];
            pointAdd(px, py, sx, sy, tx, ty);
            copyBigInt(tx, px);
            copyBigInt(ty, py);

            uint32_t digest[5];
            uint32_t finalHashBE[5];
            uint32_t finalHash[5];
            hashPublicKeyCompressed(px, py[7] & 1, digest);
            doRMD160FinalRound(digest, finalHashBE);
            // Convert to little-endian word order
            for(int j = 0; j < 5; ++j) {
                finalHash[j] = endian(finalHashBE[4 - j]);
            }

            for(uint32_t w = 0; w < windowCount; ++w) {
                TargetWindow tw = windows[w];
                Hash160 hv = hashWindow(finalHash, tw.offset, tw.bits);
                uint32_t words = (tw.bits + 31) / 32;
                bool match = true;
                for(uint32_t j = 0; j < words; ++j) {
                    if(hv.v[j] != tw.target[j]) { match = false; break; }
                }
                if(match) {
                    uint32_t idx = atomicAdd(outCount, 1u);
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
        uint32_t sx[8];
        uint32_t sy[8];
        scalarMultiplyBase(stride, sx, sy);
        for(uint32_t i = 0; i < steps; ++i) {
            uint32_t digest[5];
            uint32_t finalHashBE[5];
            uint32_t finalHash[5];
            hashPublicKeyCompressed(px, py[7] & 1, digest);
            doRMD160FinalRound(digest, finalHashBE);
            // Convert to little-endian word order
            for(int j = 0; j < 5; ++j) {
                finalHash[j] = endian(finalHashBE[4 - j]);
            }

            for(uint32_t w = 0; w < windowCount; ++w) {
                TargetWindow tw = windows[w];
                Hash160 hv = hashWindow(finalHash, tw.offset, tw.bits);
                uint32_t words = (tw.bits + 31) / 32;
                bool match = true;
                for(uint32_t j = 0; j < words; ++j) {
                    if(hv.v[j] != tw.target[j]) { match = false; break; }
                }
                if(match) {
                    uint32_t idx = atomicAdd(outCount, 1u);
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
            uint32_t tx[8];
            uint32_t ty[8];
            pointAdd(px, py, sx, sy, tx, ty);
            copyBigInt(tx, px);
            copyBigInt(ty, py);
        }
    }
}
