#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "windowKernel.h"

#include "secp256k1.cuh"

__device__ static inline bool isZero256(const uint32_t a[8]) {
    for(int i = 0; i < 8; ++i) {
        if(a[i] != 0U) return false;
    }
    return true;
}

__device__ static void setPointInfinity(uint32_t x[8], uint32_t y[8]) {
    for(int i = 0; i < 8; ++i) {
        x[i] = 0xffffffffU;
        y[i] = 0xffffffffU;
    }
}

__device__ static void pointDouble(const uint32_t x[8], const uint32_t y[8],
                                   uint32_t rx[8], uint32_t ry[8]) {
    if(isInfinity(x)) {
        setPointInfinity(rx, ry);
        return;
    }

    uint32_t x2[8];
    uint32_t three_x2[8];
    uint32_t two_y[8];
    uint32_t inv[8];
    uint32_t lambda[8];
    uint32_t lambda2[8];
    uint32_t k[8];

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

__device__ static void pointAdd(const uint32_t ax[8], const uint32_t ay[8],
                                const uint32_t bx[8], const uint32_t by[8],
                                uint32_t rx[8], uint32_t ry[8]) {
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

    uint32_t rise[8];
    uint32_t run[8];
    uint32_t inv[8];
    uint32_t lambda[8];
    uint32_t lambda2[8];
    uint32_t k[8];

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

__device__ static void scalarMultiplySmall(const uint32_t bx[8], const uint32_t by[8],
                                           const uint32_t k[8], uint32_t rx[8], uint32_t ry[8]) {
    setPointInfinity(rx, ry);
    uint32_t qx[8];
    uint32_t qy[8];
    copyBigInt(bx, qx);
    copyBigInt(by, qy);
    for(int i = 0; i < 4; ++i) {
        uint32_t word = k[i];
        for(int bit = 0; bit < 32; ++bit) {
            if(word & 1U) {
                uint32_t tx[8];
                uint32_t ty[8];
                pointAdd(rx, ry, qx, qy, tx, ty);
                copyBigInt(tx, rx);
                copyBigInt(ty, ry);
            }
            word >>= 1U;
            uint32_t tx[8];
            uint32_t ty[8];
            pointDouble(qx, qy, tx, ty);
            copyBigInt(tx, qx);
            copyBigInt(ty, qy);
        }
    }
}

__device__ static void scalarMultiplyBase(const uint32_t k[8], uint32_t rx[8], uint32_t ry[8]) {
    GLVScalarSplit split;
    splitScalar(k, split);

    uint32_t r1x[8];
    uint32_t r1y[8];
    scalarMultiplySmall(_GX, _GY, split.k1, r1x, r1y);
    if(split.k1Neg) {
        uint32_t ny[8];
        negModP(r1y, ny);
        copyBigInt(ny, r1y);
    }

    if(isZero256(split.k2)) {
        copyBigInt(r1x, rx);
        copyBigInt(r1y, ry);
        return;
    }

    uint32_t base2x[8];
    uint32_t base2y[8];
    copyBigInt(_GX, base2x);
    copyBigInt(_GY, base2y);
    mulModP(base2x, _BETA, base2x);

    uint32_t r2x[8];
    uint32_t r2y[8];
    scalarMultiplySmall(base2x, base2y, split.k2, r2x, r2y);
    if(split.k2Neg) {
        uint32_t ny[8];
        negModP(r2y, ny);
        copyBigInt(ny, r2y);
    }

    pointAdd(r1x, r1y, r2x, r2y, rx, ry);
}

// GPU kernel performing a grid-stride loop over scalars ``k`` and extracting
// window fragments from the x-coordinate of ``k * G``. Matching fragments are
// appended to ``out_buf`` using an atomic counter.
extern "C" __global__
void windowKernel(uint64_t start_k, uint64_t range_len, uint32_t ws,
                  const uint32_t* offsets, uint32_t offsets_count,
                  uint32_t mask, const uint32_t* target_frags,
                  MatchRecord* out_buf, uint32_t* out_count) {
    (void)ws; // window size is encoded in mask on device
    uint64_t idx    = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = blockDim.x * gridDim.x;
    for(uint64_t i = idx; i < range_len; i += stride) {
        uint64_t k = start_k + i;
        uint32_t X[8], Y[8];
        scalarMultiplyBase(reinterpret_cast<const uint32_t*>(&k), X, Y);
        for(uint32_t j = 0; j < offsets_count; ++j) {
            uint32_t off  = offsets[j];
            uint32_t word = off >> 5;
            uint32_t bit  = off & 31u;
            uint32_t frag = 0;
            if(word < 8) {
                frag = X[word] >> bit;
                if(bit && word + 1 < 8) {
                    frag |= X[word + 1] << (32 - bit);
                }
                frag &= mask;
                if(frag == target_frags[j]) {
                    uint32_t pos = atomicAdd(out_count, 1u);
                    out_buf[pos] = { off, frag, k };
                }
            }
        }
    }
}

// Host wrapper used to launch ``windowKernel`` with basic error checking.
extern "C" void launchWindowKernel(dim3 grid, dim3 block,
                                   uint64_t start_k, uint64_t range_len,
                                   uint32_t ws, const uint32_t* offsets,
                                   uint32_t offsets_count, uint32_t mask,
                                   const uint32_t* target_frags,
                                   MatchRecord* out_buf,
                                   uint32_t* out_count) {
    windowKernel<<<grid, block>>>(start_k, range_len, ws, offsets,
                                  offsets_count, mask, target_frags,
                                  out_buf, out_count);
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        exit(1);
    }
}

