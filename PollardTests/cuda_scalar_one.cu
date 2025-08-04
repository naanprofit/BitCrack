#include <cuda_runtime.h>
#include "../cudaMath/secp256k1.cuh"
#include "../cudaMath/sha256.cuh"
#include "../cudaMath/ripemd160.cuh"

__device__ static void doRMD160FinalRound(const unsigned int hIn[5], unsigned int hOut[5])
{
    const unsigned int iv[5] = {
        0x67452301,
        0xefcdab89,
        0x98badcfe,
        0x10325476,
        0xc3d2e1f0
    };
    for(int i = 0; i < 5; i++) {
        hOut[i] = hIn[i] + iv[(i + 1) % 5];
    }
}

__device__ void hashPublicKeyCompressed(const unsigned int *x, unsigned int yParity, unsigned int *digestOut)
{
    unsigned int hash[8];
    sha256PublicKeyCompressed(x, yParity, hash);
    for(int i = 0; i < 8; i++) {
        hash[i] = endian(hash[i]);
    }
    ripemd160sha256NoFinal(hash, digestOut);
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

__global__ void scalarOneKernel(unsigned int *outX, unsigned int *outY, unsigned int *outHash)
{
    unsigned int x[8];
    unsigned int y[8];
    scalarMultiplyBase(1ULL, x, y);

    unsigned int digest[5];
    unsigned int finalHash[5];
    hashPublicKeyCompressed(x, y[7] & 1U, digest);
    doRMD160FinalRound(digest, finalHash);

    for(int i = 0; i < 8; i++) {
        outX[i] = x[i];
        outY[i] = y[i];
    }
    for(int i = 0; i < 5; i++) {
        outHash[i] = finalHash[i];
    }
}

extern "C" bool runCudaScalarOne(unsigned int x[8], unsigned int y[8], unsigned int hash[5])
{
    unsigned int *d_x = nullptr;
    unsigned int *d_y = nullptr;
    unsigned int *d_hash = nullptr;
    if(cudaMalloc(&d_x, sizeof(unsigned int) * 8)) return false;
    if(cudaMalloc(&d_y, sizeof(unsigned int) * 8)) { cudaFree(d_x); return false; }
    if(cudaMalloc(&d_hash, sizeof(unsigned int) * 5)) { cudaFree(d_x); cudaFree(d_y); return false; }

    scalarOneKernel<<<1,1>>>(d_x, d_y, d_hash);
    if(cudaDeviceSynchronize() != cudaSuccess) {
        cudaFree(d_x); cudaFree(d_y); cudaFree(d_hash);
        return false;
    }
    if(cudaMemcpy(x, d_x, sizeof(unsigned int) * 8, cudaMemcpyDeviceToHost)) { cudaFree(d_x); cudaFree(d_y); cudaFree(d_hash); return false; }
    if(cudaMemcpy(y, d_y, sizeof(unsigned int) * 8, cudaMemcpyDeviceToHost)) { cudaFree(d_x); cudaFree(d_y); cudaFree(d_hash); return false; }
    if(cudaMemcpy(hash, d_hash, sizeof(unsigned int) * 5, cudaMemcpyDeviceToHost)) { cudaFree(d_x); cudaFree(d_y); cudaFree(d_hash); return false; }

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_hash);
    return true;
}

