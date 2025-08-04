#include <cuda.h>
#include <cuda_runtime.h>

__constant__ unsigned int _RIPEMD160_IV[5] = {
        0x67452301,
        0xefcdab89,
        0x98badcfe,
        0x10325476,
        0xc3d2e1f0
};

__constant__ unsigned int _K0 = 0x5a827999;
__constant__ unsigned int _K1 = 0x6ed9eba1;
__constant__ unsigned int _K2 = 0x8f1bbcdc;
__constant__ unsigned int _K3 = 0xa953fd4e;

__constant__ unsigned int _K4 = 0x7a6d76e9;
__constant__ unsigned int _K5 = 0x6d703ef3;
__constant__ unsigned int _K6 = 0x5c4dd124;
__constant__ unsigned int _K7 = 0x50a28be6;

