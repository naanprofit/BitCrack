typedef struct {
    ulong k[4];
    uint hash[5];
} PollardCLMatch;

typedef struct {
    ulong s0;
    ulong s1;
} RNGState;

ulong xorshift128plus(__private RNGState *state)
{
    ulong x = state->s0;
    ulong y = state->s1;
    state->s0 = y;
    x ^= x << 23;
    x ^= x >> 17;
    x ^= y;
    x ^= y >> 26;
    state->s1 = x;
    return x + y;
}

ulong next_random_step(__private RNGState *state)
{
    const ulong ORDER_MINUS_ONE = (ulong)0xBFD25E8CD0364140UL;
    return (xorshift128plus(state) % ORDER_MINUS_ONE) + 1UL;
}
unsigned int endian(unsigned int x)
{
    return (x << 24) | ((x << 8) & 0x00ff0000) | ((x >> 8) & 0x0000ff00) | (x >> 24);
}

void doRMD160FinalRound(const unsigned int hIn[5], unsigned int hOut[5])
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

void hashPublicKeyCompressed(const unsigned int x[8], unsigned int yParity, unsigned int* digestOut)
{
    unsigned int hash[8];

    sha256PublicKeyCompressed(x, yParity, hash);

    for(int i = 0; i < 8; i++) {
        hash[i] = endian(hash[i]);
    }

    ripemd160sha256NoFinal(hash, digestOut);
}

__kernel void pollard_random_walk(__global PollardCLMatch *out,
                                  __global uint *outCount,
                                  uint maxOut,
                                  __global ulong *seeds,
                                  uint steps,
                                  uint windowBits)
{
    size_t gid = get_global_id(0);
    RNGState rng = { seeds[gid] ^ (ulong)1, seeds[gid] + (ulong)1 };
    ulong scalar = 0UL;
    const ulong ORDER = (ulong)0xBFD25E8CD0364141UL;
    ulong mask = (windowBits >= 64) ? (ulong)0xFFFFFFFFFFFFFFFFUL : (((ulong)1 << windowBits) - 1UL);

    uint px[8];
    uint py[8];
    setPointInfinity(px, py);

    for(uint i = 0; i < steps; i++) {
        ulong step = next_random_step(&rng);
        scalar += step;
        scalar %= ORDER;

        uint sx[8];
        uint sy[8];
        scalarMultiplyBase(step, sx, sy);

        if(isInfinity(px)) {
            copyBigInt(sx, px);
            copyBigInt(sy, py);
        } else {
            uint tx[8];
            uint ty[8];
            pointAdd(px, py, sx, sy, tx, ty);
            copyBigInt(tx, px);
            copyBigInt(ty, py);
        }

        if((scalar & mask) == 0UL) {
            uint slot = atomic_inc(outCount);
            if(slot < maxOut) {
                uint digest[5];
                uint finalHash[5];
                hashPublicKeyCompressed(px, py[7], digest);
                doRMD160FinalRound(digest, finalHash);
                for(int w = 0; w < 5; w++) {
                    out[slot].hash[w] = finalHash[w];
                }
                out[slot].k[0] = scalar & 0xffffffffUL;
                out[slot].k[1] = scalar >> 32;
                out[slot].k[2] = 0UL;
                out[slot].k[3] = 0UL;
            }
        }
    }
}