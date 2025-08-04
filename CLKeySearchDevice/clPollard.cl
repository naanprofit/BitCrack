// Window reported back to the host when a hash window matches one of the
// provided targets.  The scalar fragment contains the low ``offset + bits``
// bits of the walk's scalar value.
typedef struct {
    uint targetIdx;
    uint offset;
    uint bits;
    uint k[8];
} PollardWindow;

// Description of a bit window to test against the RIPEMD160 of each step.
typedef struct {
    uint targetIdx;
    uint offset;
    uint bits;
    ulong target;    // expected window value (LSB order)
} TargetWindow;

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

// Extract ``bits`` bits starting at ``offset`` from the 160-bit RIPEMD160
// digest ``h``.  Bits are interpreted in little-endian order.
ulong hashWindow(const unsigned int h[5], unsigned int offset, unsigned int bits)
{
    unsigned int word = offset / 32;
    unsigned int bit  = offset % 32;
    ulong val = 0UL;
    if(word < 5) {
        val = ((ulong)h[word]) >> bit;
        if(bit + bits > 32 && word + 1 < 5) {
            val |= ((ulong)h[word + 1]) << (32 - bit);
        }
    }
    if(bit + bits > 64 && word + 2 < 5) {
        val |= ((ulong)h[word + 2]) << (64 - bit);
    }
    if(bits < 64) {
        ulong mask = (bits == 64) ? (ulong)0xffffffffffffffffUL : (((ulong)1 << bits) - 1UL);
        val &= mask;
    }
    return val;
}

__kernel void pollard_random_walk(__global PollardWindow *out,
                                  __global uint *outCount,
                                  uint maxOut,
                                  __global ulong *seeds,
                                  __global ulong *starts,
                                  __global uint *startX,
                                  __global uint *startY,
                                  uint steps,
                                  __global const TargetWindow *windows,
                                  uint windowCount)
{
    size_t gid = get_global_id(0);
    RNGState rng = { seeds[gid] ^ (ulong)1, seeds[gid] + (ulong)1 };
    ulong scalar = starts[gid];
    const ulong ORDER = (ulong)0xBFD25E8CD0364141UL;

    uint px[8];
    uint py[8];
    if(startX && startY) {
        for(int i = 0; i < 8; i++) {
            px[i] = startX[gid * 8 + i];
            py[i] = startY[gid * 8 + i];
        }
    } else {
        scalarMultiplyBase(scalar, px, py);
    }

    for(uint i = 0; i < steps; i++) {
        ulong step = next_random_step(&rng);
        scalar += step;
        scalar %= ORDER;

        uint sx[8];
        uint sy[8];
        scalarMultiplyBase(step, sx, sy);
        uint tx[8];
        uint ty[8];
        pointAdd(px, py, sx, sy, tx, ty);
        copyBigInt(tx, px);
        copyBigInt(ty, py);

        // Compute the RIPEMD160 of the current point once per step
        uint digest[5];
        uint finalHash[5];
        hashPublicKeyCompressed(px, py[7], digest);
        doRMD160FinalRound(digest, finalHash);

        // Compare all requested windows against their targets
        for(uint w = 0; w < windowCount; w++) {
            TargetWindow tw = windows[w];
            ulong hv = hashWindow(finalHash, tw.offset, tw.bits);
            if(hv == tw.target) {
                uint slot = atomic_inc(outCount);
                if(slot < maxOut) {
                    out[slot].targetIdx = tw.targetIdx;
                    out[slot].offset    = tw.offset;
                    out[slot].bits      = tw.bits;
                    uint modBits = tw.offset + tw.bits;
                    ulong mask = (modBits >= 64) ? (ulong)0xffffffffffffffffUL : (((ulong)1 << modBits) - 1UL);
                    ulong frag = scalar & mask;
                    out[slot].k[0] = (uint)(frag & 0xffffffffUL);
                    out[slot].k[1] = (uint)(frag >> 32);
                    for(int j = 2; j < 8; j++) {
                        out[slot].k[j] = 0U;
                    }
                }
            }
        }
    }
}
