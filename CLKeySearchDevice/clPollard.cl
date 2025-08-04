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
    uint target[5];    // expected window value (LSB order)
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
// digest ``h``. Bits are interpreted in little-endian order and returned as a
// 160-bit value with higher bits cleared.
typedef struct { uint v[5]; } Hash160;

Hash160 hashWindow(const unsigned int h[5], unsigned int offset, unsigned int bits)
{
    Hash160 out; 
    for(int i=0;i<5;i++) out.v[i]=0u;
    unsigned int word = offset / 32;
    unsigned int bit  = offset % 32;
    unsigned int words = (bits + 31) / 32;
    for(unsigned int i=0;i<words && word + i < 5; i++) {
        ulong val = ((ulong)h[word + i]) >> bit;
        if(bit && word + i + 1 < 5) {
            val |= ((ulong)h[word + i + 1]) << (32 - bit);
        }
        out.v[i] = (uint)(val & 0xffffffffUL);
    }
    if(bits % 32) {
        uint mask = (1u << (bits % 32)) - 1u;
        out.v[words-1] &= mask;
    }
    return out;
}

__kernel void pollard_walk(__global PollardWindow *out,
                           __global uint *outCount,
                           uint maxOut,
                           __global ulong *seeds,
                           __global ulong *starts,
                           __global uint *startX,
                           __global uint *startY,
                           uint steps,
                           __global const TargetWindow *windows,
                           uint windowCount,
                           ulong stride)
{
    size_t gid = get_global_id(0);
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

    if(stride == 0UL) {
        RNGState rng = { seeds[gid] ^ (ulong)1, seeds[gid] + (ulong)1 };
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

            uint digest[5];
            uint finalHash[5];
            hashPublicKeyCompressed(px, py[7], digest);
            doRMD160FinalRound(digest, finalHash);

            for(uint w = 0; w < windowCount; w++) {
                TargetWindow tw = windows[w];
                Hash160 hv = hashWindow(finalHash, tw.offset, tw.bits);
                uint words = (tw.bits + 31) / 32;
                int match = 1;
                for(uint j = 0; j < words; j++) {
                    if(hv.v[j] != tw.target[j]) { match = 0; break; }
                }
                if(match) {
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
    } else {
        uint sx[8];
        uint sy[8];
        scalarMultiplyBase(stride, sx, sy);
        for(uint i = 0; i < steps; i++) {
            uint digest[5];
            uint finalHash[5];
            hashPublicKeyCompressed(px, py[7], digest);
            doRMD160FinalRound(digest, finalHash);

            for(uint w = 0; w < windowCount; w++) {
                TargetWindow tw = windows[w];
                Hash160 hv = hashWindow(finalHash, tw.offset, tw.bits);
                uint words = (tw.bits + 31) / 32;
                int match = 1;
                for(uint j = 0; j < words; j++) {
                    if(hv.v[j] != tw.target[j]) { match = 0; break; }
                }
                if(match) {
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

            scalar += stride;
            scalar %= ORDER;
            uint tx[8];
            uint ty[8];
            pointAdd(px, py, sx, sy, tx, ty);
            copyBigInt(tx, px);
            copyBigInt(ty, py);
        }
    }
}
