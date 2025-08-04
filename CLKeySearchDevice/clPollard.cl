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

// secp256k1 group order in little-endian words
__constant uint ORDER[8] = {
    0xD0364141U, 0xBFD25E8CU, 0xAF48A03BU, 0xBAAEDCE6U,
    0xFFFFFFFEU, 0xFFFFFFFFU, 0xFFFFFFFFU, 0xFFFFFFFFU
};

int isZero256(const uint a[8]) {
    for(int i=0;i<8;i++) {
        if(a[i] != 0U) return 0;
    }
    return 1;
}

int ge256(const uint a[8], const uint b[8]) {
    for(int i=7;i>=0;--i) {
        if(a[i] > b[i]) return 1;
        if(a[i] < b[i]) return 0;
    }
    return 1;
}

void sub256(const uint a[8], const uint b[8], uint r[8]) {
    ulong borrow = 0UL;
    for(int i=0;i<8;i++) {
        ulong diff = (ulong)a[i] - b[i] - borrow;
        r[i] = (uint)diff;
        borrow = (diff >> 63) & 1UL;
    }
}

void addModN(const uint a[8], const uint b[8], uint r[8]) {
    ulong carry = 0UL;
    for(int i=0;i<8;i++) {
        ulong sum = (ulong)a[i] + b[i] + carry;
        r[i] = (uint)sum;
        carry = sum >> 32;
    }
    if(carry || ge256(r, ORDER)) {
        sub256(r, ORDER, r);
    }
}

void next_random_step(__private RNGState *state, uint step[8]) {
    do {
        for(int i=0;i<4;i++) {
            ulong v = xorshift128plus(state);
            step[i*2] = (uint)(v & 0xffffffffUL);
            step[i*2+1] = (uint)(v >> 32);
        }
    } while(isZero256(step) || ge256(step, ORDER));
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
        hOut[i] = hIn[i] + iv[i];
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
    unsigned int span = bit + bits;
    unsigned int words = (span + 31) / 32;
    for(unsigned int i=0;i<words && word + i < 5; i++) {
        ulong val = ((ulong)h[word + i]) >> bit;
        if(bit && word + i + 1 < 5) {
            val |= ((ulong)h[word + i + 1]) << (32 - bit);
        }
        out.v[i] = (uint)(val & 0xffffffffUL);
    }
    if(span % 32) {
        uint mask = (1u << (span % 32)) - 1u;
        out.v[words-1] &= mask;
    }
    for(unsigned int i=words;i<5;i++) {
        out.v[i]=0u;
    }
    return out;
}

__kernel void pollard_walk(__global PollardWindow *out,
                           __global volatile uint *outCount,
                           uint maxOut,
                           __global uint *seeds,
                           __global uint *starts,
                           __global uint *startX,
                           __global uint *startY,
                           uint steps,
                           __global const TargetWindow *windows,
                           uint windowCount,
                           __global uint *strides)
{
    size_t gid = get_global_id(0);
    uint scalar[8];
    for(int i=0;i<8;i++) {
        scalar[i] = starts[gid*8 + i];
    }

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

    uint stride[8];
    for(int i=0;i<8;i++) {
        stride[i] = strides ? strides[gid*8 + i] : 0U;
    }

    if(isZero256(stride)) {
        ulong s0 = ((ulong)seeds[gid*8 + 1] << 32) | seeds[gid*8 + 0];
        ulong s1 = ((ulong)seeds[gid*8 + 3] << 32) | seeds[gid*8 + 2];
        RNGState rng = { s0 ^ (ulong)1, s1 + (ulong)1 };
        for(uint i = 0; i < steps; i++) {
            uint step[8];
            next_random_step(&rng, step);
            addModN(scalar, step, scalar);

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
            for(int j = 0; j < 5; j++) {
                finalHash[j] = endian(finalHash[j]);
            }

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
                        uint wordsK = (modBits + 31) / 32;
                        for(uint j = 0; j < wordsK; j++) {
                            out[slot].k[j] = scalar[j];
                        }
                        if(modBits % 32) {
                            out[slot].k[wordsK-1] &= ((1U << (modBits % 32)) - 1U);
                        }
                        for(uint j = wordsK; j < 8; j++) {
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
            for(int j = 0; j < 5; j++) {
                finalHash[j] = endian(finalHash[j]);
            }

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
                        uint wordsK = (modBits + 31) / 32;
                        for(uint j = 0; j < wordsK; j++) {
                            out[slot].k[j] = scalar[j];
                        }
                        if(modBits % 32) {
                            out[slot].k[wordsK-1] &= ((1U << (modBits % 32)) - 1U);
                        }
                        for(uint j = wordsK; j < 8; j++) {
                            out[slot].k[j] = 0U;
                        }
                    }
                }
            }

            addModN(scalar, stride, scalar);
            uint tx[8];
            uint ty[8];
            pointAdd(px, py, sx, sy, tx, ty);
            copyBigInt(tx, px);
            copyBigInt(ty, py);
        }
    }
}
