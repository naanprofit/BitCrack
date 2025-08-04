#pragma OPENCL EXTENSION cl_khr_int64 : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

// Window reported back to the host when a hash window matches one of the
// provided targets.  The scalar fragment contains the low ``offset + bits``
// bits of the walk's scalar value.
typedef struct {
    uint targetIdx;
    uint offset;
    uint bits;
    // 256-bit scalar for this step (little-endian words)
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

static const uint A1[8] = {2458184469U,3899429092U,2815716301U,814141985U,0U,0U,0U,0U};
static const uint B1[8] = {180348099U,1867808681U,17729576U,3829628630U,0U,0U,0U,0U};
static const uint A2[8] = {2638532568U,1472270477U,2833445878U,348803319U,1U,0U,0U,0U};
static const uint B2[8] = {2458184469U,3899429092U,2815716301U,814141985U,0U,0U,0U,0U};
static const uint G1[5] = {3944037802U,2430898820U,1808656492U,3525421012U,12422U};
static const uint G2[5] = {3838059026U,2141784767U,2284351316U,2127954190U,58435U};
static const uint _BETA[8] = {0x7AE96A2BU,0x657C0710U,0x6E64479EU,0xAC3434E9U,0x9CF04975U,0x12F58995U,0xC1396C28U,0x719501EEU};

bool equal(const __private uint *a, const __private uint *b);
bool isInfinity(const __private uint *x);
void mulModP(const __private uint *a, const __private uint *b, __private uint *r);

int isZero256(const uint a[8]) {
    for(int i=0;i<8;i++) {
        if(a[i] != 0U) return 0;
    }
    return 1;
}

int ge256(const __private uint *a, const __constant uint *b) {
    for(int i=7;i>=0;--i) {
        if(a[i] > b[i]) return 1;
        if(a[i] < b[i]) return 0;
    }
    return 1;
}

void sub256(const __private uint *a, const __constant uint *b, __private uint *r) {
    uint borrow = 0U;
    for(int i=0;i<8;i++) {
        uint ai = a[i];
        uint bi = b[i];
        uint t = ai - bi;
        uint ri = t - borrow;
        borrow = (ai < bi) | (t < borrow);
        r[i] = ri;
    }
}

void addModN(const uint a[8], const uint b[8], uint r[8]) {
    uint carry = 0U;
    for(int i=0;i<8;i++) {
        uint ai = a[i];
        uint bi = b[i];
        uint s = ai + bi;
        uint ri = s + carry;
        carry = (s < ai) | (ri < s);
        r[i] = ri;
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

static inline uint256_t toUint256(const uint a[8]) {
    uint256_t r;
    for(int i=0;i<8;i++) r.v[i] = a[i];
    return r;
}

static inline void fromUint256(uint256_t a, uint r[8]) {
    for(int i=0;i<8;i++) r[i] = a.v[i];
}

void copyBigInt(const uint src[8], uint dest[8]) {
    for(int i=0;i<8;i++) dest[i] = src[i];
}

void glvMulWords(const uint *x, int xLen, const uint *y, int yLen, uint *z) {
    for(int i=0;i<xLen+yLen;i++) z[i]=0U;
    for(int i=0;i<xLen;i++) {
        uint carry=0U;
        for(int j=0;j<yLen;j++) {
            ulong p = (ulong)x[i]*(ulong)y[j] + z[i+j] + carry;
            z[i+j] = (uint)p;
            carry = (uint)(p >> 32);
        }
        z[i+yLen] = carry;
    }
}

int glvCmp(const uint *a, const uint *b, int len) {
    for(int i=len-1;i>=0;--i) {
        if(a[i]>b[i]) return 1;
        if(a[i]<b[i]) return -1;
    }
    return 0;
}

void glvAdd(const uint *a, const uint *b, int len, uint *r) {
    uint carry=0U;
    for(int i=0;i<len;i++) {
        uint ai=a[i];
        uint bi=b[i];
        uint s=ai+bi;
        uint ri=s+carry;
        carry=(s<ai)||(ri<s);
        r[i]=ri;
    }
}

void glvSub(const uint *a, const uint *b, int len, uint *r) {
    uint borrow=0U;
    for(int i=0;i<len;i++) {
        uint ai=a[i];
        uint bi=b[i];
        uint t=ai-bi;
        uint ri=t-borrow;
        borrow=(ai<bi)||(t<borrow);
        r[i]=ri;
    }
}

typedef struct {
    uint k1[8];
    uint k2[8];
    int k1Neg;
    int k2Neg;
} GLVSplit;

void splitScalar(const uint k[8], __private GLVSplit *out) {
    uint prod1[13];
    glvMulWords(k,8,G1,5,prod1);
    uint c1[8]={0};
    for(int i=0;i<5;i++) {
        uint lo=(i+8<13)?prod1[i+8]:0U;
        uint hi=(i+9<13)?prod1[i+9]:0U;
        c1[i]=(lo>>16)|(hi<<16);
    }
    uint prod2[13];
    glvMulWords(k,8,G2,5,prod2);
    uint c2[8]={0};
    for(int i=0;i<5;i++) {
        uint lo=(i+8<13)?prod2[i+8]:0U;
        uint hi=(i+9<13)?prod2[i+9]:0U;
        c2[i]=(lo>>16)|(hi<<16);
    }
    uint t1[16];
    uint t2[16];
    glvMulWords(c1,8,A1,8,t1);
    glvMulWords(c2,8,A2,8,t2);
    uint sum[16];
    glvAdd(t1,t2,16,sum);
    uint kExt[16];
    for(int i=0;i<8;i++) kExt[i]=k[i];
    for(int i=8;i<16;i++) kExt[i]=0U;
    if(glvCmp(kExt,sum,16)>=0){
        glvSub(kExt,sum,16,t1);
        for(int i=0;i<8;i++) out->k1[i]=t1[i];
        out->k1Neg=0;
    }else{
        glvSub(sum,kExt,16,t1);
        for(int i=0;i<8;i++) out->k1[i]=t1[i];
        out->k1Neg=1;
    }
    glvMulWords(c1,8,B1,8,t1);
    glvMulWords(c2,8,B2,8,t2);
    if(glvCmp(t1,t2,16)>=0){
        glvSub(t1,t2,16,sum);
        for(int i=0;i<8;i++) out->k2[i]=sum[i];
        out->k2Neg=0;
    }else{
        glvSub(t2,t1,16,sum);
        for(int i=0;i<8;i++) out->k2[i]=sum[i];
        out->k2Neg=1;
    }
}

void setPointInfinity(uint x[8], uint y[8]) {
    for(int i=0;i<8;i++) {
        x[i] = 0xffffffffU;
        y[i] = 0xffffffffU;
    }
}

void addModP(const uint a[8], const uint b[8], uint r[8]) {
    uint256_t aa = toUint256(a);
    uint256_t bb = toUint256(b);
    uint256_t cc = addModP256k(aa, bb);
    fromUint256(cc, r);
}

void subModP(const uint a[8], const uint b[8], uint r[8]) {
    uint256_t aa = toUint256(a);
    uint256_t bb = toUint256(b);
    uint256_t cc = subModP256k(aa, bb);
    fromUint256(cc, r);
}

void invModP(const uint a[8], uint r[8]) {
    uint256_t aa = toUint256(a);
    uint256_t cc = invModP256k(aa);
    fromUint256(cc, r);
}

void negModP(const uint a[8], uint r[8]) {
    subModP(_P, a, r);
}

void pointDouble(const uint x[8], const uint y[8], uint rx[8], uint ry[8]) {
    if(isInfinity(x)) {
        setPointInfinity(rx, ry);
        return;
    }

    uint x2[8];
    uint three_x2[8];
    uint two_y[8];
    uint inv[8];
    uint lambda[8];
    uint lambda2[8];
    uint k[8];

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

void pointAdd(const uint ax[8], const uint ay[8], const uint bx[8], const uint by[8], uint rx[8], uint ry[8]) {
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

    uint rise[8];
    uint run[8];
    uint inv[8];
    uint lambda[8];
    uint lambda2[8];
    uint k[8];

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

void scalarMultiplySmall(const __private uint *bx, const __private uint *by, const __private uint *k, __private uint *rx, __private uint *ry) {
    setPointInfinity(rx, ry);
    uint qx[8];
    uint qy[8];
    copyBigInt(bx, qx);
    copyBigInt(by, qy);
    for(int i=0;i<4;i++) {
        uint word = k[i];
        for(int bit=0; bit<32; ++bit) {
            if(word & 1U) {
                uint tx[8];
                uint ty[8];
                pointAdd(rx, ry, qx, qy, tx, ty);
                copyBigInt(tx, rx);
                copyBigInt(ty, ry);
            }
            word >>= 1U;
            uint tx[8];
            uint ty[8];
            pointDouble(qx, qy, tx, ty);
            copyBigInt(tx, qx);
            copyBigInt(ty, qy);
        }
    }
}

void scalarMultiplyBase(const __private uint *k, __private uint *rx, __private uint *ry) {
    GLVSplit s;
    splitScalar(k, &s);
    uint base1x[8];
    uint base1y[8];
    copyBigInt(_GX, base1x);
    copyBigInt(_GY, base1y);
    uint r1x[8];
    uint r1y[8];
    scalarMultiplySmall(base1x, base1y, s.k1, r1x, r1y);
    if(s.k1Neg) {
        uint ny[8];
        negModP(r1y, ny);
        copyBigInt(ny, r1y);
    }
    if(isZero256(s.k2)) {
        copyBigInt(r1x, rx);
        copyBigInt(r1y, ry);
        return;
    }
    uint base2x[8];
    uint base2y[8];
    uint gx[8];
    uint beta[8];
    copyBigInt(_GX, gx);
    copyBigInt(_BETA, beta);
    mulModP(gx, beta, base2x);
    copyBigInt(_GY, base2y);
    uint r2x[8];
    uint r2y[8];
    scalarMultiplySmall(base2x, base2y, s.k2, r2x, r2y);
    if(s.k2Neg) {
        uint ny[8];
        negModP(r2y, ny);
        copyBigInt(ny, r2y);
    }
    pointAdd(r1x, r1y, r2x, r2y, rx, ry);
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
// 160-bit value with higher bits cleared.  The result is stored as five
// 32-bit little-endian words.
typedef struct { uint v[5]; } Hash160;

Hash160 hashWindow(const uint h[5], uint offset, uint bits)
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
