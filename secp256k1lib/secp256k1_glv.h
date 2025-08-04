#ifndef _SECP256K1_GLV_H
#define _SECP256K1_GLV_H

#include "secp256k1.h"

namespace secp256k1 {

struct GLVSplit {
    uint256 k1;
    uint256 k2;
    bool k1Neg;
    bool k2Neg;
    GLVSplit() : k1(0), k2(0), k1Neg(false), k2Neg(false) {}
};

/*
 * Split a scalar ``k`` into ``k1`` and ``k2`` using the efficiently
 * computable endomorphism on secp256k1.  The result satisfies
 *   k * G = k1 * G + k2 * (\phi(G))
 * where ``phi`` is the endomorphism (x,y) -> (beta*x, y).
 *
 * The values ``k1`` and ``k2`` are at most 128 bits each and the sign of
 * each component is returned separately to avoid having to deal with
 * negative numbers in the multiplication routines.
 */
static inline GLVSplit splitScalar(const uint256 &k) {
    // Lattice constants from "Guide to Elliptic Curve Cryptography"
    // and libsecp256k1. All values are little endian.
    static const unsigned int A1_WORDS[8] = {
        2458184469u, 3899429092u, 2815716301u, 814141985u,
        0u, 0u, 0u, 0u
    }; // 0x3086d221a7d46bcde86c90e49284eb15
    static const unsigned int B1_WORDS[8] = {
        180348099u, 1867808681u, 17729576u, 3829628630u,
        0u, 0u, 0u, 0u
    }; // 0xe4437ed6010e88286f547fa90abfe4c3 (absolute value)
    static const unsigned int A2_WORDS[8] = {
        2638532568u, 1472270477u, 2833445878u, 348803319u,
        1u, 0u, 0u, 0u
    }; // 0x114ca50f7a8e2f3f657c1108d9d44cfd8
    static const unsigned int B2_WORDS[8] = {
        2458184469u, 3899429092u, 2815716301u, 814141985u,
        0u, 0u, 0u, 0u
    }; // equal to A1

    // Pre-computed constants for the rounding operations: floor((bi << 272)/n)
    static const unsigned int G1_WORDS[5] = {
        3944037802u, 2430898820u, 1808656492u, 3525421012u, 12422u
    }; // for b2
    static const unsigned int G2_WORDS[5] = {
        3838059026u, 2141784767u, 2284351316u, 2127954190u, 58435u
    }; // for -b1 (absolute value)

    // Helper to multiply two little endian arrays of 32-bit words.
    auto mulWords = [](const unsigned int *x, int xLen,
                       const unsigned int *y, int yLen,
                       unsigned int *z) {
        for(int i = 0; i < xLen + yLen; ++i) z[i] = 0u;
        for(int i = 0; i < xLen; ++i) {
            unsigned int carry = 0u;
            for(int j = 0; j < yLen; ++j) {
                uint64_t p = (uint64_t)x[i] * (uint64_t)y[j] + z[i+j] + carry;
                z[i+j] = (unsigned int)p;
                carry = (unsigned int)(p >> 32);
            }
            z[i + yLen] = carry;
        }
    };

    // Export the scalar ``k`` to an array for the arithmetic below.
    unsigned int kWords[8];
    k.exportWords(kWords, 8, uint256::LittleEndian);

    // Compute c1 = round(b2 * k / n) via ((k * G1) >> 272)
    unsigned int prod1[13];
    mulWords(kWords, 8, G1_WORDS, 5, prod1);
    unsigned int c1Words[8] = {0};
    for(int i = 0; i < 5; ++i) {
        unsigned int lo = (i + 8 < 13) ? prod1[i + 8] : 0u;
        unsigned int hi = (i + 9 < 13) ? prod1[i + 9] : 0u;
        c1Words[i] = (lo >> 16) | (hi << 16);
    }
    uint256 c1(c1Words);

    // Compute c2 = round(-b1 * k / n) via ((k * G2) >> 272)
    unsigned int prod2[13];
    mulWords(kWords, 8, G2_WORDS, 5, prod2);
    unsigned int c2Words[8] = {0};
    for(int i = 0; i < 5; ++i) {
        unsigned int lo = (i + 8 < 13) ? prod2[i + 8] : 0u;
        unsigned int hi = (i + 9 < 13) ? prod2[i + 9] : 0u;
        c2Words[i] = (lo >> 16) | (hi << 16);
    }
    uint256 c2(c2Words);

    // k1 = k - c1 * a1 - c2 * a2
    uint256 t1 = c1.mul(uint256(A1_WORDS));
    uint256 t2 = c2.mul(uint256(A2_WORDS));
    uint256 sum = t1.add(t2);

    GLVSplit r;
    if(k.cmp(sum) >= 0) {
        r.k1 = k.sub(sum);
        r.k1Neg = false;
    } else {
        r.k1 = sum.sub(k);
        r.k1Neg = true;
    }

    // k2 = c1 * |b1| - c2 * b2
    uint256 u1 = c1.mul(uint256(B1_WORDS));
    uint256 u2 = c2.mul(uint256(B2_WORDS));
    if(u1.cmp(u2) >= 0) {
        r.k2 = u1.sub(u2);
        r.k2Neg = false;
    } else {
        r.k2 = u2.sub(u1);
        r.k2Neg = true;
    }

    return r;
}

static inline ecpoint glvEndomorphismBasePoint() {
    static const unsigned int xWords[8] = {
        0xBCACE2E9u, 0x9DA01887u, 0xAB0102B6u, 0x96902325u,
        0x87284406u, 0x7F15E98Du, 0xA7BBA044u, 0x00B88FCBu
    };
    static const unsigned int yWords[8] = {
        0x483ADA77u, 0x26A3C465u, 0x5DA4FBFCu, 0x0E1108A8u,
        0xFD17B448u, 0xA6855419u, 0x9C47D08Fu, 0xFB10D4B8u
    };
    return ecpoint(uint256(xWords, uint256::BigEndian), uint256(yWords, uint256::BigEndian));
}

}

#endif
