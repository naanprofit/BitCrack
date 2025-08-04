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

static inline GLVSplit splitScalar(const uint256 &k) {
    GLVSplit r;
    r.k1 = k;
    r.k2 = uint256(0);
    r.k1Neg = false;
    r.k2Neg = false;
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
