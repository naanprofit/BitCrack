#ifndef POLLARD_TYPES_H
#define POLLARD_TYPES_H
#include "secp256k1.h"
struct PollardMatch {
    secp256k1::uint256 scalar;
    unsigned int hash[5];
};
#endif
