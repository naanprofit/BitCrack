#include "PollardEngine.h"
#include "secp256k1.h"
#include <cstring>

using namespace secp256k1;

PollardEngine::PollardEngine(ResultCallback cb) : _callback(cb) {
}

void PollardEngine::addConstraint(unsigned int bits, const uint256 &value) {
    Constraint c{bits, value};
    _constraints.push_back(c);
}

bool PollardEngine::reconstruct(uint256 &out) {
    if(_constraints.empty()) {
        return false;
    }

    out = uint256(0);
    unsigned int shift = 0;

    for(const Constraint &c : _constraints) {
        uint256 val = c.value;
        for(unsigned int i = 0; i < c.bits; ++i) {
            if(val.bit(i)) {
                uint256 bitVal = uint256(2).pow(i + shift);
                out = out.add(bitVal);
            }
        }
        shift += c.bits;
    }

    return true;
}

bool PollardEngine::checkPoint(const ecpoint &p) {
    // Simple window check: low 16 bits of x are zero
    return (p.x.v[0] & 0xFFFF) == 0;
}

void PollardEngine::enumerateCandidate(const uint256 &priv, const ecpoint &pub) {
    if(!_callback) {
        return;
    }

    KeySearchResult r;
    r.privateKey = priv;
    r.publicKey = pub;
    r.compressed = true;
    std::memset(r.hash, 0, sizeof(r.hash));
    r.address = "";

    _callback(r);
}

void PollardEngine::runTameWalk(const uint256 &start, uint64_t steps) {
    uint256 k = start;
    ecpoint p = multiplyPoint(k, G());

    for(uint64_t i = 0; i < steps; ++i) {
        if(checkPoint(p)) {
            uint256 rem(k.v[0] & 0xFFFF);
            addConstraint(16, rem);
            uint256 priv;
            if(reconstruct(priv)) {
                enumerateCandidate(priv, multiplyPoint(priv, G()));
            }
        }

        k = k.add(uint256(1));
        p = addPoints(p, G());
    }
}

void PollardEngine::runWildWalk(const ecpoint &start, uint64_t steps) {
    ecpoint p = start;
    uint256 k(0);

    for(uint64_t i = 0; i < steps; ++i) {
        if(checkPoint(p)) {
            uint256 rem(k.v[0] & 0xFFFF);
            addConstraint(16, rem);
            uint256 priv;
            if(reconstruct(priv)) {
                enumerateCandidate(priv, multiplyPoint(priv, G()));
            }
        }

        k = k.add(uint256(1));
        p = addPoints(p, G());
    }
}

