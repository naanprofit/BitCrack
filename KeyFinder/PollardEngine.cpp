#include "PollardEngine.h"
#include "secp256k1.h"
#include <cstring>
#include <cstdint>

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

    uint256 x(0);
    uint256 M(1);

    for(const Constraint &c : _constraints) {
        uint64_t mod = 1ULL << c.bits;
        uint64_t rem = c.value.toUint64() & (mod - 1);
        uint64_t xmod = x.mod(static_cast<uint32_t>(mod)).toUint64();
        uint64_t diff = (rem + mod - xmod) % mod;
        x = x.add(M.mul(diff));
        M = M.mul(mod);
    }

    out = x;
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

