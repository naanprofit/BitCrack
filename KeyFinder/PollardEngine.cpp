#include "PollardEngine.h"
#include "secp256k1.h"
#include <algorithm>
#include <cstring>
#include <vector>
#include <random>
#include <limits>

using namespace secp256k1;

namespace {
// Create a mask with the lowest ``bits`` bits set.
uint256 maskBits(unsigned int bits) {
    uint256 m(0);
    for(unsigned int i = 0; i < bits; ++i) {
        m.v[i / 32] |= (1u << (i % 32));
    }
    return m;
}

// Combine two congruences using a simple CRT solver tailored for moduli
// that are powers of two.  The resulting constraint contains the union of
// known low bits from ``a`` and ``b``.
bool combineCRT(const PollardEngine::Constraint &a,
                const PollardEngine::Constraint &b,
                PollardEngine::Constraint &out) {
    unsigned int overlap = std::min(a.bits, b.bits);
    uint256 mask = maskBits(overlap);

    for(int w = 0; w < 8; ++w) {
        if((a.value.v[w] & mask.v[w]) != (b.value.v[w] & mask.v[w])) {
            return false; // inconsistent constraints
        }
    }

    out = (a.bits >= b.bits) ? a : b;
    return true;
}
} // namespace

PollardEngine::PollardEngine(ResultCallback cb,
                             unsigned int windowBits,
                             const std::vector<unsigned int> &offsets)
    : _callback(cb), _windowBits(windowBits), _offsets(offsets) {}

void PollardEngine::addConstraint(unsigned int bits, const uint256 &value) {
    Constraint c{bits, value};
    _constraints.push_back(c);
}

bool PollardEngine::reconstruct(uint256 &out) {
    if(_constraints.empty()) {
        return false;
    }

    // Sort constraints by modulus size.  Since each modulus is a power of two
    // this is equivalent to sorting by the number of known low bits.
    std::sort(_constraints.begin(), _constraints.end(),
              [](const Constraint &a, const Constraint &b) {
                  return a.bits < b.bits;
              });

    Constraint acc = _constraints[0];

    for(size_t i = 1; i < _constraints.size(); ++i) {
        Constraint combined;
        if(!combineCRT(acc, _constraints[i], combined)) {
            return false;
        }
        acc = combined;
    }

    out = acc.value;
    return acc.bits >= 256;
}

bool PollardEngine::checkPoint(const ecpoint &p) {
    uint64_t mask = ((uint64_t)1 << _windowBits) - 1ULL;
    return (p.x.v[0] & mask) == 0;
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
    // Random-step walk beginning at ``start``.  At each distinguished point
    // windows from the current scalar are recorded and fed to the CRT solver.
    std::mt19937_64 rng(std::random_device{}());

    uint64_t maxStep;
    if(_windowBits >= 64) {
        maxStep = std::numeric_limits<uint64_t>::max();
    } else {
        maxStep = (1ULL << _windowBits) - 1ULL;
    }
    std::uniform_int_distribution<uint64_t> dist(1, maxStep);

    uint256 k = start;
    ecpoint p = multiplyPoint(k, G());

    for(uint64_t i = 0; i < steps; ++i) {
        uint64_t step = dist(rng);
        uint256 stepVal(step);
        k = k.add(stepVal);
        p = addPoints(p, multiplyPoint(stepVal, G()));

        if(checkPoint(p)) {
            for(unsigned int off : _offsets) {
                unsigned int modBits = off + _windowBits;
                if(modBits > 256) {
                    continue;
                }
                uint256 mask = maskBits(modBits);
                uint256 full;
                for(int w = 0; w < 8; ++w) {
                    full.v[w] = k.v[w] & mask.v[w];
                }
                addConstraint(modBits, full);
            }

            uint256 priv;
            if(reconstruct(priv)) {
                enumerateCandidate(priv, multiplyPoint(priv, G()));
            }
        }
    }
}

void PollardEngine::runWildWalk(const ecpoint &start, uint64_t steps) {
    // Wild walk begins from a supplied point and advances by random multiples
    // of G while tracking the corresponding scalar ``k``.  Window constraints
    // are gathered identically to the tame walk.
    std::mt19937_64 rng(std::random_device{}());

    uint64_t maxStep;
    if(_windowBits >= 64) {
        maxStep = std::numeric_limits<uint64_t>::max();
    } else {
        maxStep = (1ULL << _windowBits) - 1ULL;
    }
    std::uniform_int_distribution<uint64_t> dist(1, maxStep);

    ecpoint p = start;
    uint256 k(0);

    for(uint64_t i = 0; i < steps; ++i) {
        uint64_t step = dist(rng);
        uint256 stepVal(step);
        k = k.add(stepVal);
        p = addPoints(p, multiplyPoint(stepVal, G()));

        if(checkPoint(p)) {
            for(unsigned int off : _offsets) {
                unsigned int modBits = off + _windowBits;
                if(modBits > 256) {
                    continue;
                }
                uint256 mask = maskBits(modBits);
                uint256 full;
                for(int w = 0; w < 8; ++w) {
                    full.v[w] = k.v[w] & mask.v[w];
                }
                addConstraint(modBits, full);
            }

            uint256 priv;
            if(reconstruct(priv)) {
                enumerateCandidate(priv, multiplyPoint(priv, G()));
            }
        }
    }
}

