#include "PollardEngine.h"
#include "secp256k1.h"
#include <algorithm>
#include <cstring>
#include <vector>


using namespace secp256k1;

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

    uint256 x = _constraints[0].value;
    unsigned int known = _constraints[0].bits;

    for(size_t i = 1; i < _constraints.size(); ++i) {
        const Constraint &c = _constraints[i];

        unsigned int overlap = std::min(known, c.bits);

        // Mask covering the overlapping region.
        uint256 mask(0);
        for(unsigned int j = 0; j < overlap; ++j) {
            mask.v[j / 32] |= (1u << (j % 32));
        }

        // Ensure the overlapping bits agree.
        for(int w = 0; w < 8; ++w) {
            if((x.v[w] & mask.v[w]) != (c.value.v[w] & mask.v[w])) {
                return false; // inconsistent constraint
            }
        }

        if(c.bits > known) {
            // Merge in new higher bits.
            for(unsigned int bit = overlap; bit < c.bits; ++bit) {
                unsigned int word = bit / 32;
                unsigned int shift = bit % 32;
                if(c.value.v[word] & (1u << shift)) {
                    x.v[word] |= (1u << shift);
                }
            }
            known = c.bits;
        }
    }

    out = x;
    return known >= 256;
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
    // Sequential walk beginning at ``start``.  Whenever a distinguished point is
    // encountered we record windows from the current private key and feed them
    // to the CRT solver.
    uint256 k = start;
    ecpoint p = multiplyPoint(k, G());

    for(uint64_t i = 0; i < steps; ++i) {
        if(checkPoint(p)) {
            for(unsigned int off : _offsets) {
                uint64_t modBits = off + _windowBits;
                if(modBits >= 64) {
                    continue; // limitation of 64-bit accumulation
                }
                uint64_t full = k.toUint64() & ((1ULL << modBits) - 1ULL);
                addConstraint(modBits, uint256(full));
            }

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
    // Wild walk begins from a supplied point and advances by G each step while
    // tracking the corresponding scalar ``k``.  Window constraints are gathered
    // identically to the tame walk.
    ecpoint p = start;
    uint256 k(0);
    for(uint64_t i = 0; i < steps; ++i) {
        if(checkPoint(p)) {
            for(unsigned int off : _offsets) {
                uint64_t modBits = off + _windowBits;
                if(modBits >= 64) {
                    continue;
                }
                uint64_t full = k.toUint64() & ((1ULL << modBits) - 1ULL);
                addConstraint(modBits, uint256(full));
            }

            uint256 priv;
            if(reconstruct(priv)) {
                enumerateCandidate(priv, multiplyPoint(priv, G()));
            }
        }

        k = k.add(uint256(1));
        p = addPoints(p, G());
    }
}

