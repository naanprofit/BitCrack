#include "PollardEngine.h"
#include "secp256k1.h"
#include "AddressUtil.h"
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

// Extract ``bits`` bits starting at ``offset`` (LSB order) from a 160-bit
// little-endian hash represented as five 32-bit words.
uint64_t hashWindowLE(const unsigned int h[5], unsigned int offset,
                      unsigned int bits) {
    unsigned int word = offset / 32;
    unsigned int bit = offset % 32;
    uint64_t val = 0;
    if(word < 5) {
        val = ((uint64_t)h[word]) >> bit;
        if(bit + bits > 32 && word + 1 < 5) {
            val |= ((uint64_t)h[word + 1]) << (32 - bit);
        }
        if(bit + bits > 64 && word + 2 < 5) {
            val |= ((uint64_t)h[word + 2]) << (64 - bit);
        }
    }
    if(bits >= 64) {
        return val;
    }
    return val & (((uint64_t)1 << bits) - 1ULL);
}
} // namespace

PollardEngine::PollardEngine(ResultCallback cb,
                             unsigned int windowBits,
                             const std::vector<unsigned int> &offsets,
                             const std::vector<std::array<unsigned int,5>> &targets)
    : _callback(cb), _windowBits(windowBits), _offsets(offsets) {
    for(const auto &t : targets) {
        TargetState s;
        s.hash = t;
        _targets.push_back(s);
    }
}

void PollardEngine::addConstraint(size_t target, unsigned int bits,
                                  const uint256 &value) {
    if(target >= _targets.size()) {
        return;
    }
    Constraint c{bits, value};
    _targets[target].constraints.push_back(c);
}

bool PollardEngine::reconstruct(size_t target, uint256 &out) {
    if(target >= _targets.size()) {
        return false;
    }
    auto &vec = _targets[target].constraints;
    if(vec.empty()) {
        return false;
    }

    std::vector<Constraint> sorted = vec;
    std::sort(sorted.begin(), sorted.end(),
              [](const Constraint &a, const Constraint &b) {
                  return a.bits < b.bits;
              });

    Constraint acc = sorted[0];
    for(size_t i = 1; i < sorted.size(); ++i) {
        Constraint combined;
        if(!combineCRT(acc, sorted[i], combined)) {
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
    runTameWalk(start, steps, std::random_device{}());
}

void PollardEngine::runTameWalk(const uint256 &start, uint64_t steps, uint64_t seed) {
    // Random-step walk beginning at ``start``.  At each distinguished point
    // the RIPEMD160 hash of the current public key is compared against each
    // target.  When a window of bits matches, the corresponding scalar window
    // is recorded for that target.
    std::mt19937_64 rng(seed);

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
            unsigned int digest[5];
            Hash::hashPublicKeyCompressed(p, digest);

            for(size_t t = 0; t < _targets.size(); ++t) {
                for(unsigned int off : _offsets) {
                    if(off + _windowBits > 160) {
                        continue;
                    }
                    if(_targets[t].seenOffsets.count(off)) {
                        continue;
                    }
                    uint64_t want = hashWindowLE(_targets[t].hash.data(), off, _windowBits);
                    uint64_t got  = hashWindowLE(digest, off, _windowBits);
                    if(got == want) {
                        unsigned int modBits = off + _windowBits;
                        if(modBits > 256) {
                            continue;
                        }
                        uint256 mask = maskBits(modBits);
                        uint256 full;
                        for(int w = 0; w < 8; ++w) {
                            full.v[w] = k.v[w] & mask.v[w];
                        }
                        addConstraint(t, modBits, full);
                        _targets[t].seenOffsets.insert(off);

                        uint256 priv;
                        if(reconstruct(t, priv)) {
                            ecpoint pub = multiplyPoint(priv, G());
                            unsigned int h[5];
                            Hash::hashPublicKeyCompressed(pub, h);
                            bool match = true;
                            for(int w = 0; w < 5; ++w) {
                                if(h[w] != _targets[t].hash[w]) { match = false; break; }
                            }
                            if(match) {
                                enumerateCandidate(priv, pub);
                            }
                        }
                    }
                }
            }
        }
    }
}

void PollardEngine::runWildWalk(const ecpoint &start, uint64_t steps) {
    runWildWalk(start, steps, std::random_device{}());
}

void PollardEngine::runWildWalk(const ecpoint &start, uint64_t steps, uint64_t seed) {
    // Wild walk begins from a supplied point and advances by random multiples
    // of G while tracking the corresponding scalar ``k``.  Window constraints
    // are gathered identically to the tame walk using hash comparisons.
    std::mt19937_64 rng(seed);

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
            unsigned int digest[5];
            Hash::hashPublicKeyCompressed(p, digest);

            for(size_t t = 0; t < _targets.size(); ++t) {
                for(unsigned int off : _offsets) {
                    if(off + _windowBits > 160) {
                        continue;
                    }
                    if(_targets[t].seenOffsets.count(off)) {
                        continue;
                    }
                    uint64_t want = hashWindowLE(_targets[t].hash.data(), off, _windowBits);
                    uint64_t got  = hashWindowLE(digest, off, _windowBits);
                    if(got == want) {
                        unsigned int modBits = off + _windowBits;
                        if(modBits > 256) {
                            continue;
                        }
                        uint256 mask = maskBits(modBits);
                        uint256 full;
                        for(int w = 0; w < 8; ++w) {
                            full.v[w] = k.v[w] & mask.v[w];
                        }
                        addConstraint(t, modBits, full);
                        _targets[t].seenOffsets.insert(off);

                        uint256 priv;
                        if(reconstruct(t, priv)) {
                            ecpoint pub = multiplyPoint(priv, G());
                            unsigned int h[5];
                            Hash::hashPublicKeyCompressed(pub, h);
                            bool match = true;
                            for(int w = 0; w < 5; ++w) {
                                if(h[w] != _targets[t].hash[w]) { match = false; break; }
                            }
                            if(match) {
                                enumerateCandidate(priv, pub);
                            }
                        }
                    }
                }
            }
        }
    }
}

uint64_t PollardEngine::hashWindow(const unsigned int h[5], unsigned int offset,
                                   unsigned int bits) {
    return hashWindowLE(h, offset, bits);
}

