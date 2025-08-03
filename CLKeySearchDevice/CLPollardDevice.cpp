#include "CLPollardDevice.h"
#include <random>
#include <limits>
#include <cstring>
#include "AddressUtil.h"

using namespace secp256k1;

CLPollardDevice::CLPollardDevice(PollardEngine &engine,
                                 unsigned int windowBits,
                                 const std::vector<unsigned int> &offsets,
                                 const std::vector<std::array<unsigned int,5>> &targets)
    : _engine(engine), _windowBits(windowBits), _offsets(offsets), _targets(targets) {}

uint256 CLPollardDevice::maskBits(unsigned int bits) {
    uint256 m(0);
    for(unsigned int i = 0; i < bits; ++i) {
        m.v[i/32] |= (1u << (i % 32));
    }
    return m;
}

uint64_t CLPollardDevice::hashWindowLE(const unsigned int h[5], unsigned int offset, unsigned int bits) {
    unsigned int word = offset / 32;
    unsigned int bit = offset % 32;
    uint64_t val = 0;
    if(word < 5) {
        val = ((uint64_t)h[word]) >> bit;
        if(bit + bits > 32 && word + 1 < 5) {
            val |= ((uint64_t)h[word + 1]) << (32 - bit);
        }
    }
    if(bit + bits > 64 && word + 2 < 5) {
        val |= ((uint64_t)h[word + 2]) << (64 - bit);
    }
    if(bits < 64) {
        uint64_t mask = (bits == 64) ? 0xffffffffffffffffULL : ((1ULL << bits) - 1ULL);
        val &= mask;
    }
    return val;
}

void CLPollardDevice::startTameWalk(const uint256 &start, uint64_t steps, uint64_t seed) {
    uint256 k = start;
    ecpoint p = multiplyPoint(k, G());
    std::mt19937_64 rng(seed);
    uint64_t maxStep = (_windowBits >= 64) ? std::numeric_limits<uint64_t>::max() : ((1ULL << _windowBits) - 1ULL);
    std::uniform_int_distribution<uint64_t> dist(1, maxStep);

    for(uint64_t i = 0; i < steps; ++i) {
        uint64_t step = dist(rng);
        uint256 stepVal(step);
        k = k.add(stepVal);
        p = addPoints(p, multiplyPoint(stepVal, G()));
        uint64_t mask = (_windowBits >= 64) ? 0xffffffffffffffffULL : ((1ULL << _windowBits) - 1ULL);
        if((p.x.v[0] & mask) == 0) {
            PollardMatch m;
            m.scalar = k;
            Hash::hashPublicKeyCompressed(p, m.hash);
            for(size_t t = 0; t < _targets.size(); ++t) {
                for(unsigned int off : _offsets) {
                    if(off + _windowBits > 160) continue;
                    uint64_t want = hashWindowLE(_targets[t].data(), off, _windowBits);
                    uint64_t got  = hashWindowLE(m.hash, off, _windowBits);
                    if(got == want) {
                        unsigned int modBits = off + _windowBits;
                        if(modBits > 256) continue;
                        uint256 maskBitsVal = maskBits(modBits);
                        uint256 frag;
                        for(int w = 0; w < 8; ++w) {
                            frag.v[w] = m.scalar.v[w] & maskBitsVal.v[w];
                        }
                        PollardWindow w{static_cast<unsigned int>(t), off, _windowBits, frag};
                        _engine.processWindow(w);
                    }
                }
            }
        }
    }
}

void CLPollardDevice::startWildWalk(const ecpoint &start, uint64_t steps, uint64_t seed) {
    ecpoint p = start;
    uint256 k(0);
    std::mt19937_64 rng(seed);
    uint64_t maxStep = (_windowBits >= 64) ? std::numeric_limits<uint64_t>::max() : ((1ULL << _windowBits) - 1ULL);
    std::uniform_int_distribution<uint64_t> dist(1, maxStep);

    for(uint64_t i = 0; i < steps; ++i) {
        uint64_t step = dist(rng);
        uint256 stepVal(step);
        k = k.add(stepVal);
        p = addPoints(p, multiplyPoint(stepVal, G()));
        uint64_t mask = (_windowBits >= 64) ? 0xffffffffffffffffULL : ((1ULL << _windowBits) - 1ULL);
        if((p.x.v[0] & mask) == 0) {
            PollardMatch m;
            m.scalar = k;
            Hash::hashPublicKeyCompressed(p, m.hash);
            for(size_t t = 0; t < _targets.size(); ++t) {
                for(unsigned int off : _offsets) {
                    if(off + _windowBits > 160) continue;
                    uint64_t want = hashWindowLE(_targets[t].data(), off, _windowBits);
                    uint64_t got  = hashWindowLE(m.hash, off, _windowBits);
                    if(got == want) {
                        unsigned int modBits = off + _windowBits;
                        if(modBits > 256) continue;
                        uint256 maskBitsVal = maskBits(modBits);
                        uint256 frag;
                        for(int w = 0; w < 8; ++w) {
                            frag.v[w] = m.scalar.v[w] & maskBitsVal.v[w];
                        }
                        PollardWindow w{static_cast<unsigned int>(t), off, _windowBits, frag};
                        _engine.processWindow(w);
                    }
                }
            }
        }
    }
}
