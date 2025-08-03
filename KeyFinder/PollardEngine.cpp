#include "PollardEngine.h"
#include "secp256k1.h"
#include "AddressUtil.h"
#include <algorithm>
#include <cstring>
#include <vector>
#include <random>
#include <limits>
#include <chrono>
#include "../util/RingBuffer.h"
#include "../util/util.h"
#include "../Logger/Logger.h"

using namespace secp256k1;

namespace {

// Simple CPU implementation of the device interface used for unit tests and
// as a fallback when no GPU is available.
class CPUPollardDevice : public PollardDevice {
    unsigned int _windowBits;
    SimpleRingBuffer<PollardMatch> _buffer;

    bool checkPoint(const ecpoint &p) {
        uint64_t mask = ((uint64_t)1 << _windowBits) - 1ULL;
        return (p.x.v[0] & mask) == 0;
    }

public:
    explicit CPUPollardDevice(unsigned int windowBits)
        : _windowBits(windowBits), _buffer(1024) {}

    void startTameWalk(const uint256 &start, uint64_t steps, uint64_t seed) override;
    void startWildWalk(const ecpoint &start, uint64_t steps, uint64_t seed) override;
    bool popResult(PollardMatch &out) override { return _buffer.pop(out); }
};

void CPUPollardDevice::startTameWalk(const uint256 &start, uint64_t steps, uint64_t seed) {
    _buffer.clear();
    std::mt19937_64 rng(seed);
    uint64_t maxStep = (_windowBits >= 64) ? std::numeric_limits<uint64_t>::max() : ((1ULL << _windowBits) - 1ULL);
    std::uniform_int_distribution<uint64_t> dist(1, maxStep);

    uint256 k = start;
    ecpoint p = multiplyPoint(k, G());

    for(uint64_t i = 0; i < steps; ++i) {
        uint64_t step = dist(rng);
        uint256 stepVal(step);
        k = k.add(stepVal);
        p = addPoints(p, multiplyPoint(stepVal, G()));
        if(checkPoint(p)) {
            PollardMatch m;
            m.scalar = k;
            Hash::hashPublicKeyCompressed(p, m.hash);
            _buffer.push(m);
        }
    }
}

void CPUPollardDevice::startWildWalk(const ecpoint &start, uint64_t steps, uint64_t seed) {
    _buffer.clear();
    std::mt19937_64 rng(seed);
    uint64_t maxStep = (_windowBits >= 64) ? std::numeric_limits<uint64_t>::max() : ((1ULL << _windowBits) - 1ULL);
    std::uniform_int_distribution<uint64_t> dist(1, maxStep);

    ecpoint p = start;
    uint256 k(0);

    for(uint64_t i = 0; i < steps; ++i) {
        uint64_t step = dist(rng);
        uint256 stepVal(step);
        k = k.add(stepVal);
        p = addPoints(p, multiplyPoint(stepVal, G()));
        if(checkPoint(p)) {
            PollardMatch m;
            m.scalar = k;
            Hash::hashPublicKeyCompressed(p, m.hash);
            _buffer.push(m);
        }
    }
}

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
    _device = std::unique_ptr<PollardDevice>(new CPUPollardDevice(windowBits));
}

void PollardEngine::setDevice(std::unique_ptr<PollardDevice> device) {
    _device = std::move(device);
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

void PollardEngine::processWindow(const PollardWindow &w) {
    _windowsProcessed++;
    if(w.targetIdx >= _targets.size()) {
        return;
    }
    if(_targets[w.targetIdx].seenOffsets.count(w.offset)) {
        return;
    }

    unsigned int modBits = w.offset + w.bits;
    if(modBits > 256) {
        return;
    }

    addConstraint(w.targetIdx, modBits, w.scalarFragment);
    _targets[w.targetIdx].seenOffsets.insert(w.offset);

    _reconstructionAttempts++;
    uint256 priv;
    if(reconstruct(w.targetIdx, priv)) {
        ecpoint pub = multiplyPoint(priv, G());
        unsigned int h[5];
        Hash::hashPublicKeyCompressed(pub, h);
        bool match = true;
        for(int i = 0; i < 5; ++i) {
            if(h[i] != _targets[w.targetIdx].hash[i]) {
                match = false;
                break;
            }
        }
        if(match) {
            _reconstructionSuccess++;
            if(_callback) {
                KeySearchResult r;
                r.privateKey = priv;
                r.publicKey = pub;
                r.compressed = true;
                std::memcpy(r.hash, h, sizeof(h));
                r.address = Address::fromPublicKey(pub, true);
                _callback(r);
            }
        }
    }

    if((_windowsProcessed % 1000) == 0) {
        auto now = std::chrono::steady_clock::now();
        double secs = std::chrono::duration_cast<std::chrono::milliseconds>(now - _startTime).count() / 1000.0;
        if(secs > 0) {
            double rate = static_cast<double>(_windowsProcessed) / secs;
            Logger::log(LogLevel::Info,
                        "GPU throughput: " + util::format(static_cast<uint64_t>(rate)) + " windows/s");
        }
    }
}

void PollardEngine::enumerateCandidate(const uint256 &priv, const ecpoint &pub) {
    if(!_callback) {
        return;
    }
    // Compute the candidate's hash160 and ensure it matches one of the
    // supplied targets before invoking the callback.  This avoids reporting
    // non-matching candidates that may appear during the walk.
    unsigned int digest[5];
    Hash::hashPublicKeyCompressed(pub, digest);

    bool match = false;
    for(const auto &t : _targets) {
        bool same = true;
        for(int i = 0; i < 5; ++i) {
            if(digest[i] != t.hash[i]) {
                same = false;
                break;
            }
        }
        if(same) {
            match = true;
            break;
        }
    }

    if(!match) {
        return; // Candidate does not belong to the target set
    }

    KeySearchResult r;
    r.privateKey = priv;
    r.publicKey = pub;
    r.compressed = true;
    std::memcpy(r.hash, digest, sizeof(r.hash));
    r.address = Address::fromPublicKey(pub, true);

    _callback(r);
}

void PollardEngine::runTameWalk(const uint256 &start, uint64_t steps) {
    runTameWalk(start, steps, std::random_device{}());
}

void PollardEngine::runTameWalk(const uint256 &start, uint64_t steps, uint64_t seed) {
    if(!_device) {
        return;
    }

    _windowsProcessed = _reconstructionAttempts = _reconstructionSuccess = 0;
    _startTime = std::chrono::steady_clock::now();

    _device->startTameWalk(start, steps, seed);
    PollardMatch m;
    while(_device->popResult(m)) {
        for(size_t t = 0; t < _targets.size(); ++t) {
            for(unsigned int off : _offsets) {
                if(off + _windowBits > 160) {
                    continue;
                }
                if(_targets[t].seenOffsets.count(off)) {
                    continue;
                }
                uint64_t want = hashWindowLE(_targets[t].hash.data(), off, _windowBits);
                uint64_t got  = hashWindowLE(m.hash, off, _windowBits);
                if(got == want) {
                    unsigned int modBits = off + _windowBits;
                    if(modBits > 256) {
                        continue;
                    }
                    uint256 mask = maskBits(modBits);
                    uint256 full;
                    for(int w = 0; w < 8; ++w) {
                        full.v[w] = m.scalar.v[w] & mask.v[w];
                    }
                    PollardWindow w{static_cast<unsigned int>(t), off, _windowBits, full};
                    processWindow(w);
                }
            }
        }
    }

    auto end = std::chrono::steady_clock::now();
    double secs = std::chrono::duration_cast<std::chrono::milliseconds>(end - _startTime).count() / 1000.0;
    if(secs > 0) {
        double rate = static_cast<double>(_windowsProcessed) / secs;
        Logger::log(LogLevel::Info,
                    "GPU throughput: " + util::format(static_cast<uint64_t>(rate)) + " windows/s");
    }
    Logger::log(LogLevel::Info,
                "CPU reconstructions: " + util::format(_reconstructionSuccess) + "/" +
                util::format(_reconstructionAttempts));
}

void PollardEngine::runWildWalk(const ecpoint &start, uint64_t steps) {
    runWildWalk(start, steps, std::random_device{}());
}

void PollardEngine::runWildWalk(const ecpoint &start, uint64_t steps, uint64_t seed) {
    if(!_device) {
        return;
    }

    _windowsProcessed = _reconstructionAttempts = _reconstructionSuccess = 0;
    _startTime = std::chrono::steady_clock::now();

    _device->startWildWalk(start, steps, seed);
    PollardMatch m;
    while(_device->popResult(m)) {
        for(size_t t = 0; t < _targets.size(); ++t) {
            for(unsigned int off : _offsets) {
                if(off + _windowBits > 160) {
                    continue;
                }
                if(_targets[t].seenOffsets.count(off)) {
                    continue;
                }
                uint64_t want = hashWindowLE(_targets[t].hash.data(), off, _windowBits);
                uint64_t got  = hashWindowLE(m.hash, off, _windowBits);
                if(got == want) {
                    unsigned int modBits = off + _windowBits;
                    if(modBits > 256) {
                        continue;
                    }
                    uint256 mask = maskBits(modBits);
                    uint256 full;
                    for(int w = 0; w < 8; ++w) {
                        full.v[w] = m.scalar.v[w] & mask.v[w];
                    }
                    PollardWindow w{static_cast<unsigned int>(t), off, _windowBits, full};
                    processWindow(w);
                }
            }
        }
    }

    auto end = std::chrono::steady_clock::now();
    double secs = std::chrono::duration_cast<std::chrono::milliseconds>(end - _startTime).count() / 1000.0;
    if(secs > 0) {
        double rate = static_cast<double>(_windowsProcessed) / secs;
        Logger::log(LogLevel::Info,
                    "GPU throughput: " + util::format(static_cast<uint64_t>(rate)) + " windows/s");
    }
    Logger::log(LogLevel::Info,
                "CPU reconstructions: " + util::format(_reconstructionSuccess) + "/" +
                util::format(_reconstructionAttempts));
}

uint64_t PollardEngine::hashWindow(const unsigned int h[5], unsigned int offset,
                                   unsigned int bits) {
    return hashWindowLE(h, offset, bits);
}

