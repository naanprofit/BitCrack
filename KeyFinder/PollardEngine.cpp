#include "PollardEngine.h"
#include "secp256k1.h"
#include "AddressUtil.h"
#include <algorithm>
#include <cstring>
#include <vector>
#include <random>
#include <limits>
#include <chrono>
#include <thread>
#include <sstream>
#include <iomanip>
#include "../util/RingBuffer.h"
#include "../util/util.h"
#include "../Logger/Logger.h"

using namespace secp256k1;

static std::string hashToString(const unsigned int h[5]) {
    std::ostringstream ss;
    ss << std::hex << std::setfill('0');
    for(int i=0;i<5;i++) {
        ss << std::setw(8) << h[i];
    }
    return ss.str();
}

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

    void startTameWalk(const uint256 &start, uint64_t steps,
                       const uint256 &seed, bool sequential) override;
    void startWildWalk(const uint256 &start, uint64_t steps,
                       const uint256 &seed, bool sequential) override;
    bool popResult(PollardMatch &out) override { return _buffer.pop(out); }
};

void CPUPollardDevice::startTameWalk(const uint256 &start, uint64_t steps,
                                     const uint256 &seed, bool sequential) {
    _buffer.clear();
    uint256 k = start;
    ecpoint p = multiplyPoint(k, G());
    if(sequential) {
        for(uint64_t i = 0; i < steps; ++i) {
            if(checkPoint(p)) {
                PollardMatch m;
                m.scalar = k;
                unsigned int be[5];
                Hash::hashPublicKeyCompressed(p, be);
                for(int j = 0; j < 5; ++j) {
                    m.hash[j] = util::endian(be[4 - j]);
                }
                _buffer.push(m);
            }
            k = k.add(1);
            p = addPoints(p, G());
        }
        return;
    }

    std::mt19937_64 rng(seed.toUint64());
    uint64_t maxStep = (_windowBits >= 64) ? std::numeric_limits<uint64_t>::max() : ((1ULL << _windowBits) - 1ULL);
    std::uniform_int_distribution<uint64_t> dist(1, maxStep);
    for(uint64_t i = 0; i < steps; ++i) {
        uint64_t step = dist(rng);
        uint256 stepVal(step);
        k = k.add(stepVal);
        p = addPoints(p, multiplyPoint(stepVal, G()));
        if(checkPoint(p)) {
            PollardMatch m;
            m.scalar = k;
            unsigned int be[5];
            Hash::hashPublicKeyCompressed(p, be);
            for(int j = 0; j < 5; ++j) {
                m.hash[j] = util::endian(be[4 - j]);
            }
            _buffer.push(m);
        }
    }
}

void CPUPollardDevice::startWildWalk(const uint256 &start, uint64_t steps,
                                     const uint256 &seed, bool sequential) {
    _buffer.clear();
    uint256 k = start;
    ecpoint p = multiplyPoint(k, G());
    if(sequential) {
        ecpoint negG = G();
        negG.y = negModP(negG.y);
        for(uint64_t i = 0; i < steps; ++i) {
            if(checkPoint(p)) {
                PollardMatch m;
                m.scalar = k;
                unsigned int be[5];
                Hash::hashPublicKeyCompressed(p, be);
                for(int j = 0; j < 5; ++j) {
                    m.hash[j] = util::endian(be[4 - j]);
                }
                _buffer.push(m);
            }
            if(k.isZero()) {
                break;
            }
            k = k.sub(1);
            p = addPoints(p, negG);
        }
        return;
    }

    std::mt19937_64 rng(seed.toUint64());
    uint64_t maxStep = (_windowBits >= 64) ? std::numeric_limits<uint64_t>::max() : ((1ULL << _windowBits) - 1ULL);
    std::uniform_int_distribution<uint64_t> dist(1, maxStep);

    uint256 kOffset(0);
    for(uint64_t i = 0; i < steps; ++i) {
        uint64_t step = dist(rng);
        uint256 stepVal(step);
        kOffset = kOffset.add(stepVal);
        p = addPoints(p, multiplyPoint(stepVal, G()));
        if(checkPoint(p)) {
            PollardMatch m;
            m.scalar = kOffset;
            unsigned int be[5];
            Hash::hashPublicKeyCompressed(p, be);
            for(int j = 0; j < 5; ++j) {
                m.hash[j] = util::endian(be[4 - j]);
            }
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
// little-endian hash represented as five 32-bit words.  The result is
// returned in the lower bits of a uint256 value.
uint256 hashWindowLE(const unsigned int h[5], unsigned int offset,
                     unsigned int bits) {
    uint256 out(0);
    unsigned int word = offset / 32;
    unsigned int bit = offset % 32;
    unsigned int words = (bits + 31) / 32;
    // Extract required words with cross-boundary handling
    for(unsigned int i = 0; i < words && word + i < 5; ++i) {
        uint64_t val = ((uint64_t)h[word + i]) >> bit;
        if(bit && word + i + 1 < 5) {
            val |= ((uint64_t)h[word + i + 1]) << (32 - bit);
        }
        out.v[i] = static_cast<unsigned int>(val & 0xffffffffULL);
    }
    // Mask off any excess bits in the most significant word
    if(bits % 32) {
        unsigned int mask = (1u << (bits % 32)) - 1u;
        out.v[words - 1] &= mask;
    }
    // Ensure higher words are zero
    for(unsigned int i = words; i < 8; ++i) {
        out.v[i] = 0u;
    }
    return out;
}

} // namespace

void PollardEngine::handleMatch(const PollardMatch &m) {
    for(size_t t = 0; t < _targets.size(); ++t) {
        for(unsigned int off : _offsets) {
            if(off + _windowBits > 160) {
                continue;
            }
            if(_targets[t].seenOffsets.count(off)) {
                continue;
            }
            uint256 want = hashWindowLE(_targets[t].hash.data(), off, _windowBits);
            uint256 got  = hashWindowLE(m.hash, off, _windowBits);
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

void PollardEngine::pollDevice() {
    if(!_device) {
        return;
    }
    PollardMatch m;
    auto delay = std::chrono::milliseconds(_pollInterval);
    while(true) {
        size_t processed = 0;
        while(processed < _batchSize && _device->popResult(m)) {
            handleMatch(m);
            processed++;
        }
        if(processed == 0) {
            std::this_thread::sleep_for(delay);
            if(!_device->popResult(m)) {
                break;
            }
            handleMatch(m);
        }
        std::this_thread::sleep_for(delay);
    }
}

PollardEngine::PollardEngine(ResultCallback cb,
                             unsigned int windowBits,
                             const std::vector<unsigned int> &offsets,
                             const std::vector<std::array<unsigned int,5>> &targets,
                             const uint256 &L,
                             const uint256 &U,
                             unsigned int batchSize,
                             unsigned int pollInterval,
                             bool sequential,
                             bool debug)
    : _callback(cb), _windowBits(windowBits), _offsets(offsets),
      _batchSize(batchSize), _pollInterval(pollInterval), _L(L), _U(U),
      _sequential(sequential), _debug(debug) {
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

bool PollardEngine::reconstruct(size_t target, uint256 &k0, uint256 &modulus) {
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

    k0 = acc.value;
    // modulus = 2^bits
    if(acc.bits >= 256) {
        modulus = uint256(0); // represents 2^256
    } else {
        modulus = uint256(0);
        modulus.v[acc.bits / 32] = (1u << (acc.bits % 32));
    }

    return true;
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

    if(_debug) {
        Logger::log(LogLevel::Debug,
                    "Window target=" + util::format(w.targetIdx) +
                    " offset=" + util::format(w.offset) +
                    " bits=" + util::format(w.bits) +
                    " fragment=" + secp256k1::uint256(w.scalarFragment).toString(16));
    }

    unsigned int modBits = w.offset + w.bits;
    if(modBits > 256) {
        return;
    }

    uint256 val = w.scalarFragment;
    if(modBits < 256) {
        unsigned int word = modBits / 32;
        unsigned int bit = modBits % 32;
        if(bit) {
            unsigned int mask = (1u << bit) - 1u;
            val.v[word] &= mask;
            for(unsigned int i = word + 1; i < 8; ++i) {
                val.v[i] = 0u;
            }
        } else {
            for(unsigned int i = word; i < 8; ++i) {
                val.v[i] = 0u;
            }
        }
    }

    addConstraint(w.targetIdx, modBits, val);
    _targets[w.targetIdx].seenOffsets.insert(w.offset);

    _reconstructionAttempts++;
    uint256 k0;
    uint256 modulus;
    if(reconstruct(w.targetIdx, k0, modulus)) {
        enumerateCandidates(k0, modulus, _L, _U);
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
    unsigned int be[5];
    Hash::hashPublicKeyCompressed(pub, be);
    unsigned int digest[5];
    for(int i = 0; i < 5; ++i) {
        digest[i] = util::endian(be[4 - i]);
    }

    if(_debug) {
        Logger::log(LogLevel::Debug,
                    "k=" + secp256k1::uint256(priv).toString(16) +
                    " Px=" + secp256k1::uint256(pub.x).toString(16) +
                    " Py=" + secp256k1::uint256(pub.y).toString(16) +
                    " hash=" + hashToString(be));
    }

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

    _reconstructionSuccess++;

    KeySearchResult r;
    r.privateKey = priv;
    r.publicKey = pub;
    r.compressed = true;
    std::memcpy(r.hash, digest, sizeof(r.hash));
    r.address = Address::fromPublicKey(pub, true);

    _callback(r);
}

void PollardEngine::enumerateCandidates(const uint256 &k0, const uint256 &modulus,
                                       const uint256 &L, const uint256 &U) {
    // Handle full reconstruction separately (modulus == 0 represents 2^256)
    if(modulus.isZero()) {
        if(k0.cmp(L) >= 0 && k0.cmp(U) <= 0) {
            ecpoint pub = multiplyPoint(k0, G());
            enumerateCandidate(k0, pub);
        }
        return;
    }

    uint256 k = k0;
    // Move k into the search range [L, U]
    if(k.cmp(L) < 0) {
        while(k.cmp(L) < 0) {
            uint256 next = k.add(modulus);
            if(next.cmp(k) <= 0) {
                // overflow, no candidates
                return;
            }
            k = next;
        }
    }

    for(; k.cmp(U) <= 0; k = k.add(modulus)) {
        ecpoint pub = multiplyPoint(k, G());
        enumerateCandidate(k, pub);
    }
}

void PollardEngine::runTameWalk(const uint256 &start, uint64_t steps) {
    runTameWalk(start, steps, uint256(std::random_device{}()));
}

void PollardEngine::runTameWalk(const uint256 &start, uint64_t steps, const uint256 &seed) {
    if(!_device) {
        return;
    }

    _windowsProcessed = _reconstructionAttempts = _reconstructionSuccess = 0;
    _startTime = std::chrono::steady_clock::now();

    const uint256 &s = _sequential ? _L : start;
    _device->startTameWalk(s, steps, seed, _sequential);
    pollDevice();

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

void PollardEngine::runWildWalk(const uint256 &start, uint64_t steps) {
    runWildWalk(start, steps, uint256(std::random_device{}()));
}

void PollardEngine::runWildWalk(const uint256 &start, uint64_t steps, const uint256 &seed) {
    if(!_device) {
        return;
    }

    _windowsProcessed = _reconstructionAttempts = _reconstructionSuccess = 0;
    _startTime = std::chrono::steady_clock::now();

    const uint256 &s = _sequential ? _U : start;
    _device->startWildWalk(s, steps, seed, _sequential);
    pollDevice();

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

uint256 PollardEngine::hashWindow(const unsigned int h[5], unsigned int offset,
                                  unsigned int bits) {
    return hashWindowLE(h, offset, bits);
}

