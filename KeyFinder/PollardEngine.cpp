#include "PollardEngine.h"
#include "secp256k1.h"
#include "AddressUtil.h"
#if BUILD_CUDA
#include <cuda_runtime.h>
#include "windowKernel.h"
#include <cstdio>
#define CUDA_CHECK(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
        cuda_cleanup(); \
        return; \
    } \
} while(0)
#endif
#include <algorithm>
#include <cstring>
#include <vector>
#include <random>
#include <chrono>
#include <thread>
#include <sstream>
#include <iomanip>
#include <numeric>
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

uint256 maskBits(unsigned int bits);

// Simple CPU implementation of the device interface used for unit tests and
// as a fallback when no GPU is available.
class CPUPollardDevice : public PollardDevice {
    unsigned int _windowBits;
    SimpleRingBuffer<PollardMatch> _buffer;

    bool checkPoint(const ecpoint &p) {
        uint256 mask = maskBits(_windowBits);
        for(int w = 0; w < 8; ++w) {
            if((p.x.v[w] & mask.v[w]) != 0) {
                return false;
            }
        }
        return true;
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

    uint256 seedCopy = seed;
    std::mt19937_64 rng(seedCopy.toUint64());
    for(uint64_t i = 0; i < steps; ++i) {
        uint256 stepVal;
        do {
            for(int j = 0; j < 4; ++j) {
                uint64_t v = rng();
                stepVal.v[j * 2]     = static_cast<uint32_t>(v & 0xffffffffULL);
                stepVal.v[j * 2 + 1] = static_cast<uint32_t>(v >> 32);
            }
        } while(stepVal.isZero() || stepVal.cmp(N) >= 0);
        k = addModN(k, stepVal);
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

    uint256 seedCopy = seed;
    std::mt19937_64 rng(seedCopy.toUint64());

    uint256 kOffset(0);
    for(uint64_t i = 0; i < steps; ++i) {
        uint256 stepVal;
        do {
            for(int j = 0; j < 4; ++j) {
                uint64_t v = rng();
                stepVal.v[j * 2]     = static_cast<uint32_t>(v & 0xffffffffULL);
                stepVal.v[j * 2 + 1] = static_cast<uint32_t>(v >> 32);
            }
        } while(stepVal.isZero() || stepVal.cmp(N) >= 0);
        kOffset = addModN(kOffset, stepVal);
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

static bool toU128(const uint256 &x, unsigned __int128 &out) {
    if(x.v[4] || x.v[5] || x.v[6] || x.v[7]) return false;
    out = ((unsigned __int128)x.v[3] << 96) |
          ((unsigned __int128)x.v[2] << 64) |
          ((unsigned __int128)x.v[1] << 32) |
          x.v[0];
    return true;
}

static uint256 fromU128(unsigned __int128 x) {
    uint256 r(0);
    r.v[0] = (uint32_t)x;
    r.v[1] = (uint32_t)(x >> 32);
    r.v[2] = (uint32_t)(x >> 64);
    r.v[3] = (uint32_t)(x >> 96);
    return r;
}

static unsigned __int128 gcd_u128(unsigned __int128 a, unsigned __int128 b) {
    while(b != 0) {
        unsigned __int128 t = a % b;
        a = b;
        b = t;
    }
    return a;
}

static unsigned __int128 modInverse128(unsigned __int128 a, unsigned __int128 m) {
    __int128_t t = 0, newt = 1;
    __int128_t r = m, newr = a % m;
    while(newr != 0) {
        __int128_t q = r / newr;
        __int128_t tmp = t - q * newt; t = newt; newt = tmp;
        tmp = r - q * newr; r = newr; newr = tmp;
    }
    if(r != 1) return 0;
    if(t < 0) t += m;
    return (unsigned __int128)t;
}

static bool modulusBits(const uint256 &m, unsigned int &bits) {
    if(m.isZero()) { bits = 256; return true; }
    bool found = false;
    for(int i = 7; i >= 0; --i) {
        uint32_t w = m.v[i];
        if(w) {
            if((w & (w - 1)) != 0 || found) return false;
            bits = i * 32 + __builtin_ctz(w);
            found = true;
        }
    }
    return found;
}

// Combine two congruences using a generic CRT solver.  If both moduli fit
// within 128 bits a full solver with gcd and modular inverses is used.
// Otherwise a power-of-two merge is attempted.
bool combineCRT(const PollardEngine::Constraint &a,
                const PollardEngine::Constraint &b,
                PollardEngine::Constraint &out) {
    unsigned __int128 m1, m2, r1, r2;
    if(toU128(a.modulus, m1) && toU128(b.modulus, m2) &&
       toU128(a.value, r1) && toU128(b.value, r2)) {
        unsigned __int128 g = gcd_u128(m1, m2);
        if(((r1 >= r2) ? (r1 - r2) : (r2 - r1)) % g != 0) {
            return false;
        }
        unsigned __int128 lcm = m1 / g * m2;
        unsigned __int128 m1_g = m1 / g;
        unsigned __int128 m2_g = m2 / g;
        unsigned __int128 inv = modInverse128(m1_g % m2_g, m2_g);
        if(inv == 0) return false;
        unsigned __int128 t = (r2 + m2 - r1) % m2;
        t = (t / g) % m2_g;
        unsigned __int128 x = (r1 + m1 * ((t * inv) % m2_g)) % lcm;
        out.value = fromU128(x);
        out.modulus = fromU128(lcm);
        return true;
    }

    unsigned int bitsA, bitsB;
    if(!modulusBits(a.modulus, bitsA) || !modulusBits(b.modulus, bitsB)) {
        return false;
    }
    unsigned int overlap = std::min(bitsA, bitsB);
    uint256 mask = maskBits(overlap);
    for(int w = 0; w < 8; ++w) {
        if((a.value.v[w] & mask.v[w]) != (b.value.v[w] & mask.v[w])) {
            return false; // inconsistent constraints
        }
    }
    out = (bitsA >= bitsB) ? a : b;
    return true;
}

// Extract ``bits`` bits starting at ``offset`` (LSB order) from a 160-bit
// little-endian hash represented as five 32-bit words.  The result is returned
// as a ``uint256`` with any unused high words cleared.
uint256 hashWindowLE(const unsigned int h[5], unsigned int offset,
                     unsigned int bits) {
    uint256 out(0);
    unsigned int word = offset / 32;
    unsigned int bit  = offset % 32;
    unsigned int words = (bits + 31) / 32;        // number of output words
    for(unsigned int i = 0; i < words && word + i < 5; ++i) {
        uint32_t val = h[word + i] >> bit;
        if(bit && word + i + 1 < 5) {
            val |= h[word + i + 1] << (32 - bit);
        }
        out.v[i] = val;
    }
    unsigned int maskBits = bits % 32;
    if(maskBits) {
        uint32_t mask = (1u << maskBits) - 1u;
        out.v[words - 1] &= mask;
    }
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
            auto want = hashWindowLE(_targets[t].hash.data(), off, _windowBits);
            auto got  = hashWindowLE(m.hash, off, _windowBits);
            if(got == want) {
                unsigned int modBits = off + _windowBits;
                if(modBits > 256) {
                    continue;
                }
                uint256 mask = maskBits(modBits);
                uint256 rem;
                for(int w = 0; w < 8; ++w) {
                    rem.v[w] = m.scalar.v[w] & mask.v[w];
                }
                uint256 mod(0);
                if(modBits < 256) {
                    mod.v[modBits / 32] = (1u << (modBits % 32));
                }
                processWindow(t, off, {mod, rem});
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
                             bool debug,
                             unsigned int gridDim,
                             unsigned int blockDim)
    : _callback(cb), _windowBits(windowBits), _offsets(offsets),
      _batchSize(batchSize), _pollInterval(pollInterval), _L(L), _U(U),
      _sequential(sequential), _debug(debug),
      _gridDim(gridDim), _blockDim(blockDim) {
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

void PollardEngine::addConstraint(size_t target, const uint256 &modulus,
                                  const uint256 &value) {
    if(target >= _targets.size()) {
        return;
    }
    Constraint c{modulus, value};
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
                  bool aInf = a.modulus.isZero();
                  bool bInf = b.modulus.isZero();
                  if(aInf && bInf) return false;
                  if(aInf) return false;
                  if(bInf) return true;
                  return a.modulus.cmp(b.modulus) < 0;
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
    modulus = acc.modulus;
    return true;
}

bool PollardEngine::checkPoint(const ecpoint &p) {
    uint256 mask = maskBits(_windowBits);
    for(int w = 0; w < 8; ++w) {
        if((p.x.v[w] & mask.v[w]) != 0) {
            return false;
        }
    }
    return true;
}

void PollardEngine::processWindow(size_t targetIdx, unsigned int offset,
                                  const Constraint &c) {
    _windowsProcessed++;
    if(targetIdx >= _targets.size()) {
        return;
    }
    if(_targets[targetIdx].seenOffsets.count(offset)) {
        return;
    }

    if(_debug) {
        Logger::log(LogLevel::Debug,
                    "Window target=" + util::format(targetIdx) +
                    " offset=" + util::format(offset) +
                    " modulus=" + secp256k1::uint256(c.modulus).toString(16) +
                    " value=" + secp256k1::uint256(c.value).toString(16));
    }

    unsigned int modBits = offset + _windowBits;
    if(modBits > 256) {
        return;
    }

    addConstraint(targetIdx, c.modulus, c.value);
    _targets[targetIdx].seenOffsets.insert(offset);

    _reconstructionAttempts++;
    uint256 k0;
    uint256 modulus;
    if(reconstruct(targetIdx, k0, modulus)) {
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
    // Compute both compressed and uncompressed hashes for the candidate and
    // emit a result for any matching target.  This ensures targets specified
    // via --pubkey (which may be compressed or uncompressed) are handled.
    unsigned int beComp[5];
    unsigned int beUncomp[5];
    Hash::hashPublicKeyCompressed(pub, beComp);
    Hash::hashPublicKey(pub, beUncomp);

    if(_debug) {
        Logger::log(LogLevel::Debug,
                    "k=" + secp256k1::uint256(priv).toString(16) +
                    " Px=" + secp256k1::uint256(pub.x).toString(16) +
                    " Py=" + secp256k1::uint256(pub.y).toString(16) +
                    " hash=" + hashToString(beComp));
    }

    unsigned int compDigest[5];
    unsigned int uncompDigest[5];
    for(int i = 0; i < 5; ++i) {
        compDigest[i] = util::endian(beComp[4 - i]);
        uncompDigest[i] = util::endian(beUncomp[4 - i]);
    }

    bool compMatch = false;
    bool uncompMatch = false;
    for(const auto &t : _targets) {
        if(!compMatch) {
            compMatch = std::equal(compDigest, compDigest + 5, t.hash.begin());
        }
        if(!uncompMatch) {
            uncompMatch = std::equal(uncompDigest, uncompDigest + 5, t.hash.begin());
        }
        if(compMatch && uncompMatch) {
            break;
        }
    }

    if(!compMatch && !uncompMatch) {
        return; // Candidate does not belong to the target set
    }

    if(compMatch) {
        _reconstructionSuccess++;
        KeySearchResult r;
        r.privateKey = priv;
        r.publicKey = pub;
        r.compressed = true;
        std::memcpy(r.hash, compDigest, sizeof(r.hash));
        r.address = Address::fromPublicKey(pub, true);
        _callback(r);
    }

    if(uncompMatch) {
        _reconstructionSuccess++;
        KeySearchResult r;
        r.privateKey = priv;
        r.publicKey = pub;
        r.compressed = false;
        std::memcpy(r.hash, uncompDigest, sizeof(r.hash));
        r.address = Address::fromPublicKey(pub, false);
        _callback(r);
    }
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

#if BUILD_CUDA
    uint64_t start_k = k.toUint64();
    uint64_t range_len = 0;
    for(uint256 tmp = k; tmp.cmp(U) <= 0; tmp = tmp.add(modulus)) {
        range_len++;
    }

    if(range_len == 0) {
        return;
    }

    uint32_t offsetCount = static_cast<uint32_t>(_offsets.size());
    if(offsetCount == 0) {
        return;
    }
    uint32_t ws = _windowBits;
    uint32_t mask = (ws >= 32) ? 0xffffffffu : ((1u << ws) - 1u);

    std::vector<uint32_t> hostFrags(offsetCount);

    // Allocate GPU buffers.
    uint32_t *dev_offsets = nullptr;
    uint32_t *dev_target_frags = nullptr;
    MatchRecord *dev_out = nullptr;
    uint32_t *dev_count = nullptr;

    auto cuda_cleanup = [&]() {
        if(dev_offsets) cudaFree(dev_offsets);
        if(dev_target_frags) cudaFree(dev_target_frags);
        if(dev_out) cudaFree(dev_out);
        if(dev_count) cudaFree(dev_count);
    };

    CUDA_CHECK(cudaMalloc(&dev_offsets, offsetCount * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&dev_target_frags, offsetCount * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&dev_out, sizeof(MatchRecord) * 1024));
    CUDA_CHECK(cudaMalloc(&dev_count, sizeof(uint32_t)));

    CUDA_CHECK(cudaMemcpy(dev_offsets, _offsets.data(), offsetCount * sizeof(uint32_t), cudaMemcpyHostToDevice));
    for(size_t t = 0; t < _targets.size(); ++t) {
        // Pre-compute the target fragments for this hash.
        for(uint32_t i = 0; i < offsetCount; ++i) {
            auto win = hashWindow(_targets[t].hash.data(), _offsets[i], ws);
            hostFrags[i] = win.v[0] & mask;
        }
        CUDA_CHECK(cudaMemcpy(dev_target_frags, hostFrags.data(), offsetCount * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(dev_count, 0, sizeof(uint32_t)));

        // Launch the GPU kernel to perform the window/fragment matching.
        unsigned int blockSize = _blockDim ? _blockDim : 256u;
        dim3 block(blockSize);
        unsigned int gridSize = _gridDim ? _gridDim :
                               (unsigned int)((range_len + block.x - 1) / block.x);
        dim3 grid(gridSize);
        launchWindowKernel(grid, block, start_k, range_len, ws,
                           dev_offsets, offsetCount, mask,
                           dev_target_frags, dev_out, dev_count);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Retrieve matches.
        uint32_t hitCount = 0;
        CUDA_CHECK(cudaMemcpy(&hitCount, dev_count, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        hitCount = std::min(hitCount, 1024u);

        std::vector<MatchRecord> hostBuf(hitCount);
        if(hitCount != 0) {
            CUDA_CHECK(cudaMemcpy(hostBuf.data(), dev_out,
                       hitCount * sizeof(MatchRecord), cudaMemcpyDeviceToHost));
        }

        // Convert GPU match records into CRT constraints.  The full private key
        // candidates are validated later by ``processWindow`` which performs the
        // CRT merge followed by hash160 verification.
        for(const auto &rec : hostBuf) {
            uint256 priv(rec.k);
            if(priv.cmp(L) < 0 || priv.cmp(U) > 0) {
                continue; // discard matches outside the search interval
            }
            uint32_t mod32 = 1u << (rec.offset + ws);
            uint32_t rem32 = (rec.fragment << rec.offset) & (mod32 - 1u);
            Constraint c{ uint256(mod32), uint256(rem32) };
            processWindow(t, rec.offset, c);
        }
    }

    cuda_cleanup();
    return;
#else
    for(; k.cmp(U) <= 0; k = k.add(modulus)) {
        ecpoint pub = multiplyPoint(k, G());
        enumerateCandidate(k, pub);
    }
#endif
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

    if(_sequential) {
        Logger::log(LogLevel::Info,
                    "Running deterministic sequential walk using GPU kernels");
    }

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

    if(_sequential) {
        Logger::log(LogLevel::Info,
                    "Running deterministic sequential walk using GPU kernels");
    }

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

uint256 PollardEngine::publicHashWindow(const unsigned int h[5], unsigned int offset,
                                        unsigned int bits) {
    return hashWindow(h, offset, bits);
}

