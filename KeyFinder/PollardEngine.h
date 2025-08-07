#ifndef POLLARD_ENGINE_H
#define POLLARD_ENGINE_H

#include <vector>
#include <functional>
#include <cstdint>
#include <cstddef>
#include <array>
#include <set>
#include <memory>
#include <chrono>
#include "secp256k1.h"
#include "KeySearchDevice.h"
#include "PollardTypes.h"
#if BUILD_CUDA
#include "../CudaKeySearchDevice/windowKernel.h"
#endif

class PollardDevice {
public:
    virtual ~PollardDevice() {}
    virtual void startTameWalk(const secp256k1::uint256 &start, uint64_t steps,
                               const secp256k1::uint256 &seed, bool sequential) = 0;
    virtual void startWildWalk(const secp256k1::uint256 &start, uint64_t steps,
                               const secp256k1::uint256 &seed, bool sequential) = 0;
    // CPU implementations may still emit ``PollardMatch`` results which are
    // converted to ``PollardWindow`` objects by the engine.  GPU
    // implementations can enqueue ``PollardWindow`` structures directly.
    virtual bool popResult(PollardMatch &out) = 0;
};

class PollardEngine {
public:
    struct Constraint {
        secp256k1::uint256 modulus;     // modulus (0 represents 2^256)
        secp256k1::uint256 value;       // value modulo ``modulus``
    };

    using ResultCallback = std::function<void(KeySearchResult)>;

    /**
     * Construct a PollardEngine.
     *
     * @param cb          Callback invoked for every candidate key recovered.
     * @param windowBits  Size of each bit window collected from a walk.
     * @param offsets     Bit offsets (within the hash) describing where each
     *                    window is collected.
     * @param targets     RIPEMD160 hashes (five 32-bit words, little-endian)
     *                    that the walk is attempting to recover.  Each
     *                    target maintains its own set of constraints.
     */
    PollardEngine(ResultCallback cb,
                  unsigned int windowBits,
                  const std::vector<unsigned int> &offsets,
                  const std::vector<std::array<unsigned int,5>> &targets,
                  const secp256k1::uint256 &L,
                  const secp256k1::uint256 &U,
                  unsigned int batchSize = 1024,
                  unsigned int pollInterval = 100,
                  bool sequential = false,
                  bool debug = false);

    // Add a constraint of the form k \equiv value (mod ``modulus``) for ``target``
    void addConstraint(size_t target, const secp256k1::uint256 &modulus,
                       const secp256k1::uint256 &value);

    // Attempt to reconstruct the private key for ``target`` from accumulated
    // constraints using a generic CRT solver.
    bool reconstruct(size_t target, secp256k1::uint256 &k0,
                     secp256k1::uint256 &modulus);

    // Consume a window constraint produced by a device. This function
    // accumulates the constraint, attempts key reconstruction and, on success,
    // hashes the candidate to verify it belongs to the supplied target set
    // before invoking the callback.
    void processWindow(size_t targetIdx, unsigned int offset,
                       const Constraint &c);

    // Walk routines consume results produced by the configured device.  The
    // overloads with a seed parameter enable deterministic behaviour for
    // testing and for GPU implementations that require an explicit seed.
    struct Job {
        bool tame;                             // true for tame walk, false for wild
        secp256k1::uint256 start;              // starting scalar for the walk
        uint64_t steps;                        // number of steps to take
        secp256k1::uint256 seed;               // RNG seed for random walks
    };

    // Execute a pollard job described by ``job``.  The helper dispatches to
    // ``runTameWalk`` or ``runWildWalk`` as appropriate.
    void runJob(const Job &job);

    void runTameWalk(const secp256k1::uint256 &start, uint64_t steps);
    void runTameWalk(const secp256k1::uint256 &start, uint64_t steps, const secp256k1::uint256 &seed);
    void runWildWalk(const secp256k1::uint256 &start, uint64_t steps);
    void runWildWalk(const secp256k1::uint256 &start, uint64_t steps, const secp256k1::uint256 &seed);

    // Replace the underlying device used to generate walk results.  By
    // default a CPU implementation is used which enables unit tests to run
    // without a GPU.  Ownership of ``device`` is transferred to the engine.
    void setDevice(std::unique_ptr<PollardDevice> device);

    // Progress helpers
    bool allOffsetsFound() const;
    size_t foundOffsets() const;
    size_t totalOffsets() const;

    // Public wrapper exposing the internal hashWindow helper.  ``h`` must be
    // supplied in little-endian word order.  The returned array contains the
    // extracted window as five 32-bit words with unused high words set to
    // zero.
    static std::array<unsigned int,5> publicHashWindow(const unsigned int h[5], unsigned int offset,
                                                       unsigned int bits);

private:
    struct TargetState {
        std::array<unsigned int,5> hash;      // target RIPEMD160 (little-endian)
        std::vector<Constraint> constraints;  // gathered constraints
        std::set<unsigned int> seenOffsets;   // offsets already collected
    };

    ResultCallback _callback;
    unsigned int _windowBits;                 // number of bits per window
    std::vector<unsigned int> _offsets;       // bit offsets of each window
    std::vector<TargetState> _targets;        // state per target hash

    std::unique_ptr<PollardDevice> _device;   // producer of walk results

    unsigned int _batchSize;                  // windows processed per poll
    unsigned int _pollInterval;               // milliseconds between polls
    secp256k1::uint256 _L;                    // search lower bound
    secp256k1::uint256 _U;                    // search upper bound
    bool _sequential;                         // sequential walk mode
    bool _debug;                              // enable verbose logging


    // Metrics
    uint64_t _windowsProcessed = 0;           // number of windows consumed
    uint64_t _reconstructionAttempts = 0;     // number of CRT solves attempted
    uint64_t _reconstructionSuccess = 0;      // successful reconstructions
    std::chrono::steady_clock::time_point _startTime; // timing for throughput

    bool checkPoint(const secp256k1::ecpoint &p);
    void enumerateCandidate(const secp256k1::uint256 &priv,
                            const secp256k1::ecpoint &pub);
    void enumerateCandidates(const secp256k1::uint256 &k0,
                             const secp256k1::uint256 &modulus,
                             const secp256k1::uint256 &L,
                             const secp256k1::uint256 &U);
    void handleMatch(const PollardMatch &m);
    void pollDevice();

    static std::array<unsigned int,5> hashWindow(const unsigned int h[5], unsigned int offset,
                                                 unsigned int bits);
};

#endif
