#ifndef POLLARD_ENGINE_H
#define POLLARD_ENGINE_H

#include <vector>
#include <functional>
#include <cstdint>
#include <array>
#include <set>
#include "secp256k1.h"
#include "KeySearchDevice.h"

class PollardEngine {
public:
    struct Constraint {
        unsigned int bits;              // number of low bits constrained
        secp256k1::uint256 value;       // value modulo 2^bits
    };

    using ResultCallback = std::function<void(KeySearchResult)>;

    /**
     * Construct a PollardEngine.
     *
     * @param cb          Callback invoked for every candidate key recovered.
     * @param windowBits  Size of each bit window collected from a walk.
     * @param offsets     Bit offsets (within the hash) describing where each
     *                    window is collected.
     * @param targets     RIPEMD160 hashes that the walk is attempting to
     *                    recover.  Each target maintains its own set of
     *                    constraints.
     */
    PollardEngine(ResultCallback cb,
                  unsigned int windowBits,
                  const std::vector<unsigned int> &offsets,
                  const std::vector<std::array<unsigned int,5>> &targets);

    // Add a constraint of the form k \equiv value (mod 2^bits) for ``target``
    void addConstraint(size_t target, unsigned int bits,
                       const secp256k1::uint256 &value);

    // Attempt to reconstruct the private key for ``target`` from accumulated
    // constraints using a CRT solver capable of combining arbitrarily large
    // power-of-two moduli.
    bool reconstruct(size_t target, secp256k1::uint256 &out);

    // CPU based tame and wild walks using random steps.  The overloads with a
    // seed parameter enable deterministic behaviour for testing.
    void runTameWalk(const secp256k1::uint256 &start, uint64_t steps);
    void runTameWalk(const secp256k1::uint256 &start, uint64_t steps, uint64_t seed);
    void runWildWalk(const secp256k1::ecpoint &start, uint64_t steps);
    void runWildWalk(const secp256k1::ecpoint &start, uint64_t steps, uint64_t seed);

private:
    struct TargetState {
        std::array<unsigned int,5> hash;      // target RIPEMD160
        std::vector<Constraint> constraints;  // gathered constraints
        std::set<unsigned int> seenOffsets;   // offsets already collected
    };

    ResultCallback _callback;
    unsigned int _windowBits;                 // number of bits per window
    std::vector<unsigned int> _offsets;       // bit offsets of each window
    std::vector<TargetState> _targets;        // state per target hash

    bool checkPoint(const secp256k1::ecpoint &p);
    void enumerateCandidate(const secp256k1::uint256 &priv,
                            const secp256k1::ecpoint &pub);

    static uint64_t hashWindow(const unsigned int h[5], unsigned int offset,
                               unsigned int bits);
};

#endif
