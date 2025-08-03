#ifndef POLLARD_ENGINE_H
#define POLLARD_ENGINE_H

#include <vector>
#include <functional>
#include <cstdint>
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
     * @param offsets     Bit offsets describing where in the key each window
     *                    is collected.  The union of these windows is later fed
     *                    into the Chinese Remainder Theorem (CRT) solver.
     */
    PollardEngine(ResultCallback cb,
                  unsigned int windowBits,
                  const std::vector<unsigned int> &offsets);

    // Add a constraint of the form k \equiv value (mod 2^bits)
    void addConstraint(unsigned int bits, const secp256k1::uint256 &value);

    // Attempt to reconstruct the private key from accumulated constraints using
    // a CRT solver capable of combining arbitrarily large power-of-two
    // moduli.
    bool reconstruct(secp256k1::uint256 &out);

    // CPU based tame and wild walks using random steps.
    void runTameWalk(const secp256k1::uint256 &start, uint64_t steps);
    void runWildWalk(const secp256k1::ecpoint &start, uint64_t steps);

private:
    std::vector<Constraint> _constraints;
    ResultCallback _callback;
    unsigned int _windowBits;                 // number of bits per window
    std::vector<unsigned int> _offsets;       // bit offsets of each window

    bool checkPoint(const secp256k1::ecpoint &p);
    void enumerateCandidate(const secp256k1::uint256 &priv,
                            const secp256k1::ecpoint &pub);
};

#endif
