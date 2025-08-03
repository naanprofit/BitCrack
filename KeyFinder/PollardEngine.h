#ifndef POLLARD_ENGINE_H
#define POLLARD_ENGINE_H

#include <vector>
#include <functional>
#include "secp256k1.h"
#include "KeySearchDevice.h"

class PollardEngine {
public:
    struct Constraint {
        unsigned int bits;              // number of low bits constrained
        secp256k1::uint256 value;       // value modulo 2^bits
    };

    using ResultCallback = std::function<void(KeySearchResult)>;

    explicit PollardEngine(ResultCallback cb);

    // Add a constraint of the form k \equiv value (mod 2^bits)
    void addConstraint(unsigned int bits, const secp256k1::uint256 &value);

    // Attempt to reconstruct the private key from accumulated constraints.
    bool reconstruct(secp256k1::uint256 &out);

    // CPU based tame and wild walks.
    void runTameWalk(const secp256k1::uint256 &start, uint64_t steps);
    void runWildWalk(const secp256k1::ecpoint &start, uint64_t steps);

private:
    std::vector<Constraint> _constraints;
    ResultCallback _callback;

    bool checkPoint(const secp256k1::ecpoint &p);
    void enumerateCandidate(const secp256k1::uint256 &priv, const secp256k1::ecpoint &pub);
};

#endif
