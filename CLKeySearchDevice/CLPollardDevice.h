#ifndef CL_POLLARD_DEVICE_H
#define CL_POLLARD_DEVICE_H

#include <vector>
#include <array>
#include "../KeyFinder/PollardEngine.h"

class CLPollardDevice : public PollardDevice {
    PollardEngine &_engine;
    unsigned int _windowBits;
    std::vector<unsigned int> _offsets;
    // RIPEMD160 hashes in little-endian word order
    std::vector<std::array<unsigned int,5>> _targets;
    bool _debug;
public:
    CLPollardDevice(PollardEngine &engine,
                    unsigned int windowBits,
                    const std::vector<unsigned int> &offsets,
                    // ``targets`` must contain RIPEMD160 hashes in
                    // little-endian word order
                    const std::vector<std::array<unsigned int,5>> &targets,
                    bool debug);

    static secp256k1::uint256 maskBits(unsigned int bits);
    static secp256k1::uint256 hashWindowLE(const unsigned int h[5], unsigned int offset, unsigned int bits);

    void startTameWalk(const secp256k1::uint256 &start, uint64_t steps,
                       const secp256k1::uint256 &seed, bool sequential) override;
    void startWildWalk(const secp256k1::uint256 &start, uint64_t steps,
                       const secp256k1::uint256 &seed, bool sequential) override;
    bool popResult(PollardMatch &out) override { (void)out; return false; }
};

#endif
