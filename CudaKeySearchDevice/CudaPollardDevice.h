#ifndef CUDA_POLLARD_DEVICE_H
#define CUDA_POLLARD_DEVICE_H

#include <vector>
#include <array>
#include "../KeyFinder/PollardEngine.h"

class CudaPollardDevice : public PollardDevice {
    PollardEngine &_engine;
    unsigned int _windowBits;
    std::vector<unsigned int> _offsets;
    std::vector<std::array<unsigned int,5>> _targets;

    static secp256k1::uint256 maskBits(unsigned int bits);
    static uint64_t hashWindowLE(const unsigned int h[5], unsigned int offset, unsigned int bits);
public:
    CudaPollardDevice(PollardEngine &engine,
                      unsigned int windowBits,
                      const std::vector<unsigned int> &offsets,
                      const std::vector<std::array<unsigned int,5>> &targets);

    void startTameWalk(const secp256k1::uint256 &start, uint64_t steps, uint64_t seed) override;
    void startWildWalk(const secp256k1::ecpoint &start, uint64_t steps, uint64_t seed) override;
    bool popResult(PollardMatch &out) override { (void)out; return false; }
};

#endif
