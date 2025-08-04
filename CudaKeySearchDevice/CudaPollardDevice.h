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
    bool _debug;
public:
    CudaPollardDevice(PollardEngine &engine,
                      unsigned int windowBits,
                      const std::vector<unsigned int> &offsets,
                      const std::vector<std::array<unsigned int,5>> &targets,
                      bool debug);

    void startTameWalk(const secp256k1::uint256 &start, uint64_t steps,
                       uint64_t seed, bool sequential) override;
    void startWildWalk(const secp256k1::uint256 &start, uint64_t steps,
                       uint64_t seed, bool sequential) override;
    bool popResult(PollardMatch &out) override { (void)out; return false; }
};

#endif
