#ifndef CUDA_POLLARD_DEVICE_H
#define CUDA_POLLARD_DEVICE_H

#include <vector>
#include <array>
#include "../KeyFinder/PollardEngine.h"

static const unsigned int MAX_OFFSETS = 32;

class CudaPollardDevice : public PollardDevice {
    PollardEngine &_engine;
    unsigned int _windowBits;
    std::vector<unsigned int> _offsets;
    // RIPEMD160 hashes in little-endian word order
    std::vector<std::array<unsigned int,5>> _targets;
    bool _debug;
    unsigned int _gridDim;
    unsigned int _blockDim;
public:
    CudaPollardDevice(PollardEngine &engine,
                      unsigned int windowBits,
                      const std::vector<unsigned int> &offsets,
                      // ``targets`` must contain RIPEMD160 hashes in
                      // little-endian word order
                      const std::vector<std::array<unsigned int,5>> &targets,
                      bool debug,
                      unsigned int gridDim = 0,
                      unsigned int blockDim = 0);

    void startTameWalk(const secp256k1::uint256 &start, uint64_t steps,
                       const secp256k1::uint256 &seed, bool sequential) override;
    void startWildWalk(const secp256k1::uint256 &start, uint64_t steps,
                       const secp256k1::uint256 &seed, bool sequential) override;
    bool popResult(PollardMatch &out) override { (void)out; return false; }

    /**
     * Enumerate a key range using the ``windowKernel`` and forward any
     * recovered window constraints to the associated ``PollardEngine``.
     *
     * ``targetFragments`` is a two dimensional array where each row contains
     * the expected fragment for every configured offset.  The number of rows
     * must equal ``windowBits`` which also determines the mask applied to the
     * extracted fragments.  Constraints for each discovered offset are appended
     * to ``outConstraints``.
     */
    void scanKeyRange(uint64_t start_k,
                      uint64_t end_k,
                      uint32_t windowBits,
                      const uint32_t *targetFragments,
                      std::vector<PollardEngine::Constraint> &outConstraints);
};

#endif
