#include "CudaKeySearchDevice.h"
#include "Logger.h"
#include "util.h"
#include "cudabridge.h"
#include "AddressUtil.h"
#include <cstring>
#include <vector>

struct CudaPollardMatch {
    unsigned long long k[4];
    unsigned int hash[5];
};

extern "C" __global__ void pollardRandomWalk(CudaPollardMatch *out,
                                              unsigned int *outCount,
                                              unsigned int maxOut,
                                              unsigned long long seed,
                                              unsigned int steps,
                                              unsigned int windowBits);

void CudaKeySearchDevice::cudaCall(cudaError_t err)
{
    if(err) {
        std::string errStr = cudaGetErrorString(err);

        throw KeySearchException(errStr);
    }
}

CudaKeySearchDevice::CudaKeySearchDevice(int device, int threads, int pointsPerThread, int blocks)
{
    cuda::CudaDeviceInfo info;
    try {
        info = cuda::getDeviceInfo(device);
        _deviceName = info.name;
    } catch(cuda::CudaException ex) {
        throw KeySearchException(ex.msg);
    }

    if(threads <= 0 || threads % 32 != 0) {
        throw KeySearchException("The number of threads must be a multiple of 32");
    }

    if(pointsPerThread <= 0) {
        throw KeySearchException("At least 1 point per thread required");
    }

    // Specifying blocks on the commandline is depcreated but still supported. If there is no value for
    // blocks, devide the threads evenly among the multi-processors
    if(blocks == 0) {
        if(threads % info.mpCount != 0) {
            throw KeySearchException("The number of threads must be a multiple of " + util::format("%d", info.mpCount));
        }

        _threads = threads / info.mpCount;

        _blocks = info.mpCount;

        while(_threads > 512) {
            _threads /= 2;
            _blocks *= 2;
        }
    } else {
        _threads = threads;
        _blocks = blocks;
    }

    _iterations = 0;

    _device = device;

    _pointsPerThread = pointsPerThread;
}

void CudaKeySearchDevice::init(const secp256k1::uint256 &start, int compression, const secp256k1::uint256 &stride)
{
    if(start.cmp(secp256k1::N) >= 0) {
        throw KeySearchException("Starting key is out of range");
    }

    _startExponent = start;

    _compression = compression;

    _stride = stride;

    cudaCall(cudaSetDevice(_device));

    // Block on kernel calls
    cudaCall(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));

    // Use a larger portion of shared memory for L1 cache
    cudaCall(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

    generateStartingPoints();

    cudaCall(allocateChainBuf(_threads * _blocks * _pointsPerThread));

    // Set the incrementor
    secp256k1::ecpoint g = secp256k1::G();
    secp256k1::ecpoint p = secp256k1::multiplyPoint(secp256k1::uint256((uint64_t)_threads * _blocks * _pointsPerThread) * _stride, g);

    cudaCall(_resultList.init(sizeof(CudaDeviceResult), 16));

    cudaCall(setIncrementorPoint(p.x, p.y));
}


void CudaKeySearchDevice::generateStartingPoints()
{
    uint64_t totalPoints = (uint64_t)_pointsPerThread * _threads * _blocks;
    uint64_t totalMemory = totalPoints * 40;

    std::vector<secp256k1::uint256> exponents;

    Logger::log(LogLevel::Info, "Generating " + util::formatThousands(totalPoints) + " starting points (" + util::format("%.1f", (double)totalMemory / (double)(1024 * 1024)) + "MB)");

    // Generate key pairs for k, k+1, k+2 ... k + <total points in parallel - 1>
    secp256k1::uint256 privKey = _startExponent;

    exponents.push_back(privKey);

    for(uint64_t i = 1; i < totalPoints; i++) {
        privKey = privKey.add(_stride);
        exponents.push_back(privKey);
    }

    cudaCall(_deviceKeys.init(_blocks, _threads, _pointsPerThread, exponents));

    // Show progress in 10% increments
    double pct = 10.0;
    for(int i = 1; i <= 256; i++) {
        cudaCall(_deviceKeys.doStep());

        if(((double)i / 256.0) * 100.0 >= pct) {
            Logger::log(LogLevel::Info, util::format("%.1f%%", pct));
            pct += 10.0;
        }
    }

    Logger::log(LogLevel::Info, "Done");

    _deviceKeys.clearPrivateKeys();
}


void CudaKeySearchDevice::setTargets(const std::set<KeySearchTarget> &targets)
{
    _targets.clear();
    
    for(std::set<KeySearchTarget>::iterator i = targets.begin(); i != targets.end(); ++i) {
        hash160 h(i->value);
        _targets.push_back(h);
    }

    cudaCall(_targetLookup.setTargets(_targets));
}

void CudaKeySearchDevice::doStep()
{
    uint64_t numKeys = (uint64_t)_blocks * _threads * _pointsPerThread;

    try {
        if(_iterations < 2 && _startExponent.cmp(numKeys) <= 0) {
            callKeyFinderKernel(_blocks, _threads, _pointsPerThread, true, _compression);
        } else {
            callKeyFinderKernel(_blocks, _threads, _pointsPerThread, false, _compression);
        }
    } catch(cuda::CudaException ex) {
        throw KeySearchException(ex.msg);
    }

    getResultsInternal();

    _iterations++;
}

uint64_t CudaKeySearchDevice::keysPerStep()
{
    return (uint64_t)_blocks * _threads * _pointsPerThread;
}

std::string CudaKeySearchDevice::getDeviceName()
{
    return _deviceName;
}

void CudaKeySearchDevice::getMemoryInfo(uint64_t &freeMem, uint64_t &totalMem)
{
    cudaCall(cudaMemGetInfo(&freeMem, &totalMem));
}

void CudaKeySearchDevice::removeTargetFromList(const unsigned int hash[5])
{
    size_t count = _targets.size();

    while(count) {
        if(memcmp(hash, _targets[count - 1].h, 20) == 0) {
            _targets.erase(_targets.begin() + count - 1);
            return;
        }
        count--;
    }
}

bool CudaKeySearchDevice::isTargetInList(const unsigned int hash[5])
{
    size_t count = _targets.size();

    while(count) {
        if(memcmp(hash, _targets[count - 1].h, 20) == 0) {
            return true;
        }
        count--;
    }

    return false;
}

uint32_t CudaKeySearchDevice::getPrivateKeyOffset(int thread, int block, int idx)
{
    // Total number of threads
    int totalThreads = _blocks * _threads;

    int base = idx * totalThreads;

    // Global ID of the current thread
    int threadId = block * _threads + thread;

    return base + threadId;
}

void CudaKeySearchDevice::getResultsInternal()
{
    int count = _resultList.size();
    int actualCount = 0;
    if(count == 0) {
        return;
    }

    unsigned char *ptr = new unsigned char[count * sizeof(CudaDeviceResult)];

    _resultList.read(ptr, count);

    for(int i = 0; i < count; i++) {
        struct CudaDeviceResult *rPtr = &((struct CudaDeviceResult *)ptr)[i];

        // might be false-positive
        if(!isTargetInList(rPtr->digest)) {
            continue;
        }
        actualCount++;

        KeySearchResult minerResult;

        // Calculate the private key based on the number of iterations and the current thread
        secp256k1::uint256 offset = (secp256k1::uint256((uint64_t)_blocks * _threads * _pointsPerThread * _iterations) + secp256k1::uint256(getPrivateKeyOffset(rPtr->thread, rPtr->block, rPtr->idx))) * _stride;
        secp256k1::uint256 privateKey = secp256k1::addModN(_startExponent, offset);

        minerResult.privateKey = privateKey;
        minerResult.compressed = rPtr->compressed;

        memcpy(minerResult.hash, rPtr->digest, 20);

        minerResult.publicKey = secp256k1::ecpoint(secp256k1::uint256(rPtr->x, secp256k1::uint256::BigEndian), secp256k1::uint256(rPtr->y, secp256k1::uint256::BigEndian));

        removeTargetFromList(rPtr->digest);

        _results.push_back(minerResult);
    }

    delete[] ptr;

    _resultList.clear();

    // Reload the bloom filters
    if(actualCount) {
        cudaCall(_targetLookup.setTargets(_targets));
    }
}

// Verify a private key produces the public key and hash
bool CudaKeySearchDevice::verifyKey(const secp256k1::uint256 &privateKey, const secp256k1::ecpoint &publicKey, const unsigned int hash[5], bool compressed)
{
    secp256k1::ecpoint g = secp256k1::G();

    secp256k1::ecpoint p = secp256k1::multiplyPoint(privateKey, g);

    if(!(p == publicKey)) {
        return false;
    }

    unsigned int xWords[8];
    unsigned int yWords[8];

    p.x.exportWords(xWords, 8, secp256k1::uint256::BigEndian);
    p.y.exportWords(yWords, 8, secp256k1::uint256::BigEndian);

    unsigned int digest[5];
    if(compressed) {
        Hash::hashPublicKeyCompressed(xWords, yWords, digest);
    } else {
        Hash::hashPublicKey(xWords, yWords, digest);
    }

    for(int i = 0; i < 5; i++) {
        if(digest[i] != hash[i]) {
            return false;
        }
    }

    return true;
}

size_t CudaKeySearchDevice::getResults(std::vector<KeySearchResult> &resultsOut)
{
    for(int i = 0; i < _results.size(); i++) {
        resultsOut.push_back(_results[i]);
    }
    _results.clear();

    return resultsOut.size();
}

secp256k1::uint256 CudaKeySearchDevice::getNextKey()
{
    uint64_t totalPoints = (uint64_t)_pointsPerThread * _threads * _blocks;

    return _startExponent + secp256k1::uint256(totalPoints) * _iterations * _stride;
}

void CudaKeySearchDevice::runPollard(const secp256k1::uint256 &start, uint64_t steps, uint64_t seed)
{
    (void)start;
    _pollardBuffer.clear();

    CudaPollardMatch *d_out = nullptr;
    unsigned int *d_count = nullptr;
    cudaStream_t stream;
    cudaCall(cudaStreamCreate(&stream));
    cudaCall(cudaMalloc(&d_out, sizeof(CudaPollardMatch) * steps));
    cudaCall(cudaMalloc(&d_count, sizeof(unsigned int)));

    pollardRandomWalk<<<1, 1, 0, stream>>>(d_out, d_count,
                                           static_cast<unsigned int>(steps),
                                           seed,
                                           static_cast<unsigned int>(steps),
                                           8u);

    unsigned int h_count = 0;
    std::vector<CudaPollardMatch> h_out(steps);
    cudaCall(cudaMemcpyAsync(&h_count, d_count, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
    cudaCall(cudaMemcpyAsync(h_out.data(), d_out, sizeof(CudaPollardMatch) * steps, cudaMemcpyDeviceToHost, stream));
    cudaCall(cudaStreamSynchronize(stream));

    for(unsigned int i = 0; i < h_count; ++i) {
        PollardMatch m;
        for(int j = 0; j < 4; ++j) {
            m.scalar.v[j * 2]     = static_cast<unsigned int>(h_out[i].k[j] & 0xFFFFFFFFULL);
            m.scalar.v[j * 2 + 1] = static_cast<unsigned int>(h_out[i].k[j] >> 32);
        }
        std::memcpy(m.hash, h_out[i].hash, sizeof(m.hash));
        _pollardBuffer.push(m);
    }

    cudaCall(cudaFree(d_out));
    cudaCall(cudaFree(d_count));
    cudaCall(cudaStreamDestroy(stream));
}

bool CudaKeySearchDevice::getPollardResult(PollardMatch &out)
{
    return _pollardBuffer.pop(out);
}