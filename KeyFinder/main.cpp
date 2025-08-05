#include <stdio.h>
#include <inttypes.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <array>
#include <cctype>
#include <algorithm>
#include <map>
#include <iomanip>

#include "KeyFinder.h"
#include "AddressUtil.h"
#include "util.h"
#include "secp256k1.h"
#include "CmdParse.h"
#include "Logger.h"
#include "ConfigFile.h"

#if defined(BUILD_CUDA) || defined(BUILD_OPENCL)
#include "DeviceManager.h"
#endif
#include "PollardEngine.h"

#ifdef BUILD_CUDA
#include "CudaKeySearchDevice.h"
#include "CudaPollardDevice.h"
#endif

#ifdef BUILD_OPENCL
#include "CLKeySearchDevice.h"
#include "CLPollardDevice.h"
#endif

typedef struct {
    // startKey is the first key. We store it so that if the --continue
    // option is used, the correct progress is displayed. startKey and
    // nextKey are only equal at the very beginning. nextKey gets saved
    // in the checkpoint file.
    secp256k1::uint256 startKey = 1;
    secp256k1::uint256 nextKey = 1;

    // The last key to be checked
    secp256k1::uint256 endKey = secp256k1::N - 1;

    uint64_t statusInterval = 1800;
    uint64_t checkpointInterval = 60000;

    unsigned int threads = 0;
    unsigned int blocks = 0;
    unsigned int pointsPerThread = 0;
    unsigned int gridDim = 0;
    unsigned int blockDim = 0;
    
    int compression = PointCompressionType::COMPRESSED;

    std::vector<std::string> targets;
    std::vector<std::array<unsigned int,5>> hash160Targets;
    std::map<std::array<unsigned int,5>, std::string> targetTypes;

    std::string targetsFile = "";

    std::string checkpointFile = "";

    int device = 0;

    std::string resultsFile = "";

    uint64_t totalkeys = 0;
    unsigned int elapsed = 0;
    secp256k1::uint256 stride = 1;

    bool follow = false;

    bool pollard = false;
    std::vector<unsigned int> offsets; // bit offsets for CRT windows
    uint32_t windowSize = 0;
    uint32_t tames = 0;
    uint32_t wilds = 0;
    uint32_t pollBatch = 1024;
    uint32_t pollInterval = 100;
    bool full = false;
    bool deterministic = false;
    bool debug = false;
}RunConfig;

static RunConfig _config;

static bool _resultFound = false;

#if defined(BUILD_CUDA) || defined(BUILD_OPENCL)
std::vector<DeviceManager::DeviceInfo> _devices;
#endif

void writeCheckpoint(secp256k1::uint256 nextKey);

static uint64_t _lastUpdate = 0;
static uint64_t _runningTime = 0;
static uint64_t _startTime = 0;

/**
* Callback to display the private key
*/
void resultCallback(KeySearchResult info)
{
        _resultFound = true;

        std::string outFile = _config.resultsFile.length() != 0 ? _config.resultsFile : "found.txt";

        unsigned int be[5];
        if(info.compressed) {
                Hash::hashPublicKeyCompressed(info.publicKey, be);
        } else {
                Hash::hashPublicKey(info.publicKey, be);
        }

        std::array<unsigned int,5> h;
        for(int i = 0; i < 5; ++i) {
                h[i] = util::endian(be[4 - i]);
        }

        std::string source = "unknown";
        auto it = _config.targetTypes.find(h);
        if(it != _config.targetTypes.end()) {
                source = it->second;
        }

        Logger::log(LogLevel::Info, "Found key for address '" + info.address + "' (source: " + source + "). Written to '" + outFile + "'");

        std::string s = info.address + " " + info.privateKey.toString(16) + " " + info.publicKey.toString(info.compressed);
        util::appendToFile(outFile, s);

        if(_config.resultsFile.length() == 0) {
                std::string logStr = "Address:     " + info.address + "\n";
                logStr += "Private key: " + info.privateKey.toString(16) + "\n";
                logStr += "Compressed:  ";

                if(info.compressed) {
                        logStr += "yes\n";
                } else {
                        logStr += "no\n";
                }

                logStr += "Public key:  \n";

                if(info.compressed) {
                        logStr += info.publicKey.toString(true) + "\n";
                } else {
                        logStr += info.publicKey.x.toString(16) + "\n";
                        logStr += info.publicKey.y.toString(16) + "\n";
                }

                Logger::log(LogLevel::Info, logStr);
        }
}

/**
Callback to display progress
*/
void statusCallback(KeySearchStatus info)
{
	std::string speedStr;

	if(info.speed < 0.01) {
		speedStr = "< 0.01 MKey/s";
	} else {
		speedStr = util::format("%.2f", info.speed) + " MKey/s";
	}

	std::string totalStr = "(" + util::formatThousands(_config.totalkeys + info.total) + " total)";

	std::string timeStr = "[" + util::formatSeconds((unsigned int)((_config.elapsed + info.totalTime) / 1000)) + "]";

	std::string usedMemStr = util::format((info.deviceMemory - info.freeMemory) /(1024 * 1024));

	std::string totalMemStr = util::format(info.deviceMemory / (1024 * 1024));

    std::string targetStr = util::format(info.targets) + " target" + (info.targets > 1 ? "s" : "");


	// Fit device name in 16 characters, pad with spaces if less
	std::string devName = info.deviceName.substr(0, 16);
	devName += std::string(16 - devName.length(), ' ');

    const char *formatStr = NULL;

    if(_config.follow) {
        formatStr = "%s %s/%sMB | %s %s %s %s\n";
    } else {
        formatStr = "\r%s %s / %sMB | %s %s %s %s";
    }

	printf(formatStr, devName.c_str(), usedMemStr.c_str(), totalMemStr.c_str(), targetStr.c_str(), speedStr.c_str(), totalStr.c_str(), timeStr.c_str());

    if(_config.checkpointFile.length() > 0) {
        uint64_t t = util::getSystemTime();
        if(t - _lastUpdate >= _config.checkpointInterval) {
            Logger::log(LogLevel::Info, "Checkpoint");
            writeCheckpoint(info.nextKey);
            _lastUpdate = t;
        }
    }
}

/**
 * Parses the start:end key pair. Possible values are:
 start
 start:end
 start:+offset
 :end
 :+offset
 */
bool parseKeyspace(const std::string &s, secp256k1::uint256 &start, secp256k1::uint256 &end)
{
    size_t pos = s.find(':');

    if(pos == std::string::npos) {
        start = secp256k1::uint256(s);
        end = secp256k1::N - 1;
    } else {
        std::string left = s.substr(0, pos);

        if(left.length() == 0) {
            start = secp256k1::uint256(1);
        } else {
            start = secp256k1::uint256(left);
        }

        std::string right = s.substr(pos + 1);

        if(right[0] == '+') {
            end = start + secp256k1::uint256(right.substr(1));
        } else {
            end = secp256k1::uint256(right);
        }
    }

    return true;
}

void usage()
{
    printf("BitCrack OPTIONS [TARGETS]\n");
    printf("Where TARGETS is one or more addresses or specify targets with --hash160 or --pubkey\n\n");
	
    printf("--help                  Display this message\n");
    printf("-c, --compressed        Use compressed points\n");
    printf("-u, --uncompressed      Use Uncompressed points\n");
    printf("--compression  MODE     Specify compression where MODE is\n");
    printf("                          COMPRESSED or UNCOMPRESSED or BOTH\n");
    printf("-d, --device ID         Use device ID\n");
    printf("-b, --blocks N          N blocks\n");
    printf("-t, --threads N         N threads per block\n");
    printf("-p, --points N          N points per thread\n");
    printf("-i, --in FILE           Read addresses from FILE, one per line\n");
    printf("-o, --out FILE          Write keys to FILE (default: found.txt)\n");
    printf("-f, --follow            Follow text output\n");
    printf("--list-devices          List available devices\n");
    printf("--keyspace KEYSPACE     Specify the keyspace:\n");
    printf("                          START:END\n");
    printf("                          START:+COUNT\n");
    printf("                          START\n");
    printf("                          :END\n"); 
    printf("                          :+COUNT\n");
    printf("                        Where START, END, COUNT are in hex format\n");
    printf("--stride N              Increment by N keys at a time\n");
    printf("--L HEX                Lower bound for Pollard search (hex)\n");
    printf("--U HEX                Upper bound for Pollard search (hex)\n");
    printf("--share M/N             Divide the keyspace into N equal shares, process the Mth share\n");
    printf("--continue FILE         Save/load progress from FILE\n");
    printf("--hash160 HEX          Add a target specified as a 40-hex-character RIPEMD160 hash\n");
    printf("                        Hashes must be big-endian; internally the digest is stored\n");
    printf("                        as little-endian words. Provide the standard big-endian\n");
    printf("                        representation when using this option.\n");
    printf("--pubkey HEX           Add a target specified as a 33- or 65-byte public key in hex\n");
    printf("--pollard              Enable CPU-only Pollard Rho/CRT mode\n");
    printf("--offsets LIST         Comma-separated bit offsets for CRT windows (required)\n");
    printf("--window-size N        Bits per window (default 8)\n");
    printf("--tames N              Tame walk steps (0 disables)\n");
    printf("--wilds N              Wild walk steps (0 disables)\n");
    printf("--poll-batch N        Windows processed per poll (default 1024)\n");
    printf("--poll-interval MS    Polling interval in milliseconds (default 100)\n");
    printf("--grid-dim N          Override CUDA grid dimension (default auto)\n");
    printf("--block-dim N         Override CUDA block dimension (default auto)\n");
    printf("--full                 Process entire keyspace\n");
    printf("--deterministic       Use sequential deterministic walks (still uses GPU kernels)\n");
    printf("--debug               Enable verbose Pollard debugging\n");
    printf("\nAt least one target must be specified via an address, --hash160, or --pubkey.\n");
}


#if defined(BUILD_CUDA) || defined(BUILD_OPENCL)
/**
 Finds default parameters depending on the device
 */
typedef struct {
        int threads;
        int blocks;
        int pointsPerThread;
}DeviceParameters;

DeviceParameters getDefaultParameters(const DeviceManager::DeviceInfo &device)
{
        DeviceParameters p;
        p.threads = 256;
    p.blocks = 32;
        p.pointsPerThread = 32;

        return p;
}

static KeySearchDevice *getDeviceContext(DeviceManager::DeviceInfo &device, int blocks, int threads, int pointsPerThread)
{
#ifdef BUILD_CUDA
    if(device.type == DeviceManager::DeviceType::CUDA) {
        return new CudaKeySearchDevice((int)device.physicalId, threads, pointsPerThread, blocks);
    }
#endif

#ifdef BUILD_OPENCL
    if(device.type == DeviceManager::DeviceType::OpenCL) {
        return new CLKeySearchDevice(device.physicalId, threads, pointsPerThread, blocks);
    }
#endif

    return NULL;
}

static void printDeviceList(const std::vector<DeviceManager::DeviceInfo> &devices)
{
    for(int i = 0; i < devices.size(); i++) {
        printf("ID:     %d\n", devices[i].id);
        printf("Name:   %s\n", devices[i].name.c_str());
        printf("Memory: %" PRIu64 "MB\n", devices[i].memory / ((uint64_t)1024 * 1024));
        printf("Compute units: %d\n", devices[i].computeUnits);
        printf("\n");
    }
}
#endif

bool readAddressesFromFile(const std::string &fileName, std::vector<std::string> &lines)
{
    if(fileName == "-") {
        return util::readLinesFromStream(std::cin, lines);
    } else {
        return util::readLinesFromStream(fileName, lines);
    }
}

int parseCompressionString(const std::string &s)
{
    std::string comp = util::toLower(s);

    if(comp == "both") {
        return PointCompressionType::BOTH;
    }

    if(comp == "compressed") {
        return PointCompressionType::COMPRESSED;
    }

    if(comp == "uncompressed") {
        return PointCompressionType::UNCOMPRESSED;
    }

    throw std::string("Invalid compression format: '" + s + "'");
}

static std::string getCompressionString(int mode)
{
    switch(mode) {
    case PointCompressionType::BOTH:
        return "both";
    case PointCompressionType::UNCOMPRESSED:
        return "uncompressed";
    case PointCompressionType::COMPRESSED:
        return "compressed";
    }

    throw std::string("Invalid compression setting '" + util::format(mode) + "'");
}

void writeCheckpoint(secp256k1::uint256 nextKey)
{
    std::ofstream tmp(_config.checkpointFile, std::ios::out);

    tmp << "start=" << _config.startKey.toString() << std::endl;
    tmp << "next=" << nextKey.toString() << std::endl;
    tmp << "end=" << _config.endKey.toString() << std::endl;
    tmp << "blocks=" << _config.blocks << std::endl;
    tmp << "threads=" << _config.threads << std::endl;
    tmp << "points=" << _config.pointsPerThread << std::endl;
    tmp << "griddim=" << _config.gridDim << std::endl;
    tmp << "blockdim=" << _config.blockDim << std::endl;
    tmp << "compression=" << getCompressionString(_config.compression) << std::endl;
    tmp << "device=" << _config.device << std::endl;
    tmp << "elapsed=" << (_config.elapsed + util::getSystemTime() - _startTime) << std::endl;
    tmp << "stride=" << _config.stride.toString();
    tmp.close();
}

void readCheckpointFile()
{
    if(_config.checkpointFile.length() == 0) {
        return;
    }

    ConfigFileReader reader(_config.checkpointFile);

    if(!reader.exists()) {
        return;
    }

    Logger::log(LogLevel::Info, "Loading ' " + _config.checkpointFile + "'");

    std::map<std::string, ConfigFileEntry> entries = reader.read();

    _config.startKey = secp256k1::uint256(entries["start"].value);
    _config.nextKey = secp256k1::uint256(entries["next"].value);
    _config.endKey = secp256k1::uint256(entries["end"].value);

    if(_config.threads == 0 && entries.find("threads") != entries.end()) {
        _config.threads = util::parseUInt32(entries["threads"].value);
    }
    if(_config.blocks == 0 && entries.find("blocks") != entries.end()) {
        _config.blocks = util::parseUInt32(entries["blocks"].value);
    }
    if(_config.pointsPerThread == 0 && entries.find("points") != entries.end()) {
        _config.pointsPerThread = util::parseUInt32(entries["points"].value);
    }
    if(_config.gridDim == 0 && entries.find("griddim") != entries.end()) {
        _config.gridDim = util::parseUInt32(entries["griddim"].value);
    }
    if(_config.blockDim == 0 && entries.find("blockdim") != entries.end()) {
        _config.blockDim = util::parseUInt32(entries["blockdim"].value);
    }
    if(entries.find("compression") != entries.end()) {
        _config.compression = parseCompressionString(entries["compression"].value);
    }
    if(entries.find("elapsed") != entries.end()) {
        _config.elapsed = util::parseUInt32(entries["elapsed"].value);
    }
    if(entries.find("stride") != entries.end()) {
        _config.stride = util::parseUInt64(entries["stride"].value);
    }

    _config.totalkeys = (_config.nextKey - _config.startKey).toUint64();
}
#if defined(BUILD_CUDA) || defined(BUILD_OPENCL)
int runBruteForce()
{
    if(_config.device < 0 || _config.device >= _devices.size()) {
        Logger::log(LogLevel::Error, "device " + util::format(_config.device) + " does not exist");
        return 1;
    }

    Logger::log(LogLevel::Info, "Compression: " + getCompressionString(_config.compression));
    Logger::log(LogLevel::Info, "Starting at: " + _config.nextKey.toString());
    Logger::log(LogLevel::Info, "Ending at:   " + _config.endKey.toString());
    Logger::log(LogLevel::Info, "Counting by: " + _config.stride.toString());

    try {

        _lastUpdate = util::getSystemTime();
        _startTime = util::getSystemTime();

        // Use default parameters if they have not been set
        DeviceParameters params = getDefaultParameters(_devices[_config.device]);

        if(_config.blocks == 0) {
            _config.blocks = params.blocks;
        }

        if(_config.threads == 0) {
            _config.threads = params.threads;
        }

        if(_config.pointsPerThread == 0) {
            _config.pointsPerThread = params.pointsPerThread;
        }

        // Get device context
        KeySearchDevice *d = getDeviceContext(_devices[_config.device], _config.blocks, _config.threads, _config.pointsPerThread);

        KeyFinder f(_config.nextKey, _config.endKey, _config.compression, d, _config.stride);

        f.setResultCallback(resultCallback);
        f.setStatusInterval(_config.statusInterval);
        f.setStatusCallback(statusCallback);

        f.init();

        if(!_config.targetsFile.empty()) {
            f.setTargets(_config.targetsFile);
            std::ifstream inFile(_config.targetsFile.c_str());
            if(inFile.is_open()) {
                std::string line;
                while(std::getline(inFile, line)) {
                    util::removeNewline(line);
                    line = util::trim(line);
                    if(line.length() > 0) {
                        unsigned int h[5];
                        Base58::toHash160(line, h);
                        std::array<unsigned int,5> arr;
                        for(int j=0;j<5;j++) arr[j]=h[j];
                        _config.targetTypes[arr] = "address";
                    }
                }
            }
        } else if(!_config.targets.empty()) {
            f.setTargets(_config.targets);
        }

        for(const auto &h : _config.hash160Targets) {
            f.addTarget(h.data());
        }

        f.run();

        delete d;
    } catch(KeySearchException ex) {
        Logger::log(LogLevel::Info, "Error: " + ex.msg);
        return 1;
    }

    return 0;
}
#endif

int runPollard()
{
    Logger::log(LogLevel::Info, "Pollard Rho search selected");

    unsigned int window = _config.windowSize ? _config.windowSize : 8;
    std::vector<unsigned int> offsets = _config.offsets;
    uint64_t tameSteps = _config.tames;
    uint64_t wildSteps = _config.wilds;

    std::vector<std::array<unsigned int,5>> targetHashes;

    if(!_config.targetsFile.empty()) {
        std::ifstream inFile(_config.targetsFile.c_str());
        if(!inFile.is_open()) {
            Logger::log(LogLevel::Error, "Unable to open '" + _config.targetsFile + "'");
            return 1;
        }
        std::string line;
        while(std::getline(inFile, line)) {
            util::removeNewline(line);
            line = util::trim(line);
            if(line.length() > 0) {
                unsigned int h[5];
                Base58::toHash160(line, h);
                std::array<unsigned int,5> arr;
                for(int i=0;i<5;i++) {
                    arr[i] = h[i];
                }
                targetHashes.push_back(arr);
                _config.targetTypes[arr] = "address";
            }
        }
    } else {
        for(size_t i=0;i<_config.targets.size();i++) {
            unsigned int h[5];
            Base58::toHash160(_config.targets[i], h);
            std::array<unsigned int,5> arr;
            for(int j=0;j<5;j++) {
                arr[j] = h[j];
            }
            targetHashes.push_back(arr);
            _config.targetTypes[arr] = "address";
        }
    }

    for(const auto &h : _config.hash160Targets) {
        targetHashes.push_back(h);
    }

    if(!offsets.empty()) {
        std::string s;
        for(size_t i = 0; i < offsets.size(); i++) {
            if(i != 0) {
                s += ",";
            }
            s += util::format(offsets[i]);
        }
        Logger::log(LogLevel::Info, "Offsets: " + s);
    }
    Logger::log(LogLevel::Info, "Window size: " + util::format(window));
    Logger::log(LogLevel::Info, "Tame walk steps: " + util::format(tameSteps));
    Logger::log(LogLevel::Info, "Wild walk steps: " + util::format(wildSteps));
    Logger::log(LogLevel::Info, "Poll batch: " + util::format(_config.pollBatch));
    Logger::log(LogLevel::Info, "Poll interval: " + util::format(_config.pollInterval) + " ms");

    for(size_t t = 0; t < targetHashes.size(); ++t) {
        for(unsigned int off : offsets) {
            unsigned int bits = off + window;
            auto remWords = PollardEngine::publicHashWindow(targetHashes[t].data(), off, window);
            secp256k1::uint256 rem;
            for(int i = 0; i < 5; ++i) rem.v[i] = remWords[i];
            for(int i = 5; i < 8; ++i) rem.v[i] = 0u;
            std::string modStr;
            if(bits >= 256) {
                modStr = "2^256";
            } else {
                secp256k1::uint256 modVal = secp256k1::uint256(2).pow(bits);
                modStr = modVal.toString(16);
            }
            Logger::log(LogLevel::Info,
                        "Target " + util::format(t) +
                        " offset=" + util::format(off) +
                        " mod=" + modStr +
                        " remainder=" + rem.toString(16));
        }
    }

    _resultFound = false;

    secp256k1::uint256 segmentStart = _config.nextKey;

    try {
        while(segmentStart.cmp(_config.endKey) <= 0) {
            PollardEngine engine(resultCallback, window, offsets, targetHashes,
                                 segmentStart, _config.endKey,
                                 _config.pollBatch, _config.pollInterval,
                                 _config.deterministic,
                                 _config.debug);

#ifdef BUILD_CUDA
            if(_devices[_config.device].type == DeviceManager::DeviceType::CUDA) {
                engine.setDevice(std::unique_ptr<CudaPollardDevice>(
                    new CudaPollardDevice(engine, window, offsets, targetHashes,
                                           _config.debug, _config.gridDim,
                                           _config.blockDim)));
            }
#endif
#ifdef BUILD_OPENCL
            if(_devices[_config.device].type == DeviceManager::DeviceType::OpenCL) {
                engine.setDevice(std::unique_ptr<CLPollardDevice>(new CLPollardDevice(engine, window, offsets, targetHashes, _config.debug)));
            }
#endif

            if(tameSteps > 0) {
                engine.runTameWalk(segmentStart, tameSteps);
            }

            if(wildSteps > 0) {
                engine.runWildWalk(_config.endKey, wildSteps);
            }

            if(_resultFound) {
                break;
            }

            segmentStart = segmentStart.add(_config.stride);
            _config.nextKey = segmentStart;

            if(!_config.full) {
                break;
            }
        }
    } catch(const std::exception &ex) {
        Logger::log(LogLevel::Error, std::string("Pollard error: ") + ex.what());
        return 1;
    }

    return 0;
}

int run()
{
#if defined(BUILD_CUDA) || defined(BUILD_OPENCL)
    if(_config.pollard) {
        return runPollard();
    }
    return runBruteForce();
#else
    return runPollard();
#endif
}

/**
 * Parses a string in the form of x/y
 */
bool parseShare(const std::string &s, uint32_t &idx, uint32_t &total)
{
    size_t pos = s.find('/');
    if(pos == std::string::npos) {
        return false;
    }

    try {
        idx = util::parseUInt32(s.substr(0, pos));
    } catch(...) {
        return false;
    }

    try {
        total = util::parseUInt32(s.substr(pos + 1));
    } catch(...) {
        return false;
    }

    if(idx == 0 || total == 0) {
        return false;
    }

    if(idx > total) {
        return false;
    }

    return true;
}

bool parseHash160(const std::string &s, std::array<unsigned int,5> &hash)
{
    if(s.length() != 40) {
        return false;
    }

    for(char c : s) {
        if(!std::isxdigit(static_cast<unsigned char>(c))) {
            return false;
        }
    }

    std::array<unsigned char,20> bytes;
    for(int i = 0; i < 20; ++i) {
        unsigned int b = 0;
        std::stringstream ss;
        ss << std::hex << s.substr(i * 2, 2);
        ss >> b;
        bytes[i] = static_cast<unsigned char>(b);
    }

    // The internal representation is little-endian (hash[0] contains the
    // least significant 32 bits).  Reverse the byte array so that a
    // big-endian hex string is converted to this little-endian layout.
    std::reverse(bytes.begin(), bytes.end());

    for(int i = 0; i < 5; ++i) {
        hash[i] = static_cast<unsigned int>(bytes[i * 4]) |
                  (static_cast<unsigned int>(bytes[i * 4 + 1]) << 8) |
                  (static_cast<unsigned int>(bytes[i * 4 + 2]) << 16) |
                  (static_cast<unsigned int>(bytes[i * 4 + 3]) << 24);
    }

    // Reconstruct the big-endian hex string from the internal representation
    // to ensure the input was provided as a big-endian digest.
    std::array<unsigned char,20> verify;
    for(int i = 0; i < 5; ++i) {
        verify[i * 4]     = static_cast<unsigned char>(hash[i] & 0xFF);
        verify[i * 4 + 1] = static_cast<unsigned char>((hash[i] >> 8) & 0xFF);
        verify[i * 4 + 2] = static_cast<unsigned char>((hash[i] >> 16) & 0xFF);
        verify[i * 4 + 3] = static_cast<unsigned char>((hash[i] >> 24) & 0xFF);
    }
    std::reverse(verify.begin(), verify.end());
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    for(unsigned char b : verify) {
        oss << std::setw(2) << static_cast<unsigned int>(b);
    }
    std::string reconstructed = oss.str();
    std::string lowerInput = s;
    std::transform(lowerInput.begin(), lowerInput.end(), lowerInput.begin(), ::tolower);
    if(reconstructed != lowerInput) {
        Logger::log(LogLevel::Error, "hash160 arguments must be specified in big-endian order");
        return false;
    }

    return true;
}

bool parsePubKey(const std::string &s, std::array<unsigned int,5> &hash)
{
    if(s.length() != 66 && s.length() != 130) {
        return false;
    }

    for(char c : s) {
        if(!std::isxdigit(static_cast<unsigned char>(c))) {
            return false;
        }
    }

    unsigned int be[5];

    if(s.length() == 66) {
        std::array<unsigned char,33> bytes;
        for(int i = 0; i < 33; ++i) {
            unsigned int b = 0;
            std::stringstream ss;
            ss << std::hex << s.substr(i * 2, 2);
            ss >> b;
            bytes[i] = static_cast<unsigned char>(b);
        }
        Hash::hashPublicKeyCompressed(bytes.data(), be);
    } else {
        try {
            secp256k1::ecpoint p;
            p.x = secp256k1::uint256(s.substr(2, 64));
            p.y = secp256k1::uint256(s.substr(66, 64));
            Hash::hashPublicKey(p, be);
        } catch(...) {
            return false;
        }
    }

    for(int i = 0; i < 5; ++i) {
        hash[i] = util::endian(be[4 - i]);
    }

    return true;
}

int main(int argc, char **argv)
{
	bool optCompressed = false;
	bool optUncompressed = false;
    bool listDevices = false;
    bool optShares = false;
    bool optThreads = false;
    bool optBlocks = false;
    bool optPoints = false;

    uint32_t shareIdx = 0;
    uint32_t numShares = 0;

    // Catch --help first
    for(int i = 1; i < argc; i++) {
        if(std::string(argv[i]) == "--help") {
            usage();
            return 0;
        }
    }

#if defined(BUILD_CUDA) || defined(BUILD_OPENCL)
    // Check for supported devices
    try {
        _devices = DeviceManager::getDevices();

        if(_devices.size() == 0) {
            Logger::log(LogLevel::Warning, "No devices available");
        }
    } catch(DeviceManager::DeviceManagerException ex) {
        Logger::log(LogLevel::Error, "Error detecting devices: " + ex.msg);
        return 1;
    }
#endif

    // Check for arguments
    if(argc == 1) {
        Logger::log(LogLevel::Error, "No targets specified: provide addresses, --hash160, or --pubkey");
        usage();
        return 1;
    }


	CmdParse parser;
	parser.add("-d", "--device", true);
	parser.add("-t", "--threads", true);
	parser.add("-b", "--blocks", true);
	parser.add("-p", "--points", true);
	parser.add("-d", "--device", true);
	parser.add("-c", "--compressed", false);
	parser.add("-u", "--uncompressed", false);
    parser.add("", "--compression", true);
	parser.add("-i", "--in", true);
	parser.add("-o", "--out", true);
    parser.add("-f", "--follow", false);
    parser.add("", "--list-devices", false);
    parser.add("", "--keyspace", true);
    parser.add("", "--L", true);
    parser.add("", "--U", true);
    parser.add("", "--continue", true);
    parser.add("", "--share", true);
    parser.add("", "--hash160", true);
    parser.add("", "--pubkey", true);
    parser.add("", "--stride", true);
    parser.add("", "--pollard", false);
    parser.add("", "--offsets", true);
    parser.add("", "--window-size", true);
    parser.add("", "--tames", true);
    parser.add("", "--wilds", true);
    parser.add("", "--poll-batch", true);
    parser.add("", "--poll-interval", true);
    parser.add("", "--grid-dim", true);
    parser.add("", "--block-dim", true);
    parser.add("", "--full", false);
    parser.add("", "--deterministic", false);
    parser.add("", "--debug", false);

    try {
        parser.parse(argc, argv);
    } catch(std::string err) {
        Logger::log(LogLevel::Error, "Error: " + err);
        return 1;
    }

    std::vector<OptArg> args = parser.getArgs();

	for(unsigned int i = 0; i < args.size(); i++) {
		OptArg optArg = args[i];
		std::string opt = args[i].option;

		try {
			if(optArg.equals("-t", "--threads")) {
				_config.threads = util::parseUInt32(optArg.arg);
                optThreads = true;
            } else if(optArg.equals("-b", "--blocks")) {
                _config.blocks = util::parseUInt32(optArg.arg);
                optBlocks = true;
			} else if(optArg.equals("-p", "--points")) {
				_config.pointsPerThread = util::parseUInt32(optArg.arg);
                optPoints = true;
			} else if(optArg.equals("-d", "--device")) {
				_config.device = util::parseUInt32(optArg.arg);
			} else if(optArg.equals("-c", "--compressed")) {
				optCompressed = true;
            } else if(optArg.equals("-u", "--uncompressed")) {
                optUncompressed = true;
            } else if(optArg.equals("", "--compression")) {
                _config.compression = parseCompressionString(optArg.arg);
			} else if(optArg.equals("-i", "--in")) {
				_config.targetsFile = optArg.arg;
			} else if(optArg.equals("-o", "--out")) {
				_config.resultsFile = optArg.arg;
            } else if(optArg.equals("", "--list-devices")) {
                listDevices = true;
            } else if(optArg.equals("", "--continue")) {
                _config.checkpointFile = optArg.arg;
            } else if(optArg.equals("", "--L")) {
                try {
                    _config.startKey = secp256k1::uint256(optArg.arg);
                    _config.nextKey = _config.startKey;
                } catch(...) {
                    throw std::string("invalid argument: expected hex string");
                }
                if(_config.startKey.cmp(secp256k1::N) > 0 || _config.startKey.isZero()) {
                    throw std::string("argument is out of range");
                }
            } else if(optArg.equals("", "--U")) {
                try {
                    _config.endKey = secp256k1::uint256(optArg.arg);
                } catch(...) {
                    throw std::string("invalid argument: expected hex string");
                }
                if(_config.endKey.cmp(secp256k1::N) > 0) {
                    throw std::string("argument is out of range");
                }
            } else if(optArg.equals("", "--keyspace")) {
                secp256k1::uint256 start;
                secp256k1::uint256 end;

                parseKeyspace(optArg.arg, start, end);

                if(start.cmp(secp256k1::N) > 0) {
                    throw std::string("argument is out of range");
                }
                if(start.isZero()) {
                    throw std::string("argument is out of range");
                }

                if(end.cmp(secp256k1::N) > 0) {
                    throw std::string("argument is out of range");
                }

                if(start.cmp(end) > 0) {
                    throw std::string("Invalid argument");
                }

                _config.startKey = start;
                _config.nextKey = start;
                _config.endKey = end;
            } else if(optArg.equals("", "--share")) {
                if(!parseShare(optArg.arg, shareIdx, numShares)) {
                    throw std::string("Invalid argument");
                }
                optShares = true;
            } else if(optArg.equals("", "--hash160")) {
                std::array<unsigned int,5> arr;
                if(!parseHash160(optArg.arg, arr)) {
                    throw std::string("invalid argument: expected 40 hex characters");
                }
                _config.hash160Targets.push_back(arr);
                _config.targetTypes[arr] = "hash160";
            } else if(optArg.equals("", "--pubkey")) {
                std::array<unsigned int,5> arr;
                if(!parsePubKey(optArg.arg, arr)) {
                    throw std::string("invalid argument: expected 66 or 130 hex characters");
                }
                _config.hash160Targets.push_back(arr);
                _config.targetTypes[arr] = "pubkey";
            } else if(optArg.equals("", "--stride")) {
                try {
                    _config.stride = secp256k1::uint256(optArg.arg);
                } catch(...) {
                    throw std::string("invalid argument: : expected hex string");
                }

                if(_config.stride.cmp(secp256k1::N) >= 0) {
                    throw std::string("argument is out of range");
                }

                if(_config.stride.cmp(0) == 0) {
                    throw std::string("argument is out of range");
                }
            } else if(optArg.equals("-f", "--follow")) {
                _config.follow = true;
            } else if(optArg.equals("", "--pollard")) {
                _config.pollard = true;
            } else if(optArg.equals("", "--offsets")) {
                std::stringstream ss(optArg.arg);
                std::string item;
                while(std::getline(ss, item, ',')) {
                    item = util::trim(item);
                    if(item.length() > 0) {
                        try {
                            _config.offsets.push_back(util::parseUInt32(item));
                        } catch(...) {
                            throw std::string("invalid argument: expected integer");
                        }
                    }
                }
            } else if(optArg.equals("", "--window-size")) {
                try {
                    _config.windowSize = util::parseUInt32(optArg.arg);
                } catch(...) {
                    throw std::string("invalid argument");
                }
            } else if(optArg.equals("", "--tames")) {
                try {
                    _config.tames = util::parseUInt32(optArg.arg);
                } catch(...) {
                    throw std::string("invalid argument");
                }
            } else if(optArg.equals("", "--wilds")) {
                try {
                    _config.wilds = util::parseUInt32(optArg.arg);
                } catch(...) {
                    throw std::string("invalid argument");
                }
            } else if(optArg.equals("", "--poll-batch")) {
                try {
                    _config.pollBatch = util::parseUInt32(optArg.arg);
                } catch(...) {
                    throw std::string("invalid argument");
                }
            } else if(optArg.equals("", "--poll-interval")) {
                try {
                    _config.pollInterval = util::parseUInt32(optArg.arg);
                } catch(...) {
                    throw std::string("invalid argument");
                }
            } else if(optArg.equals("", "--grid-dim")) {
                _config.gridDim = util::parseUInt32(optArg.arg);
            } else if(optArg.equals("", "--block-dim")) {
                _config.blockDim = util::parseUInt32(optArg.arg);
            } else if(optArg.equals("", "--full")) {
                _config.full = true;
            } else if(optArg.equals("", "--deterministic")) {
                _config.deterministic = true;
            } else if(optArg.equals("", "--debug")) {
                _config.debug = true;
            }

		} catch(std::string err) {
			Logger::log(LogLevel::Error, "Error " + opt + ": " + err);
			return 1;
		}
        }

    if(_config.startKey.cmp(_config.endKey) > 0) {
        Logger::log(LogLevel::Error, "Invalid argument: L must be <= U");
        return 1;
    }

#if !(defined(BUILD_CUDA) || defined(BUILD_OPENCL))
    if(_config.deterministic) {
        Logger::log(LogLevel::Warning,
                    "--deterministic requested but no GPU support is available");
    }
#else
    if(_config.deterministic && _devices.size() == 0) {
        Logger::log(LogLevel::Warning,
                    "--deterministic requested but no GPU device was detected");
    }
#endif

#if defined(BUILD_CUDA) || defined(BUILD_OPENCL)
    if(listDevices) {
        printDeviceList(_devices);
        return 0;
    }

        // Verify device exists
        if(_config.device < 0 || _config.device >= _devices.size()) {
                Logger::log(LogLevel::Error, "device " + util::format(_config.device) + " does not exist");
                return 1;
        }
#endif

	// Parse operands
	std::vector<std::string> ops = parser.getOperands();

    // If there are no operands, then we must be reading from a file, otherwise
    // expect addresses on the commandline
        if(ops.size() == 0) {
                if(_config.targetsFile.length() == 0 && _config.hash160Targets.size() == 0) {
                        Logger::log(LogLevel::Error, "No targets specified: provide addresses, --hash160, or --pubkey");
                        usage();
                        return 1;
                }
        } else {
                for(unsigned int i = 0; i < ops.size(); i++) {
            if(!Address::verifyAddress(ops[i])) {
                Logger::log(LogLevel::Error, "Invalid address '" + ops[i] + "'");
                return 1;
            }
            _config.targets.push_back(ops[i]);
            unsigned int h[5];
            Base58::toHash160(ops[i], h);
            std::array<unsigned int,5> arr;
            for(int j = 0; j < 5; j++) {
                arr[j] = h[j];
            }
            _config.targetTypes[arr] = "address";
                }
        }
    
    // Calculate where to start and end in the keyspace when the --share option is used
    if(optShares) {
        Logger::log(LogLevel::Info, "Share " + util::format(shareIdx) + " of " + util::format(numShares));
        secp256k1::uint256 numKeys = _config.endKey - _config.nextKey + 1;

        secp256k1::uint256 diff = numKeys.mod(numShares);
        numKeys = numKeys - diff;

        secp256k1::uint256 shareSize = numKeys.div(numShares);

        secp256k1::uint256 startPos = _config.nextKey + (shareSize * (shareIdx - 1));

        if(shareIdx < numShares) {
            secp256k1::uint256 endPos = _config.nextKey + (shareSize * (shareIdx)) - 1;
            _config.endKey = endPos;
        }

        _config.nextKey = startPos;
        _config.startKey = startPos;
    }

	// Check option for compressed, uncompressed, or both
	if(optCompressed && optUncompressed) {
		_config.compression = PointCompressionType::BOTH;
	} else if(optCompressed) {
		_config.compression = PointCompressionType::COMPRESSED;
	} else if(optUncompressed) {
		_config.compression = PointCompressionType::UNCOMPRESSED;
	}

    if(_config.checkpointFile.length() > 0) {
        readCheckpointFile();
    }

    return run();
}