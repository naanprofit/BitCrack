#include <iostream>
#include <vector>
#include <array>
#include <cstdint>
#include <fstream>
#include <cstdio>
#include <string>
#include <sstream>
#include "AddressUtil.h"
#include "../secp256k1lib/secp256k1.h"
#include "../KeyFinder/PollardEngine.h"

#if BUILD_CUDA
#include <cuda_runtime.h>
bool runCudaScalarOne(unsigned int x[8], unsigned int y[8], unsigned int hash[5]);
#endif

#if BUILD_OPENCL
#include "clutil.h"
#include "clContext.h"
#endif

struct RefMatch {
    uint64_t k;
    std::array<unsigned int,5> h;
};

struct RNGState {
    uint64_t s0;
    uint64_t s1;
};

static inline uint64_t xorshift128plus(RNGState &st)
{
    uint64_t x = st.s0;
    uint64_t y = st.s1;
    st.s0 = y;
    x ^= x << 23;
    x ^= x >> 17;
    x ^= y;
    x ^= y >> 26;
    st.s1 = x;
    return x + y;
}

static secp256k1::ecpoint scalarMultiplyBase(uint64_t k) {
    secp256k1::uint256 scalar(k);
    auto split = secp256k1::splitScalar(scalar);
    secp256k1::ecpoint p1 = secp256k1::multiplyPointSmall(split.k1, secp256k1::G());
    secp256k1::ecpoint p2 = secp256k1::multiplyPointSmall(split.k2, secp256k1::G());
    if(!split.k2.isZero()) {
        p2 = secp256k1::glvEndomorphism(p2);
    }
    return secp256k1::glvRecombine(split, p1, p2);
}

static inline uint64_t next_random_step(RNGState &st)
{
    const uint64_t ORDER_MINUS_ONE = 0xBFD25E8CD0364140ULL;
    return (xorshift128plus(st) % ORDER_MINUS_ONE) + 1ULL;
}

static void fake_hash160(uint64_t k, std::array<unsigned int,5> &h)
{
    RNGState st{ k, k ^ 0x9E3779B97F4A7C15ULL };
    for(int i = 0; i < 5; ++i) {
        h[i] = static_cast<unsigned int>(xorshift128plus(st));
    }
}

static std::vector<RefMatch> referenceWalk(uint64_t seed, unsigned int steps, unsigned int windowBits) {
    RNGState rng{ seed ^ 1ULL, seed + 1ULL };
    uint64_t scalar = 0;
    const uint64_t ORDER = 0xBFD25E8CD0364141ULL;
    uint64_t mask = (windowBits >= 64) ? 0xffffffffffffffffULL : ((1ULL << windowBits) - 1ULL);
    std::vector<RefMatch> out;
    for(unsigned int i = 0; i < steps; ++i) {
        uint64_t step = next_random_step(rng);
        scalar += step;
        scalar %= ORDER;
        if((scalar & mask) == 0ULL) {
            RefMatch m;
            m.k = scalar;
            fake_hash160(scalar, m.h);
            out.push_back(m);
        }
    }
    return out;
}

static inline unsigned int bswap32(unsigned int x) {
    return (x << 24) | ((x << 8) & 0x00ff0000U) |
           ((x >> 8) & 0x0000ff00U) | (x >> 24);
}

static std::array<unsigned int,5> hashWindowLE(const unsigned int h[5], unsigned int offset,
                                               unsigned int bits) {
    std::array<unsigned int,5> out{};
    unsigned int word = offset / 32;
    unsigned int bit  = offset % 32;
    unsigned int words = (bits + 31) / 32;
    for(unsigned int i = 0; i < words && word + i < 5; ++i) {
        uint64_t val = ((uint64_t)h[word + i]) >> bit;
        if(bit && word + i + 1 < 5) {
            val |= ((uint64_t)h[word + i + 1]) << (32 - bit);
        }
        out[i] = static_cast<unsigned int>(val & 0xffffffffULL);
    }
    unsigned int maskBits = bits % 32;
    if(maskBits) {
        unsigned int mask = (1u << maskBits) - 1u;
        out[words - 1] &= mask;
    }
    for(unsigned int i = words; i < 5; ++i) {
        out[i] = 0u;
    }
    return out;
}

bool testDeterministicSeed() {
    auto matches = referenceWalk(2ULL, 1000, 8);
    const uint64_t expectedK[1] = {10820130499599738880ULL};
    const unsigned int expectedH[1][5] = {
        {3977112439u, 4030187129u, 4139897143u, 2627728552u, 1044400310u}
    };
    if(matches.size() != 1) return false;
    for(size_t i = 0; i < matches.size(); ++i) {
        if(matches[i].k != expectedK[i]) return false;
        for(int j = 0; j < 5; ++j) {
            if(matches[i].h[j] != expectedH[i][j]) return false;
        }
    }
    return true;
}

bool testScalarOne() {
    secp256k1::ecpoint p = scalarMultiplyBase(1ULL);
    unsigned int digest[5];
    Hash::hashPublicKeyCompressed(p, digest);
    const unsigned int expected[5] = {
        0x751e76e8u,
        0x199196d4u,
        0x54941c45u,
        0xd1b3a323u,
        0xf1433bd6u
    };
    for(int i = 0; i < 5; ++i) {
        if(digest[i] != expected[i]) return false;
    }
    return true;
}

bool testGlvMatchesClassic() {
    uint64_t sample = 0x123456789abcdefULL;
    secp256k1::uint256 k(sample);
    secp256k1::ecpoint classic = secp256k1::multiplyPoint(k, secp256k1::G());
    secp256k1::ecpoint glv = scalarMultiplyBase(sample);
    return classic.x == glv.x && classic.y == glv.y;
}

bool testHashWindowLEPython() {
    secp256k1::ecpoint p = scalarMultiplyBase(1ULL);
    unsigned int be[5];
    Hash::hashPublicKeyCompressed(p, be);
    unsigned int le[5];
    for(int i = 0; i < 5; ++i) {
        le[i] = bswap32(be[4 - i]);
    }
    auto got = hashWindowLE(le, 0, 160);

    const char *cmd =
        "python3 - <<'PY'\n"
        "import hashlib\n"
        "pub=bytes.fromhex('0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798')\n"
        "r=hashlib.new('ripemd160', hashlib.sha256(pub).digest()).digest()\n"
        "print(hex(int.from_bytes(r,'big')))\n"
        "PY";
    FILE *pipe = popen(cmd, "r");
    if(!pipe) return false;
    char buffer[128];
    std::string py;
    while(fgets(buffer, sizeof(buffer), pipe)) {
        py += buffer;
    }
    pclose(pipe);
    while(!py.empty() && (py.back() == '\n' || py.back() == '\r')) py.pop_back();
    if(py.rfind("0x", 0) == 0) py = py.substr(2);
    std::array<unsigned int,5> arr{};
    for(int i = 0; i < 5; ++i) {
        std::string word = py.substr((4 - i) * 8, 8);
        unsigned int w = 0;
        std::stringstream ss;
        ss << std::hex << word;
        ss >> w;
        arr[i] = bswap32(w);
    }
    std::array<unsigned int,5> expected = arr;
    return got == expected;
}

bool testHashWindowLEK1() {
    const unsigned int be[5] = {
        0x751e76e8u,
        0x199196d4u,
        0x54941c45u,
        0xd1b3a323u,
        0xf1433bd6u
    };
    unsigned int le[5];
    secp256k1::uint256::importBigEndian(be, 5).exportWords(le, 5);
    auto got = hashWindowLE(le, 0, 160);
    std::array<unsigned int,5> expected;
    for(int i = 0; i < 5; ++i) expected[i] = le[i];
    return got == expected;
}

bool testHashWindowK1Windows() {
    const unsigned int be[5] = {
        0x751e76e8u,
        0x199196d4u,
        0x54941c45u,
        0xd1b3a323u,
        0xf1433bd6u
    };
    unsigned int le[5];
    // Convert big-endian digest to little-endian words
    secp256k1::uint256::importBigEndian(be, 5).exportWords(le, 5);
    struct Case { unsigned int off; unsigned int bits; std::vector<unsigned int> words; };
    std::vector<Case> cases = {
        {0, 80,  {0xd63b43f1u, 0x23a3b3d1u, 0x00009454u}},
        {40, 80, {0x5423a3b3u, 0x19451c94u, 0x00009691u}},
        {80, 80, {0x9119451cu, 0x1e75d496u, 0x0000e876u}},
        {0, 96,  {0xd63b43f1u, 0x23a3b3d1u, 0x451c9454u}},
        {32, 96, {0x23a3b3d1u, 0x451c9454u, 0xd4969119u}},
        {64, 96, {0x451c9454u, 0xd4969119u, 0xe8761e75u}},
        {64, 32, {0x451c9454u}},
        {76, 19, {0x000451c9u}},
        {84, 19, {0x00019451u}}
    };
    for(const auto &c : cases) {
        auto got = hashWindowLE(le, c.off, c.bits);
        unsigned int words = (c.bits + 31) / 32;
        for(unsigned int i = 0; i < words; ++i) {
            if(got[i] != c.words[i]) return false;
        }
        for(unsigned int i = words; i < 5; ++i) {
            if(got[i] != 0u) return false;
        }
    }
    return true;
}

bool testHashWindowOffsets() {
    const unsigned int be[5] = {
        0x751e76e8u,
        0x199196d4u,
        0x54941c45u,
        0xd1b3a323u,
        0xf1433bd6u
    };
    unsigned int le[5];
    // Convert big-endian digest to little-endian words
    secp256k1::uint256::importBigEndian(be, 5).exportWords(le, 5);
    unsigned int offsets[4] = {0,20,40,60};

    const char *cmd =
        "python3 - <<'PY'\n"
        "h=bytes.fromhex('751e76e8199196d454941c45d1b3a323f1433bd6')\n"
        "w=[int.from_bytes(h[::-1][i*4:(i+1)*4],'big') for i in range(5)]\n"
        "for off in (0,20,40,60):\n"
        "    word=off//32\n"
        "    bit=off%32\n"
        "    val=w[word]>>bit\n"
        "    if bit and word+1<5:\n"
        "        val|=w[word+1]<<(32-bit)\n"
        "    val &= (1<<20)-1\n"
        "    print(hex(val))\n"
        "PY";
    FILE *pipe = popen(cmd, "r");
    if(!pipe) return false;
    char buffer[128];
    std::vector<unsigned int> expected;
    while(fgets(buffer, sizeof(buffer), pipe)) {
        std::string line(buffer);
        while(!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
        if(line.rfind("0x", 0) == 0) line = line.substr(2);
        unsigned int val = 0;
        std::stringstream ss;
        ss << std::hex << line;
        ss >> val;
        expected.push_back(val);
    }
    pclose(pipe);
    if(expected.size() != 4) return false;
    for(int i = 0; i < 4; ++i) {
        auto got = hashWindowLE(le, offsets[i], 20);
        if(got[0] != expected[i]) return false;
        for(int j = 1; j < 5; ++j) {
            if(got[j] != 0u) return false;
        }
    }
    return true;
}

class StubDevice : public PollardDevice {
    PollardMatch _m;
    bool _sent = false;
public:
    StubDevice() {
        secp256k1::uint256 k(1);
        secp256k1::ecpoint p = secp256k1::multiplyPoint(k, secp256k1::G());
        unsigned int be[5];
        Hash::hashPublicKeyCompressed(p, be);
        for(int j = 0; j < 5; ++j) {
            _m.hash[j] = bswap32(be[4 - j]);
        }
        _m.scalar = k;
    }
    void startTameWalk(const secp256k1::uint256 &, uint64_t, const secp256k1::uint256 &, bool) override {
        _sent = false;
    }
    void startWildWalk(const secp256k1::uint256 &, uint64_t, const secp256k1::uint256 &, bool) override {
        _sent = false;
    }
    bool popResult(PollardMatch &out) override {
        if(_sent) return false;
        out = _m;
        _sent = true;
        return true;
    }
};

bool testPollardHash160FindsKey() {
    unsigned int words[5];
    Base58::toHash160("1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH", words);
    std::array<unsigned int,5> target{};
    for(int i = 0; i < 5; ++i) target[i] = words[i];
    bool found = false;
    PollardEngine engine([&](KeySearchResult r){ if(r.privateKey == secp256k1::uint256(1)) found = true; },
                         8, std::vector<unsigned int>{0}, std::vector<std::array<unsigned int,5>>{target},
                         secp256k1::uint256(1), secp256k1::uint256(0xFF), 1, 100, true, false);
    engine.setDevice(std::unique_ptr<PollardDevice>(new StubDevice()));
    engine.runTameWalk(secp256k1::uint256(1), 1);
    return found;
}

static bool runOpenCLScalarOne(unsigned int x[8], unsigned int y[8], unsigned int hash[5]) {
#if BUILD_OPENCL
    try {
        auto devices = cl::getDevices();
        if(devices.empty()) return false;
        cl::CLContext ctx(devices[0].id);

        auto readFile = [](const char *path){
            std::ifstream f(path); return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        };
        std::string sha = readFile("clMath/sha256.cl");
        std::string secp = readFile("clMath/secp256k1.cl");
        std::string rmd = readFile("clMath/ripemd160.cl");
        std::string pollard = readFile("CLKeySearchDevice/clPollard.cl");
        std::string src = sha + secp + rmd + pollard + R"(
        __kernel void scalar_one(__global uint* xOut, __global uint* yOut, __global uint* hOut) {
            uint px[8]; uint py[8]; uint digest[5];
            scalarMultiplyBase((ulong)1, px, py);
            hashPublicKeyCompressed(px, py[7], digest);
            for(int i=0;i<8;i++){ xOut[i]=px[i]; yOut[i]=py[i]; }
            for(int j=0;j<5;j++){ hOut[j]=digest[j]; }
        }
        )";
        cl::CLProgram prog(ctx, src.c_str());
        cl::CLKernel k(prog, "scalar_one");
        cl_mem dx = ctx.malloc(sizeof(unsigned int)*8);
        cl_mem dy = ctx.malloc(sizeof(unsigned int)*8);
        cl_mem dh = ctx.malloc(sizeof(unsigned int)*5);
        k.set_args(dx, dy, dh);
        k.call(1,1);
        ctx.copyDeviceToHost(dx, x, sizeof(unsigned int)*8);
        ctx.copyDeviceToHost(dy, y, sizeof(unsigned int)*8);
        ctx.copyDeviceToHost(dh, hash, sizeof(unsigned int)*5);
        ctx.free(dx); ctx.free(dy); ctx.free(dh);
        return true;
    } catch(cl::CLException &) {
        return false;
    }
#else
    (void)x; (void)y; (void)hash; return false;
#endif
}

bool testGpuScalarOne() {
    unsigned int expectedHash[5] = {
        0x751e76e8u, 0x199196d4u, 0x54941c45u, 0xd1b3a323u, 0xf1433bd6u
    };
    secp256k1::ecpoint G = secp256k1::G();
    unsigned int gx[8];
    unsigned int gy[8];
    G.x.exportWords(gx, 8, secp256k1::uint256::BigEndian);
    G.y.exportWords(gy, 8, secp256k1::uint256::BigEndian);

    bool ran = false;
    bool pass = true;

#if BUILD_CUDA
    int cudaDevs = 0;
    if(cudaGetDeviceCount(&cudaDevs) == cudaSuccess && cudaDevs > 0) {
        ran = true;
        unsigned int x[8], y[8], h[5];
        if(!runCudaScalarOne(x, y, h)) return false;
        for(int i=0;i<8;i++){ if(x[i]!=gx[i] || y[i]!=gy[i]) pass=false; }
        for(int i=0;i<5;i++){ if(h[i]!=expectedHash[i]) pass=false; }
    }
#endif
#if BUILD_OPENCL
    unsigned int xcl[8], ycl[8], hcl[5];
    if(runOpenCLScalarOne(xcl, ycl, hcl)) {
        ran = true;
        for(int i=0;i<8;i++){ if(xcl[i]!=gx[i] || ycl[i]!=gy[i]) pass=false; }
        for(int i=0;i<5;i++){ if(hcl[i]!=expectedHash[i]) pass=false; }
    }
#endif
    if(!ran) {
        std::cout << "GPU test skipped" << std::endl;
        return true;
    }
    return pass;
}

int main(){
    int fails=0;
    if(!testScalarOne()) { std::cout<<"scalar one failed"<<std::endl; fails++; }
    if(!testGlvMatchesClassic()) { std::cout<<"glv compare failed"<<std::endl; fails++; }
    if(!testDeterministicSeed()) { std::cout<<"deterministic seed failed"<<std::endl; fails++; }
    if(!testPollardHash160FindsKey()) { std::cout<<"pollard hash160 failed"<<std::endl; fails++; }
    if(!testGpuScalarOne()) { std::cout<<"gpu scalar one failed"<<std::endl; fails++; }
    if(!testHashWindowLEK1()) { std::cout<<"hash window k1 failed"<<std::endl; fails++; }
    if(!testHashWindowK1Windows()) { std::cout<<"hash window segments failed"<<std::endl; fails++; }
    if(!testHashWindowOffsets()) { std::cout<<"hash window offsets failed"<<std::endl; fails++; }
    if(!testHashWindowLEPython()) { std::cout<<"hash window python failed"<<std::endl; fails++; }
    if(fails==0) {
        std::cout<<"PASS"<<std::endl;
    } else {
        std::cout<<"FAIL"<<std::endl;
    }
    return fails;
}
