#include <iostream>
#include <vector>
#include <array>
#include <cstdint>

struct RefMatch {
    uint64_t k;
    std::array<unsigned int,5> h;
};

static std::vector<RefMatch> referenceWalk(uint64_t seed, unsigned int steps, unsigned int windowBits) {
    uint64_t state = seed;
    uint64_t scalar = 0;
    uint64_t mask = (windowBits >= 64) ? 0xffffffffffffffffULL : ((1ULL << windowBits) - 1ULL);
    std::vector<RefMatch> out;
    for(unsigned int i = 0; i < steps; ++i) {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        uint64_t step = (state & mask) + 1ULL;
        scalar += step;
        if((scalar & mask) == 0ULL) {
            uint64_t x = scalar;
            RefMatch m;
            m.k = scalar;
            for(int j = 0; j < 5; ++j) {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                m.h[j] = static_cast<unsigned int>(x);
            }
            out.push_back(m);
        }
    }
    return out;
}

bool testDeterministicSeed() {
    auto matches = referenceWalk(2ULL, 1000, 8);
    const uint64_t expectedK[5] = {512ULL, 34304ULL, 42752ULL, 61696ULL, 75520ULL};
    const unsigned int expectedH[5][5] = {
        {71860740u, 538972672u, 2657931812u, 555745920u, 708371077u},
        {519636748u, 1751168514u, 696367086u, 1933610753u, 3973836383u},
        {3629475406u, 2054266626u, 8060092u, 4057507397u, 2051548745u},
        {2144907490u, 521238787u, 2486401841u, 2477650311u, 143535012u},
        {3996443982u, 1651515140u, 2636627514u, 3250143174u, 3456072761u}
    };
    if(matches.size() != 5) return false;
    for(size_t i = 0; i < matches.size(); ++i) {
        if(matches[i].k != expectedK[i]) return false;
        for(int j = 0; j < 5; ++j) {
            if(matches[i].h[j] != expectedH[i][j]) return false;
        }
    }
    return true;
}

int main(){
    int fails=0;
    if(!testDeterministicSeed()) { std::cout<<"deterministic seed failed"<<std::endl; fails++; }
    if(fails==0) {
        std::cout<<"PASS"<<std::endl;
    } else {
        std::cout<<"FAIL"<<std::endl;
    }
    return fails;
}
