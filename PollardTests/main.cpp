#include <iostream>
#include <vector>
#include "PollardEngine.h"
#include "secp256k1.h"
using namespace secp256k1;

static uint256 maskBits(unsigned int bits) {
    uint256 m(0);
    for(unsigned int i=0;i<bits;i++) {
        m.v[i/32] |= (1u << (i%32));
    }
    return m;
}

bool testReconstruct() {
    uint256 key("0x11223344556677889900aabbccddeeff00112233445566778899aabbccddeeff");
    PollardEngine eng(nullptr, 0, {});
    for(unsigned int bits=32; bits<=256; bits+=32) {
        uint256 mask = maskBits(bits);
        uint256 val;
        for(int w=0; w<8; ++w) {
            val.v[w] = key.v[w] & mask.v[w];
        }
        eng.addConstraint(bits, val);
    }
    uint256 out;
    return eng.reconstruct(out) && out==key;
}

bool testTameWalk() {
    unsigned int windowBits=1;
    std::vector<unsigned int> offsets;
    for(unsigned int i=0;i<256;i++) offsets.push_back(i);
    uint256 start("1");
    uint256 recovered(0);
    PollardEngine eng([&](KeySearchResult r){recovered=r.privateKey;}, windowBits, offsets);
    eng.runTameWalk(start, 50, 12345);
    return recovered==uint256(5);
}

bool testWildWalk() {
    unsigned int windowBits=1;
    std::vector<unsigned int> offsets;
    for(unsigned int i=0;i<256;i++) offsets.push_back(i);
    ecpoint startP = multiplyPoint(uint256("2"), G());
    uint256 recovered(0);
    PollardEngine eng([&](KeySearchResult r){recovered=r.privateKey;}, windowBits, offsets);
    eng.runWildWalk(startP, 50, 12345);
    return recovered==uint256(3);
}

int main(){
    int fails=0;
    if(!testReconstruct()) { std::cout<<"reconstruct failed"<<std::endl; fails++; }
    if(!testTameWalk()) { std::cout<<"tame walk failed"<<std::endl; fails++; }
    if(!testWildWalk()) { std::cout<<"wild walk failed"<<std::endl; fails++; }
    if(fails==0) {
        std::cout<<"PASS"<<std::endl;
    } else {
        std::cout<<"FAIL"<<std::endl;
    }
    return fails;
}
