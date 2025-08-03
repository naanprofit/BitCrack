#include <iostream>
#include <vector>
#include <array>
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
    // Instantiate engine with a single dummy target to exercise the CRT
    std::array<unsigned int,5> dummy = {0};
    PollardEngine eng(nullptr, 0, {}, {dummy});
    for(unsigned int bits=32; bits<=256; bits+=32) {
        uint256 mask = maskBits(bits);
        uint256 val;
        for(int w=0; w<8; ++w) {
            val.v[w] = key.v[w] & mask.v[w];
        }
        eng.addConstraint(0, bits, val);
    }
    uint256 out;
    return eng.reconstruct(0, out) && out==key;
}

int main(){
    int fails=0;
    if(!testReconstruct()) { std::cout<<"reconstruct failed"<<std::endl; fails++; }
    if(fails==0) {
        std::cout<<"PASS"<<std::endl;
    } else {
        std::cout<<"FAIL"<<std::endl;
    }
    return fails;
}
