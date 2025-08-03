typedef struct {
    ulong k[4];
    uint hash[5];
} PollardCLMatch;

ulong xorshift64(ulong *state)
{
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    return *state;
}

void fake_hash160(ulong k, uint hash[5])
{
    ulong x = k;
    for(int i = 0; i < 5; i++) {
        xorshift64(&x);
        hash[i] = (uint)x;
    }
}

__kernel void pollard_random_walk(__global PollardCLMatch *out,
                                  __global uint *outCount,
                                  uint maxOut,
                                  ulong seed,
                                  uint steps,
                                  uint windowBits)
{
    if(get_global_id(0) == 0) {
        ulong state = seed;
        ulong scalar = 0UL;
        ulong mask = (windowBits >= 64) ? (ulong)0xFFFFFFFFFFFFFFFFUL : (((ulong)1 << windowBits) - 1UL);
        uint count = 0;
        for(uint i = 0; i < steps && count < maxOut; i++) {
            ulong step = (xorshift64(&state) & mask) + 1UL;
            scalar += step;
            if((scalar & mask) == 0UL) {
                fake_hash160(scalar, out[count].hash);
                out[count].k[0] = scalar & 0xffffffffUL;
                out[count].k[1] = scalar >> 32;
                out[count].k[2] = 0UL;
                out[count].k[3] = 0UL;
                count++;
            }
        }
        *outCount = count;
    }
}
