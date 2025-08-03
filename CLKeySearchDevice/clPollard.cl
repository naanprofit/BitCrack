typedef struct {
    ulong k[4];
    uint hash[5];
} PollardCLMatch;

typedef struct {
    ulong s0;
    ulong s1;
} RNGState;

ulong xorshift128plus(__private RNGState *state)
{
    ulong x = state->s0;
    ulong y = state->s1;
    state->s0 = y;
    x ^= x << 23;
    x ^= x >> 17;
    x ^= y;
    x ^= y >> 26;
    state->s1 = x;
    return x + y;
}

ulong next_random_step(__private RNGState *state)
{
    const ulong ORDER_MINUS_ONE = (ulong)0xBFD25E8CD0364140UL;
    return (xorshift128plus(state) % ORDER_MINUS_ONE) + 1UL;
}

void fake_hash160(ulong k, uint hash[5])
{
    RNGState st = { k, k ^ (ulong)0x9E3779B97F4A7C15UL };
    for(int i = 0; i < 5; i++) {
        hash[i] = (uint)xorshift128plus(&st);
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
        RNGState rng = { seed ^ (ulong)1, seed + (ulong)1 };
        ulong scalar = 0UL;
        const ulong ORDER = (ulong)0xBFD25E8CD0364141UL;
        ulong mask = (windowBits >= 64) ? (ulong)0xFFFFFFFFFFFFFFFFUL : (((ulong)1 << windowBits) - 1UL);
        uint count = 0;
        for(uint i = 0; i < steps && count < maxOut; i++) {
            ulong step = next_random_step(&rng);
            scalar += step;
            scalar %= ORDER;
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
