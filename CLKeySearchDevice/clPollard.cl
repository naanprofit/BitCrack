typedef struct {
    ulong k[4];
    uint hash[5];
} PollardCLMatch;

__kernel void pollard_random_walk(__global PollardCLMatch *out, ulong seed) {
    if(get_global_id(0) == 0) {
        for(int i=0;i<4;i++) out[0].k[i] = seed;
        for(int i=0;i<5;i++) out[0].hash[i] = 0;
    }
}
