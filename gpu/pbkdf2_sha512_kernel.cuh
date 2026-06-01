
#ifndef HAS_CUDA_ROTATE
#define HAS_CUDA_ROTATE
__device__ inline unsigned int rotate(unsigned int x, unsigned int n) {
    return (x << n) | (x >> (32 - n));
}
__device__ inline unsigned long long rotate(unsigned long long x, unsigned int n) {
    return (x << n) | (x >> (64 - n));
}
#endif

/*
 * SHA-512 and PBKDF2-HMAC-SHA512 OpenCL Kernel
 * Optimized for BIP39 GPU Acceleration.
 * Supports mnemonics and passphrases up to 256 bytes.
 */

#define ror(x, n) rotate((x), (64UL - (n)))
#define Ch(x, y, z) ((x & y) ^ ((~x) & z))
#define Maj(x, y, z) ((x & y) ^ (x & z) ^ (y & z))
#define Sigma0(x) (ror(x, 28) ^ ror(x, 34) ^ ror(x, 39))
#define Sigma1(x) (ror(x, 14) ^ ror(x, 18) ^ ror(x, 41))
#define sigma0(x) (ror(x, 1) ^ ror(x, 8) ^ ((x) >> 7))
#define sigma1(x) (ror(x, 19) ^ ror(x, 61) ^ ((x) >> 6))

#define SWAP64(n) \
    (((n) << 56) | \
     (((n) & 0x000000000000FF00UL) << 40) | \
     (((n) & 0x0000000000FF0000UL) << 24) | \
     (((n) & 0x00000000FF000000UL) << 8) | \
     (((n) & 0x000000FF00000000UL) >> 8) | \
     (((n) & 0x0000FF0000000000UL) >> 24) | \
     (((n) & 0x00FF000000000000UL) >> 40) | \
     ((n) >> 56))

static __constant__ unsigned long long K[80] = {
    0x428A2F98D728AE22UL, 0x7137449123EF65CDUL, 0xB5C0FBCFEC4D3B2FUL, 0xE9B5DBA58189DBBCUL,
    0x3956C25BF348B538UL, 0x59F111F1B605D019UL, 0x923F82A4AF194F9BUL, 0xAB1C5ED5DA6D8118UL,
    0xD807AA98A3030242UL, 0x12835B0145706FBEUL, 0x243185BE4EE4B28CUL, 0x550C7DC3D5FFB4E2UL,
    0x72BE5D74F27B896FUL, 0x80DEB1FE3B1696B1UL, 0x9BDC06A725C71235UL, 0xc19bf174cf692694UL,
    0xE49B69C19EF14AD2UL, 0xEFBE4786384F25E3UL, 0x0FC19DC68B8CD5B5UL, 0x240CA1CC77AC9C65UL,
    0x2DE92C6F592B0275UL, 0x4A7484AA6EA6E483UL, 0x5CB0A9DCBD41FBD4UL, 0x76F988DA831153B5UL,
    0x983E5152EE66DFABUL, 0xA831C66D2DB43210UL, 0xB00327C898FB213FUL, 0xBF597FC7BEEF0EE4UL,
    0xC6E00BF33DA88FC2UL, 0xD5A79147930AA725UL, 0x06CA6351E003826FUL, 0x142929670A0E6E70UL,
    0x27B70A8546D22FFCUL, 0x2E1B21385C26C926UL, 0x4D2C6DFC5AC42AEDUL, 0x53380D139D95B3DFUL,
    0x650A73548BAF63DEUL, 0x766A0ABB3C77B2A8UL, 0x81C2C92E47EDAEE6UL, 0x92722C851482353BUL,
    0xA2BFE8A14CF10364UL, 0xA81A664BBC423001UL, 0xC24B8B70D0F89791UL, 0xC76C51A30654BE30UL,
    0xD192E819D6EF5218UL, 0xD69906245565A910UL, 0xF40E35855771202AUL, 0x106AA07032BBD1B8UL,
    0x19A4C116B8D2D0C8UL, 0x1E376C085141AB53UL, 0x2748774CDF8EEB99UL, 0x34B0BCB5E19B48A8UL,
    0x391C0CB3C5C95A63UL, 0x4ED8AA4AE3418ACBUL, 0x5B9CCA4F7763E373UL, 0x682E6FF3D6B2B8A3UL,
    0x748F82EE5DEFB2FCUL, 0x78A5636F43172F60UL, 0x84C87814A1F0AB72UL, 0x8CC702081A6439ECUL,
    0x90BEFFFA23631E28UL, 0xA4506CEBDE82BDE9UL, 0xBEF9A3F7B2C67915UL, 0xC67178F2E372532BUL,
    0xCA273ECEEA26619CUL, 0xD186B8C721C0C207UL, 0xEADA7DD6CDE0EB1EUL, 0xF57D4F7FEE6ED178UL,
    0x06F067AA72176FBAUL, 0x0A637DC5A2C898A6UL, 0x113F9804BEF90DAEUL, 0x1B710B35131C471BUL,
    0x28DB77F523047D84UL, 0x32CAAB7B40C72493UL, 0x3C9EBE0A15C9BEBCUL, 0x431D67C49C100D4CUL,
    0x4CC5D4BECB3E42B6UL, 0x597F299CFC657E2AUL, 0x5FCB6FAB3AD6FAECUL, 0x6C44198C4A475817UL
};

__device__ void sha512_transform( unsigned long long *state,  const unsigned long long *block) {
    unsigned long long a, b, c, d, e, f, g, h, t;
    unsigned long long w[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) w[i] = SWAP64(block[i]);
    a = state[0]; b = state[1]; c = state[2]; d = state[3];
    e = state[4]; f = state[5]; g = state[6]; h = state[7];
    for (int i = 0; i < 16; i++) {
        t = K[i] + w[i] + h + Sigma1(e) + Ch(e, f, g);
        h = g; g = f; f = e; e = d + t;
        t = t + Maj(a, b, c) + Sigma0(a);
        d = c; c = b; b = a; a = t;
    }
    for (int i = 16; i < 80; i++) {
        w[i & 15] = sigma1(w[(i - 2) & 15]) + sigma0(w[(i - 15) & 15]) + w[(i - 16) & 15] + w[(i - 7) & 15];
        t = K[i] + w[i & 15] + h + Sigma1(e) + Ch(e, f, g);
        h = g; g = f; f = e; e = d + t;
        t = t + Maj(a, b, c) + Sigma0(a);
        d = c; c = b; b = a; a = t;
    }
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

__device__ void hmac_sha512_prepare( const unsigned long long *key, unsigned int key_len,  unsigned long long *ipad_state,  unsigned long long *opad_state) {
    unsigned long long k[32]; 
    for (int i = 0; i < 32; i++) k[i] = 0;
    
    if (key_len > 128) {
        unsigned long long st[8] = { 
            0x6A09E667F3BCC908UL, 0xBB67AE8584CAA73BUL, 0x3C6EF372FE94F82BUL, 0xA54FF53A5F1D36F1UL,
            0x510E527FADE682D1UL, 0x9B05688C2B3E6C1FUL, 0x1F83D9ABFB41BD6BUL, 0x5BE0CD19137E2179UL
        };
        unsigned long long bl[16];
        unsigned int processed = 0;
        while (processed + 128 <= key_len) {
            for(int j=0; j<16; j++) bl[j] = key[processed/8 + j];
            sha512_transform(st, bl);
            processed += 128;
        }
        for(int j=0; j<16; j++) bl[j] = 0;
        unsigned int rem = key_len - processed;
        for(int j=0; j<(rem+7)/8; j++) bl[j] = key[processed/8 + j];
        (( unsigned char*)bl)[rem] = 0x80;
        bl[15] = SWAP64((unsigned long long)key_len * 8);
        sha512_transform(st, bl);
        for(int j=0; j<8; j++) k[j] = SWAP64(st[j]);
    } else {
        for (int i = 0; i < (key_len + 7) / 8; i++) k[i] = key[i];
    }
    
    unsigned long long ipad[16], opad[16];
    for (int i = 0; i < 16; i++) {
        ipad[i] = k[i] ^ 0x3636363636363636UL;
        opad[i] = k[i] ^ 0x5C5C5C5C5C5C5C5CUL;
    }
    ipad_state[0] = 0x6A09E667F3BCC908UL; ipad_state[1] = 0xBB67AE8584CAA73BUL;
    ipad_state[2] = 0x3C6EF372FE94F82BUL; ipad_state[3] = 0xA54FF53A5F1D36F1UL;
    ipad_state[4] = 0x510E527FADE682D1UL; ipad_state[5] = 0x9B05688C2B3E6C1FUL;
    ipad_state[6] = 0x1F83D9ABFB41BD6BUL; ipad_state[7] = 0x5BE0CD19137E2179UL;
    sha512_transform(ipad_state, ipad);
    opad_state[0] = 0x6A09E667F3BCC908UL; opad_state[1] = 0xBB67AE8584CAA73BUL;
    opad_state[2] = 0x3C6EF372FE94F82BUL; opad_state[3] = 0xA54FF53A5F1D36F1UL;
    opad_state[4] = 0x510E527FADE682D1UL; opad_state[5] = 0x9B05688C2B3E6C1FUL;
    opad_state[6] = 0x1F83D9ABFB41BD6BUL; opad_state[7] = 0x5BE0CD19137E2179UL;
    sha512_transform(opad_state, opad);
}

__device__ void hmac_sha512_step( const unsigned long long *ipad_state,  const unsigned long long *opad_state,  const unsigned long long *msg, unsigned int msg_len,  unsigned long long *out) {
    unsigned long long state[8];
    for(int i=0; i<8; i++) state[i] = ipad_state[i];
    unsigned long long block[16];
    for(int i=0; i<16; i++) block[i] = 0;
    for(int i=0; i<(msg_len+7)/8; i++) block[i] = msg[i];
    (( unsigned char*)block)[msg_len] = 0x80;
    block[15] = SWAP64((unsigned long long)(1024 + msg_len * 8));
    sha512_transform(state, block);
    unsigned long long out_state[8];
    for(int i=0; i<8; i++) out_state[i] = opad_state[i];
    unsigned long long out_block[16];
    for(int i=0; i<8; i++) out_block[i] = SWAP64(state[i]);
    for(int i=8; i<15; i++) out_block[i] = 0;
    (( unsigned char*)out_block)[64] = 0x80;
    out_block[15] = SWAP64(1536UL);
    sha512_transform(out_state, out_block);
    for(int j=0; j<8; j++) out[j] = SWAP64(out_state[j]);
}

__global__ void pbkdf2_hmac_sha512(
     const unsigned long long *passwords,
     const unsigned int *pass_lens,
     const unsigned long long *salts,
     const unsigned int *salt_lens,
    unsigned int iterations,
     unsigned long long *out_seeds,
    unsigned int num_keys) {
    int gid = (blockIdx.x * blockDim.x + threadIdx.x);
    if (gid >= num_keys) return;
    unsigned long long ipad_state[8], opad_state[8];
    hmac_sha512_prepare(&passwords[gid * 32], pass_lens[gid], ipad_state, opad_state);
    unsigned int s_len = salt_lens[gid];
    unsigned long long u[8], b[32]; // Increased to 32 to support 256 byte salt
    for(int i=0; i<32; i++) b[i] = salts[gid * 32 + i];
    (( unsigned char*)b)[s_len] = 0x00;
    (( unsigned char*)b)[s_len+1] = 0x00;
    (( unsigned char*)b)[s_len+2] = 0x00;
    (( unsigned char*)b)[s_len+3] = 0x01;
    hmac_sha512_step(ipad_state, opad_state, b, s_len + 4, u);
    unsigned long long t[8];
    for(int i=0; i<8; i++) t[i] = u[i];
    for (unsigned int i = 1; i < iterations; i++) {
        unsigned long long next_u[8];
        hmac_sha512_step(ipad_state, opad_state, u, 64, next_u);
        for(int j=0; j<8; j++) {
            u[j] = next_u[j];
            t[j] ^= u[j];
        }
    }
    for(int i=0; i<8; i++) out_seeds[gid * 8 + i] = t[i];
}
