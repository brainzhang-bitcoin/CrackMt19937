#ifndef BIP39_H
#define BIP39_H

#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <openssl/sha.h>
#include "wordlist_english.h"

// 32 bytes entropy -> 24 words mnemonic
void entropy_to_mnemonic(const uint8_t *entropy, char *mnemonic_out) {
    uint8_t hash[SHA256_DIGEST_LENGTH];
    SHA256(entropy, 32, hash);

    uint8_t bits[33]; 
    memcpy(bits, entropy, 32);
    bits[32] = hash[0]; // checksum is first byte

    int mnemonic_len = 0;
    for (int i = 0; i < 24; i++) {
        // extract 11 bits
        int bit_start = i * 11;
        int byte_idx = bit_start / 8;
        int bit_shift = bit_start % 8;

        uint32_t val = (bits[byte_idx] << 16) | (bits[byte_idx+1] << 8) | (bits[byte_idx+2]);
        val = (val >> (24 - bit_shift - 11)) & 0x7FF;

        const char *word = bip39_english_wordlist[val];
        mnemonic_len += snprintf(mnemonic_out + mnemonic_len, 256 - mnemonic_len, "%s%s", i == 0 ? "" : " ", word);
    }
}

// 16 bytes entropy -> 12 words mnemonic
void entropy_to_mnemonic_12word(const uint8_t *entropy, char *mnemonic_out) {
    uint8_t hash[SHA256_DIGEST_LENGTH];
    SHA256(entropy, 16, hash);

    uint8_t bits[20] = {0}; 
    memcpy(bits, entropy, 16);
    bits[16] = hash[0]; // checksum is first 4 bits

    int mnemonic_len = 0;
    for (int i = 0; i < 12; i++) {
        int bit_start = i * 11;
        int byte_idx = bit_start / 8;
        int bit_shift = bit_start % 8;

        uint32_t val = (bits[byte_idx] << 16) | (bits[byte_idx+1] << 8) | (bits[byte_idx+2]);
        val = (val >> (24 - bit_shift - 11)) & 0x7FF;

        const char *word = bip39_english_wordlist[val];
        mnemonic_len += snprintf(mnemonic_out + mnemonic_len, 256 - mnemonic_len, "%s%s", i == 0 ? "" : " ", word);
    }
}

#endif
