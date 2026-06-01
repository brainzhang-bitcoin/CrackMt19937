#ifndef CUDA_PIPELINE_CUH
#define CUDA_PIPELINE_CUH

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// The converted kernels
#include "gpu/pbkdf2_sha512_kernel.cuh"
#include "gpu/pipeline_kernel.cuh"

// Embedded kernels contains massive data arrays
#include "embedded_kernels.h"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "[CUDA ERROR] %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while (0)


typedef struct {
    uint8_t *d_precomp_buf; // secp256k1 precomputed table (61KB, uploaded once)
} CudaState;

void init_cuda(CudaState *state, int pipeline_opt_level) {

    fprintf(stderr, "[INIT] Step 1/2: Allocating device memory for precomp_g...\n");
    CUDA_CHECK(cudaMalloc((void**)&state->d_precomp_buf, precomp_g_data_len));
    fprintf(stderr, "[INIT] Step 2/2: Uploading precomp_g (%u bytes)...\n", precomp_g_data_len);
    CUDA_CHECK(cudaMemcpy(state->d_precomp_buf, precomp_g_data, precomp_g_data_len, cudaMemcpyHostToDevice));

    fprintf(stderr, "[INIT] CUDA fully initialized!\n");
    fflush(stderr);
}

// Batch version of derive_seed_cuda with custom salts
void derive_seed_batch_cuda_with_salt(CudaState *state, char **mnemonics, char **salts, int num_keys, uint8_t *seeds_out) {
    unsigned long long *pass_data = (unsigned long long*)calloc(num_keys * 32, sizeof(unsigned long long)); // 256 bytes per key
    uint32_t *pass_lens = (uint32_t*)calloc(num_keys, sizeof(uint32_t));
    unsigned long long *salt_data = (unsigned long long*)calloc(num_keys * 32, sizeof(unsigned long long)); // 256 bytes per key
    uint32_t *salt_lens = (uint32_t*)calloc(num_keys, sizeof(uint32_t));
    
    for (int i = 0; i < num_keys; i++) {
        pass_lens[i] = strlen(mnemonics[i]);
        memcpy(pass_data + i * 32, mnemonics[i], pass_lens[i]);
        
        salt_lens[i] = strlen(salts[i]);
        memcpy(salt_data + i * 32, salts[i], salt_lens[i]);
    }

    unsigned long long *d_passwords;
    uint32_t *d_pass_lens;
    unsigned long long *d_salts;
    uint32_t *d_salt_lens;
    unsigned long long *d_out;

    CUDA_CHECK(cudaMalloc((void**)&d_passwords, num_keys * 256));
    CUDA_CHECK(cudaMalloc((void**)&d_pass_lens, num_keys * 4));
    CUDA_CHECK(cudaMalloc((void**)&d_salts, num_keys * 256));
    CUDA_CHECK(cudaMalloc((void**)&d_salt_lens, num_keys * 4));
    CUDA_CHECK(cudaMalloc((void**)&d_out, 64 * num_keys));
    CUDA_CHECK(cudaMemset(d_out, 0, 64 * num_keys));

    CUDA_CHECK(cudaMemcpy(d_passwords, pass_data, num_keys * 256, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pass_lens, pass_lens, num_keys * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_salts, salt_data, num_keys * 256, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_salt_lens, salt_lens, num_keys * 4, cudaMemcpyHostToDevice));

    uint32_t iterations = 2048;
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_keys + threadsPerBlock - 1) / threadsPerBlock;

    pbkdf2_hmac_sha512<<<blocksPerGrid, threadsPerBlock>>>(
        d_passwords, d_pass_lens, d_salts, d_salt_lens, iterations, d_out, (uint32_t)num_keys
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(seeds_out, d_out, 64 * num_keys, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_passwords));
    CUDA_CHECK(cudaFree(d_pass_lens));
    CUDA_CHECK(cudaFree(d_salts));
    CUDA_CHECK(cudaFree(d_salt_lens));
    CUDA_CHECK(cudaFree(d_out));
    
    free(pass_data);
    free(pass_lens);
    free(salt_data);
    free(salt_lens);
}

// Wrapper for derive_seed_batch_cuda with default "mnemonic" salt
void derive_seed_batch_cuda(CudaState *state, char **mnemonics, int num_keys, uint8_t *seeds_out) {
    char **salts = (char**)malloc(sizeof(char*) * num_keys);
    for (int i = 0; i < num_keys; i++) {
        salts[i] = strdup("mnemonic");
    }
    derive_seed_batch_cuda_with_salt(state, mnemonics, salts, num_keys, seeds_out);
    for (int i = 0; i < num_keys; i++) {
        free(salts[i]);
    }
    free(salts);
}

typedef enum {
    MODE_BIP44,
    MODE_BIP49,
    MODE_BIP84,
    MODE_BIP32,
    MODE_CORE,
    MODE_BLOCKCHAIN
} KernelMode;

void derive_bip32_cuda(CudaState *state, KernelMode mode, uint8_t *seeds, int num_seeds, int num_addresses, uint8_t *out_privs) {
    uint8_t *d_seeds;
    uint8_t *d_out;
    uint32_t n_addrs = num_addresses;

    CUDA_CHECK(cudaMalloc((void**)&d_seeds, 64 * num_seeds));
    CUDA_CHECK(cudaMalloc((void**)&d_out, 32 * num_seeds * num_addresses));
    CUDA_CHECK(cudaMemset(d_out, 0, 32 * num_seeds * num_addresses));

    CUDA_CHECK(cudaMemcpy(d_seeds, seeds, 64 * num_seeds, cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocksPerGrid = (num_seeds + threadsPerBlock - 1) / threadsPerBlock;

    switch (mode) {
        case MODE_BIP44:
            bip32_pipeline_bip44_priv<<<blocksPerGrid, threadsPerBlock>>>(d_seeds, d_out, n_addrs, (uint32_t)num_seeds);
            break;
        case MODE_BIP49:
            bip32_pipeline_bip49_priv<<<blocksPerGrid, threadsPerBlock>>>(d_seeds, d_out, n_addrs, (uint32_t)num_seeds);
            break;
        case MODE_BIP84:
            bip32_pipeline_bip84_priv<<<blocksPerGrid, threadsPerBlock>>>(d_seeds, d_out, n_addrs, (uint32_t)num_seeds);
            break;
        case MODE_BIP32:
            bip32_pipeline_bip32_priv<<<blocksPerGrid, threadsPerBlock>>>(d_seeds, d_out, n_addrs, (uint32_t)num_seeds);
            break;
        case MODE_CORE:
            bip32_pipeline_core_priv<<<blocksPerGrid, threadsPerBlock>>>(d_seeds, d_out, n_addrs, (uint32_t)num_seeds);
            break;
        case MODE_BLOCKCHAIN:
            bip32_pipeline_blockchain_priv<<<blocksPerGrid, threadsPerBlock>>>(d_seeds, d_out, n_addrs, (uint32_t)num_seeds);
            break;
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(out_privs, d_out, 32 * num_seeds * num_addresses, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_seeds));
    CUDA_CHECK(cudaFree(d_out));
}

#endif
