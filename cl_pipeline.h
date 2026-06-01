#define CL_TARGET_OPENCL_VERSION 300
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS 1
#ifndef CL_PIPELINE_H
#define CL_PIPELINE_H

#include <CL/cl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "embedded_kernels.h"

typedef struct {
    cl_context ctx;
    cl_command_queue queue;
    cl_program pbkdf2_prog;
    cl_kernel pbkdf2_kernel;
    
    cl_program pipeline_prog;
    cl_kernel bip44_kernel;
    cl_kernel bip49_kernel;
    cl_kernel bip84_kernel;
    cl_kernel bip32_kernel;
    cl_kernel core_kernel;
    cl_kernel blockchain_kernel;
    cl_mem precomp_buf;  // secp256k1 precomputed table (61KB, uploaded once)
} OpenCLState;

void init_opencl(OpenCLState *state, int pipeline_opt_level) {
    cl_int err;

    fprintf(stderr, "[INIT] Step 1/8: create_context...\n");
    cl_device_id device;
    state->ctx = create_context(&device);
    if (!state->ctx) {
        fprintf(stderr, "[INIT] FATAL: Failed to create OpenCL context\n");
        exit(1);
    }
    fprintf(stderr, "[INIT] Step 1/8: done\n");

    fprintf(stderr, "[INIT] Step 2/8: clCreateCommandQueue...\n");
    state->queue = clCreateCommandQueue(state->ctx, device, 0, &err);
    fprintf(stderr, "[INIT] Step 2/8: done, err=%s\n", cl_error_string(err));

    // Build PBKDF2 program — full optimization (compiles fine on all GPUs)
    fprintf(stderr, "[INIT] Step 3/8: build PBKDF2 program (%u bytes)...\n", pbkdf2_kernel_src_len);
    state->pbkdf2_prog = build_program_from_source(state->ctx, device, (const char*)pbkdf2_kernel_src, pbkdf2_kernel_src_len, NULL);
    fprintf(stderr, "[INIT] Step 3/8: done\n");

    fprintf(stderr, "[INIT] Step 4/8: create PBKDF2 kernel...\n");
    state->pbkdf2_kernel = clCreateKernel(state->pbkdf2_prog, "pbkdf2_hmac_sha512", &err);
    fprintf(stderr, "[INIT] Step 4/8: done, err=%s\n", cl_error_string(err));

    // Auto-detect GPU arch for pipeline optimization
    // Ada Lovelace (RTX 40xx) ptxas hangs on complex secp256k1 math with optimization enabled
    if (pipeline_opt_level < 0) {
        // Auto-detect: check GPU name for Ada Lovelace (RTX 40xx)
        char dev_name[256] = {0};
        clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(dev_name), dev_name, NULL);
        
        // RTX 40xx = Ada Lovelace (sm_89), hangs on pipeline optimization
        if (strstr(dev_name, "4090") || strstr(dev_name, "4080") || 
            strstr(dev_name, "4070") || strstr(dev_name, "4060") ||
            strstr(dev_name, "4050") || strstr(dev_name, "L40") ||
            strstr(dev_name, "L4 ") || strstr(dev_name, "AD10")) {
            pipeline_opt_level = 0;
            fprintf(stderr, "[INIT] GPU '%s' detected as Ada Lovelace — using -cl-opt-disable for pipeline\n", dev_name);
        } else {
            pipeline_opt_level = 3;
            fprintf(stderr, "[INIT] GPU '%s' — using full optimization for pipeline\n", dev_name);
        }
    }

    // Build Pipeline Program with selected optimization level
    const char *pipeline_opts = NULL;
    if (pipeline_opt_level <= 0) {
        pipeline_opts = "-cl-opt-disable";
    }
    fprintf(stderr, "[INIT] Step 5/8: build Pipeline program (%u bytes, %s)...\n", 
            pipeline_kernel_src_len, pipeline_opts ? pipeline_opts : "full-opt");
    fflush(stderr);
    state->pipeline_prog = build_program_from_source(state->ctx, device, (const char*)pipeline_kernel_src, pipeline_kernel_src_len, pipeline_opts);
    fprintf(stderr, "[INIT] Step 5/8: done\n");

    fprintf(stderr, "[INIT] Step 6/8: create BIP44/49/84 kernels...\n");
    state->bip49_kernel = clCreateKernel(state->pipeline_prog, "bip32_pipeline_bip49_priv", &err);
    fprintf(stderr, "[INIT]   bip49: err=%s\n", cl_error_string(err));
    state->bip44_kernel = clCreateKernel(state->pipeline_prog, "bip32_pipeline_bip44_priv", &err);
    fprintf(stderr, "[INIT]   bip44: err=%s\n", cl_error_string(err));
    state->bip84_kernel = clCreateKernel(state->pipeline_prog, "bip32_pipeline_bip84_priv", &err);
    fprintf(stderr, "[INIT]   bip84: err=%s\n", cl_error_string(err));
    fprintf(stderr, "[INIT] Step 6/8: done\n");

    fprintf(stderr, "[INIT] Step 7/8: create BIP32/Core/Blockchain kernels...\n");
    state->bip32_kernel = clCreateKernel(state->pipeline_prog, "bip32_pipeline_bip32_priv", &err);
    fprintf(stderr, "[INIT]   bip32: err=%s\n", cl_error_string(err));
    state->core_kernel = clCreateKernel(state->pipeline_prog, "bip32_pipeline_core_priv", &err);
    fprintf(stderr, "[INIT]   core: err=%s\n", cl_error_string(err));
    state->blockchain_kernel = clCreateKernel(state->pipeline_prog, "bip32_pipeline_blockchain_priv", &err);
    fprintf(stderr, "[INIT]   blockchain: err=%s\n", cl_error_string(err));
    fprintf(stderr, "[INIT] Step 7/8: done\n");

    // Upload precomputed secp256k1 table to GPU (61KB, once)
    fprintf(stderr, "[INIT] Step 8/8: upload precomp_g (%u bytes)...\n", precomp_g_data_len);
    state->precomp_buf = clCreateBuffer(state->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, precomp_g_data_len, (void*)precomp_g_data, &err);
    fprintf(stderr, "[INIT] Step 8/8: done, err=%s\n", cl_error_string(err));

    fprintf(stderr, "[INIT] OpenCL fully initialized!\n");
    fflush(stderr);
}

// Convert 24-word string "word1 word2..." -> PBKDF2 -> 64 byte seed
void derive_seed_opencl(OpenCLState *state, const char *mnemonic, uint8_t *seed_out) {
    size_t global_item_size = 1;


    uint64_t pass_data[32] = {0}; // 256 bytes
    uint64_t salt_data[32] = {0}; // 256 bytes
    uint32_t pass_len = strlen(mnemonic);
    memcpy(pass_data, mnemonic, pass_len);
    
    const char *salt_base = "mnemonic";
    uint32_t salt_len = strlen(salt_base);
    memcpy(salt_data, salt_base, salt_len);
    
    cl_mem d_passwords = clCreateBuffer(state->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 256, pass_data, NULL);
    cl_mem d_pass_lens = clCreateBuffer(state->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 4, &pass_len, NULL);
    cl_mem d_salts = clCreateBuffer(state->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 256, salt_data, NULL);
    cl_mem d_salt_lens = clCreateBuffer(state->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 4, &salt_len, NULL);
    cl_mem d_out = clCreateBuffer(state->ctx, CL_MEM_WRITE_ONLY, 64, NULL, NULL);
    
    uint32_t iterations = 2048; // PBKDF2 iterations for BIP39

    clSetKernelArg(state->pbkdf2_kernel, 0, sizeof(cl_mem), &d_passwords);
    clSetKernelArg(state->pbkdf2_kernel, 1, sizeof(cl_mem), &d_pass_lens);
    clSetKernelArg(state->pbkdf2_kernel, 2, sizeof(cl_mem), &d_salts);
    clSetKernelArg(state->pbkdf2_kernel, 3, sizeof(cl_mem), &d_salt_lens);
    clSetKernelArg(state->pbkdf2_kernel, 4, sizeof(uint32_t), &iterations);
    clSetKernelArg(state->pbkdf2_kernel, 5, sizeof(cl_mem), &d_out);

    clEnqueueNDRangeKernel(state->queue, state->pbkdf2_kernel, 1, NULL, &global_item_size, NULL, 0, NULL, NULL);
    clEnqueueReadBuffer(state->queue, d_out, CL_TRUE, 0, 64, seed_out, 0, NULL, NULL);

    clReleaseMemObject(d_passwords);
    clReleaseMemObject(d_pass_lens);
    clReleaseMemObject(d_salts);
    clReleaseMemObject(d_salt_lens);
    clReleaseMemObject(d_out);
}

void derive_bip32_opencl(OpenCLState *state, cl_kernel kernel, uint8_t *seeds, int num_seeds, int num_addresses, uint8_t *out_privs) {
    size_t global_item_size = num_seeds;

    
    cl_mem d_seeds = clCreateBuffer(state->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 64 * num_seeds, seeds, NULL);
    cl_mem d_out = clCreateBuffer(state->ctx, CL_MEM_WRITE_ONLY, 32 * num_seeds * num_addresses, NULL, NULL);
    uint32_t n_addrs = num_addresses;
    
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_seeds);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_out);
    clSetKernelArg(kernel, 2, sizeof(uint32_t), &n_addrs);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &state->precomp_buf);
    
    clEnqueueNDRangeKernel(state->queue, kernel, 1, NULL, &global_item_size, NULL, 0, NULL, NULL);
    clEnqueueReadBuffer(state->queue, d_out, CL_TRUE, 0, 32 * num_seeds * num_addresses, out_privs, 0, NULL, NULL);
    
    clReleaseMemObject(d_seeds);
    clReleaseMemObject(d_out);
}


// Batch version of derive_seed_opencl for performance
void derive_seed_batch_opencl(OpenCLState *state, char **mnemonics, int num_keys, uint8_t *seeds_out) {
    size_t global_item_size = num_keys;
 // Or query device for better layout, but 1 is safe

    uint64_t *pass_data = calloc(num_keys * 32, sizeof(uint64_t)); // 256 bytes per key
    uint32_t *pass_lens = calloc(num_keys, sizeof(uint32_t));
    uint64_t *salt_data = calloc(num_keys * 32, sizeof(uint64_t)); // 256 bytes per key
    uint32_t *salt_lens = calloc(num_keys, sizeof(uint32_t));
    
    for (int i = 0; i < num_keys; i++) {
        pass_lens[i] = strlen(mnemonics[i]);
        memcpy(pass_data + i * 32, mnemonics[i], pass_lens[i]);
        
        const char *salt_base = "mnemonic";
        salt_lens[i] = strlen(salt_base);
        memcpy(salt_data + i * 32, salt_base, salt_lens[i]);
    }
    

    cl_mem d_passwords = clCreateBuffer(state->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, num_keys * 256, pass_data, NULL);
    cl_mem d_pass_lens = clCreateBuffer(state->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, num_keys * 4, pass_lens, NULL);
    cl_mem d_salts = clCreateBuffer(state->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, num_keys * 256, salt_data, NULL);
    cl_mem d_salt_lens = clCreateBuffer(state->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, num_keys * 4, salt_lens, NULL);
    cl_mem d_out = clCreateBuffer(state->ctx, CL_MEM_WRITE_ONLY, 64 * num_keys, NULL, NULL);
    
    uint32_t iterations = 2048;

    clSetKernelArg(state->pbkdf2_kernel, 0, sizeof(cl_mem), &d_passwords);
    clSetKernelArg(state->pbkdf2_kernel, 1, sizeof(cl_mem), &d_pass_lens);
    clSetKernelArg(state->pbkdf2_kernel, 2, sizeof(cl_mem), &d_salts);
    clSetKernelArg(state->pbkdf2_kernel, 3, sizeof(cl_mem), &d_salt_lens);
    clSetKernelArg(state->pbkdf2_kernel, 4, sizeof(uint32_t), &iterations);
    clSetKernelArg(state->pbkdf2_kernel, 5, sizeof(cl_mem), &d_out);

    clEnqueueNDRangeKernel(state->queue, state->pbkdf2_kernel, 1, NULL, &global_item_size, NULL, 0, NULL, NULL);
    clEnqueueReadBuffer(state->queue, d_out, CL_TRUE, 0, 64 * num_keys, seeds_out, 0, NULL, NULL);

    clReleaseMemObject(d_passwords);
    clReleaseMemObject(d_pass_lens);
    clReleaseMemObject(d_salts);
    clReleaseMemObject(d_salt_lens);
    clReleaseMemObject(d_out);
    
    free(pass_data);
    free(pass_lens);
    free(salt_data);
    free(salt_lens);
}

// Batch PBKDF2 with custom per-key salts (e.g. "mnemonicTREZOR")
void derive_seed_batch_opencl_with_salt(OpenCLState *state, char **mnemonics, char **salts_str, int num_keys, uint8_t *seeds_out) {
    size_t global_item_size = num_keys;

    uint64_t *pass_data = calloc(num_keys * 32, sizeof(uint64_t));
    uint32_t *pass_lens = calloc(num_keys, sizeof(uint32_t));
    uint64_t *salt_data = calloc(num_keys * 32, sizeof(uint64_t));
    uint32_t *salt_lens = calloc(num_keys, sizeof(uint32_t));
    
    for (int i = 0; i < num_keys; i++) {
        pass_lens[i] = strlen(mnemonics[i]);
        memcpy(pass_data + i * 32, mnemonics[i], pass_lens[i]);
        
        salt_lens[i] = strlen(salts_str[i]);
        memcpy(salt_data + i * 32, salts_str[i], salt_lens[i]);
    }

    cl_mem d_passwords = clCreateBuffer(state->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, num_keys * 256, pass_data, NULL);
    cl_mem d_pass_lens = clCreateBuffer(state->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, num_keys * 4, pass_lens, NULL);
    cl_mem d_salts = clCreateBuffer(state->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, num_keys * 256, salt_data, NULL);
    cl_mem d_salt_lens = clCreateBuffer(state->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, num_keys * 4, salt_lens, NULL);
    cl_mem d_out = clCreateBuffer(state->ctx, CL_MEM_WRITE_ONLY, 64 * num_keys, NULL, NULL);
    
    uint32_t iterations = 2048;

    clSetKernelArg(state->pbkdf2_kernel, 0, sizeof(cl_mem), &d_passwords);
    clSetKernelArg(state->pbkdf2_kernel, 1, sizeof(cl_mem), &d_pass_lens);
    clSetKernelArg(state->pbkdf2_kernel, 2, sizeof(cl_mem), &d_salts);
    clSetKernelArg(state->pbkdf2_kernel, 3, sizeof(cl_mem), &d_salt_lens);
    clSetKernelArg(state->pbkdf2_kernel, 4, sizeof(uint32_t), &iterations);
    clSetKernelArg(state->pbkdf2_kernel, 5, sizeof(cl_mem), &d_out);

    clEnqueueNDRangeKernel(state->queue, state->pbkdf2_kernel, 1, NULL, &global_item_size, NULL, 0, NULL, NULL);
    clEnqueueReadBuffer(state->queue, d_out, CL_TRUE, 0, 64 * num_keys, seeds_out, 0, NULL, NULL);

    clReleaseMemObject(d_passwords);
    clReleaseMemObject(d_pass_lens);
    clReleaseMemObject(d_salts);
    clReleaseMemObject(d_salt_lens);
    clReleaseMemObject(d_out);
    
    free(pass_data);
    free(pass_lens);
    free(salt_data);
    free(salt_lens);
}
#endif
