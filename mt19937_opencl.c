#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#include <getopt.h>
#include <time.h>
#include <omp.h>
#include <openssl/hmac.h>
#include <openssl/evp.h>

#include "ocl_helper.h"
#include "bip39.h"
#include "cl_pipeline.h"

#define MT_N 624
#define MT_M 397
#define MT_A 0x9908B0DF
#define MT_U 11
#define MT_D 0xFFFFFFFF
#define MT_S 7
#define MT_B 0x9D2C5680
#define MT_T 15
#define MT_C 0xEFC60000
#define MT_L 18
#define MT_F 1812433253
#define MT_LOWER_MASK 0x7FFFFFFF
#define MT_UPPER_MASK 0x80000000

#define REGION_START 0
#define REGION_END 4294967295
#define REGION_CHUNK 5

typedef struct {
    uint32_t MT[MT_N];
    int index;
} CppMT19937;

void eprint_ts() {
    time_t now = time(NULL);
    struct tm *tm_info = localtime(&now);
    char buffer[26];
    strftime(buffer, 26, "%Y-%m-%d %H:%M:%S", tm_info);
    fprintf(stderr, "[%s] ", buffer);
}

void print_hex(const uint8_t *data, size_t len) {
    for (size_t i = 0; i < len; i++) {
        printf("%02x", data[i]);
    }
    printf("\n");
    fflush(stdout);
}

void eprint_hex(const uint8_t *data, size_t len) {
    for (size_t i = 0; i < len; i++) {
        fprintf(stderr, "%02x", data[i]);
    }
}

void mt19937_seed(CppMT19937 *rng, uint32_t seed) {
    rng->MT[0] = seed;
    for (int i = 1; i < MT_N; i++) {
        uint32_t temp = MT_F * (rng->MT[i - 1] ^ (rng->MT[i - 1] >> 30)) + i;
        rng->MT[i] = temp;
    }
    rng->index = MT_N;
}

void mt19937_twist(CppMT19937 *rng) {
    for (int i = 0; i < MT_N; i++) {
        uint32_t x = (rng->MT[i] & MT_UPPER_MASK) + (rng->MT[(i + 1) % MT_N] & MT_LOWER_MASK);
        uint32_t xA = x >> 1;
        if ((x % 2) != 0) {
            xA = xA ^ MT_A;
        }
        rng->MT[i] = rng->MT[(i + MT_M) % MT_N] ^ xA;
    }
    rng->index = 0;
}

uint32_t mt19937_extract_number(CppMT19937 *rng) {
    if (rng->index >= MT_N) {
        mt19937_twist(rng);
    }
    uint32_t y = rng->MT[rng->index];
    y = y ^ ((y >> MT_U) & MT_D);
    y = y ^ ((y << MT_S) & MT_B);
    y = y ^ ((y << MT_T) & MT_C);
    y = y ^ (y >> MT_L);
    rng->index++;
    return y;
}

void generate_entropy(uint32_t seed, int keys, uint8_t *out_entropies) {
    CppMT19937 rng;
    mt19937_seed(&rng, seed);
    for (int k = 0; k < keys; k++) {
        for (int i = 0; i < 32; i++) {
            uint32_t val = mt19937_extract_number(&rng);
            out_entropies[k * 32 + i] = (val >> 24) & 0xFF;
        }
    }
}

#define MAX_REGIONS 64

struct {
    char mode[32];
    int regions[MAX_REGIONS];
    int num_regions;
    int maxkeys;
    uint32_t entropy;
    bool has_entropy;
    bool has_region;
    bool gpu;
    int threads;
    int pipeline_opt;  // 0=disable, 1-3=NVIDIA opt levels
} opts;

void parse_regions(const char *arg) {
    // Supports: "0", "0,1,2", "0-9", "0,3-5,8"
    char buf[256];
    strncpy(buf, arg, sizeof(buf)-1);
    buf[sizeof(buf)-1] = '\0';
    opts.num_regions = 0;
    char *token = strtok(buf, ",");
    while (token && opts.num_regions < MAX_REGIONS) {
        char *dash = strchr(token, '-');
        if (dash) {
            *dash = '\0';
            int from = atoi(token);
            int to = atoi(dash + 1);
            for (int r = from; r <= to && opts.num_regions < MAX_REGIONS; r++) {
                opts.regions[opts.num_regions++] = r;
            }
        } else {
            opts.regions[opts.num_regions++] = atoi(token);
        }
        token = strtok(NULL, ",");
    }
}

void print_help() {
    printf("Usage:\n");
    printf("  mt19937 [--mode=<mode>] [--region=<regions>] [--maxkeys=<maxkeys>] [--entropy=<entropy>] [--gpu] [--threads=<n>] [--pipeline-opt=<0-3>]\n");
    printf("  Regions: single (0), list (0,1,2), range (0-9), mixed (0,3-5,8)\n");
    printf("  Threads: number of CPU threads for PBKDF2 (default: all cores)\n");
    printf("  mt19937 -h | --help\n");
}

void ec_new(const uint8_t *entropy, uint8_t *priv_out) {
    uint8_t hmac_out[64];
    HMAC(EVP_sha512(), "Bitcoin seed", 12, entropy, 32, hmac_out, NULL);
    memcpy(priv_out, hmac_out, 32);
}

void replay_mode() {
    uint8_t *entropies = malloc(32 * opts.maxkeys);
    generate_entropy(opts.entropy, opts.maxkeys, entropies);

    OpenCLState cl_state;
    init_opencl(&cl_state, opts.pipeline_opt);

    // Exact output formatting from replay_mode in python
    for(int i=0; i<opts.maxkeys; i++) {
        eprint_ts();
        fprintf(stderr, "CppMT19937 DEBUG: entropy_key:%u, mtrandom:", opts.entropy);
        eprint_hex(entropies + i * 32, 32);
        fprintf(stderr, "\n");

        uint8_t priv[32];
        ec_new(entropies + i * 32, priv);
        print_hex(priv, 32);

        eprint_ts();
        fprintf(stderr, "priv mode:%u,", opts.entropy);
        eprint_hex(entropies + i * 32, 32);
        fprintf(stderr, ",");
        eprint_hex(priv, 32);
        fprintf(stderr, "\n");



        uint8_t seed[64];


        eprint_ts();
        fprintf(stderr, "BIP39 entropy:");
        eprint_hex(entropies + i * 32, 32);
        char mnemonic[512] = {0};
        entropy_to_mnemonic(entropies + i * 32, mnemonic);
        fprintf(stderr, ", mnemonic:%s, seed:", mnemonic);
        eprint_hex(seed, 64);
        fprintf(stderr, "\n");

        uint8_t child_priv[32];

        // bip44
        derive_bip32_opencl(&cl_state, cl_state.bip44_kernel, seed, 1, 1, child_priv);
        print_hex(child_priv, 32);
        eprint_ts();
        fprintf(stderr, "bip44 mode:%u,", opts.entropy);
        eprint_hex(seed, 64);
        fprintf(stderr, ",");
        eprint_hex(child_priv, 32);
        fprintf(stderr, ",<not-impl-addr>\n");

        // bip49
        derive_bip32_opencl(&cl_state, cl_state.bip49_kernel, seed, 1, 1, child_priv);
        print_hex(child_priv, 32);
        eprint_ts();
        fprintf(stderr, "bip49 mode:%u,", opts.entropy);
        eprint_hex(seed, 64);
        fprintf(stderr, ",");
        eprint_hex(child_priv, 32);
        fprintf(stderr, ",<not-impl-addr>\n");

        // bip84
        derive_bip32_opencl(&cl_state, cl_state.bip84_kernel, seed, 1, 1, child_priv);
        print_hex(child_priv, 32);
        eprint_ts();
        fprintf(stderr, "bip84 mode:%u,", opts.entropy);
        eprint_hex(seed, 64);
        fprintf(stderr, ",");
        eprint_hex(child_priv, 32);
        fprintf(stderr, ",<not-impl-addr>\n");
    }
}

void get_region_range(int region_id, uint32_t *out_start, uint32_t *out_end) {
    uint32_t region_chunk_count = (REGION_END - REGION_START) / REGION_CHUNK;
    *out_start = REGION_START + region_chunk_count * region_id;
    *out_end = *out_start + region_chunk_count;
}

void run_priv_mode() {
    uint32_t start_i = REGION_START;
    uint32_t end_i = REGION_END;
    if (opts.has_region) {
        get_region_range(opts.regions[0], &start_i, &end_i);
    }

    uint8_t *entropies = malloc(32 * opts.maxkeys);
    for (uint32_t i = start_i; i < end_i; i++) {
        generate_entropy(i, opts.maxkeys, entropies);
        for(int k=0; k<opts.maxkeys; k++) {
            uint8_t priv[32];
            ec_new(entropies + k * 32, priv);
            print_hex(priv, 32);
            if (i % 10000 == 0) {
                eprint_ts();
                fprintf(stderr, "%u ", i);
                eprint_hex(priv, 32);
                fprintf(stderr, "\n");
            }
        }
    }
    free(entropies);
}

void run_bip_gpu_mode_range(OpenCLState *cl_state, cl_kernel kernel, const char *mode_name, uint32_t start_i, uint32_t end_i) {

    int BATCH_SIZE = 10000 / opts.maxkeys; if (BATCH_SIZE == 0) BATCH_SIZE = 1;
    uint8_t *entropies = malloc(32 * BATCH_SIZE * opts.maxkeys);
    uint32_t *seeds_origin = malloc(4 * BATCH_SIZE);

    uint8_t *seeds_64 = malloc(64 * BATCH_SIZE * opts.maxkeys);
    uint8_t *privs_out = malloc(32 * BATCH_SIZE * opts.maxkeys);

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    int64_t total_processed = 0;

    int batch_count = 0;
    for (uint32_t i = start_i; i < end_i; i++) {
        generate_entropy(i, opts.maxkeys, entropies + batch_count * opts.maxkeys * 32);
        seeds_origin[batch_count] = i;
        batch_count++;

        if (batch_count >= BATCH_SIZE) {
            int total_keys = batch_count * opts.maxkeys;

            // PBKDF2 (GPU)
            char **mnemonics_arr = malloc(sizeof(char*) * total_keys);
            for(int j=0; j<total_keys; j++) {
                mnemonics_arr[j] = malloc(512);
                memset(mnemonics_arr[j], 0, 512);
                entropy_to_mnemonic(entropies + j * 32, mnemonics_arr[j]);
            }
            derive_seed_batch_opencl(cl_state, mnemonics_arr, total_keys, seeds_64);
            for(int j=0; j<total_keys; j++) free(mnemonics_arr[j]);
            free(mnemonics_arr);

            // BIP32 Pipeline (GPU)
            derive_bip32_opencl(cl_state, kernel, seeds_64, total_keys, 1, privs_out);

            for(int j=0; j<total_keys; j++) {
                print_hex(privs_out + j * 32, 32);
            }

            if (seeds_origin[0] % 10000 == 0) {
                 eprint_ts();
                 fprintf(stderr, "%u ", seeds_origin[0]);
                 eprint_hex(seeds_64, 64);
                 fprintf(stderr, " ");
                 eprint_hex(privs_out, 32);
                 fprintf(stderr, "\n");
            }

            total_processed += total_keys;
            clock_gettime(CLOCK_MONOTONIC, &end);
            double elapsed = (end.tv_sec - start.tv_sec) + 1e-9 * (end.tv_nsec - start.tv_nsec);
            fprintf(stderr, "[%s GPU] Processed %ld items (%.0f items/s)\n", mode_name, total_processed, total_processed / elapsed);

            batch_count = 0;
        }
    }

    // Process remaining
    if (batch_count > 0) {
        int total_keys = batch_count * opts.maxkeys;
        // PBKDF2 (GPU) - remaining
        char **mnemonics_arr = malloc(sizeof(char*) * total_keys);
        for(int j=0; j<total_keys; j++) {
            mnemonics_arr[j] = malloc(512);
            memset(mnemonics_arr[j], 0, 512);
            entropy_to_mnemonic(entropies + j * 32, mnemonics_arr[j]);
        }
        derive_seed_batch_opencl(cl_state, mnemonics_arr, total_keys, seeds_64);
        for(int j=0; j<total_keys; j++) free(mnemonics_arr[j]);
        free(mnemonics_arr);

        derive_bip32_opencl(cl_state, kernel, seeds_64, total_keys, 1, privs_out);
        for(int j=0; j<total_keys; j++) {
            print_hex(privs_out + j * 32, 32);
        }
        total_processed += total_keys;
        clock_gettime(CLOCK_MONOTONIC, &end);
        double elapsed = (end.tv_sec - start.tv_sec) + 1e-9 * (end.tv_nsec - start.tv_nsec);
        fprintf(stderr, "[%s GPU] Processed %ld items (%.0f items/s)\n", mode_name, total_processed, total_processed / elapsed);
    }

    free(entropies);
    free(seeds_origin);
    free(seeds_64);
    free(privs_out);
}

void perf_gpu_mode() {
    eprint_ts();
    fprintf(stderr, "Running GPU comparison...\n");
    OpenCLState cl_state;
    init_opencl(&cl_state, opts.pipeline_opt);

    int num_seeds = 100;
    int maxkeys = 100;
    int total_keys = num_seeds * maxkeys; // 10,000 requests

    uint8_t *entropies = malloc(32 * total_keys);
    uint8_t *seeds_64 = malloc(64 * total_keys);
    uint8_t *privs_out = malloc(32 * total_keys);

    int idx = 0;
    for(int i = 0; i < num_seeds; i++) {
        generate_entropy(REGION_START + i, maxkeys, entropies + idx * 32);
        idx += maxkeys;
    }

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    char **mnemonics_arr = malloc(sizeof(char*) * total_keys);
    for(int j=0; j<total_keys; j++) {
        mnemonics_arr[j] = malloc(512);
        memset(mnemonics_arr[j], 0, 512);
        entropy_to_mnemonic(entropies + j * 32, mnemonics_arr[j]);
    }
    derive_seed_batch_opencl(&cl_state, mnemonics_arr, total_keys, seeds_64);
    for(int j=0; j<total_keys; j++) free(mnemonics_arr[j]);
    free(mnemonics_arr);

    derive_bip32_opencl(&cl_state, cl_state.bip49_kernel, seeds_64, total_keys, 1, privs_out);

    clock_gettime(CLOCK_MONOTONIC, &end);
    double gpu_t = (end.tv_sec - start.tv_sec) + 1e-9 * (end.tv_nsec - start.tv_nsec);

    eprint_ts();
    fprintf(stderr, "GPU Time: %.2fs\n", gpu_t);
    // CPU fallback for comparison can be added if needed, but GPU is the focus.

    free(entropies);
    free(seeds_64);
    free(privs_out);
}

int main(int argc, char **argv) {
    strcpy(opts.mode, "priv");
    opts.maxkeys = 1;
    opts.has_entropy = false;
    opts.has_region = false;
    opts.gpu = false;
    opts.threads = 0;  // 0 = use all cores
    opts.pipeline_opt = -1;  // -1=auto-detect (4090→opt-disable, 3060→full-opt)

    struct option long_options[] = {
        {"help", no_argument, 0, 'h'},
        {"version", no_argument, 0, 'v'},
        {"mode", required_argument, 0, 'm'},
        {"region", required_argument, 0, 'r'},
        {"maxkeys", required_argument, 0, 'k'},
        {"entropy", required_argument, 0, 'e'},
        {"gpu", no_argument, 0, 'g'},
        {"threads", required_argument, 0, 't'},
        {"pipeline-opt", required_argument, 0, 'p'},
        {0, 0, 0, 0}
    };

    int c;
    while ((c = getopt_long(argc, argv, "hvp:", long_options, NULL)) != -1) {
        switch (c) {
            case 'h': print_help(); return 0;
            case 'v': printf("mt19937 C version 1.2\n"); return 0;
            case 'm': strncpy(opts.mode, optarg, sizeof(opts.mode)-1); break;
            case 'r': parse_regions(optarg); opts.has_region = true; break;
            case 'k': opts.maxkeys = atoi(optarg); break;
            case 'e': opts.entropy = strtoul(optarg, NULL, 10); opts.has_entropy = true; break;
            case 'g': opts.gpu = true; break;
            case 't': opts.threads = atoi(optarg); break;
            case 'p': opts.pipeline_opt = atoi(optarg); break;
            default: print_help(); return 1;
        }
    }

    if (opts.threads > 0) {
        omp_set_num_threads(opts.threads);
    }
    fprintf(stderr, "[CPU] Using %d threads for PBKDF2\n", omp_get_max_threads());

    OpenCLState cl_state;
    if (opts.gpu) {
        init_opencl(&cl_state, opts.pipeline_opt);
        eprint_ts();
        fprintf(stderr, "[GPU] OpenCL acceleration enabled\n");
    }

    if (strcmp(opts.mode, "priv") == 0) {
        run_priv_mode();
    } else if (strcmp(opts.mode, "bip44") == 0 ||
               strcmp(opts.mode, "bip49") == 0 ||
               strcmp(opts.mode, "bip84") == 0) {
        if (opts.gpu) {
            cl_kernel kernel;
            const char *mode_label;
            if (strcmp(opts.mode, "bip44") == 0) {
                kernel = cl_state.bip44_kernel; mode_label = "BIP44";
            } else if (strcmp(opts.mode, "bip49") == 0) {
                kernel = cl_state.bip49_kernel; mode_label = "BIP49";
            } else {
                kernel = cl_state.bip84_kernel; mode_label = "BIP84";
            }

            if (opts.has_region) {
                for (int ri = 0; ri < opts.num_regions; ri++) {
                    uint32_t s, e;
                    get_region_range(opts.regions[ri], &s, &e);
                    eprint_ts();
                    fprintf(stderr, "[%s GPU] Starting region %d (seeds %u-%u)\n", mode_label, opts.regions[ri], s, e-1);
                    run_bip_gpu_mode_range(&cl_state, kernel, mode_label, s, e);
                }
            } else {
                run_bip_gpu_mode_range(&cl_state, kernel, mode_label, REGION_START, REGION_END);
            }
        }
    } else if (strcmp(opts.mode, "replay") == 0) {
        if (opts.has_entropy) {
            replay_mode();
        }
    } else if (strcmp(opts.mode, "perfgpu") == 0) {
        perf_gpu_mode();
    } else {
        print_help();
    }

    return 0;
}
