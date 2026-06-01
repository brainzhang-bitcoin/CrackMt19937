#define CL_TARGET_OPENCL_VERSION 300
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS 1
#ifndef OCL_HELPER_H
#define OCL_HELPER_H

#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static const char* cl_error_string(cl_int err) {
    switch(err) {
        case 0: return "CL_SUCCESS";
        case -1: return "CL_DEVICE_NOT_FOUND";
        case -2: return "CL_DEVICE_NOT_AVAILABLE";
        case -3: return "CL_COMPILER_NOT_AVAILABLE";
        case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case -5: return "CL_OUT_OF_RESOURCES";
        case -6: return "CL_OUT_OF_HOST_MEMORY";
        case -11: return "CL_BUILD_PROGRAM_FAILURE";
        case -12: return "CL_MAP_FAILURE";
        case -30: return "CL_INVALID_VALUE";
        case -33: return "CL_INVALID_DEVICE";
        case -34: return "CL_INVALID_PLATFORM";
        case -44: return "CL_INVALID_PROGRAM";
        case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
        case -46: return "CL_INVALID_KERNEL_NAME";
        case -47: return "CL_INVALID_KERNEL_DEFINITION";
        case -48: return "CL_INVALID_KERNEL";
        default: {
            static char buf[32];
            snprintf(buf, sizeof(buf), "CL_ERROR_%d", err);
            return buf;
        }
    }
}

char* read_file(const char* filename) {
    FILE *f = fopen(filename, "r");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    rewind(f);
    char *buf = (char*)malloc(size + 1);
    if(fread(buf, 1, size, f) != size) { free(buf); fclose(f); return NULL; }
    buf[size] = '\0';
    fclose(f);
    return buf;
}

cl_context create_context(cl_device_id *device_out) {
    cl_platform_id platform;
    cl_uint num_platforms;
    cl_int err;

    fprintf(stderr, "[OCL] Step 1: clGetPlatformIDs...\n");
    err = clGetPlatformIDs(1, &platform, &num_platforms);
    fprintf(stderr, "[OCL] Step 1: done, err=%s, num_platforms=%u\n", cl_error_string(err), num_platforms);
    if(num_platforms == 0) { fprintf(stderr, "[OCL] FATAL: no platforms\n"); return NULL; }

    // Print platform info
    char platname[256] = {0};
    clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(platname), platname, NULL);
    char platver[256] = {0};
    clGetPlatformInfo(platform, CL_PLATFORM_VERSION, sizeof(platver), platver, NULL);
    fprintf(stderr, "[OCL] Platform: %s (%s)\n", platname, platver);

    cl_device_id device;
    cl_uint num_devices;

    fprintf(stderr, "[OCL] Step 2: clGetDeviceIDs (CL_DEVICE_TYPE_ALL)...\n");
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, &num_devices);
    fprintf(stderr, "[OCL] Step 2: done, err=%s, num_devices=%u\n", cl_error_string(err), num_devices);

    if(num_devices == 0) {
        fprintf(stderr, "[OCL] Step 2b: trying CL_DEVICE_TYPE_CPU...\n");
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, &num_devices);
        fprintf(stderr, "[OCL] Step 2b: done, err=%s, num_devices=%u\n", cl_error_string(err), num_devices);
        if(num_devices == 0) { fprintf(stderr, "[OCL] FATAL: no devices\n"); return NULL; }
    }

    // Print device info
    char devname[256] = {0};
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(devname), devname, NULL);
    cl_uint compute_units = 0;
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, NULL);
    cl_ulong global_mem = 0;
    clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem), &global_mem, NULL);
    fprintf(stderr, "[OCL] Device: %s (CU=%u, VRAM=%luMB)\n", devname, compute_units, (unsigned long)(global_mem/1024/1024));

    *device_out = device;

    fprintf(stderr, "[OCL] Step 3: clCreateContext...\n");
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    fprintf(stderr, "[OCL] Step 3: done, err=%s, ctx=%p\n", cl_error_string(err), (void*)context);
    return context;
}

cl_program build_program(cl_context context, cl_device_id device, const char *source_path) {
    char *source = read_file(source_path);
    if(!source) {
        fprintf(stderr, "Failed to read %s\n", source_path);
        return NULL;
    }
    
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source, NULL, NULL);
    cl_int err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    
    if (err != CL_SUCCESS) {
        size_t len;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
        char *buffer = (char*)malloc(len);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, len, buffer, NULL);
        fprintf(stderr, "CL Build Error: %s\n", buffer);
        free(buffer);
        exit(1);
    }
    free(source);
    return program;
}

cl_program build_program_from_source(cl_context context, cl_device_id device, const char *source, size_t source_len, const char *build_opts) {
    cl_int err;
    struct timespec t0, t1;

    fprintf(stderr, "[OCL] build_program_from_source: source_len=%zu opts=%s\n", source_len, build_opts ? build_opts : "(default)");
    fprintf(stderr, "[OCL]   clCreateProgramWithSource...\n");
    cl_program program = clCreateProgramWithSource(context, 1, &source, &source_len, &err);
    fprintf(stderr, "[OCL]   clCreateProgramWithSource: done, err=%s, prog=%p\n", cl_error_string(err), (void*)program);

    if (err != CL_SUCCESS) {
        fprintf(stderr, "[OCL] FATAL: clCreateProgramWithSource failed\n");
        exit(1);
    }

    fprintf(stderr, "[OCL]   clBuildProgram (this may take a while for large kernels)...\n");
    fflush(stderr);
    clock_gettime(CLOCK_MONOTONIC, &t0);
    err = clBuildProgram(program, 1, &device, build_opts, NULL, NULL);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec);
    fprintf(stderr, "[OCL]   clBuildProgram: done, err=%s (%.1fs)\n", cl_error_string(err), elapsed);
    
    if (err != CL_SUCCESS) {
        size_t len;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
        char *buffer = (char*)malloc(len);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, len, buffer, NULL);
        fprintf(stderr, "[OCL] CL Build Error (log %zu bytes):\n%s\n", len, buffer);
        free(buffer);
        exit(1);
    }
    return program;
}

#endif
