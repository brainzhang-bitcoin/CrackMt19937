# 编译器设置
CC = gcc
CXX = g++
NVCC = nvcc

# 基础 C 编译选项
CFLAGS = -Wall -O3 -march=native -fopenmp
LDFLAGS = -lOpenCL -lcrypto

# --- 关键修改：NVCC 编译选项 ---
# 1. -ccbin: 明确告诉 nvcc 使用 g++-10 避免找不到 cc1plus
# 2. -cudart static: 静态链接 CUDA 运行时，这样目标机器即使没装 CUDA Toolkit 也能跑
# 3. -gencode: 同时生成 sm_35 (GT720) 和 sm_86 (RTX3060) 的机器码，并保留虚拟 PTX 实现前向兼容
NVCCFLAGS = -O3 -std=c++14 \
    -ccbin $(CXX) \
    -cudart static \
    -Wno-deprecated-gpu-targets \
    -gencode arch=compute_35,code=sm_35 \
    -gencode arch=compute_86,code=sm_86 \
    -gencode arch=compute_86,code=compute_86

all: mt19937_opencl mt19937_cuda

mt19937_opencl: mt19937_opencl.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

mt19937_cuda: mt19937_cuda.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@ -lcrypto -Xcompiler -fopenmp

clean:
	rm -f mt19937_opencl mt19937_cuda 

.PHONY: all clean
