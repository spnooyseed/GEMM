#include "stdio.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>
#define N 32 * 1024 * 1024
#define BLOCK_SIZE 256

__device__ void warpReduce(volatile float *cache, unsigned int tid) {
  cache[tid] += cache[tid + 32];
  cache[tid] += cache[tid + 16];
  cache[tid] += cache[tid + 8];
  cache[tid] += cache[tid + 4];
  cache[tid] += cache[tid + 2];
  cache[tid] += cache[tid + 1];
}

__global__ void reduce_v5_unroll_warp(float *g_idata, float *g_odata) {
  __shared__ float sdata[BLOCK_SIZE];

  // each thread loads one element from global to shared mem
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
  sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];
  __syncthreads();

  // do reduction in shared mem
  if (BLOCK_SIZE >= 512) {
    if (tid < 256) {
      sdata[tid] += sdata[tid + 256];
    }
    __syncthreads();
  }
  if (BLOCK_SIZE >= 256) {
    if (tid < 128) {
      sdata[tid] += sdata[tid + 128];
    }
    __syncthreads();
  }
  if (BLOCK_SIZE >= 128) {
    if (tid < 64) {
      sdata[tid] += sdata[tid + 64];
    }
    __syncthreads();
  }

  if (tid < 32) {
    warpReduce(sdata, tid);
  }

  // write result for this block to global mem
  if (tid == 0)
    g_odata[blockIdx.x] = sdata[0];
}

bool check(float *output_host, float *res, int n) {
  for (int i = 0; i < n; ++i) {
    if (output_host[i] != res[i]) {
      return false;
    }
  }
  return true;
}

int main() {
  float *input_host = (float *)malloc(N * sizeof(float));
  float *input_device;
  cudaMalloc((void **)&input_device, N * sizeof(float));
  for (int i = 0; i < N; i++)
    input_host[i] = 2.0;
  cudaMemcpy(input_device, input_host, N * sizeof(float),
             cudaMemcpyHostToDevice);

  int num_per_block = 2 * BLOCK_SIZE;
  int block_num = N / num_per_block;

  float *output_host = (float *)malloc(block_num * sizeof(float));
  float *res = (float *)malloc(block_num * sizeof(float));
  float *output_device;
  cudaMalloc((void **)&output_device, block_num * sizeof(float));

  dim3 grid(block_num, 1);
  dim3 block(BLOCK_SIZE, 1);
  cudaEvent_t start, stop;
  float msecTotal = 0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  int nIter = 1000;
  for (int i = 1; i <= nIter; ++i) {
    reduce_v5_unroll_warp<<<grid, block>>>(input_device, output_device);
  }

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&msecTotal, start, stop);
  cudaMemcpy(output_host, output_device, block_num * sizeof(float),
             cudaMemcpyDeviceToHost);
  msecTotal /= nIter;
  for (int i = 0; i < block_num; ++i) {
    for (int j = 0; j < num_per_block; ++j) {
      res[i] += input_host[i * num_per_block + j];
    }
  }
  if (check(output_host, res, block_num)) {
    // bandwidth maybe error
    printf("reduce_v5_unroll_warp , use time = %.3fms , and bandwidth = "
           "%.3fGB/s\n",
           msecTotal, N * sizeof(float) * 1e-9 / (msecTotal * 1e-3f));
  } else {
    printf("reduce_v5_unroll_warp , this is error\n");
    for (int i = 0; i < block_num; ++i) {
      if (output_host[i] != res[i])
        printf("output_host[%d] = %.0f , res[%d] = %.0f\n", i, output_host[i],
               i, res[i]);
    }
  }

  return 0;
}