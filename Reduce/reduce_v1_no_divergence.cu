#include "stdio.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>
#define N 32 * 1024 * 1024
#define BLOCK_SIZE 256

__global__ void reduce_v1_no_divergence(float *g_idata, float *g_odata) {
  __shared__ float sdata[BLOCK_SIZE];

  // each thread loads one element from global to shared mem
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  sdata[tid] = g_idata[i];
  __syncthreads();

  // do reduction in shared mem
  for (unsigned int s = 1; s < blockDim.x; s *= 2) {
    int index = 2 * s * tid;
    if (index < blockDim.x) {
      sdata[index] += sdata[index + s];
    }
    __syncthreads();
  }

  // write result for this block to global mem
  if (tid == 0)
    g_odata[blockIdx.x] = sdata[0];
}

bool check(float *output_host) {
  for (int i = 0; i < N / BLOCK_SIZE; ++i) {
    if (output_host[i] != BLOCK_SIZE * 2.0) {
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

  int32_t block_num = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
  float *output_host = (float *)malloc((N / BLOCK_SIZE) * sizeof(float));
  float *output_device;
  cudaMalloc((void **)&output_device, (N / BLOCK_SIZE) * sizeof(float));

  dim3 grid(N / BLOCK_SIZE, 1);
  dim3 block(BLOCK_SIZE, 1);
  cudaEvent_t start, stop;
  float msecTotal = 0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  int nIter = 1000;
  for (int i = 1; i <= nIter; ++i) {
    reduce_v1_no_divergence<<<grid, block>>>(input_device, output_device);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&msecTotal, start, stop);
  cudaMemcpy(output_host, output_device, block_num * sizeof(float),
             cudaMemcpyDeviceToHost);
  msecTotal /= nIter;
  if (check(output_host)) {
    // bandwidth maybe error
    printf("reduce_v1_no_divergence , use time = %.3fms , and bandwidth = "
           "%.3fGB/s\n",
           msecTotal, N * sizeof(float) * 1e-9 / (msecTotal * 1e-3f));
  } else {
    printf("reduce_v1_no_divergence , this is error\n");
  }

  return 0;
}