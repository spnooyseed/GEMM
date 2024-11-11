#include "stdio.h"
#include <math_constants.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>
#include <cudnn.h>
#include <iostream>
#define N 32 * 12 * 108
#define M 1024
#define BLOCK_SIZE 1024
#define Fetch_Float4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])

const int packed_size = 4;
const int kWarpSize = 32 ;

template<typename T>
struct SumOp{
    __device__ __forceinline__ T operator()(const T&a , const T&b){
        return a + b ;
    }
};

template<typename T>
struct MaxOp{
    __device__ __forceinline__ T operator()(const T&a , const T&b) {
        return max(a , b);
    }
};

template<template<typename> class ReduceOp , typename T>
__inline__ __device__ T warpReduce(T val){
    for (int i = kWarpSize / 2 ; i > 0 ; i /= 2) {
        val = ReduceOp<T>()(val , __shfl_xor_sync(0xffffffff,val , i)) ;
    }
    return val ;
}

template<const int n , const int m>
__global__ void softmax_v0(float *g_idata, float *g_odata) {
    __shared__ float tmp[n][m] ;
    int tx = threadIdx.x , ty = threadIdx.y ;
    int bx = blockIdx.x;
    g_idata = &g_idata[bx * blockDim.y * m];
    g_odata = &g_odata[bx * blockDim.y * m];

    float threadMax = -10000.0 ;
#pragma unroll
    for (int i = 0 ; i < m / 32 / packed_size ; ++ i) {
       Fetch_Float4(tmp[ty][tx * packed_size + i * 32 * packed_size]) = Fetch_Float4(g_idata[ty * m + tx * packed_size + i * 32 * packed_size]) ;
       for (int j = 0 ; j < packed_size ; ++ j){
          threadMax = max(threadMax ,tmp[ty][tx * packed_size + i * 32 * packed_size + j]);
       }
    }
    __syncthreads() ;
    threadMax = warpReduce<MaxOp , float>(threadMax) ;
    float threadSum = 0. ;
#pragma unroll
    for (int i = 0 ; i < m / 32 ; ++ i) {
        float t = exp(tmp[ty][tx + i * 32] - threadMax);
        threadSum += t ;
        tmp[ty][tx + i * 32] = t ;
    }
    __syncthreads();
    threadSum = warpReduce<SumOp , float>(threadSum) ;
#pragma unroll
    for (int i = 0 ; i < m / 32 ; ++ i) {
         tmp[ty][tx + i * 32] /= threadSum ;
    }
#pragma unroll
    for (int i = 0 ; i < m / 32 / packed_size ; ++ i) {
        Fetch_Float4(g_odata[ty * m + tx * packed_size + i * 32 * packed_size]) =
        Fetch_Float4(tmp[ty][tx * packed_size + i * 32 * packed_size]) ;
    }
    return ;
}

template<const int n , const int m>
__global__ void softmax_v1_no_bank(float *g_idata, float *g_odata) {
    __shared__ float tmp[n][m] ;
    int tx = threadIdx.x , ty = threadIdx.y ;
    int bx = blockIdx.x;
    g_idata = &g_idata[bx * blockDim.y * m];
    g_odata = &g_odata[bx * blockDim.y * m];

    float threadMax = -10000.0 ;
#pragma unroll
    for (int i = 0 ; i < m / 32 / packed_size ; ++ i) {
       float t[packed_size] ;
       Fetch_Float4(t) = Fetch_Float4(g_idata[ty * m + tx * packed_size + i * 32 * packed_size]) ;
#pragma unroll
       for (int j = 0 ; j < packed_size ; ++ j){
          tmp[ty][tx + i * 32 * packed_size + j * 32] = t[j] ;
          threadMax = max(threadMax , t[j]);//,tmp[ty][tx * packed_size + i * 32 * packed_size + j]);
       }
    }
    __syncthreads() ;
    threadMax = warpReduce<MaxOp , float>(threadMax) ;
    float threadSum = 0. ;
#pragma unroll
    for (int i = 0 ; i < m / 32 ; ++ i) {
        float t = exp(tmp[ty][tx + i * 32] - threadMax);
        threadSum += t ;
        tmp[ty][tx + i * 32] = t ;
    }
    __syncthreads();
    threadSum = warpReduce<SumOp , float>(threadSum) ;
#pragma unroll
    for (int i = 0 ; i < m / 32 ; ++ i) {
         tmp[ty][tx + i * 32] /= threadSum ;
    }
#pragma unroll
    for (int i = 0 ; i < m / 32 / packed_size ; ++ i) {
        float t[packed_size];
#pragma unroll
        for (int j = 0 ; j < packed_size ; ++ j) {
            t[j] = tmp[ty][tx + i * 32 * packed_size + j * 32] ;
        }
        Fetch_Float4(g_odata[ty * m + tx * packed_size + i * 32 * packed_size]) = Fetch_Float4(t) ;
    }
    return ;
}

// 检查 CUDA 错误
#define CHECK_CUDA(call) \
  { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
      std::cerr << "Error: " << __FILE__ << ":" << __LINE__ << ", " \
                << cudaGetErrorString(error) << std::endl; \
      exit(1); \
    } \
  }

// 检查 cuDNN 错误
#define CHECK_CUDNN(call) \
  { \
    const cudnnStatus_t status = call; \
    if (status != CUDNN_STATUS_SUCCESS) { \
      std::cerr << "Error: " << __FILE__ << ":" << __LINE__ << ", " \
                << cudnnGetErrorString(status) << std::endl; \
      exit(1); \
    } \
  }

void softmax_cudnn(float* input, float* output, int n, int m) {

  cudnnHandle_t cudnn;
  CHECK_CUDNN(cudnnCreate(&cudnn));

  cudnnTensorDescriptor_t input_desc;
  cudnnTensorDescriptor_t output_desc;
  CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc));
  CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_desc));

  int dims[4] = {n, m, 1, 1}; // N x M matrix
  int strides[4] = {m, 1, 1, 1}; // Strides for N x M matrix
  CHECK_CUDNN(cudnnSetTensorNdDescriptor(input_desc, CUDNN_DATA_FLOAT, 4, dims, strides));
  CHECK_CUDNN(cudnnSetTensorNdDescriptor(output_desc, CUDNN_DATA_FLOAT, 4, dims, strides));

  float* d_input;
  float* d_output;
  CHECK_CUDA(cudaMalloc(&d_input, n * m * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_output, n * m * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_input, input, n * m * sizeof(float), cudaMemcpyHostToDevice));

  float alpha = 1.0f;
  float beta = 0.0f;
  cudaEvent_t start, stop;
  float msecTotal = 0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  int nIter = 1000;
  for (int i = 0 ; i < nIter;  ++ i)
    CHECK_CUDNN(cudnnSoftmaxForward(cudnn, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
                                  &alpha, input_desc, d_input, &beta, output_desc, d_output));
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&msecTotal, start, stop);
  msecTotal /= nIter;
  printf("softmax_cudnn , use time = %.3fms , and bandwidth = %.3fGB/s\n",
           msecTotal,2 * n * m * sizeof(float) * 1e-9 / (msecTotal * 1e-3f));
  CHECK_CUDA(cudaMemcpy(output, d_output, n * m * sizeof(float), cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(d_input));
  CHECK_CUDA(cudaFree(d_output));
  CHECK_CUDNN(cudnnDestroyTensorDescriptor(input_desc));
  CHECK_CUDNN(cudnnDestroyTensorDescriptor(output_desc));
  CHECK_CUDNN(cudnnDestroy(cudnn));

}

bool check(float* a , float* b) {
    for (int i = 0 ; i < N * M ; ++ i) {
        if (abs(a[i] - b[i]) > 0.00001) {
           std::cout << a[i] << " " << b[i] << "\n";
           return false ;
        }
    }
    return true ;
}


int main() {
  float *input_host = (float *)malloc(N * M * sizeof(float));
  float *input_device;
  cudaMalloc((void **)&input_device, N * M * sizeof(float));
  for (int i = 0; i < N * M ; i++)
    input_host[i] = i % 49 - 25;
  cudaMemcpy(input_device, input_host, N * M  * sizeof(float),
             cudaMemcpyHostToDevice);

  float *output_host = (float *)malloc((N * M) * sizeof(float));
  float *output_host_cudnn = (float *)malloc((N * M) * sizeof(float));
  float *output_device;
  cudaMalloc((void **)&output_device, N * M * sizeof(float));
  softmax_cudnn(input_host, output_host_cudnn, N , M) ;
  const int num_rows = 8 ;
  dim3 grid(N / num_rows, 1);
  dim3 block(32 , num_rows, 1);
  cudaEvent_t start, stop;
  float msecTotal = 0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  int nIter = 1000;
  for (int i = 1; i <= nIter; ++i) {
    softmax_v0<num_rows , M><<<grid, block>>>(input_device, output_device);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&msecTotal, start, stop);
  msecTotal /= nIter;
  cudaMemcpy(output_host, output_device, N * M * sizeof(float),
             cudaMemcpyDeviceToHost);
  if (check(output_host , output_host_cudnn)) {
    // bandwidth maybe error
    printf("softmax_v0_base , use time = %.3fms , and bandwidth = %.3fGB/s\n",
           msecTotal, 2 * N * M * sizeof(float) * 1e-9 / (msecTotal * 1e-3f));
  } else {
    printf("softmax_v0_base , this is error\n");
  }
  cudaEventRecord(start);

  for (int i = 1; i <= nIter; ++i) {
    softmax_v1_no_bank<num_rows , M><<<grid, block>>>(input_device, output_device);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&msecTotal, start, stop);
  msecTotal /= nIter;
  cudaMemcpy(output_host, output_device, N * M * sizeof(float),
             cudaMemcpyDeviceToHost);
  if (check(output_host , output_host_cudnn)) {
    // bandwidth maybe error
    printf("softmax_v1_no_bank , use time = %.3fms , and bandwidth = %.3fGB/s\n",
           msecTotal, 2 * N * M * sizeof(float) * 1e-9 / (msecTotal * 1e-3f));
  } else {
    printf("softmax_v1_no_bank , this is error\n");
  }

  return 0;
}