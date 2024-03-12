#include "stdio.h"
#include "stdlib.h"

#include "assert.h"

#include "cublas_v2.h"
#include "cuda_runtime.h"
#include <iostream>
#define checkCudaErrors(func)                                                  \
  {                                                                            \
    cudaError_t e = func;                                                      \
    if (e != cudaSuccess) {                                                    \
      printf("%s , %d CUDA: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    }                                                                          \
  }

// transfer float4
#define Fetch_Float4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])
#define Offset(row, col, k) ((row) * (k) + (col))
template <
    const int Block_Size_N, /*height of block of C that each thread block
                               calculate*/
    const int Block_Size_K, /*width of block of A that each thread block load
                         into shared memory*/

    const int Block_Size_M,  /*width of block of C that each thread block
                                calculate*/
    const int Thread_Size_X, /*height of block of C that each thread calculate*/
    const int Thread_Size_Y, /* width of block of C that each thread calculate*/
    const bool double_buffer /*whether enable double buffering or not*/>

__global__ void Gemm(float *__restrict__ A, float *__restrict__ B,
                     float *__restrict__ C, const int N, const int M,
                     const int K) {
  // block index
  int bx = blockIdx.x, by = blockIdx.y;
  // thread index
  int tx = threadIdx.x, ty = threadIdx.y;

  int row = by * blockDim.y + ty, col = bx * blockDim.x + tx;
  __shared__ float As[Block_Size_N][Block_Size_K];
  __shared__ float Bs[Block_Size_K][Block_Size_M];
  float tmp = 0.0;
  for (int i = 0; i < K; i += Block_Size_K) {
    As[ty][tx] = A[Offset(row, tx + i, K)];
    Bs[ty][tx] = B[Offset(ty + i, col, M)];
    __syncthreads();
    for (int j = 0; j < Block_Size_K; ++j) {
      tmp += As[ty][j] * Bs[j][tx];
    }
  }
  C[Offset(row, col, M)] = tmp;
}

int main(int argc, char **argv) {
  if (argc != 4) {
    printf("usage: ./main [N] [K] [M]");
    exit(0);
  }
  size_t N = atoi(argv[1]), K = atoi(argv[2]), M = atoi(argv[3]);

  float *h_a = new float[N * K], *h_b = new float[K * M],
        *h_c = new float[N * M], *h_c1 = new float[N * M];

  float *d_a, *d_b, *d_c;
  size_t bytes_A = sizeof(float) * N * K, bytes_B = sizeof(float) * M * K,
         bytes_C = sizeof(float) * N * M;
  checkCudaErrors(cudaMalloc(&d_a, sizeof(float) * N * K));
  checkCudaErrors(cudaMalloc(&d_b, sizeof(float) * M * K));
  checkCudaErrors(cudaMalloc(&d_c, sizeof(float) * N * M));

  const int Block_Size_N = 32, Block_Size_M = 32, Block_Size_K = 32,
            Thread_Size_X = 8, Thread_Size_Y = 8;
  const bool Enable_Double_Buffer = false;
  // generate A
  for (int i = 0; i < N * K; ++i) {
    h_a[i] = i;
  }
  // generate B
  for (int i = 0; i < M * K; ++i) {
    h_b[i] = i;
  }
  checkCudaErrors(cudaMemcpy(d_a, h_a, bytes_A, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_b, h_b, bytes_B, cudaMemcpyHostToDevice));

  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));
  float msecTotal = 0;
  int nIter = 1000;

  checkCudaErrors(cudaMemcpy(d_c, h_c, bytes_C, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaEventRecord(start));

  for (int run = 0; run < nIter; ++run) {
    dim3 dimBlock(Block_Size_N, Block_Size_M);
    dim3 dimGrid(N / Block_Size_N, M / Block_Size_M);
    Gemm<Block_Size_N, Block_Size_K, Block_Size_M, Thread_Size_X, Thread_Size_Y,
         Enable_Double_Buffer><<<dimGrid, dimBlock>>>(d_a, d_b, d_c, N, M, K);
  }

  checkCudaErrors(cudaEventRecord(stop));
  checkCudaErrors(cudaEventSynchronize(stop));
  checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
  checkCudaErrors(cudaMemcpy(h_c, d_c, bytes_C, cudaMemcpyDeviceToHost));
  // puts("generate A");
  // // generate A
  // for (int i = 0; i < N * K; ++i) {
  //   if (i % K == 0)
  //     puts("");
  //   std::cout << h_a[i] << " ";
  // }
  // puts("\ngenerate B");
  // // generate B
  // for (int i = 0; i < M * K; ++i) {
  //   if (i % M == 0)
  //     puts("");
  //   std::cout << h_b[i] << " ";
  // }

  // puts("\nNative Gemm output = \n");
  // for (int i = 0; i < N; ++i) {
  //   for (int j = 0; j < M; ++j) {
  //     double ans = 0;
  //     for (int k = 0; k < K; ++k) {
  //       ans += h_a[i * K + k] * h_b[k * M + j];
  //     }
  //     printf("%.0f ", ans);
  //   }
  //   puts("");
  // }
  // puts("\n MyGemm output = ");
  // for (int j = 0; j < N; ++j) {
  //   for (int k = 0; k < M; ++k) {
  //     printf("%.0f ", h_c[j * M + k]);
  //   }
  //   printf("\n");
  // }

  double msecPerMatrixMul[2] = {0, 0};
  double gigaFlops[2] = {0, 0};
  double flopsPerMatrixMul = 2.0 * M * N * K;
  msecPerMatrixMul[0] = msecTotal / nIter;
  gigaFlops[0] =
      (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul[0] / 1000.0f);
  printf(
      "My gemm Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
      gigaFlops[0], msecPerMatrixMul[0], flopsPerMatrixMul);

  // cublas
  cublasHandle_t blas_handle;
  cublasCreate(&blas_handle);
  float alpha = 1.0;
  float beta = 0;
  checkCudaErrors(cudaMemcpy(d_c, h_c, bytes_C, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaEventRecord(start));
  for (int run = 0; run < nIter; run++) {
    cublasSgemm(blas_handle, CUBLAS_OP_T, CUBLAS_OP_T, N, M, K, &alpha, d_a, K,
                d_b, M, &beta, d_c, N);
  }
  checkCudaErrors(cudaEventRecord(stop));
  checkCudaErrors(cudaEventSynchronize(stop));
  checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

  checkCudaErrors(cudaMemcpy(h_c1, d_c, bytes_C, cudaMemcpyDeviceToHost));
  // puts("cublas");
  // for (int j = 0; j < N; ++j) {
  //   for (int k = 0; k < M; ++k) {
  //     //   // printf("%d %d" , j , k) ;
  //     printf("%.0f ", h_c1[j * M + k]);
  //     // C[j][k] = As[k][j] ;
  //   }
  //   printf("\n");
  // }
  msecPerMatrixMul[1] = msecTotal / nIter;
  gigaFlops[1] =
      (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul[1] / 1000.0f);
  printf("CuBlas Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
         gigaFlops[1], msecPerMatrixMul[1], flopsPerMatrixMul);

  cublasDestroy(blas_handle);

  double eps = 1.e-6; // machine zero
  bool correct = true;
  for (int i = 0; i < N * M; i++) {
    int row = i / M;
    int col = i % M;
    double abs_err = fabs(h_c[i] - h_c1[col * N + row]);
    double dot_length = M;
    double abs_val = fabs(h_c[i]);
    double rel_err = abs_err / abs_val / dot_length;
    if (rel_err > eps) {
      printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n", i,
             h_c[i], h_c1[col * N + row], eps);
      correct = false;
      break;
    }
  }

  printf("%s\n", correct ? "Result= PASS" : "Result= FAIL");
  printf("ratio= %f\n", gigaFlops[0] / gigaFlops[1]);

  delete h_a, h_b, h_c, h_c1;
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}