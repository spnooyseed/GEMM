// optimize sgemm

#include "assert.h"
#include <stdio.h>
#include <stdlib.h>

// CUDA runtime
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define checkCudaErrors(func)                                                  \
  {                                                                            \
    cudaError_t e = (func);                                                    \
    if (e != cudaSuccess)                                                      \
      printf("%s %d CUDA: %s\n", __FILE__, __LINE__, cudaGetErrorString(e));   \
  }

// cal offset from row col and ld , in row-major matrix, ld is the width of the
// matrix
#define Offset(row, col, k) ((row) * (k) + (col))

// transfer float4
#define Fetch_Float4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])

// K: ldA
// M: ldB
template <
    const int Block_Size_N,  // height of block of C that each thread block
                             // calculate
    const int Block_Size_K,  // width of block of A that each thread block load
                             // into shared memory
    const int Block_Size_M,  // width of block of C that each thread block
                             // calculate
    const int Thread_Size_X, // height of block of C that each thread calculate
    const int Thread_Size_Y, // width of block of C that each thread calculate
    const bool double_buffer // whether enable double buffering or not
    >
__global__ void Gemm(float *__restrict__ A, float *__restrict__ B,
                     float *__restrict__ C, const int N, const int M,
                     const int K) {
  // Block index
  int bx = blockIdx.x;
  int by = blockIdx.y;

  // Thread index
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // the threads number in Block of X,Y
  const int thread_num_x_per_block = Block_Size_N / Thread_Size_X;
  const int thread_num_y_per_block = Block_Size_M / Thread_Size_Y;
  const int thread_num_per_block =
      thread_num_y_per_block * thread_num_x_per_block;

  // thread id in cur Block
  const int tid = ty * thread_num_y_per_block + tx;

  // shared memory
  __shared__ float As[2][Block_Size_K][Block_Size_N];
  __shared__ float Bs[2][Block_Size_K][Block_Size_M];
  // registers for C
  float accum[Thread_Size_X][Thread_Size_Y] = {0};
  // registers for A and B
  float frag_a[2][Thread_Size_X];
  float frag_b[2][Thread_Size_Y];
  // registers load global memory
  const int load_num_A =
      Block_Size_N * Block_Size_K / (thread_num_per_block * 4);
  const int load_num_B =
      Block_Size_K * Block_Size_M / (thread_num_per_block * 4);
  float load_A_reg[4 * load_num_A];
  float load_B_reg[4 * load_num_B];

  // threads number in one row
  const int A_Tile_Thread_Per_Row = Block_Size_K / 4;
  const int B_Tile_Thread_Per_Row = Block_Size_M / 4;

  // row number and col number that needs to be loaded by this thread
  const int A_Tile_Row = tid / A_Tile_Thread_Per_Row;
  const int B_Tile_Row = tid / B_Tile_Thread_Per_Row;

  const int A_Tile_Col = tid % A_Tile_Thread_Per_Row * 4;
  const int B_Tile_Col = tid % B_Tile_Thread_Per_Row * 4;

  // row stride that thread uses to load multiple rows of a tile
  const int A_Tile_Row_Stride = thread_num_per_block / A_Tile_Thread_Per_Row;
  const int B_Tile_Row_Stride = thread_num_per_block / B_Tile_Thread_Per_Row;

  A = &A[(Block_Size_N * by) * K];
  B = &B[Block_Size_M * bx];

// transfer first tile from global mem to shared mem
//  load A from global memory to shared memory
#pragma unroll
  for (int i = 0; i < Block_Size_N; i += A_Tile_Row_Stride) {
    int ldg_index = i / A_Tile_Row_Stride * 4;
    Fetch_Float4(load_A_reg[ldg_index]) =
        Fetch_Float4(A[Offset(A_Tile_Row + i, // row
                              A_Tile_Col,     // col
                              K)]);
    As[0][A_Tile_Col][A_Tile_Row + i] = load_A_reg[ldg_index];
    As[0][A_Tile_Col + 1][A_Tile_Row + i] = load_A_reg[ldg_index + 1];
    As[0][A_Tile_Col + 2][A_Tile_Row + i] = load_A_reg[ldg_index + 2];
    As[0][A_Tile_Col + 3][A_Tile_Row + i] = load_A_reg[ldg_index + 3];
  }
// load B from global memory to shared memory
#pragma unroll
  for (int i = 0; i < Block_Size_K; i += B_Tile_Row_Stride) {
    Fetch_Float4(Bs[0][B_Tile_Row + i][B_Tile_Col]) =
        Fetch_Float4(B[Offset(B_Tile_Row + i, // row
                              B_Tile_Col,     // col
                              M)]);
  }
  __syncthreads();
// load A from shared memory to register
#pragma unroll
  for (int thread_y = 0; thread_y < Thread_Size_X; thread_y += 4) {
    Fetch_Float4(frag_a[0][thread_y]) =
        Fetch_Float4(As[0][0][Thread_Size_X * ty + thread_y]);
  }
// load B from shared memory to register
#pragma unroll
  for (int thread_x = 0; thread_x < Thread_Size_Y; thread_x += 4) {
    Fetch_Float4(frag_b[0][thread_x]) =
        Fetch_Float4(Bs[0][0][Thread_Size_Y * tx + thread_x]);
  }

  int write_stage_idx = 1;
  int tile_idx = 0;
  do {
    tile_idx += Block_Size_K;
    // load next tile from global mem
    if (tile_idx < K) {
#pragma unroll
      for (int i = 0; i < Block_Size_N; i += A_Tile_Row_Stride) {
        int ldg_index = i / A_Tile_Row_Stride * 4;
        Fetch_Float4(load_A_reg[ldg_index]) =
            Fetch_Float4(A[Offset(A_Tile_Row + i,        // row
                                  A_Tile_Col + tile_idx, // col
                                  K)]);
      }
#pragma unroll
      for (int i = 0; i < Block_Size_K; i += B_Tile_Row_Stride) {
        int ldg_index = i / B_Tile_Row_Stride * 4;
        Fetch_Float4(load_B_reg[ldg_index]) =
            Fetch_Float4(B[Offset(tile_idx + B_Tile_Row + i, // row
                                  B_Tile_Col,                // col
                                  M)]);
      }
    }

    int load_stage_idx = write_stage_idx ^ 1;

#pragma unroll
    for (int j = 0; j < Block_Size_K - 1; ++j) {
// load next tile from shared mem to register
// load A from shared memory to register
#pragma unroll
      for (int thread_y = 0; thread_y < Thread_Size_X; thread_y += 4) {
        Fetch_Float4(frag_a[(j + 1) % 2][thread_y]) = Fetch_Float4(
            As[load_stage_idx][j + 1][Thread_Size_X * ty + thread_y]);
      }
// load B from shared memory to register
#pragma unroll
      for (int thread_x = 0; thread_x < Thread_Size_Y; thread_x += 4) {
        Fetch_Float4(frag_b[(j + 1) % 2][thread_x]) = Fetch_Float4(
            Bs[load_stage_idx][j + 1][Thread_Size_Y * tx + thread_x]);
      }
// compute C Thread_Size_Y x Thread_Size_X
#pragma unroll
      for (int thread_y = 0; thread_y < Thread_Size_X; ++thread_y) {
#pragma unroll
        for (int thread_x = 0; thread_x < Thread_Size_Y; ++thread_x) {
          accum[thread_y][thread_x] +=
              frag_a[j % 2][thread_y] * frag_b[j % 2][thread_x];
        }
      }
    }

    if (tile_idx < K) {
#pragma unroll
      for (int i = 0; i < Block_Size_N; i += A_Tile_Row_Stride) {
        int ldg_index = i / A_Tile_Row_Stride * 4;
        As[write_stage_idx][A_Tile_Col][A_Tile_Row + i] = load_A_reg[ldg_index];
        As[write_stage_idx][A_Tile_Col + 1][A_Tile_Row + i] =
            load_A_reg[ldg_index + 1];
        As[write_stage_idx][A_Tile_Col + 2][A_Tile_Row + i] =
            load_A_reg[ldg_index + 2];
        As[write_stage_idx][A_Tile_Col + 3][A_Tile_Row + i] =
            load_A_reg[ldg_index + 3];
      }
// load B from global memory to shared memory
#pragma unroll
      for (int i = 0; i < Block_Size_K; i += B_Tile_Row_Stride) {
        int ldg_index = i / B_Tile_Row_Stride * 4;
        Fetch_Float4(Bs[write_stage_idx][B_Tile_Row + i][B_Tile_Col]) =
            Fetch_Float4(load_B_reg[ldg_index]);
      }
      // use double buffer, only need one sync
      __syncthreads();
      // switch
      write_stage_idx ^= 1;
    }

// load first tile from shared mem to register of next iter
// load A from shared memory to register
#pragma unroll
    for (int thread_y = 0; thread_y < Thread_Size_X; thread_y += 4) {
      Fetch_Float4(frag_a[0][thread_y]) = Fetch_Float4(
          As[load_stage_idx ^ 1][0][Thread_Size_X * ty + thread_y]);
    }
// load B from shared memory to register
#pragma unroll
    for (int thread_x = 0; thread_x < Thread_Size_Y; thread_x += 4) {
      Fetch_Float4(frag_b[0][thread_x]) = Fetch_Float4(
          Bs[load_stage_idx ^ 1][0][Thread_Size_Y * tx + thread_x]);
    }
// compute last tile mma Thread_Size_Y x Thread_Size_X
#pragma unroll
    for (int thread_y = 0; thread_y < Thread_Size_X; ++thread_y) {
#pragma unroll
      for (int thread_x = 0; thread_x < Thread_Size_Y; ++thread_x) {
        accum[thread_y][thread_x] += frag_a[1][thread_y] * frag_b[1][thread_x];
      }
    }
  } while (tile_idx < K);

// store back to C
#pragma unroll
  for (int thread_y = 0; thread_y < Thread_Size_X; ++thread_y) {
#pragma unroll
    for (int thread_x = 0; thread_x < Thread_Size_Y; thread_x += 4) {
      Fetch_Float4(
          C[Offset(Block_Size_N * by + ty * Thread_Size_X + thread_y,
                   Block_Size_M * bx + tx * Thread_Size_Y + thread_x, M)]) =
          Fetch_Float4(accum[thread_y][thread_x]);
    }
  }
}

int main(int argc, char **argv) {
  if (argc != 4) {
    printf("usage: ./main [N] [K] [M]\n");
    exit(0);
  }
  size_t N = atoi(argv[1]);
  size_t K = atoi(argv[2]);
  size_t M = atoi(argv[3]);

  assert(N % 8 == 0);
  assert(M % 8 == 0);
  assert(K % 8 == 0);

  size_t bytes_A = sizeof(float) * N * K;
  size_t bytes_B = sizeof(float) * K * M;
  size_t bytes_C = sizeof(float) * N * M;
  float *h_a = (float *)malloc(bytes_A);
  float *h_b = (float *)malloc(bytes_B);
  float *h_c = (float *)malloc(bytes_C);
  float *h_c1 = (float *)malloc(bytes_C);

  float *d_a;
  float *d_b;
  float *d_c;

  checkCudaErrors(cudaMalloc(&d_a, bytes_A));
  checkCudaErrors(cudaMalloc(&d_b, bytes_B));
  checkCudaErrors(cudaMalloc(&d_c, bytes_C));
  double msecPerMatrixMul[2] = {0, 0};
  double gigaFlops[2] = {0, 0};
  double flopsPerMatrixMul = 2.0 * N * M * K;

  const int Block_Size_N = 128;
  const int Block_Size_K = 8;
  const int Block_Size_M = 128;
  const int Thread_Size_Y = 8;
  const int Thread_Size_X = 8;
  const bool Enable_Double_Buffer = false;

  // generate A
  for (int i = 0; i < N * K; i++) {
    h_a[i] = i / 13;
  }

  // generate B
  for (int i = 0; i < K * M; i++) {
    h_b[i] = i % 13;
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
  for (int run = 0; run < nIter; run++) {
    dim3 dimBlock(Block_Size_M / Thread_Size_Y, Block_Size_N / Thread_Size_X);
    dim3 dimGrid(M / Block_Size_M, N / Block_Size_N);
    Gemm<Block_Size_N, Block_Size_K, Block_Size_M, Thread_Size_X, Thread_Size_Y,
         Enable_Double_Buffer><<<dimGrid, dimBlock>>>(d_a, d_b, d_c, N, M, K);
  }
  checkCudaErrors(cudaEventRecord(stop));
  checkCudaErrors(cudaEventSynchronize(stop));
  checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

  checkCudaErrors(cudaMemcpy(h_c, d_c, bytes_C, cudaMemcpyDeviceToHost));

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
                d_b, M, &beta, d_c, M);
  }
  checkCudaErrors(cudaEventRecord(stop));
  checkCudaErrors(cudaEventSynchronize(stop));
  checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

  checkCudaErrors(cudaMemcpy(h_c1, d_c, bytes_C, cudaMemcpyDeviceToHost));

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
    double dot_length = N;
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

  // Free Memory
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  free(h_a);
  free(h_b);
  free(h_c);
  free(h_c1);
}