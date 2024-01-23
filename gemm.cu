#include "stdio.h"
#include "stdlib.h"

#include "assert.h"

#include "cublas_v2.h"
#include "cuda_runtime.h"

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

  // X、Y threads number in block
  const int thread_num_x_per_block = Block_Size_N / Thread_Size_X;
  const int thread_num_y_per_block = Block_Size_M / Thread_Size_Y;
  const int thread_num_per_block =
      thread_num_x_per_block * thread_num_y_per_block;

  // thread Id in cur Block
  // const int tid = ty * thread_num_y_per_block + tx;
  const int tid = tx * thread_num_y_per_block + ty;

  // shared memory , A_shared , B_shared
  __shared__ float As[2][Block_Size_K][Block_Size_N];
  __shared__ float Bs[2][Block_Size_K][Block_Size_M];

  // register for C
  float accum[Thread_Size_X][Thread_Size_Y] = {0};
  // register for A , B
  float reg_a[2][Thread_Size_X], reg_b[2][Thread_Size_Y];

  // resgister for load global memory
  const int load_num_A =
      Block_Size_N * Block_Size_K / (thread_num_per_block * 4);
  const int load_num_B =
      Block_Size_K * Block_Size_M / (thread_num_per_block * 4);

  float load_A_reg[4 * load_num_A], load_B_reg[4 * load_num_B];

  // threads num in one row
  const int A_Tile_Thread_Per_Row = Block_Size_K / 4,
            B_Tile_Thread_Per_Row = Block_Size_M / 4;

  const int A_Tile_Row = tid / A_Tile_Thread_Per_Row,
            B_Tile_Row = tid / B_Tile_Thread_Per_Row;
  const int A_Tile_Col = tid % A_Tile_Thread_Per_Row * 4;
  const int B_Tile_Col = tid % B_Tile_Thread_Per_Row * 4;

  // row stride that thread uses to load multiple rows of a tile
  const int A_Tile_Row_Stride = thread_num_per_block / A_Tile_Thread_Per_Row;
  const int B_Tile_Row_Stride = thread_num_per_block / B_Tile_Thread_Per_Row;

  A = &A[Block_Size_N * bx * K];
  B = &B[Block_Size_M * by];
// printf("A_Tile_Row_Stride = %d , __LINE__ = %d\n" , A_Tile_Row_Stride , __LINE__) ;
// load A from global mem to shared mem
#pragma unroll
  for (int i = 0; i < Block_Size_N; i += A_Tile_Row_Stride) {
    int load_indx = i / A_Tile_Row_Stride * 4;
    Fetch_Float4(load_A_reg[load_indx]) =
        Fetch_Float4(A[Offset(A_Tile_Row + i, A_Tile_Col, K)]);

    As[0][A_Tile_Col][A_Tile_Row + i] = load_A_reg[load_indx];
    As[0][A_Tile_Col + 1][A_Tile_Row + i] = load_A_reg[load_indx + 1];
    As[0][A_Tile_Col + 1][A_Tile_Row + i] = load_A_reg[load_indx + 2];
    As[0][A_Tile_Col + 1][A_Tile_Row + i] = load_A_reg[load_indx + 3];
  }

  // load B from global mem to shared mem
#pragma unroll
  for (int i = 0; i < Block_Size_K; i += B_Tile_Row_Stride) {
    Fetch_Float4(Bs[0][B_Tile_Row + i][B_Tile_Col]) =
        Fetch_Float4(B[Offset(B_Tile_Row + i, B_Tile_Col, M)]);
  }
  __syncthreads();

  // load A from shared mem to register

#pragma unroll
  for (int thread_x = 0; thread_x < Thread_Size_X; thread_x += 4) {
    Fetch_Float4(reg_a[0][thread_x]) =
        Fetch_Float4(As[0][0][Thread_Size_X * tx + thread_x]);
  }
#pragma unroll
  for (int thread_y = 0; thread_y < Thread_Size_Y; thread_y += 4) {
    Fetch_Float4(reg_b[0][thread_y]) =
        Fetch_Float4(Bs[0][0][Thread_Size_Y * ty + thread_y]);
  }

  int write_stage_idx = 1;

  for (int tile_idx = Block_Size_K; tile_idx < K; tile_idx += Block_Size_K) {
#pragma unroll
    for (int i = 0; i < Block_Size_N; i += A_Tile_Row_Stride) {
      int load_idx = i / A_Tile_Row_Stride * 4;
      Fetch_Float4(load_A_reg[load_idx]) =
          Fetch_Float4(A[Offset(A_Tile_Row + i, A_Tile_Col + tile_idx, K)]);
    }

#pragma unroll
    for (int i = 0; i < Block_Size_K; i += B_Tile_Row_Stride) {
      int load_idx = i / B_Tile_Row_Stride * 4;
      Fetch_Float4(load_B_reg[load_idx]) =
          Fetch_Float4(B[Offset(tile_idx + B_Tile_Row + i, B_Tile_Col, M)]);
    }

    int load_stage_idx = write_stage_idx ^ 1;

#pragma unroll
    for (int j = 1; j < Block_Size_K; ++j) {
#pragma unroll
      // load A from shared memory to register
      for (int thread_x = 0; thread_x < Thread_Size_X; thread_x += 4) {
        Fetch_Float4(reg_a[j % 2][thread_x]) =
            Fetch_Float4(As[load_stage_idx][j][Thread_Size_X * tx + thread_x]);
      }

#pragma unroll
      // load B from shared memory to register
      for (int thread_y = 0; thread_y < Thread_Size_Y; thread_y += 4) {
        Fetch_Float4(reg_b[j % 2][thread_y]) =
            Fetch_Float4(Bs[load_stage_idx][j][Thread_Size_Y * ty + thread_y]);
      }

#pragma unroll
      // computer C = A * B , in thread , use register
      for (int thread_x = 0; thread_x < Thread_Size_X; ++thread_x) {
      #pragma unroll
          for (int thread_y = 0; thread_y < Thread_Size_Y; ++thread_y) {
          accum[thread_x][thread_y] +=
              reg_a[(j + 1) % 2][thread_x] * reg_b[(j + 1) % 2][thread_y];
        }
      }

#pragma unroll
      for (int i = 0; i < Block_Size_N; i += A_Tile_Row_Stride) {
        int load_index = i / A_Tile_Row_Stride * 4;
        As[write_stage_idx][A_Tile_Col][A_Tile_Row + i] =
            load_A_reg[load_index];
        As[write_stage_idx][A_Tile_Col + 1][A_Tile_Row + i] =
            load_A_reg[load_index + 1];
        As[write_stage_idx][A_Tile_Col + 2][A_Tile_Row + i] =
            load_A_reg[load_index + 2];
        As[write_stage_idx][A_Tile_Col + 3][A_Tile_Row + i] =
            load_A_reg[load_index + 3];
      }
#pragma unroll
      // load B from register to shared memory
      for (int i = 0; i < Block_Size_K; i += B_Tile_Row_Stride) {
        int load_index = i / B_Tile_Row_Stride * 4;
        Fetch_Float4(Bs[write_stage_idx][B_Tile_Row + i][B_Tile_Col]) =
            Fetch_Float4(load_B_reg[load_index]);
      }

      __syncthreads();
      write_stage_idx ^= 1;
    }

#pragma unroll
    // load A from shared memory to register
    for (int thread_x = 0; thread_x < Thread_Size_X; thread_x += 4) {
      Fetch_Float4(reg_a[0][thread_x]) = Fetch_Float4(
          As[load_stage_idx ^ 1][0][Thread_Size_X * tx + thread_x]);
    }

#pragma unroll
    // load B from shared memory to register
    for (int thread_y = 0; thread_y < Thread_Size_Y; thread_y += 4) {
      Fetch_Float4(reg_b[0][thread_y]) = Fetch_Float4(
          Bs[load_stage_idx ^ 1][0][Thread_Size_Y * ty + thread_y]);
    }

#pragma unroll
    // computer C = A * B , in thread , use register
    for (int thread_x = 0; thread_x < Thread_Size_X; ++thread_x) {
      #pragma unroll
      for (int thread_y = 0; thread_y < Thread_Size_Y; ++thread_y) {
        accum[thread_x][thread_y] += reg_a[1][thread_x] * reg_b[1][thread_y];
      }
    }
  }

#pragma unroll
for (int thread_x = 0; thread_x < Thread_Size_X; thread_x += 4) {
#pragma unroll
      for (int thread_y = 0; thread_y < Thread_Size_Y; ++thread_y) {
      Fetch_Float4(
          C[Offset(Block_Size_N * bx + tx * Thread_Size_X + thread_x,
                   Block_Size_M * by + ty * Thread_Size_Y + thread_y, M)]) =
          Fetch_Float4(accum[thread_x][thread_y]);
    }
  }
}

int main(int argc, char **argv) {
  if (argc != 4) {
    printf("usage: ./main [N] [K] [M]");
    exit(0);
  }
  size_t N = atoi(argv[1]), K = atoi(argv[2]), M = atoi(argv[3]);

  assert(M % 8 == 0);
  assert(N % 8 == 0);
  assert(K % 8 == 0);

  float *h_a = new float[N * K], *h_b = new float[K * M],
        *h_c = new float[N * M], *h_c1 = new float[N * M];

  float *d_a, *d_b, *d_c;
  size_t bytes_A = sizeof(float) * N * K, bytes_B = sizeof(float) * M * K,
         bytes_C = sizeof(float) * N * M;
  checkCudaErrors(cudaMalloc(&d_a, sizeof(float) * N * K));
  checkCudaErrors(cudaMalloc(&d_b, sizeof(float) * M * K));
  checkCudaErrors(cudaMalloc(&d_c, sizeof(float) * N * M));

  const int Block_Size_N = 128, Block_Size_M = 128, Block_size_K = 8,
            Thread_Size_X = 8, Thread_Size_Y = 8;
  const bool Enable_Double_Buffer = false;

  // generate A
  for (int i = 0; i < N * K; ++i) {
    h_a[i] = i / 13;
  }

  // generate B
  for (int i = 0; i < M * K; ++i) {
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

  for (int run = 0; run < nIter; ++run) {
    dim3 dimBlock(Block_Size_N / Thread_Size_X, Block_Size_M / Thread_Size_Y);
    dim3 dimGrid(N / Block_Size_N, M / Block_Size_M);
    Gemm<Block_Size_N, Block_size_K, Block_Size_M, Thread_Size_X, Thread_Size_Y,
         Enable_Double_Buffer><<<dimGrid, dimBlock>>>(d_a, d_b, d_c, N, M, K);
  }

  checkCudaErrors(cudaEventRecord(stop));
  checkCudaErrors(cudaEventSynchronize(stop));
  checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
  checkCudaErrors(cudaMemcpy(h_c, d_c, bytes_C, cudaMemcpyDeviceToHost));

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
    cublasSgemm(blas_handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, &alpha, d_a, K,
                d_b, N, &beta, d_c, N);
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
  for (int i = 0; i < M * N; i++) {
    int row = i / N;
    int col = i % N;
    double abs_err = fabs(h_c[i] - h_c1[col * M + row]);
    double dot_length = M;
    double abs_val = fabs(h_c[i]);
    double rel_err = abs_err / abs_val / dot_length;
    if (rel_err > eps) {
      printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n", i,
             h_c[i], h_c1[col * M + row], eps);
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