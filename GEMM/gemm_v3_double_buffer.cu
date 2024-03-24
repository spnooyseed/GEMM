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
      printf("gemm_v3_double_buffer , %s , %d CUDA: %s\n", __FILE__, __LINE__, \
             cudaGetErrorString(e));                                           \
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
    const int Thread_Size_Y, /*height of block of C that each thread calculate*/
    const int Thread_Size_X, /* width of block of C that each thread calculate*/
    const bool double_buffer /*whether enable double buffering or not*/>

__global__ void Gemm(float *__restrict__ A, float *__restrict__ B,
                     float *__restrict__ C, const int N, const int M,
                     const int K) {
  // block index
  int bx = blockIdx.x, by = blockIdx.y;
  // thread index
  int tx = threadIdx.x, ty = threadIdx.y;

  __shared__ float As[2][Block_Size_K][Block_Size_N];
  __shared__ float Bs[2][Block_Size_K][Block_Size_M];
  const int loadN = 4;

  A = &A[Offset(by * Block_Size_N, 0, K)];
  B = &B[Offset(0, bx * Block_Size_M, M)];

  const int thread_num_y_per_block = Block_Size_N / Thread_Size_Y,
            thread_num_x_per_block = Block_Size_M / Thread_Size_X;
  const int thread_num_per_block =
      thread_num_x_per_block * thread_num_y_per_block;
  const int tid = ty * thread_num_x_per_block + tx;

  float accum[Thread_Size_Y][Thread_Size_X] = {0};

  float reg_A[2][Thread_Size_Y], reg_B[2][Thread_Size_X];

  const int load_num_A = Block_Size_N * Block_Size_K / thread_num_per_block;
  const int load_num_B = Block_Size_M * Block_Size_K / thread_num_per_block;
  float load_A_reg[load_num_A], load_B_reg[load_num_B];

  const int A_Tile_Thread_Per_Row = Block_Size_K / loadN,
            B_Tile_Thread_Per_Row = Block_Size_M / loadN;

  const int A_Tile_Row = tid / A_Tile_Thread_Per_Row,
            B_Tile_Row = tid / B_Tile_Thread_Per_Row;
  const int A_Tile_Col = tid % A_Tile_Thread_Per_Row * loadN,
            B_Tile_Col = tid % B_Tile_Thread_Per_Row * loadN;
  const int A_Tile_Row_Stride = thread_num_per_block / A_Tile_Thread_Per_Row,
            B_Tile_Row_Stride = thread_num_per_block / B_Tile_Thread_Per_Row;
#pragma unroll
  // prefetch A data from global memory to shared memory
  for (int tile_idx_bn = 0; tile_idx_bn < Block_Size_N;
       tile_idx_bn += A_Tile_Row_Stride) {
    int index = tile_idx_bn / A_Tile_Row_Stride * loadN;
    Fetch_Float4(load_A_reg[index]) =
        Fetch_Float4(A[Offset(A_Tile_Row + tile_idx_bn, // row
                              A_Tile_Col,               // col
                              K)]);
#pragma unroll
    for (int load_N = 0; load_N < loadN; ++load_N) {
      As[0][A_Tile_Col + load_N][A_Tile_Row + tile_idx_bn] =
          load_A_reg[index + load_N];
    }
  }
#pragma unroll
  // prefetch B data from global memory to shared memory
  for (int tile_idx_bk = 0; tile_idx_bk < Block_Size_K;
       tile_idx_bk += B_Tile_Row_Stride) {
    Fetch_Float4(Bs[0][B_Tile_Row + tile_idx_bk][B_Tile_Col]) =
        Fetch_Float4(B[Offset(tile_idx_bk + B_Tile_Row, // row
                              B_Tile_Col,               // col
                              M)]);
  }
  __syncthreads();

// prefetch data from shared memory rigster
#pragma unroll
  for (int tile_idx_rn = 0; tile_idx_rn < Thread_Size_Y; tile_idx_rn += loadN) {
    Fetch_Float4(reg_A[0][tile_idx_rn]) =
        Fetch_Float4(As[0][0][Thread_Size_Y * ty + tile_idx_rn]);
  }
#pragma unroll
  // load B from shared to register
  for (int tile_idx_rm = 0; tile_idx_rm < Thread_Size_X; tile_idx_rm += loadN) {
    Fetch_Float4(reg_B[0][tile_idx_rm]) =
        Fetch_Float4(Bs[0][0][Thread_Size_X * tx + tile_idx_rm]);
  }

  int write_stage_idx = 1;
#pragma unroll
  for (int tile_idx_k = 0; tile_idx_k < K; tile_idx_k += Block_Size_K) {

    // load next tile A and B from global to register tmp
    if (tile_idx_k + Block_Size_K < K) {
#pragma unroll
      for (int tile_idx_bn = 0; tile_idx_bn < Block_Size_N;
           tile_idx_bn += A_Tile_Row_Stride) {
        int index = tile_idx_bn / A_Tile_Row_Stride * loadN;
        Fetch_Float4(load_A_reg[index]) =
            Fetch_Float4(A[Offset(A_Tile_Row + tile_idx_bn,               // row
                                  A_Tile_Col + tile_idx_k + Block_Size_K, // col
                                  K)]);
      }
// load B from global to shared
#pragma unroll
      for (int tile_idx_bk = 0; tile_idx_bk < Block_Size_K;
           tile_idx_bk += B_Tile_Row_Stride) {
        int index = tile_idx_bk / B_Tile_Row_Stride * loadN;
        Fetch_Float4(load_B_reg[index]) = Fetch_Float4(B[Offset(
            tile_idx_k + Block_Size_K + tile_idx_bk + B_Tile_Row, // row
            B_Tile_Col,                                           // col
            M)]);
      }
    }

    int load_stage_idx = write_stage_idx ^ 1;
#pragma unroll
    for (int tile_idx_bk = 0; tile_idx_bk < Block_Size_K; ++tile_idx_bk) {

      // prefetch next tile bk from As and Bs to register
      if (tile_idx_bk + 1 < Block_Size_K) {
#pragma unroll
        for (int tile_idx_rn = 0; tile_idx_rn < Thread_Size_Y;
             tile_idx_rn += loadN) {
          Fetch_Float4(reg_A[(tile_idx_bk + 1) % 2][tile_idx_rn]) =
              Fetch_Float4(As[load_stage_idx][tile_idx_bk + 1]
                             [Thread_Size_Y * ty + tile_idx_rn]);
        }
#pragma unroll
        for (int tile_idx_rm = 0; tile_idx_rm < Thread_Size_X;
             tile_idx_rm += loadN) {
          Fetch_Float4(reg_B[(tile_idx_bk + 1) % 2][tile_idx_rm]) =
              Fetch_Float4(Bs[load_stage_idx][tile_idx_bk + 1]
                             [Thread_Size_X * tx + tile_idx_rm]);
        }
      }

// compute this tile rn , rm
#pragma unroll
      for (int tile_idx_rn = 0; tile_idx_rn < Thread_Size_Y; ++tile_idx_rn) {
#pragma unroll
        for (int tile_idx_rm = 0; tile_idx_rm < Thread_Size_X; ++tile_idx_rm) {
          accum[tile_idx_rn][tile_idx_rm] +=
              reg_A[tile_idx_bk % 2][tile_idx_rn] *
              reg_B[tile_idx_bk % 2][tile_idx_rm];
        }
      }
    }

    // load next tile A and B from register tmp to shared memory
    if (tile_idx_k + Block_Size_K < K) {
#pragma unroll
      for (int tile_idx_bn = 0; tile_idx_bn < Block_Size_N;
           tile_idx_bn += A_Tile_Row_Stride) {
        int index = tile_idx_bn / A_Tile_Row_Stride * loadN;
#pragma unroll
        for (int load_N = 0; load_N < loadN; ++load_N) {
          As[write_stage_idx][A_Tile_Col + load_N][A_Tile_Row + tile_idx_bn] =
              load_A_reg[index + load_N];
        }
      }

// load B from global to shared
#pragma unroll
      for (int tile_idx_bk = 0; tile_idx_bk < Block_Size_K;
           tile_idx_bk += B_Tile_Row_Stride) {
        int index = tile_idx_bk / B_Tile_Row_Stride * loadN;
#pragma unroll
        for (int load_N = 0; load_N < loadN; ++load_N) {
          Bs[write_stage_idx][B_Tile_Row + tile_idx_bk][B_Tile_Col + load_N] =
              load_B_reg[index + load_N];
        }
      }
    }

    __syncthreads();
    write_stage_idx ^= 1;
  }

// store back to C
#pragma unroll
  for (int tile_idx_rn = 0; tile_idx_rn < Thread_Size_Y; ++tile_idx_rn) {
#pragma unroll
    for (int tile_idx_rm = 0; tile_idx_rm < Thread_Size_X;
         tile_idx_rm += loadN) {
      Fetch_Float4(
          C[Offset(by * Block_Size_N + Thread_Size_Y * ty + tile_idx_rn,
                   bx * Block_Size_M + Thread_Size_X * tx + tile_idx_rm, M)]) =
          Fetch_Float4(accum[tile_idx_rn][tile_idx_rm]);
    }
  }
}

int main(int argc, char **argv) {
  if (argc != 4) {
    printf("gemm_v3_double_buffer , usage: ./main [N] [K] [M]");
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

  const int Block_Size_N = 128, Block_Size_M = 128, Block_Size_K = 8,
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
    dim3 dimBlock(Block_Size_N / Thread_Size_Y, Block_Size_M / Thread_Size_X);
    dim3 dimGrid(N / Block_Size_N, M / Block_Size_M);
    Gemm<Block_Size_N, Block_Size_K, Block_Size_M, Thread_Size_Y, Thread_Size_X,
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
      "gemm_v3_double_buffer , My gemm Performance= %.2f GFlop/s, Time= %.3f "
      "msec, Size= %.0f Ops,\n",
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
  printf("gemm_v3_double_buffer , CuBlas Performance= %.2f GFlop/s, Time= %.3f "
         "msec, Size= %.0f Ops,\n",
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
      printf("gemm_v3_double_buffer , Error! Matrix[%05d]=%.8f, ref=%.8f error "
             "term is > %E\n",
             i, h_c[i], h_c1[col * N + row], eps);
      correct = false;
      break;
    }
  }

  printf("gemm_v3_double_buffer , %s\n",
         correct ? "Result= PASS" : "Result= FAIL");
  printf("gemm_v3_double_buffer , ratio= %f\n", gigaFlops[0] / gigaFlops[1]);

  delete h_a, h_b, h_c, h_c1;
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}