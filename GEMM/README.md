### learning to Implement and optimize GEMM

### how to use
```bash
1. clang-format -i gemm.cu

// run one cuda kernel
1. nvcc -std=c++11 -lcublas -o a gemm.cu
2. ./a 2048 2048 2048

// run all cuda kernel
1. make
```