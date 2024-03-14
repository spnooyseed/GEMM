### learning to Implement and optimize GEMM

### how to use
```bash
1. clang-format -i gemm.cu
2. nvcc -std=c++11 -lcublas -o a gemm.cu
3. ./a 2048 2048 2048
```