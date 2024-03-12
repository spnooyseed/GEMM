### learning to Implement and optimize GEMM

### how to use
```bash
1. clang-format -i gemm.cu
2. nvcc -std=c++11 -lcublas gemm.cu -o a
3. ./a 2048 2048 2048
```