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

![GEMM](GEMM.jpg)


### Referecnce
1. https://zhuanlan.zhihu.com/p/435908830
2. https://zhuanlan.zhihu.com/p/442930482
3. https://zhuanlan.zhihu.com/p/481600052
4. https://zhuanlan.zhihu.com/p/614109686