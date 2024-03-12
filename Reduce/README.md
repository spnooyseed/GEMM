### do reduce operation optimization follow :
1. do block_size = 256 , N = 32 * 1024 * 1024 , output.shape = 256 , baseline reduce
2. many thread in one warp dont compute , lead to warp divergence
3. this solve bank conflict
4. unroll last warp
5. unroll all warp

### how to use
```c++
// run all reduce cuda file
1. make
// run one reduce cuda file
1. nvcc file.cu -o file
2. ./file


// clang-format file 
clang-format -i file
```

![reduce](reduce.png)
### References
1. https://zhuanlan.zhihu.com/p/426978026
2. https://zhuanlan.zhihu.com/p/596012674