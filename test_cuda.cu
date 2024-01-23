#include <cuda_runtime.h>
#include "stdio.h"
__global__ void kernel() {
    // 使用内置变量 blockDim 输出块的维度
    printf("Block dimensions: (%d, %d) , ThreadIdx = (%d , %d)\n", blockDim.x, blockDim.y , threadIdx.x , threadIdx.y);

    // 其他内核逻辑
}

int main() {
    // 设置网格和块的维度
    dim3 gridDim(3, 2);
    dim3 blockDim(2, 3);

    // 调用 GPU 内核函数
    kernel<<<gridDim, blockDim>>>();
    cudaDeviceSynchronize(); // 等待 GPU 完成

    return 0;
}
