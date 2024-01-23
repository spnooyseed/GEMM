#include <iostream>
#include <cmath>

// 定义矩阵的维度
#define N 3

// CUDA核函数：矩阵乘法
__global__ void matrixMultiply(int *a, int *b, int *c)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int sum = 0;
    for (int i = 0; i < N; ++i)
    {
        sum += a[row * N + i] * b[i * N + col];
    }

    c[row * N + col] = sum;
}

int main()
{
    int *h_a, *h_b, *h_c; // 主机上的矩阵
    int *d_a, *d_b, *d_c; // 设备上的矩阵
    // 分配主机上的内存
    h_a = new int[N * N];
    h_b = new int[N * N];
    h_c = new int[N * N];
    // int *d_a;
    // std::cout << (d_a) << " " << (&d_a) << " " << (void **)(&d_a) << "\n";

    // 初始化矩阵数据
    for (int i = 0; i < N * N; ++i)
    {
        h_a[i] = i;
        h_b[i] = i;
    }

    // 分配设备上的内存
    cudaMalloc((void **)&d_a, N * N * sizeof(int));
    cudaMalloc((void **)&d_b, N * N * sizeof(int));
    cudaMalloc((void **)&d_c, N * N * sizeof(int));

    // 将数据从主机复制到设备
    cudaMemcpy(d_a, h_a, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * N * sizeof(int), cudaMemcpyHostToDevice);

    // 定义线程块和网格的大小
    dim3 threadsPerBlock(N, N);
    dim3 numBlocks(1, 1);

    // 调用CUDA核函数
    matrixMultiply<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c);

    // 将结果从设备复制回主机
    cudaMemcpy(h_c, d_c, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    // 打印结果
    std::cout << "Matrix C (Result):" << std::endl;
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            std::cout << h_c[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    // 释放内存
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
