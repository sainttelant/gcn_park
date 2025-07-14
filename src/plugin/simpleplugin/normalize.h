#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

/**
 * @brief 对输入张量按指定维度进行L2归一化（对齐PyTorch行为）
 * @param input     输入张量指针（设备内存）
 * @param output    输出张量指针（设备内存）
 * @param num_vectors 归一化方向的数量（如对形状[N, C]按dim=1归一化时，num_vectors = N）
 * @param dim_size  归一化维度的长度（如对形状[N, C]按dim=1归一化时，dim_size = C）
 * @param stride    相邻归一化向量的内存步长（元素数量，非字节）
 * @param eps       数值稳定项（避免除零）
 * @param stream    CUDA流
 */
template <typename T>
void cuda_normalize(
    const T* input, T* output,
    size_t num_vectors, size_t dim_size, size_t stride,
    float eps = 1e-12f, cudaStream_t stream = nullptr
);


