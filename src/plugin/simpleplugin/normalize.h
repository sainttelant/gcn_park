#pragma once
#include <cuda_runtime.h>

template <typename T>
void cuda_normalize(
    const T* input, 
    T* output,
    size_t num_vectors,  // 替换 num_elements
    size_t dim_size,      // 归一化维度长度
    size_t stride,        // 向量间步长（字节）
    float p,             // 范数阶数
    float eps = 1e-12f,
    cudaStream_t stream = nullptr
);