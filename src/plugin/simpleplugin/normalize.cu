#include "normalize.h"
#include <cub/block/block_reduce.cuh>
#include <math.h>

// 1. 计算单向量L2范数的核函数
template <typename T, int BLOCK_SIZE>
__global__ void compute_l2_norm_kernel(
    const T* input, float* norms, 
    size_t dim_size, size_t stride, float eps
) {
    // 共享内存存储每个线程的局部平方和
    __shared__ typename cub::BlockReduce<float, BLOCK_SIZE>::TempStorage temp_storage;
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;  // 向量索引
    const T* vec_start = input + bid * stride;  // 当前向量的起始地址

    // 每个线程计算局部平方和
    float local_sum = 0.0f;
    for (int i = tid; i < dim_size; i += BLOCK_SIZE) {
        float val = static_cast<float>(vec_start[i]);
        local_sum += val * val;
    }

    // 块内规约求平方和
    float block_sum = cub::BlockReduce<float, BLOCK_SIZE>(temp_storage).Sum(local_sum);
    
    // 计算L2范数（含eps保护）
    if (tid == 0) {
        norms[bid] = sqrtf(block_sum + eps);
    }
}

// 2. 归一化执行核函数
template <typename T>
__global__ void normalize_kernel(
    const T* input, T* output, const float* norms,
    size_t dim_size, size_t stride
) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const T* vec_in = input + bid * stride;
    T* vec_out = output + bid * stride;

    const float norm_val = norms[bid];  // 当前向量的范数
    
    // 每个线程处理部分元素
    for (int i = tid; i < dim_size; i += blockDim.x) {
        float val = static_cast<float>(vec_in[i]);
        vec_out[i] = static_cast<T>(val / norm_val);
    }
}

// 3. 主机函数封装
template <typename T>
void cuda_normalize(
    const T* input, T* output,
    size_t num_vectors, size_t dim_size, size_t stride,
    float eps, cudaStream_t stream
) {
    // 配置内核参数（每向量一个线程块）
    const int BLOCK_SIZE = 256;
    dim3 grid(num_vectors);  // 每个向量分配一个线程块
    dim3 block(BLOCK_SIZE);

    // 分配范数缓存（每个向量一个范数值）
    float* d_norms;
    cudaMallocAsync(&d_norms, num_vectors * sizeof(float), stream);

    // 步骤1：计算每个向量的L2范数
    compute_l2_norm_kernel<T, BLOCK_SIZE><<<grid, block, 0, stream>>>(
        input, d_norms, dim_size, stride, eps
    );

    // 步骤2：执行归一化
    normalize_kernel<T><<<grid, block, 0, stream>>>(
        input, output, d_norms, dim_size, stride
    );

    // 异步释放范数缓存
    cudaFreeAsync(d_norms, stream);
}

// 显式实例化
template void cuda_normalize<float>(const float*, float*, size_t, size_t, size_t, float, cudaStream_t);
template void cuda_normalize<half>(const half*, half*, size_t, size_t, size_t, float, cudaStream_t);