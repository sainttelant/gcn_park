#include "normalize.h"
#include <cub/block/block_reduce.cuh>
#include <math.h>

// 1. 范数计算核函数
template <typename T, int BLOCK_SIZE>
__global__ void compute_norm_kernel(
    const T* input, float* norms, size_t dim_size, size_t stride, float p) 
{
    // 每个线程计算局部值 (无需共享内存)
    float local_val = 0.0f;
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const T* vec_start = input + bid * stride;

    for (int i = tid; i < dim_size; i += BLOCK_SIZE) {
        float val = static_cast<float>(vec_start[i]);
        if (p == 1.0f)      local_val += fabs(val);
        else if (p == 2.0f) local_val += val * val;
        else if (isinf(p))  local_val = fmaxf(local_val, fabs(val));
    }

    // 使用CUB规约（关键修正：传入局部变量）
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    float block_sum = BlockReduce(temp_storage).Sum(local_val); // 改为 local_val

    if (tid == 0) {
        norms[bid] = (p == 2.0f) ? sqrtf(block_sum) : block_sum;
    }
}

// 2. 归一化核函数
template <typename T>
__global__ void normalize_kernel(
    const T* input,
    T* output,
    const float* norms,
    size_t dim_size,
    size_t stride,        // 元素步长
    float eps
) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    const T* vec_in = input + bid * stride;
    T* vec_out = output + bid * stride;

    __shared__ float norm_val;
    if (tid == 0) norm_val = fmaxf(norms[bid], eps);
    __syncthreads();

    for (int i = tid; i < dim_size; i += blockDim.x) {
        vec_out[i] = static_cast<T>(vec_in[i] / norm_val);
    }
}

// 3. 主机封装函数
template <typename T>
void cuda_normalize(
    const T* input, 
    T* output,
    size_t num_vectors,   // 向量数量
    size_t dim_size,      // 每个向量长度
    size_t stride_bytes,  // 向量间字节步长
    float p, 
    float eps, 
    cudaStream_t stream
) {
    // 计算元素步长（重要！）
    size_t stride = stride_bytes / sizeof(T);

    dim3 block(256);
    dim3 grid(num_vectors);

    // 分配范数缓存
    float* d_norms;
    cudaMallocAsync(&d_norms, num_vectors * sizeof(float), stream);

    // 计算范数
    size_t shmem_size = block.x * sizeof(float);
    compute_norm_kernel<T, 256><<<grid, block, shmem_size, stream>>>(
        input, d_norms, dim_size, stride, p
    );

    // 执行归一化
    normalize_kernel<T><<<grid, block, 0, stream>>>(
        input, output, d_norms, dim_size, stride, eps
    );

    // 异步释放内存
    cudaFreeAsync(d_norms, stream);
}

// 显式实例化
template void cuda_normalize<float>(const float*, float*, size_t, size_t, size_t, float, float, cudaStream_t);
