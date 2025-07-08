#include "normalize.h"


__global__ void normalize_kernel(
    float* input, const int dim, const float eps, const int total_elements
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    
    // 共享内存加速归约
    extern __shared__ float s_norm[];
    float norm = 0.0f;
    for (int i = 0; i < dim; ++i) {
        const float val = input[idx * dim + i];
        norm += val * val;
    }
    s_norm[threadIdx.x] = norm;
    __syncthreads();
    
    // 块内归约
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) s_norm[threadIdx.x] += s_norm[threadIdx.x + s];
        __syncthreads();
    }
    
    // 归一化
    if (threadIdx.x == 0) {
        const float global_norm = sqrtf(s_norm[0] + eps);
        for (int i = 0; i < dim; ++i) {
            input[idx * dim + i] /= global_norm;
        }
    }
}

void normalize_cuda(
    float* data, int num_elements, int dim, float p, float eps,
    cudaStream_t stream
) {
    const int threads = 256;
    const int blocks = (num_elements + threads - 1) / threads;
    const size_t smem_size = threads * sizeof(float);
    normalize_kernel<<<blocks, threads, smem_size, stream>>>(
        data, dim, eps, num_elements
    );
}
