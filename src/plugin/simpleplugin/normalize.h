
#pragma once

#include <cuda_runtime.h>


#include <cuda_fp16.h>
#include <stdio.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "common.h"


void normalize_cuda(
    float* data, int num_elements, int dim, float p, float eps,
    cudaStream_t stream = 0
);
