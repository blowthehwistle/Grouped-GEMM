#pragma once

#include <cstdio>
#include "cuda_runtime.h"
#include "cuda_fp16.h"
#include "cuda_bf16.h"

__global__ void naive_fp(const float *__restrict A, const float *__restrict B, float *C, int m, int n, int k, float alpha, float beta) {
    const uint row = blockDim.x * blockIdx.x + threadIdx.x;
    const uint col = blockDim.y * blockIdx.y + threadIdx.y;
    
    if(row < m && col < n) {
        float tmp = 0.0f;
        for(int i = 0; i < k; i++)
            tmp += A[row * k + i] * B[i * n + col];
        C[row * n + col] = alpha * tmp + beta * C[row * n + col];
    }
}

__global__ void naive_bf(const __nv_bfloat16 *__restrict A, const __nv_bfloat16 *__restrict B, float *C, int m, int n, int k, float alpha, float beta) {
    const uint row = blockDim.x * blockIdx.x + threadIdx.x;
    const uint col = blockDim.y * blockIdx.y + threadIdx.y;

    if(row < m && col < n) {
        float tmp = 0.0f;
        for(int i = 0; i < k; i++)
            tmp += __bfloat162float(A[row * k + i]) * __bfloat162float(B[i * n + col]);
        C[row * n + col] = alpha * tmp + beta * C[row * n + col];
    }
}

__global__ void naive_h(const __half *__restrict A, const __half *__restrict B, float *C, int m, int n, int k, float alpha, float beta) {
    const uint row = blockDim.x * blockIdx.x + threadIdx.x;
    const uint col = blockDim.y * blockIdx.y + threadIdx.y;

    if(row < m && col < n) {
        float tmp = 0.0f;
        for(int i = 0; i < k; i++)
            tmp += __half2float(A[row * k + i]) * __half2float(B[i * n + col]);
        C[row * n + col] = alpha * tmp + beta * C[row * n + col];
    }
}