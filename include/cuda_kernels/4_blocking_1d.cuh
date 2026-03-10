#pragma once

#include <cstdio>
#include "cuda_runtime.h"
#include "cuda_fp16.h"
#include "cuda_bf16.h"

template <const uint bm, const uint bn, const uint bk, const uint tw>
__global__ void blocking_1d_fp(const float *__restrict A, const float *__restrict B, float *C, int m, int n, int k, float alpha, float beta) {
    __shared__ float A_shared[bm * bk];
    __shared__ float B_shared[bk * bn];
    float inter_result[tw] = {0.0f};

    // coordinate of block in grid
    const int brow = blockIdx.y;
    const int bcol = blockIdx.x;

    // coordinate of thread in thread block
    const int trow = threadIdx.x / bn;
    const int tcol = threadIdx.x % bn;

    // coordinate of thread in thread block for each shared memory block
    const int A_trow = threadIdx.x / bk;
    const int A_tcol = threadIdx.x % bk;
    const int B_trow = threadIdx.x / bn;
    const int B_tcol = threadIdx.x % bn;

    // staring pointer of each matrcies
    A += brow * bm * k; // grid dimension = (m / bm) * (n / bn) => to recover the original shape, mulitply bm
    B += bcol * bn;
    C += brow * bm * n + bcol * bn;

    // outer loop
    // divide whole matrix into blocks
    for(int i = 0; i < k; i += bk) {
        // load the data into shared memory
        A_shared[A_trow * bk + A_tcol] = A[A_trow * k + A_tcol];
        B_shared[B_trow * bn + B_tcol] = B[B_trow * n + B_tcol];
        // wait unit the whole threads in single thread block finish the data loading
        // unit of shared memory = thread block
        __syncthreads();
        
        // inner loop
        // divide block into vector where single thread responsible for 
        for(int j = 0; j < bk; j++) {
            float tmp = B_shared[j * bn + tcol];
            // single thread multiply elements in the vector
            for(int k = 0; k < tw; k++)
                inter_result[k] += A_shared[(trow * tw + k) * bk + j] * tmp;
        }
        __syncthreads();

        // move onto next block
        A += bk;
        B += bk * n;
    }

    for(int k = 0; k < tw; k++)
        C[(trow * tw + k) * n + tcol] = alpha * inter_result[k] + beta * C[(trow * tw + k) * n + tcol];
}

template <const uint bm, const uint bn, const uint bk, const uint tw>
__global__ void blocking_1d_bf(const __nv_bfloat16 *__restrict A, const __nv_bfloat16 *__restrict B, float *C, int m, int n, int k, float alpha, float beta) {
    __shared__ __nv_bfloat16 A_shared[bm * bk];
    __shared__ __nv_bfloat16 B_shared[bk * bn];
    float inter_result[tw] = {0.0f};

    // coordinate of block in grid
    const int brow = blockIdx.y;
    const int bcol = blockIdx.x;

    // coordinate of thread in thread block
    const int trow = threadIdx.x / bn;
    const int tcol = threadIdx.x % bn;

    // coordinate of thread in thread block for each shared memory block
    const int A_trow = threadIdx.x / bk;
    const int A_tcol = threadIdx.x % bk;
    const int B_trow = threadIdx.x / bn;
    const int B_tcol = threadIdx.x % bn;

    // staring pointer of each matrcies
    A += brow * bm * k; // grid dimension = (m / bm) * (n / bn) => to recover the original shape, mulitply bm
    B += bcol * bn;
    C += brow * bm * n + bcol * bn;

    // outer loop
    // divide whole matrix into blocks
    for(int i = 0; i < k; i += bk) {
        // load the data into shared memory
        A_shared[A_trow * bk + A_tcol] = A[A_trow * k + A_tcol];
        B_shared[B_trow * bn + B_tcol] = B[B_trow * n + B_tcol];
        // wait unit the whole threads in single thread block finish the data loading
        // unit of shared memory = thread block
        __syncthreads();
        
        // inner loop
        // divide block into vector where single thread responsible for 
        for(int j = 0; j < bk; j++) {
            float tmp = __bfloat162float(B_shared[j * bn + tcol]);
            // single thread multiply elements in the vector
            for(int k = 0; k < tw; k++)
                inter_result[k] += __bfloat162float(A_shared[(trow * tw + k) * bk + j]) * tmp;
        }
        __syncthreads();

        // move onto next block
        A += bk;
        B += bk * n;
    }

    for(int k = 0; k < tw; k++)
        C[(trow * tw + k) * n + tcol] = alpha * inter_result[k] + beta * C[(trow * tw + k) * n + tcol];
}

template <const uint bm, const uint bn, const uint bk, const uint tw>
__global__ void blocking_1d_h(const __half *__restrict A, const __half *__restrict B, float *C, int m, int n, int k, float alpha, float beta) {
    __shared__ __half A_shared[bm * bk];
    __shared__ __half B_shared[bk * bn];
    float inter_result[tw] = {0.0f};

    // coordinate of block in grid
    const int brow = blockIdx.y;
    const int bcol = blockIdx.x;

    // coordinate of thread in thread block
    const int trow = threadIdx.x / bn;
    const int tcol = threadIdx.x % bn;

    // coordinate of thread in thread block for each shared memory block
    const int A_trow = threadIdx.x / bk;
    const int A_tcol = threadIdx.x % bk;
    const int B_trow = threadIdx.x / bn;
    const int B_tcol = threadIdx.x % bn;

    // staring pointer of each matrcies
    A += brow * bm * k; // grid dimension = (m / bm) * (n / bn) => to recover the original shape, mulitply bm
    B += bcol * bn;
    C += brow * bm * n + bcol * bn;

    // outer loop
    // divide whole matrix into blocks
    for(int i = 0; i < k; i += bk) {
        // load the data into shared memory
        A_shared[A_trow * bk + A_tcol] = A[A_trow * k + A_tcol];
        B_shared[B_trow * bn + B_tcol] = B[B_trow * n + B_tcol];
        // wait unit the whole threads in single thread block finish the data loading
        // unit of shared memory = thread block
        __syncthreads();
        
        // inner loop
        // divide block into vector where single thread responsible for 
        for(int j = 0; j < bk; j++) {
            float tmp = __half2float(B_shared[j * bn + tcol]);
            // single thread multiply elements in the vector
            for(int k = 0; k < tw; k++)
                inter_result[k] += __half2float(A_shared[(trow * tw + k) * bk + j]) * tmp;
        }
        __syncthreads();

        // move onto next block
        A += bk;
        B += bk * n;
    }

    for(int k = 0; k < tw; k++)
        C[(trow * tw + k) * n + tcol] = alpha * inter_result[k] + beta * C[(trow * tw + k) * n + tcol];
}