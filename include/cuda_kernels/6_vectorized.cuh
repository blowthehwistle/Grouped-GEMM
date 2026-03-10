#pragma once

#include <cstdio>
#include "cuda_runtime.h"
#include "cuda_fp16.h"
#include "cuda_bf16.h"
#include "mma.h"

template <const uint bm, const uint bn, const uint bk, const uint tw_m, const uint tw_n>
__global__ void vectorized_fp(float *__restrict A, float *__restrict B, float *C, int m, int n, int k, float alpha, float beta) {
    __shared__ float AT_shared[bk * bm];
    __shared__ float B_shared[bk * bn];
    float per_thread_result[tw_m * tw_n] = {0.0f};
    float reg_A[tw_m] = {0.0f};
    float reg_B[tw_n] = {0.0f};

    // coordinate of block in grid
    const int brow = blockIdx.y;
    const int bcol = blockIdx.x;

    // coordinate of thread in thread block
    const int trow = threadIdx.x / (bn / tw_n);
    const int tcol = threadIdx.x % (bn / tw_n);

    // coordinate of thread in thread block for each shared memory block to use LSD.128
    // 128bits = 32bits * 4
    // single load instruction can load 4 float type elements using vectorization
    const int A_trow = threadIdx.x / (bk / 4);
    const int A_tcol = threadIdx.x % (bk / 4);
    const int B_trow = threadIdx.x / (bn / 4);
    const int B_tcol = threadIdx.x % (bn / 4);

    // staring pointer of each matrcies
    A += brow * bm * k; // grid dimension = (m / bm) * (n / bn) => to recover the original shape, mulitply bm
    B += bcol * bn;
    C += brow * bm * n + bcol * bn;

    // out most loop
    // divide whole matrix into blocks
    for(int i = 0; i < k; i += bk) {
        // load the data into shared memory
        float4 A_tmp = reinterpret_cast<float4 *>(&A[A_trow * k + A_tcol * 4])[0];
        AT_shared[(A_tcol * 4 + 0) * bm + A_trow] = A_tmp.x;
        AT_shared[(A_tcol * 4 + 1) * bm + A_trow] = A_tmp.y;
        AT_shared[(A_tcol * 4 + 2) * bm + A_trow] = A_tmp.z;
        AT_shared[(A_tcol * 4 + 3) * bm + A_trow] = A_tmp.w;
        
        reinterpret_cast<float4 *>(&B_shared[B_trow * bn + B_tcol * 4])[0] = reinterpret_cast<float4 *>(&B[B_trow * n + B_tcol * 4])[0];
        // wait unit the whole threads in single thread block finish the data loading
        // unit of shared memory = thread block
        __syncthreads();
        
        // inner loop
        // divide block into small matrices where single thread responsible for 
        for(int j = 0; j < bk; j++) {
            // load data in shared memory into local register
            for(int k = 0; k < tw_m; k++)
                reg_A[k] = AT_shared[j * bm + trow * tw_m + k];
            for(int k = 0; k < tw_n; k++)
                reg_B[k] = B_shared[j * bn + tcol * tw_n + k];
            // outer product two vectors to make matrix
            for(int inter_m = 0; inter_m < tw_m; inter_m++) {
                for(int inter_n = 0; inter_n < tw_n; inter_n++)
                    per_thread_result[inter_m * tw_n + inter_n] += reg_A[inter_m] * reg_B[inter_n];
            }
        }
        __syncthreads();

        // move onto next block
        A += bk;
        B += bk * n;
    }

    for(int i = 0; i < tw_m; i++) {
        for(int j = 0; j < tw_n; j += 4) {
            // load vector into register
            float4 tmp = reinterpret_cast<float4 *>(&C[(trow * tw_m + i) * n + tcol * tw_n + j])[0];

            // write the result into register
            tmp.x = alpha * per_thread_result[i * tw_n + j + 0] + beta * tmp.x;
            tmp.y = alpha * per_thread_result[i * tw_n + j + 1] + beta * tmp.y;
            tmp.z = alpha * per_thread_result[i * tw_n + j + 2] + beta * tmp.z;
            tmp.w = alpha * per_thread_result[i * tw_n + j + 3] + beta * tmp.w;

            // write the result into matrix C
            reinterpret_cast<float4 *>(&C[(trow * tw_m + i) * n + tcol * tw_n + j])[0] = tmp;
        }
    }
}

template <const uint bm, const uint bn, const uint bk, const uint tw_m, const uint tw_n>
__global__ void vectorized_bf(__nv_bfloat16 *__restrict A, __nv_bfloat16 *__restrict B, float *C, int m, int n, int k, float alpha, float beta) {
    __shared__ __nv_bfloat16 AT_shared[bk * bm];
    __shared__ __nv_bfloat16 B_shared[bk * bn];
    float per_thread_result[tw_m * tw_n] = {0.0f};
    float reg_A[tw_m] = {0.0f};
    float reg_B[tw_n] = {0.0f};

    // coordinate of block in grid
    const int brow = blockIdx.y;
    const int bcol = blockIdx.x;

    // coordinate of thread in thread block
    const int trow = threadIdx.x / (bn / tw_n);
    const int tcol = threadIdx.x % (bn / tw_n);

    // coordinate of thread in thread block for each shared memory block to use LSD.128
    // 128bits = 32bits * 4
    // single load instruction can load 4 float type elements using vectorization
    const int A_trow = threadIdx.x / (bk / 4);
    const int A_tcol = threadIdx.x % (bk / 4);
    const int B_trow = threadIdx.x / (bn / 4);
    const int B_tcol = threadIdx.x % (bn / 4);

    // staring pointer of each matrcies
    A += brow * bm * k; // grid dimension = (m / bm) * (n / bn) => to recover the original shape, mulitply bm
    B += bcol * bn;
    C += brow * bm * n + bcol * bn;

    // out most loop
    // divide whole matrix into blocks
    for(int i = 0; i < k; i += bk) {
        // load the data into shared memory
        short4 A_tmp = reinterpret_cast<short4 *>(&A[A_trow * k + A_tcol * 4])[0];
        AT_shared[(A_tcol * 4 + 0) * bm + A_trow] = __short2bfloat16_ru(A_tmp.x);
        AT_shared[(A_tcol * 4 + 1) * bm + A_trow] = __short2bfloat16_ru(A_tmp.y);
        AT_shared[(A_tcol * 4 + 2) * bm + A_trow] = __short2bfloat16_ru(A_tmp.z);
        AT_shared[(A_tcol * 4 + 3) * bm + A_trow] = __short2bfloat16_ru(A_tmp.w);
        
        reinterpret_cast<short4 *>(&B_shared[B_trow * bn + B_tcol * 4])[0] = reinterpret_cast<short4 *>(&B[B_trow * n + B_tcol * 4])[0];
        // wait unit the whole threads in single thread block finish the data loading
        // unit of shared memory = thread block
        __syncthreads();

        // inner loop
        // divide block into small matrices where single thread responsible for 
        for(int j = 0; j < bk; j++) {
            // load data in shared memory into local register
            for(int k = 0; k < tw_m; k++)
                reg_A[k] = __bfloat162float(AT_shared[j * bn + trow * tw_m + k]);
            for(int k = 0; k < tw_n; k++)
                reg_B[k] = __bfloat162float(__short2bfloat16_ru(B_shared[j * bn + tcol * tw_n + k]));
            // outer product two vectors to make matrix
            for(int inter_m = 0; inter_m < tw_m; inter_m++) {
                for(int inter_n = 0; inter_n < tw_n; inter_n++)
                    per_thread_result[inter_m * tw_n + inter_n] += reg_A[inter_m] * reg_B[inter_n];
            }
        }
        __syncthreads();

        // move onto next block
        A += bk;
        B += bk * n;
    }

    for(int i = 0; i < tw_m; i++) {
        for(int j = 0; j < tw_n; j += 4) {
            // load vector into register
            float4 tmp = reinterpret_cast<float4 *>(&C[(trow * tw_m + i) * n + tcol * tw_n + j])[0];

            // write the result into register
            tmp.x = alpha * per_thread_result[i * tw_n + j + 0] + beta * tmp.x;
            tmp.y = alpha * per_thread_result[i * tw_n + j + 1] + beta * tmp.y;
            tmp.z = alpha * per_thread_result[i * tw_n + j + 2] + beta * tmp.z;
            tmp.w = alpha * per_thread_result[i * tw_n + j + 3] + beta * tmp.w;

            // write the result into matrix C
            reinterpret_cast<float4 *>(&C[(trow * tw_m + i) * n + tcol * tw_n + j])[0] = tmp;
        }
    }
}

template <const uint bm, const uint bn, const uint bk, const uint tw_m, const uint tw_n>
__global__ void vectorized_h(__half *__restrict A, __half *__restrict B, float *C, int m, int n, int k, float alpha, float beta) {
    __shared__ __half AT_shared[bk * bm];
    __shared__ __half B_shared[bk * bn];
    float per_thread_result[tw_m * tw_n] = {0.0f};
    float reg_A[tw_m] = {0.0f};
    float reg_B[tw_n] = {0.0f};

    // coordinate of block in grid
    const int brow = blockIdx.y;
    const int bcol = blockIdx.x;

    // coordinate of thread in thread block
    const int trow = threadIdx.x / (bn / tw_n);
    const int tcol = threadIdx.x % (bn / tw_n);

    // coordinate of thread in thread block for each shared memory block to use LSD.128
    // 128bits = 32bits * 4
    // single load instruction can load 4 float type elements using vectorization
    const int A_trow = threadIdx.x / (bk / 4);
    const int A_tcol = threadIdx.x % (bk / 4);
    const int B_trow = threadIdx.x / (bn / 4);
    const int B_tcol = threadIdx.x % (bn / 4);

    // staring pointer of each matrcies
    A += brow * bm * k; // grid dimension = (m / bm) * (n / bn) => to recover the original shape, mulitply bm
    B += bcol * bn;
    C += brow * bm * n + bcol * bn;

    // out most loop
    // divide whole matrix into blocks
    for(int i = 0; i < k; i += bk) {
        // load the data into shared memory
        short4 A_tmp = reinterpret_cast<short4 *>(&A[A_trow * k + A_tcol * 4])[0];
        AT_shared[(A_tcol * 4 + 0) * bm + A_trow] = A_tmp.x;
        AT_shared[(A_tcol * 4 + 1) * bm + A_trow] = A_tmp.y;
        AT_shared[(A_tcol * 4 + 2) * bm + A_trow] = A_tmp.z;
        AT_shared[(A_tcol * 4 + 3) * bm + A_trow] = A_tmp.w;
        
        reinterpret_cast<short4 *>(&B_shared[B_trow * bn + B_tcol * 4])[0] = reinterpret_cast<short4 *>(&B[B_trow * n + B_tcol * 4])[0];
        // wait unit the whole threads in single thread block finish the data loading
        // unit of shared memory = thread block
        __syncthreads();
        
        // inner loop
        // divide block into small matrices where single thread responsible for 
        for(int j = 0; j < bk; j++) {
            // load data in shared memory into local register
            for(int k = 0; k < tw_m; k++)
                reg_A[k] = __half2float(AT_shared[j * bn + trow * tw_m + k]);
            for(int k = 0; k < tw_n; k++)
                reg_B[k] = __half2float(B_shared[j * bn + tcol * tw_n + k]);
            // outer product two vectors to make matrix
            for(int inter_m = 0; inter_m < tw_m; inter_m++) {
                for(int inter_n = 0; inter_n < tw_n; inter_n++)
                    per_thread_result[inter_m * tw_n + inter_n] += reg_A[inter_m] * reg_B[inter_n];
            }
        }
        __syncthreads();

        // move onto next block
        A += bk;
        B += bk * n;
    }

    for(int i = 0; i < tw_m; i++) {
        for(int j = 0; j < tw_n; j += 4) {
            // load vector into register
            float4 tmp = reinterpret_cast<float4 *>(&C[(trow * tw_m + i) * n + tcol * tw_n + j])[0];

            // write the result into register
            tmp.x = alpha * per_thread_result[i * tw_n + j + 0] + beta * tmp.x;
            tmp.y = alpha * per_thread_result[i * tw_n + j + 1] + beta * tmp.y;
            tmp.z = alpha * per_thread_result[i * tw_n + j + 2] + beta * tmp.z;
            tmp.w = alpha * per_thread_result[i * tw_n + j + 3] + beta * tmp.w;

            // write the result into matrix C
            reinterpret_cast<float4 *>(&C[(trow * tw_m + i) * n + tcol * tw_n + j])[0] = tmp;
        }
    }
}