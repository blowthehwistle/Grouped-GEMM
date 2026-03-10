#pragma once

#include <cstdio>
#include "cuda_runtime.h"
#include "cuda_fp16.h"
#include "cuda_bf16.h"

template <const uint bm, const uint bn, const uint bk, const uint tw_m, const uint tw_n>
__global__ void blocking_2d_fp(const float *__restrict A, const float *__restrict B, float *C, int m, int n, int k, float alpha, float beta) {
    __shared__ float A_shared[bm * bk];
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

    // coordinate of thread in thread block for each shared memory block
    const int A_trow = threadIdx.x / bk;
    const int A_tcol = threadIdx.x % bk;
    const int B_trow = threadIdx.x / bn;
    const int B_tcol = threadIdx.x % bn;

    // total number of thread in thread block = total number of blocks in shared memory for each matrix
    const int total_num_thread = (bm * bn) / (tw_m * tw_n);
    const int A_row_num = total_num_thread / bk;
    const int B_row_num = total_num_thread / bn; 

    // staring pointer of each matrcies
    A += brow * bm * k; // grid dimension = (m / bm) * (n / bn) => to recover the original shape, mulitply bm
    B += bcol * bn;
    C += brow * bm * n + bcol * bn;
    
    // out most loop
    // divide whole matrix into blocks
    for(int i = 0; i < k; i += bk) {
        // load the data into shared memory
        for(int j = 0; j < bm; j += A_row_num)
            A_shared[(A_trow + j) * bk + A_tcol] = A[(A_trow + j) * k + A_tcol];
        for(int j = 0; j < bk; j += B_row_num)
            B_shared[(B_trow + j) * bn + B_tcol] = B[(B_trow + j) * n + B_tcol];
        // wait unit the whole threads in single thread block finish the data loading
        // unit of shared memory = thread block
        __syncthreads();
        
        // inner loop
        // divide block into small matrices where single thread responsible for 
        for(int j = 0; j < bk; j++) {
            // load data in shared memory into local register
            for(int k = 0; k < tw_m; k++)
                reg_A[k] = A_shared[(trow * tw_m + k) * bk + j];
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
        for(int j = 0; j < tw_n; j++)
            C[(trow * tw_m + i) * n + tcol * tw_n + j] = alpha * per_thread_result[i * tw_n + j] + beta * C[(trow * tw_m + i) * n + tcol * tw_n + j];
    }
}

template <const uint bm, const uint bn, const uint bk, const uint tw_m, const uint tw_n>
__global__ void blocking_2d_bf(const __nv_bfloat16 *__restrict A, const __nv_bfloat16 *__restrict B, float *C, int m, int n, int k, float alpha, float beta) {
    __shared__ __nv_bfloat16 A_shared[bm * bk];
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

    // coordinate of thread in thread block for each shared memory block
    const int A_trow = threadIdx.x / bk;
    const int A_tcol = threadIdx.x % bk;
    const int B_trow = threadIdx.x / bn;
    const int B_tcol = threadIdx.x % bn;

    // total number of thread in thread block = total number of blocks in shared memory for each matrix
    const int total_num_thread = (bm * bn) / (tw_m * tw_n);
    const int A_row_num = total_num_thread / bk;
    const int B_row_num = total_num_thread / bn; 

    // staring pointer of each matrcies
    A += brow * bm * k; // grid dimension = (m / bm) * (n / bn) => to recover the original shape, mulitply bm
    B += bcol * bn;
    C += brow * bm * n + bcol * bn;
    
    // out most loop
    // divide whole matrix into blocks
    for(int i = 0; i < k; i += bk) {
        // load the data into shared memory
        for(int j = 0; j < bm; j += A_row_num)
            A_shared[(A_trow + j) * bk + A_tcol] = A[(A_trow + j) * k + A_tcol];
        for(int j = 0; j < bk; j += B_row_num)
            B_shared[(B_trow + j) * bn + B_tcol] = B[(B_trow + j) * n + B_tcol];
        // wait unit the whole threads in single thread block finish the data loading
        // unit of shared memory = thread block
        __syncthreads();
        
        // inner loop
        // divide block into small matrices where single thread responsible for 
        for(int j = 0; j < bk; j++) {
            // load data in shared memory into local register
            for(int k = 0; k < tw_m; k++)
                reg_A[k] = __bfloat162float(A_shared[(trow * tw_m + k) * bk + j]);
            for(int k = 0; k < tw_n; k++)
                reg_B[k] = __bfloat162float(B_shared[j * bn + tcol * tw_n + k]);
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
        for(int j = 0; j < tw_n; j++)
            C[(trow * tw_m + i) * n + tcol * tw_n + j] = alpha * per_thread_result[i * tw_n + j] + beta * C[(trow * tw_m + i) * n + tcol * tw_n + j];
    }
}

template <const uint bm, const uint bn, const uint bk, const uint tw_m, const uint tw_n>
__global__ void blocking_2d_h(const __half *__restrict A, const __half *__restrict B, float *C, int m, int n, int k, float alpha, float beta) {
    __shared__ __half A_shared[bm * bk];
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

    // coordinate of thread in thread block for each shared memory block
    const int A_trow = threadIdx.x / bk;
    const int A_tcol = threadIdx.x % bk;
    const int B_trow = threadIdx.x / bn;
    const int B_tcol = threadIdx.x % bn;

    // total number of thread in thread block = total number of blocks in shared memory for each matrix
    const int total_num_thread = (bm * bn) / (tw_m * tw_n);
    const int A_row_num = total_num_thread / bk;
    const int B_row_num = total_num_thread / bn; 

    // staring pointer of each matrcies
    A += brow * bm * k; // grid dimension = (m / bm) * (n / bn) => to recover the original shape, mulitply bm
    B += bcol * bn;
    C += brow * bm * n + bcol * bn;
    
    // out most loop
    // divide whole matrix into blocks
    for(int i = 0; i < k; i += bk) {
        // load the data into shared memory
        for(int j = 0; j < bm; j += A_row_num)
            A_shared[(A_trow + j) * bk + A_tcol] = A[(A_trow + j) * k + A_tcol];
        for(int j = 0; j < bk; j += B_row_num)
            B_shared[(B_trow + j) * bn + B_tcol] = B[(B_trow + j) * n + B_tcol];
        // wait unit the whole threads in single thread block finish the data loading
        // unit of shared memory = thread block
        __syncthreads();
        
        // inner loop
        // divide block into small matrices where single thread responsible for 
        for(int j = 0; j < bk; j++) {
            // load data in shared memory into local register
            for(int k = 0; k < tw_m; k++)
                reg_A[k] = __half2float(A_shared[(trow * tw_m + k) * bk + j]);
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
        for(int j = 0; j < tw_n; j++)
            C[(trow * tw_m + i) * n + tcol * tw_n + j] = alpha * per_thread_result[i * tw_n + j] + beta * C[(trow * tw_m + i) * n + tcol * tw_n + j];
    }
}