#pragma once

#include <cstdio>
#include "cuda_runtime.h"
#include "cuda_fp16.h"
#include "cuda_bf16.h"
#include "mma.h"

const uint WARP_SIZE = 32;

template <const uint bm, const uint bn, const uint bk, const int wm, const int wn, const int wniter, 
            const uint tw_m, const uint tw_n, const int num_threads>
__global__ void warptiling_fp(float *__restrict A, float *__restrict B, float *C, int m, int n, int k, float alpha, float beta) {    
    __shared__ float AT_shared[bk * bm];
    __shared__ float B_shared[bk * bn];
    
    // coordinate of block in grid
    const int brow = blockIdx.y;
    const int bcol = blockIdx.x;

    // coordinate of warp in thread block
    const int widx = threadIdx.x / WARP_SIZE;
    const int wrow = widx / (bn / wn);
    const int wcol = widx % (bn / wn);

    // size of subtile in the warp region
    const int wmiter = (wm * wn) / (WARP_SIZE * tw_m * tw_n * wniter);
    const int subm = wm / wmiter;
    const int subn = wn / wniter;

    // coordinate of thread in warp tile which is in the thread block
    const int wtidx = threadIdx.x % WARP_SIZE;
    const int wtrow = wtidx / (subn / tw_n);
    const int wtcol = wtidx % (subn / tw_n);

    // allocate the local register for each threads
    float per_thread_result[(wmiter * wniter) * (tw_m * tw_n)] = {0.0};
    float reg_A[wmiter * tw_m] = {0.0};
    float reg_B[wniter * tw_n] = {0.0};

    // coordinate of thread in thread block for each shared memory block to use LSD.128
    // 128bits = 32bits * 4
    // single load instruction can load 4 float type elements using vectorization
    const int A_trow = threadIdx.x / (bk / 4);
    const int A_tcol = threadIdx.x % (bk / 4);
    const int B_trow = threadIdx.x / (bn / 4);
    const int B_tcol = threadIdx.x % (bn / 4);

    const int strideA = (num_threads * 4) / bk;
    const int strideB = num_threads / (bn / 4);

    // staring pointer of each matrcies
    A += brow * bm * k; // grid dimension = (m / bm) * (n / bn) => to recover the original shape, mulitply bm
    B += bcol * bn;
    C += (brow * bm + wrow * wm) * n + bcol * bn + wcol * wn;

    // out most loop
    // divide whole matrix into blocks
    for(int i = 0; i < k; i += bk) {
        // load data from global memory to shared memory
        for(int offset = 0; offset + strideA <= bm; offset += strideA) {
            const float4 A_tmp = reinterpret_cast<const float4 *>(&A[(A_trow + offset) * k + A_tcol * 4])[0];
            AT_shared[(A_tcol * 4 + 0) * bm + A_trow + offset] = A_tmp.x;
            AT_shared[(A_tcol * 4 + 1) * bm + A_trow + offset] = A_tmp.y;
            AT_shared[(A_tcol * 4 + 2) * bm + A_trow + offset] = A_tmp.z;
            AT_shared[(A_tcol * 4 + 3) * bm + A_trow + offset] = A_tmp.w;
        }
        for(int offset = 0; offset + strideB <= bk; offset += strideB)
            reinterpret_cast<float4 *>(&B_shared[(B_trow + offset) * bn + B_tcol * 4])[0] = reinterpret_cast<const float4 *>(&B[(B_trow + offset) * n + B_tcol * 4])[0];
        // wait unit the whole threads in single thread block finish the data loading
        // unit of shared memory = thread block
        __syncthreads();

        // perform matrix multiplication
        for(int sidx = 0; sidx < bk; sidx++) {
            // load data from shared memory to local register
			for(int sub_rowidx = 0; sub_rowidx < wmiter; sub_rowidx++) {
                for(int j = 0; j < tw_m; j++)
                    reg_A[sub_rowidx * tw_m + j] = AT_shared[(sidx * bm) + wrow * wm + sub_rowidx * subm + wtrow * tw_m + j];
            }
            for(int sub_colidx = 0; sub_colidx < wniter; sub_colidx++) {
                for(int j = 0; j < tw_n; j++)
                    reg_B[sub_colidx * tw_n + j] = B_shared[(sidx * bn) + wcol * wn + sub_colidx * subn + wtcol * tw_n + j];
            }
			// execute multiplication
            for(int sub_rowidx = 0; sub_rowidx < wmiter; sub_rowidx++) {
            	for(int sub_colidx = 0; sub_colidx < wniter; sub_colidx++) {
					for(int inter_m = 0; inter_m < tw_m; inter_m++) {
						for(int inter_n = 0; inter_n < tw_n; inter_n++)
							per_thread_result[(sub_rowidx * tw_m + inter_m) * (wniter * tw_n) + (sub_colidx * tw_n) + inter_n] 
								+= reg_A[sub_rowidx * tw_m + inter_m] * reg_B[sub_colidx * tw_n + inter_n];
					}
            	}
        	}
        }
        __syncthreads();

        // move onto next block
        A += bk;
        B += bk * n;
    }
    
    for(int sub_rowidx = 0; sub_rowidx < wmiter; sub_rowidx++) {           
        for(int sub_colidx = 0; sub_colidx < wniter; sub_colidx++) {
            float *inter_C = C + (sub_rowidx * subm) * n + sub_colidx * subn;
            for(int inter_m = 0; inter_m < tw_m; inter_m++) {
                for(int inter_n = 0; inter_n < tw_n; inter_n += 4) {
                    // load vector into register
                    float4 tmp = reinterpret_cast<float4 *>(&inter_C[(wtrow * tw_m + inter_m) * n + wtcol * tw_n + inter_n])[0];

                    // write the result into register
                    const int idx = (sub_rowidx * tw_m + inter_m) * (wniter * tw_n) + sub_colidx * tw_n + inter_n;
                    tmp.x = alpha * per_thread_result[idx + 0] + beta * tmp.x;
                    tmp.y = alpha * per_thread_result[idx + 1] + beta * tmp.y;
                    tmp.z = alpha * per_thread_result[idx + 2] + beta * tmp.z;
                    tmp.w = alpha * per_thread_result[idx + 3] + beta * tmp.w;

                    // write the result into matrix C
                    reinterpret_cast<float4 *>(&inter_C[(wtrow * tw_m + inter_m) * n + wtcol * tw_n + inter_n])[0] = tmp;
                }
            }
       }
    }
}