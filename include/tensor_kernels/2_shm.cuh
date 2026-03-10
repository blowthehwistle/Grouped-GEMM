#pragma once

#include <cstdio>
#include "mma.h"

using namespace nvcuda;

template <const uint bm, const uint bn, const uint bk, const int wm, const int wn, const int num_threads>
__global__ void shm_tf(const float *A, const float *B, float *C, int m, int n, int k, float alpha, float beta) {
    __shared__ float AT_shared[bk * bm];
    __shared__ float B_shared[bk * bn];

    // coordinate of block in grid
    const int brow = blockIdx.y;
    const int bcol = blockIdx.x;

    // starting pointer of each thread block in output matrix
    A += brow * bm * k;
    B += bcol * bn;
    C += brow * bm * n + bcol * bn;

    // coordinate of each thread in thread block for using LSD.128
    // single load instruction load 4 float elements
    const int A_trow = threadIdx.x / (bk / 4);
    const int A_tcol = threadIdx.x % (bk / 4);
    const int B_trow = threadIdx.x / (bn / 4);
    const int B_tcol = threadIdx.x % (bn / 4);
    
    // stride for loading input matrices to shared memory
    const int strideA = (num_threads * 4) / bk;
    const int strideB = (num_threads * 4) / bn;

    // coordinate of warp in thread block 
    const int widx = threadIdx.x / WARP_SIZE;
    const int wrow = widx / (bn / wn);
    const int wcol = widx % (bn / wn);

    // inner warp related variables
    const int wsubm = 16;
    const int wsubn = 16;
    const int wmiter = wm / wsubm;
    const int wniter = wn / wsubn;

    // Define WMMA fragment types for TF32 precision
    wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::col_major> a_frag[wmiter];
    wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major> b_frag[wniter];
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> c_frag[wmiter * wniter];

    // Initialize output fragment
    for(int i = 0; i < wmiter * wniter ;i++)
        wmma::fill_fragment(c_frag[i], 0.0f);

    for(int i = 0; i < k; i += bk) {
        // load elements from global memory to shared memory
        for(int offset = 0; offset + strideA <= bm; offset += strideA) {
            const float4 A_tmp = reinterpret_cast<const float4 *>(&A[(A_trow + offset) * k + A_tcol * 4])[0];
            AT_shared[(A_tcol * 4 + 0) * bm + A_trow + offset] = A_tmp.x;
            AT_shared[(A_tcol * 4 + 1) * bm + A_trow + offset] = A_tmp.y;
            AT_shared[(A_tcol * 4 + 2) * bm + A_trow + offset] = A_tmp.z;
            AT_shared[(A_tcol * 4 + 3) * bm + A_trow + offset] = A_tmp.w;
        }
        for(int offset = 0; offset + strideB <= bk; offset += strideB)
            reinterpret_cast<float4 *>(&B_shared[(B_trow + offset) * bn + B_tcol * 4])[0]
             = reinterpret_cast<const float4 *>(&B[(B_trow + offset) * n + B_tcol * 4])[0];
        __syncthreads();

        // perform matrix multiplication using WMMA API
        for(int frag = 0; frag < bk; frag += 8) {
            // load elements into fragments
            for(int innerw_row = 0; innerw_row < wmiter; innerw_row++) {
                wmma::load_matrix_sync(a_frag[innerw_row], (AT_shared + wrow * wm) + frag * bm + innerw_row * 16, bm);
            }
            for(int innerw_col = 0; innerw_col < wniter; innerw_col++) {
                wmma::load_matrix_sync(b_frag[innerw_col], (B_shared + wcol * wn) + frag * bn + innerw_col * 16, bn);
            }
            // multiplicate fragments
            for(int innerw_row = 0; innerw_row < wmiter; innerw_row++) {
                for(int innerw_col = 0; innerw_col < wniter; innerw_col++) {
                    wmma::mma_sync(c_frag[innerw_row * wniter + innerw_col], a_frag[innerw_row], b_frag[innerw_col], c_frag[innerw_row * wniter + innerw_col]);
                }
            }
        }

        // move on to the next block
        A += bk;
        B += bk * n;
    }

    //Store the result back to the output matrix
    for(int innerw_row = 0; innerw_row < wmiter; innerw_row++) {
        for(int innerw_col = 0; innerw_col < wniter; innerw_col++) {
            wmma::store_matrix_sync(C + (wrow * wm) * n + wcol * wn + (innerw_row * 16) * n + innerw_col * 16, c_frag[innerw_row * wniter + innerw_col], n, wmma::mem_row_major);
        }
    }
}