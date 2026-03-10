#pragma once
#pragma nv_diag_suppress static_var_with_dynamic_init

#include <cstdio>
#include <cooperative_groups.h>
#include <cuda/barrier>
#include "mma.h"
using namespace nvcuda;

template <const uint bm, const uint bn, const uint bk, const int wm, const int wn, const int num_threads>
__global__ void shm_revised_bypass_tf(float *__restrict A, float *__restrict B, float *C, int m, int n, int k, float alpha, float beta) {
    auto block = cooperative_groups::this_thread_block();
    __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> barrier;
    
    auto barrier_ptr = &barrier;
    if(block.thread_rank() == 0)
        init(&barrier, block.size());
    block.sync();

    __shared__ float A_shared[bm * bk];
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
    // const int wsubm = 16;
    // const int wsubn = 16;
    const int wmiter = wm / 16;
    const int wniter = wn / 16;

    // Define WMMA fragment types for TF32 precision
    wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> a_frag[wmiter];
    wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major> b_frag[wniter];
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> c_frag[wmiter * wniter];

    // Initialize output fragment
#pragma unroll
    for(int i = 0; i < wmiter * wniter ;i++)
        wmma::fill_fragment(c_frag[i], 0.0f);

#pragma unroll
    for(int i = 0; i < k; i += bk) {
        // load elements from global memory to shared memory
#pragma unroll
        for(int offset = 0; offset + strideA <= bm; offset += strideA) {
            // reinterpret_cast<float4 *>(&A_shared[A_trow * bk + A_tcol * 4 + offset * bk])[0] 
            // = reinterpret_cast<const float4 *>(&A[(A_trow + offset) * k + A_tcol * 4])[0];
            cuda::memcpy_async(&A_shared[A_trow * bk + A_tcol * 4 + offset * bk],
                                &A[(A_trow + offset) * k + A_tcol * 4],
                                sizeof(float4),
                                barrier);
        }
#pragma unroll
        for(int offset = 0; offset + strideB <= bk; offset += strideB) {
            // reinterpret_cast<float4 *>(&B_shared[(B_trow + offset) * bn + B_tcol * 4])[0]
            //  = reinterpret_cast<const float4 *>(&B[(B_trow + offset) * n + B_tcol * 4])[0];
            cuda::memcpy_async(&B_shared[(B_trow + offset) * bn + B_tcol * 4],
                                &B[(B_trow + offset) * n + B_tcol * 4],
                                sizeof(float4),
                                barrier);
        }
        
        //__syncthreads();
        (*barrier_ptr).arrive_and_wait();

        // perform matrix multiplication using WMMA API
#pragma unroll
        for(int frag = 0; frag < bk; frag += 8) {
            // load elements into fragments
#pragma unroll
            for(int innerw_row = 0; innerw_row < wmiter; innerw_row++) {
                wmma::load_matrix_sync(a_frag[innerw_row], A_shared + wrow * wm * bk + frag + innerw_row * 16 * bk, bk);
            }
#pragma unroll
            for(int innerw_col = 0; innerw_col < wniter; innerw_col++) {
                wmma::load_matrix_sync(b_frag[innerw_col], B_shared + wcol * wn + frag * bn + innerw_col * 16, bn);
            }
            // multiplicate fragments
#pragma unroll
            for(int innerw_row = 0; innerw_row < wmiter; innerw_row++) {
#pragma unroll
                for(int innerw_col = 0; innerw_col < wniter; innerw_col++) {
                    wmma::mma_sync(c_frag[innerw_row * wniter + innerw_col], a_frag[innerw_row], b_frag[innerw_col], c_frag[innerw_row * wniter + innerw_col]);
                }
            }
        }
        // move on to the next block
        A += bk;
        B += bk * n;

        block.sync();
    }

    //Store the result back to the output matrix
#pragma unroll
    for(int innerw_row = 0; innerw_row < wmiter; innerw_row++) {
#pragma unroll
        for(int innerw_col = 0; innerw_col < wniter; innerw_col++) {
            wmma::store_matrix_sync(C + (wrow * wm) * n + wcol * wn + (innerw_row * 16) * n + innerw_col * 16, c_frag[innerw_row * wniter + innerw_col], n, wmma::mem_row_major);
            //__syncwarp();
        }
    }
}