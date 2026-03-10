#pragma once

#include <cstdio>
#include "mma.h"

using namespace nvcuda;

__global__ void global_tf(const float *A, const float *B, float *C, int m, int n, int k, float alpha, float beta) {
    // Define WMMA fragment types for TF32 precision
    wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> c_frag;

    // Initialize output fragment
    wmma::fill_fragment(c_frag, 0.0f);
    
    // warp index among total threads 
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    
    // index of each matrices' tile
    const int Crow = warp_id / (n / 16);
    const int Ccol = warp_id % (n / 16);
    int Arow = Crow * (16 * k);
    int Bcol = Ccol * 16;
    
    for(int i = 0; i < k; i += 8) {
        // Load the input matrices into fragments
        wmma::load_matrix_sync(a_frag, A + Arow + i, k);
        wmma::load_matrix_sync(b_frag, B + Bcol + i * n, n);
        // Perform the matrix multiplication using Tensor Cores
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    // for(int i = 0; i < c_frag.num_elements; i++)
    //     c_frag.x[i] = alpha * c_frag.x[i] + beta * C[Crow * n + Ccol + i];

    // Store the result back to the output matrix
    wmma::store_matrix_sync(C + (Crow * n + Ccol) * 16, c_frag, n, wmma::mem_row_major);    
}

__global__ void global_bf(const __nv_bfloat16 *A, const __nv_bfloat16 *B, float *C, int m, int n, int k, float alpha, float beta) {
    // Define WMMA fragment types for TF32 precision
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    // Initialize output fragment
    wmma::fill_fragment(c_frag, 0.0f);
    
    // warp index among total threads 
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    
    // index of each matrices' tile
    const int Crow = warp_id / (n / 16);
    const int Ccol = warp_id % (n / 16);
    int Arow = Crow * (16 * k);
    int Bcol = Ccol * 16;
    
    for(int i = 0; i < k; i += 16) {
        // Load the input matrices into fragments
        wmma::load_matrix_sync(a_frag, A + Arow + i, k);
        wmma::load_matrix_sync(b_frag, B + Bcol + i * n, n);
        // Perform the matrix multiplication using Tensor Cores
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    for(int i = 0; i < c_frag.num_elements; i++)
        c_frag.x[i] = alpha * c_frag.x[i] + beta * C[Crow * n + Ccol + i];

    // Store the result back to the output matrix
    wmma::store_matrix_sync(C + (Crow * n + Ccol) * 16, c_frag, n, wmma::mem_row_major);
}

__global__ void global_h(const half *A, const half *B, float *C, int m, int n, int k, float alpha, float beta) {
    // Define WMMA fragment types for TF32 precision
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    // Initialize output fragment
    wmma::fill_fragment(c_frag, 0.0f);
    
    // warp index among total threads 
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    
    // index of each matrices' tile
    const int Crow = warp_id / (n / 16);
    const int Ccol = warp_id % (n / 16);
    int Arow = Crow * (16 * k);
    int Bcol = Ccol * 16;
    
    for(int i = 0; i < k; i += 16) {
        // Load the input matrices into fragments
        wmma::load_matrix_sync(a_frag, A + Arow + i, k);
        wmma::load_matrix_sync(b_frag, B + Bcol + i * n, n);
        // Perform the matrix multiplication using Tensor Cores
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    for(int i = 0; i < c_frag.num_elements; i++)
        c_frag.x[i] = alpha * c_frag.x[i] + beta * C[Crow * n + Ccol + i];

    // Store the result back to the output matrix
    wmma::store_matrix_sync(C + (Crow * n + Ccol) * 16, c_frag, n, wmma::mem_row_major);
}