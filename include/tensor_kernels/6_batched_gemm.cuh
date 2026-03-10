#pragma once
#pragma nv_diag_suppress static_var_with_dynamic_init

#include <cooperative_groups.h>

#include <cstdio>
#include <cuda/barrier>
#include <cuda/pipeline>

#include "mma.h"
using namespace nvcuda;

template <const uint bm, const uint bn, const uint bk, const int wm, const int wn,
          const int num_threads>
__global__ void double_buffering_tf_batched(float *__restrict A, float *__restrict B, float *C,
                                            int m, int n, int k, int batch_size, float alpha,
                                            float beta) {
  auto thread = cooperative_groups::this_thread();
  auto block = cooperative_groups::this_thread_block();
  // printf("check\n");

  constexpr size_t stages_count = 2;
  __shared__ cuda::pipeline_shared_state<cuda::thread_scope::thread_scope_block, stages_count>
      shared_state;
  auto pipeline = cuda::make_pipeline(block, &shared_state);

  extern __shared__ __align__(128) float shared_memory[];
  float *A_shared = shared_memory;
  float *B_shared = shared_memory + (2 * bm * bk);

  // coordinate of block in grid
  // const int brow = blockIdx.y;
  // const int bcol = blockIdx.x;
  const int brow = blockIdx.x / ceil_div(n, bn);
  const int bcol = blockIdx.x % ceil_div(n, bn);
  const int b_batch = blockIdx.y;

  // starting pointer of each thread block in output matrix
  A += m * k * b_batch + brow * bm * k;
  B += k * n * b_batch + bcol * bn;
  C += m * n * b_batch + brow * bm * n + bcol * bn;

  // coordinate of each thread in thread block for using LSD.128
  // single load instruction load 4 float elements
  const int A_trow = threadIdx.x / (bk / 4);
  const int A_tcol = threadIdx.x % (bk / 4) * 4;
  const int B_trow = threadIdx.x / (bn / 4);
  const int B_tcol = threadIdx.x % (bn / 4) * 4;

  // stride for loading input matrices to shared memory
  const int strideA = (num_threads * 4) / bk;
  const int strideB = (num_threads * 4) / bn;

  // coordinate of warp in thread block
  const int widx = threadIdx.x / WARP_SIZE;
  const int wrow = widx / (bn / wn);
  const int wcol = widx % (bn / wn);

  // inner warp related variables
  const int wmiter = wm / 16;
  const int wniter = wn / 16;

  // Define WMMA fragment types for TF32 precision
  wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> a_frag[wmiter];
  wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major> b_frag[wniter];
  wmma::fragment<wmma::accumulator, 16, 16, 8, float> c_frag[wmiter * wniter];

  int shm_offset = 0;

  // Initialize output fragment
#pragma unroll
  for (int i = 0; i < wmiter * wniter; i++) wmma::fill_fragment(c_frag[i], 0.0f);

  pipeline.producer_acquire();
  for (int offset = 0; offset + strideA <= bm; offset += strideA) {
    cuda::memcpy_async(
        thread, reinterpret_cast<float4 *>(&A_shared[A_trow * bk + A_tcol + offset * bk]),
        reinterpret_cast<float4 *>(&A[(A_trow + offset) * k + A_tcol]), sizeof(float4), pipeline);
  }
  for (int offset = 0; offset + strideB <= bk; offset += strideB) {
    cuda::memcpy_async(
        thread, reinterpret_cast<float4 *>(&B_shared[(B_trow + offset) * bn + B_tcol]),
        reinterpret_cast<float4 *>(&B[(B_trow + offset) * n + B_tcol]), sizeof(float4), pipeline);
  }
  pipeline.producer_commit();

#pragma unroll
  for (int i = 0; i < k - bk; i += bk) {
    // load elements from global memory to shared memory
    pipeline.producer_acquire();
    for (int offset = 0; offset + strideA <= bm; offset += strideA) {
      cuda::memcpy_async(
          thread,
          reinterpret_cast<float4 *>(
              &A_shared[(1 - shm_offset) * bm * bk + A_trow * bk + A_tcol + offset * bk]),
          reinterpret_cast<float4 *>(&A[(A_trow + offset) * k + A_tcol + bk]), sizeof(float4),
          pipeline);
    }
    for (int offset = 0; offset + strideB <= bk; offset += strideB) {
      cuda::memcpy_async(
          thread,
          reinterpret_cast<float4 *>(
              &B_shared[(1 - shm_offset) * bk * bn + (B_trow + offset) * bn + B_tcol]),
          reinterpret_cast<float4 *>(&B[(B_trow + offset) * n + B_tcol + bk * n]), sizeof(float4),
          pipeline);
    }
    pipeline.producer_commit();

    pipeline.consumer_wait();

    // perform matrix multiplication using WMMA API
    for (int frag = 0; frag < bk; frag += 8) {
// load elements into fragments
#pragma unroll
      for (int innerw_row = 0; innerw_row < wmiter; innerw_row++) {
        wmma::load_matrix_sync(
            a_frag[innerw_row],
            A_shared + shm_offset * bm * bk + wrow * wm * bk + frag + innerw_row * 16 * bk, bk);
      }
#pragma unroll
      for (int innerw_col = 0; innerw_col < wniter; innerw_col++) {
        wmma::load_matrix_sync(
            b_frag[innerw_col],
            B_shared + shm_offset * bk * bn + wcol * wn + frag * bn + innerw_col * 16, bn);
      }
// multiplicate fragments
#pragma unroll
      for (int innerw_row = 0; innerw_row < wmiter; innerw_row++) {
#pragma unroll
        for (int innerw_col = 0; innerw_col < wniter; innerw_col++) {
          wmma::mma_sync(c_frag[innerw_row * wniter + innerw_col], a_frag[innerw_row],
                         b_frag[innerw_col], c_frag[innerw_row * wniter + innerw_col]);
        }
      }
    }
    pipeline.consumer_release();

    // move on to the next block
    A += bk;
    B += bk * n;

    shm_offset = 1 - shm_offset;
  }

  pipeline.consumer_wait();

  // perform matrix multiplication using WMMA API
  for (int frag = 0; frag < bk; frag += 8) {
// load elements into fragments
#pragma unroll
    for (int innerw_row = 0; innerw_row < wmiter; innerw_row++) {
      wmma::load_matrix_sync(
          a_frag[innerw_row],
          A_shared + shm_offset * bm * bk + wrow * wm * bk + frag + innerw_row * 16 * bk, bk);
    }
#pragma unroll
    for (int innerw_col = 0; innerw_col < wniter; innerw_col++) {
      wmma::load_matrix_sync(
          b_frag[innerw_col],
          B_shared + shm_offset * bk * bn + wcol * wn + frag * bn + innerw_col * 16, bn);
    }
// multiplicate fragments
#pragma unroll
    for (int innerw_row = 0; innerw_row < wmiter; innerw_row++) {
#pragma unroll
      for (int innerw_col = 0; innerw_col < wniter; innerw_col++) {
        wmma::mma_sync(c_frag[innerw_row * wniter + innerw_col], a_frag[innerw_row],
                       b_frag[innerw_col], c_frag[innerw_row * wniter + innerw_col]);
      }
    }
  }
  pipeline.consumer_release();

// Store the result back to the output matrix
#pragma unroll
  for (int innerw_row = 0; innerw_row < wmiter; innerw_row++) {
#pragma unroll
    for (int innerw_col = 0; innerw_col < wniter; innerw_col++) {
      wmma::store_matrix_sync(
          C + (wrow * wm) * n + wcol * wn + (innerw_row * 16) * n + innerw_col * 16,
          c_frag[innerw_row * wniter + innerw_col], n, wmma::mem_row_major);
    }
  }
}