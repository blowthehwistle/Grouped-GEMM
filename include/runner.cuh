#include "cuda_kernels.cuh"
#include "helpers.h"
#include "tensor_kernels.cuh"

void run_naive_fp(float *A, float *B, float *C, int m, int n, int k, float alpha, float beta) {
  dim3 blockDim(32, 32);
  dim3 gridDim(ceil_div(n, 32), ceil_div(m, 32));
  naive_fp<<<gridDim, blockDim>>>(A, B, C, m, n, k, alpha, beta);
}

void run_global_coalesce_fp(float *A, float *B, float *C, int m, int n, int k, float alpha,
                            float beta) {
  dim3 blockDim(32 * 32);
  dim3 gridDim(ceil_div(n, 32), ceil_div(m, 32));
  global_coalesce_fp<32><<<gridDim, blockDim>>>(A, B, C, m, n, k, alpha, beta);
}

void run_shared_caching_fp(float *A, float *B, float *C, int m, int n, int k, float alpha,
                           float beta) {
  dim3 blockDim(32 * 32);
  dim3 gridDim(ceil_div(n, 32), ceil_div(m, 32));
  shared_caching_fp<32><<<gridDim, blockDim>>>(A, B, C, m, n, k, alpha, beta);
}

void run_blocking_1d_fp(float *A, float *B, float *C, int m, int n, int k, float alpha,
                        float beta) {
  const uint bm = 64;
  const uint bn = 64;
  const uint bk = 8;
  const uint tw = 8;
  dim3 blockDim((bm / tw) * bn);
  dim3 gridDim(ceil_div(n, bn), ceil_div(m, bm));
  blocking_1d_fp<bm, bn, bk, tw><<<gridDim, blockDim>>>(A, B, C, m, n, k, alpha, beta);
}

void run_blocking_2d_fp(float *A, float *B, float *C, int m, int n, int k, float alpha,
                        float beta) {
  const uint bm = 128;
  const uint bn = 128;
  const uint bk = 8;
  const uint tw_m = 8;
  const uint tw_n = 8;
  dim3 blockDim((bm / tw_m) * (bn / tw_n));
  dim3 gridDim(ceil_div(n, bn), ceil_div(m, bm));
  blocking_2d_fp<bm, bn, bk, tw_m, tw_n><<<gridDim, blockDim>>>(A, B, C, m, n, k, alpha, beta);
}

void run_vectorized_fp(float *A, float *B, float *C, int m, int n, int k, float alpha, float beta) {
  const uint k6_bm = 128;
  const uint k6_bn = 128;
  const uint k6_bk = 8;
  const uint k6_tw_m = 8;
  const uint k6_tw_n = 8;
  dim3 blockDim((k6_bm / k6_tw_m) * (k6_bn / k6_tw_n));
  dim3 gridDim(ceil_div(n, k6_bn), ceil_div(m, k6_bm));
  vectorized_fp<k6_bm, k6_bn, k6_bk, k6_tw_m, k6_tw_n>
      <<<gridDim, blockDim>>>(A, B, C, m, n, k, alpha, beta);
}

void run_vectorized_fp_revised(float *A, float *B, float *C, int m, int n, int k, float alpha,
                               float beta) {
  const uint k7_bm = 128;
  const uint k7_bn = 128;
  const uint k7_bk = 8;
  const uint k7_tw_m = 8;
  const uint k7_tw_n = 8;
  dim3 blockDim((k7_bm / k7_tw_m) * (k7_bn / k7_tw_n));
  dim3 gridDim(ceil_div(n, k7_bn), ceil_div(m, k7_bm));
  vectorized_fp_revised<k7_bm, k7_bn, k7_bk, k7_tw_m, k7_tw_n>
      <<<gridDim, blockDim>>>(A, B, C, m, n, k, alpha, beta);
}

void run_resolve_bank_conflict(float *A, float *B, float *C, int m, int n, int k, float alpha,
                               float beta) {
  const uint k8_bm = 128;
  const uint k8_bn = 128;
  const uint k8_bk = 8;
  const uint k8_tw_m = 8;
  const uint k8_tw_n = 8;
  dim3 blockDim((k8_bm / k8_tw_m) * (k8_bn / k8_tw_n));
  dim3 gridDim(ceil_div(n, k8_bn), ceil_div(m, k8_bm));
  resolve_bank_conflict<k8_bm, k8_bn, k8_bk, k8_tw_m, k8_tw_n>
      <<<gridDim, blockDim>>>(A, B, C, m, n, k, alpha, beta);
}

// 4k best : num_threads=128, bm=128, bn=128, bk=16, wm=64, wn=64, wniter=2, tw=4, tn=8
void run_warptiling(float *A, float *B, float *C, int m, int n, int k, float alpha, float beta) {
  const uint k10_num_threads = 128;
  const uint k10_bm = 256;
  const uint k10_bn = 64;
  const uint k10_bk = 16;
  const uint k10_wm = 128;
  const uint k10_wn = 32;
  const uint k10_wniter = 4;
  const uint k10_tw_m = 4;
  const uint k10_tw_n = 4;
  dim3 blockDim(k10_num_threads);
  dim3 gridDim(ceil_div(n, k10_bn), ceil_div(m, k10_bm));
  warptiling_fp<k10_bm, k10_bn, k10_bk, k10_wm, k10_wn, k10_wniter, k10_tw_m, k10_tw_n,
                k10_num_threads><<<gridDim, blockDim>>>(A, B, C, m, n, k, alpha, beta);
}

void run_double_buffering(float *A, float *B, float *C, int m, int n, int k, float alpha,
                          float beta) {
  const uint k11_num_threads = 128;
  const uint k11_bm = 128;
  const uint k11_bn = 128;
  const uint k11_bk = 16;
  const uint k11_wm = 64;
  const uint k11_wn = 64;
  const uint k11_wniter = 2;
  const uint k11_tw_m = 4;
  const uint k11_tw_n = 8;
  dim3 blockDim(k11_num_threads);
  dim3 gridDim(ceil_div(n, k11_bn), ceil_div(m, k11_bm));
  double_buffering<k11_bm, k11_bn, k11_bk, k11_wm, k11_wn, k11_wniter, k11_tw_m, k11_tw_n,
                   k11_num_threads><<<gridDim, blockDim>>>(A, B, C, m, n, k, alpha, beta);
}

void runCublasFP32(cublasHandle_t handle, float *A, float *B, float *C, int m, int n, int k,
                   float alpha, float beta) {
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, B, CUDA_R_32F, n, A, CUDA_R_32F,
               k, &beta, C, CUDA_R_32F, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

void run_global_tf(float *A, float *B, float *C, int m, int n, int k, float alpha, float beta) {
  dim3 blockDim(16 * 16);
  dim3 gridDim(ceil_div(n * m, 16 * 16 * 8));
  global_tf<<<gridDim, blockDim>>>(A, B, C, m, n, k, alpha, beta);
  cudaDeviceSynchronize();
}

/* current best for 4k : num_threads=128, bm=64, bn=256, bk=8, wm=64, wn=64
                       num_threads=128, bm=128, bn=128, bk=8, wm=64, wn=64 */
void run_shared_tf(float *A, float *B, float *C, int m, int n, int k, float alpha, float beta) {
  const uint t_k2_bm = 64;
  const uint t_k2_bn = 256;
  const uint t_k2_bk = 8;
  const uint t_k2_wm = 64;
  const uint t_k2_wn = 64;
  const uint t_k2_num_threads = 128;
  dim3 blockDim(t_k2_num_threads);
  dim3 gridDim(ceil_div(n, t_k2_bn), ceil_div(m, t_k2_bm));
  shm_tf<t_k2_bm, t_k2_bn, t_k2_bk, t_k2_wm, t_k2_wn, t_k2_num_threads>
      <<<gridDim, blockDim>>>(A, B, C, m, n, k, alpha, beta);
}

// RTX 4060
// current best for 4k : num_threads=128, bm=64, bn=256, bk=8, wm=64, wn=64
// current best for 8k : num_threads=128, bm=128, bn=128, bk=16, wm=32, wn=128
// current best for 16k : num_threads=128, bm=128, bn=128, bk=8, wm=64, wn=64

// RTX 4090
// current best for 4k : num_threads=128, bm=128, bn=128, bk=8, wm=128, wn=32
// current best for 8k : num_threads=128, bm=128, bn=128, bk=8, wm=64, wn=64
// current best for 16k : num_threads=128, bm=256, bn=64, bk=8, wm=128, wn=32
void run_shared_revised_tf(float *A, float *B, float *C, int m, int n, int k, float alpha,
                           float beta) {
  const uint t_k3_bm = 64;
  const uint t_k3_bn = 256;
  const uint t_k3_bk = 8;
  const uint t_k3_wm = 64;
  const uint t_k3_wn = 64;
  const uint t_k3_num_threads = 128;
  dim3 blockDim(t_k3_num_threads);
  dim3 gridDim(ceil_div(n, t_k3_bn), ceil_div(m, t_k3_bm));
  shm_revised_tf<t_k3_bm, t_k3_bn, t_k3_bk, t_k3_wm, t_k3_wn, t_k3_num_threads>
      <<<gridDim, blockDim>>>(A, B, C, m, n, k, alpha, beta);
}

// for 4090
// 4k : NUM_THREADS=128 BM=128 BN=128 BK=32 WM=64 WN=64
void run_shared_bypass_tf(float *A, float *B, float *C, int m, int n, int k, float alpha,
                          float beta) {
  const uint t_k4_bm = 128;
  const uint t_k4_bn = 128;
  const uint t_k4_bk = 32;
  const uint t_k4_wm = 64;
  const uint t_k4_wn = 64;
  const uint t_k4_num_threads = 128;
  dim3 blockDim(t_k4_num_threads);
  dim3 gridDim(ceil_div(n, t_k4_bn), ceil_div(m, t_k4_bm));
  shm_revised_bypass_tf<t_k4_bm, t_k4_bn, t_k4_bk, t_k4_wm, t_k4_wn, t_k4_num_threads>
      <<<gridDim, blockDim>>>(A, B, C, m, n, k, alpha, beta);
}

void run_double_buffering_tf(float *A, float *B, float *C, int m, int n, int k, float alpha,
                             float beta) {
  const uint t_k5_bm = 128;
  const uint t_k5_bn = 128;
  const uint t_k5_bk = 16;
  const uint t_k5_wm = 64;
  const uint t_k5_wn = 64;
  const uint t_k5_num_threads = 128;
  const int shm_size = sizeof(float) * (2 * t_k5_bm * t_k5_bk + 2 * t_k5_bk * t_k5_bn);
  dim3 blockDim(t_k5_num_threads);
  dim3 gridDim(ceil_div(n, t_k5_bn), ceil_div(m, t_k5_bm));
  double_buffering_tf<t_k5_bm, t_k5_bn, t_k5_bk, t_k5_wm, t_k5_wn, t_k5_num_threads>
      <<<gridDim, blockDim, shm_size>>>(A, B, C, m, n, k, alpha, beta);
}

void run_double_buffering_tf_batched(float *A, float *B, float *C, int m, int n, int k,
                                     int batch_size, float alpha, float beta) {
  const uint t_k5_bm = 128;
  const uint t_k5_bn = 128;
  const uint t_k5_bk = 16;
  const uint t_k5_wm = 64;
  const uint t_k5_wn = 64;
  const uint t_k5_num_threads = 128;
  const int shm_size = sizeof(float) * (2 * t_k5_bm * t_k5_bk + 2 * t_k5_bk * t_k5_bn);
  dim3 blockDim(t_k5_num_threads);
  dim3 gridDim(ceil_div(n, t_k5_bn) * ceil_div(m, t_k5_bm), batch_size);
  double_buffering_tf_batched<t_k5_bm, t_k5_bn, t_k5_bk, t_k5_wm, t_k5_wn, t_k5_num_threads>
      <<<gridDim, blockDim, shm_size>>>(A, B, C, m, n, k, batch_size, alpha, beta);
}

void run_double_buffering_tf_grouped(float *A, float *B, float *C, int *m_list, int *n_list, int k,
                                     int batch_size, int *A_offset, int *B_offset, int *C_offset,
                                     int *d_m_list, int *d_n_list, int *d_A_offset, int *d_B_offset,
                                     int *d_C_offset, float alpha, float beta) {
  const uint t_k5_bm = 128;
  const uint t_k5_bn = 128;
  const uint t_k5_bk = 16;
  const uint t_k5_wm = 64;
  const uint t_k5_wn = 64;
  const uint t_k5_num_threads = 128;
  const int shm_size = sizeof(float) * (2 * t_k5_bm * (t_k5_bk + 8) + 2 * t_k5_bk * (t_k5_bn + 16));

  // cudaDeviceProp prop;
  // cudaGetDeviceProperties(&prop, 0);
  // printf("shm_size: %d\n", shm_size);
  // printf("Max shared memory per block: %zu bytes\n", prop.sharedMemPerBlock);

  uint grid_dim = 0;
  for (int batch = 0; batch < batch_size; batch++) {
    grid_dim += ceil_div(n_list[batch], t_k5_bn) * ceil_div(m_list[batch], t_k5_bm);
  }
  dim3 blockDim(t_k5_num_threads);
  // dim3 gridDim(ceil_div(n_list[0], t_k5_bn) * ceil_div(m_list[0], t_k5_bm), batch_size);
  dim3 gridDim(grid_dim);
  double_buffering_tf_vbatched<t_k5_bm, t_k5_bn, t_k5_bk, t_k5_wm, t_k5_wn, t_k5_num_threads>
      <<<gridDim, blockDim, shm_size>>>(A, B, C, d_m_list, d_n_list, k, batch_size, d_A_offset,
                                        d_B_offset, d_C_offset, alpha, beta);

  cudaDeviceSynchronize();
  // Check for the last error
  // cudaError_t err = cudaGetLastError();
  // if (err != cudaSuccess) {
  //   std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
  //   exit(1);
  // } else {
  //   std::cout << "No CUDA errors detected." << std::endl;
  // }
}

void runCublasTF32_with_TC(cublasHandle_t handle, float *A, float *B, float *C, int m, int n, int k,
                           float alpha, float beta) {
  cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, B, CUDA_R_32F, n, A, CUDA_R_32F,
               k, &beta, C, CUDA_R_32F, n, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT);
  // cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
}

void runCublasTF32_with_TC(cublasHandle_t handle, float *A, float *B, float *C, int m, int n, int k,
                           int batch_size, float alpha, float beta) {
  /**
   * @brief Performs a strided batched GEMM (General Matrix Multiply) operation using cuBLAS.
   * This function computes multiple matrix multiplications in a batched manner.
   */
  cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
  cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, B, CUDA_R_32F, n,
                             k * n, A, CUDA_R_32F, k, m * k, &beta, C, CUDA_R_32F, n, m * n,
                             batch_size, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT);
  // cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
}

void runCublasTF32_with_TC(cublasHandle_t handle, float *A, float *B, float *C, int *m_list,
                           int *n_list, int *k_list, int batch_size, int *A_offset, int *B_offset,
                           int *C_offset, float alpha, float beta) {
  /**
   * @brief Performs a strided batched GEMM (General Matrix Multiply) operation using cuBLAS.
   * This function computes multiple matrix multiplications in a batched manner.
   */
  cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
  for (int batch = 0; batch < batch_size; batch++) {
    int m = m_list[batch];
    int n = n_list[batch];
    int k = k_list[batch];
    float *A_ptr = A + A_offset[batch];
    float *B_ptr = B + B_offset[batch];
    float *C_ptr = C + C_offset[batch];
    // printf("%d %d %d %d %d\n", m, n, A_offset[batch], B_offset[batch], C_offset[batch]);
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, B_ptr, CUDA_R_32F, n, A_ptr,
                 CUDA_R_32F, k, &beta, C_ptr, CUDA_R_32F, n, CUBLAS_COMPUTE_32F_FAST_TF32,
                 CUBLAS_GEMM_DEFAULT);
  }
  // cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
}

void runCublasGroupedTF32_with_TC(cublasHandle_t handle, float *A, float *B, float *C, int *m_list,
                                  int *n_list, int *k_list, int batch_size, int *A_offset,
                                  int *B_offset, int *C_offset, float alpha, float bet) {
  cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);

  // Create array for grouped matrices
  float *A_array[batch_size], *B_array[batch_size], *C_array[batch_size];

  for (int batch = 0; batch < batch_size; batch++) {
    A_array[batch] = A + A_offset[batch];
    B_array[batch] = B + B_offset[batch];
    C_array[batch] = C + C_offset[batch];
  }

  float **d_A_array = nullptr;
  float **d_B_array = nullptr;
  float **d_C_array = nullptr;

  CHECK_CUDA(cudaMalloc(&d_A_array, batch_size * sizeof(float *)));
  CHECK_CUDA(cudaMalloc(&d_B_array, batch_size * sizeof(float *)));
  CHECK_CUDA(cudaMalloc(&d_C_array, batch_size * sizeof(float *)));

  CHECK_CUDA(cudaMemcpy(d_A_array, A_array, batch_size * sizeof(float *), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_B_array, B_array, batch_size * sizeof(float *), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_C_array, C_array, batch_size * sizeof(float *), cudaMemcpyHostToDevice));

  // Set grouped batched gemm configurations for cublasGemmGroupedBatchedEx
  cublasOperation_t transa_array[batch_size];
  cublasOperation_t transb_array[batch_size];
  int lda_array[batch_size];
  int ldb_array[batch_size];
  int ldc_array[batch_size];
  float alpha_array[batch_size];
  float beta_array[batch_size];
  int group_size[batch_size];

  for (int batch = 0; batch < batch_size; batch++) {
    transa_array[batch] = CUBLAS_OP_N;
    transb_array[batch] = CUBLAS_OP_N;
    lda_array[batch] = k_list[batch];   // A(M,K) row-major
    ldb_array[batch] = n_list[batch];   // B(K,N) row-major
    ldc_array[batch] = n_list[batch];   // C(M,N) row-major
    alpha_array[batch] = 1.0;
    beta_array[batch] = 0.0;
    group_size[batch] = 1;
  }

  cublasGemmGroupedBatchedEx(
      handle, transa_array, transb_array, m_list, n_list, k_list, alpha_array,
      (void const *const *)d_A_array, CUDA_R_32F, lda_array, (void const *const *)d_B_array,
      CUDA_R_32F, ldb_array, beta_array, (void *const *)d_C_array, CUDA_R_32F, ldc_array,
      batch_size, group_size, CUBLAS_COMPUTE_32F_FAST_TF32);

  cudaFree(d_A_array);
  cudaFree(d_B_array);
  cudaFree(d_C_array);
}

void runCublasTF16_with_TC(cublasHandle_t handle, __half *A, __half *B, __half *C, int *m_list,
                           int *n_list, int k, int batch_size, int *A_offset, int *B_offset,
                           int *C_offset, float alpha, float beta) {
  /**
   * @brief Performs a strided batched GEMM (General Matrix Multiply) operation using cuBLAS.
   * This function computes multiple matrix multiplications in a batched manner.
   */
  // cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
  for (int batch = 0; batch < batch_size; batch++) {
    int m = m_list[batch];
    int n = n_list[batch];
    __half *A_ptr = A + A_offset[batch];
    __half *B_ptr = B + B_offset[batch];
    __half *C_ptr = C + C_offset[batch];
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, B_ptr, CUDA_R_16F, n, A_ptr,
                 CUDA_R_16F, k, &beta, C_ptr, CUDA_R_16F, n, CUBLAS_COMPUTE_32F,
                 CUBLAS_GEMM_DEFAULT);
  }
}

void runCublasGroupedTF16_with_TC(cublasHandle_t handle, __half *A, __half *B, __half *C,
                                  int *m_list, int *n_list, int *k_list, int batch_size,
                                  int *A_offset, int *B_offset, int *C_offset, float alpha,
                                  float bet) {
  // cublasGemmGroupedBatchedEx는 cuBLAS 12+에서만 제공되므로, 호환성을 위해 루프로 cublasGemmEx 사용
  cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
  for (int batch = 0; batch < batch_size; batch++) {
    int m = m_list[batch];
    int n = n_list[batch];
    int k = k_list[batch];
    __half *A_ptr = A + A_offset[batch];
    __half *B_ptr = B + B_offset[batch];
    __half *C_ptr = C + C_offset[batch];
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, B_ptr, CUDA_R_16F, n, A_ptr,
                 CUDA_R_16F, k, &bet, C_ptr, CUDA_R_16F, n, CUBLAS_COMPUTE_32F,
                 CUBLAS_GEMM_DEFAULT);
  }
}