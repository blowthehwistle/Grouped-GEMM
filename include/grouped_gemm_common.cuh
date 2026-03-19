/**
 * Shared setup/teardown and reference run for grouped GEMM programs.
 * Used by grouped_gemm_loop, grouped_gemm_grouped, grouped_gemm_custom.
 */
#pragma once

#include "config_io.h"
#include "helpers.h"
#include "runner.cuh"
#include "nvtx3/nvToolsExt.h"

#include <cstdlib>
#include <cstring>
#include <iostream>

struct GroupedGemmParams {
  cublasHandle_t handle{};
  float alpha = 1.0f, beta = 0.0f;
  long batch_size = 0;
  int *m_list = nullptr;
  int *n_list = nullptr;
  int *k_list = nullptr;
  int *A_offset = nullptr;
  int *B_offset = nullptr;
  int *C_offset = nullptr;
  size_t size_A = 0, size_B = 0, size_C = 0, size_C_ref = 0;

  float *A = nullptr;
  float *B = nullptr;
  float *C = nullptr;
  float *C_ref = nullptr;
  float *d_A = nullptr;
  float *d_B = nullptr;
  float *d_C = nullptr;
  float *d_C_ref = nullptr;
  int *d_m_list = nullptr;
  int *d_n_list = nullptr;
  int *d_k_list = nullptr;
  int *d_A_offset = nullptr;
  int *d_B_offset = nullptr;
  int *d_C_offset = nullptr;

  cudaEvent_t start = nullptr, stop = nullptr;
  const char *result_mode = "p";
  int repeat_time = 1000;
};

// Same warmup as Torch/Triton/Cutlass (50). Used by grouped_gemm_loop/grouped/custom.
constexpr int GROUPED_GEMM_WARMUP = 50;

static void randomize_matrix_s(int N, float *M) {
  struct timeval time {};
  gettimeofday(&time, nullptr);
  srand(static_cast<unsigned>(time.tv_usec));
  for (int i = 0; i < N; i++) {
    float tmp = (float)(rand() % 5) + 0.01f * (rand() % 5);
    tmp = (rand() % 2 == 0) ? tmp : -tmp;
    M[i] = tmp;
  }
}

/** Run cuBLAS loop once, writing result to d_C_out (for ref or timed). */
static void run_loop_to(GroupedGemmParams *p, float *d_C_out) {
  runCublasTF32_with_TC(p->handle, p->d_A, p->d_B, d_C_out, p->m_list, p->n_list, p->k_list,
                        static_cast<int>(p->batch_size), p->A_offset, p->B_offset, p->C_offset,
                        p->alpha, p->beta);
}

/** Run loop once into d_C_ref (for validation reference). */
static void run_ref_loop(GroupedGemmParams *p) {
  run_loop_to(p, p->d_C_ref);
}

/** Load config, alloc, copy, convert. Returns params (caller must call teardown). */
static GroupedGemmParams *grouped_gemm_setup(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <result_mode> [config_file | batch_size m n k]\n"
              << "  result_mode: p=compact, else=verbose\n"
              << "  config_file: one line per 'M N K' per batch\n";
    return nullptr;
  }
  auto *p = new GroupedGemmParams();
  p->result_mode = argv[1];

  if (argc >= 3) {
    int loaded = load_mnk_from_file(argv[2], &p->m_list, &p->n_list, &p->k_list);
    if (loaded > 0) {
      p->batch_size = loaded;
    } else if (argc >= 6) {
      p->batch_size = std::atol(argv[2]);
      int cfg_m = std::atoi(argv[3]);
      int cfg_n = std::atoi(argv[4]);
      int cfg_k = std::atoi(argv[5]);
      p->m_list = (int *)malloc(sizeof(int) * (p->batch_size + 1));
      p->n_list = (int *)malloc(sizeof(int) * (p->batch_size + 1));
      p->k_list = (int *)malloc(sizeof(int) * (p->batch_size + 1));
      for (long i = 0; i < p->batch_size; i++) {
        p->m_list[i] = cfg_m;
        p->n_list[i] = cfg_n;
        p->k_list[i] = cfg_k;
      }
    } else {
      std::cerr << "Warning: Could not load config, using defaults (1024,4096,14336 x8)\n";
      p->batch_size = 8;
      p->m_list = (int *)malloc(sizeof(int) * 9);
      p->n_list = (int *)malloc(sizeof(int) * 9);
      p->k_list = (int *)malloc(sizeof(int) * 9);
      for (int i = 0; i < 8; i++) {
        p->m_list[i] = 1024;
        p->n_list[i] = 4096;
        p->k_list[i] = 14336;
      }
    }
  } else {
    p->batch_size = 8;
    p->m_list = (int *)malloc(sizeof(int) * 9);
    p->n_list = (int *)malloc(sizeof(int) * 9);
    p->k_list = (int *)malloc(sizeof(int) * 9);
    for (int i = 0; i < 8; i++) {
      p->m_list[i] = 1024;
      p->n_list[i] = 4096;
      p->k_list[i] = 14336;
    }
  }

  p->A_offset = (int *)malloc(sizeof(int) * (p->batch_size + 1));
  p->B_offset = (int *)malloc(sizeof(int) * (p->batch_size + 1));
  p->C_offset = (int *)malloc(sizeof(int) * (p->batch_size + 1));
  int sum_A = 0, sum_B = 0, sum_C = 0;
  for (long b = 0; b <= p->batch_size; b++) {
    p->A_offset[b] = sum_A;
    p->B_offset[b] = sum_B;
    p->C_offset[b] = sum_C;
    sum_A += p->m_list[b] * p->k_list[b];
    sum_B += p->k_list[b] * p->n_list[b];
    sum_C += p->m_list[b] * p->n_list[b];
  }
  p->size_A = sum_A;
  p->size_B = sum_B;
  p->size_C = p->size_C_ref = sum_C;

  printf("      B          M     N     K\n");
  int cp = 0;
  for (long b = 1; b < p->batch_size; b++) {
    if (p->m_list[b] != p->m_list[cp] || p->n_list[b] != p->n_list[cp] ||
        p->k_list[b] != p->k_list[cp]) {
      printf("%6d-%-5ld%6d%6d%6d\n", cp, b - 1, p->m_list[b - 1], p->n_list[b - 1], p->k_list[b - 1]);
      cp = static_cast<int>(b);
    }
  }
  printf("%6d-%-5ld%6d%6d%6d\n", cp, p->batch_size - 1, p->m_list[cp], p->n_list[cp],
         p->k_list[cp]);

  if (cublasCreate(&p->handle) != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "Create cublas handle error.\n";
    delete p;
    return nullptr;
  }

  p->A = (float *)malloc(p->size_A * sizeof(float));
  p->B = (float *)malloc(p->size_B * sizeof(float));
  p->C = (float *)calloc(p->size_C, sizeof(float));
  p->C_ref = (float *)calloc(p->size_C_ref, sizeof(float));
  randomize_matrix_s(static_cast<int>(p->size_A), p->A);
  randomize_matrix_s(static_cast<int>(p->size_B), p->B);

  CHECK_CUDA(cudaMalloc((void **)&p->d_A, p->size_A * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)&p->d_B, p->size_B * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)&p->d_C, p->size_C * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)&p->d_C_ref, p->size_C_ref * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)&p->d_m_list, p->batch_size * sizeof(int)));
  CHECK_CUDA(cudaMalloc((void **)&p->d_n_list, p->batch_size * sizeof(int)));
  CHECK_CUDA(cudaMalloc((void **)&p->d_k_list, p->batch_size * sizeof(int)));
  CHECK_CUDA(cudaMalloc((void **)&p->d_A_offset, (p->batch_size + 1) * sizeof(int)));
  CHECK_CUDA(cudaMalloc((void **)&p->d_B_offset, (p->batch_size + 1) * sizeof(int)));
  CHECK_CUDA(cudaMalloc((void **)&p->d_C_offset, (p->batch_size + 1) * sizeof(int)));

  CHECK_CUDA(cudaMemcpy(p->d_A, p->A, p->size_A * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(p->d_B, p->B, p->size_B * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(p->d_C, 0, p->size_C * sizeof(float)));
  CHECK_CUDA(cudaMemset(p->d_C_ref, 0, p->size_C_ref * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(p->d_m_list, p->m_list, p->batch_size * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(p->d_n_list, p->n_list, p->batch_size * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(p->d_k_list, p->k_list, p->batch_size * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(p->d_A_offset, p->A_offset, (p->batch_size + 1) * sizeof(int),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(p->d_B_offset, p->B_offset, (p->batch_size + 1) * sizeof(int),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(p->d_C_offset, p->C_offset, (p->batch_size + 1) * sizeof(int),
                        cudaMemcpyHostToDevice));

  convert_to_tf32(p->d_A, p->d_B, static_cast<int>(p->size_A), static_cast<int>(p->size_B));
  CHECK_CUDA(cudaEventCreate(&p->start));
  CHECK_CUDA(cudaEventCreate(&p->stop));
  return p;
}

static void grouped_gemm_teardown(GroupedGemmParams *p) {
  if (!p) return;
  free(p->A);
  free(p->B);
  free(p->C);
  free(p->C_ref);
  free(p->m_list);
  free(p->n_list);
  free(p->k_list);
  free(p->A_offset);
  free(p->B_offset);
  free(p->C_offset);
  cudaFree(p->d_A);
  cudaFree(p->d_B);
  cudaFree(p->d_C);
  cudaFree(p->d_C_ref);
  cudaFree(p->d_m_list);
  cudaFree(p->d_n_list);
  cudaFree(p->d_k_list);
  cudaFree(p->d_A_offset);
  cudaFree(p->d_B_offset);
  cudaFree(p->d_C_offset);
  cublasDestroy(p->handle);
  cudaEventDestroy(p->start);
  cudaEventDestroy(p->stop);
  delete p;
}

static void grouped_gemm_verify_and_print(GroupedGemmParams *p, float elapsed_time) {
  long flops = 0;
  for (long b = 0; b < p->batch_size; b++)
    flops += 2L * p->m_list[b] * p->n_list[b] * p->k_list[b];
  float t_kernel = elapsed_time / 1000.0f / static_cast<float>(p->repeat_time);
  float gflops = (p->repeat_time * flops * 1e-9f) / (elapsed_time / 1000.0f);

  bool ok = verify_matrix(p->C_ref, p->C, p->m_list, p->n_list, static_cast<int>(p->batch_size));
  std::cout << (ok ? "Result is correct\n" : "Result is different\n");

  if (strcmp(p->result_mode, "p") == 0) {
    std::cout << "\n--- Performance ---\n";
    std::cout << "  Kernel: " << (t_kernel * 1000.0f) << " ms/iter,  " << gflops << " GFLOPS\n";
    std::cout << "  [raw] " << t_kernel << "," << gflops << "\n";
  } else {
    std::cout << "Kernel time (s): " << t_kernel << "\nKernel GFLOPS: " << gflops << "\n";
  }
}
