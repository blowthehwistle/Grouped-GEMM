#include "config_io.h"
#include "helpers.h"
#include "runner.cuh"
#include "nvtx3/nvToolsExt.h"

void randomize_matrix_half(int N, __half *M) {
  struct timeval time{};
  gettimeofday(&time, nullptr);
  srand(time.tv_usec);
  for (int i = 0; i < N; i++) {
    float tmp = (float)(rand() % 5) + 0.01f * (rand() % 5);
    tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.f);
    M[i] = __float2half(tmp);
  }
}

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <kernel_number> <result_mode> [config_file | batch_size m n k]\n"
              << "  kernel_number: 0=cuBLAS Loop, 1=cuBLAS Grouped, 2=all (for Nsight profiling)\n"
              << "  result_mode: p=compact\n"
              << "  config_file: 한 줄당 'M N K'\n";
    return EXIT_FAILURE;
  }
  int kernel_number = std::atoi(argv[1]);
  if (kernel_number > 2) {
    std::cerr << "Kernel 0, 1, or 2 only." << std::endl;
    return EXIT_FAILURE;
  }

  cublasHandle_t handle;
  if (cublasCreate(&handle)) {
    std::cerr << "Create cublas handle error." << std::endl;
    return EXIT_FAILURE;
  }

  float alpha = 1.0f, beta = 0.0f;
  long batch_size;
  int *m_list = nullptr, *n_list = nullptr, *k_list = nullptr;

  if (argc >= 4) {
    int loaded = load_mnk_from_file(argv[3], &m_list, &n_list, &k_list);
    if (loaded > 0) {
      batch_size = loaded;
    } else if (argc >= 7) {
      batch_size = std::atol(argv[3]);
      int cfg_m = std::atoi(argv[4]), cfg_n = std::atoi(argv[5]), cfg_k = std::atoi(argv[6]);
      m_list = (int *)malloc(sizeof(int) * (batch_size + 1));
      n_list = (int *)malloc(sizeof(int) * (batch_size + 1));
      k_list = (int *)malloc(sizeof(int) * (batch_size + 1));
      for (int i = 0; i < batch_size; i++) {
        m_list[i] = cfg_m;
        n_list[i] = cfg_n;
        k_list[i] = cfg_k;
      }
    } else {
      std::cerr << "Warning: Could not load config from '" << argv[3]
                << "', using defaults (1024,4096,14336 x8)\n";
      batch_size = 8;
      m_list = (int *)malloc(sizeof(int) * 9);
      n_list = (int *)malloc(sizeof(int) * 9);
      k_list = (int *)malloc(sizeof(int) * 9);
      for (int i = 0; i < 8; i++) {
        m_list[i] = 1024;
        n_list[i] = 4096;
        k_list[i] = 14336;
      }
    }
  } else {
    batch_size = 8;
    m_list = (int *)malloc(sizeof(int) * 9);
    n_list = (int *)malloc(sizeof(int) * 9);
    k_list = (int *)malloc(sizeof(int) * 9);
    for (int i = 0; i < 8; i++) {
      m_list[i] = 1024;
      n_list[i] = 4096;
      k_list[i] = 14336;
    }
  }

  int *A_offset = (int *)malloc(sizeof(int) * (batch_size + 1));
  int *B_offset = (int *)malloc(sizeof(int) * (batch_size + 1));
  int *C_offset = (int *)malloc(sizeof(int) * (batch_size + 1));
  for (int b = 0; b <= batch_size; b++) {
    A_offset[b] = (b == 0) ? 0 : A_offset[b - 1] + m_list[b - 1] * k_list[b - 1];
    B_offset[b] = (b == 0) ? 0 : B_offset[b - 1] + k_list[b - 1] * n_list[b - 1];
    C_offset[b] = (b == 0) ? 0 : C_offset[b - 1] + m_list[b - 1] * n_list[b - 1];
  }

  int size_A = A_offset[batch_size];
  int size_B = B_offset[batch_size];
  int size_C = C_offset[batch_size];

  printf("      B          M     N     K\n");
  int cp = 0;
  for (int b = 1; b < batch_size; b++) {
    if (m_list[b] != m_list[cp] || n_list[b] != n_list[cp] || k_list[b] != k_list[cp]) {
      printf("%6d-%-5d%6d%6d%6d\n", cp, b - 1, m_list[b - 1], n_list[b - 1], k_list[b - 1]);
      cp = b;
    }
  }
  printf("%6d-%-5ld%6d%6d%6d\n", cp, batch_size - 1, m_list[cp], n_list[cp], k_list[cp]);

  __half *A = (__half *)malloc(size_A * sizeof(__half));
  __half *B = (__half *)malloc(size_B * sizeof(__half));
  __half *C = (__half *)calloc(size_C, sizeof(__half));
  __half *C_ref = (__half *)calloc(size_C, sizeof(__half));
  randomize_matrix_half(size_A, A);
  randomize_matrix_half(size_B, B);

  __half *d_A = nullptr, *d_B = nullptr, *d_C = nullptr, *d_C_ref = nullptr;
  int *d_m_list = nullptr, *d_n_list = nullptr, *d_k_list = nullptr;
  int *d_A_offset = nullptr, *d_B_offset = nullptr, *d_C_offset = nullptr;

  CHECK_CUDA(cudaMalloc((void **)&d_A, size_A * sizeof(__half)));
  CHECK_CUDA(cudaMalloc((void **)&d_B, size_B * sizeof(__half)));
  CHECK_CUDA(cudaMalloc((void **)&d_C, size_C * sizeof(__half)));
  CHECK_CUDA(cudaMalloc((void **)&d_C_ref, size_C * sizeof(__half)));
  CHECK_CUDA(cudaMalloc((void **)&d_m_list, batch_size * sizeof(int)));
  CHECK_CUDA(cudaMalloc((void **)&d_n_list, batch_size * sizeof(int)));
  CHECK_CUDA(cudaMalloc((void **)&d_k_list, batch_size * sizeof(int)));
  CHECK_CUDA(cudaMalloc((void **)&d_A_offset, (batch_size + 1) * sizeof(int)));
  CHECK_CUDA(cudaMalloc((void **)&d_B_offset, (batch_size + 1) * sizeof(int)));
  CHECK_CUDA(cudaMalloc((void **)&d_C_offset, (batch_size + 1) * sizeof(int)));

  CHECK_CUDA(cudaMemcpy(d_A, A, size_A * sizeof(__half), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_B, B, size_B * sizeof(__half), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(d_C, 0, size_C * sizeof(__half)));
  CHECK_CUDA(cudaMemset(d_C_ref, 0, size_C * sizeof(__half)));
  CHECK_CUDA(cudaMemcpy(d_m_list, m_list, batch_size * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_n_list, n_list, batch_size * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_k_list, k_list, batch_size * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_A_offset, A_offset, (batch_size + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_B_offset, B_offset, (batch_size + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_C_offset, C_offset, (batch_size + 1) * sizeof(int), cudaMemcpyHostToDevice));

  float elapsed_time1, elapsed_time2;
  bool ok = false;
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  const int warmup = (kernel_number == 2) ? 2 : 50;
  for (int i = 0; i < warmup; i++)
    runCublasGroupedTF16_with_TC(handle, d_A, d_B, d_C_ref, m_list, n_list, k_list, batch_size,
                                 A_offset, B_offset, C_offset, alpha, beta);
  cudaDeviceSynchronize();
  CHECK_CUDA(cudaMemset(d_C_ref, 0, size_C * sizeof(__half)));

  int repeat = (kernel_number == 2) ? 5 : 1000;
  bool k_uniform = true;
  for (int i = 1; i < batch_size && k_uniform; i++)
    if (k_list[i] != k_list[0]) k_uniform = false;

  if (kernel_number == 2) {
    for (int k = 0; k <= 1; k++) {
      int eff = (k == 0 && !k_uniform) ? 1 : k;
      nvtxRangePushA((eff == 0) ? "FP16_Loop" : "FP16_Grouped");
      for (int i = 0; i < repeat; i++) {
        if (eff == 0)
          runCublasTF16_with_TC(handle, d_A, d_B, d_C, m_list, n_list, k_list[0], batch_size,
                                A_offset, B_offset, C_offset, alpha, beta);
        else
          runCublasGroupedTF16_with_TC(handle, d_A, d_B, d_C, m_list, n_list, k_list, batch_size,
                                       A_offset, B_offset, C_offset, alpha, beta);
      }
      nvtxRangePop();
      CHECK_CUDA(cudaDeviceSynchronize());
    }
    std::cout << "All FP16 kernels (0,1) run for profiling.\n";
    ok = true;
    goto cleanup;
  }

  nvtxRangePushA("cuBLAS");
  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < repeat; i++)
    runCublasGroupedTF16_with_TC(handle, d_A, d_B, d_C_ref, m_list, n_list, k_list, batch_size,
                                 A_offset, B_offset, C_offset, alpha, beta);
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));
  CHECK_CUDA(cudaEventElapsedTime(&elapsed_time1, start, stop));
  nvtxRangePop();

  CHECK_CUDA(cudaMemcpy(C_ref, d_C_ref, size_C * sizeof(__half), cudaMemcpyDeviceToHost));

  int eff = (kernel_number == 0 && !k_uniform) ? 1 : kernel_number;
  if (kernel_number == 0 && !k_uniform)
    std::cerr << "Note: Kernel 0 requires uniform K; using Kernel 1 (Grouped) instead.\n";

  nvtxRangePushA("Kernel");
  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < repeat; i++) {
    if (eff == 0) {
      runCublasTF16_with_TC(handle, d_A, d_B, d_C, m_list, n_list, k_list[0], batch_size,
                            A_offset, B_offset, C_offset, alpha, beta);
    } else {
      runCublasGroupedTF16_with_TC(handle, d_A, d_B, d_C, m_list, n_list, k_list, batch_size,
                                   A_offset, B_offset, C_offset, alpha, beta);
    }
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));
  CHECK_CUDA(cudaEventElapsedTime(&elapsed_time2, start, stop));
  nvtxRangePop();

  CHECK_CUDA(cudaMemcpy(C, d_C, size_C * sizeof(__half), cudaMemcpyDeviceToHost));

  ok = verify_matrix(C_ref, C, m_list, n_list, batch_size);
  elapsed_time1 /= 1000.f;
  elapsed_time2 /= 1000.f;
  long flops = 0;
  for (int b = 0; b < batch_size; b++)
    flops += 2L * m_list[b] * n_list[b] * k_list[b];
  float gflops_ref = (repeat * flops * 1e-9f) / elapsed_time1;
  float gflops_ker = (repeat * flops * 1e-9f) / elapsed_time2;

  std::cout << (ok ? "Result is correct\n" : "Result is different\n");

  if (strcmp(argv[2], "p") == 0) {
    float t_ref = elapsed_time1 / repeat;
    float t_ker = elapsed_time2 / repeat;
    std::cout << "\n--- Performance ---\n";
    std::cout << "  Baseline (cuBLAS Loop FP16): " << (t_ref * 1000) << " ms/iter, " << gflops_ref << " GFLOPS\n";
    std::cout << "  Kernel (0/1):                " << (t_ker * 1000) << " ms/iter, " << gflops_ker << " GFLOPS\n";
    std::cout << "  [raw] " << t_ref << "," << t_ker << "," << gflops_ref << "," << gflops_ker << "\n";
  }

cleanup:
  free(A);
  free(B);
  free(C);
  free(C_ref);
  free(m_list);
  free(n_list);
  free(k_list);
  free(A_offset);
  free(B_offset);
  free(C_offset);
  CHECK_CUDA(cudaFree(d_A));
  CHECK_CUDA(cudaFree(d_B));
  CHECK_CUDA(cudaFree(d_C));
  CHECK_CUDA(cudaFree(d_C_ref));
  CHECK_CUDA(cudaFree(d_m_list));
  CHECK_CUDA(cudaFree(d_n_list));
  CHECK_CUDA(cudaFree(d_k_list));
  CHECK_CUDA(cudaFree(d_A_offset));
  CHECK_CUDA(cudaFree(d_B_offset));
  CHECK_CUDA(cudaFree(d_C_offset));
  cublasDestroy(handle);
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));

  return ok ? 0 : EXIT_FAILURE;
}
