#include <fstream>
#include <sstream>
#include <vector>

#include "helpers.h"
#include "runner.cuh"
#include "nvtx3/nvToolsExt.h"

/**
 * config 파일에서 M,N,K 로드. 한 줄당 "M N K" (배치 1개).
 * 반환: batch_size, 또는 -1 (오류)
 */
int load_mnk_from_file(const char *path, int **m_list, int **n_list, int **k_list) {
  std::ifstream f(path);
  if (!f) return -1;
  std::vector<int> m_vec, n_vec, k_vec;
  std::string line;
  while (std::getline(f, line)) {
    std::istringstream ss(line);
    int a, b, c;
    if (ss >> a >> b >> c) {
      m_vec.push_back(a);
      n_vec.push_back(b);
      k_vec.push_back(c);
    }
  }
  if (m_vec.empty()) return -1;
  int batch_size = static_cast<int>(m_vec.size());
  *m_list = (int *)malloc(sizeof(int) * (batch_size + 1));
  *n_list = (int *)malloc(sizeof(int) * (batch_size + 1));
  *k_list = (int *)malloc(sizeof(int) * (batch_size + 1));
  for (int i = 0; i < batch_size; i++) {
    (*m_list)[i] = m_vec[i];
    (*n_list)[i] = n_vec[i];
    (*k_list)[i] = k_vec[i];
  }
  return batch_size;
}

void randomize_matrix_s(int N, float *M) {
  struct timeval time{};

  gettimeofday(&time, nullptr);
  srand(time.tv_usec);

  for (int i = 0; i < N; i++) {
    float tmp = (float)(rand() % 5) + 0.01 * (rand() % 5);
    tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
    M[i] = tmp;
  }
}

void set_mnk(int *M, int *N, int *K, const int batch_size, const int *m_value,
             const int *m_batch_offset, const int *n_value, const int *n_batch_offset,
             const int *k_value, const int *k_batch_offset) {
  int m_value_idx = 0, n_value_idx = 0, k_value_idx = 0;
  for (int batch = 0; batch < batch_size; batch++) {
    if (batch == m_batch_offset[m_value_idx + 1]) {
      m_value_idx++;
    }
    if (batch == n_batch_offset[n_value_idx + 1]) {
      n_value_idx++;
    }
    if (batch == k_batch_offset[k_value_idx + 1]) {
      k_value_idx++;
    }
    M[batch] = m_value[m_value_idx];
    N[batch] = n_value[n_value_idx];
    K[batch] = k_value[k_value_idx];
  }
}

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <kernel_number> <result_mode> [config_file | batch_size m n k]\n"
              << "  kernel_number: 0=cuBLAS, 1=GroupedTF32, 2=custom\n"
              << "  result_mode: p=compact, else=verbose\n"
              << "  config_file: 한 줄당 'M N K' (배치별 상이한 M,N,K)\n"
              << "  또는 batch_size m n k (uniform, 모든 배치 동일)\n";
    return EXIT_FAILURE;
  }
  int kernel_number = std::atoi(argv[1]);

  // create cuBLAS handle
  cublasHandle_t handle;
  if (cublasCreate(&handle)) {
    std::cerr << "Create cublas handle error." << std::endl;
    return EXIT_FAILURE;
  };

  // set constants for GEMM
  float alpha = 1.0, beta = 0.0;

  long batch_size;
  int *m_list = nullptr;
  int *n_list = nullptr;
  int *k_list = nullptr;

  // Config: config 파일 또는 argv (uniform) 또는 기본값
  if (argc >= 4) {
    int loaded = load_mnk_from_file(argv[3], &m_list, &n_list, &k_list);
    if (loaded > 0) {
      batch_size = loaded;
    } else if (argc >= 7) {
      batch_size = std::atol(argv[3]);
      int cfg_m = std::atoi(argv[4]);
      int cfg_n = std::atoi(argv[5]);
      int cfg_k = std::atoi(argv[6]);
      m_list = (int *)malloc(sizeof(int) * (batch_size + 1));
      n_list = (int *)malloc(sizeof(int) * (batch_size + 1));
      k_list = (int *)malloc(sizeof(int) * (batch_size + 1));
      for (int i = 0; i < batch_size; i++) {
        m_list[i] = cfg_m;
        n_list[i] = cfg_n;
        k_list[i] = cfg_k;
      }
    } else {
      batch_size = 8;
      m_list = (int *)malloc(sizeof(int) * (batch_size + 1));
      n_list = (int *)malloc(sizeof(int) * (batch_size + 1));
      k_list = (int *)malloc(sizeof(int) * (batch_size + 1));
      for (int i = 0; i < batch_size; i++) {
        m_list[i] = 1024;
        n_list[i] = 4096;
        k_list[i] = 14336;
      }
    }
  } else {
    batch_size = 8;
    m_list = (int *)malloc(sizeof(int) * (batch_size + 1));
    n_list = (int *)malloc(sizeof(int) * (batch_size + 1));
    k_list = (int *)malloc(sizeof(int) * (batch_size + 1));
    for (int i = 0; i < batch_size; i++) {
      m_list[i] = 1024;
      n_list[i] = 4096;
      k_list[i] = 14336;
    }
  }

  int *A_offset = (int *)malloc(sizeof(int) * (batch_size + 1));
  int *B_offset = (int *)malloc(sizeof(int) * (batch_size + 1));
  int *C_offset = (int *)malloc(sizeof(int) * (batch_size + 1));

  printf("      B          M     N     K\n");
  int check_point = 0;
  for (int batch = 1; batch < batch_size; batch++) {
    if (m_list[batch] != m_list[check_point] || n_list[batch] != n_list[check_point]
        || k_list[batch] != k_list[check_point]) {
      printf("%6d-%-5d%6d%6d%6d\n", check_point, batch - 1, m_list[batch - 1], n_list[batch - 1],
             k_list[batch]);
      check_point = batch;
    }
  }
  printf("%6d-%-5ld%6d%6d%6d\n", check_point, batch_size - 1, m_list[check_point],
         n_list[check_point], k_list[check_point]);

  // Calculate offsets for each batch
  int sum_mat_A_offset = 0;
  int sum_mat_B_offset = 0;
  int sum_mat_C_offset = 0;
  for (int batch = 0; batch <= batch_size; batch++) {
    A_offset[batch] = sum_mat_A_offset;
    B_offset[batch] = sum_mat_B_offset;
    C_offset[batch] = sum_mat_C_offset;
    sum_mat_A_offset += m_list[batch] * k_list[batch];
    sum_mat_B_offset += k_list[batch] * n_list[batch];
    sum_mat_C_offset += m_list[batch] * n_list[batch];
  }

  // kernel repeat time for averaging the elapsed time
  int repeat_time = 1000;

  // Size of matrices
  int size_A = A_offset[batch_size];
  int size_B = B_offset[batch_size];
  int size_C = C_offset[batch_size];
  int size_C_ref = C_offset[batch_size];

  // allocate the host memory
  float *A = (float *)malloc(size_A * sizeof(float));
  float *B = (float *)malloc(size_B * sizeof(float));
  float *C = (float *)calloc(size_C, sizeof(float));
  float *C_ref = (float *)calloc(size_C_ref, sizeof(float));

  // initialize input matrices
  randomize_matrix_s(size_A, A);
  randomize_matrix_s(size_B, B);

  // allocate the device memory
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
  CHECK_CUDA(cudaMalloc((void **)&d_A, size_A * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)&d_B, size_B * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)&d_C, size_C * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)&d_C_ref, size_C_ref * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)&d_m_list, batch_size * sizeof(int)));
  CHECK_CUDA(cudaMalloc((void **)&d_n_list, batch_size * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_k_list, batch_size * sizeof(int)));
  CHECK_CUDA(cudaMalloc((void **)&d_A_offset, (batch_size + 1) * sizeof(int)));
  CHECK_CUDA(cudaMalloc((void **)&d_B_offset, (batch_size + 1) * sizeof(int)));
  CHECK_CUDA(cudaMalloc((void **)&d_C_offset, (batch_size + 1) * sizeof(int)));

  // copy input matrices from host to device
  CHECK_CUDA(cudaMemcpy(d_A, A, size_A * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_B, B, size_B * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(d_C, 0, size_C * sizeof(float)));
  CHECK_CUDA(cudaMemset(d_C_ref, 0, size_C_ref * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_m_list, m_list, batch_size * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_n_list, n_list, batch_size * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_k_list, k_list, batch_size * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(d_A_offset, A_offset, (batch_size + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(d_B_offset, B_offset, (batch_size + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(d_C_offset, C_offset, (batch_size + 1) * sizeof(int), cudaMemcpyHostToDevice));

  // convert precision from fp32 to tf32
  convert_to_tf32(d_A, d_B, size_A, size_B);  // tensor core 활용

  // create CUDA event for calculating elapsed time
  float elapsed_time1, elapsed_time2;
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  // warming up the device for 50 times
  for (int i = 0; i < 50; i++)
    runCublasTF32_with_TC(handle, d_A, d_B, d_C_ref, m_list, n_list, k_list, batch_size, A_offset,
                          B_offset, C_offset, alpha, beta);
  cudaDeviceSynchronize();

  CHECK_CUDA(cudaMemset(d_C_ref, 0, size_C_ref * sizeof(float)));

  // baseline
  // execute cuBLAS kernel and calculate execution time
  nvtxRangePushA("cuBLAS");
  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < repeat_time; i++)
    runCublasTF32_with_TC(handle, d_A, d_B, d_C_ref, m_list, n_list, k_list, batch_size, A_offset,
                          B_offset, C_offset, alpha, beta);
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(start));
  CHECK_CUDA(cudaEventSynchronize(stop));
  CHECK_CUDA(cudaEventElapsedTime(&elapsed_time1, start, stop));
  nvtxRangePop();

  // copy the result of cuBLAS kernel from device to host for validation
  CHECK_CUDA(cudaMemcpy(C_ref, d_C_ref, size_C_ref * sizeof(float), cudaMemcpyDeviceToHost));

  // kernel 2는 단일 K만 지원. K가 배치별로 다르면 kernel 1로 폴백
  bool k_uniform = true;
  for (int i = 1; i < batch_size && k_uniform; i++) {
    if (k_list[i] != k_list[0]) k_uniform = false;
  }
  int effective_kernel = (kernel_number == 2 && !k_uniform) ? 1 : kernel_number;
  if (kernel_number == 2 && !k_uniform) {
    std::cerr << "Note: kernel 2 requires uniform K; using kernel 1 (cuBLAS Grouped) instead.\n";
  }

  // kernel to be compared
  nvtxRangePushA("Kernel");
  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < repeat_time; i++) {
    switch (effective_kernel) {
      case 0:
        runCublasTF32_with_TC(handle, d_A, d_B, d_C, m_list, n_list, k_list, batch_size, A_offset,
                              B_offset, C_offset, alpha, beta);
        break;
      case 1:
        runCublasGroupedTF32_with_TC(handle, d_A, d_B, d_C, m_list, n_list, k_list, batch_size,
                                     A_offset, B_offset, C_offset, alpha, beta);
        break;
      case 2:
        run_double_buffering_tf_grouped(d_A, d_B, d_C, m_list, n_list, k_list[0], batch_size,
                                        A_offset, B_offset, C_offset, d_m_list, d_n_list,
                                        d_A_offset, d_B_offset, d_C_offset, alpha, beta);
        break;
      default:
        std::cerr << "Invalid kernel number." << std::endl;
        return EXIT_FAILURE;
    }
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(start));
  CHECK_CUDA(cudaEventSynchronize(stop));
  CHECK_CUDA(cudaEventElapsedTime(&elapsed_time2, start, stop));
  nvtxRangePop();

  // copy the result of the kernel from device to host for validation
  CHECK_CUDA(cudaMemcpy(C, d_C, sizeof(float) * size_C, cudaMemcpyDeviceToHost));

  // calculate the kernel's GFLOPS
  elapsed_time1 /= 1000;
  elapsed_time2 /= 1000;
  long flops = 0;
  for (int batch = 0; batch < batch_size; batch++) {
    flops += (long)2 * m_list[batch] * n_list[batch] * k_list[batch];
  }
  float gflops_ref = (repeat_time * flops * 1e-9) / elapsed_time1;
  float gflops_kernel = (repeat_time * flops * 1e-9) / elapsed_time2;

  // validate kernel result
  bool validation;
  if (!verify_matrix(C_ref, C, m_list, n_list, batch_size)) {
    validation = 1;
    std::cout << "Result is different" << std::endl;

  } else {
    validation = 1;
    std::cout << "Result is correct" << std::endl;
  }

  // display the performance of each kernels
  if (validation == 1) {
    if (strcmp(argv[2], "p") == 0) {
      std::cout << elapsed_time1 / repeat_time << "," << elapsed_time2 / repeat_time << ",";
      std::cout << gflops_ref << "," << gflops_kernel << std::endl;
    } else {
      std::cout << elapsed_time1 / repeat_time << std::endl;
      std::cout << gflops_ref << std::endl;
      std::cout << elapsed_time2 / repeat_time << std::endl;
      std::cout << gflops_kernel << std::endl;
    }
  }

  free(A);
  free(B);
  free(C);
  free(C_ref);
  free(m_list);
  free(n_list);
  free(k_list);
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

  // destroy cuBLAS handle
  if (cublasDestroy(handle) != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "CUBLAS destruction failed\n");
    return EXIT_FAILURE;
  }

  // destroy CUDA event
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));

  return 0;
}
