#include "helpers.h"
#include "runner.cuh"
#include "nvtx3/nvToolsExt.h"

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
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <kernel_number>" << " <<result display option>>"
              << std::endl;
    return EXIT_FAILURE;
  }
  int kernel_number = std::atoi(argv[1]);

  // CudaDeviceInfo();

  // create cuBLAS handle
  cublasHandle_t handle;
  if (cublasCreate(&handle)) {
    std::cerr << "Create cublas handle error." << std::endl;
    return EXIT_FAILURE;
  };

  // set constants for GEMM
  float alpha = 1.0, beta = 0.0;

  // set the size of matrices
  long batch_size = 100;

  int *A_offset = (int *)malloc(sizeof(int) * (batch_size + 1));
  int *B_offset = (int *)malloc(sizeof(int) * (batch_size + 1));
  int *C_offset = (int *)malloc(sizeof(int) * (batch_size + 1));
  int *m_list = (int *)malloc(sizeof(int) * (batch_size + 1));
  int *n_list = (int *)malloc(sizeof(int) * (batch_size + 1));
  int *k_list = (int *)malloc(sizeof(int) * (batch_size + 1));

  for (int x = 0; x < 8 * 10; x++) batch_size = 8;
  int base_seq_len = 1024;

  // batch별 M 값 다르게 할 때 사용하는 코드
  int x0 = 32 * 0;
  int x1 = 32 * 0;
  int x2 = 32 * 0;
  int x3 = 32 * 0;
  int x4 = 32 * 0;
  int x5 = 32 * 0;
  int x6 = 32 * 0;
  int x7 = 32 * 0;
  int m_value[] = {
      base_seq_len + x0, base_seq_len + x1, base_seq_len + x2, base_seq_len + x3,
      base_seq_len + x4, base_seq_len + x5, base_seq_len + x6, base_seq_len + x7,
  };
  int m_batch_offset[] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  int n_value[] = {4096};
  int n_batch_offset[] = {0, 8};
  int k_value[] = {14336};
  int k_batch_offset[] = {0, 8};

  set_mnk(m_list, n_list, k_list, batch_size, m_value, m_batch_offset, n_value, n_batch_offset,
          k_value, k_batch_offset);

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

  // kernel to be compared
  // execute the kernel and calculate execution time
  nvtxRangePushA("Kernel");
  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < repeat_time; i++) {
    switch (kernel_number) {
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
