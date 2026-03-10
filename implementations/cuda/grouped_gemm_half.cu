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

void randomize_matrix_s(int N, __half *M) {
  struct timeval time{};

  gettimeofday(&time, nullptr);
  srand(time.tv_usec);

  for (int i = 0; i < N; i++) {
    float tmp = (float)(rand() % 5) + 0.01 * (rand() % 5);
    tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
    M[i] = (__half)tmp;
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
  long batch_size = 8;

  int *A_offset = (int *)malloc(sizeof(int) * (batch_size + 1));
  int *B_offset = (int *)malloc(sizeof(int) * (batch_size + 1));
  int *C_offset = (int *)malloc(sizeof(int) * (batch_size + 1));
  int *m_list = (int *)malloc(sizeof(int) * (batch_size + 1));
  int *n_list = (int *)malloc(sizeof(int) * (batch_size + 1));
  int *k_list = (int *)malloc(sizeof(int) * (batch_size + 1));

  int m_value[] = {64, 128, 256, 512, 1024};
  int m_batch_offset[] = {0, 20, 40, 60, 80, 100};
  int n_value[] = {1024};
  int n_batch_offset[] = {0, 100};
  int k_value[] = {1024};
  int k_batch_offset[] = {0, 100};

  set_mnk(m_list, n_list, k_list, batch_size, m_value, m_batch_offset, n_value, n_batch_offset,
          k_value, k_batch_offset);

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
  int repeat_time = 100;

  // Size of matrices
  int size_A = A_offset[batch_size];
  int size_B = B_offset[batch_size];
  int size_C = C_offset[batch_size];
  int size_C_ref = C_offset[batch_size];

  // allocate the host memory
  __half *A = (__half *)malloc(size_A * sizeof(__half));
  __half *B = (__half *)malloc(size_B * sizeof(__half));
  __half *C = (__half *)calloc(size_C, sizeof(__half));
  __half *C_ref = (__half *)calloc(size_C_ref, sizeof(__half));

  // initialize input matrices
  randomize_matrix_s(size_A, A);
  randomize_matrix_s(size_B, B);

  // allocate the device memory
  __half *d_A = nullptr;
  __half *d_B = nullptr;
  __half *d_C = nullptr;
  __half *d_C_ref = nullptr;
  int *d_m_list = nullptr;
  int *d_n_list = nullptr;
  int *d_k_list = nullptr;
  int *d_A_offset = nullptr;
  int *d_B_offset = nullptr;
  int *d_C_offset = nullptr;
  CHECK_CUDA(cudaMalloc((void **)&d_A, size_A * sizeof(__half)));
  CHECK_CUDA(cudaMalloc((void **)&d_B, size_B * sizeof(__half)));
  CHECK_CUDA(cudaMalloc((void **)&d_C, size_C * sizeof(__half)));
  CHECK_CUDA(cudaMalloc((void **)&d_C_ref, size_C_ref * sizeof(__half)));
  CHECK_CUDA(cudaMalloc((void **)&d_m_list, batch_size * sizeof(int)));
  CHECK_CUDA(cudaMalloc((void **)&d_n_list, batch_size * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_k_list, batch_size * sizeof(int)));
  CHECK_CUDA(cudaMalloc((void **)&d_A_offset, (batch_size + 1) * sizeof(int)));
  CHECK_CUDA(cudaMalloc((void **)&d_B_offset, (batch_size + 1) * sizeof(int)));
  CHECK_CUDA(cudaMalloc((void **)&d_C_offset, (batch_size + 1) * sizeof(int)));

  // copy input matrices from host to device
  CHECK_CUDA(cudaMemcpy(d_A, A, size_A * sizeof(__half), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_B, B, size_B * sizeof(__half), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(d_C, 0, size_C * sizeof(__half)));
  CHECK_CUDA(cudaMemset(d_C_ref, 0, size_C_ref * sizeof(__half)));
  CHECK_CUDA(cudaMemcpy(d_m_list, m_list, batch_size * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_n_list, n_list, batch_size * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_k_list, k_list, batch_size * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(d_A_offset, A_offset, (batch_size + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(d_B_offset, B_offset, (batch_size + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(d_C_offset, C_offset, (batch_size + 1) * sizeof(int), cudaMemcpyHostToDevice));

  // create CUDA event for calculating elapsed time
  float elapsed_time1, elapsed_time2;
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  // warming up the device for 50 times
  for (int i = 0; i < 50; i++)
    runCublasTF16_with_TC(handle, d_A, d_B, d_C_ref, m_list, n_list, k_list[0], batch_size,
                          A_offset, B_offset, C_offset, alpha, beta);
  cudaDeviceSynchronize();

  CHECK_CUDA(cudaMemset(d_C_ref, 0, size_C_ref * sizeof(__half)));

  // execute cuBLAS kernel and calculate execution time
  nvtxRangePushA("cuBLAS");
  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < repeat_time; i++)
    runCublasTF16_with_TC(handle, d_A, d_B, d_C_ref, m_list, n_list, k_list[0], batch_size,
                          A_offset, B_offset, C_offset, alpha, beta);
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(start));
  CHECK_CUDA(cudaEventSynchronize(stop));
  CHECK_CUDA(cudaEventElapsedTime(&elapsed_time1, start, stop));
  nvtxRangePop();

  // copy the result of cuBLAS kernel from device to host for validation
  CHECK_CUDA(cudaMemcpy(C_ref, d_C_ref, size_C_ref * sizeof(__half), cudaMemcpyDeviceToHost));

  // execute the kernel and calculate execution time
  nvtxRangePushA("Kernel");
  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < repeat_time; i++) {
    switch (kernel_number) {
      case 0:
        runCublasTF16_with_TC(handle, d_A, d_B, d_C, m_list, n_list, k_list[0], batch_size,
                              A_offset, B_offset, C_offset, alpha, beta);
        break;
      case 1:
        runCublasGroupedTF16_with_TC(handle, d_A, d_B, d_C, m_list, n_list, k_list, batch_size,
                                     A_offset, B_offset, C_offset, alpha, beta);
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
  CHECK_CUDA(cudaMemcpy(C, d_C, sizeof(__half) * size_C, cudaMemcpyDeviceToHost));

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
      std::cout << "cuBLAS Execution Time : " << elapsed_time1 / repeat_time << " seconds"
                << std::endl;
      std::cout << "cuBLAS Performance : " << gflops_ref << " GFLOPS" << std::endl << std::endl;
      std::cout << "Kernel Execution Time : " << elapsed_time2 / repeat_time << " seconds"
                << std::endl;
      std::cout << "Kernel Performance : " << gflops_kernel << " GFLOPS" << std::endl << std::endl;
      std::cout << "Relative GFLOPS Performance : " << (gflops_kernel / gflops_ref) * 100 << "%"
                << std::endl;
    } else {
      std::cout << elapsed_time1 / repeat_time << std::endl;
      std::cout << gflops_ref << std::endl;
      std::cout << elapsed_time2 / repeat_time << std::endl;
      std::cout << gflops_kernel << std::endl;
    }
  }

  // deallocate the memory
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
