#include "helpers.h"
#include "runner.cuh"

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
  long m = 4096, n = 4096, k = 4096, batch_size = 16;
  //   long m = 1024, n = 4096, k = 12288, batch_size = 32;
  // long m = 8192, n = 8192, k = 8192;
  // long m = 16384, n = 16384, k = 16384;

  // kernel repeat time for averaging the elapsed time
  int repeat_time = 1000;

  // Size of matrices
  int size_A = m * k * batch_size;
  int size_B = k * n * batch_size;
  int size_C = m * n * batch_size;
  int size_C_ref = m * n * batch_size;

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
  CHECK_CUDA(cudaMalloc((void **)&d_A, size_A * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)&d_B, size_B * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)&d_C, size_C * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)&d_C_ref, size_C_ref * sizeof(float)));

  // copy input matrices from host to device
  CHECK_CUDA(cudaMemcpy(d_A, A, size_A * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_B, B, size_B * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(d_C, 0, size_C * sizeof(float)));
  CHECK_CUDA(cudaMemset(d_C_ref, 0, size_C_ref * sizeof(float)));

  // convert precision from fp32 to tf32
  convert_to_tf32(d_A, d_B, m, n, k, batch_size);

  // create CUDA event for calculating elapsed time
  float elapsed_time1, elapsed_time2;
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  // warming up the device for 50 times
  for (int i = 0; i < 50; i++)
    runCublasTF32_with_TC(handle, d_A, d_B, d_C_ref, m, n, k, batch_size, alpha, beta);
  cudaDeviceSynchronize();

  CHECK_CUDA(cudaMemset(d_C_ref, 0, size_C_ref * sizeof(float)));

  // execute cuBLAS kernel and calculate execution time
  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < repeat_time; i++)
    runCublasTF32_with_TC(handle, d_A, d_B, d_C_ref, m, n, k, batch_size, alpha, beta);
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(start));
  CHECK_CUDA(cudaEventSynchronize(stop));
  CHECK_CUDA(cudaEventElapsedTime(&elapsed_time1, start, stop));

  // copy the result of cuBLAS kernel from device to host for validation
  CHECK_CUDA(cudaMemcpy(C_ref, d_C_ref, size_C_ref * sizeof(float), cudaMemcpyDeviceToHost));

  // execute the kernel and calculate execution time
  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < repeat_time; i++) {
    switch (kernel_number) {
      case 0:
        run_double_buffering_tf_batched(d_A, d_B, d_C, m, n, k, batch_size, alpha, beta);
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

  // copy the result of the kernel from device to host for validation
  CHECK_CUDA(cudaMemcpy(C, d_C, sizeof(float) * m * n, cudaMemcpyDeviceToHost));

  // calculate the kernel's GFLOPS
  elapsed_time1 /= 1000;
  elapsed_time2 /= 1000;
  long flops = 2 * m * n * k * batch_size;
  float gflops_ref = (repeat_time * flops * 1e-9) / elapsed_time1;
  float gflops_kernel = (repeat_time * flops * 1e-9) / elapsed_time2;

  // validate kernel result
  bool validation;
  if (!verify_matrix(C_ref, C, m, n)) {
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
