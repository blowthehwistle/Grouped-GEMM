#pragma once

#include <cooperative_groups.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <sys/time.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda/barrier>
#include <iostream>
#include <vector>

#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

#define ceil_div(a, b) (((a) + (b) - 1) / (b))

typedef enum _precision_type { FP32 = 0, TF32 = 1, BF16 = 2, FP16 = 3 } precision_type;

typedef struct _matrix {
  float *fp = nullptr;
  float *tf = nullptr;
  __nv_bfloat16 *bf = nullptr;
  __half *h = nullptr;
} Matrix;

typedef struct _matrix_c {
  float *fp = nullptr;
  float *tf = nullptr;
  float *bf = nullptr;
  float *h = nullptr;
} Matrix_C;

void CudaDeviceInfo() {
  int deviceId;

  cudaGetDevice(&deviceId);

  cudaDeviceProp props{};
  cudaGetDeviceProperties(&props, deviceId);

  printf(
      "Device ID: %d\n\
    Name: %s\n\
    Compute Capability: %d.%d\n\
    memory Bus Width: %d bits\n\
    max Threads Per Block: %d\n\
    max Threads Per Streaming MultiProcessor: %d\n\
    max Regs Per Block: %d\n\
    max Regs Per Streaming MultiProcessor: %d\n\
    total Global Mem: %zu MB\n\
    shared Mem Per Block: %zu KB\n\
    shared Mem Per Streaming Multiprocessor: %zu KB\n\
    total Const Mem: %zu KB\n\
    multi Processor Count: %d\n\
    Warp Size: %d\n\
    max warps per multiprocessor: %d\n\
    reg allocation unit size: %d\n\
    reg allocation granularity: %s\n\
    CUDA runtime shared mem overhead per block: %d B\n\
    warp allocation granularity: %d\n\n",
      deviceId, props.name, props.major, props.minor, props.memoryBusWidth,
      props.maxThreadsPerBlock, props.maxThreadsPerMultiProcessor, props.regsPerBlock,
      props.regsPerMultiprocessor, props.totalGlobalMem / 1024 / 1024,
      props.sharedMemPerBlock / 1024, props.sharedMemPerMultiprocessor / 1024,
      props.totalConstMem / 1024, props.multiProcessorCount, props.warpSize,
      props.maxThreadsPerMultiProcessor / props.warpSize,
      256,     // reg allocation unit size
      "warp",  // reg allocation granularity
      1024,    // CUDA runtime shared mem overhead per block
      4        // warp allocation granularity
  );
}

void randomize_matrix(int N, Matrix *M) {
  struct timeval time{};

  gettimeofday(&time, nullptr);
  srand(time.tv_usec);

  for (int i = 0; i < N; i++) {
    float tmp = (float)(rand() % 5) + 0.01 * (rand() % 5);
    tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
    M->fp[i] = tmp;
    M->tf[i] = tmp;
    M->bf[i] = __float2bfloat16(M->fp[i]);
    M->h[i] = __float2half(M->fp[i]);
  }
}

void initialize_host_matrix(long max_size, Matrix *A, Matrix *B, Matrix_C *C, Matrix_C *C_ref) {
  A->fp = (float *)malloc(sizeof(float) * max_size * max_size);
  B->fp = (float *)malloc(sizeof(float) * max_size * max_size);
  C->fp = (float *)malloc(sizeof(float) * max_size * max_size);
  C_ref->fp = (float *)malloc(sizeof(float) * max_size * max_size);

  A->tf = (float *)malloc(sizeof(float) * max_size * max_size);
  B->tf = (float *)malloc(sizeof(float) * max_size * max_size);
  C->tf = (float *)malloc(sizeof(float) * max_size * max_size);
  C_ref->tf = (float *)malloc(sizeof(float) * max_size * max_size);

  A->bf = (__nv_bfloat16 *)malloc(sizeof(__nv_bfloat16) * max_size * max_size);
  B->bf = (__nv_bfloat16 *)malloc(sizeof(__nv_bfloat16) * max_size * max_size);
  C->bf = (float *)malloc(sizeof(float) * max_size * max_size);
  C_ref->bf = (float *)malloc(sizeof(float) * max_size * max_size);

  A->h = (__half *)malloc(sizeof(__half) * max_size * max_size);
  B->h = (__half *)malloc(sizeof(__half) * max_size * max_size);
  C->h = (float *)malloc(sizeof(float) * max_size * max_size);
  C_ref->h = (float *)malloc(sizeof(float) * max_size * max_size);

  randomize_matrix(max_size * max_size, A);
  randomize_matrix(max_size * max_size, B);
}

void allocate_device_matrix(long max_size, Matrix *d_A, Matrix *d_B, Matrix_C *d_C,
                            Matrix_C *d_C_ref) {
  CHECK_CUDA(cudaMalloc((void **)&d_A->fp, sizeof(float) * max_size * max_size));
  CHECK_CUDA(cudaMalloc((void **)&d_B->fp, sizeof(float) * max_size * max_size));
  CHECK_CUDA(cudaMalloc((void **)&d_C->fp, sizeof(float) * max_size * max_size));
  CHECK_CUDA(cudaMalloc((void **)&d_C_ref->fp, sizeof(float) * max_size * max_size));

  CHECK_CUDA(cudaMalloc((void **)&d_A->tf, sizeof(float) * max_size * max_size));
  CHECK_CUDA(cudaMalloc((void **)&d_B->tf, sizeof(float) * max_size * max_size));
  CHECK_CUDA(cudaMalloc((void **)&d_C->tf, sizeof(float) * max_size * max_size));
  CHECK_CUDA(cudaMalloc((void **)&d_C_ref->tf, sizeof(float) * max_size * max_size));

  CHECK_CUDA(cudaMalloc((void **)&d_A->bf, sizeof(__nv_bfloat16) * max_size * max_size));
  CHECK_CUDA(cudaMalloc((void **)&d_B->bf, sizeof(__nv_bfloat16) * max_size * max_size));
  CHECK_CUDA(cudaMalloc((void **)&d_C->bf, sizeof(float) * max_size * max_size));
  CHECK_CUDA(cudaMalloc((void **)&d_C_ref->bf, sizeof(float) * max_size * max_size));

  CHECK_CUDA(cudaMalloc((void **)&d_A->h, sizeof(__half) * max_size * max_size));
  CHECK_CUDA(cudaMalloc((void **)&d_B->h, sizeof(__half) * max_size * max_size));
  CHECK_CUDA(cudaMalloc((void **)&d_C->h, sizeof(float) * max_size * max_size));
  CHECK_CUDA(cudaMalloc((void **)&d_C_ref->h, sizeof(float) * max_size * max_size));
}

void copy_host_to_device(long max_size, Matrix *A, Matrix *B, Matrix *d_A, Matrix *d_B) {
  CHECK_CUDA(
      cudaMemcpy(d_A->fp, A->fp, sizeof(float) * max_size * max_size, cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(d_B->fp, B->fp, sizeof(float) * max_size * max_size, cudaMemcpyHostToDevice));

  CHECK_CUDA(
      cudaMemcpy(d_A->tf, A->tf, sizeof(float) * max_size * max_size, cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(d_B->tf, B->tf, sizeof(float) * max_size * max_size, cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMemcpy(d_A->bf, A->bf, sizeof(__nv_bfloat16) * max_size * max_size,
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_B->bf, B->bf, sizeof(__nv_bfloat16) * max_size * max_size,
                        cudaMemcpyHostToDevice));

  CHECK_CUDA(
      cudaMemcpy(d_A->h, A->h, sizeof(__half) * max_size * max_size, cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(d_B->h, B->h, sizeof(__half) * max_size * max_size, cudaMemcpyHostToDevice));
}

void free_and_reallocate_C(int op, long max_size, int prec, Matrix_C *C, Matrix_C *C_ref,
                           Matrix_C *d_C, Matrix_C *d_C_ref) {
  if (prec == FP32) {
    CHECK_CUDA(cudaFree(d_C->fp));
    CHECK_CUDA(cudaMalloc((void **)&d_C->fp, sizeof(float) * max_size * max_size));
    if (op == 0) {
      free(C->fp);
      free(C_ref->fp);
      C->fp = (float *)malloc(sizeof(float) * max_size * max_size);
      C_ref->fp = (float *)malloc(sizeof(float) * max_size * max_size);

    } else {
      CHECK_CUDA(cudaFree(d_C_ref->fp));
      CHECK_CUDA(cudaMalloc((void **)&d_C_ref->fp, sizeof(float) * max_size * max_size));
    }
  } else if (prec == BF16) {
    CHECK_CUDA(cudaFree(d_C->bf));
    CHECK_CUDA(cudaMalloc((void **)&d_C->bf, sizeof(float) * max_size * max_size));
    if (op == 0) {
      free(C->bf);
      free(C_ref->bf);
      C->bf = (float *)malloc(sizeof(float) * max_size * max_size);
      C_ref->bf = (float *)malloc(sizeof(float) * max_size * max_size);

    } else {
      CHECK_CUDA(cudaFree(d_C_ref->bf));
      CHECK_CUDA(cudaMalloc((void **)&d_C_ref->bf, sizeof(float) * max_size * max_size));
    }
  }
  if (prec == FP16) {
    CHECK_CUDA(cudaFree(d_C->h));
    CHECK_CUDA(cudaMalloc((void **)&d_C->h, sizeof(float) * max_size * max_size));
    if (op == 0) {
      free(C->h);
      free(C_ref->h);
      C->h = (float *)malloc(sizeof(float) * max_size * max_size);
      C_ref->h = (float *)malloc(sizeof(float) * max_size * max_size);

    } else {
      CHECK_CUDA(cudaFree(d_C_ref->h));
      CHECK_CUDA(cudaMalloc((void **)&d_C_ref->h, sizeof(float) * max_size * max_size));
    }
  }
}

bool verify_matrix(void *matRef, void *matOut, int m, int n) {
  double diff = 0.0;
  int i;
  for (i = 0; i < m * n; i++) {
    float ref_val = ((float *)matRef)[i];
    float out_val = ((float *)matOut)[i];
    if (std::isnan(ref_val) || std::isnan(out_val)) {
      printf("NAN Detected\n");
      return false;
    }
    diff = std::fabs(ref_val - out_val) / (std::fabs(ref_val) + 1e-6);  // Avoid division by zero
    if (diff > 0.0001) {
      printf("Divergence! Should %.6f, Is %.6f (Relative Diff %.6f) at %d, (%d, %d)\n", ref_val,
             out_val, diff, i, i / n, i % n);
      return false;
    }
    // else {
    // 	printf("Correct! Should %.6f, Is %.6f (Relative Diff %.6f) at %d, row : %d\n",
    //           ref_val, out_val, diff, i, i / n);
    // }
  }
  return true;
}

bool verify_matrix(void *matRef, void *matOut, int m, int n, int batch_size) {
  double diff = 0.0;
  int i;
  for (int batch_base = 0; batch_base < m * n * batch_size; batch_base += m * n) {
    for (i = 0; i < m * n; i++) {
      float ref_val = ((float *)matRef)[batch_base + i];
      float out_val = ((float *)matOut)[batch_base + i];
      if (std::isnan(ref_val) || std::isnan(out_val)) {
        printf("NAN Detected\n");
        return false;
      }
      diff = std::fabs(ref_val - out_val) / (std::fabs(ref_val) + 1e-6);  // Avoid division by zero
      if (diff > 0.0001) {
        printf("Divergence! Should %.6f, Is %.6f (Relative Diff %.6f) at %d, (%d, %d)\n", ref_val,
               out_val, diff, i, i / n, i % n);
        return false;
      }
      // else {
      // 	printf("Correct! Should %.6f, Is %.6f (Relative Diff %.6f) at %d, row : %d\n",
      //           ref_val, out_val, diff, i, i / n);
      // }
    }
  }
  return true;
}

bool verify_matrix(void *matRef, void *matOut, int *m_list, int *n_list, int batch_size) {
  double diff = 0.0;
  int i, m, n;
  int offset = 0;
  for (int batch = 0; batch < batch_size; batch++) {
    m = m_list[batch];
    n = n_list[batch];
    for (i = 0; i < m * n; i++) {
      float ref_val = ((float *)matRef)[offset + i];
      float out_val = ((float *)matOut)[offset + i];
      if (std::isnan(ref_val) || std::isnan(out_val)) {
        printf("NAN Detected\n");
        return false;
      }
      diff = std::fabs(ref_val - out_val) / (std::fabs(ref_val) + 1e-6);  // Avoid division by zero
      if (diff > 0.0001) {
        printf("Divergence! Should %.6f, Is %.6f (Relative Diff %.6f) at %d, (%d, %d, %d)\n",
               ref_val, out_val, diff, offset + i, batch, i / n, i % n);
        return false;
      }
      // else {
      // 	printf("Correct! Should %.6f, Is %.6f (Relative Diff %.6f) at %d, row : %d\n",
      //           ref_val, out_val, diff, i, i / n);
      // }
    }
    offset += m * n;
  }
  return true;
}

__global__ void convert_fp32_to_tf32(float *input, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float tmp;
    tmp = input[idx];
    input[idx] = nvcuda::wmma::__float_to_tf32(tmp);
  }
}

void convert_to_tf32(float *A, float *B, int m, int n, int k) {
  dim3 blockDim(1024);
  dim3 gridDim(ceil_div(m * k, blockDim.x));
  convert_fp32_to_tf32<<<gridDim, blockDim>>>(A, m * k);
  gridDim = {(ceil_div(k * n, blockDim.x))};
  convert_fp32_to_tf32<<<gridDim, blockDim>>>(B, k * n);
  CHECK_CUDA(cudaDeviceSynchronize());
}

void convert_to_tf32(float *A, float *B, int m, int n, int k, int batch_size) {
  dim3 blockDim(1024);
  dim3 gridDim(ceil_div(m * k * batch_size, blockDim.x));
  convert_fp32_to_tf32<<<gridDim, blockDim>>>(A, m * k * batch_size);
  gridDim = {(ceil_div(k * n * batch_size, blockDim.x))};
  convert_fp32_to_tf32<<<gridDim, blockDim>>>(B, k * n * batch_size);
  CHECK_CUDA(cudaDeviceSynchronize());
}

void convert_to_tf32(float *A, float *B, int size_A, int size_B) {
  dim3 blockDim(1024);
  dim3 gridDim(ceil_div(size_A, blockDim.x));
  convert_fp32_to_tf32<<<gridDim, blockDim>>>(A, size_A);
  gridDim = {(ceil_div(size_B, blockDim.x))};
  convert_fp32_to_tf32<<<gridDim, blockDim>>>(B, size_B);
  CHECK_CUDA(cudaDeviceSynchronize());
}