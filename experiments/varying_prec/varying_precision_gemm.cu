#include "helpers.h"
#include "cuda_kernels.cuh"
#include "tensor_kernels.cuh"

void run_naive_fp(float *A, float *B, float *C, int m, int n, int k, float alpha, float beta) {
    dim3 blockDim(32, 32);
    dim3 gridDim(ceil_div(n, 32), ceil_div(m, 32));
    naive_fp<<<gridDim, blockDim>>>(A, B, C, m, n, k, alpha, beta);
}

void run_naive_bf(__nv_bfloat16 *A, __nv_bfloat16 *B, float *C, int m, int n, int k, float alpha, float beta) {
    dim3 blockDim(32, 32);
    dim3 gridDim(ceil_div(n, 32), ceil_div(m, 32));
    naive_bf<<<gridDim, blockDim>>>(A, B, C, m, n, k, alpha, beta);
}

void run_naive_h(__half *A, __half *B, float *C, int m, int n, int k, float alpha, float beta) {
    dim3 blockDim(32, 32);
    dim3 gridDim(ceil_div(n, 32), ceil_div(m, 32));
    naive_h<<<gridDim, blockDim>>>(A, B, C, m, n, k, alpha, beta);
}

void run_global_coalesce_fp(float *A, float *B, float *C, int m, int n, int k, float alpha, float beta) {
    dim3 blockDim(32 * 32);
    dim3 gridDim(ceil_div(n, 32), ceil_div(m, 32));
    global_coalesce_fp<32> <<<gridDim, blockDim>>>(A, B, C, m, n, k, alpha, beta);
}

void run_global_coalesce_bf(__nv_bfloat16 *A, __nv_bfloat16 *B, float *C, int m, int n, int k, float alpha, float beta) {
    dim3 blockDim(32 * 32);
    dim3 gridDim(ceil_div(n, 32), ceil_div(m, 32));
    global_coalesce_bf<32> <<<gridDim, blockDim>>>(A, B, C, m, n, k, alpha, beta);
}

void run_global_coalesce_h(__half *A, __half *B, float *C, int m, int n, int k, float alpha, float beta) {
    dim3 blockDim(32 * 32);
    dim3 gridDim(ceil_div(n, 32), ceil_div(m, 32));
    global_coalesce_h<32> <<<gridDim, blockDim>>>(A, B, C, m, n, k, alpha, beta);
}

void run_shared_caching_fp(float *A, float *B, float *C, int m, int n, int k, float alpha, float beta) {
    dim3 blockDim(32 * 32);
    dim3 gridDim(ceil_div(n, 32), ceil_div(m, 32));
    shared_caching_fp<32> <<<gridDim, blockDim>>>(A, B, C, m, n, k, alpha, beta);
}

void run_shared_caching_bf(__nv_bfloat16 *A, __nv_bfloat16 *B, float *C, int m, int n, int k, float alpha, float beta) {
    dim3 blockDim(32 * 32);
    dim3 gridDim(ceil_div(n, 32), ceil_div(m, 32));
    shared_caching_bf<32> <<<gridDim, blockDim>>>(A, B, C, m, n, k, alpha, beta);
}

void run_shared_caching_h(__half *A, __half *B, float *C, int m, int n, int k, float alpha, float beta) {
    dim3 blockDim(32 * 32);
    dim3 gridDim(ceil_div(n, 32), ceil_div(m, 32));
    shared_caching_h<32> <<<gridDim, blockDim>>>(A, B, C, m, n, k, alpha, beta);
}

void run_blocking_1d_fp(float *A, float *B, float *C, int m, int n, int k, float alpha, float beta) {
    const uint bm = 64;
    const uint bn = 64;
    const uint bk = 8;
    const uint tw = 8;
    dim3 blockDim((bm / tw) * bn);
    dim3 gridDim(ceil_div(n, bn), ceil_div(m, bm));
    blocking_1d_fp<bm, bn, bk, tw> <<<gridDim, blockDim>>>(A, B, C, m, n, k, alpha, beta);
}

void run_blocking_1d_bf(__nv_bfloat16 *A, __nv_bfloat16 *B, float *C, int m, int n, int k, float alpha, float beta) {
    const uint bm = 64;
    const uint bn = 64;
    const uint bk = 16;
    const uint tw = 8;
    dim3 blockDim((bm / tw) * bn);
    dim3 gridDim(ceil_div(n, bn), ceil_div(m, bm));
    blocking_1d_bf<bm, bn, bk, tw> <<<gridDim, blockDim>>>(A, B, C, m, n, k, alpha, beta);
}

void run_blocking_1d_h(__half *A, __half *B, float *C, int m, int n, int k, float alpha, float beta) {
    const uint bm = 64;
    const uint bn = 64;
    const uint bk = 16;
    const uint tw = 8;
    dim3 blockDim((bm / tw) * bn);
    dim3 gridDim(ceil_div(n, bn), ceil_div(m, bm));
    blocking_1d_h<bm, bn, bk, tw> <<<gridDim, blockDim>>>(A, B, C, m, n, k, alpha, beta);
}

void run_blocking_2d_fp(float *A, float *B, float *C, int m, int n, int k, float alpha, float beta) {
    const uint bm = 128;
    const uint bn = 128;
    const uint bk = 8;
    const uint tw_m = 8;
    const uint tw_n = 8;
    dim3 blockDim((bm / tw_m) * (bn / tw_n));
    dim3 gridDim(ceil_div(n, bn), ceil_div(m, bm));
    blocking_2d_fp<bm, bn, bk, tw_m, tw_n> <<<gridDim, blockDim>>>(A, B, C, m, n, k, alpha, beta);
}

void run_blocking_2d_bf(__nv_bfloat16 *A, __nv_bfloat16 *B, float *C, int m, int n, int k, float alpha, float beta) {
    const uint bm = 64;
    const uint bn = 64;
    const uint bk = 8;
    const uint tw_m = 8;
    const uint tw_n = 8;
    dim3 blockDim((bm / tw_m) * (bn / tw_n));
    dim3 gridDim(ceil_div(n, bn), ceil_div(m, bm));
    blocking_2d_bf<bm, bn, bk, tw_m, tw_n> <<<gridDim, blockDim>>>(A, B, C, m, n, k, alpha, beta);
}

void run_blocking_2d_h(__half *A, __half *B, float *C, int m, int n, int k, float alpha, float beta) {
    const uint bm = 64;
    const uint bn = 64;
    const uint bk = 8;
    const uint tw_m = 8;
    const uint tw_n = 8;
    dim3 blockDim((bm / tw_m) * (bn / tw_n));
    dim3 gridDim(ceil_div(n, bn), ceil_div(m, bm));
    blocking_2d_h<bm, bn, bk, tw_m, tw_n> <<<gridDim, blockDim>>>(A, B, C, m, n, k, alpha, beta);
}

void run_vectorized_fp(float *A, float *B, float *C, int m, int n, int k, float alpha, float beta) {
    const uint bm = 128;
    const uint bn = 128;
    const uint bk = 8;
    const uint tw_m = 8;
    const uint tw_n = 8;
    dim3 blockDim((bm / tw_m) * (bn / tw_n));
    dim3 gridDim(ceil_div(n, bn), ceil_div(m, bm));
    vectorized_fp<bm, bn, bk, tw_m, tw_n> <<<gridDim, blockDim>>>(A, B, C, m, n, k, alpha, beta);
}

void run_vectorized_fp_revised(float *A, float *B, float *C, int m, int n, int k, float alpha, float beta) {
    const uint bm = 128;
    const uint bn = 128;
    const uint bk = 8;
    const uint tw_m = 8;
    const uint tw_n = 8;
    dim3 blockDim((bm / tw_m) * (bn / tw_n));
    dim3 gridDim(ceil_div(n, bn), ceil_div(m, bm));
    vectorized_fp_revised<bm, bn, bk, tw_m, tw_n> <<<gridDim, blockDim>>>(A, B, C, m, n, k, alpha, beta);
}

void run_vectorized_bf(__nv_bfloat16 *A, __nv_bfloat16 *B, float *C, int m, int n, int k, float alpha, float beta) {
    const uint bm = 128;
    const uint bn = 128;
    const uint bk = 8;
    const uint tw_m = 8;
    const uint tw_n = 8;
    dim3 blockDim((bm / tw_m) * (bn / tw_n));
    dim3 gridDim(ceil_div(n, bn), ceil_div(m, bm));
    vectorized_bf<bm, bn, bk, tw_m, tw_n> <<<gridDim, blockDim>>>(A, B, C, m, n, k, alpha, beta);
}

void run_vectorized_h(__half *A, __half *B, float *C, int m, int n, int k, float alpha, float beta) {
    const uint bm = 128;
    const uint bn = 128;
    const uint bk = 8;
    const uint tw_m = 8;
    const uint tw_n = 8;
    dim3 blockDim((bm / tw_m) * (bn / tw_n));
    dim3 gridDim(ceil_div(n, bn), ceil_div(m, bm));
    vectorized_h<bm, bn, bk, tw_m, tw_n> <<<gridDim, blockDim>>>(A, B, C, m, n, k, alpha, beta);
}

void run_resolve_bank_conflict(float *A, float *B, float *C, int m, int n, int k, float alpha, float beta) {
    const uint bm = 128;
    const uint bn = 128;
    const uint bk = 8;
    const uint tw_m = 8;
    const uint tw_n = 8;
    dim3 blockDim((bm / tw_m) * (bn / tw_n));
    dim3 gridDim(ceil_div(n, bn), ceil_div(m, bm));
    resolve_bank_conflict<bm, bn, bk, tw_m, tw_n> <<<gridDim, blockDim>>>(A, B, C, m, n, k, alpha, beta);
}

void run_warptiling(float *A, float *B, float *C, int m, int n, int k, float alpha, float beta) {
    const uint num_threads = 128;
    const uint bm = 256;
    const uint bn = 64;
    const uint bk = 16;
    const uint wm = 128;
    const uint wn = 32;
    const uint wniter = 4;
    const uint tw_m = 8;
    const uint tw_n = 4;
    dim3 blockDim(num_threads);
    dim3 gridDim(ceil_div(n, bn), ceil_div(m, bm));
    warptiling_fp<bm, bn, bk, wm, wn, wniter, tw_m, tw_n, num_threads> <<<gridDim, blockDim>>>(A, B, C, m, n, k, alpha, beta);
}

void run_global_tf(float *A, float *B, float *C, int m, int n, int k, float alpha, float beta) {
    dim3 blockDim(16 * 16);
    dim3 gridDim(ceil_div(n * m, 16 * 16 * 8));
    global_tf <<<gridDim, blockDim>>>(A, B, C, m, n, k, alpha, beta);
}

void run_global_bf(__nv_bfloat16 *A, __nv_bfloat16 *B, float *C, int m, int n, int k, float alpha, float beta) {
    dim3 blockDim(16 * 16);
    dim3 gridDim(ceil_div(n * m, 16 * 16 * 8));
    global_bf <<<gridDim, blockDim>>>(A, B, C, m, n, k, alpha, beta);
}

void run_global_h(__half *A, __half *B, float *C, int m, int n, int k, float alpha, float beta) {
    dim3 blockDim(16 * 16);
    dim3 gridDim(ceil_div(n * m, 16 * 16 * 8));
    global_h <<<gridDim, blockDim>>>(A, B, C, m, n, k, alpha, beta);
}

void run_shared_tf(float *A, float *B, float *C, int m, int n, int k, float alpha, float beta) {
    const uint t_k2_bm = 64;
    const uint t_k2_bn = 256;
    const uint t_k2_bk = 8;
    const uint t_k2_wm = 64;
    const uint t_k2_wn = 64;
    const uint t_k2_num_threads = 128;
    dim3 blockDim(t_k2_num_threads);
    dim3 gridDim(ceil_div(n, t_k2_bn), ceil_div(m, t_k2_bm));
    shm_tf<t_k2_bm, t_k2_bn, t_k2_bk, t_k2_wm, t_k2_wn, t_k2_num_threads> <<<gridDim, blockDim>>>(A, B, C, m, n, k, alpha, beta);
}

void run_shared_revised_tf(float *A, float *B, float *C, int m, int n, int k, float alpha, float beta) {
    const uint t_k3_bm = 64;
    const uint t_k3_bn = 256;
    const uint t_k3_bk = 8;
    const uint t_k3_wm = 64;
    const uint t_k3_wn = 64;
    const uint t_k3_num_threads = 128;
    dim3 blockDim(t_k3_num_threads);
    dim3 gridDim(ceil_div(n, t_k3_bn), ceil_div(m, t_k3_bm));
    shm_revised_tf<t_k3_bm, t_k3_bn, t_k3_bk, t_k3_wm, t_k3_wn, t_k3_num_threads> <<<gridDim, blockDim>>>(A, B, C, m, n, k, alpha, beta);
}

void runCublasTF32_with_TC(cublasHandle_t handle, float *A, float *B, float *C, int m, int n, int k, float alpha, float beta) {
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, B, CUDA_R_32F,
                n, A, CUDA_R_32F, k, &beta, C, CUDA_R_32F, n,
                CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
}

void runCublasBF16_with_TC(cublasHandle_t handle, __nv_bfloat16 *A, __nv_bfloat16 *B, float *C, int m, int n, int k, float alpha, float beta) {
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, B, CUDA_R_16BF,
                n, A, CUDA_R_16BF, k, &beta, C, CUDA_R_32F, n,
                CUBLAS_COMPUTE_32F_FAST_16BF, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
}

void runCublasFP16_with_TC(cublasHandle_t handle, __half *A, __half *B, float *C, int m, int n, int k, float alpha, float beta) {
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, B, CUDA_R_16F,
                n, A, CUDA_R_16F, k, &beta, C, CUDA_R_32F, n,
                CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
}

void runCublasFP32(cublasHandle_t handle, float *A, float *B, float *C, int m, int n, int k, float alpha, float beta) {
    //nvtxRangePushA("outer");
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, B, CUDA_R_32F,
                n, A, CUDA_R_32F, k, &beta, C, CUDA_R_32F, n, 
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    //nvtxRangePop();
}

void runCublasTF32(cublasHandle_t handle, float *A, float *B, float *C, int m, int n, int k, float alpha, float beta) {
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, B, CUDA_R_32F,
                n, A, CUDA_R_32F, k, &beta, C, CUDA_R_32F, n,
                CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT);
}

void runCublasBF16(cublasHandle_t handle, __nv_bfloat16 *A, __nv_bfloat16 *B, float *C, int m, int n, int k, float alpha, float beta) {
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, B, CUDA_R_16BF,
                n, A, CUDA_R_16BF, k, &beta, C, CUDA_R_32F, n,
                CUBLAS_COMPUTE_32F_FAST_16BF, CUBLAS_GEMM_DEFAULT);
}

void runCublasFP16(cublasHandle_t handle, __half *A, __half *B, float *C, int m, int n, int k, float alpha, float beta) {
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, B, CUDA_R_16F,
                n, A, CUDA_R_16F, k, &beta, C, CUDA_R_32F, n,
                CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT);
}

void launch_kernel_with_option_fp(int op, cublasHandle_t handle, float *A, float *B, float *C, int m, int n, int k, float alpha, float beta) {
    switch(op) {
        case 0 :
            runCublasFP32(handle, A, B, C, m, n, k, alpha, beta);
            break;
        case 1 :
            run_naive_fp(A, B, C, m, n, k, alpha, beta);
            break;
        case 2 :
            run_global_coalesce_fp(A, B, C, m, n, k, alpha, beta);
            break;
        case 3 :
            run_shared_caching_fp(A, B, C, m, n, k, alpha, beta);
            break;
        case 4 :
            run_blocking_1d_fp(A, B, C, m, n, k, alpha, beta);
            break;
        case 5 :
            run_blocking_2d_fp(A, B, C, m, n, k, alpha, beta);
            break;
        case 6 :
            run_vectorized_fp(A, B, C, m, n, k, alpha, beta);
            break;
        case 7:
            run_vectorized_fp_revised(A, B, C, m, n, k, alpha, beta);
            break;
        case 8 :
            run_resolve_bank_conflict(A, B, C, m, n, k, alpha, beta);
            break;
        case 9 :
            run_warptiling(A, B, C, m, n, k, alpha, beta);
            break;
    }
}

void launch_kernel_with_option_tf(int op, cublasHandle_t handle, float *A, float *B, float *C, int m, int n, int k, float alpha, float beta) {
    switch(op) {
        case 0 :
            runCublasTF32(handle, A, B, C, m, n, k, alpha, beta);
            break;
        case 1 :
            run_global_tf(A, B, C, m, n, k, alpha, beta);
            break;
        case 2 :
            run_shared_tf(A, B, C, m, n, k, alpha, beta);
            break;
        case 3 :
            run_shared_revised_tf(A, B, C, m, n, k, alpha, beta);
            break;
        default :
            break;
    }
}

void launch_kernel_with_option_bf(int op, cublasHandle_t handle, __nv_bfloat16 *A, __nv_bfloat16 *B, float *C, int m, int n, int k, float alpha, float beta) {
    switch(op) {
        case 0 :
            runCublasBF16(handle, A, B, C, m, n, k, alpha, beta);
            break;
        case 1 :
            run_naive_bf(A, B, C, m, n, k, alpha, beta);
            break;
        case 2 :
            run_global_coalesce_bf(A, B, C, m, n, k, alpha, beta);
            break;
        case 3 :
            run_shared_caching_bf(A, B, C, m, n, k, alpha, beta);
            break;
        case 4 :
            run_blocking_1d_bf(A, B, C, m, n, k, alpha, beta);
            break;
        case 5 :
            run_blocking_2d_bf(A, B, C, m, n, k, alpha, beta);
            break;
        case 6 :
            run_vectorized_bf(A, B, C, m, n, k, alpha, beta);
            break;
        case 7 : 
            run_global_bf(A, B, C, m, n, k, alpha, beta);
            break;
    }
}

void launch_kernel_with_option_h(int op, cublasHandle_t handle, __half *A, __half *B, float *C, int m, int n, int k, float alpha, float beta) {
    switch(op) {
        case 0 :
            runCublasFP16(handle, A, B, C, m, n, k, alpha, beta);
            break;
        case 1 :
            run_naive_h(A, B, C, m, n, k, alpha, beta);
            break;
        case 2 :
            run_global_coalesce_h(A, B, C, m, n, k, alpha, beta);
            break;
        case 3 :
            run_shared_caching_h(A, B, C, m, n, k, alpha, beta);
            break;
        case 4 :
            run_blocking_1d_h(A, B, C, m, n, k, alpha, beta);
            break;
        case 5 :
            run_blocking_2d_h(A, B, C, m, n, k, alpha, beta);
            break;
        case 6 :
            run_vectorized_h(A, B, C, m, n, k, alpha, beta);
            break;
        case 7 : 
            run_global_h(A, B, C, m, n, k, alpha, beta);
            break;
    }
}

typedef struct _execution_result {
    float gflops;
    float time;
} result;

int main(int argc, char **argv) {
    // print the device information
    CudaDeviceInfo();

    // for checking the execution time
    float elapsed_time;
    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);

    // create the cuBLAS handle
    cublasHandle_t handle;
    if(cublasCreate(&handle)) {
        std::cerr << "Create cublas handle error." << std::endl;
        exit(EXIT_FAILURE);
    };

    // constant value for GEMM
    float alpha = 1.0, beta = 0.0;

    // host memory
    Matrix *A = (Matrix*)malloc(sizeof(Matrix));
    Matrix *B = (Matrix*)malloc(sizeof(Matrix));
    Matrix_C *C = (Matrix_C*)malloc(sizeof(Matrix_C));
    Matrix_C *C_ref = (Matrix_C*)malloc(sizeof(Matrix_C));

    // device Memory
    Matrix *d_A = (Matrix*)malloc(sizeof(Matrix));
    Matrix *d_B = (Matrix*)malloc(sizeof(Matrix));
    Matrix_C *d_C = (Matrix_C*)malloc(sizeof(Matrix));
    Matrix_C *d_C_ref = (Matrix_C*)malloc(sizeof(Matrix));

    // define the various matrix size
    std::vector<int> SIZE = {128, 256, 512, 1024, 2048, 4096};
    //std::vector<int> SIZE = {2048, 4096, 8192};
    long max_size;
    max_size = SIZE[SIZE.size() - 1];

    // allocate and initialize host memory
    initialize_host_matrix(max_size, A, B, C, C_ref);

    // allocate device memory
    allocate_device_matrix(max_size, d_A, d_B, d_C, d_C_ref);

    // copy the host memory to device memory
    copy_host_to_device(max_size, A, B, d_A, d_B);

    convert_to_tf32(d_A -> tf, d_B -> tf, max_size, max_size, max_size);

    // number of precisions
    // fp32, tf32, bf16, fp16
    int precision_num = 2;

    // number of kernels
    int op_num = 3;

    // for storing the gflops and elapsed_time
    result ***exe_results = (result***)malloc(sizeof(result**) * precision_num);
    for(int i = 0; i < precision_num; i++) {
        exe_results[i] = (result**)malloc(sizeof(result*) * (op_num + 1));
        for(int j = 0; j <= op_num; j++)
            exe_results[i][j] = (result*)malloc(sizeof(result) * SIZE.size());
    }

    // repeat same kernel as 'repeat'
    int repeat = 100;

    // index for accessing the 'exe_results', tracking the matrix size
    int cnt = 0;
    
    // execute kernels with different precisions
    for(int prec = 0; prec < precision_num; prec++) {
        cnt = 0;
        if(prec == FP32)
            printf("This is FP32\n");
        else if(prec == TF32)
            printf("This is TF32\n");
        else if(prec == BF16)
            printf("This is BF16\n");
        else if(prec == FP16)
            printf("This is FP16\n");
        
        // execute kernels from small size to largest 
        for(int size : SIZE) {
            long m, n, k;
            m = n = k = size;
            std::cout << "size : " << size << std::endl;

            // warm up the device and compare the result
            for(int warm_up = 0; warm_up < 10; warm_up++) {
                if(prec == FP32)
                    launch_kernel_with_option_fp(0, handle, d_A -> fp, d_B -> fp, d_C_ref -> fp, m, n, k, alpha, beta);
                else if(prec == TF32)
                    launch_kernel_with_option_tf(0, handle, d_A -> tf, d_B -> tf, d_C_ref -> tf, m, n, k, alpha, beta);
                else if(prec == BF16)
                    launch_kernel_with_option_bf(0, handle, d_A -> bf, d_B -> bf, d_C_ref -> bf, m, n, k, alpha, beta);
                else if(prec == FP16)
                    launch_kernel_with_option_h(0, handle, d_A -> h, d_B -> h, d_C_ref -> h, m, n, k, alpha, beta);
                CHECK_CUDA(cudaDeviceSynchronize());
            }
            
            for(int i = 0; i <= op_num; i++) {
                std::cout << "This is op " << i << std::endl;
                if(prec == FP32)
                    launch_kernel_with_option_fp(i, handle, d_A -> fp, d_B -> fp, d_C -> fp, m, n, k, alpha, beta);
                else if(prec == TF32)
                    launch_kernel_with_option_tf(i, handle, d_A -> tf, d_B -> tf, d_C -> tf, m, n, k, alpha, beta);
                else if(prec == BF16)
                    launch_kernel_with_option_bf(i, handle, d_A -> bf, d_B -> bf, d_C -> bf, m, n, k, alpha, beta);
                else if(prec == FP16)
                    launch_kernel_with_option_h(i, handle, d_A -> h, d_B -> h, d_C -> h, m, n, k, alpha, beta);
                CHECK_CUDA(cudaDeviceSynchronize());

                // copy kernel results to host
                if(prec == FP32)
                    CHECK_CUDA(cudaMemcpy(C -> fp, d_C -> fp, sizeof(float) * m * n, cudaMemcpyDeviceToHost));
                else if(prec == TF32)
                    CHECK_CUDA(cudaMemcpy(C -> tf, d_C -> tf, sizeof(float) * m * n, cudaMemcpyDeviceToHost));
                else if(prec == BF16)
                    CHECK_CUDA(cudaMemcpy(C -> bf, d_C -> bf, sizeof(float) * m * n, cudaMemcpyDeviceToHost));
                else if(prec == FP16)
                    CHECK_CUDA(cudaMemcpy(C -> h, d_C -> h, sizeof(float) * m * n, cudaMemcpyDeviceToHost));

                if(prec == FP32)
                    CHECK_CUDA(cudaMemcpy(C_ref -> fp, d_C_ref -> fp, sizeof(float) * m * n, cudaMemcpyDeviceToHost));
                if(prec == TF32)
                    CHECK_CUDA(cudaMemcpy(C_ref -> tf, d_C_ref -> tf, sizeof(float) * m * n, cudaMemcpyDeviceToHost));
                if(prec == BF16)
                    CHECK_CUDA(cudaMemcpy(C_ref -> bf, d_C_ref -> bf, sizeof(float) * m * n, cudaMemcpyDeviceToHost));
                if(prec == FP16)
                    CHECK_CUDA(cudaMemcpy(C_ref -> h, d_C_ref -> h, sizeof(float) * m * n, cudaMemcpyDeviceToHost));

                // verify the result between kernels and cuBLAS
                if(prec == FP32) {
                    if(!verify_matrix(C_ref -> fp, C -> fp, n))
                        std::cout << "Result is different" << std::endl << std::endl; 
                    else
                        std::cout << "Result is same" << std::endl << std::endl;
                }
                else if(prec == TF32) {
                    if(!verify_matrix(C_ref -> tf, C -> tf, n))
                        std::cout << "Result is different" << std::endl << std::endl; 
                    else
                        std::cout << "Result is same" << std::endl << std::endl;
                }
                else if(prec == BF16) {
                    if(!verify_matrix(C_ref -> bf, C -> bf, n))
                        std::cout << "Result is different" << std::endl << std::endl; 
                    else
                        std::cout << "Result is same" << std::endl << std::endl;
                }
                else if(prec == FP16) {
                    if(!verify_matrix(C_ref -> h, C -> h, n))
                        std::cout << "Result is different" << std::endl << std::endl; 
                    else
                        std::cout << "Result is same" << std::endl << std::endl;
                }
                free_and_reallocate_C(0, max_size, prec, C, C_ref, d_C, d_C_ref);
            }
            
            // for checking the correct result, deallocate the memory 
            if(prec == FP32) {
                CHECK_CUDA(cudaFree(d_C_ref -> fp));
                CHECK_CUDA(cudaMalloc((void**)&d_C_ref -> fp, sizeof(float) * max_size * max_size));
            }
            else if(prec == TF32) {
                CHECK_CUDA(cudaFree(d_C_ref -> tf));
                CHECK_CUDA(cudaMalloc((void**)&d_C_ref -> tf, sizeof(float) * max_size * max_size));
            }
            else if(prec == BF16) {
                CHECK_CUDA(cudaFree(d_C_ref -> bf));
                CHECK_CUDA(cudaMalloc((void**)&d_C_ref -> bf, sizeof(float) * max_size * max_size));
            }
            else if(prec == FP16) {
                CHECK_CUDA(cudaFree(d_C_ref -> h));
                CHECK_CUDA(cudaMalloc((void**)&d_C_ref -> h, sizeof(float) * max_size * max_size));
            }

            // for comparing the GFLOPS
            for(int i = 0; i <= op_num; i++) {
                CHECK_CUDA(cudaEventRecord(begin));
                for(int j = 0; j < repeat; j++) {
                    if(i == 0) {
                        if(prec == FP32)
                            launch_kernel_with_option_fp(0, handle, d_A -> fp, d_B -> fp, d_C_ref -> fp, m, n, k, alpha, beta);
                        else if(prec == TF32)
                            launch_kernel_with_option_tf(0, handle, d_A -> tf, d_B -> tf, d_C_ref -> tf, m, n, k, alpha, beta);
                        else if(prec == BF16)
                            launch_kernel_with_option_bf(0, handle, d_A -> bf, d_B -> bf, d_C_ref -> bf, m, n, k, alpha, beta);
                        else if(prec == FP16)
                            launch_kernel_with_option_h(0, handle, d_A -> h, d_B -> h, d_C_ref -> h, m, n, k, alpha, beta);
                    }
                    else {
                        if(prec == FP32)
                            launch_kernel_with_option_fp(i, handle, d_A -> fp, d_B -> fp, d_C -> fp, m, n, k, alpha, beta);
                        else if(prec == TF32)
                            launch_kernel_with_option_tf(i, handle, d_A -> tf, d_B -> tf, d_C_ref -> tf, m, n, k, alpha, beta);
                        else if(prec == BF16)
                            launch_kernel_with_option_bf(i, handle, d_A -> bf, d_B -> bf, d_C -> bf, m, n, k, alpha, beta);
                        else if(prec == FP16)
                            launch_kernel_with_option_h(i, handle, d_A -> h, d_B -> h, d_C -> h, m, n, k, alpha, beta);
                    }
                }
                CHECK_CUDA(cudaEventRecord(end));
                CHECK_CUDA(cudaEventSynchronize(begin));
                CHECK_CUDA(cudaEventSynchronize(end));
                CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, begin, end));
                
                // print the results for each size
                elapsed_time /= 1000;
                long flops = 2 * m * n * k;
                printf("op : %d, Average time : (%7.6f)sec, performance : (%7.1f) GFLOPs/s\n", i, elapsed_time / repeat, (repeat * flops * 1e-9) / elapsed_time);
                fflush(stdout);
                
                // store the results
                exe_results[prec][i][cnt].gflops = (repeat * flops * 1e-9) / elapsed_time;
                exe_results[prec][i][cnt].time = elapsed_time;
            }
            printf("\n");
            cnt++;
            
            // for checking the correct result, deallocate the memory 
            free_and_reallocate_C(1, max_size, prec, C, C_ref, d_C, d_C_ref);
        }
    }

    printf("execution finished\n\n");

    // display the result of each kernels
    for(int i = 0; i < precision_num; i++) {
        if(i == 0)
            printf("This is FP32\n");
        else if(i == 1)
            printf("This is TF32\n");
        else if(i == 2)
            printf("This is BF16\n");
        else if(i == 3)
            printf("This is FP16\n");
        for(int j = 0; j <= op_num; j++) {
            printf("op : %d\nperformance : ", j);
            for(int k = 0; k < SIZE.size(); k++)
                printf("%7.1f GFLOPs/s ", exe_results[i][j][k].gflops);
            printf("\n");
        }
        printf("\n");
    }
    printf("\n");

    // summarize the execution results
    for(int i = 0; i < precision_num; i++) {
        printf("precision : %d\n", precision_type(i));
        for(int j = 0; j <= op_num; j++) {
            printf("kernel\tsize\tGFLOPs/s\tSpeed UP(time) relative to cuBLAS\tPerformance(Gflops) relative to cuBLAS\n");
            for(int k = 0; k < SIZE.size(); k++) {
                printf("  %d\t%d\t%7.1f\t\t\t   %f\t\t\t\t   %9.6f%%\n", j, SIZE[k], exe_results[i][j][k].gflops, 
                                                    exe_results[i][0][k].time / exe_results[i][j][k].time, (exe_results[i][j][k].gflops / exe_results[i][0][k].gflops) * 100);
            }
            printf("\n");
        }
    }
    
    FILE *fp;
    if(argc == 2) {
        if(strcmp("-p", argv[1]) == 0) {
            fp = fopen("result.txt", "w+");
            if(fp == NULL)
                printf("failed to open\n");
            for(int i = 0; i < precision_num; i++) {
                fprintf(fp, "precision : %d\n", precision_type(i));
                for(int j = 0; j <= op_num; j++) {
                    fprintf(fp, "kernel\tsize\tGFLOPs/s\tSpeed UP(time) relative to cuBLAS\tPerformance(Gflops) relative to cuBLAS\n");
                    for(int k = 0; k < SIZE.size(); k++) {
                        fprintf(fp, "  %d\t%d\t%7.1f\t\t\t   %f\t\t\t\t   %9.6f%%\n", j, SIZE[k], exe_results[i][j][k].gflops, 
                                    exe_results[i][0][k].time / exe_results[i][j][k].time, (exe_results[i][j][k].gflops / exe_results[i][0][k].gflops) * 100);
                    }
                    fprintf(fp, "\n");
                }
            }
        }
    }

    // deallocate the host memory
    free(A);
	free(B);
	free(C);
	free(C_ref);
	for(int i = 0; i < precision_num; i++) {
		for(int j = 0; j < op_num; j++)
        	free(exe_results[i][j]);
    	free(exe_results[i]);
	}
	free(exe_results);

    // deallocate the device memory
    CHECK_CUDA(cudaFree(d_A -> fp));
    CHECK_CUDA(cudaFree(d_A -> tf));
    CHECK_CUDA(cudaFree(d_A -> bf));
    CHECK_CUDA(cudaFree(d_A -> h));

    CHECK_CUDA(cudaFree(d_B -> fp));
    CHECK_CUDA(cudaFree(d_B -> tf));
    CHECK_CUDA(cudaFree(d_B -> bf));
    CHECK_CUDA(cudaFree(d_B -> h));

    CHECK_CUDA(cudaFree(d_C -> fp));
    CHECK_CUDA(cudaFree(d_C -> tf));
    CHECK_CUDA(cudaFree(d_C -> bf));
    CHECK_CUDA(cudaFree(d_C -> h));

    CHECK_CUDA(cudaFree(d_C_ref -> fp));
    CHECK_CUDA(cudaFree(d_C_ref -> tf));
    CHECK_CUDA(cudaFree(d_C_ref -> bf));
    CHECK_CUDA(cudaFree(d_C_ref -> h));

    // destroy the event variables
    CHECK_CUDA(cudaEventDestroy(begin));
    CHECK_CUDA(cudaEventDestroy(end));
    
    // destroy the cublas handle
    if (cublasDestroy(handle) != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "CUBLAS destruction failed\n");
        return EXIT_FAILURE;
    }

    return 0;
}