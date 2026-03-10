#include "runner.cuh"

void randomize_matrix_s(int N, float *M) {
	struct timeval time {};
	
	gettimeofday(&time, nullptr);
	srand(time.tv_usec);

	for (int i = 0; i < N; i++) {
		float tmp = (float)(rand() % 5) + 0.01 * (rand() % 5);
		tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
		M[i] = tmp;
	}
}

bool verify_matrix_s(void *matRef, void *matOut, int N) {
	double diff = 0.0;
	int i;
	for (i = 0; i < N; i++) {
		diff = std::fabs(((float*)matRef)[i] - ((float*)matOut)[i]);
		if (diff > 0.001) {
			printf("Divergence! Should %.6f, Is %.6f (Diff %.6f) at %d\n",
				((float*)matRef)[i], ((float*)matOut)[i], diff, i);
			return false;
		}
	}
	return true;
}

int main(int argc, char **argv) {
    // 실행할 커널을 선택하기 위해 커널 번호를 인자로 받습니다.
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <kernel_number>" << std::endl;
        return EXIT_FAILURE;
    }
    int kernel_number = std::atoi(argv[1]);
    //printf("kernel : %d\n", kernel_number);
    // 디바이스 정보 출력
    //CudaDeviceInfo();

    // CuBLAS 핸들 생성
    cublasHandle_t handle;
    if(cublasCreate(&handle)) {
        std::cerr << "Create cublas handle error." << std::endl;
        return EXIT_FAILURE;
    };

    // 행렬 곱셈 상수 값
    float alpha = 1.0, beta = 0.0;

    // 행렬 크기 설정 (예시: 4096x4096 크기의 행렬)
    long m = 4096, n = 4096, k = 4096;
    //long m = 8192, n = 8192, k = 8192;
    //long m = 16384, n = 16384, k = 16384;
    

    // repeats
    int repeat_time = 100;

    // 호스트 메모리 할당
    float *A = (float*)malloc(sizeof(float) * m * k);
    float *B = (float*)malloc(sizeof(float) * k * n);
    float *C = (float*)calloc(m * n, sizeof(float));
    float *C_ref = (float*)calloc(m * n, sizeof(float));

    randomize_matrix_s(m * k, A);
	randomize_matrix_s(k * n, B);

    // 디바이스 메모리 할당
    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;
    float *d_C_ref = nullptr;

    CHECK_CUDA(cudaMalloc((void**)&d_A, sizeof(float) * m * k));
	CHECK_CUDA(cudaMalloc((void**)&d_B, sizeof(float) * k * n));
	CHECK_CUDA(cudaMalloc((void**)&d_C, sizeof(float) * m * n));
	CHECK_CUDA(cudaMalloc((void**)&d_C_ref, sizeof(float) * m * n));

    CHECK_CUDA(cudaMemcpy(d_A, A, sizeof(float) * m * k, cudaMemcpyHostToDevice));
	CHECK_CUDA(cudaMemcpy(d_B, B, sizeof(float) * k * n, cudaMemcpyHostToDevice));

    // CUDA 이벤트 생성
    float elapsed_time1, elapsed_time2;
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // 커널 실행 및 시간 측정
    CHECK_CUDA(cudaEventRecord(start));
    for(int i = 0; i < repeat_time; i++)
        runCublasFP32(handle, d_A, d_B, d_C_ref, m, n, k, alpha, beta);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(start));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time1, start, stop));
    
    cudaMemcpy(C_ref, d_C_ref, sizeof(float) * m * n, cudaMemcpyDeviceToHost);

    CHECK_CUDA(cudaEventRecord(start));
    for(int i = 0; i < repeat_time; i++) {
        switch (kernel_number) {
            case 1 :
                run_naive_fp(d_A, d_B, d_C, m, n, k, alpha, beta);
                break;
            case 2 :
                run_global_coalesce_fp(d_A, d_B, d_C, m, n, k, alpha, beta);
                break;
            case 3 :
                run_shared_caching_fp(d_A, d_B, d_C, m, n, k, alpha, beta);
                break;
            case 4 :
                run_blocking_1d_fp(d_A, d_B, d_C, m, n, k, alpha, beta);
                break;
            case 5 :
                run_blocking_2d_fp(d_A, d_B, d_C, m, n, k, alpha, beta);
                break;
            case 6 :
                run_vectorized_fp(d_A, d_B, d_C, m, n, k, alpha, beta);
                break;
            case 7:
                run_vectorized_fp_revised(d_A, d_B, d_C, m, n, k, alpha, beta);
                break;
            case 8 :
                run_resolve_bank_conflict(d_A, d_B, d_C, m, n, k, alpha, beta);
                break;
            case 10 :
                run_warptiling(d_A, d_B, d_C, m, n, k, alpha, beta);
                break;
            case 11 :
                run_double_buffering(d_A, d_B, d_C, m, n, k, alpha, beta);
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

    // 커널 실행 완료 후, 결과를 호스트 메모리로 복사
    CHECK_CUDA(cudaMemcpy(C, d_C, sizeof(float) * m * n, cudaMemcpyDeviceToHost));

    // GFLOPS 계산
    long flops = 2 * m * n * k;
    elapsed_time1 /= 1000;
    elapsed_time2 /= 1000;
    float gflops_ref = (repeat_time * flops * 1e-9) / elapsed_time1;
    float gflops_kernel = (repeat_time * flops * 1e-9) / elapsed_time2;

    // 결과 검증 (예시로 FP32 결과를 사용)
    if(!verify_matrix(C_ref, C, m, n))
        std::cout << "Result is different" << std::endl;
    else
        std::cout << "Result is correct" << std::endl;

    // 성능 출력
    std::cout << "cuBLAS Execution Time : " << elapsed_time1 / repeat_time << " seconds" << std::endl;
    std::cout << "cuBLAS Performance : " << gflops_ref << " GFLOPS" << std::endl << std::endl;
    std::cout << "Kernel Execution Time : " << elapsed_time2 / repeat_time << " seconds" << std::endl;
    std::cout << "Kernel Performance : " << gflops_kernel << " GFLOPS" << std::endl;

    // 메모리 해제
    free(A);
    free(B);
    free(C);
    free(C_ref);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFree(d_C_ref));

    // CuBLAS 핸들 해제
    if (cublasDestroy(handle) != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "CUBLAS destruction failed\n");
        return EXIT_FAILURE;
    }

    // CUDA 이벤트 해제
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}
