# GEMM CUDA 실험 저장소

이 저장소는 CUDA 기반 GEMM 커널을 단계적으로 최적화하고, `cuBLAS`와 성능 및 정확도를 비교하기 위한 실험 코드 모음입니다.

이번 정리에서는 **삭제 없이** 파일을 역할 기준으로 재배치했습니다.

## 현재 구조

```text
gemm/
├── bin/
├── experiments/
│   ├── triton/
│   └── varying_prec/
├── include/
│   ├── cuda_kernels/
│   ├── tensor_kernels/
│   ├── cuda_kernels.cuh
│   ├── tensor_kernels.cuh
│   ├── runner.cuh
│   └── helpers.h
├── profiles/
│   ├── nsight/
│   ├── report1.nsys-rep
│   └── report2.nsys-rep
├── results/
│   ├── benchmark/
│   └── tuned/
├── scripts/
│   ├── autotuner/
│   └── profile/
├── src/
├── tools/
├── Makefile
└── README.md
```

## 폴더별 역할

### `src/`

실행 진입점 역할의 CUDA 소스가 모여 있습니다.

- `src/cuda_kernel.cu`
  - FP32 GEMM 최적화 커널 실험용 메인
- `src/tensor_kernel_gemm.cu`
  - 단일 GEMM Tensor Core 실험용 메인
- `src/tensor_kernel_gemm_batched.cu`
  - batched GEMM 실험용 메인
- `src/tensor_kernel_gemm_grouped.cu`
  - grouped GEMM 실험용 메인
- `src/tensor_kernel_gemm_grouped_half.cu`
  - half precision grouped GEMM 실험용 메인

### `include/`

핵심 CUDA 헤더가 모여 있습니다.

- `include/cuda_kernels/`
  - FP32 계열 최적화 커널
- `include/tensor_kernels/`
  - Tensor Core, batched, grouped GEMM 커널
- `include/runner.cuh`
  - 커널 launch 래퍼와 `cuBLAS` 비교 실행 함수
- `include/helpers.h`
  - CUDA 체크, 메모리 초기화, precision 변환, 결과 검증 유틸

### `scripts/`

실험 실행용 스크립트입니다.

- `scripts/autotuner/`
  - 커널 파라미터 autotuning 스크립트
- `scripts/profile/`
  - Nsight Systems / Nsight Compute 실행 스크립트

### `tools/`

결과 후처리용 Python 스크립트입니다.

- `tools/cuda_extractor.py`
- `tools/cuda_extractor_10.py`
- `tools/tensor_extractor.py`

이 스크립트들은 `results/benchmark/`의 로그를 읽고 `results/tuned/`에 정리 결과를 저장합니다.

### `profiles/`

프로파일 결과 보관용 폴더입니다.

- `profiles/nsight/`
  - `*.ncu-rep`, `*.nsys-rep`
- `profiles/report1.nsys-rep`
- `profiles/report2.nsys-rep`

### `results/`

실험 산출물 저장 위치입니다.

- `results/benchmark/`
  - 원시 벤치마크 로그
- `results/tuned/`
  - 정리된 결과

### `experiments/`

주 실험 흐름과 분리된 부가 실험입니다.

- `experiments/varying_prec/`
  - 정밀도 변화 실험용 별도 `Makefile`
- `experiments/triton/grouped_gemm.py`
  - Triton 기반 grouped GEMM 실험

### `bin/`

빌드된 실행 파일 위치입니다.

- `bin/cmain_4k`
- `bin/tmain_4k`
- `bin/tmain_batched_4k`
- `bin/tmain_grouped`
- `bin/tmain_grouped_half`

## 빌드

루트에서 기본 빌드:

```bash
make
```

FP32 커널 실험용 실행 파일만 빌드:

```bash
make fp32
```

Grouped GEMM만 빌드:

```bash
make grouped
```

정리:

```bash
make clean
```

`experiments/varying_prec/`는 별도 빌드가 가능합니다.

```bash
cd experiments/varying_prec
make
```

## 추천 읽기 순서

1. `README.md`
2. `Makefile`
3. `include/helpers.h`
4. `include/runner.cuh`
5. `include/cuda_kernels.cuh`
6. `include/tensor_kernels.cuh`
7. `src/tensor_kernel_gemm.cu`
8. `src/tensor_kernel_gemm_grouped.cu`

## 실행 흐름

1. `src/`의 메인 `.cu` 파일이 입력 크기와 반복 횟수를 설정합니다.
2. `include/helpers.h`가 메모리 준비와 검증을 보조합니다.
3. `include/runner.cuh`가 특정 커널 또는 `cuBLAS`를 launch 합니다.
4. 실행 결과를 비교하고 성능을 출력합니다.
5. `scripts/autotuner/`와 `tools/`가 결과를 수집하고 정리합니다.
6. `scripts/profile/`과 `profiles/`로 병목을 분석합니다.

## 메모

- autotuner 스크립트 일부는 여전히 커널 파라미터나 matrix size를 소스 내부 상수에 의존합니다.
- 즉, 디렉터리 구조는 정리됐지만 모든 실험 스크립트가 완전히 현대화된 것은 아닙니다.
- 현재 기준으로는 `src/`, `include/`, `scripts/`, `tools/`, `profiles/`, `results/`, `experiments/`, `bin/` 구조가 기본입니다.
