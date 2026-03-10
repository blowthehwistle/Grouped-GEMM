# GEMM CUDA 실험 저장소

이 저장소는 CUDA 기반 GEMM 커널을 단계적으로 최적화하고, cuBLAS와 성능 및 정확도를 비교하기 위한 실험 코드 모음입니다.

**추가 예정**: Cutlass, PyTorch, Triton 버전 Grouped GEMM을 구현하고 각각의 퍼포먼스를 비교합니다.

## 현재 구조

```text
gemm/
├── bin/
├── implementations/           # 백엔드별 GEMM 구현체
│   ├── cuda/                  # 커스텀 CUDA
│   │   ├── single_gemm.cu     # FP32 single GEMM
│   │   ├── tensor_single_gemm.cu
│   │   ├── batched_gemm.cu
│   │   ├── grouped_gemm.cu    # FP32 grouped GEMM
│   │   └── grouped_gemm_half.cu
│   ├── cutlass/               # Cutlass (예정)
│   ├── torch/                 # PyTorch (예정)
│   └── triton/                # Triton
│       └── grouped_gemm.py
├── experiments/
│   ├── compare_grouped/       # Grouped GEMM 백엔드 비교
│   │   ├── run_all.py
│   │   ├── compare.py
│   │   └── config.yaml
│   └── varying_prec/
├── include/
│   ├── cuda_kernels/
│   ├── tensor_kernels/
│   ├── cuda_kernels.cuh
│   ├── tensor_kernels.cuh
│   ├── runner.cuh
│   └── helpers.h
├── profiles/
├── results/
│   ├── benchmark/
│   │   ├── cuda/
│   │   ├── cutlass/
│   │   ├── torch/
│   │   └── triton/
│   └── tuned/
├── scripts/
│   ├── autotuner/
│   └── profile/
├── tools/
├── Makefile
└── README.md
```

## 폴더별 역할

### `implementations/`

백엔드별 GEMM 구현체가 분리되어 있습니다.

- **cuda/** - 커스텀 CUDA (FP32, Tensor Core, batched, grouped)
- **cutlass/** - Cutlass 기반 (예정)
- **torch/** - PyTorch 기반 (예정)
- **triton/** - Triton 기반 grouped GEMM

### `include/`

핵심 CUDA 헤더가 모여 있습니다.

- `include/cuda_kernels/` - FP32 계열 최적화 커널
- `include/tensor_kernels/` - Tensor Core, batched, grouped GEMM 커널
- `include/runner.cuh` - 커널 launch 래퍼와 cuBLAS 비교 실행 함수
- `include/helpers.h` - CUDA 체크, 메모리 초기화, precision 변환, 결과 검증 유틸

### `experiments/compare_grouped/`

CUDA, Cutlass, PyTorch, Triton Grouped GEMM 성능 비교용 스크립트입니다.

- `run_all.py` - 모든 백엔드 벤치마크 실행
- `compare.py` - 결과 비교 및 시각화
- `config.yaml` - 공통 설정

### `scripts/`, `tools/`, `profiles/`, `results/`

기존과 동일합니다.

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

## Grouped GEMM 비교 실행

```bash
# 1. CUDA 빌드
make grouped

# 2. 모든 백엔드 벤치마크 실행
python experiments/compare_grouped/run_all.py

# 3. 결과 비교
python experiments/compare_grouped/compare.py
```

## 추천 읽기 순서

1. `README.md`
2. `Makefile`
3. `include/helpers.h`
4. `include/runner.cuh`
5. `include/cuda_kernels.cuh`
6. `include/tensor_kernels.cuh`
7. `implementations/cuda/tensor_single_gemm.cu`
8. `implementations/cuda/grouped_gemm.cu`

## 메모

- autotuner 스크립트 일부는 커널 파라미터나 matrix size를 소스 내부 상수에 의존합니다.
- Cutlass, PyTorch grouped GEMM 구현은 추후 추가 예정입니다.
