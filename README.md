# compare GEMM repository

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

## Grouped GEMM M, N, K 설정 방법

Grouped GEMM의 행렬 크기(M, N, K)는 **config 파일** 또는 **CLI 인자**로 지정합니다. `run_all.py`로 한 번에 실행할 때나, 여러 백엔드를 같은 설정으로 돌릴 때 유용합니다.

### 1. config.yaml 사용

`experiments/compare_grouped/config.yaml`의 `grouped` 섹션에서 설정합니다.

#### Uniform 모드 (모든 배치 동일한 M, N, K)

```yaml
grouped:
  batch_size: 8
  m: 1024
  n: 4096
  k: 14336
```

#### 리스트 모드 (배치별로 다른 M, N, K)

```yaml
grouped:
  m: [1024, 512, 256, 128]
  n: [1024, 512, 256, 128]
  k: [1024, 512, 256, 128]
```

```bash
python experiments/compare_grouped/run_all.py --config experiments/compare_grouped/config.yaml
```

### 2. CUDA 전용 config 파일 (grouped_config.txt)

`run_all.py`는 `results/benchmark/cuda/grouped_config.txt`에 M, N, K를 저장하고 CUDA 바이너리에 전달합니다. CUDA를 직접 실행할 때도 이 형식의 파일을 사용할 수 있습니다.

**형식**: 한 줄당 `M N K` (공백 구분, 배치 1개)

```
1024 4096 14336
1024 4096 14336
512  2048 8192
256  1024 4096
```

**직접 실행**:

```bash
make grouped
./bin/tmain_grouped 2 p results/benchmark/cuda/grouped_config.txt
```

**인자 형식**:

- `./bin/tmain_grouped <kernel> <mode> <config_file>` — config 파일에서 배치별 M,N,K 로드
- `./bin/tmain_grouped <kernel> <mode> <batch_size> <m> <n> <k>` — uniform (모든 배치 동일)
- `kernel`: 0=cuBLAS 루프, 1=cuBLAS Grouped, 2=custom
- `mode`: `p`=compact 출력

> **참고**: kernel 2 (custom)는 단일 K만 지원합니다. 배치별로 K가 다르면 자동으로 kernel 1로 폴백됩니다.

### 3. CLI 인자로 지정 (Triton, Torch)

`--m`, `--n`, `--k`로 콤마 구분 리스트를 넘깁니다.

```bash
python implementations/torch/torch_grouped_gemm.py --m 1024,512,256 --n 1024,512,256 --k 1024,512,256
python implementations/triton/triton_grouped_gemm.py --benchmark-only --m 1024,512 --n 1024,512 --k 1024,512
```

### 4. 우선순위

CLI 인자 > config 파일 > 기본값 (batch_size=8, M=1024, N=4096, K=14336)

### 5. Python 코드에서 사용

```python
# experiments/compare_grouped 기준으로 실행하거나, sys.path에 experiments 추가 후
from grouped_config import load_grouped_sizes

m_list, n_list, k_list = load_grouped_sizes(
    config_path="config.yaml",
    # CLI 스타일 override
    m="1024,512,256",
    n="1024,512,256",
    k="1024,512,256",
)
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
