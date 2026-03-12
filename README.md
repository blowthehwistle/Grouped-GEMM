# Grouped GEMM Benchmark

Grouped GEMM(배치별 상이한 M,N,K)에 대해 CUDA, Cutlass, PyTorch, Triton 백엔드를 동일 config로 벤치마크하고 Nsight Compute로 프로파일링합니다.

## Quick Start

```bash
# 1. 가상환경 생성 및 활성화
python3 -m venv .venv
source .venv/bin/activate   # Linux/macOS
# Windows: .venv\Scripts\activate

# 2. 의존성 설치
pip install -r requirements.txt
pip install torch   # CUDA 지원: pip install torch (자동으로 Triton 포함)

# 3. 빌드 및 실행
make grouped
python experiments/compare_grouped/run_all.py
```

## 빌드

| Target | 빌드 산출물 |
|--------|-------------|
| `make grouped` | tmain_grouped (TF32), tmain_grouped_half (FP16), tmain_cutlass_grouped |
| `make fp32` | FP32 단일 GEMM |
| `make clean` | 빌드 산출물 삭제 |

**CUTLASS**: `Makefile`의 `CUTLASS_ROOT` (기본: `./cutlass`). 재정의: `CUTLASS_ROOT=/path/to/cutlass make grouped` (CUTLASS 2.x, sm80+ 필요)

## 벤치마크 실행

### 기본

```bash
python experiments/compare_grouped/run_all.py [--config config.yaml] [--backend cuda|triton|torch|cutlass|all]
```

- `config.yaml`의 `grouped` 섹션에서 M,N,K 로드 (PyYAML 필요)
- `--backend all`(기본): CUDA → Cutlass → Torch → Triton 순 실행
- `--results-subdir <name>`: 결과를 `results/benchmark/<name>/`에 저장 (config별 분리 시)
- `--no-compare`: 통합 리포트 생략
- 백엔드당 timeout: 300초 (큰 N,K config 대비)

### Nsight Compute 프로파일링

```bash
python experiments/compare_grouped/run_all.py --nsight [--nsight-out <dir>] [--ncu-extra "-c 100"]
```

모든 백엔드를 Nsight Compute로 프로파일, `.ncu-rep` 저장:

| 백엔드 | 리포트 파일 | 저장 경로 |
|--------|-------------|----------|
| CUDA TF32 | grouped_gemm_tf32_all | kernel 0,1,2 한 리포트에 |
| CUDA FP16 | grouped_gemm_fp16_all | Loop, Grouped 한 리포트에 |
| Cutlass | cutlass_grouped | results/benchmark/cutlass/nsight/ |
| Triton | triton_grouped | results/benchmark/triton/nsight/ |
| Torch | torch_grouped | results/benchmark/torch/nsight/ |

- `--ncu-extra`: 추가 ncu 옵션 (예: `-c 100`, `--set full`)

### 단일-커널 비교 (공정한 벤치마크)

각 백엔드가 **1개의 커널**로 전체 grouped GEMM을 처리하는 경우만 비교:

```bash
python experiments/compare_grouped/run_single_kernel_compare.py [--config config.yaml]
```

- 결과: `results/benchmark/single_kernel/`
- CUDA kernel 1 (cuBLAS Grouped), Cutlass, Triton, Torch grouped_mm
- 커널 런치 오버헤드 제거 → 순수 계산 성능 비교 가능

### 여러 config 순차 실행

`experiments/compare_grouped/configs/` 내 YAML로 config별 벤치마크 실행, 결과는 config명 서브디렉터리에 저장:

```bash
python experiments/compare_grouped/run_multi_config.py
python experiments/compare_grouped/run_multi_config.py --config-dir configs
python experiments/compare_grouped/run_multi_config.py --configs configs/default.yaml configs/expert_skewed.yaml
python experiments/compare_grouped/run_multi_config.py --backend cuda --summary
python experiments/compare_grouped/run_multi_config.py --summary-csv results/summary.csv
```

| 옵션 | 설명 |
|------|------|
| `--config-dir` | config 디렉터리 (기본: configs) |
| `--configs` | 실행할 config 파일 목록 (명시적 지정) |
| `--backend` | cuda \| triton \| torch \| cutlass \| all |
| `--no-compare` | config별 통합 리포트 생략 |
| `--summary` | 마지막에 전체 요약 출력 |
| `--summary-csv` | config×백엔드 결과 CSV 저장 |

### 결과 재확인

```bash
python experiments/compare_grouped/compare.py [--benchmark-dir results/benchmark/<subdir>]
```

## config.yaml

`experiments/compare_grouped/config.yaml`:

```yaml
grouped:
  m: [8192, 4096, 2048, 2048]
  n: [8192, 4096, 2048, 2048]
  k: [2048, 2048, 2048, 2048]
```

- **리스트**: 배치별 상이한 M,N,K
- **스칼라 + batch_size**: 모든 배치 동일 크기

### 사전 정의 config (`configs/`)

| Config | 설명 |
|--------|------|
| `default` | 기본 (M,N,K 최대 8192) |
| `expert_skewed` | Expert 0,1에 부하 집중 (M 8192→8 감소) |
| `expert_even` | 전문가별 토큰 고르게 분배, M 1024→64 |
| `fragmented_16` | 16그룹 파편화 워크로드 |
| `extreme_thin` | M≤32 (추론 초기/소규모 배치) |

- N=14336: LLaMA-7B, Mixtral-8x7B hidden (SwiGLU)
- N=11008: LLaMA intermediate size
- N=4096: 테스트/리소스 절약용 (일부 config)

## 백엔드 요약

| 백엔드 | Precision | 커널 |
|--------|-----------|------|
| CUDA TF32 | TF32 | cuBLAS Loop, cuBLAS Grouped API, Custom Double Buffering |
| CUDA FP16 | FP16 | cuBLAS Loop (K uniform), cuBLAS Grouped |
| Cutlass | FP16 | CUTLASS Grouped |
| Torch | BF16 | grouped_mm / matmul loop |
| Triton | FP16 | Triton 커널 |

- 정량 비교 시 precision 차이 주의 (TF32 vs BF16 vs FP16)
- **Triton**: expert_skewed, extreme_thin 등 small M 지원 (마스크 기반 경계 처리, tl.dot ≥16)

### 백엔드별 특징 및 차이점

| 구분 | CUDA | Cutlass | Torch | Triton |
|------|------|---------|-------|--------|
| **Precision** | TF32, FP16 | FP16 | BF16 | FP16 |
| **실행 형태** | C++ 바이너리 | C++ 바이너리 | Python | Python |
| **라이브러리** | cuBLAS, Custom | CUTLASS 2.x | PyTorch | Triton |
| **단일 커널** | cuBLAS Grouped, Custom | ✓ | grouped_mm (조건부) | ✓ |
| **다중 커널** | cuBLAS Loop | ✗ | matmul loop (fallback) | ✗ |
| **GPU 최소** | sm_86 | sm_80 (Ampere) | sm_80 (grouped_mm) | sm_70+ |
| **추가 의존성** | CUDA | CUTLASS | PyTorch | PyTorch+Triton |

#### CUDA

- **3가지 TF32 커널**: Loop(개별 cublasGemmEx), Grouped(cublasGemmGroupedBatchedEx), Custom(자체 Double Buffering)
- **2가지 FP16 커널**: Loop(K uniform 시), Grouped(항상)
- **특징**: NVIDIA 공식 cuBLAS 기반, 안정적. Custom은 연구용 최적화.
- **제약**: Loop는 커널 수 = batch_size, Grouped/Custom은 1 커널.
- **Validation FAIL (expert_skewed 등)**: cuBLAS Grouped API, Custom Double Buffering은 **M/N < tile size(128)** 인 config에서 검증 실패 가능. Custom 커널은 partial tile 경계 처리 미구현. Grouped API는 cuBLAS 내부 동작 차이로 Loop와 미세 불일치 가능.

#### Cutlass

- **1 커널**: CUTLASS `DefaultGemmGrouped` (128×128×32 threadblock, Tensor Core)
- **특징**: NVIDIA CUTLASS 라이브러리, FP16 전용, 고정 tile.
- **제약**: sm_80 이상, M/N/K 제약 있음 (대형 tile).

#### Torch

- **grouped_mm**: 동일 K·N, BF16, sm80+ 시 `torch.nn.functional.grouped_mm` 1회 호출
- **fallback**: K/N 상이 시 `for i: torch.matmul(A[i], B[i])` (batch_size개 커널)
- **특징**: PyTorch 내장, 별도 빌드 없음.
- **제약**: grouped_mm은 **동일 K, 동일 N** 필요. MoE에서 K/N 불일치 시 loop로 떨어짐.

#### Triton

- **1 커널**: `grouped_matmul_kernel` (autotune, SM 수 기반 스케줄링)
- **특징**: tl.dot + 마스크로 partial tile 지원. BLOCK_SIZE 16 이상 (tl.dot 제약).
- **제약**: M/N < 16은 마스크로 처리(비효율). JIT 컴파일로 첫 실행 지연.
- **TMA**: sm90+ (Hopper)에서 별도 TMA 커널 옵션.



## 프로젝트 구조

```
gemm/
├── implementations/           # 백엔드별 구현
│   ├── cuda/                  grouped_gemm.cu (TF32), grouped_gemm_half.cu (FP16)
│   ├── cutlass/               cutlass_grouped_gemm.cu
│   ├── torch/                 torch_grouped_gemm.py
│   └── triton/                triton_grouped_gemm.py
├── experiments/compare_grouped/
│   ├── run_all.py             벤치마크 + Nsight 프로파일링
│   ├── run_multi_config.py    여러 config 순차 실행, config별 결과 저장
│   ├── run_single_kernel_compare.py   단일-커널 백엔드만 비교
│   ├── compare.py             결과 통합 리포트 (--benchmark-dir 지원)
│   ├── configs/               default, expert_skewed, expert_even, fragmented_16, extreme_thin
│   ├── grouped_config.py      M,N,K config 로더 (YAML/CLI)
│   ├── script_utils.py        implementations/에서 config 로드용
│   ├── config.yaml
│   └── requirements.txt       PyYAML
├── include/
│   ├── config_io.h            config 파일 I/O (grouped_gemm 공용)
│   ├── helpers.h
│   ├── runner.cuh
│   └── cuda_kernels/, tensor_kernels/
├── bin/                       빌드 산출물
├── results/benchmark/         벤치마크 결과, Nsight 리포트 (--results-subdir 시 <name>/)
├── requirements.txt           PyYAML (루트)
└── Makefile
```

## 바이너리 직접 실행

```bash
./bin/tmain_grouped <kernel> p [config.txt]
# kernel: 0=cuBLAS Loop, 1=Grouped API, 2=Custom, 3=all(프로파일용)

./bin/tmain_grouped_half <kernel> p [config.txt]
# kernel: 0=Loop, 1=Grouped, 2=all(프로파일용)

./bin/tmain_cutlass_grouped config.txt p
```

## 가상환경 세팅

```bash
# 프로젝트 루트에서
cd /path/to/gemm

# venv 생성
python3 -m venv .venv

# 활성화 (매번 새 터미널마다)
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# 패키지 설치
pip install --upgrade pip
pip install -r requirements.txt
pip install torch   # Triton은 PyTorch와 함께 설치됨
```

- PyTorch CUDA 버전: [pytorch.org](https://pytorch.org)에서 환경에 맞는 설치 명령 확인
- `.venv`는 `.gitignore`에 포함됨

| 항목 | 버전 |
|------|------|
| CUDA | 12.x (Makefile 기본: 12.5) |
| cuBLAS | CUDA 번들 |
| Python | 3.8+ |
| PyYAML | >= 5.1 |
| PyTorch | CUDA 지원 빌드 (grouped_mm: sm80+) |
| Triton | PyTorch 번들 또는 별도 설치 |
| CUTLASS | 2.x (Cutlass 백엔드, sm80+) |

- **GPU**: sm_80 (Ampere) 이상 권장 (Makefile: sm_86)
- CUDA 경로: `Makefile`의 `CUDA_PATH` 수정 또는 `export CUDA_PATH=/path/to/cuda`
