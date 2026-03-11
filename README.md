# compare GEMM

CUDA 기반 GEMM 커널을 cuBLAS와 비교하고, Grouped GEMM에 대해 CUDA, Cutlass, PyTorch, Triton 백엔드의 성능을 통합 비교하는 실험 코드입니다.

## 구조

```
gemm/
├── bin/                        # 빌드된 실행 파일
├── implementations/
│   ├── cuda/
│   │   ├── single_gemm.cu
│   │   ├── tensor_single_gemm.cu
│   │   ├── batched_gemm.cu
│   │   ├── grouped_gemm.cu     # TF32 grouped GEMM
│   │   └── grouped_gemm_half.cu # FP16 grouped GEMM
│   ├── cutlass/
│   │   ├── cutlass_grouped_gemm.cu   # CUTLASS Grouped GEMM (align 버전)
│   │   └── original_cutlass_grouped_gemm.cu
│   ├── torch/                  # PyTorch grouped GEMM
│   │   └── torch_grouped_gemm.py
│   └── triton/                 # Triton grouped GEMM
│       └── triton_grouped_gemm.py
├── experiments/
│   ├── compare_grouped/        # Grouped GEMM 통합 벤치마크
│   │   ├── run_all.py         # 모든 백엔드 실행
│   │   ├── compare.py         # 결과 통합 리포트
│   │   ├── grouped_config.py  # config 로더
│   │   └── config.yaml
│   └── varying_prec/
├── include/
│   ├── runner.cuh
│   ├── helpers.h
│   ├── cuda_kernels.cuh
│   ├── tensor_kernels.cuh
│   └── cuda_kernels/, tensor_kernels/
├── Makefile
└── README.md
```

## 빌드

```bash
make                    # 전체 빌드
make fp32               # FP32 커널만
make grouped            # Grouped GEMM (CUDA TF32/FP16 + Cutlass)
make clean
```

`make grouped`는 다음을 빌드합니다.

- `bin/tmain_grouped` - CUDA TF32 (kernel 0/1/2)
- `bin/tmain_grouped_half` - CUDA FP16 (kernel 0/1)
- `bin/tmain_cutlass_grouped` - Cutlass (CUTLASS_ROOT 필요)

**CUTLASS**: Makefile의 `CUTLASS_ROOT`를 CUTLASS 2.x 경로로 맞추거나 `CUTLASS_ROOT=/path/to/cutlass make grouped`로 지정합니다.

## Grouped GEMM 벤치마크

### 전체 실행

```bash
make grouped
python experiments/compare_grouped/run_all.py
```

- config.yaml의 `grouped` 설정으로 M,N,K 로드
- CUDA, Cutlass, Torch, Triton 순으로 실행
- 끝에 통합 리포트 출력 (Validation, Time, GFLOPS)

### 백엔드만 선택

```bash
python experiments/compare_grouped/run_all.py --backend cuda
python experiments/compare_grouped/run_all.py --backend triton --config ...
```

### 결과 비교 (재실행 없이)

```bash
python experiments/compare_grouped/compare.py
```

## 백엔드별 요약

| 백엔드 | Precision | 커널 | 참고 |
|--------|-----------|------|------|
| **CUDA** | TF32 | cuBLAS Loop, cuBLAS Grouped API, Custom | Baseline = cuBLAS Loop |
| **CUDA FP16** | FP16 | cuBLAS Loop, cuBLAS Grouped API | K uniform 필요(Loop) |
| **Cutlass** | FP16 | CUTLASS Grouped | sm80+ 필요 |
| **Torch** | BF16 | `grouped_mm` 또는 `matmul` loop | SM≥80 |
| **Triton** | FP16 | Triton 커널 | autotune, TMA 지원 |

**정량 비교 시 주의**

- precision이 다름: CUDA=TF32, Torch=BF16, Triton/Cutlass=FP16
- 같은 precision끼리 비교하는 것이 타당함 (예: CUDA TF32끼리, FP16끼리)

## M,N,K 설정

### config.yaml

`experiments/compare_grouped/config.yaml`의 `grouped` 섹션:

```yaml
# uniform (모든 배치 동일)
grouped:
  batch_size: 8
  m: 1024
  n: 4096
  k: 14336

# 리스트 (배치별 상이)
grouped:
  m: [8192, 4096, 2048, 2048]
  n: [8192, 4096, 2048, 2048]
  k: [2048, 2048, 2048, 2048]
```

### grouped_config.txt (CUDA/Cutlass용)

한 줄당 `M N K`:

```
1024 4096 14336
512  2048 8192
```

### CLI 인자 (Torch/Triton)

```bash
python implementations/torch/torch_grouped_gemm.py --m 1024,512 --n 1024,512 --k 1024,512
python implementations/triton/triton_grouped_gemm.py --benchmark-only --config experiments/compare_grouped/config.yaml
```

## 바이너리 직접 실행

### tmain_grouped (CUDA TF32)

```bash
./bin/tmain_grouped <kernel> <mode> [config_file | batch_size m n k]
# kernel: 0=cuBLAS Loop, 1=cuBLAS Grouped API, 2=Custom
# mode: p=compact
./bin/tmain_grouped 2 p results/benchmark/cuda/grouped_config.txt
```

- Kernel 2(Custom): 단일 K만 지원, K가 다르면 자동으로 kernel 1로 fallback

### tmain_grouped_half (CUDA FP16)

```bash
./bin/tmain_grouped_half <kernel> <mode> [config_file]
# kernel: 0=Loop (uniform K 필요), 1=Grouped
```

### tmain_cutlass_grouped

```bash
./bin/tmain_cutlass_grouped results/benchmark/cuda/grouped_config.txt p
```

## 통합 리포트 형식

`run_all.py` 후 출력 예:

```
--- Validation ---
  cuBLAS Loop             : PASS
  cuBLAS Grouped API      : PASS
  Custom Double Buffering : PASS
  CUDA FP16 Loop          : PASS
  CUDA FP16 Grouped       : PASS
  Torch                   : PASS
  Triton                  : PASS
  Cutlass                 : PASS

Backend                  |  Time (ms) |       GFLOPS | Validation
--------------------------------------------------------------
cuBLAS Loop              |      28.41 |      13305.1 |     PASS
...
```

## 의존성

- CUDA 12.x
- cuBLAS
- PyTorch (Torch, Triton 백엔드)
- Triton (Triton 백엔드)
- CUTLASS 2.x (Cutlass 백엔드, sm80+)
