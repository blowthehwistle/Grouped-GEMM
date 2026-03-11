# Cutlass GEMM Implementation

## cutlass_grouped_gemm.cu

original_cutlass_grouped_gemm.cu(CUTLASS 예제)를 기반으로, 우리 코드베이스와 align한 버전.

- **Config**: grouped_config.txt (한 줄당 "M N K", CUDA/torch/triton과 동일)
- **CLI**: `./bin/tmain_cutlass_grouped <config_file> [p|v]`
- **Output**: CUDA와 동일 형식 (Batch size, Shapes, Performance, [raw])

### 빌드

```bash
export CUTLASS_ROOT=/path/to/cutlass  # CUTLASS 2.x
make cutlass_grouped
```

### 실행

```bash
./bin/tmain_cutlass_grouped results/benchmark/cuda/grouped_config.txt p
```

## 의존성

- NVIDIA CUTLASS 2.x (Ampere sm80+)
