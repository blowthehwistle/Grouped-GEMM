# Nsight Compute 프로파일 결과 분석 (benchmark_kernel_only.py --nsight)

`benchmark_kernel_only.py --nsight`로 얻은 .ncu-rep에서 **커널 이름이 5개 모두 같을 때**와 **다를 때**가 있는 이유, 그리고 백엔드별 요약입니다.

---

## 0. NVTX 구간으로 GEMM만 수집 (원리, 모든 백엔드 공통)

ncu는 옵션 없이 돌리면 앱 실행 동안 **모든 커널**을 기록합니다.  
**Triton, Torch, CUDA(cuBLAS/Custom), Cutlass** 모두 **동일한 NVTX 구간 이름 `"grouped_gemm"`** 으로 "GEMM 벤치 구간만" 수집합니다.

- **동작**: 각 구현에서 벤치 루프(또는 타임 측정 구간)를 `nvtxRangePushA("grouped_gemm")` / `nvtxRangePop()` (또는 Triton/Torch의 `range_push("grouped_gemm")` / `range_pop()`) 으로 감싸 두었고, ncu에는 `--nvtx --nvtx-include "grouped_gemm"` (및 `--nvtx-push-pop-scope process`) 를 넘깁니다. ncu는 **"grouped_gemm" 구간이 열려 있는 동안 런치된 커널만** 기록합니다.
- **결과**: 초기화·warmup은 구간 밖이라 제외되고, **실제 GEMM 커널만** 리포트에 나옵니다. 구간 안이면 **커널 이름과 무관하게 전부** 수집되므로 robust합니다.
- **전부 보고 싶을 때**: `--nsight-all-kernels` 를 주면 NVTX 필터를 쓰지 않아, 모든 커널이 기록됩니다.

---

## 1. 왜 어떤 건 5번 다 똑같고 어떤 건 달라?

### 1) `--launch-count 5` 의미

스크립트는 Nsight를 다음처럼 실행합니다:

```text
ncu --export <rep_path> --force-overwrite --launch-count 5 -- <cmd>
```

- **launch-count 5** = 전체 프로세스(벤치마크 한 번 돌리기)를 **5번 반복** 실행하고, 그동안 실행된 **모든 커널 런치**를 한 리포트에 모아 줍니다.
- 따라서 "5개"는 보통 다음 둘 중 하나입니다:
  - 반복 5번 × (런치당 커널 1개), 또는
  - 한 번의 실행 안에서 서로 다른 커널이 여러 번 뜨는 것.

이 구조를 전제로 보면, "이름이 5번 똑같다" vs "5개가 다르다"의 차이가 설명됩니다.

---

### 2) **이름이 5번 모두 같은 경우**

- **같은 커널**이 **5번 런치**된 상황입니다.
- Nsight Summary 테이블의 "Function Name"이 하나로 통일되어 보입니다.

**예: `grouped_gemm_fp16_grouped.ncu-rep`**

- 5개 모두: `Kernel2` (demangled: `cutlass_80_tensorop_f16_s16816gemm_relu_f16_64x64_32x6_nn_align8`)
- 의미: **한 가지 CUTLASS FP16 grouped GEMM 커널**만 사용하고, 벤치마크를 5번 돌리면서 그 커널이 5번 실행됨.
- Duration이 0.37 ms, 1.61 ms, 6.42 ms, … 처럼 다른 이유:
  - **배치 내 문제 크기(M,N,K)가 달라서** 각 런치의 연산량이 다르거나,
  - 5번 실행 간 GPU 상태/스케줄링 차이로 인한 변동.

**예: `torch_grouped_single.ncu-rep`, `triton_grouped_single.ncu-rep`**

- 5개 모두: `distribution_elementwise_grid_stride_kernel` (curand 기반)
- 의미: **GEMM이 아니라 데이터 초기화(랜덤 생성)** 커널이 5번 잡힌 것입니다.
- PyTorch/Triton 스크립트는 `torch.randn` 등으로 입력 텐서를 만들기 때문에, 프로파일 구간 안에서 **같은 초기화 커널이 여러 번** 실행되고, Summary 상위에 그게 5개 보이는 경우입니다.
- 실제 grouped GEMM 커널은 같은 리포트의 **아래쪽**에 있거나, 커널 이름으로 필터해 보면 따로 보입니다.

정리하면:

- **CUDA fp16_grouped**: 동일한 1개 커널이 5번 실행 → 이름 5번 동일.
- **Torch/Triton**: 초기화 커널이 5번 잡힌 경우 → 이름 5번 동일 (GEMM 아님).

---

### 3) **이름이 다르게 보이는 경우 (또는 Demangled만 다른 경우)**

**예: `grouped_gemm_tf32_custom.ncu-rep`**

- Function Name은 모두 `Kernel2`로 같지만,
- **Demangled Name**을 보면:
  - `cutlass_80_tensorop_s1688gemm_256x128_16x3_nn_align4`
  - `cutlass_80_tensorop_s1688gemm_128x128_16x5_nn_align4`
  - `cutlass_80_tensorop_s1688gemm_64x256_16x4_nn_align4`
- 즉 **같은 "Kernel2" 템플릿**이지만 **타일 크기(256×128, 128×128, 64×256)** 가 다른 **서로 다른 인스턴스**가 각각 실행된 것입니다.
- Custom TF32 경로에서는 **문제 shape에 따라 서로 다른 CUTLASS 설정**을 선택하기 때문에, 한 번의 실행 안에서 여러 종류의 `Kernel2<...>`가 나옵니다.
- 추가로 `convert_fp32_to_tf32` 같은 변환 커널도 같이 잡혀서, "이름이 5개가 완전히 같지 않고, 혼합"되어 보입니다.

정리:

- **tf32_custom**: shape별로 다른 CUTLASS 커널(다른 타일) 사용 → Function Name은 같아도 Demangled Name이 다르고, 런치 종류가 여러 개.

---

## 2. 백엔드별 요약

| 리포트 | 커널 이름 패턴 | 설명 |
|--------|----------------|------|
| **fp16_grouped** | NVTX 구간 내 GEMM만 (5개 동일 등) | FP16 벤치 구간이 "grouped_gemm" NVTX로 감싸져 있음. 단일 CUTLASS grouped 커널만 사용 시 5회 반복. |
| **tf32_custom** | NVTX 구간 내 GEMM만 (Kernel2 여러 타일 등) | Custom TF32 벤치 구간이 "grouped_gemm" NVTX로 감싸져 있음. shape별 다른 CUTLASS 타일 + TF32 변환. |
| **tf32_grouped** | NVTX 구간 내 GEMM만 | cuBLAS Grouped 벤치 구간이 "grouped_gemm" NVTX로 감싸져 있음. |
| **torch_grouped_single** | NVTX 구간 내 GEMM만 | NVTX "grouped_gemm" 구간만 수집 → GEMM 커널만 나옴. |
| **triton_grouped_single** | NVTX 구간 내 GEMM만 | 위와 동일. |
| **cutlass_grouped_single** | NVTX 구간 내 GEMM만 | CUTLASS 벤치 루프가 "grouped_gemm" NVTX로 감싸져 있음. |

---

## 3. 성능 관련 인사이트 (스크린샷 기준)

- **fp16_grouped**
  - Theoretical Occupancy 16.7%, 한계 원인: **shared memory**.
  - "2.00 theoretical warps per scheduler vs 12 max" → SM당 워프 수를 늘리려면 shared memory 사용을 줄이거나 블록 구성을 조정해야 함.

- **tf32_custom**
  - 256×128 타일이 12.73 ms로 가장 느리고, 128×128이 3.28 ms로 가장 빠름.
  - 동일 반복에서 shape별로 다른 타일을 쓰기 때문에, **shape별로 어떤 타일이 선택되는지**가 전체 구간 시간에 큰 영향을 줌.

- **Torch/Triton**
  - "Tail Effect": 1 full wave + partial wave 72 blocks → 최대 약 50% 시간이 partial wave에서 소모된다는 의미. 그리드 크기를 SM 수에 맞게 조정하면 개선 여지가 있음 (주로 초기화 커널 쪽).

---

## 4. 실제 계산 커널만 보기 (NVTX 기본, 모든 백엔드)

- **기본 동작**: `benchmark_kernel_only.py --nsight`로 **모든 백엔드**(Triton, Torch, CUDA cuBLAS/Custom, Cutlass)를 돌리면 **NVTX 구간 "grouped_gemm"** 만 프로파일합니다. 각 구현의 벤치 루프가 이 구간에 있으므로 **GEMM 커널만** .ncu-rep에 수집됩니다.
- **전체 커널(초기화 포함) 보고 싶을 때**: `--nsight-all-kernels` 를 주면 NVTX 필터를 쓰지 않아, 모든 커널이 기록됩니다.

```bash
python experiments/compare_grouped/benchmark_kernel_only.py --nsight --backend triton --nsight-all-kernels
```

### 기타 Nsight 옵션

- `--nsight-launch-count 1` (또는 `--ncu-launch-count 1`): 앱 1회만 실행 → 한 번 실행에서 뜨는 커널 개수만 확인.
- `--nsight-launch-skip 20`: 처음 20개 커널 런치 스킵 후 수집.

---

이 문서는 `experiments/compare_grouped/NSIGHT_ANALYSIS.md`에 두었습니다. 벤치마크/프로파일 재실행 시에도 같은 해석이 적용됩니다.
