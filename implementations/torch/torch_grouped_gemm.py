"""
PyTorch Grouped GEMM - Triton 스타일 API

torch._grouped_mm을 사용하여 Triton grouped_gemm과 동일한 인터페이스 제공.
- group_gemm_fn(group_A, group_B): 리스트 형태 입력 → 리스트 형태 출력
- torch._grouped_mm은 BF16, SM>=80, 동일 N/K 가정 (MoE forward 패턴)
"""

from typing import List, Optional

import torch
from torch import Tensor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _supports_grouped_mm() -> bool:
    """torch._grouped_mm 사용 가능 여부 (BF16, SM>=80)"""
    if not torch.cuda.is_available():
        return False
    try:
        cap = torch.cuda.get_device_capability()
        return cap[0] >= 8  # SM >= 80
    except Exception:
        return False


def group_gemm_fn(
    group_A: List[Tensor],
    group_B: List[Tensor],
    use_grouped_mm: bool = True,
) -> List[Tensor]:
    """
    Triton group_gemm_fn과 동일한 API.

    group_A[i] @ group_B[i] for each i.
    - group_A[i]: (M_i, K_i)
    - group_B[i]: (K_i, N_i)

    torch._grouped_mm 사용 조건: 동일 K, 동일 N, BF16, SM>=80.
    조건 미충족 시 loop fallback.
    """
    assert len(group_A) == len(group_B)
    group_size = len(group_A)

    # torch._grouped_mm 사용 가능 여부 체크
    can_use_grouped_mm = (
        use_grouped_mm
        and _supports_grouped_mm()
        and group_A[0].dtype == torch.bfloat16
        and group_B[0].dtype == torch.bfloat16
    )

    # 모든 그룹이 동일 K, 동일 N인지 확인
    K0, N0 = group_A[0].shape[1], group_B[0].shape[1]
    for i in range(group_size):
        if group_A[i].shape[1] != K0 or group_B[i].shape[1] != N0:
            can_use_grouped_mm = False
            break

    if can_use_grouped_mm:
        # torch._grouped_mm 경로
        # mat_a: (sum(M_i), K) - concat along rows
        # mat_b: (num_groups, N, K) - stack B_i (각 B_i는 KxN -> 전치하여 NxK)
        # out = mat_a @ mat_b^T per group -> (sum(M_i), N)
        mat_a = torch.cat(group_A, dim=0)  # (sum(M_i), K)
        mat_b = torch.stack([b.T for b in group_B], dim=0)  # (num_groups, N, K)

        offs = torch.cumsum(
            torch.tensor([a.shape[0] for a in group_A], device=DEVICE, dtype=torch.int32),
            dim=0,
        )

        out = torch.nn.functional.grouped_mm(mat_a, mat_b, offs=offs)

        # split back to list
        M_list = [a.shape[0] for a in group_A]
        return list(out.split(M_list, dim=0))

    # Fallback: loop
    return [torch.matmul(a, b) for a, b in zip(group_A, group_B)]


def group_gemm_fn_loop(group_A: List[Tensor], group_B: List[Tensor]) -> List[Tensor]:
    """항상 loop 사용 (baseline / fallback)."""
    return [torch.matmul(a, b) for a, b in zip(group_A, group_B)]


# --- Triton과 동일한 벤치마크 구조 ---

group_m = [1024, 512, 256, 128]
group_n = [1024, 512, 256, 128]
group_k = [1024, 512, 256, 128]

# BF16으로 생성 (torch._grouped_mm 요구사항)
group_A: List[Tensor] = []
group_B: List[Tensor] = []
group_size = len(group_m)

for i in range(group_size):
    M, N, K = group_m[i], group_n[i], group_k[i]
    A = torch.rand((M, K), device=DEVICE, dtype=torch.bfloat16)
    B = torch.rand((K, N), device=DEVICE, dtype=torch.bfloat16)
    group_A.append(A)
    group_B.append(B)

# Validation
torch_out = group_gemm_fn(group_A, group_B)
ref_out = [torch.matmul(a, b) for a, b in zip(group_A, group_B)]

for i in range(group_size):
    assert torch.allclose(torch_out[i].float(), ref_out[i].float(), atol=1e-1, rtol=1e-1), (
        f"Mismatch at group {i}"
    )

print("Torch grouped GEMM validation passed.")


def torch_perf_fn(group_A: List[Tensor], group_B: List[Tensor]) -> None:
    """벤치마크용: torch._grouped_mm 또는 loop."""
    group_gemm_fn(group_A, group_B)


def torch_loop_perf_fn(group_A: List[Tensor], group_B: List[Tensor]) -> None:
    """벤치마크용: loop만 (baseline)."""
    group_gemm_fn_loop(group_A, group_B)


if __name__ == "__main__":
    import time

    warmup = 50
    repeat = 100

    for _ in range(warmup):
        torch_perf_fn(group_A, group_B)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(repeat):
        torch_perf_fn(group_A, group_B)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / repeat * 1000  # ms

    flops = sum(2 * m * n * k for m, n, k in zip(group_m, group_n, group_k))
    gflops = flops * 1e-9 / (elapsed / 1000)
    print(f"Torch grouped GEMM: {elapsed:.3f} ms, {gflops:.1f} GFLOPS")
