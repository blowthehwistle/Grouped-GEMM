"""
PyTorch Grouped GEMM - Triton 스타일 API

torch._grouped_mm을 사용하여 Triton grouped_gemm과 동일한 인터페이스 제공.
- group_gemm_fn(group_A, group_B): 리스트 형태 입력 → 리스트 형태 출력
- M, N, K는 config/CLI로 주입 가능 (run_all, 통합 실험용)
"""

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch import Tensor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# grouped_config import
_config_module = Path(__file__).resolve().parent.parent.parent / "experiments" / "compare_grouped" / "grouped_config.py"
if _config_module.exists():
    import importlib.util
    spec = importlib.util.spec_from_file_location("grouped_config", _config_module)
    _gc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_gc)
    load_grouped_sizes = _gc.load_grouped_sizes
    add_grouped_args = _gc.add_grouped_args
else:
    load_grouped_sizes = add_grouped_args = None


def _supports_grouped_mm() -> bool:
    """torch._grouped_mm 사용 가능 여부 (BF16, SM>=80)"""
    if not torch.cuda.is_available():
        return False
    try:
        cap = torch.cuda.get_device_capability()
        return cap[0] >= 8  # SM >= 80
    except Exception:
        return False


def make_grouped_matrices(
    m_list: List[int],
    n_list: List[int],
    k_list: List[int],
    dtype: torch.dtype = torch.bfloat16,
    device: Optional[torch.device] = None,
):
    """M, N, K 리스트로 group_A, group_B 생성."""
    device = device or DEVICE
    group_A = []
    group_B = []
    for M, N, K in zip(m_list, n_list, k_list):
        A = torch.rand((M, K), device=device, dtype=dtype)
        B = torch.rand((K, N), device=device, dtype=dtype)
        group_A.append(A)
        group_B.append(B)
    return group_A, group_B


def group_gemm_fn(
    group_A: List[Tensor],
    group_B: List[Tensor],
    use_grouped_mm: bool = True,
) -> List[Tensor]:
    """
    Triton group_gemm_fn과 동일한 API.

    group_A[i] @ group_B[i] for each i.
    torch._grouped_mm 사용 조건: 동일 K, 동일 N, BF16, SM>=80.
    """
    assert len(group_A) == len(group_B)
    group_size = len(group_A)

    can_use_grouped_mm = (
        use_grouped_mm
        and _supports_grouped_mm()
        and group_A[0].dtype == torch.bfloat16
        and group_B[0].dtype == torch.bfloat16
    )

    K0, N0 = group_A[0].shape[1], group_B[0].shape[1]
    for i in range(group_size):
        if group_A[i].shape[1] != K0 or group_B[i].shape[1] != N0:
            can_use_grouped_mm = False
            break

    if can_use_grouped_mm:
        mat_a = torch.cat(group_A, dim=0)
        mat_b = torch.stack([b.T for b in group_B], dim=0)
        offs = torch.cumsum(
            torch.tensor([a.shape[0] for a in group_A], device=DEVICE, dtype=torch.int32),
            dim=0,
        )
        out = torch.nn.functional.grouped_mm(mat_a, mat_b, offs=offs)
        M_list = [a.shape[0] for a in group_A]
        return list(out.split(M_list, dim=0))

    return [torch.matmul(a, b) for a, b in zip(group_A, group_B)]


def group_gemm_fn_loop(group_A: List[Tensor], group_B: List[Tensor]) -> List[Tensor]:
    """항상 loop 사용 (baseline / fallback)."""
    return [torch.matmul(a, b) for a, b in zip(group_A, group_B)]


def torch_perf_fn(group_A: List[Tensor], group_B: List[Tensor]) -> None:
    """벤치마크용."""
    group_gemm_fn(group_A, group_B)


def run_benchmark(
    m_list: List[int],
    n_list: List[int],
    k_list: List[int],
    warmup: int = 50,
    repeat: int = 100,
) -> Tuple[float, float]:
    """벤치마크 실행. (elapsed_ms, gflops) 반환."""
    group_A, group_B = make_grouped_matrices(m_list, n_list, k_list)

    for _ in range(warmup):
        torch_perf_fn(group_A, group_B)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    import time
    start = time.perf_counter()
    for _ in range(repeat):
        torch_perf_fn(group_A, group_B)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) / repeat * 1000

    flops = sum(2 * m * n * k for m, n, k in zip(m_list, n_list, k_list))
    gflops = flops * 1e-9 / (elapsed_ms / 1000)
    return elapsed_ms, gflops


if __name__ == "__main__":
    m_list = [1024, 512, 256, 128]
    n_list = [1024, 512, 256, 128]
    k_list = [1024, 512, 256, 128]
    warmup, repeat = 50, 100

    if load_grouped_sizes and add_grouped_args:
        parser = argparse.ArgumentParser()
        add_grouped_args(parser)
        parser.add_argument("--warmup", type=int, default=50)
        parser.add_argument("--repeat", type=int, default=100)
        parser.add_argument("--no-config", action="store_true", help="config 로드 안 함")
        args = parser.parse_args()
        warmup, repeat = args.warmup, args.repeat
        if not args.no_config or args.config or args.m or args.n or args.k:
            default_cfg = Path(__file__).resolve().parent.parent.parent / "experiments" / "compare_grouped" / "config.yaml"
            m_list, n_list, k_list = load_grouped_sizes(
                config_path=args.config if not args.no_config else None,
                m=args.m,
                n=args.n,
                k=args.k,
                default_config=default_cfg,
            )

    # Validation
    group_A, group_B = make_grouped_matrices(m_list, n_list, k_list)
    torch_out = group_gemm_fn(group_A, group_B)
    ref_out = [torch.matmul(a, b) for a, b in zip(group_A, group_B)]
    for i in range(len(m_list)):
        assert torch.allclose(torch_out[i].float(), ref_out[i].float(), atol=1e-1, rtol=1e-1)
    print("Torch grouped GEMM validation passed.")

    elapsed_ms, gflops = run_benchmark(m_list, n_list, k_list, warmup=warmup, repeat=repeat)
    print(f"M,N,K: {m_list}, {n_list}, {k_list}")
    print(f"Torch grouped GEMM: {elapsed_ms:.3f} ms, {gflops:.1f} GFLOPS")
