"""
PyTorch Grouped GEMM. torch.grouped_mm or matmul loop(fallback)

reference:
https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.grouped_mm.html

"""

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch import Tensor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# grouped_config (implementations/에서 실행 시)
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "experiments" / "compare_grouped"))
try:
    from script_utils import load_grouped_config_module, DEFAULT_CONFIG
    _load, _add_args = load_grouped_config_module()
    load_grouped_sizes = _load
    add_grouped_args = _add_args
except ImportError:
    load_grouped_sizes = add_grouped_args = None
    DEFAULT_CONFIG = None


def _supports_grouped_mm() -> bool:
    """torch.nn.functional.grouped_mm 사용 가능 여부 (PyTorch 2.10+, BF16, SM>=80)"""
    if not torch.cuda.is_available():
        return False
    if not hasattr(torch.nn.functional, 'grouped_mm'):
        return False  # PyTorch < 2.10
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

    # check if all experts have the same K and N
    K0, N0 = group_A[0].shape[1], group_B[0].shape[1]
    for i in range(group_size):
        if group_A[i].shape[1] != K0 or group_B[i].shape[1] != N0:
            can_use_grouped_mm = False
            break


    # torch.nn.functional.grouped_mm expects mat_b shaped [group, K, N]
    # (not transposed). Each group uses A_i [M_i, K] and B_i [K, N].

    if can_use_grouped_mm:
        mat_a = torch.cat(group_A, dim=0)
        mat_b = torch.stack(group_B, dim=0)

        # offset calculation (A의 axis 0으로 누적 합).
        offs = torch.cumsum(
            torch.tensor([a.shape[0] for a in group_A], device=DEVICE, dtype=torch.int32),
            dim=0,
        )
        offs = offs.to(torch.int32).contiguous()
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
    repeat: int = 1000,
) -> Tuple[float, float]:
    """벤치마크 실행. (elapsed_ms, gflops) 반환. CUDA Event 기반 측정 (cuBLAS와 동일)."""
    group_A, group_B = make_grouped_matrices(m_list, n_list, k_list)

    for _ in range(warmup):
        torch_perf_fn(group_A, group_B)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    start_ev.record()
    # NVTX: repeat 구간만 감싸서 프로파일 시 warmup이 아닌 실제 벤치 커널만 수집
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push("grouped_gemm")
    for _ in range(repeat):
        torch_perf_fn(group_A, group_B)
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_pop()
    end_ev.record()
    torch.cuda.synchronize()
    elapsed_ms = start_ev.elapsed_time(end_ev) / repeat

    flops = sum(2 * m * n * k for m, n, k in zip(m_list, n_list, k_list))
    gflops = flops * 1e-9 / (elapsed_ms / 1000)
    return elapsed_ms, gflops


if __name__ == "__main__":
    m_list, n_list, k_list = [1024] * 8, [4096] * 8, [14336] * 8
    warmup, repeat = 50, 1000

    if load_grouped_sizes and add_grouped_args and DEFAULT_CONFIG:
        parser = argparse.ArgumentParser()
        add_grouped_args(parser)
        parser.add_argument("--warmup", type=int, default=50)
        parser.add_argument("--repeat", type=int, default=1000)
        parser.add_argument("--no-config", action="store_true")
        args = parser.parse_args()
        warmup, repeat = args.warmup, args.repeat
        if args.config or args.m or args.n or args.k or not args.no_config:
            m_list, n_list, k_list = load_grouped_sizes(
                config_path=None if args.no_config else (args.config or DEFAULT_CONFIG),
                m=args.m, n=args.n, k=args.k,
                default_config=DEFAULT_CONFIG,
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
