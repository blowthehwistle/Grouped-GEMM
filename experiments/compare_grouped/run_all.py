#!/usr/bin/env python3
"""
Grouped GEMM 백엔드별 벤치마크 실행 스크립트.

CUDA, Cutlass, PyTorch, Triton 백엔드를 순차 실행하고 결과를 results/benchmark/에 저장합니다.

Usage:
    python run_all.py [--config config.yaml]
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def run_cuda_grouped():
    """CUDA grouped GEMM 실행 (bin/tmain_grouped)"""
    exe = REPO_ROOT / "bin" / "tmain_grouped"
    if not exe.exists():
        print(f"Warning: {exe} not found. Run 'make grouped' first.")
        return None
    out_dir = REPO_ROOT / "results" / "benchmark" / "cuda"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "grouped_gemm.txt"
    with open(out_file, "w") as f:
        # kernel 2 = custom grouped, "p" = print mode
        subprocess.run([str(exe), "2", "p"], cwd=REPO_ROOT, stdout=f, stderr=subprocess.STDOUT)
    print(f"CUDA result saved to {out_file}")
    return out_file


def run_triton_grouped():
    """Triton grouped GEMM 벤치마크 실행"""
    script = REPO_ROOT / "implementations" / "triton" / "grouped_gemm.py"
    if not script.exists():
        print(f"Warning: {script} not found.")
        return None
    out_dir = REPO_ROOT / "results" / "benchmark" / "triton"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "grouped_gemm.txt"
    try:
        with open(out_file, "w") as f:
            subprocess.run(
                [sys.executable, str(script)],
                cwd=REPO_ROOT,
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=300,
            )
        print(f"Triton result saved to {out_file}")
        return out_file
    except subprocess.TimeoutExpired:
        print("Triton benchmark timed out.")
        return None
    except Exception as e:
        print(f"Triton benchmark failed: {e}")
        return None


def run_torch_grouped():
    """PyTorch grouped GEMM - TODO: 구현 후 추가"""
    print("PyTorch grouped GEMM: not implemented yet (placeholder)")
    return None


def run_cutlass_grouped():
    """Cutlass grouped GEMM - TODO: 구현 후 추가"""
    print("Cutlass grouped GEMM: not implemented yet (placeholder)")
    return None


def main():
    parser = argparse.ArgumentParser(description="Run Grouped GEMM benchmarks for all backends")
    parser.add_argument("--config", type=Path, default=None, help="Config YAML path")
    parser.add_argument("--backend", choices=["cuda", "triton", "torch", "cutlass", "all"],
                        default="all", help="Which backend to run")
    args = parser.parse_args()

    results = []

    if args.backend in ("cuda", "all"):
        results.append(("cuda", run_cuda_grouped()))
    if args.backend in ("triton", "all"):
        results.append(("triton", run_triton_grouped()))
    if args.backend in ("torch", "all"):
        results.append(("torch", run_torch_grouped()))
    if args.backend in ("cutlass", "all"):
        results.append(("cutlass", run_cutlass_grouped()))

    success = sum(1 for _, r in results if r is not None)
    print(f"\nCompleted: {success}/{len(results)} backends")
    return 0 if success > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
