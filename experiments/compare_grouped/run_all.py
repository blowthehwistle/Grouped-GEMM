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


def run_triton_grouped(config_path=None):
    """Triton grouped GEMM 벤치마크 실행"""
    script = REPO_ROOT / "implementations" / "triton" / "triton_grouped_gemm.py"
    if not script.exists():
        print(f"Warning: {script} not found.")
        return None
    out_dir = REPO_ROOT / "results" / "benchmark" / "triton"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "grouped_gemm.txt"
    try:
        cmd = [sys.executable, str(script), "--benchmark-only"]
        if config_path:
            cmd.extend(["--config", str(config_path)])
        with open(out_file, "w") as f:
            subprocess.run(
                cmd,
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


def run_torch_grouped(config_path=None):
    """PyTorch grouped GEMM 벤치마크 실행"""
    script = REPO_ROOT / "implementations" / "torch" / "torch_grouped_gemm.py"
    if not script.exists():
        print(f"Warning: {script} not found.")
        return None
    out_dir = REPO_ROOT / "results" / "benchmark" / "torch"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "grouped_gemm.txt"
    try:
        cmd = [sys.executable, str(script)]
        if config_path:
            cmd.extend(["--config", str(config_path)])
        with open(out_file, "w") as f:
            subprocess.run(cmd, cwd=REPO_ROOT, stdout=f, stderr=subprocess.STDOUT, timeout=120)
        print(f"Torch result saved to {out_file}")
        return out_file
    except Exception as e:
        print(f"Torch benchmark failed: {e}")
        return None


def run_cutlass_grouped():
    """Cutlass grouped GEMM - TODO: 구현 후 추가"""
    print("Cutlass grouped GEMM: not implemented yet (placeholder)")
    return None


def main():
    parser = argparse.ArgumentParser(description="Run Grouped GEMM benchmarks for all backends")
    parser.add_argument("--config", type=Path, default=None,
                        help="Config YAML path (grouped.m, grouped.n, grouped.k)")
    parser.add_argument("--backend", choices=["cuda", "triton", "torch", "cutlass", "all"],
                        default="all", help="Which backend to run")
    args = parser.parse_args()

    config_path = args.config
    if not config_path and (REPO_ROOT / "experiments" / "compare_grouped" / "config.yaml").exists():
        config_path = REPO_ROOT / "experiments" / "compare_grouped" / "config.yaml"

    results = []

    if args.backend in ("cuda", "all"):
        results.append(("cuda", run_cuda_grouped()))
    if args.backend in ("triton", "all"):
        results.append(("triton", run_triton_grouped(config_path)))
    if args.backend in ("torch", "all"):
        results.append(("torch", run_torch_grouped(config_path)))
    if args.backend in ("cutlass", "all"):
        results.append(("cutlass", run_cutlass_grouped()))

    success = sum(1 for _, r in results if r is not None)
    print(f"\nCompleted: {success}/{len(results)} backends")
    return 0 if success > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
