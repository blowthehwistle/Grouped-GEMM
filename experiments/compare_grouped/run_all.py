#!/usr/bin/env python3
"""
Grouped GEMM 백엔드별 벤치마크 실행 스크립트.

CUDA, Cutlass, PyTorch, Triton 백엔드가 동일한 config로 벤치마크 (config.yaml 기준).

Usage:
    python run_all.py [--config config.yaml]
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = REPO_ROOT / "experiments" / "compare_grouped"


def load_unified_config(config_path):
    """통합 config 로드. (m_list, n_list, k_list) 반환."""
    sys.path.insert(0, str(CONFIG_DIR))
    try:
        from grouped_config import load_grouped_sizes
        return load_grouped_sizes(
            config_path=config_path,
            default_config=CONFIG_DIR / "config.yaml",
        )
    finally:
        if str(CONFIG_DIR) in sys.path:
            sys.path.remove(str(CONFIG_DIR))


KERNEL_NAMES = {
    0: "cuBLAS Loop",
    1: "cuBLAS Grouped API",
    2: "Custom Double Buffering",
}


def run_cuda_grouped(config_path=None):
    """CUDA grouped GEMM 실행 (bin/tmain_grouped). Kernel 0,1,2 모두 실행."""
    exe = REPO_ROOT / "bin" / "tmain_grouped"
    if not exe.exists():
        print(f"Warning: {exe} not found. Run 'make grouped' first.")
        return None
    m_list, n_list, k_list = load_unified_config(config_path)
    out_dir = REPO_ROOT / "results" / "benchmark" / "cuda"
    out_dir.mkdir(parents=True, exist_ok=True)
    cuda_config_file = out_dir / "grouped_config.txt"
    with open(cuda_config_file, "w") as f:
        for m, n, k in zip(m_list, n_list, k_list):
            f.write(f"{m} {n} {k}\n")

    out_file = out_dir / "grouped_gemm.txt"
    lines = [
        "=== CUDA Grouped GEMM Benchmark ===",
        f"Batch size: {len(m_list)}",
        f"Shapes: M={list(m_list)}, N={list(n_list)}, K={list(k_list)}",
        "",
    ]

    for kernel_num in (0, 1, 2):
        section = f"--- Kernel {kernel_num} ({KERNEL_NAMES[kernel_num]}) ---"
        lines.append(section)
        cmd = [str(exe), str(kernel_num), "p", str(cuda_config_file.resolve())]
        result = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        out = (result.stdout or "") + (result.stderr or "")
        lines.append(out.rstrip())
        if result.returncode != 0:
            lines.append(f"[Kernel {kernel_num} failed with exit code {result.returncode}]")
        lines.append("")

    with open(out_file, "w") as f:
        f.write("\n".join(lines))

    print(f"CUDA result saved to {out_file} (kernels 0, 1, 2)")
    return out_file


def run_triton_grouped(config_path=None):
    """Triton grouped GEMM 벤치마크 (config와 동일 M,N,K)"""
    script = REPO_ROOT / "implementations" / "triton" / "triton_grouped_gemm.py"
    if not script.exists():
        print(f"Warning: {script} not found.")
        return None
    out_dir = REPO_ROOT / "results" / "benchmark" / "triton"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "grouped_gemm.txt"
    config = config_path or (CONFIG_DIR / "config.yaml")
    try:
        cmd = [sys.executable, str(script), "--benchmark-only", "--config", str(config),
               "--repeat", "1000", "--warmup", "50"]
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
    """PyTorch grouped GEMM 벤치마크 (config와 동일 M,N,K)"""
    script = REPO_ROOT / "implementations" / "torch" / "torch_grouped_gemm.py"
    if not script.exists():
        print(f"Warning: {script} not found.")
        return None
    out_dir = REPO_ROOT / "results" / "benchmark" / "torch"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "grouped_gemm.txt"
    config = config_path or (CONFIG_DIR / "config.yaml")
    try:
        cmd = [sys.executable, str(script), "--config", str(config),
               "--repeat", "1000", "--warmup", "50"]
        with open(out_file, "w") as f:
            subprocess.run(cmd, cwd=REPO_ROOT, stdout=f, stderr=subprocess.STDOUT, timeout=120)
        print(f"Torch result saved to {out_file}")
        return out_file
    except Exception as e:
        print(f"Torch benchmark failed: {e}")
        return None


def run_cutlass_grouped(config_path=None):
    """Cutlass grouped GEMM (bin/tmain_cutlass_grouped). CUDA와 동일 config 사용."""
    exe = REPO_ROOT / "bin" / "tmain_cutlass_grouped"
    if not exe.exists():
        print("Warning: Cutlass binary not found. Run 'make cutlass_grouped' first (requires CUTLASS_ROOT).")
        return None
    m_list, n_list, k_list = load_unified_config(config_path)
    out_dir = REPO_ROOT / "results" / "benchmark" / "cutlass"
    out_dir.mkdir(parents=True, exist_ok=True)
    config_file = out_dir / "grouped_config.txt"
    with open(config_file, "w") as f:
        for m, n, k in zip(m_list, n_list, k_list):
            f.write(f"{m} {n} {k}\n")
    out_file = out_dir / "grouped_gemm.txt"
    try:
        cmd = [str(exe), str(config_file.resolve()), "p"]
        with open(out_file, "w") as f:
            subprocess.run(cmd, cwd=REPO_ROOT, stdout=f, stderr=subprocess.STDOUT, timeout=120)
        print(f"Cutlass result saved to {out_file}")
        return out_file
    except Exception as e:
        print(f"Cutlass benchmark failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Run Grouped GEMM benchmarks for all backends")
    parser.add_argument("--config", type=Path, default=None,
                        help="Config YAML path (grouped.m, grouped.n, grouped.k)")
    parser.add_argument("--backend", choices=["cuda", "triton", "torch", "cutlass", "all"],
                        default="all", help="Which backend to run")
    parser.add_argument("--no-compare", action="store_true",
                        help="Skip printing unified comparison at the end")
    args = parser.parse_args()

    config_path = args.config
    if not config_path and (REPO_ROOT / "experiments" / "compare_grouped" / "config.yaml").exists():
        config_path = REPO_ROOT / "experiments" / "compare_grouped" / "config.yaml"

    results = []

    if args.backend in ("cuda", "all"):
        results.append(("cuda", run_cuda_grouped(config_path)))
    if args.backend in ("triton", "all"):
        results.append(("triton", run_triton_grouped(config_path)))
    if args.backend in ("torch", "all"):
        results.append(("torch", run_torch_grouped(config_path)))
    if args.backend in ("cutlass", "all"):
        results.append(("cutlass", run_cutlass_grouped(config_path)))

    success = sum(1 for _, r in results if r is not None)
    print(f"\nCompleted: {success}/{len(results)} backends")

    if not args.no_compare and success > 0:
        sys.path.insert(0, str(CONFIG_DIR))
        try:
            from compare import load_and_parse, format_unified_report
            config_str, rows = load_and_parse()
            print("\n" + format_unified_report(config_str, rows))
        except ImportError:
            pass
        finally:
            if str(CONFIG_DIR) in sys.path:
                sys.path.remove(str(CONFIG_DIR))

    return 0 if success > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
