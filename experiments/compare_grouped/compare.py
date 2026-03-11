#!/usr/bin/env python3
"""
Grouped GEMM 백엔드별 결과 비교. 통일된 형식으로 출력합니다.

results/benchmark/{cuda,torch,triton}/ 의 결과를 파싱해 config 한 번, 단위 통일(ms, GFLOPS)로 표시.

Usage:
    python compare.py [--output results/tuned/compare_grouped.txt]
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BENCHMARK_DIR = REPO_ROOT / "results" / "benchmark"

# CUDA kernel display names
KERNEL_NAMES = {0: "cuBLAS Loop", 1: "cuBLAS Grouped API", 2: "Custom Double Buffering"}


def _parse_cuda(content: str) -> Tuple[Optional[Dict[str, Tuple[float, float]]], Optional[str]]:
    """CUDA 결과 파싱. (backend_name -> (ms, gflops)) dict, config_str 반환."""
    config = None
    results = {}

    if "Batch size:" in content:
        m = re.search(r"Batch size: (\d+)\nShapes: (.+)", content)
        if m:
            config = f"Batch size: {m.group(1)}, Shapes: {m.group(2).strip()}"
        else:
            m = re.search(r"Batch size: (\d+)", content)
            config = f"Batch size: {m.group(1)}" if m else ""

    # Kernel별 [raw] 라인 파싱: t_ref,t_kernel,gflops_ref,gflops_kernel (초 단위)
    for k in (0, 1, 2):
        pat = rf"--- Kernel {k} \([^)]+\) ---.*?\[raw\]\s+([\d.]+),([\d.]+),([\d.]+),([\d.]+)"
        m = re.search(pat, content, re.DOTALL)
        if m:
            _, t_kernel, _, gflops_kernel = map(float, m.groups())
            ms = t_kernel * 1000
            results[KERNEL_NAMES[k]] = (ms, gflops_kernel)

    return results if results else None, config


def _parse_torch(content: str) -> Optional[Tuple[float, float]]:
    """Torch: 'Torch grouped GEMM: X ms, Y GFLOPS' 파싱."""
    m = re.search(r"Torch grouped GEMM:\s*([\d.]+)\s*ms,\s*([\d.]+)\s*GFLOPS", content)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None


def _parse_triton(content: str) -> Optional[Tuple[float, float]]:
    """Triton: 'Triton grouped GEMM: X ms, Y GFLOPS' 파싱."""
    m = re.search(r"Triton grouped GEMM:\s*([\d.]+)\s*ms,\s*([\d.]+)\s*GFLOPS", content)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None


def _parse_cutlass(content: str) -> Optional[Tuple[float, float]]:
    """Cutlass: 'Cutlass grouped GEMM: X ms, Y GFLOPS' 또는 [raw] 0,t,0,gflops 파싱."""
    m = re.search(r"Cutlass grouped GEMM:\s*([\d.]+)\s*ms,\s*([\d.]+)\s*GFLOPS", content)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = re.search(r"\[raw\]\s+[\d.]+,([\d.]+),[\d.]+,([\d.]+)", content)
    if m:
        t_sec = float(m.group(1))
        gflops = float(m.group(2))
        return t_sec * 1000, gflops
    return None


def load_and_parse():
    """각 백엔드 결과 파일 파싱. config, rows 반환."""
    config_str = None
    rows = []  # (name, ms, gflops)

    for backend in ["cuda", "torch", "triton", "cutlass"]:
        path = BENCHMARK_DIR / backend / "grouped_gemm.txt"
        if not path.exists():
            continue
        try:
            content = path.read_text()
        except Exception as e:
            print(f"Warning: could not read {path}: {e}")
            continue

        if backend == "cuda":
            parsed, cfg = _parse_cuda(content)
            if cfg:
                config_str = cfg
            if parsed:
                for name, (ms, gflops) in parsed.items():
                    rows.append((name, ms, gflops))
        elif backend == "torch":
            parsed = _parse_torch(content)
            if parsed:
                ms, gflops = parsed
                rows.append(("Torch", ms, gflops))
        elif backend == "triton":
            parsed = _parse_triton(content)
            if parsed:
                ms, gflops = parsed
                rows.append(("Triton", ms, gflops))
        elif backend == "cutlass":
            parsed = _parse_cutlass(content)
            if parsed:
                ms, gflops = parsed
                rows.append(("Cutlass", ms, gflops))
            if not config_str and "Batch size:" in content:
                m = re.search(r"Batch size: (\d+)\nShapes: (.+)", content)
                if m:
                    config_str = f"Batch size: {m.group(1)}, Shapes: {m.group(2).strip()}"

    return config_str, rows


def format_unified_report(config_str: Optional[str], rows: List) -> str:
    """통일된 형식 리포트 문자열 생성."""
    lines = [
        "=== Grouped GEMM Benchmark ===",
        "",
    ]
    if config_str:
        lines.append(f"Config: {config_str}")
        lines.append("")

    if not rows:
        return "\n".join(lines) + "\n(No benchmark results parsed)\n"

    lines.append(f"{'Backend':<24} | {'Time (ms)':>10} | {'GFLOPS':>12}")
    lines.append("-" * 52)
    for name, ms, gflops in rows:
        lines.append(f"{name:<24} | {ms:>10.2f} | {gflops:>12.1f}")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare Grouped GEMM benchmark results")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output file path (default: results/tuned/compare_grouped.txt)")
    parser.add_argument("--no-write", action="store_true",
                        help="Only print to stdout, do not write file")
    args = parser.parse_args()

    config_str, rows = load_and_parse()
    report = format_unified_report(config_str, rows)

    if not args.no_write:
        out_path = args.output or REPO_ROOT / "results" / "tuned" / "compare_grouped.txt"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report)
        print(f"Comparison written to {out_path}")

    print(report)
    return 0 if rows else 1


if __name__ == "__main__":
    exit(main())
