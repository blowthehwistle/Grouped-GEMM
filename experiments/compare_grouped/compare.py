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
KERNEL_NAMES = {0: "cuBLAS Loop (TF32)", 1: "cuBLAS Grouped API (TF32)", 2: "Custom Double Buffering (TF32)"}


def _check_validation(text: str) -> Optional[bool]:
    """'Result is correct' / 'validation passed' -> True, 'Result is different' -> False."""
    if "Result is correct" in text or "validation passed" in text:
        return True
    if "Result is different" in text:
        return False
    return None


def _parse_cuda(content: str) -> Tuple[Optional[Dict[str, Tuple[float, float, Optional[bool]]]], Optional[str]]:
    """CUDA 결과 파싱. (backend_name -> (ms, gflops, validation)) dict, config_str 반환."""
    config = None
    results = {}

    if "Batch size:" in content:
        m = re.search(r"Batch size: (\d+)\nShapes: (.+)", content)
        if m:
            config = f"Batch size: {m.group(1)}, Shapes: {m.group(2).strip()}"
        else:
            m = re.search(r"Batch size: (\d+)", content)
            config = f"Batch size: {m.group(1)}" if m else ""

    # Match [raw] only within the same section. Do NOT stop at "--- Performance ---"
    # (which appears before [raw]); stop only at next section header (Kernel N or CUDA FP16).
    _section_body = r"(?:(?!\n--- (?:Kernel \d \(|CUDA FP16 (?:Loop|Grouped) |cuBLAS (?:Loop|Grouped API) \())[\s\S])*?"
    for k in (0, 1, 2):
        pat = rf"--- Kernel {k} \([^)]+\) ---{_section_body}\[raw\]\s+([\d.]+),([\d.]+),([\d.]+),([\d.]+)"
        m = re.search(pat, content)
        if m:
            full_match = m.group(0)
            section = full_match[:full_match.index("[raw]")]
            t_kernel = float(m.group(2))
            gflops_kernel = float(m.group(4))
            ms = t_kernel * 1000
            val = _check_validation(section)
            results[KERNEL_NAMES[k]] = (ms, gflops_kernel, val)

    for old_hdr, display_name in [
        ("CUDA FP16 Loop", "cuBLAS Loop (FP16)"),
        ("CUDA FP16 Grouped", "cuBLAS Grouped API (FP16)"),
    ]:
        pat = rf"--- (?:{re.escape(old_hdr)}|{re.escape(display_name)}) ---{_section_body}\[raw\]\s+([\d.]+),([\d.]+),([\d.]+),([\d.]+)"
        m = re.search(pat, content)
        if m:
            full_match = m.group(0)
            section = full_match[:full_match.index("[raw]")]
            t_kernel = float(m.group(2))
            gflops_kernel = float(m.group(4))
            ms = t_kernel * 1000
            val = _check_validation(section)
            results[display_name] = (ms, gflops_kernel, val)

    return results if results else None, config


def _parse_torch(content: str) -> Optional[Tuple[float, float, Optional[bool]]]:
    """Torch: perf + validation 파싱."""
    m = re.search(r"Torch grouped GEMM:\s*([\d.]+)\s*ms,\s*([\d.]+)\s*GFLOPS", content)
    if m:
        val = _check_validation(content)
        return float(m.group(1)), float(m.group(2)), val
    return None


def _parse_triton(content: str) -> Optional[Tuple[float, float, Optional[bool]]]:
    """Triton: perf + validation 파싱."""
    m = re.search(r"Triton grouped GEMM:\s*([\d.]+)\s*ms,\s*([\d.]+)\s*GFLOPS", content)
    if m:
        val = _check_validation(content)
        return float(m.group(1)), float(m.group(2)), val
    return None


def _parse_cutlass(content: str) -> Optional[Tuple[float, float, Optional[bool]]]:
    """Cutlass: perf + validation 파싱."""
    val = _check_validation(content)
    m = re.search(r"Cutlass grouped GEMM:\s*([\d.]+)\s*ms,\s*([\d.]+)\s*GFLOPS", content)
    if m:
        return float(m.group(1)), float(m.group(2)), val
    m = re.search(r"\[raw\]\s+[\d.]+,([\d.]+),[\d.]+,([\d.]+)", content)
    if m:
        return float(m.group(1)) * 1000, float(m.group(2)), val
    return None


def load_and_parse(benchmark_dir=None):
    """각 백엔드 결과 파일 파싱. config, rows 반환. rows: (name, ms, gflops, validation)."""
    config_str = None
    rows = []  # (name, ms, gflops, validation: True/False/None)
    base = Path(benchmark_dir) if benchmark_dir is not None else BENCHMARK_DIR

    for backend in ["cuda", "torch", "triton", "cutlass"]:
        path = base / backend / "grouped_gemm.txt"
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
                for name, (ms, gflops, val) in parsed.items():
                    rows.append((name, ms, gflops, val))
        elif backend == "torch":
            parsed = _parse_torch(content)
            if parsed:
                ms, gflops, val = parsed
                rows.append(("Torch", ms, gflops, val))
        elif backend == "triton":
            parsed = _parse_triton(content)
            if parsed:
                ms, gflops, val = parsed
                rows.append(("Triton", ms, gflops, val))
        elif backend == "cutlass":
            parsed = _parse_cutlass(content)
            if parsed:
                ms, gflops, val = parsed
                rows.append(("Cutlass", ms, gflops, val))
            if not config_str and "Batch size:" in content:
                m = re.search(r"Batch size: (\d+)\nShapes: (.+)", content)
                if m:
                    config_str = f"Batch size: {m.group(1)}, Shapes: {m.group(2).strip()}"

    return config_str, rows


def _validation_str(val: Optional[bool]) -> str:
    if val is True:
        return "PASS"
    if val is False:
        return "FAIL"
    return "N/A"


def write_results_csv(rows: List, path: Path) -> None:
    """Write benchmark rows to CSV. rows: (name, ms, gflops, validation)."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("Backend,Time_ms,GFLOPS,Validation\n")
        for name, ms, gflops, val in rows:
            vstr = _validation_str(val)
            f.write(f"{name},{ms:.4f},{gflops:.2f},{vstr}\n")


def format_unified_report(config_str: Optional[str], rows: List) -> str:
    """통일된 형식 리포트 문자열 생성. rows: (name, ms, gflops, validation)."""
    lines = [
        "=== Grouped GEMM Benchmark ===",
        "",
    ]
    if config_str:
        lines.append(f"Config: {config_str}")
        lines.append("")

    if not rows:
        return "\n".join(lines) + "\n(No benchmark results parsed)\n"

    lines.append("--- Validation ---")
    for name, _, _, val in rows:
        lines.append(f"  {name:<24}: {_validation_str(val)}")
    lines.append("")

    lines.append(f"{'Backend':<24} | {'Time (ms)':>10} | {'GFLOPS':>12} | {'Validation':>8}")
    lines.append("-" * 62)
    for name, ms, gflops, val in rows:
        lines.append(f"{name:<24} | {ms:>10.2f} | {gflops:>12.1f} | {_validation_str(val):>8}")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare Grouped GEMM benchmark results")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output file path (default: results/tuned/compare_grouped.txt)")
    parser.add_argument("--benchmark-dir", type=Path, default=None,
                        help="Benchmark results dir (default: results/benchmark)")
    parser.add_argument("--no-write", action="store_true",
                        help="Only print to stdout, do not write file")
    args = parser.parse_args()

    config_str, rows = load_and_parse(benchmark_dir=args.benchmark_dir)
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
