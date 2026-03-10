#!/usr/bin/env python3
"""
Grouped GEMM 백엔드별 결과 비교 및 시각화.

results/benchmark/{cuda,cutlass,torch,triton}/ 의 결과를 읽어 비교합니다.

Usage:
    python compare.py [--output results/tuned/compare_grouped.png]
"""

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BENCHMARK_DIR = REPO_ROOT / "results" / "benchmark"


def load_results():
    """각 백엔드별 결과 파일에서 GFLOPS 등 추출"""
    backends = ["cuda", "cutlass", "torch", "triton"]
    data = {}
    for backend in backends:
        path = BENCHMARK_DIR / backend / "grouped_gemm.txt"
        if path.exists():
            try:
                with open(path) as f:
                    content = f.read()
                # 간단한 파싱 (형식에 따라 조정 필요)
                data[backend] = content
            except Exception as e:
                print(f"Warning: could not read {path}: {e}")
        else:
            print(f"Info: {path} not found, skipping {backend}")
    return data


def main():
    parser = argparse.ArgumentParser(description="Compare Grouped GEMM benchmark results")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output plot path")
    args = parser.parse_args()

    data = load_results()
    if not data:
        print("No benchmark results found. Run experiments/compare_grouped/run_all.py first.")
        return 1

    out_path = args.output or REPO_ROOT / "results" / "tuned" / "compare_grouped.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        f.write("Grouped GEMM Benchmark Comparison\n")
        f.write("=" * 60 + "\n\n")
        for backend, content in data.items():
            f.write(f"--- {backend} ---\n")
            f.write(content[:2000])  # 처음 2000자
            if len(content) > 2000:
                f.write("\n... (truncated)\n")
            f.write("\n\n")

    print(f"Comparison written to {out_path}")
    return 0


if __name__ == "__main__":
    exit(main())
