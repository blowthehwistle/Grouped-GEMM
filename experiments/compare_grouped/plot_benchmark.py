#!/usr/bin/env python3
"""
벤치마크 결과 CSV를 읽어 Backend별 GFLOPS 막대그래프를 그립니다.

Usage:
    # 단일 config 결과 (benchmark_results.csv)
    python plot_benchmark.py
    python plot_benchmark.py --csv results/benchmark/benchmark_results.csv

    # 여러 config 요약 (run_multi_config.py --summary-csv로 생성)
    python plot_benchmark.py --csv results/summary.csv
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")  # headless

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def load_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    """CSV 로드. (컬럼명 리스트, 행 딕셔너리 리스트) 반환."""
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        for row in reader:
            rows.append(dict(row))
    return cols, rows


def plot_single_config(rows: List[Dict], out_path: Path, title: Optional[str]) -> None:
    """Backend별 GFLOPS 막대그래프 (단일 config)."""
    backends = [r["Backend"] for r in rows]
    gflops = [float(r["GFLOPS"]) for r in rows]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(backends, gflops, color=["#2ecc71", "#3498db", "#e74c3c", "#9b59b6", "#f39c12", "#1abc9c"][: len(backends)])
    ax.set_ylabel("GFLOPS", fontsize=12)
    ax.set_xlabel("Backend", fontsize=12)
    if title:
        ax.set_title(title, fontsize=14)
    ax.set_ylim(bottom=0)

    for bar, val in zip(bars, gflops):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02 * max(gflops),
                f"{val:.0f}", ha="center", va="bottom", fontsize=10)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def plot_multi_config(rows: List[Dict], out_path: Path, title: Optional[str]) -> None:
    """Config×Backend별 GFLOPS 그룹 막대그래프."""
    configs = sorted(set(r["Config"] for r in rows))
    backends = []
    for r in rows:
        b = r["Backend"]
        if b not in backends:
            backends.append(b)

    import numpy as np
    x = np.arange(len(configs))
    width = 0.8 / len(backends)

    fig, ax = plt.subplots(figsize=(max(10, len(configs) * 1.5), 6))
    colors = ["#2ecc71", "#3498db", "#e74c3c", "#9b59b6", "#f39c12", "#1abc9c"]
    for i, backend in enumerate(backends):
        vals = []
        for cfg in configs:
            row = next((r for r in rows if r["Config"] == cfg and r["Backend"] == backend), None)
            vals.append(float(row["GFLOPS"]) if row else 0)
        off = (i - len(backends) / 2 + 0.5) * width
        bars = ax.bar(x + off, vals, width, label=backend, color=colors[i % len(colors)])

    ax.set_ylabel("GFLOPS", fontsize=12)
    ax.set_xlabel("Config", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=25, ha="right")
    ax.legend()
    ax.set_ylim(bottom=0)
    if title:
        ax.set_title(title, fontsize=14)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot benchmark GFLOPS from CSV")
    parser.add_argument("--csv", type=Path, default=None,
                        help="Input CSV (default: results/benchmark/benchmark_results.csv)")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output image path (default: same dir as CSV, .png)")
    parser.add_argument("--title", type=str, default=None, help="Chart title")
    args = parser.parse_args()

    csv_path = args.csv or (REPO_ROOT / "results" / "benchmark" / "benchmark_results.csv")
    if not csv_path.is_absolute():
        csv_path = REPO_ROOT / csv_path
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        print("  Run: python experiments/compare_grouped/run_all.py (or run_multi_config.py --summary-csv)")
        return 1

    cols, rows = load_csv(csv_path)
    if not rows:
        print("No data in CSV")
        return 1

    if "Config" in cols and len(set(r.get("Config", "") for r in rows)) > 1:
        plot_multi_config(rows, args.output or csv_path.with_suffix(".png"), args.title)
    else:
        plot_single_config(rows, args.output or csv_path.with_suffix(".png"), args.title)

    return 0


if __name__ == "__main__":
    exit(main())
