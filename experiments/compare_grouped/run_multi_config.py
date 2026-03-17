#!/usr/bin/env python3
from __future__ import annotations

"""
여러 config로 benchmark_end_to_end.py를 순차 실행하고, config별로 결과를 저장합니다.

Usage:
    python run_multi_config.py
    python run_multi_config.py --config-dir configs
    python run_multi_config.py --configs configs/default.yaml configs/expert_skewed.yaml
    python run_multi_config.py --backend cuda
"""

import argparse
import subprocess
import sys
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent
REPO_ROOT = CONFIG_DIR.parent.parent
DEFAULT_CONFIGS_DIR = CONFIG_DIR / "configs"


def discover_configs(config_dir: Path) -> list[tuple[Path, str]]:
    """config_dir 내 *.yaml 파일을 (path, stem) 리스트로 반환."""
    if not config_dir.exists():
        return []
    out = []
    for p in sorted(config_dir.glob("*.yaml")):
        out.append((p, p.stem))
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Run benchmark for multiple configs, saving results per config."
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=DEFAULT_CONFIGS_DIR,
        help=f"Config directory (default: {DEFAULT_CONFIGS_DIR})",
    )
    parser.add_argument(
        "--configs",
        type=Path,
        nargs="+",
        default=None,
        help="Explicit list of config files (overrides --config-dir)",
    )
    parser.add_argument(
        "--backend",
        choices=["cuda", "triton", "torch", "cutlass", "all"],
        default="all",
        help="Backend to run",
    )
    parser.add_argument(
        "--no-compare",
        action="store_true",
        help="Skip per-config unified report",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print summary table at the end (all configs)",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Write aggregated CSV (Config,Backend,Time_ms,GFLOPS,Validation)",
    )
    parser.add_argument(
        "--nsight",
        action="store_true",
        help="Profile with Nsight Compute (per config)",
    )
    args = parser.parse_args()

    if args.configs:
        config_list = [(Path(p), Path(p).stem) for p in args.configs]
    else:
        config_list = discover_configs(args.config_dir)

    if not config_list:
        print("No config files found.", file=sys.stderr)
        if not args.configs:
            print(f"  Add YAML files to {args.config_dir}", file=sys.stderr)
        return 1

    results_base = REPO_ROOT / "results" / "benchmark"
    summary_data = []  # (config_name, config_str, rows)

    for config_path, config_name in config_list:
        if not config_path.exists():
            print(f"Skip (not found): {config_path}", flush=True)
            continue
        config_path = config_path.resolve()
        print(f"\n{'='*60}")
        print(f"Config: {config_name} ({config_path})")
        print("=" * 60, flush=True)

        cmd = [
            sys.executable,
            str(CONFIG_DIR / "benchmark_end_to_end.py"),
            "--config",
            str(config_path),
            "--results-subdir",
            config_name,
            "--backend",
            args.backend,
        ]
        if args.no_compare:
            cmd.append("--no-compare")
        if args.nsight:
            cmd.append("--nsight")

        ret = subprocess.run(cmd, cwd=REPO_ROOT)
        if ret.returncode != 0:
            print(f"[WARN] {config_name} exited with {ret.returncode}", flush=True)

        if (args.summary or args.summary_csv) and not args.no_compare:
            try:
                sys.path.insert(0, str(CONFIG_DIR))
                from compare import load_and_parse, format_unified_report
                cfg_dir = results_base / config_name
                config_str, rows = load_and_parse(benchmark_dir=cfg_dir)
                summary_data.append((config_name, config_str, rows))
            except ImportError:
                pass
            finally:
                if str(CONFIG_DIR) in sys.path:
                    sys.path.remove(str(CONFIG_DIR))

    print(f"\n{'='*60}")
    print("All configs completed.")
    print(f"Results saved under: {results_base}/")
    for _, name in config_list:
        print(f"  - {name}/")
    print("=" * 60)

    # CUDA 요청했는데 결과가 없으면 빌드 안내 (다른 머신에서 make grouped 누락 시)
    if args.backend in ("cuda", "all"):
        cuda_exists = any(
            (results_base / name / "cuda" / "grouped_gemm.txt").exists()
            for _, name in config_list
        )
        if not cuda_exists and not (REPO_ROOT / "bin" / "tmain_grouped").exists():
            print("\n[Tip] CUDA results missing. Run 'make grouped' on this machine to build bin/tmain_grouped.", flush=True)

    if args.summary and summary_data:
        print("\n--- Summary ---")
        for config_name, config_str, rows in summary_data:
            print(f"\n[{config_name}]")
            if config_str:
                print(f"  {config_str[:80]}...")
            if rows:
                best = max(rows, key=lambda r: r[2])  # max by GFLOPS
                print(f"  Best: {best[0]} {best[2]:.1f} GFLOPS")

    if args.summary_csv and summary_data:
        csv_path = Path(args.summary_csv)
        if not csv_path.is_absolute():
            csv_path = REPO_ROOT / csv_path
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w") as f:
            f.write("Config,Backend,Time_ms,GFLOPS,Validation\n")
            for config_name, _, rows in summary_data:
                for name, ms, gflops, val in rows:
                    vstr = "PASS" if val is True else ("FAIL" if val is False else "N/A")
                    f.write(f"{config_name},{name},{ms:.4f},{gflops:.2f},{vstr}\n")
        print(f"\nSummary CSV: {csv_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
