#!/usr/bin/env python3
"""
Grouped GEMM 단일-커널 백엔드만 비교 (공정한 비교용).

각 백엔드가 전체 grouped GEMM을 1개의 커널 런치로 처리하는 경우만 실행:
  - CUDA: kernel 1 (cuBLAS Grouped), kernel 2 (Custom Double Buffering) — TF32
  - CUDA FP16: kernel 1 (cuBLAS Grouped)
  - Cutlass: GemmGrouped (1 kernel)
  - Triton: grouped_matmul_kernel (1 kernel)
  - Torch: grouped_mm (1 kernel) — 조건: BF16, 동일 K·N일 때만. 아니면 loop fallback

benchmark_end_to_end.py와 동일 config 사용. 결과는 results/benchmark/single_kernel/ 에 저장.
--nsight 시 .ncu-rep 생성.

공정성: 커널 런치 오버헤드를 제거해, 순수 계산 성능만 비교 가능.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime

from benchmark_utils import (
    REPO_ROOT,
    CONFIG_DIR,
    CONFIGS_DIR,
    DEFAULT_CONFIG,
    NSIGHT_NVTX_RANGE,
    load_unified_config,
    run_cmd,
    run_ncu,
    run_backend_with_nsight,
    nsight_launch_skip_tf32_grouped,
    nsight_launch_skip_tf32_custom,
    nsight_launch_skip_fp16_grouped,
    nsight_launch_skip_cutlass,
    format_unified_report,
    parse_one_result_file,
)

# This script's result directory
BENCHMARK_DIR = REPO_ROOT / "results" / "benchmark" / "single_kernel"


def run_cuda_single_kernel(config_path=None, results_base=None, nsight_opts=None):
    """CUDA: kernel 1 (cuBLAS Grouped), kernel 2 (Custom) — TF32, FP16. 통합 바이너리 사용."""
    exe = REPO_ROOT / "bin" / "tmain_grouped"
    exe_half = REPO_ROOT / "bin" / "tmain_grouped_half"
    if not exe.exists():
        print("Warning: tmain_grouped not found. Run 'make grouped' first.")
        return None

    config_resolved = (Path(config_path) if config_path else DEFAULT_CONFIG).resolve()
    m_list, n_list, k_list = load_unified_config(config_path)
    shapes = list(zip(m_list, n_list, k_list))
    print(f"Config: {config_resolved}, shapes: {shapes[:4]}{'...' if len(shapes) > 4 else ''}", flush=True)

    benchmark_base = results_base or BENCHMARK_DIR
    out_dir = benchmark_base / "cuda"
    nsight_dir = Path(nsight_opts["out"]) if nsight_opts and nsight_opts.get("enabled") else out_dir / "nsight"
    out_dir.mkdir(parents=True, exist_ok=True)

    cuda_config = out_dir / "grouped_config.txt"
    with open(cuda_config, "w") as f:
        for m, n, k in zip(m_list, n_list, k_list):
            f.write(f"{m} {n} {k}\n")

    cfg_arg = str(cuda_config.resolve())
    batch_size = len(m_list)
    lines = [
        "=== CUDA Grouped GEMM (Single Kernel Only) ===",
        f"Batch size: {len(m_list)}",
        f"Shapes: {list(zip(m_list, n_list, k_list))}",
        "",
    ]

    if nsight_opts and nsight_opts.get("enabled"):
        nsight_dir.mkdir(parents=True, exist_ok=True)
        runs = [
            ("TF32 Grouped", exe, ["1", "p", cfg_arg], "tf32_grouped", nsight_launch_skip_tf32_grouped(batch_size)),
            ("TF32 Custom", exe, ["2", "p", cfg_arg], "tf32_custom", nsight_launch_skip_tf32_custom(batch_size)),
        ]
        nvtx_cuda = None
        for label, exe_path, args, suffix, launch_skip in runs:
            rep_path = nsight_dir / f"grouped_gemm_{suffix}"
            expected_rep = rep_path if str(rep_path).endswith(".ncu-rep") else Path(str(rep_path) + ".ncu-rep")
            effective_skip = launch_skip if nsight_opts.get("launch_skip") is None else nsight_opts["launch_skip"]
            opts = {**nsight_opts, "launch_skip": effective_skip}
            cmd = [str(exe_path)] + args
            print(f"[Nsight] CUDA single-kernel {label} starting (launch_skip={opts['launch_skip']})...", flush=True)
            r = run_ncu(cmd, rep_path, opts, nvtx_include=nvtx_cuda, timeout=600)
            out = (r.stdout or "") + (r.stderr or "")
            lines.append(f"--- Nsight ({label}) ---")
            lines.append(out.rstrip())
            lines.append("")
            if r.returncode == 0:
                if expected_rep.exists():
                    print(f"[Nsight] CUDA single-kernel {label} done -> {expected_rep}", flush=True)
                else:
                    print(f"[Nsight] CUDA single-kernel {label} ncu exited 0 but file missing: {expected_rep}", flush=True)
                    print(f"          Check ncu output in {out_dir / 'grouped_gemm.txt'}", flush=True)
            else:
                print(f"[Nsight] CUDA single-kernel {label} FAILED (exit {r.returncode}) -> {expected_rep}", flush=True)
        if exe_half.exists():
            rep_path = nsight_dir / "grouped_gemm_fp16_grouped"
            expected_rep = Path(str(rep_path) + ".ncu-rep")
            effective_skip = nsight_launch_skip_fp16_grouped(batch_size) if nsight_opts.get("launch_skip") is None else nsight_opts["launch_skip"]
            opts = {**nsight_opts, "launch_skip": effective_skip}
            cmd = [str(exe_half), "1", "p", cfg_arg]
            print(f"[Nsight] CUDA single-kernel FP16 Grouped starting (launch_skip={opts['launch_skip']})...", flush=True)
            r = run_ncu(cmd, rep_path, opts, nvtx_include=nvtx_cuda, timeout=600)
            out = (r.stdout or "") + (r.stderr or "")
            lines.append("--- Nsight (FP16 Grouped) ---")
            lines.append(out.rstrip())
            lines.append("")
            if r.returncode == 0 and expected_rep.exists():
                print(f"[Nsight] CUDA single-kernel FP16 Grouped done -> {expected_rep}", flush=True)
            elif r.returncode != 0:
                print(f"[Nsight] CUDA single-kernel FP16 Grouped FAILED (exit {r.returncode})", flush=True)

    # TF32 kernel 1 (cuBLAS Grouped)
    lines.append("--- Kernel 1 (cuBLAS Grouped API) ---")
    out, ret = run_cmd([str(exe), "1", "p", cfg_arg], timeout=300)
    lines.append(out.rstrip())
    lines.append("")

    # TF32 kernel 2 (Custom Double Buffering)
    lines.append("--- Kernel 2 (Custom Double Buffering) ---")
    out, ret = run_cmd([str(exe), "2", "p", cfg_arg], timeout=300)
    lines.append(out.rstrip())
    lines.append("")

    # FP16 kernel 1 (Grouped)
    if exe_half.exists():
        lines.append("--- cuBLAS Grouped API (FP16) ---")
        out, ret = run_cmd([str(exe_half), "1", "p", cfg_arg], timeout=300)
        lines.append(out.rstrip())
        lines.append("")

    out_file = out_dir / "grouped_gemm.txt"
    with open(out_file, "w") as f:
        f.write("\n".join(lines))
    return out_file


def run_cutlass_single_kernel(config_path=None, results_base=None, nsight_opts=None):
    """Cutlass: 1 kernel (GemmGrouped)."""
    exe = REPO_ROOT / "bin" / "tmain_cutlass_grouped"
    if not exe.exists():
        print("Warning: Cutlass binary not found. Run 'make cutlass_grouped' first.")
        return None

    m_list, n_list, k_list = load_unified_config(config_path)
    benchmark_base = results_base or BENCHMARK_DIR
    out_dir = benchmark_base / "cutlass"
    nsight_dir = Path(nsight_opts["out"]) if nsight_opts and nsight_opts.get("enabled") else out_dir / "nsight"
    out_dir.mkdir(parents=True, exist_ok=True)

    config_file = out_dir / "grouped_config.txt"
    with open(config_file, "w") as f:
        for m, n, k in zip(m_list, n_list, k_list):
            f.write(f"{m} {n} {k}\n")

    cmd = [str(exe), str(config_file.resolve()), "p"]
    try:
        if nsight_opts and nsight_opts.get("enabled"):
            nsight_dir.mkdir(parents=True, exist_ok=True)
            batch_size = len(m_list)
            effective_skip = nsight_launch_skip_cutlass(batch_size) if nsight_opts.get("launch_skip") is None else nsight_opts["launch_skip"]
            cutlass_opts = {**nsight_opts, "launch_skip": effective_skip}
            run_backend_with_nsight(
                cmd, out_dir / "grouped_gemm.txt",
                nsight_dir / "cutlass_grouped_single", "Cutlass", cutlass_opts, nvtx_include=None)
        else:
            with open(out_dir / "grouped_gemm.txt", "w") as f:
                subprocess.run(cmd, cwd=REPO_ROOT, stdout=f, stderr=subprocess.STDOUT, timeout=300)
            print("[Cutlass] done (1 kernel)", flush=True)
        return out_dir / "grouped_gemm.txt"
    except Exception as e:
        print(f"Cutlass failed: {e}")
        return None


def run_triton_single_kernel(config_path=None, results_base=None, nsight_opts=None):
    """Triton: 1 kernel (grouped_matmul_kernel)."""
    script = REPO_ROOT / "implementations" / "triton" / "triton_grouped_gemm.py"
    if not script.exists():
        print("Warning: triton_grouped_gemm.py not found.")
        return None

    benchmark_base = results_base or BENCHMARK_DIR
    out_dir = benchmark_base / "triton"
    nsight_dir = Path(nsight_opts["out"]) if nsight_opts and nsight_opts.get("enabled") else out_dir / "nsight"
    out_dir.mkdir(parents=True, exist_ok=True)
    config = (Path(config_path) if config_path else DEFAULT_CONFIG).resolve()
    cmd = [sys.executable, str(script), "--benchmark-only", "--config", str(config),
           "--repeat", "1000", "--warmup", "50"]

    try:
        if nsight_opts and nsight_opts.get("enabled"):
            nsight_dir.mkdir(parents=True, exist_ok=True)
            nvtx = None if not nsight_opts.get("kernel_filter", True) else NSIGHT_NVTX_RANGE
            run_backend_with_nsight(
                cmd, out_dir / "grouped_gemm.txt",
                nsight_dir / "triton_grouped_single", "Triton", nsight_opts, nvtx_include=nvtx)
        else:
            with open(out_dir / "grouped_gemm.txt", "w") as f:
                subprocess.run(cmd, cwd=REPO_ROOT, stdout=f, stderr=subprocess.STDOUT, timeout=300)
            print("[Triton] done (1 kernel)", flush=True)
        return out_dir / "grouped_gemm.txt"
    except subprocess.TimeoutExpired:
        print("Triton timed out.")
        return None
    except Exception as e:
        print(f"Triton failed: {e}")
        return None


def run_torch_single_kernel(config_path=None, results_base=None, nsight_opts=None):
    """Torch: grouped_mm (1 kernel) — 조건: BF16, 동일 K·N. 아니면 loop fallback."""
    script = REPO_ROOT / "implementations" / "torch" / "torch_grouped_gemm.py"
    if not script.exists():
        print("Warning: torch_grouped_gemm.py not found.")
        return None

    m_list, n_list, k_list = load_unified_config(config_path)
    same_k = len(set(k_list)) == 1
    same_n = len(set(n_list)) == 1
    if not (same_k and same_n):
        print("[Torch] Config has varying K/N -> grouped_mm unavailable (loop fallback). Skipping Torch.", flush=True)
        print("        For single-kernel comparison, use uniform M,N,K or same K and N per batch.", flush=True)
        return None

    benchmark_base = results_base or BENCHMARK_DIR
    out_dir = benchmark_base / "torch"
    nsight_dir = Path(nsight_opts["out"]) if nsight_opts and nsight_opts.get("enabled") else out_dir / "nsight"
    out_dir.mkdir(parents=True, exist_ok=True)
    config = (Path(config_path) if config_path else DEFAULT_CONFIG).resolve()
    cmd = [sys.executable, str(script), "--config", str(config), "--repeat", "1000", "--warmup", "50"]

    try:
        if nsight_opts and nsight_opts.get("enabled"):
            nsight_dir.mkdir(parents=True, exist_ok=True)
            nvtx = None if not nsight_opts.get("kernel_filter", True) else NSIGHT_NVTX_RANGE
            run_backend_with_nsight(
                cmd, out_dir / "grouped_gemm.txt",
                nsight_dir / "torch_grouped_single", "Torch", nsight_opts, nvtx_include=nvtx)
        else:
            with open(out_dir / "grouped_gemm.txt", "w") as f:
                subprocess.run(cmd, cwd=REPO_ROOT, stdout=f, stderr=subprocess.STDOUT, timeout=300)
            print("[Torch] done (1 kernel, grouped_mm)", flush=True)
        return out_dir / "grouped_gemm.txt"
    except Exception as e:
        print(f"Torch failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Grouped GEMM single-kernel comparison (fair benchmark)")
    parser.add_argument("--config", type=Path, default=None, help="Config YAML path")
    parser.add_argument(
        "--config-name",
        type=str,
        default=None,
        help="Config name under experiments/compare_grouped/configs (e.g., 'default', 'expert_even')",
    )
    parser.add_argument("--results-subdir", type=str, default=None,
                        help="Save results to results/benchmark/single_kernel/<subdir>/ instead of results/benchmark/single_kernel/")
    parser.add_argument("--backend", choices=["cuda", "triton", "torch", "cutlass", "all"],
                        action="append", default=None,
                        help="Backend(s) to run; repeat for multiple (e.g. --backend triton --backend torch). Default: all")
    parser.add_argument("--no-compare", action="store_true", help="Skip unified report")
    # Nsight Compute (한 그룹으로 정리)
    g = parser.add_argument_group("Nsight Compute")
    g.add_argument("--nsight", action="store_true", help="Profile with Nsight (.ncu-rep)")
    g.add_argument("--nsight-out", type=Path, default=None, help="Output dir for .ncu-rep (default: results/nsight/<timestamp>)")
    g.add_argument("--nsight-all-kernels", action="store_true",
                   help="Triton/Torch: do not use NVTX range (collect all kernels; default: NVTX 'grouped_gemm' only)")
    g.add_argument("--nsight-launch-count", "--ncu-launch-count", type=int, default=20, dest="nsight_launch_count",
                    help="Number of kernel launches to profile (default: 20). Use 1 for single kernel. "
                         "If --nsight-all-kernels is set and this is too small, it may capture only init kernels.")
    g.add_argument("--nsight-launch-skip", "--ncu-launch-skip", type=int, default=None, dest="nsight_launch_skip",
                    help="Skip this many kernel launches before profiling. Default: per-backend (compute kernel only)")
    g.add_argument("--nsight-extra", "--ncu-extra", type=str, default=None, dest="nsight_extra",
                    help="Extra ncu options (e.g. '-c 1')")
    args = parser.parse_args()

    # Resolve config path: --config | --config-name | default = configs/default.yaml
    config_path = None
    if args.config:
        config_path = args.config
    elif args.config_name:
        cand_yaml = CONFIGS_DIR / f"{args.config_name}.yaml"
        cand_yml = CONFIGS_DIR / f"{args.config_name}.yml"
        if cand_yaml.exists():
            config_path = cand_yaml
        elif cand_yml.exists():
            config_path = cand_yml
        else:
            name_path = CONFIGS_DIR / args.config_name
            if name_path.exists() and name_path.suffix in {".yaml", ".yml"}:
                config_path = name_path
            else:
                available = sorted(p.stem for p in CONFIGS_DIR.glob("*.y*ml"))
                print(f"[Config] '{args.config_name}' not found under {CONFIGS_DIR}.", flush=True)
                if available:
                    print(f"[Config] Available: {', '.join(available)}", flush=True)
                return 1
    else:
        config_path = DEFAULT_CONFIG

    results_base = (BENCHMARK_DIR / args.results_subdir) if args.results_subdir else None

    nsight_out = args.nsight_out
    if args.nsight and not nsight_out:
        nsight_out = REPO_ROOT / "results" / "nsight" / datetime.now().strftime("%Y%m%d-%H%M%S")
    # --nsight-all-kernels 시 초기화 커널만 잡히지 않도록 수집 런치 수를 늘림 (GEMM까지 포함)
    launch_count = args.nsight_launch_count
    if args.nsight and args.nsight_all_kernels and launch_count <= 20:
        launch_count = 50
    nsight_opts = {
        "enabled": args.nsight,
        "out": nsight_out,
        "extra": args.nsight_extra,
        "launch_count": launch_count,
        "launch_skip": args.nsight_launch_skip,  # None => use per-backend default
        "kernel_filter": not args.nsight_all_kernels,
    } if args.nsight else None

    run_opts = {"results_base": results_base, "nsight_opts": nsight_opts}

    # --backend 여러 개 지원 (--backend triton --backend torch → 둘 다 실행)
    backends = args.backend if args.backend else ["all"]
    if "all" in backends:
        backends = ["cuda", "cutlass", "triton", "torch"]

    print("=== Single-Kernel Grouped GEMM Benchmark ===\n", flush=True)

    results = []
    if "cuda" in backends:
        results.append(("cuda", run_cuda_single_kernel(config_path, **run_opts)))
    if "cutlass" in backends:
        results.append(("cutlass", run_cutlass_single_kernel(config_path, **run_opts)))
    if "triton" in backends:
        results.append(("triton", run_triton_single_kernel(config_path, **run_opts)))
    if "torch" in backends:
        results.append(("torch", run_torch_single_kernel(config_path, **run_opts)))

    success = sum(1 for _, r in results if r is not None)
    benchmark_dir = results_base if results_base is not None else BENCHMARK_DIR
    print(f"\nCompleted: {success}/{len(results)} backends")
    print(f"Results: {benchmark_dir}", flush=True)

    if not args.no_compare and success > 0:
        config_str = None
        rows = []
        for backend, out_file in results:
            if out_file is None:
                continue
            c, r = parse_one_result_file(backend, out_file)
            if c and config_str is None:
                config_str = c
            rows.extend(r)
        if rows:
            print("\n" + format_unified_report(config_str, rows))
            sys.path.insert(0, str(CONFIG_DIR))
            try:
                from compare import write_results_csv
                csv_path = benchmark_dir / "results.csv"
                write_results_csv(rows, csv_path)
                print(f"CSV written to {csv_path}")
            except ImportError:
                pass
            finally:
                if str(CONFIG_DIR) in sys.path:
                    sys.path.remove(str(CONFIG_DIR))

    if args.nsight and success > 0 and nsight_opts and nsight_opts.get("out"):
        nsight_out = Path(nsight_opts["out"]).resolve()
        reps = sorted(nsight_out.glob("*.ncu-rep"))
        if reps:
            print(f"Nsight reports: {nsight_out}", flush=True)
            for f in reps:
                print(f"  {f.name}", flush=True)
        else:
            print(f"Nsight output dir (no .ncu-rep found): {nsight_out}", flush=True)

    return 0 if success > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
