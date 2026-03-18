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

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = REPO_ROOT / "experiments" / "compare_grouped"
BENCHMARK_DIR = REPO_ROOT / "results" / "benchmark" / "single_kernel"
CONFIGS_DIR = CONFIG_DIR / "configs"
# 기본 설정: configs/default.yaml (없으면 default.yml)
_DEFAULT_YAML = CONFIGS_DIR / "default.yaml"
_DEFAULT_YML = CONFIGS_DIR / "default.yml"
DEFAULT_CONFIG = _DEFAULT_YAML if _DEFAULT_YAML.exists() else _DEFAULT_YML

DEFAULT_SHAPES = ([1024] * 8, [4096] * 8, [14336] * 8)

# NVTX 구간 이름: Triton/Torch 벤치마크 루프에 range_push/pop으로 감싸져 있음.
# Nsight CLI: Push/Pop 구간은 이름 뒤에 '/'를 붙여야 함 (그렇지 않으면 Start/End로 해석되어 커널이 0개 수집됨).
NSIGHT_NVTX_RANGE = "grouped_gemm/"


def load_unified_config(config_path):
    """config 로드 → (m_list, n_list, k_list). config_path None이면 DEFAULT_CONFIG 사용."""
    sys.path.insert(0, str(CONFIG_DIR))
    try:
        from grouped_config import load_grouped_sizes
        return load_grouped_sizes(
            config_path=config_path,
            default_config=DEFAULT_CONFIG,
        )
    finally:
        if str(CONFIG_DIR) in sys.path:
            sys.path.remove(str(CONFIG_DIR))


def _run_cmd(cmd, capture=True, timeout=300):
    """subprocess 실행. capture 시 (stdout+stderr) 반환."""
    r = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=capture, text=True, timeout=timeout)
    return (r.stdout or "") + (r.stderr or ""), r.returncode


def _run_ncu(cmd, rep_path, opts, nvtx_include=None, timeout=600):
    """
    ncu 실행. opts: nsight_opts (out, extra, launch_count, launch_skip).
    nvtx_include: 이 NVTX 구간 안에서 런치된 커널만 수집 (Triton/Torch용, 초기화 제외).
    """
    rep_path = Path(rep_path).resolve()
    export_path = rep_path if str(rep_path).endswith(".ncu-rep") else Path(str(rep_path) + ".ncu-rep")
    ncu_cmd = [
        "ncu", "--export", str(export_path), "--force-overwrite",
        "--launch-count", str(opts.get("launch_count", 5)),
    ]
    if opts.get("launch_skip", 0) > 0:
        ncu_cmd.extend(["--launch-skip-before-match", str(opts["launch_skip"])])
    if nvtx_include:
        ncu_cmd.extend(["--nvtx", "--nvtx-push-pop-scope", "process", "--nvtx-include", nvtx_include])
    if opts.get("extra"):
        ncu_cmd.extend(opts["extra"].split())
    ncu_cmd.extend(["--"] + cmd)
    return subprocess.run(ncu_cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout)


def _run_backend_with_nsight(cmd, out_file, rep_path, label, opts, nvtx_include=None):
    """cmd를 ncu로 프로파일. nvtx_include 시 해당 NVTX 구간만 수집 (GEMM만)."""
    rep_path = Path(rep_path).resolve()
    expected_rep = rep_path if str(rep_path).endswith(".ncu-rep") else Path(str(rep_path) + ".ncu-rep")
    print(f"[Nsight] Profiling {label} -> {expected_rep}", flush=True)
    r = _run_ncu(cmd, rep_path, opts, nvtx_include=nvtx_include, timeout=600)
    out = (r.stdout or "") + (r.stderr or "")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        f.write(out)
    ok = r.returncode == 0
    if ok:
        if expected_rep.exists():
            print(f"[Nsight] {label} done -> {expected_rep}", flush=True)
        else:
            print(f"[Nsight] {label} ncu exited 0 but file missing: {expected_rep}", flush=True)
            print(f"          Check ncu output in {out_file}", flush=True)
    else:
        print(f"[Nsight] {label} FAILED (exit {r.returncode})", flush=True)
        # 에러 원인: ncu는 stdout/stderr 모두에 출력할 수 있음
        combined = ((r.stdout or "") + "\n" + (r.stderr or "")).strip()
        lines = [x for x in combined.split("\n") if x.strip()]
        if lines:
            tail = lines[-25:] if len(lines) > 25 else lines
            for line in tail:
                print(f"  | {line}", flush=True)
        else:
            print(f"  | (no output, see {out_file} for full capture)", flush=True)
    return ok


def run_cuda_single_kernel(config_path=None, results_base=None, nsight_opts=None):
    """CUDA: kernel 1 (cuBLAS Grouped), kernel 2 (Custom) — TF32, FP16."""
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
    lines = [
        "=== CUDA Grouped GEMM (Single Kernel Only) ===",
        f"Batch size: {len(m_list)}",
        f"Shapes: {list(zip(m_list, n_list, k_list))}",
        "",
    ]

    if nsight_opts and nsight_opts.get("enabled"):
        nsight_dir.mkdir(parents=True, exist_ok=True)
        # C++ 바이너리는 NVTX가 ncu에 매칭되지 않아 No kernels were profiled 나옴 → NVTX 미사용
        nvtx_cuda = None
        for label, exe_path, kernel_arg, suffix in [
                ("TF32 Grouped", exe, "1", "tf32_grouped"),
                ("TF32 Custom", exe, "2", "tf32_custom"),
                ("FP16 Grouped", exe_half if exe_half.exists() else None, "1", "fp16_grouped"),
        ]:
            if exe_path is None:
                continue
            rep_path = nsight_dir / f"grouped_gemm_{suffix}"
            expected_rep = Path(str(rep_path) + ".ncu-rep")
            cmd = [str(exe_path), kernel_arg, "p", cfg_arg]
            print(f"[Nsight] CUDA single-kernel {label} starting (this may take a while)...", flush=True)
            r = _run_ncu(cmd, rep_path, nsight_opts, nvtx_include=nvtx_cuda, timeout=600)
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

    # TF32 kernel 1 (cuBLAS Grouped)
    lines.append("--- Kernel 1 (cuBLAS Grouped API) ---")
    out, ret = _run_cmd([str(exe), "1", "p", cfg_arg], timeout=300)
    lines.append(out.rstrip())
    lines.append("")

    # TF32 kernel 2 (Custom Double Buffering)
    lines.append("--- Kernel 2 (Custom Double Buffering) ---")
    out, ret = _run_cmd([str(exe), "2", "p", cfg_arg], timeout=300)
    lines.append(out.rstrip())
    lines.append("")

    # FP16 kernel 1
    if exe_half.exists():
        lines.append("--- CUDA FP16 Grouped ---")
        out, ret = _run_cmd([str(exe_half), "1", "p", cfg_arg], timeout=300)
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
            # C++ 바이너리에서 NVTX가 ncu에 인식되지 않아 No kernels were profiled → NVTX 미사용, 처음 N개 커널 수집
            _run_backend_with_nsight(
                cmd, out_dir / "grouped_gemm.txt",
                nsight_dir / "cutlass_grouped_single", "Cutlass", nsight_opts, nvtx_include=None)
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
            _run_backend_with_nsight(
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
        print("[Torch] Config has varying K/N -> grouped_mm unavailable, uses loop (multiple kernels).", flush=True)
        print("        For single-kernel comparison, use uniform M,N,K or same K and N per batch.", flush=True)

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
            _run_backend_with_nsight(
                cmd, out_dir / "grouped_gemm.txt",
                nsight_dir / "torch_grouped_single", "Torch", nsight_opts, nvtx_include=nvtx)
        else:
            with open(out_dir / "grouped_gemm.txt", "w") as f:
                subprocess.run(cmd, cwd=REPO_ROOT, stdout=f, stderr=subprocess.STDOUT, timeout=300)
            label = "1 kernel (grouped_mm)" if (same_k and same_n) else "loop (fallback)"
            print(f"[Torch] done ({label})", flush=True)
        return out_dir / "grouped_gemm.txt"
    except Exception as e:
        print(f"Torch failed: {e}")
        return None


def format_unified_report(config_str, rows):
    """통일된 형식 리포트. rows: (name, ms, gflops, validation)."""
    lines = [
        "=== Grouped GEMM Single-Kernel Comparison ===",
        "",
        "Config: " + (config_str or "N/A"),
        "",
        f"{'Backend':<28} | {'Time (ms)':>10} | {'GFLOPS':>12} | {'Validation':>8}",
        "-" * 66,
    ]
    for name, ms, gflops, val in rows:
        vstr = "PASS" if val is True else ("FAIL" if val is False else "N/A")
        lines.append(f"{name:<28} | {ms:>10.2f} | {gflops:>12.1f} | {vstr:>8}")
    lines.append("")
    return "\n".join(lines)


def _check_validation(text):
    if "Result is correct" in text or "validation passed" in text:
        return True
    if "Result is different" in text:
        return False
    return None


def parse_one_result_file(backend, path):
    """한 개 결과 파일 파싱 → (config_str 또는 None, [(name, ms, gflops, val), ...])."""
    import re
    if path is None or not Path(path).exists():
        return None, []
    content = Path(path).read_text()
    config_str = None
    rows = []
    m = re.search(r"Batch size: (\d+)\nShapes: (.+)", content)
    if m:
        config_str = f"Batch size: {m.group(1)}, Shapes: {m.group(2).strip()}"

    if backend == "cuda":
        for name, pat in [
            ("CUDA cuBLAS Grouped (TF32)", r"--- Kernel 1 \(cuBLAS Grouped API\) ---(.*?)\[raw\]\s+[\d.]+,([\d.]+),[\d.]+,([\d.]+)"),
            ("CUDA Custom (TF32)", r"--- Kernel 2 \(Custom Double Buffering\) ---(.*?)\[raw\]\s+[\d.]+,([\d.]+),[\d.]+,([\d.]+)"),
            ("CUDA FP16 Grouped", r"--- CUDA FP16 Grouped ---(.*?)\[raw\]\s+[\d.]+,([\d.]+),[\d.]+,([\d.]+)"),
        ]:
            m = re.search(pat, content, re.DOTALL)
            if m:
                t = float(m.group(2))
                gflops = float(m.group(3))
                ms = t * 1000
                val = _check_validation(m.group(1))
                rows.append((name, ms, gflops, val))
    elif backend == "cutlass":
        val = _check_validation(content)
        m = re.search(r"Cutlass grouped GEMM:\s*([\d.]+)\s*ms,\s*([\d.]+)\s*GFLOPS", content)
        if m:
            rows.append(("Cutlass", float(m.group(1)), float(m.group(2)), val))
        else:
            m = re.search(r"\[raw\]\s+[\d.]+,([\d.]+),[\d.]+,([\d.]+)", content)
            if m:
                rows.append(("Cutlass", float(m.group(1)) * 1000, float(m.group(2)), val))
    elif backend == "triton":
        m = re.search(r"Triton grouped GEMM:\s*([\d.]+)\s*ms,\s*([\d.]+)\s*GFLOPS", content)
        if m:
            val = _check_validation(content)
            rows.append(("Triton", float(m.group(1)), float(m.group(2)), val))
    elif backend == "torch":
        m = re.search(r"Torch grouped GEMM:\s*([\d.]+)\s*ms,\s*([\d.]+)\s*GFLOPS", content)
        if m:
            val = _check_validation(content)
            rows.append(("Torch", float(m.group(1)), float(m.group(2)), val))
    return config_str, rows


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
    g.add_argument("--nsight-launch-skip", "--ncu-launch-skip", type=int, default=0, dest="nsight_launch_skip",
                    help="Skip this many kernel launches before profiling (e.g. 20 to skip init)")
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
        "launch_skip": args.nsight_launch_skip,
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
