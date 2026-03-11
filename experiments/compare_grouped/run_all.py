#!/usr/bin/env python3
"""
Grouped GEMM 백엔드별 벤치마크 실행.

CUDA, Cutlass, PyTorch, Triton 백엔드가 config.yaml로 동일 M,N,K 벤치마크.
--nsight 시 Nsight Compute로 프로파일링 (.ncu-rep).
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = REPO_ROOT / "experiments" / "compare_grouped"

KERNEL_NAMES = {0: "cuBLAS Loop", 1: "cuBLAS Grouped API", 2: "Custom Double Buffering"}
FP16_KERNEL_NAMES = {0: "CUDA FP16 Loop", 1: "CUDA FP16 Grouped"}
DEFAULT_SHAPES = ([1024] * 8, [4096] * 8, [14336] * 8)


def load_unified_config(config_path):
    """config 로드 → (m_list, n_list, k_list)"""
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


def _wrap_ncu(cmd, rep_path, ncu_extra=None, timeout=600):
    """
    -o는 모호하므로 --export를 사용합니다. 
    이미 파일이 있을 경우 덮어쓰려면 --force-overwrite(-f)를 추가하는 것이 안전합니다.
    """
    # -o 대신 --export 사용 (또는 --output-file)
    ncu_cmd = ["ncu", "--export", str(rep_path), "--force-overwrite", "--launch-count", "5"]
    
    if ncu_extra:
        ncu_cmd.extend(ncu_extra.split())
    ncu_cmd.extend(["--"] + cmd)
    
    return subprocess.run(ncu_cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout)


def _run_cmd(cmd, capture=True, timeout=120):
    """subprocess 실행. capture 시 (stdout+stderr) 반환."""
    r = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=capture, text=True, timeout=timeout)
    return (r.stdout or "") + (r.stderr or ""), r.returncode


def _warn_if_config_fallback(out, label=""):
    if "Could not load config" in out or "using defaults (1024,4096,14336" in out:
        print(f"[{label}] WARNING: Config file open failed, binary used defaults", flush=True)


def _run_backend_with_nsight(cmd, out_file, rep_path, label, ncu_extra=None):
    """cmd를 ncu로 프로파일 후 결과를 out_file에 저장."""
    print(f"[Nsight] Profiling {label} -> {rep_path}.ncu-rep", flush=True)
    r = _wrap_ncu(cmd, rep_path, ncu_extra, timeout=600)
    out = (r.stdout or "") + (r.stderr or "")
    with open(out_file, "w") as f:
        f.write(out)
    ok = r.returncode == 0
    if ok:
        print(f"[Nsight] {label} done", flush=True)
    else:
        print(f"[Nsight] {label} FAILED (exit {r.returncode})", flush=True)
        combined = ((r.stdout or "") + "\n" + (r.stderr or "")).strip()
        lines = [x for x in combined.split("\n") if x.strip()]
        if lines:
            for line in (lines[-25:] if len(lines) > 25 else lines):
                print(f"  | {line}", flush=True)
        else:
            print(f"  | (see {out_file} for full output)", flush=True)
    return ok


def run_cuda_grouped(config_path=None, nsight=False, nsight_out=None, ncu_extra=None):
    exe = REPO_ROOT / "bin" / "tmain_grouped"
    exe_half = REPO_ROOT / "bin" / "tmain_grouped_half"
    if not exe.exists():
        print("Warning: tmain_grouped not found. Run 'make grouped' first.")
        return None

    config_resolved = (Path(config_path) if config_path else CONFIG_DIR / "config.yaml").resolve()
    print(f"Config: {config_resolved}", flush=True)
    m_list, n_list, k_list = load_unified_config(config_path)
    shapes = list(zip(m_list, n_list, k_list))
    print(f"Loaded shapes: {shapes[:4]}{'...' if len(shapes) > 4 else ''}", flush=True)
    if (m_list, n_list, k_list) == DEFAULT_SHAPES:
        print("WARNING: Loaded defaults - config not found or PyYAML missing", flush=True)

    out_dir = REPO_ROOT / "results" / "benchmark" / "cuda"
    nsight_dir = Path(nsight_out) if nsight_out else out_dir / "nsight"
    out_dir.mkdir(parents=True, exist_ok=True)
    if nsight:
        nsight_dir.mkdir(parents=True, exist_ok=True)

    cuda_config = out_dir / "grouped_config.txt"
    with open(cuda_config, "w") as f:
        for m, n, k in zip(m_list, n_list, k_list):
            f.write(f"{m} {n} {k}\n")

    out_file = out_dir / "grouped_gemm.txt"
    lines = [
        "=== CUDA Grouped GEMM Benchmark ===",
        f"Batch size: {len(m_list)}",
        f"Shapes: {list(zip(m_list, n_list, k_list))}",
        "",
    ]
    cfg_arg = str(cuda_config.resolve())
    timeout = 600 if nsight else 120

    # Nsight: kernel 3 (TF32 all), kernel 2 (FP16 all)
    if nsight:
        for label, exe_path, kernel_arg in [
            ("TF32", exe, "3"),
            ("FP16", exe_half if exe_half.exists() else None, "2"),
        ]:
            if exe_path is None:
                continue
            rep_path = nsight_dir / f"grouped_gemm_{label.lower()}_all"
            cmd = [str(exe_path), kernel_arg, "p", cfg_arg]
            r = _wrap_ncu(cmd, rep_path, ncu_extra, timeout=600)
            out = (r.stdout or "") + (r.stderr or "")
            lines.append(f"--- Nsight (all {label}) ---")
            lines.append(out.rstrip())
            lines.append("")
            print(f"[Nsight] {label} {'done' if r.returncode == 0 else f'FAILED ({r.returncode})'}", flush=True)

    # TF32 kernels 0,1,2 (벤치마크)
    for k in (0, 1, 2):
        lines.append(f"--- Kernel {k} ({KERNEL_NAMES[k]}) ---")
        out, ret = _run_cmd([str(exe), str(k), "p", cfg_arg], timeout=timeout)
        lines.append(out.rstrip())
        _warn_if_config_fallback(out, "CUDA")
        status = "FAILED" if ret != 0 else "done"
        print(f"[CUDA] Kernel {k} ({KERNEL_NAMES[k]}) {status}", flush=True)
        lines.append("")

    # FP16
    if exe_half.exists():
        for k in (0, 1):
            lines.append(f"--- {FP16_KERNEL_NAMES[k]} ---")
            out, ret = _run_cmd([str(exe_half), str(k), "p", cfg_arg], timeout=timeout)
            lines.append(out.rstrip())
            _warn_if_config_fallback(out, "CUDA FP16")
            print(f"[CUDA FP16] {FP16_KERNEL_NAMES[k]} {'done' if ret == 0 else 'FAILED'}", flush=True)
            lines.append("")

    with open(out_file, "w") as f:
        f.write("\n".join(lines))
    if nsight:
        print(f"[CUDA] Nsight reports: {nsight_dir}/*.ncu-rep", flush=True)
    return out_file


def run_triton_grouped(config_path=None, nsight=False, nsight_out=None, ncu_extra=None):
    script = REPO_ROOT / "implementations" / "triton" / "triton_grouped_gemm.py"
    if not script.exists():
        print("Warning: triton_grouped_gemm.py not found.")
        return None

    out_dir = REPO_ROOT / "results" / "benchmark" / "triton"
    nsight_dir = Path(nsight_out) if nsight_out else out_dir / "nsight"
    out_dir.mkdir(parents=True, exist_ok=True)
    config = (Path(config_path) if config_path else CONFIG_DIR / "config.yaml").resolve()
    cmd = [sys.executable, str(script), "--benchmark-only", "--config", str(config),
           "--repeat", "1000", "--warmup", "50"]

    try:
        if nsight:
            nsight_dir.mkdir(parents=True, exist_ok=True)
            _run_backend_with_nsight(cmd, out_dir / "grouped_gemm.txt",
                                     nsight_dir / "triton_grouped", "Triton", ncu_extra)
        else:
            with open(out_dir / "grouped_gemm.txt", "w") as f:
                subprocess.run(cmd, cwd=REPO_ROOT, stdout=f, stderr=subprocess.STDOUT, timeout=300)
            print("[Triton] done", flush=True)
        return out_dir / "grouped_gemm.txt"
    except subprocess.TimeoutExpired:
        print("Triton timed out.")
        return None
    except Exception as e:
        print(f"Triton failed: {e}")
        return None


def run_torch_grouped(config_path=None, nsight=False, nsight_out=None, ncu_extra=None):
    script = REPO_ROOT / "implementations" / "torch" / "torch_grouped_gemm.py"
    if not script.exists():
        print("Warning: torch_grouped_gemm.py not found.")
        return None

    out_dir = REPO_ROOT / "results" / "benchmark" / "torch"
    nsight_dir = Path(nsight_out) if nsight_out else out_dir / "nsight"
    out_dir.mkdir(parents=True, exist_ok=True)
    config = (Path(config_path) if config_path else CONFIG_DIR / "config.yaml").resolve()
    cmd = [sys.executable, str(script), "--config", str(config), "--repeat", "1000", "--warmup", "50"]

    try:
        if nsight:
            nsight_dir.mkdir(parents=True, exist_ok=True)
            _run_backend_with_nsight(cmd, out_dir / "grouped_gemm.txt",
                                     nsight_dir / "torch_grouped", "Torch", ncu_extra)
        else:
            with open(out_dir / "grouped_gemm.txt", "w") as f:
                subprocess.run(cmd, cwd=REPO_ROOT, stdout=f, stderr=subprocess.STDOUT, timeout=120)
            print("[Torch] done", flush=True)
        return out_dir / "grouped_gemm.txt"
    except Exception as e:
        print(f"Torch failed: {e}")
        return None


def run_cutlass_grouped(config_path=None, nsight=False, nsight_out=None, ncu_extra=None):
    exe = REPO_ROOT / "bin" / "tmain_cutlass_grouped"
    if not exe.exists():
        print("Warning: Cutlass binary not found. Run 'make cutlass_grouped' first.")
        return None

    m_list, n_list, k_list = load_unified_config(config_path)
    out_dir = REPO_ROOT / "results" / "benchmark" / "cutlass"
    nsight_dir = Path(nsight_out) if nsight_out else out_dir / "nsight"
    out_dir.mkdir(parents=True, exist_ok=True)

    config_file = out_dir / "grouped_config.txt"
    with open(config_file, "w") as f:
        for m, n, k in zip(m_list, n_list, k_list):
            f.write(f"{m} {n} {k}\n")

    cmd = [str(exe), str(config_file.resolve()), "p"]
    try:
        if nsight:
            nsight_dir.mkdir(parents=True, exist_ok=True)
            _run_backend_with_nsight(cmd, out_dir / "grouped_gemm.txt",
                                     nsight_dir / "cutlass_grouped", "Cutlass", ncu_extra)
        else:
            with open(out_dir / "grouped_gemm.txt", "w") as f:
                subprocess.run(cmd, cwd=REPO_ROOT, stdout=f, stderr=subprocess.STDOUT, timeout=120)
            print("[Cutlass] done", flush=True)
        return out_dir / "grouped_gemm.txt"
    except Exception as e:
        print(f"Cutlass failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Grouped GEMM benchmark (all backends)")
    parser.add_argument("--config", type=Path, default=None, help="Config YAML path")
    parser.add_argument("--backend", choices=["cuda", "triton", "torch", "cutlass", "all"],
                        default="all", help="Backend to run")
    parser.add_argument("--no-compare", action="store_true", help="Skip unified report")
    parser.add_argument("--nsight", action="store_true", help="Profile with Nsight Compute")
    parser.add_argument("--nsight-out", type=Path, default=None, help="Dir for .ncu-rep files")
    parser.add_argument("--ncu-extra", type=str, default=None, help="Extra ncu options")
    args = parser.parse_args()

    config_path = args.config or (CONFIG_DIR / "config.yaml" if (CONFIG_DIR / "config.yaml").exists() else None)
    nsight_opts = dict(nsight=args.nsight, nsight_out=args.nsight_out, ncu_extra=args.ncu_extra)

    results = []
    if args.backend in ("cuda", "all"):
        results.append(("cuda", run_cuda_grouped(config_path, **nsight_opts)))
    if args.backend in ("triton", "all"):
        results.append(("triton", run_triton_grouped(config_path, **nsight_opts)))
    if args.backend in ("torch", "all"):
        results.append(("torch", run_torch_grouped(config_path, **nsight_opts)))
    if args.backend in ("cutlass", "all"):
        results.append(("cutlass", run_cutlass_grouped(config_path, **nsight_opts)))

    success = sum(1 for _, r in results if r is not None)
    print(f"\nCompleted: {success}/{len(results)} backends")

    if not args.no_compare and success > 0:
        sys.path.insert(0, str(CONFIG_DIR))
        try:
            from compare import load_and_parse, format_unified_report, write_results_csv
            config_str, rows = load_and_parse()
            print("\n" + format_unified_report(config_str, rows))
            if rows:
                csv_path = REPO_ROOT / "results" / "benchmark" / "benchmark_results.csv"
                write_results_csv(rows, csv_path)
                print(f"CSV written to {csv_path}")
        except ImportError:
            pass
        finally:
            if str(CONFIG_DIR) in sys.path:
                sys.path.remove(str(CONFIG_DIR))

    return 0 if success > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
