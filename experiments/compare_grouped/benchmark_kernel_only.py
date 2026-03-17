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


def _run_cmd(cmd, capture=True, timeout=300):
    """subprocess 실행. capture 시 (stdout+stderr) 반환."""
    r = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=capture, text=True, timeout=timeout)
    return (r.stdout or "") + (r.stderr or ""), r.returncode


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


def _run_backend_with_nsight(cmd, out_file, rep_path, label, ncu_extra=None):
    """cmd를 ncu로 프로파일 후 결과를 out_file에 저장."""
    print(f"[Nsight] Profiling {label} -> {rep_path}.ncu-rep", flush=True)
    r = _wrap_ncu(cmd, rep_path, ncu_extra, timeout=600)
    out = (r.stdout or "") + (r.stderr or "")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        f.write(out)
    ok = r.returncode == 0
    if ok:
        print(f"[Nsight] {label} done", flush=True)
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


def run_cuda_single_kernel(config_path=None, nsight=False, nsight_out=None, ncu_extra=None):
    """CUDA: kernel 1 (cuBLAS Grouped), kernel 2 (Custom) — TF32, FP16."""
    exe = REPO_ROOT / "bin" / "tmain_grouped"
    exe_half = REPO_ROOT / "bin" / "tmain_grouped_half"
    if not exe.exists():
        print("Warning: tmain_grouped not found. Run 'make grouped' first.")
        return None

    config_resolved = (Path(config_path) if config_path else CONFIG_DIR / "config.yaml").resolve()
    m_list, n_list, k_list = load_unified_config(config_path)
    shapes = list(zip(m_list, n_list, k_list))
    print(f"Config: {config_resolved}, shapes: {shapes[:4]}{'...' if len(shapes) > 4 else ''}", flush=True)

    out_dir = BENCHMARK_DIR / "cuda"
    nsight_dir = (Path(nsight_out) / "cuda") if nsight_out else out_dir / "nsight"
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

    if nsight:
        nsight_dir.mkdir(parents=True, exist_ok=True)
        # Nsight profiling: kernel 1 (Grouped), kernel 2 (Custom) each → .ncu-rep
        for label, exe_path, kernel_arg, suffix in [
            ("TF32 Grouped", exe, "1", "tf32_grouped"),
            ("TF32 Custom", exe, "2", "tf32_custom"),
            ("FP16 Grouped", exe_half if exe_half.exists() else None, "1", "fp16_grouped"),
        ]:
            if exe_path is None:
                continue
            rep_path = nsight_dir / f"grouped_gemm_{suffix}"
            cmd = [str(exe_path), kernel_arg, "p", cfg_arg]
            print(f"[Nsight] CUDA single-kernel {label} starting (this may take a while)...", flush=True)
            r = _wrap_ncu(cmd, rep_path, ncu_extra, timeout=600)
            out = (r.stdout or "") + (r.stderr or "")
            lines.append(f"--- Nsight ({label}) ---")
            lines.append(out.rstrip())
            lines.append("")
            if r.returncode == 0:
                print(f"[Nsight] CUDA single-kernel {label} done -> {rep_path}.ncu-rep", flush=True)
            else:
                print(f"[Nsight] CUDA single-kernel {label} FAILED (exit {r.returncode}) -> {rep_path}.ncu-rep", flush=True)

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


def run_cutlass_single_kernel(config_path=None, nsight=False, nsight_out=None, ncu_extra=None):
    """Cutlass: 1 kernel (GemmGrouped)."""
    exe = REPO_ROOT / "bin" / "tmain_cutlass_grouped"
    if not exe.exists():
        print("Warning: Cutlass binary not found. Run 'make cutlass_grouped' first.")
        return None

    m_list, n_list, k_list = load_unified_config(config_path)
    out_dir = BENCHMARK_DIR / "cutlass"
    nsight_dir = (Path(nsight_out) / "cutlass") if nsight_out else out_dir / "nsight"
    out_dir.mkdir(parents=True, exist_ok=True)

    config_file = out_dir / "grouped_config.txt"
    with open(config_file, "w") as f:
        for m, n, k in zip(m_list, n_list, k_list):
            f.write(f"{m} {n} {k}\n")

    cmd = [str(exe), str(config_file.resolve()), "p"]
    try:
        if nsight:
            nsight_dir.mkdir(parents=True, exist_ok=True)
            _run_backend_with_nsight(
                cmd, out_dir / "grouped_gemm.txt",
                nsight_dir / "cutlass_grouped_single", "Cutlass", ncu_extra)
        else:
            with open(out_dir / "grouped_gemm.txt", "w") as f:
                subprocess.run(cmd, cwd=REPO_ROOT, stdout=f, stderr=subprocess.STDOUT, timeout=300)
            print("[Cutlass] done (1 kernel)", flush=True)
        return out_dir / "grouped_gemm.txt"
    except Exception as e:
        print(f"Cutlass failed: {e}")
        return None


def run_triton_single_kernel(config_path=None, nsight=False, nsight_out=None, ncu_extra=None):
    """Triton: 1 kernel (grouped_matmul_kernel)."""
    script = REPO_ROOT / "implementations" / "triton" / "triton_grouped_gemm.py"
    if not script.exists():
        print("Warning: triton_grouped_gemm.py not found.")
        return None

    out_dir = BENCHMARK_DIR / "triton"
    nsight_dir = (Path(nsight_out) / "triton") if nsight_out else out_dir / "nsight"
    out_dir.mkdir(parents=True, exist_ok=True)
    config = (Path(config_path) if config_path else CONFIG_DIR / "config.yaml").resolve()
    cmd = [sys.executable, str(script), "--benchmark-only", "--config", str(config),
           "--repeat", "1000", "--warmup", "50"]

    try:
        if nsight:
            nsight_dir.mkdir(parents=True, exist_ok=True)
            _run_backend_with_nsight(
                cmd, out_dir / "grouped_gemm.txt",
                nsight_dir / "triton_grouped_single", "Triton", ncu_extra)
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


def run_torch_single_kernel(config_path=None, nsight=False, nsight_out=None, ncu_extra=None):
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

    out_dir = BENCHMARK_DIR / "torch"
    nsight_dir = (Path(nsight_out) / "torch") if nsight_out else out_dir / "nsight"
    out_dir.mkdir(parents=True, exist_ok=True)
    config = (Path(config_path) if config_path else CONFIG_DIR / "config.yaml").resolve()
    cmd = [sys.executable, str(script), "--config", str(config), "--repeat", "1000", "--warmup", "50"]

    try:
        if nsight:
            nsight_dir.mkdir(parents=True, exist_ok=True)
            _run_backend_with_nsight(
                cmd, out_dir / "grouped_gemm.txt",
                nsight_dir / "torch_grouped_single", "Torch", ncu_extra)
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


def load_and_parse_single_kernel():
    """single_kernel 디렉터리에서 결과 파싱. compare.py와 동일 로직."""
    import re
    config_str = None
    rows = []

    def _check_validation(text):
        if "Result is correct" in text or "validation passed" in text:
            return True
        if "Result is different" in text:
            return False
        return None

    dir_path = BENCHMARK_DIR

    # CUDA: kernel 1 (TF32), FP16 grouped
    cuda_path = dir_path / "cuda" / "grouped_gemm.txt"
    if cuda_path.exists():
        content = cuda_path.read_text()
        if "Batch size:" in content:
            m = re.search(r"Batch size: (\d+)\nShapes: (.+)", content)
            if m:
                config_str = f"Batch size: {m.group(1)}, Shapes: {m.group(2).strip()}"
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

    # Cutlass
    cut_path = dir_path / "cutlass" / "grouped_gemm.txt"
    if cut_path.exists():
        content = cut_path.read_text()
        val = _check_validation(content)
        m = re.search(r"Cutlass grouped GEMM:\s*([\d.]+)\s*ms,\s*([\d.]+)\s*GFLOPS", content)
        if m:
            rows.append(("Cutlass", float(m.group(1)), float(m.group(2)), val))
        else:
            m = re.search(r"\[raw\]\s+[\d.]+,([\d.]+),[\d.]+,([\d.]+)", content)
            if m:
                rows.append(("Cutlass", float(m.group(1)) * 1000, float(m.group(2)), val))

    # Triton
    tri_path = dir_path / "triton" / "grouped_gemm.txt"
    if tri_path.exists():
        content = tri_path.read_text()
        m = re.search(r"Triton grouped GEMM:\s*([\d.]+)\s*ms,\s*([\d.]+)\s*GFLOPS", content)
        if m:
            val = _check_validation(content)
            rows.append(("Triton", float(m.group(1)), float(m.group(2)), val))

    # Torch
    torch_path = dir_path / "torch" / "grouped_gemm.txt"
    if torch_path.exists():
        content = torch_path.read_text()
        m = re.search(r"Torch grouped GEMM:\s*([\d.]+)\s*ms,\s*([\d.]+)\s*GFLOPS", content)
        if m:
            val = _check_validation(content)
            rows.append(("Torch", float(m.group(1)), float(m.group(2)), val))

    return config_str, rows


def main():
    parser = argparse.ArgumentParser(
        description="Grouped GEMM single-kernel comparison (fair benchmark)")
    parser.add_argument("--config", type=Path, default=None, help="Config YAML path")
    parser.add_argument("--backend", choices=["cuda", "triton", "torch", "cutlass", "all"],
                        default="all", help="Backend to run")
    parser.add_argument("--no-compare", action="store_true", help="Skip unified report")
    parser.add_argument("--nsight", action="store_true",
                        help="Profile with Nsight Compute (.ncu-rep)")
    parser.add_argument("--nsight-out", type=Path, default=None, help="Dir for .ncu-rep files")
    parser.add_argument("--ncu-extra", type=str, default=None, help="Extra ncu options")
    args = parser.parse_args()

    config_path = args.config or (CONFIG_DIR / "config.yaml" if (CONFIG_DIR / "config.yaml").exists() else None)

    # Nsight 결과 기본 위치: results/nsight/<timestamp>/ (백엔드별 서브폴더는 각 함수에서 생성)
    if args.nsight and not args.nsight_out:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        default_nsight_out = REPO_ROOT / "results" / "nsight" / ts
    else:
        default_nsight_out = args.nsight_out

    nsight_opts = dict(nsight=args.nsight, nsight_out=default_nsight_out, ncu_extra=args.ncu_extra)

    print("=== Single-Kernel Grouped GEMM Benchmark ===\n", flush=True)

    results = []
    if args.backend in ("cuda", "all"):
        results.append(("cuda", run_cuda_single_kernel(config_path, **nsight_opts)))
    if args.backend in ("cutlass", "all"):
        results.append(("cutlass", run_cutlass_single_kernel(config_path, **nsight_opts)))
    if args.backend in ("triton", "all"):
        results.append(("triton", run_triton_single_kernel(config_path, **nsight_opts)))
    if args.backend in ("torch", "all"):
        results.append(("torch", run_torch_single_kernel(config_path, **nsight_opts)))

    success = sum(1 for _, r in results if r is not None)
    print(f"\nCompleted: {success}/{len(results)} backends")
    print(f"Results: {BENCHMARK_DIR}", flush=True)

    if not args.no_compare and success > 0:
        config_str, rows = load_and_parse_single_kernel()
        if rows:
            print("\n" + format_unified_report(config_str, rows))
            sys.path.insert(0, str(CONFIG_DIR))
            try:
                from compare import write_results_csv
                csv_path = BENCHMARK_DIR / "results.csv"
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
