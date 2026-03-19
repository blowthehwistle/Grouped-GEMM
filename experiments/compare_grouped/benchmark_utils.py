"""
Shared helpers for benchmark/experiment scripts (e.g. benchmark_kernel_only.py, benchmark_end_to_end.py).

- Paths, config loading, Nsight launch-skip formulas
- Subprocess: run_cmd, run_ncu, run_backend_with_nsight
- Report: format_unified_report, check_validation, parse_one_result_file
"""

import sys
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple, Any

# Paths (experiments/compare_grouped 기준)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = REPO_ROOT / "experiments" / "compare_grouped"
CONFIGS_DIR = CONFIG_DIR / "configs"
_DEFAULT_YAML = CONFIGS_DIR / "default.yaml"
_DEFAULT_YML = CONFIGS_DIR / "default.yml"
DEFAULT_CONFIG = _DEFAULT_YAML if _DEFAULT_YAML.exists() else _DEFAULT_YML

# NVTX range name for Nsight (Triton/Torch). Push/Pop 구간은 이름 뒤 '/' 필요.
NSIGHT_NVTX_RANGE = "grouped_gemm/"

# Warmup count: implementations use 50.
WARMUP_COUNT = 50


# ---------- Nsight launch-skip (batch_size-dependent) ----------
# grouped_gemm.cu 실행 순서: convert_to_tf32(2) -> warmup(50 * runCublasTF32) -> baseline(1000 * runCublasTF32) -> kernel
# runCublasTF32_with_TC = loop over batch, cublasGemmEx per batch = batch_size launches/iter.
# cuBLAS은 내부적으로 CUTLASS 커널 사용 → skip 부족 시 CUTLASS가 리포트에 섞임.

REPEAT_TIME = 1000  # grouped_gemm.cu repeat_time


def nsight_launch_skip_tf32_loop(batch_size: int) -> int:
    """convert 2 + warmup*B + baseline*B (baseline = loop)."""
    return 2 + WARMUP_COUNT * batch_size + REPEAT_TIME * batch_size


def nsight_launch_skip_tf32_grouped(batch_size: int) -> int:
    """convert 2 + warmup*B + baseline*B (baseline = loop, 같은 순서)."""
    return 2 + WARMUP_COUNT * batch_size + REPEAT_TIME * batch_size


def nsight_launch_skip_tf32_custom(batch_size: int) -> int:
    """convert 2 + warmup*B + baseline*B. Custom도 warmup/baseline에 runCublasTF32(loop=B) 사용."""
    return 2 + WARMUP_COUNT * batch_size + REPEAT_TIME * batch_size


def nsight_launch_skip_fp16_grouped(batch_size: int) -> int:
    """No convert; warmup*B + baseline*B (runCublasTF16_Loop = B launches/iter)."""
    return WARMUP_COUNT * batch_size + REPEAT_TIME * batch_size


def nsight_launch_skip_cutlass(_batch_size: int) -> int:
    """Warmup only; 1 kernel per gemm.run()."""
    return WARMUP_COUNT


# ---------- Config ----------

def load_unified_config(
    config_path: Optional[Path] = None,
    default_config: Optional[Path] = None,
    config_dir: Optional[Path] = None,
) -> Tuple[List[int], List[int], List[int]]:
    """Config 로드 → (m_list, n_list, k_list). config_path None이면 default_config 사용."""
    config_dir = config_dir or CONFIG_DIR
    default_config = default_config or DEFAULT_CONFIG
    sys.path.insert(0, str(config_dir))
    try:
        from grouped_config import load_grouped_sizes
        return load_grouped_sizes(
            config_path=config_path,
            default_config=default_config,
        )
    finally:
        if str(config_dir) in sys.path:
            sys.path.remove(str(config_dir))


# ---------- Subprocess / Nsight ----------

def run_cmd(cmd: List[str], capture: bool = True, timeout: int = 300, cwd: Optional[Path] = None) -> Tuple[str, int]:
    """Subprocess 실행. capture 시 (stdout+stderr) 반환, returncode."""
    r = subprocess.run(
        cmd,
        cwd=cwd or REPO_ROOT,
        capture_output=capture,
        text=True,
        timeout=timeout,
    )
    return (r.stdout or "") + (r.stderr or ""), r.returncode


def run_ncu(
    cmd: List[str],
    rep_path: Path,
    opts: dict,
    nvtx_include: Optional[str] = None,
    timeout: int = 600,
    cwd: Optional[Path] = None,
) -> subprocess.CompletedProcess:
    """
    ncu 실행. opts: out, extra, launch_count, launch_skip.
    nvtx_include: 이 NVTX 구간 안에서 런치된 커널만 수집 (Triton/Torch용).
    """
    rep_path = Path(rep_path).resolve()
    export_path = rep_path if str(rep_path).endswith(".ncu-rep") else Path(str(rep_path) + ".ncu-rep")
    ncu_cmd = [
        "ncu", "--export", str(export_path), "--force-overwrite",
        "--launch-count", str(opts.get("launch_count", 5)),
    ]
    skip = opts.get("launch_skip")
    if (skip or 0) > 0:
        ncu_cmd.extend(["--launch-skip-before-match", str(skip)])
    if nvtx_include:
        ncu_cmd.extend(["--nvtx", "--nvtx-push-pop-scope", "process", "--nvtx-include", nvtx_include])
    if opts.get("extra"):
        ncu_cmd.extend(opts["extra"].split())
    ncu_cmd.extend(["--"] + cmd)
    return subprocess.run(
        ncu_cmd,
        cwd=cwd or REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def run_backend_with_nsight(
    cmd: List[str],
    out_file: Path,
    rep_path: Path,
    label: str,
    opts: dict,
    nvtx_include: Optional[str] = None,
    cwd: Optional[Path] = None,
) -> bool:
    """cmd를 ncu로 프로파일. nvtx_include 시 해당 NVTX 구간만 수집. 성공 여부 반환."""
    rep_path = Path(rep_path).resolve()
    expected_rep = rep_path if str(rep_path).endswith(".ncu-rep") else Path(str(rep_path) + ".ncu-rep")
    print(f"[Nsight] Profiling {label} -> {expected_rep}", flush=True)
    r = run_ncu(cmd, rep_path, opts, nvtx_include=nvtx_include, timeout=600, cwd=cwd)
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
        combined = ((r.stdout or "") + "\n" + (r.stderr or "")).strip()
        lines = [x for x in combined.split("\n") if x.strip()]
        if lines:
            tail = lines[-25:] if len(lines) > 25 else lines
            for line in tail:
                print(f"  | {line}", flush=True)
        else:
            print(f"  | (no output, see {out_file} for full capture)", flush=True)
    return ok


# ---------- Report parsing / formatting ----------

def check_validation(text: str) -> Optional[bool]:
    """Result text에서 validation 통과/실패 여부. True/False/None(알 수 없음)."""
    if "Result is correct" in text or "validation passed" in text:
        return True
    if "Result is different" in text:
        return False
    return None


def format_unified_report(config_str: Optional[str], rows: List[Tuple[str, float, float, Any]]) -> str:
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


def parse_one_result_file(backend: str, path: Optional[Path]) -> Tuple[Optional[str], List[Tuple[str, float, float, Any]]]:
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
            ("cuBLAS Grouped API (FP16)", r"--- (?:CUDA FP16 Grouped|cuBLAS Grouped API \(FP16\)) ---(.*?)\[raw\]\s+[\d.]+,([\d.]+),[\d.]+,([\d.]+)"),
        ]:
            m = re.search(pat, content, re.DOTALL)
            if m:
                t = float(m.group(2))
                gflops = float(m.group(3))
                ms = t * 1000
                val = check_validation(m.group(1))
                rows.append((name, ms, gflops, val))
    elif backend == "cutlass":
        val = check_validation(content)
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
            val = check_validation(content)
            rows.append(("Triton", float(m.group(1)), float(m.group(2)), val))
    elif backend == "torch":
        m = re.search(r"Torch grouped GEMM:\s*([\d.]+)\s*ms,\s*([\d.]+)\s*GFLOPS", content)
        if m:
            val = check_validation(content)
            rows.append(("Torch", float(m.group(1)), float(m.group(2)), val))
    return config_str, rows
