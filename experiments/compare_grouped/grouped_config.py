"""Grouped GEMM M,N,K 설정 로더. configs/default.yaml 또는 CLI로 로드."""

import sys
from pathlib import Path
from typing import List, Optional, Tuple

# 기본값 (CUDA grouped_gemm.cu와 동일: batch_size=8, M=1024, N=4096, K=14336)
DEFAULT_BATCH_SIZE = 8
DEFAULT_M = 1024
DEFAULT_N = 4096
DEFAULT_K = 14336


def _parse_list(s: str) -> List[int]:
    """'1024,512,256' 또는 '1024 512 256' -> [1024, 512, 256]"""
    s = s.replace(",", " ").strip()
    return [int(x) for x in s.split() if x]


def load_from_yaml(path: Path) -> Tuple[List[int], List[int], List[int]]:
    """
    YAML config에서 grouped 설정 로드 (예: configs/default.yaml).

    지원 형식:
    1) uniform: batch_size, m, n, k (스칼라) → 8개 GEMM, 모두 (1024,4096,14336)
    2) list: m, n, k (리스트) → 각 배치별 크기
    """
    try:
        import yaml
    except ImportError:
        print("grouped_config: PyYAML not installed, using defaults (1024,4096,14336 x8)", file=sys.stderr)
        return (
            [DEFAULT_M] * DEFAULT_BATCH_SIZE,
            [DEFAULT_N] * DEFAULT_BATCH_SIZE,
            [DEFAULT_K] * DEFAULT_BATCH_SIZE,
        )

    path = Path(path)
    if not path.exists():
        print(f"grouped_config: Config file not found: {path.resolve()}, using defaults (1024,4096,14336 x8)", file=sys.stderr)
        return (
            [DEFAULT_M] * DEFAULT_BATCH_SIZE,
            [DEFAULT_N] * DEFAULT_BATCH_SIZE,
            [DEFAULT_K] * DEFAULT_BATCH_SIZE,
        )

    with open(path) as f:
        data = yaml.safe_load(f) or {}

    grouped = data.get("grouped", {})
    batch_size = grouped.get("batch_size", DEFAULT_BATCH_SIZE)
    m_val = grouped.get("m", DEFAULT_M)
    n_val = grouped.get("n", DEFAULT_N)
    k_val = grouped.get("k", DEFAULT_K)

    # 리스트 형식인지 스칼라인지 판별
    def to_list(v):
        if isinstance(v, list):
            return [int(x) for x in v]
        if isinstance(v, str):
            parsed = _parse_list(v)
            return parsed if len(parsed) > 1 else parsed * batch_size
        return [int(v)] * batch_size

    m_list = to_list(m_val)
    n_list = to_list(n_val)
    k_list = to_list(k_val)

    return m_list, n_list, k_list


def load_grouped_sizes(
    config_path: Optional[Path] = None,
    m: Optional[str] = None,
    n: Optional[str] = None,
    k: Optional[str] = None,
    default_config: Optional[Path] = None,
) -> Tuple[List[int], List[int], List[int]]:
    """
    M, N, K 리스트 로드. 우선순위: CLI(m,n,k) > config_path > default_config > 기본값.

    Args:
        config_path: YAML config 파일 경로 (없으면 default_config 사용)
        m: "1024,512,256,128" 형태의 M 리스트
        n: N 리스트
        k: K 리스트
        default_config: config_path가 없을 때 사용할 기본 config 경로

    Returns:
        (m_list, n_list, k_list)
    """
    m_list = [DEFAULT_M] * DEFAULT_BATCH_SIZE
    n_list = [DEFAULT_N] * DEFAULT_BATCH_SIZE
    k_list = [DEFAULT_K] * DEFAULT_BATCH_SIZE

    # 1. config 파일에서 로드
    path = config_path or default_config
    if path:
        path = Path(path)
        if path.is_absolute() or path.exists():
            actual_path = path
        else:
            # experiments/compare_grouped/configs/ 기준
            base = Path(__file__).resolve().parent
            actual_path = base / path
        m_list, n_list, k_list = load_from_yaml(actual_path)

    # 2. CLI 인자로 override
    if m is not None:
        m_list = _parse_list(m)
    if n is not None:
        n_list = _parse_list(n)
    if k is not None:
        k_list = _parse_list(k)

    # 검증
    size = len(m_list)
    if len(n_list) != size or len(k_list) != size:
        raise ValueError(
            f"m, n, k 길이 불일치: m={len(m_list)}, n={len(n_list)}, k={len(k_list)}"
        )

    return m_list, n_list, k_list


def add_grouped_args(parser):
    """argparse에 grouped sizes 인자 추가."""
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Config YAML 경로 (grouped.m, grouped.n, grouped.k)",
    )
    parser.add_argument(
        "--m",
        type=str,
        default=None,
        help="M 리스트 (예: 1024,512,256,128)",
    )
    parser.add_argument(
        "--n",
        type=str,
        default=None,
        help="N 리스트",
    )
    parser.add_argument(
        "--k",
        type=str,
        default=None,
        help="K 리스트",
    )
