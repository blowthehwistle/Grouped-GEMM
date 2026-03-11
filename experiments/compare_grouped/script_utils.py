"""implementations/ 스크립트에서 grouped_config 로드용 유틸."""

import importlib.util
from pathlib import Path
from typing import Optional, Tuple, Callable, Any

CONFIG_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = CONFIG_DIR / "config.yaml"


def load_grouped_config_module() -> Tuple[Optional[Callable], Optional[Callable]]:
    """
    implementations/에서 호출 시 (load_grouped_sizes, add_grouped_args) 반환.
    실패 시 (None, None).
    """
    cfg_path = CONFIG_DIR / "grouped_config.py"
    if not cfg_path.exists():
        return None, None
    try:
        spec = importlib.util.spec_from_file_location("grouped_config", cfg_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.load_grouped_sizes, mod.add_grouped_args
    except Exception:
        return None, None
