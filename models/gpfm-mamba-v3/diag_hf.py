#!/usr/bin/env python3
"""diag_hf.py — gpfm-mamba-v3 高频诊断入口包装（薄变体；同 train/evaluate 包装模式）。

diag_hf 按 cfg.variant_name 定位 checkpoint，故 v3 run 须经本包装（config 遮蔽）执行，
否则 v1 config 会在 v3lp- run 目录下找 gpfm-mamba-f0-best.pt（FileNotFoundError 根因）。

用法：
  python3 models/gpfm-mamba-v3/diag_hf.py --run-id v3lp-main-01
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

RUN_PREFIX = "v3lp-"

_HERE = Path(__file__).resolve().parent
_V1 = _HERE.parent / "gpfm-mamba-v1"


def _shadow_v3_config() -> None:
    sys.path.insert(0, str(_V1))
    sys.path.insert(0, str(_HERE))
    import config as _C  # noqa: F401
    if Path(_C.__file__).resolve().parent != _HERE:
        raise RuntimeError(f"config 遮蔽失败：import 到 {_C.__file__}")


def _guard_run_id(argv: list) -> None:
    args = argv[1:]
    if "--run-id" not in args:
        sys.exit("[v3 wrapper] diag_hf 必须显式传 --run-id（v3lp- 前缀）")
    run_id = args[args.index("--run-id") + 1]
    if not run_id.startswith(RUN_PREFIX):
        sys.exit(f"[v3 wrapper] --run-id 须以 '{RUN_PREFIX}' 开头，got: {run_id}")


def main() -> None:
    _shadow_v3_config()
    _guard_run_id(sys.argv)
    spec = importlib.util.spec_from_file_location("gpfm_mamba_v1_diag_hf", str(_V1 / "diag_hf.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()


if __name__ == "__main__":
    main()
