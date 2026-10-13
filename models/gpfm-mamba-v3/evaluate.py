#!/usr/bin/env python3
"""evaluate.py — gpfm-mamba-v3（prior_bandlimit 40Hz 消融）评估入口包装（薄变体，不复制逻辑）。

依赖方式（同 v2 包装）：v3 config 遮蔽 sys.modules['config'] 后，以独立模块名加载
v1 evaluate.py（Pass A/B + GC + PSD + 空间归因 + 全部图）——build_eval_prior 经
getattr 透传 prior_bandlimit_hz=40，推理出发点 x0 与训练同滤一致；config_hash 严格校验。

run_id 前缀守卫：--run-id 必填且以 `v3lp-` 开头。

用法：
  python3 models/gpfm-mamba-v3/evaluate.py --run-id v3lp-main-01
  python3 models/gpfm-mamba-v1/fc_metric.py --run-id v3lp-main-01   # 指标 1 v2.1（共享 outputs 根）
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

RUN_PREFIX = "v3lp-"

_HERE = Path(__file__).resolve().parent          # models/gpfm-mamba-v3
_V1 = _HERE.parent / "gpfm-mamba-v1"             # 只读依赖的 v1 目录


def _shadow_v3_config() -> None:
    """v3 config 优先：占据 sys.path[0] 与 sys.modules['config']（先于插入 v1 目录）。"""
    sys.path.insert(0, str(_V1))
    sys.path.insert(0, str(_HERE))
    import config as _C  # noqa: F401  -> 解析到 v3/config.py（sys.path[0]）
    if Path(_C.__file__).resolve().parent != _HERE:
        raise RuntimeError(f"config 遮蔽失败：import 到 {_C.__file__}（应为 {_HERE / 'config.py'}）")


def _guard_run_id(argv: list) -> None:
    args = argv[1:]
    if "--run-id" not in args:
        sys.exit("[v3 wrapper] evaluate 必须显式传 --run-id（v3 run 一律 v3lp- 前缀，"
                 "如 --run-id v3lp-main-01）")
    run_id = args[args.index("--run-id") + 1]
    if not run_id.startswith(RUN_PREFIX):
        sys.exit(f"[v3 wrapper] --run-id 须以 '{RUN_PREFIX}' 开头，got: {run_id}")


def _load_v1_evaluate():
    """以独立模块名加载 v1 evaluate.py（其 import config 命中 v3 缓存）。"""
    spec = importlib.util.spec_from_file_location("gpfm_mamba_v1_evaluate", str(_V1 / "evaluate.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    _shadow_v3_config()
    _guard_run_id(sys.argv)
    mod = _load_v1_evaluate()
    mod.main()


if __name__ == "__main__":
    main()
