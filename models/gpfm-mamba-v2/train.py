#!/usr/bin/env python3
"""train.py — gpfm-mamba-v2（d_model=62 消融）训练入口包装（薄变体，不复制逻辑）。

依赖方式（design §7 v2 行）：本脚本只做三件事——
1. sys.path 令 **v2 的 config.py 优先于 v1 的**（同名模块遮蔽）：先 import 本目录 config
   并占据 sys.modules["config"]，再加载 v1 train.py；其内部 `import config as C`
   命中缓存即得 v2 config（d_model=62、variant_name=gpfm-mamba-v2d62）——
   data/gpr_prior/model/losses/train 循环本体全部只读复用 v1。
2. run_id 前缀守卫：v2 与 v1 共享任务 outputs 根 `/root/npi/outputs/gpfm-mamba-f0/`，
   强制 run_id 以 `v2d62-` 开头（--resume 派生的 run 名同样校验），防止误写 v1 run 目录。
3. 以独立模块名（gpfm_mamba_v1_train）importlib 加载 v1 train.py 并调用其 main()。

用法（与 v1 train.py 完全同名参数）：
  python3 models/gpfm-mamba-v2/train.py --smoke --run-id v2d62-smoke-a
  python3 models/gpfm-mamba-v2/train.py --epochs 1 --bench --run-id v2d62-bench-1ep
  nohup python3 models/gpfm-mamba-v2/train.py --run-id v2d62-main-01 \
      > /root/npi/outputs/gpfm-mamba-f0/v2d62-main-01/logs/train.log 2>&1 &
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time
from pathlib import Path

RUN_PREFIX = "v2d62-"

_HERE = Path(__file__).resolve().parent          # models/gpfm-mamba-v2
_V1 = _HERE.parent / "gpfm-mamba-v1"             # 只读依赖的 v1 目录


def _shadow_v2_config() -> None:
    """v2 config 优先：占据 sys.path[0] 与 sys.modules['config']（先于插入 v1 目录）。"""
    sys.path.insert(0, str(_V1))
    sys.path.insert(0, str(_HERE))
    import config as _C  # noqa: F401  -> 解析到 v2/config.py（sys.path[0]）
    if Path(_C.__file__).resolve().parent != _HERE:
        raise RuntimeError(f"config 遮蔽失败：import 到 {_C.__file__}（应为 {_HERE / 'config.py'}）")


def _guard_run_id(argv: list) -> None:
    """run_id 前缀守卫（path-isolation）：v2 run 目录一律 v2d62- 前缀。"""
    args = argv[1:]
    run_id = args[args.index("--run-id") + 1] if "--run-id" in args else None
    resume = args[args.index("--resume") + 1] if "--resume" in args else None
    if run_id is not None and not run_id.startswith(RUN_PREFIX):
        sys.exit(f"[v2 wrapper] --run-id 须以 '{RUN_PREFIX}' 开头（与 v1 run 目录隔离），got: {run_id}")
    if run_id is None:
        if resume is not None:
            derived = Path(resume).resolve().parents[1].name  # <run>/checkpoints/x.pt -> <run>
            if not derived.startswith(RUN_PREFIX):
                sys.exit(f"[v2 wrapper] --resume 所属 run 目录须以 '{RUN_PREFIX}' 开头，got: {derived}")
        else:
            argv.extend(["--run-id", f"{RUN_PREFIX}run_{time.strftime('%Y%m%d-%H%M%S')}"])


def _load_v1_train():
    """以独立模块名加载 v1 train.py（避免与本包装同名冲突；其 import config 命中 v2 缓存）。"""
    spec = importlib.util.spec_from_file_location("gpfm_mamba_v1_train", str(_V1 / "train.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    _shadow_v2_config()
    _guard_run_id(sys.argv)
    mod = _load_v1_train()
    mod.main()


if __name__ == "__main__":
    main()
