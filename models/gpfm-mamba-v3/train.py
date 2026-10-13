#!/usr/bin/env python3
"""train.py — gpfm-mamba-v3（prior_bandlimit 40Hz 消融）训练入口包装（薄变体，不复制逻辑）。

依赖方式（同 v2 包装）：v3 的 config.py 遮蔽 sys.modules['config'] 后，以独立模块名加载
v1 train.py——其内部 `import config as C` 命中缓存即得 v3 config（prior_bandlimit_hz=40.0、
variant_name=gpfm-mamba-v3lp、d_model=16 与 v1 一致）；train 循环本体全部只读复用 v1。
先验构造处 `bandlimit_hz=getattr(cfg, "prior_bandlimit_hz", None)` 由此启用采样后带限。

run_id 前缀守卫：与 v1/v2 共享任务 outputs 根，强制 run_id 以 `v3lp-` 开头。

用法：
  python3 models/gpfm-mamba-v3/train.py --smoke --run-id v3lp-smoke-a
  nohup python3 models/gpfm-mamba-v3/train.py --run-id v3lp-main-01 \
      > /root/npi/outputs/gpfm-mamba-f0/v3lp-main-01/logs/train.log 2>&1 &
"""

from __future__ import annotations

import importlib.util
import sys
import time
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
    """run_id 前缀守卫（path-isolation）：v3 run 目录一律 v3lp- 前缀。"""
    args = argv[1:]
    run_id = args[args.index("--run-id") + 1] if "--run-id" in args else None
    resume = args[args.index("--resume") + 1] if "--resume" in args else None
    if run_id is not None and not run_id.startswith(RUN_PREFIX):
        sys.exit(f"[v3 wrapper] --run-id 须以 '{RUN_PREFIX}' 开头（与 v1/v2 run 目录隔离），got: {run_id}")
    if run_id is None:
        if resume is not None:
            derived = Path(resume).resolve().parents[1].name  # <run>/checkpoints/x.pt -> <run>
            if not derived.startswith(RUN_PREFIX):
                sys.exit(f"[v3 wrapper] --resume 所属 run 目录须以 '{RUN_PREFIX}' 开头，got: {derived}")
        else:
            argv.extend(["--run-id", f"{RUN_PREFIX}run_{time.strftime('%Y%m%d-%H%M%S')}"])


def _load_v1_train():
    """以独立模块名加载 v1 train.py（避免与本包装同名冲突；其 import config 命中 v3 缓存）。"""
    spec = importlib.util.spec_from_file_location("gpfm_mamba_v1_train", str(_V1 / "train.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    _shadow_v3_config()
    _guard_run_id(sys.argv)
    mod = _load_v1_train()
    mod.main()


if __name__ == "__main__":
    main()
