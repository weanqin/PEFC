#!/usr/bin/env python3
"""train.py — gpfm-mamba-v1-fmt（新预处理范式 edra_processed_fmt）训练入口包装（薄变体，不复制逻辑）。

依赖方式（v4a 式**单遮蔽**——v1-fmt 目录无 model.py，模型经 v1 model.py 的 norm_type
default-off 钩子构建）：本脚本只做三件事——
1. sys.path 令 **v1-fmt 的 config.py 优先于 v1 的**（同名模块遮蔽）：先 import 本目录 config
   并占据 sys.modules["config"]，再以独立模块名加载 v1 train.py；其内部 `import config as C`
   命中缓存即得 v1-fmt config（data_root=fmt 数据集、period_p=25、norm_type="layernorm"、
   variant_name=gpfm-mamba-v1-fmt）——train 循环本体全部只读复用 v1（train.py 的
   `norm_type=getattr(cfg, ...)` 等钩子生效）。无 --rank（v4a-fmt 才有瓶颈 r 透传）。
2. run_id 前缀守卫：v1-fmt 与 v4a-fmt 共享本任务 outputs 根 `/root/npi/outputs/gpfm-mamba-fmt-f0/`，
   强制 run_id 以 `v1fmt-` 开头（--resume 派生的 run 名同样校验），防止误写其他变体 run 目录。
3. 以独立模块名（gpfm_mamba_v1_train）importlib 加载 v1 train.py 并调用其 main()。

用法（与 v1 train.py 完全同名参数）：
  python3 models/gpfm-mamba-v1-fmt/train.py --smoke --run-id v1fmt-smoke-01
  python3 models/gpfm-mamba-v1-fmt/train.py --epochs 1 --bench --run-id v1fmt-bench-1ep
  nohup python3 models/gpfm-mamba-v1-fmt/train.py --run-id v1fmt-main-01 \
      > /root/npi/outputs/gpfm-mamba-fmt-f0/v1fmt-main-01/logs/train.log 2>&1 &
"""

from __future__ import annotations

import os

# BLAS 线程上限（environment.md 惯例：有效 16 核 cgroup，须在 numpy 导入前设置；可覆盖）
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "8")

import importlib.util  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Optional  # noqa: E402

RUN_PREFIX = "v1fmt-"

_HERE = Path(__file__).resolve().parent          # models/gpfm-mamba-v1-fmt
_V1 = _HERE.parent / "gpfm-mamba-v1"             # 只读依赖的 v1 目录


def _shadow_fmt_config() -> None:
    """v1-fmt config 单遮蔽：占据 sys.path[0] 与 sys.modules['config']（先于插入 v1 目录）。

    与 v4 同款双遮蔽不同：v1-fmt 目录无 model.py——`import model` 须解析到 v1（norm_type
    钩子所在），此处显式自检（遮蔽不足/过度均报错）。
    """
    sys.path.insert(0, str(_V1))
    sys.path.insert(0, str(_HERE))
    import config as _C  # noqa: F401  -> 解析到 v1-fmt/config.py（sys.path[0]）
    if Path(_C.__file__).resolve().parent != _HERE:
        raise RuntimeError(f"config 遮蔽失败：import 到 {_C.__file__}（应为 {_HERE / 'config.py'}）")
    import model as _M  # noqa: F401  -> v1-fmt 目录无 model.py，须解析到 v1（norm_type 钩子）
    if Path(_M.__file__).resolve().parent != _V1:
        raise RuntimeError(f"model 解析异常：import 到 {_M.__file__}（应为 {_V1 / 'model.py'}）")


def _effective_run_id(argv: list) -> Optional[str]:
    """生效 run_id：显式 --run-id；否则 --resume 派生（<run>/checkpoints/x.pt -> <run>）；均无则 None。"""
    args = argv[1:]
    if "--run-id" in args:
        return args[args.index("--run-id") + 1]
    if "--resume" in args:
        return Path(args[args.index("--resume") + 1]).resolve().parents[1].name
    return None


def _guard_run_id(argv: list) -> None:
    """run_id 前缀守卫（path-isolation：本任务 outputs 根与 v4a-fmt 共享，前缀隔离）。"""
    run_id = _effective_run_id(argv)
    if run_id is not None and not run_id.startswith(RUN_PREFIX):
        sys.exit(f"[v1fmt wrapper] --run-id 须以 '{RUN_PREFIX}' 开头（与其他变体 run 目录隔离），got: {run_id}")
    if run_id is None:
        argv.extend(["--run-id", f"{RUN_PREFIX}run_{time.strftime('%Y%m%d-%H%M%S')}"])


def _load_v1_train():
    """以独立模块名加载 v1 train.py（避免与本包装同名冲突；其 import config 命中 v1-fmt 缓存）。"""
    spec = importlib.util.spec_from_file_location("gpfm_mamba_v1_train", str(_V1 / "train.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    _shadow_fmt_config()
    _guard_run_id(sys.argv)
    mod = _load_v1_train()
    mod.main()


if __name__ == "__main__":
    main()
