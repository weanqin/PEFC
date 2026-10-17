#!/usr/bin/env python3
"""evaluate.py — gpfm-mamba-v1-fmt（新预处理范式 edra_processed_fmt）评估入口包装（薄变体，不复制逻辑）。

依赖方式（同 train.py 包装，单遮蔽）：v1-fmt config 占据 sys.modules['config'] 后，以独立
模块名加载 v1 evaluate.py（Pass A/B + PSD + 空间归因 + 图）。v1 evaluate.py 内部
`from sample import …` 经 sys.path 命中 **v1 sample.py**（v1-fmt 目录无 sample.py，不遮蔽）；
sample.py 的 load_eval_model 经 `norm_type=getattr(cfg, ...)` 钩子构建 LayerNorm 结构装载。
checkpoint config_hash 严格校验。PSD/轨迹图 fs 经 cfg.fs=250 透传（v1 evaluate.py 钩子）。

正式评估一律带 `--skip-gc-only`（D1：genGC vs real GC 对比阻塞——PSD fs=250 与空间结构归因
照跑，summary 记 "gc": "skipped(blocked)"，不落任何 GC 产物；与 --skip-gc 互斥）。

run_id 前缀守卫：--run-id 必填且以 `v1fmt-` 开头（防止默认 latest-run 逻辑误评其他变体 run）。

用法：
  python3 models/gpfm-mamba-v1-fmt/evaluate.py --run-id v1fmt-main-01 --skip-gc-only
  python3 models/gpfm-mamba-v1-fmt/evaluate.py --run-id v1fmt-main-01 --figures-only  # 指标 2 重渲染
"""

from __future__ import annotations

import os

# BLAS 线程上限（environment.md 惯例：有效 16 核 cgroup，须在 numpy 导入前设置；可覆盖）
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "8")

import importlib.util  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

RUN_PREFIX = "v1fmt-"

_HERE = Path(__file__).resolve().parent          # models/gpfm-mamba-v1-fmt
_V1 = _HERE.parent / "gpfm-mamba-v1"             # 只读依赖的 v1 目录


def _shadow_fmt_config() -> None:
    """v1-fmt config 单遮蔽（与 train.py 包装同款，含 model→v1 自检）。"""
    sys.path.insert(0, str(_V1))
    sys.path.insert(0, str(_HERE))
    import config as _C  # noqa: F401  -> 解析到 v1-fmt/config.py（sys.path[0]）
    if Path(_C.__file__).resolve().parent != _HERE:
        raise RuntimeError(f"config 遮蔽失败：import 到 {_C.__file__}（应为 {_HERE / 'config.py'}）")
    import model as _M  # noqa: F401  -> v1-fmt 目录无 model.py，须解析到 v1（norm_type 钩子）
    if Path(_M.__file__).resolve().parent != _V1:
        raise RuntimeError(f"model 解析异常：import 到 {_M.__file__}（应为 {_V1 / 'model.py'}）")


def _guard_run_id(argv: list) -> None:
    """--run-id 必填 + `v1fmt-` 前缀守卫（path-isolation：与 v4a-fmt 共享 outputs 根）。"""
    args = argv[1:]
    if "--run-id" not in args:
        sys.exit("[v1fmt wrapper] evaluate 必须显式传 --run-id（v1-fmt run 一律 v1fmt- 前缀，"
                 "如 --run-id v1fmt-main-01）")
    run_id = args[args.index("--run-id") + 1]
    if not run_id.startswith(RUN_PREFIX):
        sys.exit(f"[v1fmt wrapper] --run-id 须以 '{RUN_PREFIX}' 开头，got: {run_id}")


def _load_v1_evaluate():
    """以独立模块名加载 v1 evaluate.py（避免与本包装同名冲突；其 import config 命中 v1-fmt 缓存）。"""
    spec = importlib.util.spec_from_file_location("gpfm_mamba_v1_evaluate", str(_V1 / "evaluate.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    _shadow_fmt_config()
    _guard_run_id(sys.argv)
    mod = _load_v1_evaluate()
    mod.main()


if __name__ == "__main__":
    main()
