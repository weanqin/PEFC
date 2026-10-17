#!/usr/bin/env python3
"""evaluate.py — gpfm-mamba-v4a-fmt-adaln（5 norm 位 adaLN(t) scale/shift）评估入口包装（薄变体，不复制逻辑）。

依赖方式（同 train.py 包装，单遮蔽）：本变体 config 占据 sys.modules['config'] 后，以独立
模块名加载 v1 evaluate.py。v1 evaluate.py 内部 `from sample import …` 经 sys.path 命中
**v1 sample.py**（本目录无 sample.py）；sample.py 的 load_eval_model 经 `bottleneck_rank`/
`norm_type`/`cond_adaln` getattr 钩子构建低秩 + LayerNorm + adaLN 调制结构装载（checkpoint
config_hash 严格校验天然隔离跨变体 ckpt）。PSD/轨迹图 fs 经 cfg.fs=250 透传。

正式评估一律带 `--skip-gc-only`（GC 对比阻塞延续 v4a-fmt D1 口径——PSD fs=250 与空间结构
归因照跑，summary 记 "gc": "skipped(blocked)"，不落任何 GC 产物；与 --skip-gc 互斥）。

run_id 前缀守卫：--run-id 必填且以 `v4afmt-adaln-` 开头（**`v4afmt-adaln-*` 归本任务根**
`/root/npi/outputs/gpfm-mamba-adaln-f0`**）；已知局限：本前缀嵌套于 v4a-fmt 的 `v4afmt-`
之下——若误经 v4a-fmt wrapper 传 adaln run_id 其守卫放行，晚失败兜底非静默（design §1.2）。
无 --rank（r=16 固定，比 v4a-fmt wrapper 简一档）。

用法：
  python3 models/gpfm-mamba-v4a-fmt-adaln/evaluate.py --run-id v4afmt-adaln-main-01 --skip-gc-only
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

RUN_PREFIX = "v4afmt-adaln-"  # 本任务根唯一前缀（嵌套于 v4afmt- 之下的已知局限见 docstring）

_HERE = Path(__file__).resolve().parent          # models/gpfm-mamba-v4a-fmt-adaln
_V1 = _HERE.parent / "gpfm-mamba-v1"             # 只读依赖的 v1 目录


def _shadow_adaln_config() -> None:
    """本变体 config 单遮蔽（与 train.py 包装同款，含 model→v1 自检）。"""
    sys.path.insert(0, str(_V1))
    sys.path.insert(0, str(_HERE))
    import config as _C  # noqa: F401  -> 解析到本变体 config.py（sys.path[0]）
    if Path(_C.__file__).resolve().parent != _HERE:
        raise RuntimeError(f"config 遮蔽失败：import 到 {_C.__file__}（应为 {_HERE / 'config.py'}）")
    import model as _M  # noqa: F401  -> 本目录无 model.py，须解析到 v1（cond_adaln 钩子）
    if Path(_M.__file__).resolve().parent != _V1:
        raise RuntimeError(f"model 解析异常：import 到 {_M.__file__}（应为 {_V1 / 'model.py'}）")


def _guard_run_id(argv: list) -> None:
    """--run-id 必填 + `v4afmt-adaln-` 前缀守卫（path-isolation；--resume 派生名同检）。"""
    args = argv[1:]
    if "--run-id" not in args:
        sys.exit("[v4afmt-adaln wrapper] evaluate 必须显式传 --run-id（本变体 run 一律 v4afmt-adaln- 前缀，"
                 "如 --run-id v4afmt-adaln-main-01）")
    run_id = args[args.index("--run-id") + 1]
    if not run_id.startswith(RUN_PREFIX):
        sys.exit(f"[v4afmt-adaln wrapper] --run-id 须以 '{RUN_PREFIX}' 开头，got: {run_id}")


def _load_v1_evaluate():
    """以独立模块名加载 v1 evaluate.py（避免与本包装同名冲突；其 import config 命中本变体缓存）。"""
    spec = importlib.util.spec_from_file_location("gpfm_mamba_v1_evaluate", str(_V1 / "evaluate.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    _shadow_adaln_config()
    _guard_run_id(sys.argv)
    mod = _load_v1_evaluate()
    mod.main()


if __name__ == "__main__":
    main()
