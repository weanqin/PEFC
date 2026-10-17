#!/usr/bin/env python3
"""evaluate.py — gpfm-mamba-v4a-fmt（新预处理范式 edra_processed_fmt）评估入口包装（薄变体，不复制逻辑）。

依赖方式（同 train.py 包装，单遮蔽）：v4a-fmt config 占据 sys.modules['config'] 后，以独立
模块名加载 v1 evaluate.py。v1 evaluate.py 内部 `from sample import …` 经 sys.path 命中
**v1 sample.py**（v4a-fmt 目录无 sample.py）；sample.py 的 load_eval_model 经
`bottleneck_rank`/`norm_type` getattr 钩子构建低秩 + LayerNorm 结构装载。checkpoint
config_hash 严格校验（r≠16 的 run 评估须带 --rank R，否则 hash 校验拒绝——防 r 串用）。
PSD/轨迹图 fs 经 cfg.fs=250 透传。

正式评估一律带 `--skip-gc-only`（D1：GC 对比阻塞——PSD fs=250 与空间结构归因照跑，
summary 记 "gc": "skipped(blocked)"，不落任何 GC 产物；与 --skip-gc 互斥）。

--rank {8,16,32}（r 扫描 run 用；默认 16 = config 文件值）与 train.py 包装同款：解析移除 +
改写被遮蔽 config 的 build_config + 生效 run_id 须含匹配 rXX 标识。

run_id 前缀守卫：--run-id 必填且以 `v4afmt-` 开头（防止默认 latest-run 逻辑误评其他变体 run）。

用法：
  python3 models/gpfm-mamba-v4a-fmt/evaluate.py --run-id v4afmt-r16-main-01 --skip-gc-only
  python3 models/gpfm-mamba-v4a-fmt/evaluate.py --rank 8 --run-id v4afmt-r8-main-01 --skip-gc-only
"""

from __future__ import annotations

import os

# BLAS 线程上限（environment.md 惯例：有效 16 核 cgroup，须在 numpy 导入前设置；可覆盖）
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Optional  # noqa: E402

RUN_PREFIX = "v4afmt-"
RANK_CHOICES = (8, 16, 32)

_HERE = Path(__file__).resolve().parent          # models/gpfm-mamba-v4a-fmt
_V1 = _HERE.parent / "gpfm-mamba-v1"             # 只读依赖的 v1 目录


def _shadow_fmt_config() -> None:
    """v4a-fmt config 单遮蔽（与 train.py 包装同款，含 model→v1 自检）。"""
    sys.path.insert(0, str(_V1))
    sys.path.insert(0, str(_HERE))
    import config as _C  # noqa: F401  -> 解析到 v4a-fmt/config.py（sys.path[0]）
    if Path(_C.__file__).resolve().parent != _HERE:
        raise RuntimeError(f"config 遮蔽失败：import 到 {_C.__file__}（应为 {_HERE / 'config.py'}）")
    import model as _M  # noqa: F401  -> v4a-fmt 目录无 model.py，须解析到 v1（bottleneck_rank/norm_type 钩子）
    if Path(_M.__file__).resolve().parent != _V1:
        raise RuntimeError(f"model 解析异常：import 到 {_M.__file__}（应为 {_V1 / 'model.py'}）")


def _pop_rank_arg(argv: list) -> Optional[int]:
    """解析并从 argv 移除 --rank {8,16,32}（v1 evaluate.py 的 argparse 不识别该参数）。"""
    if not any(a == "--rank" or a.startswith("--rank=") for a in argv[1:]):
        return None
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--rank", type=int, choices=RANK_CHOICES)
    ns, rest = p.parse_known_args(argv[1:])
    argv[1:] = rest
    return ns.rank


def _apply_rank_override(rank: int) -> None:
    """--rank 覆写（与 train.py 包装同款）：改写被遮蔽 config 模块的 build_config。"""
    import config as _C
    _orig_build = _C.build_config

    def _build_with_rank(**overrides):
        if "bottleneck_rank" in overrides:
            raise KeyError("bottleneck_rank 与 --rank 冲突（r 覆写仅经 wrapper --rank 进入）")
        return _orig_build(bottleneck_rank=rank, **overrides)

    _C.build_config = _build_with_rank


def _guard_run_id(argv: list, rank: Optional[int]) -> None:
    """--run-id 必填 + `v4afmt-` 前缀守卫 + --rank 的 rXX 标识一致性校验。"""
    args = argv[1:]
    if "--run-id" not in args:
        sys.exit("[v4afmt wrapper] evaluate 必须显式传 --run-id（v4a-fmt run 一律 v4afmt- 前缀，"
                 "如 --run-id v4afmt-r16-main-01）")
    run_id = args[args.index("--run-id") + 1]
    if not run_id.startswith(RUN_PREFIX):
        sys.exit(f"[v4afmt wrapper] --run-id 须以 '{RUN_PREFIX}' 开头，got: {run_id}")
    if rank is not None and f"r{rank}" not in run_id:
        sys.exit(f"[v4afmt wrapper] --rank {rank} 与 run_id 缺少 'r{rank}' 标识不一致（防 r 串用），got: {run_id}")


def _load_v1_evaluate():
    """以独立模块名加载 v1 evaluate.py（避免与本包装同名冲突；其 import config 命中 v4a-fmt 缓存）。"""
    spec = importlib.util.spec_from_file_location("gpfm_mamba_v1_evaluate", str(_V1 / "evaluate.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    rank = _pop_rank_arg(sys.argv)
    _shadow_fmt_config()
    if rank is not None:
        _apply_rank_override(rank)
    _guard_run_id(sys.argv, rank)
    mod = _load_v1_evaluate()
    mod.main()


if __name__ == "__main__":
    main()
