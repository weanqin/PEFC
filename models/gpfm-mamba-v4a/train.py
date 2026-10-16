#!/usr/bin/env python3
"""train.py — gpfm-mamba-v4a（v4-A：低秩输入瓶颈 r=16）训练入口包装（薄变体，不复制逻辑）。

依赖方式（v2 式**单遮蔽**——v4a 目录无 model.py，模型经 v1 model.py 的 bottleneck_rank
default-off 钩子构建）：本脚本只做四件事——
1. sys.path 令 **v4a 的 config.py 优先于 v1 的**（同名模块遮蔽）：先 import 本目录 config
   并占据 sys.modules["config"]，再加载 v1 train.py；其内部 `import config as C`
   命中缓存即得 v4a config（d_model=62、bottleneck_rank=16、variant_name=gpfm-mamba-v4a）
   ——model/data/gpr_prior/losses/train 循环本体全部只读复用 v1（v1 train.py 的
   `bottleneck_rank=getattr(cfg, "bottleneck_rank", None)` 钩子构建低秩 in_proj）。
2. **--rank {8,16,32} 覆写**（r 扫描条件扩展；config.py 文件默认唯一 16）：从 argv 解析并
   移除 --rank（v1 train.py 不识别该参数），改写被遮蔽 config 模块的 build_config——
   返回实例的 bottleneck_rank=R（resolved config 与 config_hash 随之反映实际 r；
   resume 的 config_hash 校验天然防不同 r 的 ckpt 串用）；同时校验生效 run_id 含
   匹配 `r{R}` 标识（如 v4a-r8-main-01），防 run 目录与实际结构错记。
3. run_id 前缀守卫：v4a 与 v1/v2 共享任务 outputs 根 `/root/npi/outputs/gpfm-mamba-f0/`，
   强制 run_id 以 `v4a-` 开头（--resume 派生的 run 名同样校验），防止误写其他变体 run 目录。
4. 以独立模块名（gpfm_mamba_v1_train）importlib 加载 v1 train.py 并调用其 main()。

用法（与 v1 train.py 完全同名参数 + 可选 --rank）：
  python3 models/gpfm-mamba-v4a/train.py --smoke --run-id v4a-smoke-a
  python3 models/gpfm-mamba-v4a/train.py --epochs 1 --bench --run-id v4a-r16-bench-1ep
  python3 models/gpfm-mamba-v4a/train.py --run-id v4a-r16-main-01            # r=16（config 默认）
  python3 models/gpfm-mamba-v4a/train.py --rank 8 --run-id v4a-r8-main-01    # r 扫描（须含 r8 标识）
  nohup python3 models/gpfm-mamba-v4a/train.py --run-id v4a-r16-main-01 \
      > /root/npi/outputs/gpfm-mamba-f0/v4a-r16-main-01/logs/train.log 2>&1 &
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
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Optional  # noqa: E402

RUN_PREFIX = "v4a-"
RANK_CHOICES = (8, 16, 32)  # r 扫描网格（design §4.2；主判读后议，config.py 文件默认唯一 16）

_HERE = Path(__file__).resolve().parent          # models/gpfm-mamba-v4a
_V1 = _HERE.parent / "gpfm-mamba-v1"             # 只读依赖的 v1 目录


def _shadow_v4a_config() -> None:
    """v4a config 单遮蔽：占据 sys.path[0] 与 sys.modules['config']（先于插入 v1 目录）。

    与 v4b 的双遮蔽不同：v4a 目录无 model.py——`import model` 须解析到 v1（低秩钩子所在），
    此处显式自检（遮蔽不足/过度均报错）。
    """
    sys.path.insert(0, str(_V1))
    sys.path.insert(0, str(_HERE))
    import config as _C  # noqa: F401  -> 解析到 v4a/config.py（sys.path[0]）
    if Path(_C.__file__).resolve().parent != _HERE:
        raise RuntimeError(f"config 遮蔽失败：import 到 {_C.__file__}（应为 {_HERE / 'config.py'}）")
    import model as _M  # noqa: F401  -> v4a 目录无 model.py，须解析到 v1（bottleneck_rank 钩子）
    if Path(_M.__file__).resolve().parent != _V1:
        raise RuntimeError(f"model 解析异常：import 到 {_M.__file__}（应为 {_V1 / 'model.py'}）")


def _pop_rank_arg(argv: list) -> Optional[int]:
    """解析并从 argv 移除 --rank {8,16,32}（v1 train.py 的 argparse 不识别该参数）。

    兼容 `--rank 8` 与 `--rank=8` 两种传法；非法取值（如 --rank 5）在构建期即退出
    （argparse choices）；未传时返回 None（config 默认 16）。
    """
    if not any(a == "--rank" or a.startswith("--rank=") for a in argv[1:]):
        return None
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--rank", type=int, choices=RANK_CHOICES)
    ns, rest = p.parse_known_args(argv[1:])
    argv[1:] = rest
    return ns.rank


def _apply_rank_override(rank: int) -> None:
    """--rank 覆写：改写被遮蔽 config 模块的 build_config——返回实例 bottleneck_rank=rank。

    config.py 文件默认唯一 16（薄变体契约，不随 r 扫描改文件）；r 只经本运行时改写进入，
    resolved config 与 config_hash 均反映实际 r（v1 train.py 落盘的 config.yaml/checkpoint
    一致，resume/评估 hash 校验天然防 r 串用）。
    """
    import config as _C
    _orig_build = _C.build_config

    def _build_with_rank(**overrides):
        if "bottleneck_rank" in overrides:
            raise KeyError("bottleneck_rank 与 --rank 冲突（r 覆写仅经 wrapper --rank 进入）")
        return _orig_build(bottleneck_rank=rank, **overrides)

    _C.build_config = _build_with_rank


def _effective_run_id(argv: list) -> Optional[str]:
    """生效 run_id：显式 --run-id；否则 --resume 派生（<run>/checkpoints/x.pt -> <run>）；均无则 None。"""
    args = argv[1:]
    if "--run-id" in args:
        return args[args.index("--run-id") + 1]
    if "--resume" in args:
        return Path(args[args.index("--resume") + 1]).resolve().parents[1].name
    return None


def _guard_run_id(argv: list, rank: Optional[int]) -> None:
    """run_id 前缀守卫（path-isolation）+ --rank 的 rXX 标识一致性校验（防 r 串用）。"""
    run_id = _effective_run_id(argv)
    if run_id is not None and not run_id.startswith(RUN_PREFIX):
        sys.exit(f"[v4a wrapper] --run-id 须以 '{RUN_PREFIX}' 开头（与其他变体 run 目录隔离），got: {run_id}")
    if run_id is None:
        argv.extend(["--run-id", f"{RUN_PREFIX}run_{time.strftime('%Y%m%d-%H%M%S')}"])
    elif rank is not None and f"r{rank}" not in run_id:
        # r 标识校验：显式 --rank R 时 run_id 须含 "rR"（v4a-r8-…/v4a-r16-…/v4a-r32-…）——
        # 防 run 目录名与实际结构错记；未传 --rank（config 默认 16）不强制（冒烟/任意命名合法）。
        sys.exit(f"[v4a wrapper] --rank {rank} 与 run_id 缺少 'r{rank}' 标识不一致（防 r 串用），got: {run_id}")


def _load_v1_train():
    """以独立模块名加载 v1 train.py（避免与本包装同名冲突；其 import config 命中 v4a 缓存）。"""
    spec = importlib.util.spec_from_file_location("gpfm_mamba_v1_train", str(_V1 / "train.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    rank = _pop_rank_arg(sys.argv)
    _shadow_v4a_config()
    if rank is not None:
        _apply_rank_override(rank)
    _guard_run_id(sys.argv, rank)
    mod = _load_v1_train()
    mod.main()


if __name__ == "__main__":
    main()
