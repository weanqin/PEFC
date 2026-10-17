#!/usr/bin/env python3
"""train.py — gpfm-mamba-v4a-fmt-adaln（5 norm 位 adaLN(t) scale/shift）训练入口包装（薄变体，不复制逻辑）。

依赖方式（v4a-fmt 式**单遮蔽**——本目录无 model.py，模型经 v1 model.py/mamba.py 的
cond_adaln default-off 钩子构建）：本脚本只做三件事——
1. sys.path 令 **本变体 config.py 优先于 v1 的**（同名模块遮蔽）：先 import 本目录 config
   并占据 sys.modules["config"]，再加载 v1 train.py；其内部 `import config as C` 命中缓存即得
   本变体 config（data_root=fmt、period_p=25、norm_type="layernorm"、cond_adaln="scale_shift"、
   bottleneck_rank=16、d_model=62、variant_name=gpfm-mamba-v4a-fmt-adaln）——train 循环本体
   全部只读复用 v1。**无 --rank**（r=16 固定，wrapper 比 v4a-fmt 简一档）。
2. run_id 前缀守卫：本变体独占本任务 outputs 根 `/root/npi/outputs/gpfm-mamba-adaln-f0/`，
   强制 run_id 以 `v4afmt-adaln-` 开头（--resume 派生的 run 名同样校验）。**`v4afmt-adaln-*`
   归本任务根**；已知局限（design §1.2 登记）：本前缀嵌套于 v4a-fmt 的 `v4afmt-` 之下——若误经
   v4a-fmt wrapper 传 adaln run_id 其守卫放行，但晚失败兜底齐全（独立 outputs 根
   FileNotFoundError + ckpt 名不同 + config_hash 严格校验，非静默）。
3. 以独立模块名（gpfm_mamba_v1_train）importlib 加载 v1 train.py 并调用其 main()。

用法（与 v1 train.py 完全同名参数）：
  python3 models/gpfm-mamba-v4a-fmt-adaln/train.py --smoke --run-id v4afmt-adaln-smoke-01
  python3 models/gpfm-mamba-v4a-fmt-adaln/train.py --epochs 1 --bench --run-id v4afmt-adaln-bench-1ep
  python3 models/gpfm-mamba-v4a-fmt-adaln/train.py --run-id v4afmt-adaln-main-01
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

RUN_PREFIX = "v4afmt-adaln-"  # 本任务根唯一前缀（嵌套于 v4afmt- 之下的已知局限见 docstring）

_HERE = Path(__file__).resolve().parent          # models/gpfm-mamba-v4a-fmt-adaln
_V1 = _HERE.parent / "gpfm-mamba-v1"             # 只读依赖的 v1 目录


def _shadow_adaln_config() -> None:
    """本变体 config 单遮蔽：占据 sys.path[0] 与 sys.modules['config']（先于插入 v1 目录）。"""
    sys.path.insert(0, str(_V1))
    sys.path.insert(0, str(_HERE))
    import config as _C  # noqa: F401  -> 解析到本变体 config.py（sys.path[0]）
    if Path(_C.__file__).resolve().parent != _HERE:
        raise RuntimeError(f"config 遮蔽失败：import 到 {_C.__file__}（应为 {_HERE / 'config.py'}）")
    import model as _M  # noqa: F401  -> 本目录无 model.py，须解析到 v1（cond_adaln 钩子）
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
    """run_id 前缀守卫（path-isolation）：强制 `v4afmt-adaln-` 前缀（--resume 派生名同检）。"""
    run_id = _effective_run_id(argv)
    if run_id is not None and not run_id.startswith(RUN_PREFIX):
        sys.exit(f"[v4afmt-adaln wrapper] --run-id 须以 '{RUN_PREFIX}' 开头（v4afmt-adaln-* 归本任务根"
                 f" /root/npi/outputs/gpfm-mamba-adaln-f0，与其他变体 run 目录隔离），got: {run_id}")
    if run_id is None:
        argv.extend(["--run-id", f"{RUN_PREFIX}run_{time.strftime('%Y%m%d-%H%M%S')}"])


def _load_v1_train():
    """以独立模块名加载 v1 train.py（避免与本包装同名冲突；其 import config 命中本变体缓存）。"""
    spec = importlib.util.spec_from_file_location("gpfm_mamba_v1_train", str(_V1 / "train.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    _shadow_adaln_config()
    _guard_run_id(sys.argv)
    mod = _load_v1_train()
    mod.main()


if __name__ == "__main__":
    main()
