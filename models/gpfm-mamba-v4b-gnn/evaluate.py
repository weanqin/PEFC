#!/usr/bin/env python3
"""evaluate.py — gpfm-mamba-v4b-gnn（v4b d_ch=1 基座 + GNN 先验图头）评估入口包装（薄变体）。

依赖方式（同 train.py 包装，config + model 双遮蔽）：v4bg config/model 占据 sys.modules 后，
以独立模块名加载 v1 evaluate.py（Pass A/B + GC + PSD + 空间归因 + 全部图）。v1 evaluate.py
内部 `from sample import …` 经 sys.path 命中 **v1 sample.py**（本目录无 sample.py，不遮蔽）；
sample.py 内部 `from model import EMA, VelocityNetwork` 再命中 v4bg 缓存——装载 v4bg 结构
（其无条件透传的 bottleneck_rank/norm_type/cond_adaln 三 kwargs 由 v4bg 模型按 §2.5 契约
接受并断言默认值）。checkpoint config_hash 严格校验（v4bg state_dict 键与 v1/v2/v4 完全
不同，天然防串用）。

run_id 前缀守卫：--run-id 必填且以 `v4bg-` 开头（防止默认 latest-run 逻辑误评其他变体 run）。

用法：
  python3 models/gpfm-mamba-v4b-gnn/evaluate.py --run-id v4bg-main-01
  python3 models/gpfm-mamba-v1/fc_metric.py --run-id v4bg-main-01   # 指标 1 v2.1（fc_metric
                                                                    # 硬编码共享 outputs 根 + run_id
                                                                    # 形参，天然兼容 v4bg run）
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

RUN_PREFIX = "v4bg-"

_HERE = Path(__file__).resolve().parent          # models/gpfm-mamba-v4b-gnn
_V1 = _HERE.parent / "gpfm-mamba-v1"             # 只读依赖的 v1 目录

_CONSISTENCY_FIELDS = (
    ("d_model", "d_model"),
    ("graph_kernel", "graph_kernel"),
    ("graph_k", "graph_k"),
    ("graph_hidden", "graph_hidden"),
    ("graph_gamma", "graph_gamma"),
    ("graph_gamma_m", "graph_gamma_m"),
    ("graph_w_init", "graph_w_init"),
    ("geometry_path", "geometry_path"),
    ("data_root", "data_root"),
)


def _shadow_v4bg() -> None:
    """v4bg 的 config 与 model 双遮蔽：占据 sys.path[0] 与 sys.modules（先于插入 v1 目录）。"""
    sys.path.insert(0, str(_V1))
    sys.path.insert(0, str(_HERE))
    import config as _C  # noqa: F401  -> 解析到本目录 config.py（sys.path[0]）
    import model as _M   # noqa: F401  -> 解析到本目录 model.py（sys.path[0]）
    if Path(_C.__file__).resolve().parent != _HERE:
        raise RuntimeError(f"config 遮蔽失败：import 到 {_C.__file__}（应为 {_HERE / 'config.py'}）")
    if Path(_M.__file__).resolve().parent != _HERE:
        raise RuntimeError(f"model 遮蔽失败：import 到 {_M.__file__}（应为 {_HERE / 'model.py'}）")
    import mamba as _mb  # noqa: F401  -> 本目录无 mamba.py，须解析到 v1（CI 骨干复用）
    if Path(_mb.__file__).resolve().parent != _V1:
        raise RuntimeError(f"mamba 解析异常：import 到 {_mb.__file__}（应为 {_V1 / 'mamba.py'}）")


def _assert_model_config_consistency() -> None:
    """d_model/graph_* 双处一致性防线（与 train.py 包装同款，design §2.5）。"""
    import inspect

    import config as _C
    import model as _M
    cfg = _C.Config()
    sig = inspect.signature(_M.VelocityNetwork.__init__)
    problems = []
    for cfg_key, param_name in _CONSISTENCY_FIELDS:
        p_def = sig.parameters[param_name].default
        c_val = getattr(cfg, cfg_key)
        if p_def != c_val:
            problems.append(f"{cfg_key}: model 类默认 {p_def!r} != cfg.{cfg_key} {c_val!r}")
    if problems:
        sys.exit("[v4bg wrapper] config/model 双处一致性断裂："
                 + "；".join(problems)
                 + "——graph_* 变更属口径变更（须用户确认），且须 config.py 与 model.py 双处同改并重跑全部单测")


def _guard_run_id(argv: list) -> None:
    args = argv[1:]
    if "--run-id" not in args:
        sys.exit("[v4bg wrapper] evaluate 必须显式传 --run-id（v4bg run 一律 v4bg- 前缀，"
                 "如 --run-id v4bg-main-01）")
    run_id = args[args.index("--run-id") + 1]
    if not run_id.startswith(RUN_PREFIX):
        sys.exit(f"[v4bg wrapper] --run-id 须以 '{RUN_PREFIX}' 开头，got: {run_id}")


def _load_v1_evaluate():
    """以独立模块名加载 v1 evaluate.py（避免与本包装同名冲突；其 import config/model 命中 v4bg 缓存）。"""
    spec = importlib.util.spec_from_file_location("gpfm_mamba_v1_evaluate", str(_V1 / "evaluate.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    _shadow_v4bg()
    _assert_model_config_consistency()
    _guard_run_id(sys.argv)
    mod = _load_v1_evaluate()
    mod.main()


if __name__ == "__main__":
    main()
