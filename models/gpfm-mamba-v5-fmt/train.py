#!/usr/bin/env python3
"""train.py — gpfm-mamba-v5-fmt（v5 轴分解交替 × edra_fmt/250 Hz）训练入口包装
（薄变体，不复制训练逻辑；实现自 v5 train.py 对齐，任务 10-20-gpfm-mamba-v5-fmt-f0）。

依赖方式（v5 双遮蔽惯例）：本脚本只做四件事——
1. sys.path 令 **v5-fmt 的 config.py / model.py / graph.py 优先于 v1 的**（config+model 双遮蔽；
   graph 为 model 的依赖、随 sys.path[0] 命中 v5-fmt）：先占据 sys.modules["config"] 与
   sys.modules["model"]（解析路径自检断言命中 v5-fmt 目录），再以 importlib 加载 v1 train.py；
   其内部 `import config as C` / `from model import EMA, VelocityNetwork` 命中缓存即得 v5-fmt 版本
   ——**v5 交替结构（fmt 口径 LayerNorm）进入 v1 训练循环，v1 代码零改动**；mamba/data/gpr_prior/
   losses 等其余模块经 sys.path 复用 v1。fmt 训练循环先例核验：v1-fmt 无独立训练循环（其 train.py
   亦 importlib 加载 v1 train.py，仅 config 单遮蔽——norm_type/lr/早停经 v1 循环 getattr 钩子生效），
   故本包装链 = v5 双遮蔽 + fmt config 字段，钩子同点生效。
2. run_id 前缀守卫：v5-fmt 与 v1/v2/v4*/v5 共享任务 outputs 根 `/root/npi/outputs/gpfm-mamba-f0/`，
   **前缀 `v5-` 沿用**（PRD R1：v5-fmt-01 与 v5-main-01 同根并列；checkpoint 主名
   `gpfm-mamba-v5-fmt-{best,last}.pt` 与 v5 的 `gpfm-mamba-v5-{best,last}.pt` 天然隔离；
   --resume 派生的 run 名同样校验）。
3. **加载 v1 主流程前运行时断言**（config↔model 双处一致性防线，v4/v5 惯例）：model 类默认
   d_model/graph_* 与 cfg 对应字段一致，不一致即退出——v1 train.py 构建模型不透传 graph_*，
   此断言是 config 与 model 两处不漂移的唯一防线（state_dict 自洽时 resume/eval 不报警的静默风险）。
   norm_type 不入 pairs（v1 循环经 cfg.norm_type="layernorm" 直传构建——config 即生效源，钩子
   断言放行两值；单测另有正反例防线）。
4. 以独立模块名（gpfm_mamba_v1_train）importlib 加载 v1 train.py 并调用其 main()。

用法（与 v1 train.py 完全同名参数；主 Run 须用户确认后启动——R9 沿 v5-main 先例属执行期
用户确认项，本轮 S1 结构落地不跑任何训练）：
  python3 models/gpfm-mamba-v5-fmt/train.py --smoke --run-id v5-fmt-smoke-a
  python3 models/gpfm-mamba-v5-fmt/train.py --epochs 1 --bench --run-id v5-fmt-bench-1ep
  nohup python3 models/gpfm-mamba-v5-fmt/train.py --run-id v5-fmt-01 \
      > /root/npi/outputs/gpfm-mamba-f0/v5-fmt-01/logs/train.log 2>&1 &
"""

from __future__ import annotations

import os

# BLAS 线程上限（environment.md 惯例：有效 16 核 cgroup，须在 numpy 导入前设置；可覆盖）
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "8")

import importlib.util  # noqa: E402
import inspect  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

RUN_PREFIX = "v5-"

_HERE = Path(__file__).resolve().parent          # models/gpfm-mamba-v5-fmt
_V1 = _HERE.parent / "gpfm-mamba-v1"             # 只读依赖的 v1 目录


def _shadow_v5_fmt() -> None:
    """v5-fmt 的 config 与 model 双遮蔽：占据 sys.path[0] 与 sys.modules（先于插入 v1 目录）。"""
    sys.path.insert(0, str(_V1))
    sys.path.insert(0, str(_HERE))
    import config as _C  # noqa: F401  -> 解析到 v5-fmt/config.py（sys.path[0]）
    import model as _M   # noqa: F401  -> 解析到 v5-fmt/model.py（sys.path[0]；其 graph 依赖同路命中 v5-fmt）
    if Path(_C.__file__).resolve().parent != _HERE:
        raise RuntimeError(f"config 遮蔽失败：import 到 {_C.__file__}（应为 {_HERE / 'config.py'}）")
    if Path(_M.__file__).resolve().parent != _HERE:
        raise RuntimeError(f"model 遮蔽失败：import 到 {_M.__file__}（应为 {_HERE / 'model.py'}）")
    import graph as _G  # noqa: F401  -> v5-fmt 目录 graph.py（model 的 A_eff 机制依赖）
    if Path(_G.__file__).resolve().parent != _HERE:
        raise RuntimeError(f"graph 解析异常：import 到 {_G.__file__}（应为 {_HERE / 'graph.py'}）")
    import mamba as _mb  # noqa: F401  -> v5-fmt 目录无 mamba.py，须解析到 v1（时间块骨干复用）
    if Path(_mb.__file__).resolve().parent != _V1:
        raise RuntimeError(f"mamba 解析异常：import 到 {_mb.__file__}（应为 {_V1 / 'mamba.py'}）")


def _assert_model_config_consistency() -> None:
    """d_model/graph_* 双处一致性防线：model 类默认 == config 字段，不一致即退出（v4/v5 惯例）。

    v1 train.py 构建模型不透传 graph_*（钩子透传仅 bottleneck_rank/norm_type/cond_adaln；
    graph_* 由类默认承接、norm_type 经 cfg 直传——模型构造断言钉死合法值集）——graph 口径
    变更须 config.py 与 model.py 双处同改并过单测。
    """
    import config as _C
    import model as _M
    cfg = _C.Config()
    sig = inspect.signature(_M.VelocityNetwork.__init__)
    pairs = [("d_model", cfg.d_model), ("graph_kernel", cfg.graph_kernel),
             ("graph_gamma", cfg.graph_gamma), ("graph_gamma_m", cfg.graph_gamma_m),
             ("graph_w_init", cfg.graph_w_init), ("geometry_path", cfg.geometry_path)]
    problems = []
    for name, cfg_val in pairs:
        model_def = sig.parameters[name].default
        if model_def != cfg_val:
            problems.append(f"{name}: model 类默认 {model_def!r} != cfg {cfg_val!r}")
    if cfg.d_model != cfg.n_channels:
        problems.append(f"d_model {cfg.d_model} != n_channels {cfg.n_channels}（v5 轴分解不变量）")
    if problems:
        sys.exit("[v5-fmt wrapper] config/model 双处一致性断裂："
                 + "；".join(problems)
                 + "——口径变更属实验口径变更（须用户确认），且须 config.py 与 model.py 双处同改并重跑全部单测")


def _guard_run_id(argv: list) -> None:
    """run_id 前缀守卫（path-isolation）：v5-fmt run 目录一律 v5- 前缀（含 --resume 派生 run 名）。

    前缀 `v5-` 沿用（PRD R1）：v5-fmt-01 与 v5-main-01 同 outputs 根并列，隔离层 =
    checkpoint 主名（gpfm-mamba-v5-fmt-* vs gpfm-mamba-v5-*）。边界注记：前缀守卫不区分
    v5 与 v5-fmt 的 run 目录名——误传 --run-id v5-main-01 会向 v5 的 run 目录追加
    logs/checkpoint（ckpt 主名不同不覆盖，但 train.log 追加混写）——run_id 纪律由
    Run 块登记口径约束（本变体 run 一律 v5-fmt-* 命名）。
    """
    args = argv[1:]
    run_id = args[args.index("--run-id") + 1] if "--run-id" in args else None
    resume = args[args.index("--resume") + 1] if "--resume" in args else None
    if run_id is not None and not run_id.startswith(RUN_PREFIX):
        sys.exit(f"[v5-fmt wrapper] --run-id 须以 '{RUN_PREFIX}' 开头（与 v1/v2/v4* run 目录隔离），got: {run_id}")
    if run_id is None:
        if resume is not None:
            derived = Path(resume).resolve().parents[1].name  # <run>/checkpoints/x.pt -> <run>
            if not derived.startswith(RUN_PREFIX):
                sys.exit(f"[v5-fmt wrapper] --resume 所属 run 目录须以 '{RUN_PREFIX}' 开头，got: {derived}")
        else:
            argv.extend(["--run-id", f"{RUN_PREFIX}run_{time.strftime('%Y%m%d-%H%M%S')}"])


def _load_v1_train():
    """以独立模块名加载 v1 train.py（避免与本包装同名冲突；其 import config/model 命中 v5-fmt 缓存）。"""
    spec = importlib.util.spec_from_file_location("gpfm_mamba_v1_train", str(_V1 / "train.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    _shadow_v5_fmt()
    _assert_model_config_consistency()
    _guard_run_id(sys.argv)
    mod = _load_v1_train()
    mod.main()


if __name__ == "__main__":
    main()
