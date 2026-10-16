#!/usr/bin/env python3
"""train.py — gpfm-mamba-v4（v4-B：CI 骨干 + cMLP 读出）训练入口包装（薄变体，不复制逻辑）。

依赖方式（design §4.2，v2 单遮蔽扩展为 **config + model 双遮蔽**）：本脚本只做四件事——
1. sys.path 令 **v4 的 config.py 与 model.py 优先于 v1 的**（同名模块双遮蔽）：先占据
   sys.modules["config"] 与 sys.modules["model"]（解析路径自检断言命中 v4 目录），再以
   importlib 加载 v1 train.py；其内部 `import config as C` / `from model import EMA,
   VelocityNetwork` 命中缓存即得 v4 版本——**v4 model.py 的新结构（CI+head）进入 v1 训练
   循环，v1 代码零改动**；mamba/data/gpr_prior/losses 等其余模块经 sys.path 复用 v1。
2. run_id 前缀守卫：v4 与 v1/v2 共享任务 outputs 根 `/root/npi/outputs/gpfm-mamba-f0/`，
   强制 run_id 以 `v4b-` 开头（--resume 派生的 run 名同样校验），防止误写其他变体 run 目录。
3. **加载 v1 主流程前运行时断言**（d_z/d_ch 双处一致性防线，design §4.3）：model 模块
   VelocityNetwork 类默认 d_z/head_act 与 cfg 对应字段一致，不一致即退出——防 config 与
   model 两处漂移（state_dict 形状自洽时 resume/eval 均不报警的静默风险）。
4. 以独立模块名（gpfm_mamba_v1_train）importlib 加载 v1 train.py 并调用其 main()。

用法（与 v1 train.py 完全同名参数）：
  python3 models/gpfm-mamba-v4/train.py --smoke --run-id v4b-smoke-a
  python3 models/gpfm-mamba-v4/train.py --epochs 1 --bench --run-id v4b-bench-1ep
  nohup python3 models/gpfm-mamba-v4/train.py --run-id v4b-main-01 \
      > /root/npi/outputs/gpfm-mamba-f0/v4b-main-01/logs/train.log 2>&1 &
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

RUN_PREFIX = "v4b-"

_HERE = Path(__file__).resolve().parent          # models/gpfm-mamba-v4
_V1 = _HERE.parent / "gpfm-mamba-v1"             # 只读依赖的 v1 目录


def _shadow_v4() -> None:
    """v4 的 config 与 model 双遮蔽：占据 sys.path[0] 与 sys.modules（先于插入 v1 目录）。"""
    sys.path.insert(0, str(_V1))
    sys.path.insert(0, str(_HERE))
    import config as _C  # noqa: F401  -> 解析到 v4/config.py（sys.path[0]）
    import model as _M   # noqa: F401  -> 解析到 v4/model.py（sys.path[0]）
    if Path(_C.__file__).resolve().parent != _HERE:
        raise RuntimeError(f"config 遮蔽失败：import 到 {_C.__file__}（应为 {_HERE / 'config.py'}）")
    if Path(_M.__file__).resolve().parent != _HERE:
        raise RuntimeError(f"model 遮蔽失败：import 到 {_M.__file__}（应为 {_HERE / 'model.py'}）")
    import mamba as _mb  # noqa: F401  -> v4 目录无 mamba.py，须解析到 v1（CI 骨干复用）
    if Path(_mb.__file__).resolve().parent != _V1:
        raise RuntimeError(f"mamba 解析异常：import 到 {_mb.__file__}（应为 {_V1 / 'mamba.py'}）")


def _assert_model_config_consistency() -> None:
    """d_z/head_act 双处一致性防线：model 类默认 == config 字段，不一致即退出（design §4.3）。"""
    import config as _C
    import model as _M
    cfg = _C.Config()
    sig = inspect.signature(_M.VelocityNetwork.__init__)
    d_z_def = sig.parameters["d_z"].default
    act_def = sig.parameters["head_act"].default
    dm_def = sig.parameters["d_model"].default
    problems = []
    if d_z_def != cfg.d_z:
        problems.append(f"d_z: model 类默认 {d_z_def} != cfg.d_z {cfg.d_z}")
    if act_def != cfg.head_act:
        problems.append(f"head_act: model 类默认 {act_def!r} != cfg.head_act {cfg.head_act!r}")
    if dm_def != cfg.d_model:
        problems.append(f"d_model(d_ch): model 类默认 {dm_def} != cfg.d_model {cfg.d_model}")
    if problems:
        sys.exit("[v4 wrapper] config/model 双处一致性断裂："
                 + "；".join(problems)
                 + "——d_z/d_ch 变更属口径变更（须用户确认），且须 config.py 与 model.py 双处同改并重跑全部单测")


def _guard_run_id(argv: list) -> None:
    """run_id 前缀守卫（path-isolation）：v4 run 目录一律 v4b- 前缀。"""
    args = argv[1:]
    run_id = args[args.index("--run-id") + 1] if "--run-id" in args else None
    resume = args[args.index("--resume") + 1] if "--resume" in args else None
    if run_id is not None and not run_id.startswith(RUN_PREFIX):
        sys.exit(f"[v4 wrapper] --run-id 须以 '{RUN_PREFIX}' 开头（与 v1/v2 run 目录隔离），got: {run_id}")
    if run_id is None:
        if resume is not None:
            derived = Path(resume).resolve().parents[1].name  # <run>/checkpoints/x.pt -> <run>
            if not derived.startswith(RUN_PREFIX):
                sys.exit(f"[v4 wrapper] --resume 所属 run 目录须以 '{RUN_PREFIX}' 开头，got: {derived}")
        else:
            argv.extend(["--run-id", f"{RUN_PREFIX}run_{time.strftime('%Y%m%d-%H%M%S')}"])


def _load_v1_train():
    """以独立模块名加载 v1 train.py（避免与本包装同名冲突；其 import config/model 命中 v4 缓存）。"""
    spec = importlib.util.spec_from_file_location("gpfm_mamba_v1_train", str(_V1 / "train.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    _shadow_v4()
    _assert_model_config_consistency()
    _guard_run_id(sys.argv)
    mod = _load_v1_train()
    mod.main()


if __name__ == "__main__":
    main()
