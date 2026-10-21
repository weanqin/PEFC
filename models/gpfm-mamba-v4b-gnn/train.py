#!/usr/bin/env python3
"""train.py — gpfm-mamba-v4b-gnn（v4b d_ch=1 基座 + GNN 先验图头）训练入口包装（薄变体，不复制逻辑）。

依赖方式（v4 双遮蔽惯例，design §1/§2.5）：本脚本只做五件事——
1. sys.path 令 **本目录 config.py 与 model.py 优先于 v1 的**（同名模块双遮蔽）：先占据
   sys.modules["config"] 与 sys.modules["model"]（解析路径自检断言命中本目录），再以
   importlib 加载 v1 train.py；其内部 `import config as C` / `from model import EMA,
   VelocityNetwork` 命中缓存即得 v4bg 版本——**v4bg 结构（CI trunk + GraphHead）进入 v1
   训练循环，v1 代码零改动**；mamba/data/gpr_prior/losses 等其余模块经 sys.path 复用 v1。
2. run_id 前缀守卫：与其他变体共享任务 outputs 根 `/root/npi/outputs/gpfm-mamba-f0/`，
   强制 run_id 以 `v4bg-` 开头（--resume 派生的 run 名同样校验），防止误写其他变体 run 目录。
3. **加载 v1 主流程前运行时断言**（d_model/graph_* 双处一致性防线，沿 v4 d_z/head_act 先例）：
   model 模块 VelocityNetwork 类默认 d_model 与 graph_* / geometry_path / data_root 与 cfg
   对应字段一致，不一致即退出——防 config 与 model 两处漂移。
4. **W 范数轨迹双口径仪表（B-5，R9 材料）**：以子类替换（仅本进程运行时，v1 文件零改动）
   v1 train.py 命名空间里的 `EMA` —— `update()` 在 super() 后追加记录 graph_head.W 的
   raw 范数与 EMA shadow 范数（同一 step 双口径；训练循环 per-step 调用 ⇒ epoch 内多点），
   run 结束落 `logs/w_norm_traj.jsonl`。checkpoint 的 "ema" 内容/格式不变（子类仅加记录）。
5. 以独立模块名（gpfm_mamba_v1_train）importlib 加载 v1 train.py 并调用其 main()。

用法（与 v1 train.py 完全同名参数）：
  python3 models/gpfm-mamba-v4b-gnn/train.py --smoke --run-id v4bg-smoke-a
  python3 models/gpfm-mamba-v4b-gnn/train.py --epochs 1 --bench --run-id v4bg-bench-1ep
  nohup python3 models/gpfm-mamba-v4b-gnn/train.py --run-id v4bg-main-01 \
      > /root/npi/outputs/gpfm-mamba-f0/v4bg-main-01/logs/train.log 2>&1 &
"""

from __future__ import annotations

import os

# BLAS 线程上限（environment.md 惯例：有效 16 核 cgroup，须在 numpy 导入前设置；可覆盖）
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "8")

import importlib.util  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402  # 仪表子类 @torch.no_grad 依赖（BLAS 守卫后导入）

RUN_PREFIX = "v4bg-"

_HERE = Path(__file__).resolve().parent          # models/gpfm-mamba-v4b-gnn
_V1 = _HERE.parent / "gpfm-mamba-v1"             # 只读依赖的 v1 目录

# ---- W 范数轨迹（B-5）：模块级记录缓冲，run 结束落盘 ----
_W_TRAJ: list = []


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


_CONSISTENCY_FIELDS = (
    # (config 字段, __init__ 形参名)：类默认须与 cfg 同值（d_z/head_act 先例延续）
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


def _assert_model_config_consistency() -> None:
    """d_model/graph_* 双处一致性防线：model 类默认 == config 字段，不一致即退出（design §2.5）。"""
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
    """run_id 前缀守卫（path-isolation）：v4bg run 目录一律 v4bg- 前缀。"""
    args = argv[1:]
    run_id = args[args.index("--run-id") + 1] if "--run-id" in args else None
    resume = args[args.index("--resume") + 1] if "--resume" in args else None
    if run_id is not None and not run_id.startswith(RUN_PREFIX):
        sys.exit(f"[v4bg wrapper] --run-id 须以 '{RUN_PREFIX}' 开头（与其他变体 run 目录隔离），got: {run_id}")
    if run_id is None:
        if resume is not None:
            derived = Path(resume).resolve().parents[1].name  # <run>/checkpoints/x.pt -> <run>
            if not derived.startswith(RUN_PREFIX):
                sys.exit(f"[v4bg wrapper] --resume 所属 run 目录须以 '{RUN_PREFIX}' 开头，got: {derived}")
        else:
            argv.extend(["--run-id", f"{RUN_PREFIX}run_{time.strftime('%Y%m%d-%H%M%S')}"])


def _load_v1_train():
    """以独立模块名加载 v1 train.py（避免与本包装同名冲突；其 import config/model 命中 v4bg 缓存）。"""
    spec = importlib.util.spec_from_file_location("gpfm_mamba_v1_train", str(_V1 / "train.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _instrument_wnorm(mod) -> None:
    """W 范数轨迹双口径仪表（B-5）：运行时以子类替换 mod.EMA（v1 train.py 文件零改动）。

    v1 训练循环每 step `ema.update(model)`（opt.step 之后）——子类在 super().update() 后记录
    graph_head.W 的 raw 范数（当前步后参数）与 EMA shadow 范数（decay·旧 + (1−decay)·新），
    同一 step 双口径；不改变任何训练数值/ RNG / checkpoint 内容（"ema" 键结构不变）。
    --resume 时轨迹自续跑点重新累计（只记本次进程步），Run 块照登。
    """
    import model as _M

    class _WNormEMA(_M.EMA):
        @torch.no_grad()
        def update(self, model) -> None:  # type: ignore[override]
            super().update(model)
            w_name = None
            w_param = None
            for name, p in model.named_parameters():
                if name == "W" or name.endswith(".W"):  # 全模型 "graph_head.W" / 裸 GraphHead "W"
                    w_name, w_param = name, p
                    break
            if w_name is None:
                raise RuntimeError("[v4bg wrapper] 未找到 graph_head.W（W 范数仪表失效——结构漂移？）")
            _W_TRAJ.append({
                "step": len(_W_TRAJ),
                "w_norm_raw": float(w_param.detach().norm()),
                "w_norm_ema": float(self.shadow[w_name].norm()),
            })

    mod.EMA = _WNormEMA


def _dump_wnorm_traj() -> None:
    """run 结束落盘 W 范数轨迹到 <outputs_root>/<run_id>/logs/w_norm_traj.jsonl + 摘要行。"""
    import config as _C
    args = sys.argv[1:]
    run_id = args[args.index("--run-id") + 1] if "--run-id" in args else None
    if run_id is None or not _W_TRAJ:
        return
    run_dir = Path(_C.Config().outputs_root) / run_id
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    path = run_dir / "logs" / "w_norm_traj.jsonl"
    with open(path, "w") as f:
        for rec in _W_TRAJ:
            f.write(json.dumps(rec) + "\n")
    first, last = _W_TRAJ[0], _W_TRAJ[-1]
    print(f"[wnorm] W 范数轨迹（raw/EMA 双口径，{len(_W_TRAJ)} 点）已写 {path} | "
          f"raw {first['w_norm_raw']:.6f} -> {last['w_norm_raw']:.6f} | "
          f"ema {first['w_norm_ema']:.6f} -> {last['w_norm_ema']:.6f}", flush=True)


def main() -> None:
    _shadow_v4bg()
    _assert_model_config_consistency()
    _guard_run_id(sys.argv)
    mod = _load_v1_train()
    _instrument_wnorm(mod)
    mod.main()
    _dump_wnorm_traj()


if __name__ == "__main__":
    main()
