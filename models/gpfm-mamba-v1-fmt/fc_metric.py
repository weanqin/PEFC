#!/usr/bin/env python3
"""fc_metric.py — gpfm-mamba-v1-fmt 指标 1（FC 矩阵一致性 v2.1）入口包装（薄变体，不复制逻辑）。

v1 fc_metric.py 无 config 依赖（独立读 eval/gc_input.npz），仅 outputs 根原为硬编码——
经 v1 fc_metric.py 的 `--outputs-root` default-off 参数传本任务新根
`/root/npi/outputs/gpfm-mamba-fmt-f0`（默认值 = 原硬编码路径，不传时行为零漂移）。

本包装做三件事：①v1 目录入 sys.path（gc_psd 依赖）；②run_id 前缀守卫 `v1fmt-`；
③向 argv 注入 `--outputs-root <本任务新根>` 后以独立模块名加载 v1 fc_metric.py 调其 main()。

用法：
  python3 models/gpfm-mamba-v1-fmt/fc_metric.py --run-id v1fmt-main-01
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
FMT_OUTPUTS_ROOT = "/root/npi/outputs/gpfm-mamba-fmt-f0"  # 本任务 outputs 根（与 v4a-fmt 共享，前缀隔离）

_HERE = Path(__file__).resolve().parent          # models/gpfm-mamba-v1-fmt
_V1 = _HERE.parent / "gpfm-mamba-v1"             # 只读依赖的 v1 目录


def _guard_run_id(argv: list) -> str:
    """--run-id 必填 + `v1fmt-` 前缀守卫（path-isolation）。"""
    args = argv[1:]
    if "--run-id" not in args:
        sys.exit("[v1fmt wrapper] fc_metric 必须显式传 --run-id（v1-fmt run 一律 v1fmt- 前缀，"
                 "如 --run-id v1fmt-main-01）")
    run_id = args[args.index("--run-id") + 1]
    if not run_id.startswith(RUN_PREFIX):
        sys.exit(f"[v1fmt wrapper] --run-id 须以 '{RUN_PREFIX}' 开头，got: {run_id}")
    return run_id


def _inject_outputs_root(argv: list) -> None:
    """向 argv 注入 --outputs-root 本任务新根（用户显式传了则以显式值优先，不覆盖）。"""
    if not any(a == "--outputs-root" or a.startswith("--outputs-root=") for a in argv[1:]):
        argv.extend(["--outputs-root", FMT_OUTPUTS_ROOT])


def _load_v1_fc_metric():
    """以独立模块名加载 v1 fc_metric.py（其 `from gc_psd import …` 经 sys.path 命中 v1）。"""
    sys.path.insert(0, str(_V1))
    spec = importlib.util.spec_from_file_location("gpfm_mamba_v1_fc_metric", str(_V1 / "fc_metric.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    _guard_run_id(sys.argv)
    _inject_outputs_root(sys.argv)
    mod = _load_v1_fc_metric()
    mod.main()


if __name__ == "__main__":
    main()
