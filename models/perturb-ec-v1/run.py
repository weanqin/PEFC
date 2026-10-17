#!/usr/bin/env python3
"""run.py — perturb-ec-v1 CLI 入口（design §1；B0 骨架：参数解析 / BLAS 护栏 / 入口结构
完整，各 stage 主体 B1+ 按 implement.md 逐阶段实现）。

用法（implement.md 验证命令速查）：
  python3 models/perturb-ec-v1/run.py --stage smoke
  python3 models/perturb-ec-v1/run.py --stage bench --subjects 1
  python3 models/perturb-ec-v1/run.py --stage readout / refs / split-half / compare / figs
  mkdir -p outputs/perturb-ec-f0/main-01/logs && nohup python3 run.py --stage main ...
"""

from __future__ import annotations

import os

# ---- BLAS 护栏（environment.md / _blas_guard 惯例：必须在 numpy/torch 导入之前；----
# ---- 有效 16 核 cgroup 限 8 线程，setdefault 允许外部环境变量覆盖）            ----
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse  # noqa: E402

STAGES = ("smoke", "bench", "main", "readout", "refs", "split-half", "compare", "figs")


def main() -> None:
    p = argparse.ArgumentParser(
        description="fold0 虚拟扰动→EC 有效路径推断（v4a-fmt 冻结代理脑，任务 10-17-perturb-ec-f0）")
    p.add_argument("--stage", required=True, choices=STAGES,
                   help="执行阶段（implement.md B1+：smoke=1 例 2 窗全链 / bench=N_w=32 计时 / "
                        "main=10 例扰动批采样 / readout=三 readout+有效路径 / refs=GC·gPDC 参考 / "
                        "split-half=fmt 天花板 / compare=比较器 / figs=诊断图")
    p.add_argument("--subjects", type=int, default=None,
                   help="例数上限（smoke/bench；main 默认 fold0 test 全 10 例）")
    p.add_argument("--run-id", type=str, default=None,
                   help="outputs/perturb-ec-f0/<run-id>/（experiment.md Run 块先登记后跑）")
    args = p.parse_args()
    print(f"[run] stage={args.stage} subjects={args.subjects} run_id={args.run_id}")
    print(f"[run] stage {args.stage!r} not implemented in B0（脚手架阶段；主体见 implement.md B1+）")


if __name__ == "__main__":
    main()
