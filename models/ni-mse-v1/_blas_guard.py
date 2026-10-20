"""_blas_guard.py — BLAS 线程护栏（environment.md / perturb-ec-v1 惯例，任务 10-20-ni-mse-v5fmt-f0）。

全部入口 / 单测在 **numpy 导入之前** `import _blas_guard`：有效 16 核 cgroup 限 8 线程，
setdefault 允许外部环境变量覆盖。本模块自身只 import os，绝不触发 numpy/torch 装载。
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "8")
