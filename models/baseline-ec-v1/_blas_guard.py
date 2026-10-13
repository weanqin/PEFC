"""BLAS 线程护栏 —— 必须在任何 numpy/scipy 导入之前 import。

本机 cgroup 有效算力 16 核（cpu.max=1600000/100000），而 numpy/OpenBLAS 默认按可见
128 核起线程 → 线程超订把整机钉在 100% CPU（2026-10-13 冒烟实测）。统一限 8 线程
（顺序流水足够；setdefault 允许外部环境变量覆盖）。
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "8")
