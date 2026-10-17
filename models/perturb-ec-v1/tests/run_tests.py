#!/usr/bin/env python3
"""run_tests.py — perturb-ec-v1 单测启动器（implement.md B0 验证命令入口）。

用法：python3 models/perturb-ec-v1/tests/run_tests.py
（GPU 段需 CUDA——mamba_ssm kernel 必须 CUDA；无 CUDA 自动跳过并注明；
 BLAS 护栏须在本启动器 numpy/torch 导入之前生效，子进程经环境继承）
"""

from __future__ import annotations

import os

# ---- BLAS 护栏（environment.md / _blas_guard 惯例：numpy/torch 导入前；限 8 线程可覆盖）----
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "8")

import sys
from pathlib import Path

_TESTS = Path(__file__).resolve().parent
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

import test_perturb_ec as T  # noqa: E402


def main() -> None:
    T.test_crn_contract()    # ① CRN 恒等/确定性（GPU 段无 CUDA 自动跳过）
    T.test_calibers()        # ② 口径（hash/模块归属/posterior_mean/gen_lowpass）
    T.test_readout_math()    # ③ readout/response 合成数学（CPU）
    T.test_gpu_smoke()       # ckpt 装载 GPU 冒烟（1 batch + 497 变体并批）
    print("[PASS] perturb-ec-v1 B0 单测全部通过（跳过段已注明）")


if __name__ == "__main__":
    main()
