#!/usr/bin/env python3
"""run_tests.py — ec-omtime25-v1 单测启动器（implement.md B0 验证入口）。

用法：cd models/ec-omtime25-v1 && python3 tests/run_tests.py
（GPU 段需 CUDA；本机 base 无 pytest，沿既定模式直调测试函数；七项 = ①–⑦。）
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "8")

import sys
from pathlib import Path

_TESTS = Path(__file__).resolve().parent
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

import test_ec_omtime25 as T  # noqa: E402


def main() -> None:
    T.test_01_bias_operator()          # ① O 算子契约
    T.test_02_batch()                   # ② 批 497 行
    T.test_03_crn_determinism()         # ③ CRN 确定性（GPU）
    T.test_04_m_time25()              # ④ M-time@25 解析值
    T.test_05_readout()                 # ⑤ readout 合成全套
    T.test_06_loader_calibers()         # ⑥ loader 口径 + 窗枚举 0..79
    T.test_07_output_level_identity()   # ⑦ 输出级 r=0 恒等（GPU）
    print("[PASS] ec-omtime25-v1 B0 单测全部通过（跳过段已注明）")


if __name__ == "__main__":
    main()
