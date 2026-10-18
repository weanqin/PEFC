#!/usr/bin/env python3
"""run_tests.py — perturb-search-v1 单测启动器（implement.md B0 验证命令入口）。

用法：cd models/perturb-search-v1 && python3 tests/run_tests.py
（GPU 段需 CUDA——mamba_ssm kernel 必须 CUDA；无 CUDA 自动跳过并注明；
 BLAS 护栏须在本启动器 numpy/torch 导入之前生效，子进程经环境继承）。

注：本机 base 环境（environment.md 契约）无 pytest，沿 perturb-ec-v1/tests/run_tests.py
既定模式直调测试函数；九个测试项 = implement.md B0 ①–⑨。
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

import test_perturb_search as T  # noqa: E402


def main() -> None:
    T.test_01_r0_bitwise_identity()   # ① r=0 短路 clone 逐位恒等（六算子）
    T.test_02_only_source_channel_touched()   # ② 只动源通道 + 波形事实
    T.test_03_crn_determinism()       # ③ 显式 eps/eta 两次调用逐位相同（GPU）
    T.test_04_metrics_synthetic()     # ④ 度量合成 sanity（解析值/恒等 0/null 同构）
    T.test_05_batch_index_map()       # ⑤ 批构造行索引/逆映射（559/1737，r 网格化）
    T.test_06_loader_calibers()       # ⑥ gen_lowpass None + config_hash + 窗枚举
    T.test_07_output_level_identity() # ⑦ 输出级 r=0 同批逐位恒等（GPU）
    T.test_08_gate_d_monotonicity()   # ⑧ Gate D 单调性（网格推导 + 判定器合成数据）
    T.test_09_gate_b_spearman_merged_pool()   # ⑨ Gate B 合并池 Spearman 平铺配对回归
    print("[PASS] perturb-search-v1 B0 单测全部通过（跳过段已注明）")


if __name__ == "__main__":
    main()
