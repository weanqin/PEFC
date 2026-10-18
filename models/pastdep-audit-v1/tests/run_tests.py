#!/usr/bin/env python3
"""run_tests.py — pastdep-audit-v1 单测启动器（①–⑥ 直调）。"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "8")
import sys
from pathlib import Path
_T = Path(__file__).resolve().parent
if str(_T) not in sys.path:
    sys.path.insert(0, str(_T))
import test_pastdep_audit as T

def main() -> None:
    T.test_01_operators()
    T.test_02_batch()
    T.test_03_crn()
    T.test_04_e3_mapping()
    T.test_05_loader()
    T.test_06_metrics()
    print("[PASS] pastdep-audit-v1 B0 单测全部通过（跳过段已注明）")

if __name__ == "__main__":
    main()
