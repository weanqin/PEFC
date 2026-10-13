"""test_split_half —— ①分段与两半形状 ②合成平稳系统的天花板 >0.5。
运行：python3 models/baseline-ec-v1/tests/test_split_half.py（退出码 0 = 通过）
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _blas_guard  # noqa: F401,E402  # 须在 numpy 之前：BLAS 线程护栏

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import split_half as sh  # noqa: E402


def gen_stationary(T=7500, C=3, seed=2026):
    """平稳耦合 AR(2)：0→1（lag2）。60 s @125 Hz。"""
    rng = np.random.default_rng(seed)
    A = np.zeros((C, C, 2))
    A[0, 0, 0], A[0, 0, 1] = 0.9, -0.7
    A[1, 1, 0], A[1, 1, 1] = 0.85, -0.6
    A[2, 2, 0], A[2, 2, 1] = 0.8, -0.5
    A[1, 0, 1] = 0.35
    X = np.zeros((T, C))
    for t in range(2, T):
        X[t] = A[:, :, 0] @ X[t - 1] + A[:, :, 1] @ X[t - 2] + 0.4 * rng.standard_normal(C)
    return X.T


def main():
    # ① 形状：T=5000, seg=250 → 20 段 → A/B 各 10 段
    x = gen_stationary(T=5000)
    h = sh.build_halves(x)
    assert h["A"].shape == (10, 3, 250) and h["B"].shape == (10, 3, 250), \
        (h["A"].shape, h["B"].shape)
    # 偶段内容核对：A[0] == 原第 0 段
    segs = x[:, :20 * 250].reshape(3, 20, 250).transpose(1, 0, 2)
    assert np.array_equal(h["A"][0], segs[0]) and np.array_equal(h["B"][0], segs[1])
    print("① 分段/两半形状与内容 OK")

    # ② 平稳系统天花板（p=2，与主口径同阶；λ_rel=1e-2 为 62ch 真实数据定标，
    #    3ch 合成系统受收缩略压——数据量加倍后应显著超阈）
    c = sh.ceiling_for_subject(gen_stationary(T=15000), p=2)
    assert c["gc"] > 0.5, f"gc 天花板过低: {c['gc']:.3f}"
    assert c["gpdc"] > 0.5, f"gpdc 天花板过低: {c['gpdc']:.3f}"
    print(f"② 合成平稳系统天花板 OK（gc={c['gc']:.3f}, gpdc={c['gpdc']:.3f}）")

    print("[PASS] test_split_half")


if __name__ == "__main__":
    main()
