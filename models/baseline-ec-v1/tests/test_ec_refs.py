"""test_ec_refs —— 合成已知方向系统上的参考读出正确性。

①条件 GC 方向（Schur 主口径，行=源）②gPDC² 值域/行和=1（源归一）
③gPDC 方向 ④BIC 恢复真阶附近 ⑤mne-c 二元交叉核对方向一致 ⑥残差白性护栏。
运行：python3 models/baseline-ec-v1/tests/test_ec_refs.py（退出码 0 = 通过）
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _blas_guard  # noqa: F401,E402  # 须在 numpy 之前：BLAS 线程护栏

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ec_refs  # noqa: E402


def gen_3node(T=30000, seed=2026, noise=0.4):
    """0→1（lag2 单向耦合）+ 节点2 独立的 3 通道 AR(2)。"""
    rng = np.random.default_rng(seed)
    A = np.zeros((3, 3, 2))
    A[0, 0, 0], A[0, 0, 1] = 0.9, -0.7
    A[1, 1, 0], A[1, 1, 1] = 0.85, -0.6
    A[2, 2, 0], A[2, 2, 1] = 0.8, -0.5
    A[1, 0, 1] = 0.35                     # 0 → 1 @ lag2
    X = np.zeros((T, 3))
    for t in range(2, T):
        X[t] = A[:, :, 0] @ X[t - 1] + A[:, :, 1] @ X[t - 2] + noise * rng.standard_normal(3)
    return X.T                             # (C, T)


def main():
    X = gen_3node()

    # ① 条件 GC 方向（行=源）：gc[0,1] 显著大于反向与背景
    gram = ec_refs.fit_gram(X, 2)
    gc = ec_refs.conditional_gc_matrix(gram, 2)
    off = gc[~np.eye(3, dtype=bool)]
    assert gc[0, 1] > gc[1, 0], f"方向反了: gc01={gc[0,1]:.4f} gc10={gc[1,0]:.4f}"
    assert gc[0, 1] > 3 * np.median(off), f"耦合边不突出: {gc[0,1]:.4f} vs med={np.median(off):.4f}"
    assert gc[0, 1] > 0.05, f"gc01 过小 {gc[0,1]:.4f}"
    print(f"① 条件GC方向 OK（gc[0,1]={gc[0,1]:.4f}, gc[1,0]={gc[1,0]:.4f}）")

    # ②③ gPDC：|gPDC|² 值域 [0,1]、源行和=1（列=源归一 → 行=源存储后行和=1）、方向
    co = ec_refs.fit_coeffs(gram, 2)
    gp = ec_refs.gpdc_squared(co["B"], co["Sigma"])
    bb = gp["broadband"]
    assert bb.min() >= -1e-9 and bb.max() <= 1 + 1e-6, f"值域越界 [{bb.min():.2e},{bb.max():.4f}]"
    assert np.allclose(bb.sum(axis=1), 1.0, atol=1e-6), \
        f"源行和≠1: {bb.sum(axis=1) - 1}"
    assert bb[0, 1] > bb[1, 0], f"gPDC 方向反了: {bb[0,1]:.4f} vs {bb[1,0]:.4f}"
    offb = bb[~np.eye(3, dtype=bool)]
    assert bb[0, 1] > np.quantile(offb, 0.75), "耦合边未进前 25%"
    print(f"②③ gPDC 值域/行和/方向 OK（bb[0,1]={bb[0,1]:.4f}, bb[1,0]={bb[1,0]:.4f}, "
          f"max={bb.max():.4f}）")

    # ④ BIC 恢复真阶附近（真阶 2；允许 ≤5 的过选）
    r = ec_refs.bic_pass(gen_3node(T=60000, seed=7))
    assert 2 <= r["p_star"] <= 5, f"BIC 选阶异常 p*={r['p_star']}"
    print(f"④ BIC 选阶 OK（p*={r['p_star']}，真阶 2）")

    # ⑤ mne-c 二元交叉核对：方向一致（seed→target=行=源）
    M = ec_refs.mnec_gc_matrix(X, 2)
    offm = M[~np.eye(3, dtype=bool)]
    assert M[0, 1] > M[1, 0], f"mne-c 方向反了: {M[0,1]:.4f} vs {M[1,0]:.4f}"
    assert M[0, 1] > np.median(offm), "mne-c 耦合边不突出"
    print(f"⑤ mne-c 交叉核对方向 OK（M[0,1]={M[0,1]:.4f}）")

    # ⑥ 残差白性护栏：白性成立时 lb_frac 应低
    frac = ec_refs.ljung_box_frac(X, 2, co["B"])
    assert frac <= 0.34, f"残差白性异常（p<0.01 通道占比 {frac:.2f}）"
    print(f"⑥ 残差白性 OK（lb_frac={frac:.2f}）")

    print("[PASS] test_ec_refs")


if __name__ == "__main__":
    main()
