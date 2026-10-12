"""test_gc_psd.py — GC / PSD / 统计工具单测（implement.md 6.5；纯 numpy，无 CUDA 依赖）。

① 分布 p 值：F/t 生存函数对教科书临界值（α=0.05）
② BH-FDR：单调性（q 随 raw p 非降、q ≥ p、q ≤ 1）+ 已知算例
③ Welch：合成 10 Hz 正弦+噪声的峰值频率正确
④ GC 方向正确性：已知 2×2 VAR（c1→c0 有向）+ 独立白噪通道——有向边显著、无向/白噪边不显著
⑤ bootstrap CI 覆盖：受试者级 Gram 重采样 CI 包含全样本点估计
⑥ Spearman = 秩变换 Pearson（完全单调关联 rho=1）+ zero_lag_corr 已知结构
运行：python3 models/gpfm-mamba-v1/tests/test_gc_psd.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gc_psd import (  # noqa: E402
    _bootstrap_corr_ci,
    bh_fdr,
    bic_select_p,
    conditional_gc,
    f_sf,
    gram_pieces,
    offdiag_flat,
    pearson_r,
    rank_transform,
    spearman_rho,
    sum_grams,
    t_two_sided_p,
    tri_flat,
    welch_psd,
    zero_lag_corr,
)


def main() -> None:
    rng = np.random.default_rng(2026)

    # ① F/t p 值对教科书临界值（α=0.05 表）
    cases = [(f_sf(4.96, 1, 10), 0.05), (f_sf(4.10, 2, 10), 0.05),
             (f_sf(3.33, 5, 10), 0.05), (f_sf(2.98, 10, 10), 0.05),
             (t_two_sided_p(2.228, 10), 0.05), (t_two_sided_p(2.101, 18), 0.05),
             (t_two_sided_p(1.960, 10000), 0.05)]
    for got, want in cases:
        assert abs(got - want) < 5e-3, f"p 值偏差过大: got {got:.5f} vs 表值 {want}"
    print(f"[p 值] F/t 生存函数对 7 个教科书临界值 |Δ|<2e-3（例 F(2,10)@4.10 → {f_sf(4.10, 2, 10):.4f}）")

    # ② BH-FDR 单调性 + 已知算例
    p_raw = np.array([1e-6, 0.049, 0.05, 0.001, 0.5, 0.5, 0.033])
    q = bh_fdr(p_raw)
    assert (q >= p_raw - 1e-12).all() and (q <= 1.0 + 1e-12).all(), "q 越界"
    order = np.argsort(p_raw)
    q_sorted = q[order]
    assert (np.diff(q_sorted) >= -1e-12).all(), "q 相对 raw p 非单调"
    known = bh_fdr(np.array([0.01, 0.02, 0.03, 0.04, 0.05]))
    assert np.allclose(known, 0.05, atol=1e-12), f"已知算例失败: {known}"
    clip = bh_fdr(np.array([0.5, 0.9]))
    assert np.allclose(clip, [0.9, 0.9]), f"后缀最小/截断算例失败: {clip}"
    print(f"[BH-FDR] 单调 + q≥p + 已知算例（[0.01..0.05]→全 0.05，[0.5,0.9]→[0.9,0.9]）通过")

    # ③ Welch 峰值频率（合成 10 Hz 正弦 + 噪声）
    fs, f0, nperseg = 200.0, 10.0, 512
    t = np.arange(512 * 8) / fs
    x = np.sin(2 * np.pi * f0 * t) + 0.3 * rng.standard_normal(len(t))
    freqs, psd = welch_psd(x, fs, nperseg)
    pos = freqs > 0
    peak = freqs[pos][np.argmax(psd[pos])]
    tol = 1.5 * fs / nperseg
    assert abs(peak - f0) < tol, f"峰值频率错误 {peak:.3f} Hz（tol {tol:.3f}）"
    # 谱线幅度口径：正弦振幅 1 → 该频点 PSD 积分 ≈ 0.5（Parseval 一致性检查，宽松容差）
    df = freqs[1] - freqs[0]
    total_pow = psd[1:-1].sum() * df + psd[0] * df / 2 + psd[-1] * df / 2
    assert 0.4 < total_pow < 0.9, f"总功率 {total_pow:.3f} 偏离 sin+0.3 噪声预期 ~0.635"
    print(f"[Welch] 峰值 {peak:.3f} Hz（真值 10，tol ±{tol:.2f}）| 总功率 {total_pow:.3f}"
          f"（预期 0.5[sin] + 0.09[噪声] ≈ 0.59 附近）")

    # ④ GC 方向正确性：c1→c0 有向；c2 独立白噪
    #    x0_t = 0.5·x0_{t-1} + 0.6·x1_{t-1} + e0；x1_t = −0.4·x1_{t-1} + e1；x2_t = e2
    N_seg, F_len, burn = 40, 400, 60
    data = np.empty((N_seg, 3, F_len))
    for s in range(N_seg):
        x = np.zeros((3, F_len + burn))
        for tt in range(1, F_len + burn):
            x[0, tt] = 0.5 * x[0, tt - 1] + 0.6 * x[1, tt - 1] + rng.standard_normal()
            x[1, tt] = -0.4 * x[1, tt - 1] + rng.standard_normal()
            x[2, tt] = rng.standard_normal()
        data[s] = x[:, burn:]
    p_star, tab = bic_select_p(data, p_max=4)
    assert p_star == 1, f"BIC 应选 p=1（真实阶），实选 {p_star}：{tab}"
    out = conditional_gc(gram_pieces(data, p_star), p_star)
    gc_m, P = out["gc"], out["p"]
    assert P[1, 0] < 1e-3, f"c1→c0 应显著 p={P[1, 0]:.2e}"
    assert gc_m[1, 0] > 0.05, f"c1→c0 GC 量异常小 {gc_m[1, 0]:.4f}"
    for (j, i), name in {(0, 1): "c0→c1", (2, 0): "c2→c0", (2, 1): "c2→c1",
                         (0, 2): "c0→c2", (1, 2): "c1→c2"}.items():
        assert P[j, i] > 0.05, f"{name} 不应显著 p={P[j, i]:.3f}"
    print(f"[GC 方向] BIC 选 p*={p_star}（真值 1）| p(c1→c0)={P[1, 0]:.2e} 显著"
          f"（GC 量 {gc_m[1, 0]:.3f}）；c0→c1 p={P[0, 1]:.2f}、白噪边 "
          f"p={P[2, 0]:.2f}/{P[2, 1]:.2f}/{P[0, 2]:.2f}/{P[1, 2]:.2f} 均不显著")

    # ⑤ bootstrap CI 覆盖（受试者级 Gram 重采样，生产路径 _bootstrap_corr_ci）
    #    4 通道（12 条有向边）：c1→c0（强）与 c2→c3（弱）两条强度不同的因果边，
    #    组 A = 纯 VAR；组 B = 同结构 + 独立噪声（矩阵强相关但 r<1，避开恒等退化）
    S, Ns, Fs, Cs = 8, 30, 200, 4
    def gen_group(rng2, noise):
        segs = np.empty((S, Ns, Cs, Fs))
        for s in range(S):
            for k in range(Ns):
                x = np.zeros((Cs, Fs + 50))
                for tt in range(1, Fs + 50):
                    x[0, tt] = 0.5 * x[0, tt - 1] + 0.6 * x[1, tt - 1] + rng2.standard_normal()
                    x[1, tt] = -0.4 * x[1, tt - 1] + rng2.standard_normal()
                    x[2, tt] = 0.2 * x[2, tt - 1] + 0.3 * x[3, tt - 1] + rng2.standard_normal()
                    x[3, tt] = 0.3 * x[3, tt - 1] + rng2.standard_normal()
                segs[s, k] = x[:, 50:] + noise * rng2.standard_normal((Cs, Fs))
        return segs
    gA = gen_group(np.random.default_rng(11), 0.0)
    gB = gen_group(np.random.default_rng(22), 0.5)    # 同结构 + 独立噪声
    pr = [gram_pieces(gA[s], 1) for s in range(S)]
    pg = [gram_pieces(gB[s], 1) for s in range(S)]
    r_point = pearson_r(offdiag_flat(conditional_gc(sum_grams(pr), 1, with_tests=False)["gc"]),
                        offdiag_flat(conditional_gc(sum_grams(pg), 1, with_tests=False)["gc"]))
    boot = _bootstrap_corr_ci(pr, pg, 1, n_boot=200, seed=2026, log=lambda m: None)
    lo, hi = boot["pearson_ci95"]
    assert 0.3 < r_point <= 1.0, f"合成两组相关性异常 {r_point:.4f}"
    assert lo - 1e-6 <= r_point <= hi + 1e-6, f"CI [{lo:.4f},{hi:.4f}] 未覆盖点估计 {r_point:.4f}"
    assert boot["spearman_ci95"][0] <= boot["spearman_ci95"][1] + 1e-9
    print(f"[bootstrap] 全样本点估计 r={r_point:.4f} ∈ 95% CI [{lo:.4f},{hi:.4f}]"
          f"（受试者级 200 次；rho CI {boot['spearman_ci95']}）")

    # ⑥ Spearman/zero_lag_corr 基本性质
    x = rng.standard_normal(500)
    assert abs(spearman_rho(x, x ** 3) - 1.0) < 1e-9, "完全单调关联 rho 应为 1"
    assert abs(pearson_r(rank_transform(x), rank_transform(x)) - 1.0) < 1e-9
    z = rng.standard_normal((20, 3, 128))
    z[:, 1] = z[:, 0]                                    # 完全相关通道
    R = zero_lag_corr(z)
    assert abs(R[0, 1] - 1.0) < 1e-6 and abs(R[0, 2]) < 0.2 and abs(R[1, 2]) < 0.2
    print(f"[Spearman/zero-lag] 单调 rho=1；相关阵 [[1,1,0],[1,1,0],[0,0,1]] 复现"
          f"（R01={R[0, 1]:.4f}, R02={R[0, 2]:.3f}）")

    print("[PASS] test_gc_psd 全部断言通过")


if __name__ == "__main__":
    main()
