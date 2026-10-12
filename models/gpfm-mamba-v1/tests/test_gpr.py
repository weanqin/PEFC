"""test_gpr.py — 时空可分 GPR 条件先验单测（design §2.1 / implement.md 2.2）。

① A 与 numpy 直接求解一致（rtol 1e-5）
② 可分性：K_s=I 时与逐通道实现逐元素一致；K_s≠I 时 A 不变（K_s 消去）
③ 采样经验协方差 ≈ K_s ⊗ K'_{t,f|p}（边缘方差 / 随机元素对 / 通道相关 / 时间相关）
④ K_s 真实电极坐标 PSD（σ∈{0.045,0.080,0.137}，min_eig>−1e-8，单位对角，+jitter Cholesky 成功）
⑤ μ 随 past 线性
运行：python3 models/gpfm-mamba-v1/tests/test_gpr.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as C  # noqa: E402
from gpr_prior import GPRPrior, ricker3d_kernel, time_kernel  # noqa: E402


def build(cfg, spatial="ricker", sigma_s=None) -> GPRPrior:
    coords = np.load(f"{cfg.data_root}/chan_coords_62.npy")
    return GPRPrior(
        coords,
        past_len=cfg.past_len,
        future_len=cfg.future_len,
        kernel_t=cfg.kernel_t,
        gamma=cfg.gamma,
        period_p=cfg.period_p,
        iso=cfg.iso,
        pp_jitter=cfg.pp_jitter,
        sigma_s=cfg.sigma_s if sigma_s is None else sigma_s,
        jitter_s=cfg.jitter_s,
        spatial=spatial,
    )


def main() -> None:
    cfg = C.Config()
    torch.manual_seed(7)
    prior = build(cfg)
    P, F = cfg.past_len, cfg.future_len
    T = P + F

    # ① A vs numpy 独立直接求解
    k = time_kernel(cfg.kernel_t, cfg.gamma)
    tau = np.arange(T, dtype=np.float64) * (np.pi / cfg.period_p)
    Kn = k(np.abs(tau[:, None] - tau[None, :])) + cfg.iso * np.eye(T)
    ppn = Kn[:P, :P] + cfg.pp_jitter * np.eye(P)
    A_np = np.linalg.solve(ppn, Kn[:P, P:]).T  # (K'_fp inv(K'_pp))ᵀ = inv(K'_pp) K'_pf 的转置
    A_got = prior.A.double().numpy()
    denom = np.abs(A_np).max()
    assert np.abs(A_got - A_np).max() < 1e-5 * denom, f"A 与 numpy 直接解不一致: {np.abs(A_got - A_np).max():.3e}"
    print(f"[A] max|Δ|/max|A| = {np.abs(A_got - A_np).max() / denom:.2e}（rtol 1e-5 内）")

    # 后验时间协方差 numpy 独立对照
    Kp_np = Kn[P:, P:] - Kn[P:, :P] @ np.linalg.inv(ppn) @ Kn[:P, P:]
    assert np.allclose(prior.K_post_t, Kp_np, rtol=1e-6, atol=1e-8), "K'_{t,f|p} 与 numpy 对照不一致"
    post_std = np.sqrt(np.diag(prior.K_post_t))
    print(f"[K_post] post_std mean={post_std.mean():.4f} min={post_std.min():.4f} max={post_std.max():.4f}")

    # ② 可分性
    prior_i = build(cfg, spatial="identity")
    assert torch.equal(prior.A, prior_i.A), "K_s≠I 时 A 必须不变（K_s 在 Σ_fp Σ_pp⁻¹ 中消去）"
    B = 5
    xp = torch.randn(B, 62, P, dtype=torch.float64)
    eps = torch.randn(B, 62, F, dtype=torch.float64)
    mu = torch.einsum("bcp,fp->bcf", xp, prior_i.A.double())
    per_channel = mu + eps @ prior_i.L_t.double().T  # 逐通道实现：μ + ε L_tᵀ
    got_i = prior_i.double().sample(xp, eps=eps)
    assert torch.allclose(got_i, per_channel, atol=1e-10), "K_s=I 时与逐通道实现不一致"
    got_r = prior.double().sample(xp, eps=eps)
    assert not torch.allclose(got_r, got_i), "K_s≠I 采样应不同于逐通道（跨通道耦合生效检查）"
    print("[可分性] K_s=I 与逐通道实现逐元素一致；K_s≠I 时 A 严格不变")

    # ③ 经验协方差 ≈ K_s ⊗ K'_{t,f|p}（N=4000，float64，μ=0）
    N = 4000
    pr = prior.double()
    x0 = pr.sample(torch.zeros(1, 62, P, dtype=torch.float64), eps=torch.randn(N, 62, F, dtype=torch.float64))
    x0f = x0.reshape(N, -1).numpy()  # 展平索引 = c*F + t
    Ks = prior.K_s
    Kt_ = prior.K_post_t
    # 边缘方差
    var_true = np.outer(np.diag(Ks), np.diag(Kt_)).ravel()
    var_emp = x0f.var(axis=0)
    rel = np.abs(var_emp - var_true) / var_true
    assert rel.max() < 0.10, f"边缘方差偏差过大 {rel.max():.3f}"
    print("[边缘方差] Var_emp≈diag(K_s)·diag(K'_{t,f|p})，最大相对偏差 %.3f" % rel.max())
    # 随机元素对（全 Kronecker 直接抽查）
    rng = np.random.default_rng(0)
    idx = rng.integers(0, 62 * F, size=(600, 2))
    cov_emp = np.array([np.mean(x0f[:, i] * x0f[:, j]) for i, j in idx])
    cov_true = np.array([Ks[i // F, j // F] * Kt_[i % F, j % F] for i, j in idx])
    err = np.abs(cov_emp - cov_true).max()
    assert err < 0.08, f"Kronecker 协方差抽查误差过大 {err:.3f}"
    print(f"[Kronecker] 600 随机元素对 max|Cov_emp − K_s⊗K'| = {err:.3f}（N=4000 抽样噪声 ~0.02）")
    # 通道相关（同时刻 t）与时间相关（同通道 c）
    t0 = 64
    Xc = x0[:, :, t0].numpy()  # (N,62)
    Cemp = np.corrcoef(Xc.T)
    Ctrue = Ks / np.sqrt(np.outer(np.diag(Ks), np.diag(Ks)))
    m_cc = np.abs(Cemp - Ctrue)[np.triu_indices(62, 1)].max()
    assert m_cc < 0.08, f"通道间经验相关偏差过大 {m_cc:.3f}"
    c0 = 30
    Tt = x0[:, c0, :].numpy()
    Temp = np.corrcoef(Tt.T)
    Ttrue = Kt_ / np.sqrt(np.outer(np.diag(Kt_), np.diag(Kt_)))
    m_tc = np.abs(Temp - Ttrue)[np.triu_indices(F, 1)].max()
    assert m_tc < 0.08, f"时间侧经验相关偏差过大 {m_tc:.3f}"
    print(f"[相关结构] 通道 max|Δρ|={m_cc:.3f}（t 固定）；时间 max|Δρ|={m_tc:.3f}（c 固定）")

    # ④ K_s 真实坐标 PSD 三档扫描
    coords = np.load(f"{cfg.data_root}/chan_coords_62.npy")
    for sig in (0.045, 0.080, 0.137):
        Ks_raw = ricker3d_kernel(coords, sig)
        w = np.linalg.eigvalsh(Ks_raw)
        assert w.min() > -1e-8, f"σ={sig}: K_s 非 PSD（min_eig={w.min():.3e}）"
        assert np.allclose(np.diag(Ks_raw), 1.0, atol=1e-12), "K_s 须单位对角"
        p_j = torch.linalg.cholesky(torch.from_numpy(Ks_raw + 1e-3 * np.eye(62)))
        erank = float(np.exp(-(w / w.sum()) @ np.log(w / w.sum())))
        print(f"[K_s PSD] σ={sig:.3f}: min_eig={w.min():+.2e}, +jitter Cholesky OK, erank={erank:.2f}/62")

    # ⑤ μ 线性
    x1 = torch.randn(2, 62, P)
    x2 = torch.randn(2, 62, P)
    m = prior.posterior_mean
    assert torch.allclose(m(2.5 * x1), 2.5 * m(x1), atol=1e-5)
    assert torch.allclose(m(x1 + x2), m(x1) + m(x2), atol=1e-5)
    print("[线性] μ=A·y^p 齐次+可加验证通过")
    print("[PASS] test_gpr 全部断言通过")


if __name__ == "__main__":
    main()
