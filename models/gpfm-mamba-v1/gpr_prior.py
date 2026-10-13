"""GPR 条件先验 — 时空可分核 K_s ⊗ K_t（design §2.1 逐字实现）。

- 时间网格 τ_i = i·π/p（p=50，禁止整窗 [0,1] 归一）；K_t=OU（γ=1）默认，Matérn-3/2 备选
- 白噪 iso 折叠进时间核 K'_t = K_t + iso·I（保持 Kronecker 结构，禁止联合恒等白噪）
- K'_t,pp 求逆前 +1e-4·I；A = K'_t,fp · cholesky_solve(K'_t,pp)（128×384，纯时间侧）
- L_t = chol(K'_{t,f|p})（128×128）；K_s = Ricker 三维正定形式 (1−d²/(3σ²))e^{−d²/(2σ²)}
  （σ=0.080 m + jitter_s·I=1e-3；严禁严格 1D Ricker (1−d²/σ²)e^{−d²/2σ²}——实测非 PSD）
- 全部 float64 构造 → float32 buffer；Cholesky 失败即报错不静默
- 采样 x0^f = μ + L_s·ε·L_tᵀ（ε~N(0,I_{62×128})）；μ = A·y^p（A 与 K_s 无关）
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn


def time_kernel(name: str, gamma: float) -> np.ndarray:
    """返回函数 d -> k(d)。"""
    if name == "ou":
        return lambda d: np.exp(-gamma * np.abs(d))
    if name == "matern32":
        c = np.sqrt(3.0 * gamma)
        return lambda d: (1.0 + c * np.abs(d)) * np.exp(-c * np.abs(d))
    raise ValueError(f"未知时间核: {name}（注册表仅 'ou' | 'matern32'）")


def ricker3d_kernel(coords: np.ndarray, sigma: float) -> np.ndarray:
    """Ricker 三维正定形式（= -∇²高斯，FT=k²·Gaussian≥0，任意 ℝ³ 点集保证 PSD）。

    K_s(c,c') = (1 − d²/(3σ²))·exp(−d²/(2σ²))；不加 jitter，单位对角。
    """
    coords = np.asarray(coords, dtype=np.float64)
    diff = coords[:, None, :] - coords[None, :, :]
    d2 = (diff ** 2).sum(-1)
    return (1.0 - d2 / (3.0 * sigma ** 2)) * np.exp(-d2 / (2.0 * sigma ** 2))


class GPRPrior(nn.Module):
    """全观测可分 GPR 条件先验（past 62×384 全观测 ⇒ A/L_t/L_s 均为固定算子预计算）。

    bandlimit_hz（Run 015 设计，v3 变体启用）：对**采样输出**做末维 reflect-pad 砖墙低通
    （>bandlimit_hz 置零）——条件化仍用原核完成（满秩、无解析延拓塌缩），等价于把条件
    先验分布的协方差带限为 B·K_post·Bᵀ。默认 None = 关闭（v1/v2 行为逐位不变）；
    posterior_mean 不滤（Run 015 探针：μ 本就平滑，滤波不动数值，保基线口径）。
    """

    def __init__(
        self,
        coords: np.ndarray,
        past_len: int = 384,
        future_len: int = 128,
        kernel_t: str = "ou",
        gamma: float = 1.0,
        period_p: int = 50,
        iso: float = 1e-2,
        pp_jitter: float = 1e-4,
        sigma_s: float = 0.080,
        jitter_s: float = 1e-3,
        spatial: str = "ricker",  # 'ricker' | 'identity'（identity 仅单测/消融口径）
        bandlimit_hz: Optional[float] = None,
        fs: float = 500.0,
        bandlimit_pad: int = 64,
    ):
        super().__init__()
        self.past_len = past_len
        self.future_len = future_len
        T = past_len + future_len
        # ---- 时间侧（float64）----
        tau = np.arange(T, dtype=np.float64) * (np.pi / period_p)  # τ_i = i·π/p
        k = time_kernel(kernel_t, gamma)
        Kt = k(np.abs(tau[:, None] - tau[None, :]))
        Kt = Kt + iso * np.eye(T)  # 白噪折叠进时间核
        pp = Kt[:past_len, :past_len] + pp_jitter * np.eye(past_len)
        pf = Kt[:past_len, past_len:]  # (384,128)
        ff = Kt[past_len:, past_len:]  # (128,128)
        pp_t = torch.from_numpy(pp)
        pf_t = torch.from_numpy(pf)
        L_pp = torch.linalg.cholesky(pp_t)  # 失败即抛错，不静默
        X = torch.cholesky_solve(pf_t, L_pp)  # X = inv(K'_pp) K'_pf
        A_t = X.T.contiguous()  # A = K'_fp inv(K'_pp)（128×384）
        K_post_t = torch.from_numpy(ff) - torch.from_numpy(Kt[past_len:, :past_len]) @ X
        L_t = torch.linalg.cholesky(K_post_t)
        # ---- 空间侧（float64）----
        if spatial == "ricker":
            Ks = ricker3d_kernel(coords, sigma_s)
            Ks_jit = Ks + jitter_s * np.eye(Ks.shape[0])
        elif spatial == "identity":
            Ks = np.eye(len(coords))
            Ks_jit = Ks.copy()
        else:
            raise ValueError(f"未知 spatial: {spatial}")
        L_s = torch.linalg.cholesky(torch.from_numpy(Ks_jit))
        # ---- 诊断用 float64 存档（非 buffer）----
        self.K_t_full = Kt.copy()
        self.K_post_t = K_post_t.numpy().copy()
        self.K_s_prejitter = Ks.copy()
        self.K_s = Ks_jit.copy()
        self.sigma_s = sigma_s
        # ---- 带限（默认关；Run 015）----
        self.bandlimit_hz = bandlimit_hz
        self.fs = fs
        self.bandlimit_pad = bandlimit_pad
        # ---- float32 buffers（训练用）----
        self.register_buffer("A", A_t.float())
        self.register_buffer("L_t", L_t.float())
        self.register_buffer("L_s", L_s.float())

    def bandlimit(self, x: torch.Tensor) -> torch.Tensor:
        """末维 reflect-pad 砖墙低通（self.bandlimit_hz=None 时原样返回）。

        reflect 填充抑制 DFT 循环卷积段边界振铃（Run 013 实测边缘/中段偏差比 1.56→1.09）。
        训练（prior.sample 末端）与推理（EulerSampler x0 装配末端）共用本方法，保证同滤一致。
        """
        if self.bandlimit_hz is None:
            return x
        n = x.shape[-1]
        pad = self.bandlimit_pad
        xp = torch.nn.functional.pad(x, (pad, pad), mode="reflect")
        freqs = torch.fft.rfftfreq(xp.shape[-1], d=1.0 / self.fs, device=xp.device)
        spec = torch.fft.rfft(xp, dim=-1)
        spec[..., freqs > self.bandlimit_hz] = 0
        return torch.fft.irfft(spec, n=xp.shape[-1], dim=-1)[..., pad:pad + n]

    @torch.no_grad()
    def posterior_mean(self, x_past: torch.Tensor) -> torch.Tensor:
        """μ = A·y^p：x_past (B,C,384) -> (B,C,128)。A 与 K_s 无关。"""
        return torch.einsum("bcp,fp->bcf", x_past.to(self.A.dtype), self.A)

    def sample(
        self,
        x_past: torch.Tensor,
        eps: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """x0^f = LP(μ + L_s·ε·L_tᵀ)（bandlimit_hz=None 时 LP=恒等）。"""
        mu = self.posterior_mean(x_past)
        B, C, F = mu.shape
        if eps is None:
            eps = torch.randn(B, C, F, device=mu.device, dtype=mu.dtype, generator=generator)
        return self.bandlimit(mu + torch.einsum("ij,bjk,lk->bil", self.L_s, eps, self.L_t))
