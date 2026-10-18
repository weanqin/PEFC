"""candidates.py — N 全段/子窗掩码算子 + CRN + E1b/E2 批构造（10-18-pastdep-audit-f0）。

算子契约同前任务：只动源通道；r==0 短路 clone；新张量。行序：0 = baseline；
1 + a·N_ARMS + k，k = ARMS 索引（3 剂量全段在前、4 子窗在后）。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent


def load_config():
    name = "pd_config"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, str(_HERE / "config.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


_NOISE_WAVEFORM: torch.Tensor | None = None


def noise_waveform() -> torch.Tensor:
    """g(t)：固定 seed FFT 1–40 Hz 掩码、RMS=1、全局共享（常量与前两任务逐位一致）。"""
    global _NOISE_WAVEFORM
    if _NOISE_WAVEFORM is None:
        SC = load_config()
        rng = np.random.default_rng(SC.NOISE_WAVEFORM_SEED)
        white = rng.standard_normal(SC.NOISE_LEN)
        spec = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(SC.NOISE_LEN, d=1.0 / SC.FS)
        mask = (freqs >= SC.NOISE_BAND[0]) & (freqs <= SC.NOISE_BAND[1])
        band = np.fft.irfft(spec * mask, n=SC.NOISE_LEN)
        band = band / np.sqrt(np.mean(band ** 2))
        _NOISE_WAVEFORM = torch.from_numpy(band.astype(np.float32))
    return _NOISE_WAVEFORM


def _sigma_a(x: torch.Tensor, ch: int) -> torch.Tensor:
    return x[..., ch, :].std(dim=-1)          # 全 past std（design §2 钉死）


def apply_noise(x: torch.Tensor, ch: int, r: float) -> torch.Tensor:
    """N 全段：x_A ← x_A + r·σ_A·g(t)。"""
    out = x.clone()
    out[..., ch, :] = out[..., ch, :] \
        + r * _sigma_a(x, ch) * noise_waveform().to(device=x.device, dtype=x.dtype)
    return out


def apply_noise_window(x: torch.Tensor, ch: int, r: float, sub: int) -> torch.Tensor:
    """N 子窗掩码（E2）：注入 = r·σ_A·(g·1_W)/RMS_W——boxcar 掩码后子窗内 RMS 重归一到 1。

    注入支集 ⊆ W；子窗外逐位不动；σ_A 仍用全 past std（跨子窗可比）。"""
    SC = load_config()
    lo, hi = SC.SUBWINDOWS[sub]
    g = noise_waveform().to(device=x.device, dtype=x.dtype)
    gw = torch.zeros_like(g)
    gw[lo:hi] = g[lo:hi]
    gw[lo:hi] = gw[lo:hi] / torch.sqrt((gw[lo:hi] ** 2).mean())
    out = x.clone()
    out[..., ch, lo:hi] = out[..., ch, lo:hi] + r * _sigma_a(x, ch) * gw[lo:hi]
    return out


def apply_paradigm(x_past: torch.Tensor, ch: int, r: float, arm) -> torch.Tensor:
    """统一入口：arm = ("dose", r) 全段 / ("win", i) 子窗@E2_DOSE；r==0 短路 clone。"""
    if r == 0 and arm[0] == "dose":
        return x_past.clone()
    if arm[0] == "dose":
        return apply_noise(x_past, ch, r)
    return apply_noise_window(x_past, ch, r, arm[1])


def crn_noise(seed: int, n_samples: int, n_channels: int, future_len: int, device):
    """显式 eps/eta（(1,K,C,F)）；seed 由调用方给（批 = SEED+win_idx；E3 臂 = +偏置）。"""
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    eps = torch.randn(1, n_samples, n_channels, future_len, device=device, generator=g)
    eta = torch.randn(1, n_samples, n_channels, future_len, device=device, generator=g)
    return eps, eta


def variant_index(a: int, k: int) -> int:
    return 1 + a * load_config().N_ARMS + k


def build_batch(x_past: torch.Tensor) -> torch.Tensor:
    """E1b/E2 批：(435, C, P)；行 0 = baseline 逐位复制。"""
    SC = load_config()
    x = x_past.unsqueeze(0) if x_past.dim() == 2 else x_past
    assert x.dim() == 3 and x.shape[0] == 1, f"单窗接口，输入形状 {tuple(x.shape)}"
    batch = x.repeat(SC.BATCH_ROWS, 1, 1)
    for a in range(SC.N_SRC):
        for k, arm in enumerate(SC.ARMS):
            r = arm[1] if arm[0] == "dose" else SC.E2_DOSE
            batch[variant_index(a, k)] = apply_paradigm(x, a, r, arm)[0]
    return batch
