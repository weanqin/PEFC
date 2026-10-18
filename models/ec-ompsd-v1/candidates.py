"""candidates.py — O 单范式算子（DC 偏置）+ CRN 噪声 + 变体批构造（10-18-ec-ompsd-f0）。

算子契约（沿 perturb-ec-pipeline.md）：只动源通道逐元素、其余通道逐位不动；
r==0 短路返回 clone；返回新张量。批行序：行 0 = baseline；行 1 + a·len(R_SAMPLED) + j。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent


def load_config():
    """本任务 config 唯一装载入口（唯一名 `ec_config`，幂等；sys.modules 先注册后 exec）。"""
    name = "ec_config"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, str(_HERE / "config.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


_NOISE_WAVEFORM: torch.Tensor | None = None


def noise_waveform() -> torch.Tensor:
    """N 的 g(t)（常量与 perturb-search-v1 登记逐位一致 ⇒ 同一波形实例）：
    固定 seed FFT 域 1–40 Hz 掩码、RMS 重归一 1、全局共享。float32 torch (384,)。"""
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
    """σ_A = 该窗源通道 past 段 std（z-score 后 ≈1，仍逐窗实算）。"""
    return x[..., ch, :].std(dim=-1)


def apply_bias(x: torch.Tensor, ch: int, r: float) -> torch.Tensor:
    """O：x_A ← x_A + r·σ_A（全段常数 DC 平移；常量同 perturb-search-v1 登记）。"""
    out = x.clone()
    out[..., ch, :] = out[..., ch, :] + r * _sigma_a(x, ch)
    return out


def apply_paradigm(x_past: torch.Tensor, ch: int, r: float) -> torch.Tensor:
    """统一入口（单范式）：r==0 短路返回 clone（恒等构造的根基）。"""
    if r == 0:
        return x_past.clone()
    return apply_bias(x_past, ch, r)


def crn_noise(win_idx: int, n_samples: int, n_channels: int, future_len: int,
              device, seed: int = None):
    """逐窗一组 eps/eta（(1, K, C, F)，baseline 与全部变体行共享）。
    确定性：torch.Generator(device) manual_seed(SEED + win_idx)，先 eps 后 eta。"""
    SC = load_config()
    g = torch.Generator(device=device)
    g.manual_seed((SC.SEED if seed is None else seed) + win_idx)
    eps = torch.randn(1, n_samples, n_channels, future_len, device=device, generator=g)
    eta = torch.randn(1, n_samples, n_channels, future_len, device=device, generator=g)
    return eps, eta


def variant_index(a: int, j: int, n_r: int) -> int:
    """行 0 = baseline；行 1 + a·n_r + j（源主序、r 次序）。"""
    return 1 + a * n_r + j


def variant_of(idx: int, n_r: int):
    """逆映射：idx ≥ 1 → (a, j)。"""
    assert idx >= 1, "idx 0 = baseline，无 (a, j) 分解"
    return divmod(idx - 1, n_r)


def build_variant_batch(x_past: torch.Tensor) -> torch.Tensor:
    """单窗 N 批：(1 + 62×8, C, P) = 497 行。行 0 = baseline 逐位复制；
    行 variant_index(a, j) = 对通道 a 施加 O @ R_SAMPLED[j]。输入 (C, P) 或 (1, C, P)。"""
    SC = load_config()
    x = x_past.unsqueeze(0) if x_past.dim() == 2 else x_past
    assert x.dim() == 3 and x.shape[0] == 1, f"单窗接口，输入形状 {tuple(x.shape)}"
    n_r = len(SC.R_SAMPLED)
    batch = x.repeat(SC.BATCH_ROWS, 1, 1)
    for a in range(SC.N_SRC):
        for j, r in enumerate(SC.R_SAMPLED):
            batch[variant_index(a, j, n_r)] = apply_paradigm(x, a, r)[0]
    return batch
