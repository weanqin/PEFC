"""candidates.py — 6 个输入端扰动算子（design §3）+ CRN 噪声 + 变体批构造（本任务自实现）。

算子实现契约（design §3 预注册）：
- 只动通道 A 逐元素、其余通道逐位不动（单测断言）；返回新张量（不改输入）。
- **r==0 短路返回 clone**（规避 L 减加往返与 ±0.0 翻转的浮点陷阱；L 实现代数形式
  x_A − r·(x_A − μ_A)）；六算子统一经 `apply_paradigm` 调度器短路。
- G = 只读复用 `perturb-ec-v1::apply_gain`（纯函数、无 config 访问；importlib 唯一名装载，
  **不把 perturb-ec-v1 目录进 sys.path**）。crn_noise / 批构造为本任务自实现（读自身
  config；不复用 perturb-ec-v1 的 build_variant_batch/crn_noise——二者读旧任务 config）。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt

_HERE = Path(__file__).resolve().parent


def load_config():
    """搜索侧 config 唯一装载入口（唯一名 `search_config`，幂等复用同实例）。

    配方：sys.modules **先注册后 exec**；不占据 `config` 模块名（留给 gpfm-mamba-v1）。
    """
    name = "search_config"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, str(_HERE / "config.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod           # 先注册
        spec.loader.exec_module(mod)
    return sys.modules[name]


# ---------------------------------------------------------------------------
# 只读复用：perturb-ec-v1::apply_gain（纯函数叶子）
# ---------------------------------------------------------------------------

def _load_gain_leaf():
    """importlib 唯一名 `perturb_ec_gain_leaf` 装载 perturb-ec-v1/perturb.py——该文件仅
    顶层 import stdlib+torch，`apply_gain` 为纯函数（无 config 访问），安全只读复用。"""
    SC = load_config()
    name = "perturb_ec_gain_leaf"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, str(Path(SC.EC_PERTURB_LEAF)))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


apply_gain = _load_gain_leaf().apply_gain   # G 算子（现行 A 案，只读复用）


# ---------------------------------------------------------------------------
# 固定波形（全局共享单例，deterministic）
# ---------------------------------------------------------------------------

_NOISE_WAVEFORM: torch.Tensor | None = None
_PULSE_WAVEFORM: torch.Tensor | None = None
_ALPHA_SOS = None


def noise_waveform() -> torch.Tensor:
    """N 算子的 g(t)（design §3 钉死）：固定 seed FFT 域 1–40 Hz 掩码生成的 384 样本波形、
    带限后 RMS 重归一到 1、**全局共享单波形**（跨 (A, 窗) 不变）。float32 torch (384,)。"""
    global _NOISE_WAVEFORM
    if _NOISE_WAVEFORM is None:
        SC = load_config()
        rng = np.random.default_rng(SC.NOISE_WAVEFORM_SEED)
        white = rng.standard_normal(SC.NOISE_LEN)
        spec = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(SC.NOISE_LEN, d=1.0 / SC.FS)
        mask = (freqs >= SC.NOISE_BAND[0]) & (freqs <= SC.NOISE_BAND[1])
        band = np.fft.irfft(spec * mask, n=SC.NOISE_LEN)
        band = band / np.sqrt(np.mean(band ** 2))          # RMS 重归一到 1
        _NOISE_WAVEFORM = torch.from_numpy(band.astype(np.float32))
    return _NOISE_WAVEFORM


def pulse_waveform() -> torch.Tensor:
    """P 算子的 h(t)（design §3 钉死）：20 样本（80 ms）对称汉宁脉冲（端点为零），中心 =
    past 末端 −40 样本（起止 [334, 354)）。float32 torch (384,)，全局共享。"""
    global _PULSE_WAVEFORM
    if _PULSE_WAVEFORM is None:
        SC = load_config()
        h = torch.zeros(SC.PAST_LEN)
        start = SC.PAST_LEN - SC.PULSE_CENTER_OFFSET - SC.PULSE_LEN // 2   # 384-40-10 = 334
        h[start:start + SC.PULSE_LEN] = torch.hann_window(SC.PULSE_LEN, periodic=False)
        _PULSE_WAVEFORM = h
    return _PULSE_WAVEFORM


def alpha_bandpass_sos():
    """B 算子带通（design §3 钉死）：Butterworth 阶 BP_BUTTER_ORDER（≤6 契约内取 3）+
    sos 形式（filtfilt 零相位用）。带 = ALPHA_BAND (8–13 Hz) @ fs=250。"""
    global _ALPHA_SOS
    if _ALPHA_SOS is None:
        SC = load_config()
        _ALPHA_SOS = butter(SC.BP_BUTTER_ORDER, [SC.ALPHA_BAND[0], SC.ALPHA_BAND[1]],
                            btype="bandpass", fs=SC.FS, output="sos")
    return _ALPHA_SOS


def alpha_bandpass(x: torch.Tensor) -> torch.Tensor:
    """α 带 (8–13 Hz) 零相位带通（sosfiltfilt，默认 padlen）：x (..., P) → 同形状新张量。

    float64 内部计算（IIR 数值稳定），返回与输入同 dtype/device。最后一维 = 时间。
    """
    SC = load_config()
    dev, dt = x.device, x.dtype
    arr = x.detach().to(torch.float64).cpu().numpy()
    out = np.ascontiguousarray(sosfiltfilt(alpha_bandpass_sos(), arr, axis=-1))
    return torch.from_numpy(out).to(device=dev, dtype=dt)


# ---------------------------------------------------------------------------
# 六算子（N/O/P/L/B 新实现；G = apply_gain 复用）。签名统一 (x_past, ch, r) → 新张量。
# x_past (C, P) 或 (B, C, P)；只动通道 ch；σ_A 逐窗实算（torch .std(dim=-1)，无偏 ddof=1）。
# ---------------------------------------------------------------------------

def _sigma_a(x: torch.Tensor, ch: int) -> torch.Tensor:
    """σ_A = 该窗源通道 past 段 std（z-score 后 ≈1，仍逐窗实算）。"""
    return x[..., ch, :].std(dim=-1)


def apply_noise(x: torch.Tensor, ch: int, r: float) -> torch.Tensor:
    """N：x_A ← x_A + r·σ_A·g(t)（g = 全局共享 1–40 Hz 带限 RMS=1 固定波形）。"""
    SC = load_config()
    out = x.clone()
    sig = _sigma_a(x, ch)
    g = noise_waveform().to(device=x.device, dtype=x.dtype)
    out[..., ch, :] = out[..., ch, :] + r * sig * g
    return out


def apply_bias(x: torch.Tensor, ch: int, r: float) -> torch.Tensor:
    """O：x_A ← x_A + r·σ_A（全段常数，DC 平移；数据 1–40 Hz 带限下 DC 属外推输入）。"""
    out = x.clone()
    sig = _sigma_a(x, ch)
    out[..., ch, :] = out[..., ch, :] + r * sig
    return out


def apply_pulse(x: torch.Tensor, ch: int, r: float) -> torch.Tensor:
    """P：x_A ← x_A + r·σ_A·h(t)（h = 20 样本汉宁脉冲，中心 past 末端 −40 样本）。"""
    out = x.clone()
    sig = _sigma_a(x, ch)
    h = pulse_waveform().to(device=x.device, dtype=x.dtype)
    out[..., ch, :] = out[..., ch, :] + r * sig * h
    return out


def apply_lesion(x: torch.Tensor, ch: int, r: float) -> torch.Tensor:
    """L：x_A ← μ_A + (1−r)·(x_A − μ_A)，**实现取代数形式 x_A − r·(x_A − μ_A)**（预注册：
    规避减加往返与 ±0.0 翻转的浮点陷阱）；μ_A = 该窗 x_A 均值。"""
    out = x.clone()
    xa = x[..., ch, :]
    mu = xa.mean(dim=-1, keepdim=True)
    out[..., ch, :] = xa - r * (xa - mu)
    return out


def apply_band_gain(x: torch.Tensor, ch: int, r: float) -> torch.Tensor:
    """B：x_A ← x_A + r·BP_{8–13Hz}(x_A)（α 带分量增益 1+r；Butterworth sos + filtfilt 零相位）。"""
    out = x.clone()
    bp = alpha_bandpass(x[..., ch, :])
    out[..., ch, :] = out[..., ch, :] + r * bp
    return out


_RAW_OPS = {
    "G": apply_gain,
    "N": apply_noise,
    "O": apply_bias,
    "P": apply_pulse,
    "L": apply_lesion,
    "B": apply_band_gain,
}


def apply_paradigm(key: str, x_past: torch.Tensor, ch: int, r: float) -> torch.Tensor:
    """统一调度器：**r==0 短路返回 clone**（六算子统一契约，含 G）；否则走对应原始算子。"""
    SC = load_config()
    assert key in SC.PARADIGMS, f"未知范式 {key!r}"
    if r == 0:
        return x_past.clone()
    return _RAW_OPS[key](x_past, ch, r)


# ---------------------------------------------------------------------------
# CRN 噪声（本任务自实现；读自身 config）
# ---------------------------------------------------------------------------

def crn_noise(win_idx: int, n_samples: int, n_channels: int, future_len: int,
              device, seed: int = None):
    """逐窗一组 eps/eta（(1, K, C, F)，跨 baseline 与全部变体行共享——design §2 预注册）。

    确定性方案：`torch.Generator(device)` manual_seed(SEED + win_idx)，**先 eps 后 eta**；
    batch 维 = 1（sample() 内对 (B,1,C,F) 先验均值广播 ⇒ 全部变体行同噪声，CRN 配对）。
    两阶段同 win_idx ⇒ 同张量（跨阶段逐位相同，sha256 断言）。
    """
    SC = load_config()
    g = torch.Generator(device=device)
    g.manual_seed((SC.SEED if seed is None else seed) + win_idx)
    eps = torch.randn(1, n_samples, n_channels, future_len, device=device, generator=g)
    eta = torch.randn(1, n_samples, n_channels, future_len, device=device, generator=g)
    return eps, eta


# ---------------------------------------------------------------------------
# 变体批构造（本任务自实现；沿 perturb-ec-v1::build_variant_batch 同构扩 r 维，读自身 config）
# ---------------------------------------------------------------------------

def variant_specs(paradigms) -> list:
    """变体批平铺格点表（顺序固定）：[(p_index, key, r), ...]——范式次序、r 末序（R_GRID[p] 升序）。

    长度 N_RT = Σ_p len(R_GRID[p])：阶段一 9（G4+N5）、阶段二 28。Gate D 的 r 网格数据源。
    """
    SC = load_config()
    return [(p, key, r) for p, key in enumerate(paradigms) for r in SC.R_GRID[key]]


def variant_index(a: int, j: int, n_rt: int) -> int:
    """变体批索引表（顺序固定）：index 0 = baseline；其后**源 A 主序、(范式, r) 格点次序**，
    idx = 1 + a·N_RT + j。逆映射 variant_of。"""
    return 1 + a * n_rt + j


def variant_of(idx: int, n_rt: int):
    """逆映射：idx ≥ 1 → (a, j)（idx 0 = baseline，不接受）。"""
    assert idx >= 1, "idx 0 = baseline，无 (a, j) 分解"
    return divmod(idx - 1, n_rt)


def build_variant_batch(x_past: torch.Tensor, paradigms, r_grid: dict = None,
                        timing: dict = None) -> torch.Tensor:
    """单窗变体批：(1 + N_SRC·N_RT, C, P)——阶段一 559 行、阶段二 1737 行（N_RT = Σ格点数）。

    行 0 = baseline（逐位复制）；行 variant_index(a, j) = 对通道 a 施加格点 j = (范式 p, r)，
    其余通道与 baseline 逐位相同；r 取 R_GRID[p]（r0 本点并入网格：Gate A/B/C 取 r0 切片、
    Gate D 用全网格——design §5/§6）。输入 (C, P) 或 (1, C, P)。主 Run 每窗**一次** sample()
    调用消费本批（跨 r 的 CRN 逐位配对 = 「仅改变 r」）。timing（可选 dict）逐范式累计算子
    施加秒数（scorecard 成本列数据源；含全部格点）。
    """
    import time
    SC = load_config()
    rg = {p: SC.R_GRID[p] for p in paradigms} if r_grid is None else r_grid
    specs = [(p, key, r) for p, key in enumerate(paradigms) for r in rg[key]]
    x = x_past.unsqueeze(0) if x_past.dim() == 2 else x_past
    assert x.dim() == 3 and x.shape[0] == 1, f"单窗接口，输入形状 {tuple(x.shape)}"
    n_rt = len(specs)
    n_rows = 1 + SC.N_SRC * n_rt
    batch = x.repeat(n_rows, 1, 1)          # 全部行先逐位复制 baseline（含行 0）
    for a in range(SC.N_SRC):
        for j, (p, key, r) in enumerate(specs):
            row = variant_index(a, j, n_rt)
            t0 = time.perf_counter()
            one = apply_paradigm(key, x, a, r)             # (1, C, P) 新张量，只动通道 a
            if timing is not None:
                timing[key] = timing.get(key, 0.0) + (time.perf_counter() - t0)
            batch[row] = one[0]
    return batch
