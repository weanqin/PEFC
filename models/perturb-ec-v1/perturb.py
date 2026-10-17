"""perturb.py — 扰动算子 P(X, A, r)（design §2.1）+ 单窗变体批构造 + config 唯一装载配方。

**A 案：增益调制（用户 2026-10-17 拍板，默认且唯一实现）**
    x_A(t) ← (1+r)·x_A(t)，t ∈ past 全段（384 样本 @250 Hz），主流程仅正方向 r ≥ 0。
    - 注入层 = 模型输入层 X_past 通道 A（框架 Stage II 正典位置）；r 为相对增益（无量纲）。
    - r=0 数值恒等（×1.0 位保持）；只动源通道 A，其余通道逐位不动。
    - 谱域在流形上 ≠ 幅度域在流形上：r ≥ 0.32 属网络外推行为（design §2.1 警示，判读随附）。
**B 案：加性带限固定波形 x_A + r·σ_A·g(t)（g = 1–40 Hz 带限固定 seed 归一化波形）——
文档级预留，不实现**；启用须另走口径变更纪律 + 实现 + 重登记（design §2.1）。

本模块另承载扰动侧 config 的唯一装载配方（importlib 唯一名 `perturb_config`，
sys.modules 先注册后 exec——design §1 四行配方；不占据 `config` 模块名）。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent


def load_config():
    """扰动侧 config 唯一装载入口（唯一名 `perturb_config`，幂等复用同实例）。

    配方（design §1）：sys.modules **先注册后 exec**——dataclasses 按 cls.__module__ 查
    sys.modules；本 config 虽为纯常量模块仍守配方，防日后演化踩坑。
    """
    name = "perturb_config"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, str(_HERE / "config.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod           # 先注册（design §1 配方）
        spec.loader.exec_module(mod)
    return sys.modules[name]


def apply_gain(x_past: torch.Tensor, ch: int, r: float) -> torch.Tensor:
    """A 案增益算子 P(X, A, r)：x[..., ch, :] ← (1+r)·x[..., ch, :]（past 全段）。

    - 只动通道 ch，其余通道逐位不动；返回新张量（不改输入）。
    - r=0 数值恒等（×1.0 位保持，CRN 下 D(0)=0 按构造）。
    - x_past (C, P) 或 (B, C, P)。
    """
    out = x_past.clone()
    out[..., ch, :] = out[..., ch, :] * (1.0 + r)
    return out


def variant_index(a: int, j: int) -> int:
    """变体批索引表（顺序固定，design §2.2）：index 0 = baseline；其后**源 A 主序、
    r 档次序**，idx = 1 + a·N_R_PERT + j。逆映射 variant_of。"""
    PC = load_config()
    return 1 + a * PC.N_R_PERT + j


def variant_of(idx: int):
    """逆映射：idx ≥ 1 → (a, j)（idx 0 = baseline，不接受）。"""
    PC = load_config()
    assert idx >= 1, "idx 0 = baseline，无 (a, j) 分解"
    return divmod(idx - 1, PC.N_R_PERT)


def build_variant_batch(x_past: torch.Tensor) -> torch.Tensor:
    """单窗变体批：(1 + N_SRC×N_R_PERT, C, P) = (497, C, P)。

    行 0 = baseline（逐位复制）；行 variant_index(a, j) 仅通道 a 乘 (1+r_j)，其余通道
    与 baseline 逐位相同。输入 (C, P) 或 (1, C, P)。主 Run 每窗**一次** sample() 调用消费
    本批（design §2.2 批采样纪律：r=0 恒等断言在同批形内成立的前提）。
    """
    PC = load_config()
    x = x_past.unsqueeze(0) if x_past.dim() == 2 else x_past
    assert x.dim() == 3 and x.shape[0] == 1, f"单窗接口，输入形状 {tuple(x.shape)}"
    batch = x.repeat(PC.N_VARIANTS, 1, 1)          # 全部行先逐位复制 baseline（含行 0）
    gains = torch.as_tensor([1.0 + r for r in PC.PERT_R_GRID],
                            dtype=x.dtype, device=x.device).view(-1, 1)   # (8,1)
    for a in range(PC.N_SRC):
        sl = slice(1 + a * PC.N_R_PERT, 1 + (a + 1) * PC.N_R_PERT)
        batch[sl, a, :] = batch[sl, a, :] * gains  # (8,P)×(8,1) 广播；只动通道 a
    return batch
