"""官方 mamba-ssm 适配层（mamba_ssm 2.3.1 + causal_conv1d 1.7.0，2026-10-12 已装并验证）。

- MambaBlock = 预 norm（默认 RMSNorm，norm_type 可选 LayerNorm——fmt 变体 default-off 钩子）+
  mamba_ssm.Mamba + 残差（因果单向）
- 输入/输出 (B, L, d_model)；past 前缀天然作为条件被 future 位置经递归看到

norm_type default-off 钩子（用户 2026-10-16 拍板 D4：fmt 变体 RMSNorm→LayerNorm）：
默认 "rmsnorm" = nn.RMSNorm 原结构逐位一致（state_dict 键/旧变体 config_hash 零漂移）；
"layernorm" = nn.LayerNorm（含 bias ⇒ state_dict 多 *.norm.bias 键，仅 fmt ckpt）；
先例 bottleneck_rank/early_stop_patience/lr_schedule（getattr(cfg, "norm_type", "rmsnorm") 透传）。

cond_adaln default-off 钩子（用户 2026-10-17 拍板 A 案 scale/shift，任务 10-17-gpfm-mamba-adaln-f0）：
默认 "off" = 原结构逐位一致（不建任何新参数/子模块，state_dict 键/旧变体 config_hash 零漂移）；
"scale_shift" = MambaBlock pre-norm 位换 adaLN——LN(x) 经 modulate(x, s, b) = (1+s)⊙x+b 调制，
(s,b) 由 forward 传入的 (B, mod_dim) 条件张量经逐 site 调制 MLP Sequential(SiLU, Linear(mod_dim,
2*d_model))（末层零初始化 ⇒ 初始严格等价纯 LayerNorm）自产切分；state_dict 键 blocks.{i}.mod.1.*
（SiLU 无参占 0，Linear 落索引 1）。**RNG 中性构造契约**：调制 MLP 必须在
torch.random.fork_rng(devices=[]) 内构建——nn.Linear 构建期 reset_parameters 无条件消耗全局 RNG，
fork_rng 隔离随机流 ⇒ 同 seed 下 "off"/"scale_shift" 两分支共享权重逐位相同（review Major）。
"zero"（adaLN-Zero B 案）值域预留、本任务不启用：传入报 NotImplementedError；非法值 ValueError
（先例 build_norm/build_lr_scheduler）。枚举字符串先例 lr_schedule/kernel_t。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from mamba_ssm import Mamba


def modulate(x: torch.Tensor, s: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """adaLN 调制（DiT 官方同式）：x * (1 + s) + b；s/b 形状 (B, d) -> (B, 1, d) 广播到 (B, L, d)。"""
    return x * (1 + s.unsqueeze(1)) + b.unsqueeze(1)


def build_norm(d_model: int, norm_type: str = "rmsnorm") -> nn.Module:
    """norm 构建钩子：'rmsnorm'（默认，原结构）| 'layernorm'（fmt 变体）；其他值构建期报错。"""
    if norm_type == "rmsnorm":
        return nn.RMSNorm(d_model)
    if norm_type == "layernorm":
        return nn.LayerNorm(d_model)
    raise ValueError(f"unknown norm_type: {norm_type}（可选 'rmsnorm' | 'layernorm'）")


class MambaBlock(nn.Module):
    def __init__(
        self,
        d_model: int = 16,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        use_fast_path: bool = True,
        norm_type: str = "rmsnorm",  # default-off 钩子（D4）；默认 = 原 nn.RMSNorm 逐位一致
        cond_adaln: str = "off",     # default-off 钩子（A 案）；默认 = 原结构逐位一致（无新参数）
        mod_dim: int | None = None,  # 调制条件维度（= time_emb_dim，仅 cond_adaln 启用时使用）
    ):
        super().__init__()
        self.norm = build_norm(d_model, norm_type)
        if cond_adaln == "scale_shift":
            # RNG 中性构造（design §1.1 契约）：fork_rng 隔离构建期随机流——同 seed 下
            # 启用/关闭两分支共享权重逐位相同（Linear reset_parameters 消耗不泄漏到全局流）。
            with torch.random.fork_rng(devices=[]):
                self.mod = nn.Sequential(
                    nn.SiLU(), nn.Linear(mod_dim, 2 * d_model)
                )
                nn.init.zeros_(self.mod[-1].weight)  # 零初始化 ⇒ s=b=0 ⇒ 初始 = 纯 LayerNorm
                nn.init.zeros_(self.mod[-1].bias)
        elif cond_adaln == "off":
            pass  # 不建任何新参数/子模块（零漂移）
        elif cond_adaln == "zero":
            raise NotImplementedError(
                f'cond_adaln="zero"（adaLN-Zero B 案）未拍板实现，本任务仅接 "off"|"scale_shift"'
            )
        else:
            raise ValueError(f"unknown cond_adaln: {cond_adaln}（可选 'off' | 'scale_shift' | 'zero'(预留)）")
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            use_fast_path=use_fast_path,
        )

    def forward(self, x, mod=None):
        # mod is None：原式逐位不变（default-off）；非 None：adaLN 调制（s,b 由 mod 经逐 site MLP 自产切分）
        if mod is None:
            return x + self.mamba(self.norm(x))
        s, b = self.mod(mod).chunk(2, dim=-1)  # (B,2d) -> s/b 各 (B,d)
        return x + self.mamba(modulate(self.norm(x), s, b))


def build_mamba_backbone(
    n_layers: int,
    d_model: int = 16,
    d_state: int = 16,
    d_conv: int = 4,
    expand: int = 2,
    use_fast_path: bool = True,
    norm_type: str = "rmsnorm",  # default-off 钩子（D4）：透传各 MambaBlock pre-norm
    cond_adaln: str = "off",     # default-off 钩子（A 案）：透传各 MambaBlock（kwargs——v4b 六位置参调用向后兼容）
    mod_dim: int | None = None,  # 调制条件维度（仅 cond_adaln 启用时使用）
) -> nn.Sequential:
    return nn.Sequential(
        *[
            MambaBlock(d_model, d_state, d_conv, expand, use_fast_path, norm_type,
                       cond_adaln=cond_adaln, mod_dim=mod_dim)
            for _ in range(n_layers)
        ]
    )
