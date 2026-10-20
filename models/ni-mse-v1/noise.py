"""noise.py — 无偏高斯噪声注入算子 P(X,a,c) + 种子树（design.md §1b-1/2 + v1.5 §1.2/§3）。

算子（v1.2 重定义，标准化空间——注入位于 dataloader z-score 之后、模型 forward 之前）：
    P(X, a, c)：X'[a, :] = X[a, :] + c·ξ，ξ ~ N(0, I_L)，past 全段（384 样本）；
    σ_p ≡ c 无量纲；c=0 ⇒ P(X,a,0) ≡ X（R2 恒等断言的算子侧前提）；
    clone 后改（禁原位）；只动源通道 a，其余通道逐位不动。

种子树（master=2026；分支整数映射固定并落 manifest，design §1b-2）：
    ε_ref=10 / ε_null=11 / ξ_pert=12 / η=13 / win-null=20 / win-pert=21
    统一合法形式 `np.random.SeedSequence(entropy=2026, spawn_key=(<branch_int>, …))`
    （字符串 spawn_key 本机非法——审查补钉）。

spawn_key 尾部钉定（Run 000 登记）：
    ε_ref  (10, sid)                     —— 逐被试一份固定 ε 网格，全窗复用；
    η_ref  (13, sid)                     —— 同构；
    ε_null (11, sid, w)                  —— 逐窗一次物化 (K_null, C, F)（k 经抽取次序）；
    η_null (13, sid, w)                  —— 同构；
    ξ_pert (12, sid, w, a, k)            —— 审查补钉四元组（k 显式进 spawn_key）；
    η      (13, sid, w, a, k)            —— 扰动侧 η 与 ξ 同构四元组；
    响应侧 fresh ε (11, sid, w, a, k)    —— ε_null 支（无 c ⇒ 跨 c 臂 CRN 配对）。
    w = 被试窗全局索引（0..929，loader shuffle=False 确定性序）。
ξ/η/ε 均不含 c——同一 (sid,w,a,k) 的噪声在全部 c 臂间逐位共享（剂量-响应无蒙特卡洛抖动）。
"""

from __future__ import annotations

import _blas_guard  # noqa: F401  必须最先

import numpy as np
import torch

MASTER_ENTROPY = 2026

# 分支整数映射（design §1b-2 固定；manifest 逐字登记）
BRANCH = {
    "eps_ref": 10,
    "eps_null": 11,
    "xi_pert": 12,
    "eta": 13,
    "win_null": 20,
    "win_pert": 21,
}


def spawn_key_for(branch: int, *ids: int) -> tuple:
    """合法 spawn_key：(branch_int, *ids)，全部 int。"""
    return (int(branch),) + tuple(int(i) for i in ids)


def rng_for(branch: int, *ids: int) -> np.random.Generator:
    """种子树派生 rng（确定性；同 (branch, *ids) 必逐位复现）。"""
    return np.random.default_rng(
        np.random.SeedSequence(entropy=MASTER_ENTROPY, spawn_key=spawn_key_for(branch, *ids))
    )


def apply_noise(x_past: torch.Tensor, a: int, c: float, xi: torch.Tensor) -> torch.Tensor:
    """P(X, a, c)：X'[a,:] = X[a,:] + c·ξ（past 全段；标准化空间；clone 后改，禁原位）。

    x_past (C, P) 或 (B, C, P)（B 行可各自独立 ξ——扰动期 (w,k) 展平批每行独立注入）；
    xi 与 x_past[..., a, :] 可广播（(P,) / (B, P)）；返回新张量。
    """
    out = x_past.clone()
    out[..., a, :] = out[..., a, :] + c * xi
    return out


def draw_xi(sid: int, w: int, a: int, k: int, past_len: int) -> np.ndarray:
    """ξ_{sid,w,a,k} ~ N(0, I_P)（spawn_key=(12, sid, w, a, k)，逐位可复现）。"""
    return rng_for(BRANCH["xi_pert"], sid, w, a, k).standard_normal(past_len)


def draw_eta(sid: int, w: int, a: int, k: int, shape: tuple) -> np.ndarray:
    """η_{sid,w,a,k}（spawn_key=(13, sid, w, a, k)，与 ξ 同构四元组；σ_min 噪声显式支）。"""
    return rng_for(BRANCH["eta"], sid, w, a, k).standard_normal(shape)


def draw_eps_pert(sid: int, w: int, a: int, k: int, shape: tuple) -> np.ndarray:
    """响应侧 fresh ε（ε_null 支，spawn_key=(11, sid, w, a, k)；可交换 null 口径）。"""
    return rng_for(BRANCH["eps_null"], sid, w, a, k).standard_normal(shape)
