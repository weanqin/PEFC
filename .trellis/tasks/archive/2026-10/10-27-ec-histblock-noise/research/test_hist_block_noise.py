#!/usr/bin/env python3
"""test_hist_block_noise.py — hist-block 历史段分组共享噪声构造检查（CPU，无模型依赖）。

任务 10-27-ec-histblock-noise（AC2/AC3/AC4/AC6 构造面 + AC5 CRN 结构面）。
V1 通道共享 / V2 被试共享 / V3 组内恒等 / V4 组数与组间独立 / V5 CRN 结构与
口径断言负例。集成面（GPU 全链路 + iid 对拍）由 run 命令另行执行（见
experiment.md Run 002/003）。

Run:
  OMP_NUM_THREADS=8 /root/miniconda3/bin/python \
    .trellis/tasks/10-27-ec-histblock-noise/research/test_hist_block_noise.py
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "8")

import sys
from pathlib import Path

import numpy as np
import torch

VARIANT_DIR = Path("/root/npi/models/latent-flow-v1")
sys.path.insert(0, str(VARIANT_DIR))

from ec_mild_unbiased import (  # noqa: E402
    _HIST_XI_SEED_OFFSET,
    _caliber_asserts,
    build_hist_block_xi,
    parse_args,
)
from ec_mild_unbiased import HistBlockProbeSite  # noqa: E402

N_TRIALS, HIST_LEN, GROUP_SIZE, N_GROUPS, SEED = 30, 375, 25, 15, 42
PASS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    assert cond, f"[FAIL] {name} {detail}"
    PASS.append(name)
    print(f"[PASS] {name}" + (f" | {detail}" if detail else ""), flush=True)


# --- V4 前置：构造性基抽取（供 V3/V4 复用同一生成序） -------------------------
gen = torch.Generator().manual_seed(SEED + _HIST_XI_SEED_OFFSET)
base_ref = torch.randn(N_TRIALS, N_GROUPS, generator=gen)

xi = build_hist_block_xi(N_TRIALS, HIST_LEN, GROUP_SIZE, seed=SEED, device="cpu")
check("V0 形状", tuple(xi.shape) == (N_TRIALS, HIST_LEN), f"{tuple(xi.shape)}")
check("V0 构造性", torch.equal(xi, base_ref.repeat_interleave(GROUP_SIZE, dim=1)),
      "与独立复刻的 randn(30,15)+repeat_interleave 逐位一致")

# --- V1 通道共享：provider 忽略 source，跨源同一引用 ---------------------------
def provider(src, _xi=xi):
    return _xi


site = HistBlockProbeSite.__new__(HistBlockProbeSite)   # 免模型实例化（只测 override）
site.N, site.C = N_TRIALS, 128
site.x_obs = torch.zeros(N_TRIALS, HIST_LEN)
site._xi_provider = provider
site.set_source(0)
xi_src0 = site._xi
site.set_source(1)
xi_src1 = site._xi
site.set_source(127)
xi_src127 = site._xi
check("V1 同被试跨通道逐元素相同",
      xi_src0 is xi and torch.equal(xi_src0, xi_src1)
      and torch.equal(xi_src0, xi_src127),
      "src=0/1/127 ξ 同引用且 torch.equal（全 128 源同一构造）")


def bad_provider(src):
    return torch.zeros(N_TRIALS, HIST_LEN - 1)


site._xi_provider = bad_provider
try:
    site.set_source(0)
    raised = False
except AssertionError:
    raised = True
check("V1b 形状断言", raised, "错误形状 provider 被 set_source 断言拦截")

# --- V2 被试共享：两次独立同参数重建逐位一致 -----------------------------------
xi_rebuild = build_hist_block_xi(N_TRIALS, HIST_LEN, GROUP_SIZE, seed=SEED,
                                 device="cpu")
check("V2 跨被试逐元素相同（独立重建）", torch.equal(xi, xi_rebuild),
      "同 seed 同形状两次构造 torch.equal ⇒ 跨被试切片自同一构造逐位一致")
xi_subj5 = build_hist_block_xi(N_TRIALS, HIST_LEN, GROUP_SIZE, seed=SEED,
                               device="cpu", subject_key=5)
check("V2b 被试键控退化", not torch.equal(xi, xi_subj5),
      "--no-hist-share-subjects 路径产出不同构造（键控生效）")
xi_src9 = build_hist_block_xi(N_TRIALS, HIST_LEN, GROUP_SIZE, seed=SEED,
                              device="cpu", source_key=9)
xi_src9_subj5 = build_hist_block_xi(N_TRIALS, HIST_LEN, GROUP_SIZE, seed=SEED,
                                    device="cpu", source_key=9, subject_key=5)
check("V2c 源键控退化 + 两键正交叠加",
      not torch.equal(xi_src9, xi) and not torch.equal(xi_src9, xi_src9_subj5),
      "--no-hist-share-channels 生效，且与被试键叠加互不抵消")

# --- V3 组内 25 点恒等 --------------------------------------------------------
ok = all(
    torch.equal(xi[:, g * GROUP_SIZE:(g + 1) * GROUP_SIZE],
                xi[:, g * GROUP_SIZE:g * GROUP_SIZE + 1].expand(N_TRIALS, GROUP_SIZE))
    for g in range(N_GROUPS))
check("V3 组内完全相同（逐组 15/15）", ok,
      "每组 25 个连续采样点 == 该组首点广播；等价式 "
      "xi == xi[:, ::25].repeat_interleave(25, 1) 亦成立："
      f"{torch.equal(xi, xi[:, ::GROUP_SIZE].repeat_interleave(GROUP_SIZE, dim=1))}")

# --- V4 15 组且组间独立 -------------------------------------------------------
check("V4a 组数", xi.shape[1] // GROUP_SIZE == N_GROUPS and HIST_LEN == 15 * GROUP_SIZE,
      "375 = 15 组 × 25 点")
reps = xi[:, ::GROUP_SIZE]                     # (N, 15) 组代表
check("V4b 组代表 = 基抽取", torch.equal(reps, base_ref),
      "ξ[:, ::25] 与独立 randn(30,15) 逐位一致（组间来自独立抽样，构造性）")
big = build_hist_block_xi(200_000, HIST_LEN, GROUP_SIZE, seed=SEED + 1,
                          device="cpu")[:, ::GROUP_SIZE].numpy()
corr = np.corrcoef(big.T)
off = corr[~np.eye(N_GROUPS, dtype=bool)]
check("V4c 组间独立 sanity", float(np.abs(off).max()) < 0.01,
      f"N=200k 组代表 15×15 Pearson |offdiag|max = {np.abs(off).max():.4f} < 0.01")

# --- V5 CRN 结构 + 口径断言负例（AC6） ----------------------------------------
site._xi_provider = provider
site.set_source(3)
d1 = site._xi
site.set_source(3)                              # 同源重进（模拟跨 σ 循环再入）
check("V5a CRN 结构", site._xi is d1 and torch.equal(site._xi, d1),
      "同源重复 set_source 幂等；ξ 引用不变 ⇒ 跨 σ 逐位一致（继承结构）")

a_ok = parse_args(["--mode", "main", "--out-dir", "/tmp/x",
                   "--noise-mode", "hist-block"])
_caliber_asserts(a_ok)                          # 默认 375/25 通过
try:
    _caliber_asserts(parse_args(["--mode", "main", "--out-dir", "/tmp/x",
                                 "--noise-mode", "hist-block",
                                 "--hist-len", "350"]))
    raised = False
except AssertionError:
    raised = True
check("V5b hist_len != obs_len 拒跑", raised)
try:
    _caliber_asserts(parse_args(["--mode", "main", "--out-dir", "/tmp/x",
                                 "--noise-mode", "hist-block",
                                 "--hist-group-size", "20"]))
    raised = False
except AssertionError:
    raised = True
check("V5c 非整除 group-size 拒跑", raised, "375 % 20 != 0")
a_iid = parse_args(["--mode", "main", "--out-dir", "/tmp/x"])
check("V5d 默认面", a_iid.noise_mode == "iid" and a_iid.hist_len == 375
      and a_iid.hist_group_size == 25 and a_iid.hist_share_channels
      and a_iid.hist_share_subjects, "5 旗标默认 = iid/375/25/共享/共享")

print(f"\nALL {len(PASS)} CHECKS PASS", flush=True)
