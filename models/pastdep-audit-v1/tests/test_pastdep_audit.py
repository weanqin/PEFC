#!/usr/bin/env python3
"""test_pastdep_audit.py — pastdep-audit-v1 B0 契约单测（implement.md ①–⑥）。"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "8")

import sys
from pathlib import Path

import numpy as np
import torch

_TESTS = Path(__file__).resolve().parent
_PKG = _TESTS.parent
_MODELS = _PKG.parent
_V1 = _MODELS / "gpfm-mamba-v1"

for _p in (str(_PKG), str(_V1)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import candidates as CD   # noqa: E402
import loader as LD      # noqa: E402
from run import _energy_dist, _t50   # noqa: E402

SC = CD.load_config()


def _synth(seed: int = 2026) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, SC.N_CHANNELS, SC.PAST_LEN, generator=g)


def test_01_operators() -> None:
    x = _synth()
    a = 17
    assert torch.equal(CD.apply_paradigm(x, a, 0.0, ("dose", 0.0)), x), "r=0 非恒等"
    out = CD.apply_noise(x, a, 0.5)
    others = [c for c in range(SC.N_CHANNELS) if c != a]
    assert torch.equal(out[0, others], x[0, others]), "全段 N 动了非源通道"
    xa = x[0, a]
    assert torch.allclose(out[0, a], xa + 0.5 * xa.std() * CD.noise_waveform(),
                          rtol=1e-6, atol=1e-7), "全段公式失配"
    for sub in range(4):
        lo, hi = SC.SUBWINDOWS[sub]
        ow = CD.apply_noise_window(x, a, 1.0, sub)
        assert torch.equal(ow[0, others], x[0, others]), f"子窗 {sub} 动了非源通道"
        d = (ow[0, a] - xa)
        assert (d[:lo] == 0).all() and (d[hi:] == 0).all(), f"子窗 {sub} 支集外注入泄漏"
        rms = float(torch.sqrt((d[lo:hi] ** 2).mean()))
        assert abs(rms - float(xa.std())) / float(xa.std()) < 1e-3, \
            f"子窗 {sub} 注入 RMS {rms} != σ_A（重归一失配）"
    print("[① 算子] r=0 恒等；全段/子窗只动源通道；子窗支集无泄漏；RMS 重归一 = r·σ_A")


def test_02_batch() -> None:
    x = _synth(seed=2027)
    vb = CD.build_batch(x)
    assert vb.shape == (SC.BATCH_ROWS, SC.N_CHANNELS, SC.PAST_LEN) and SC.BATCH_ROWS == 435
    assert torch.equal(vb[0], x[0]), "行 0 非逐位 baseline"
    touched = (vb[1:] != x[0]).any(dim=-1)
    for v in range(1, SC.BATCH_ROWS):
        a, k = divmod(v - 1, SC.N_ARMS)
        assert CD.variant_index(a, k) == v, f"索引不自洽 {v}"
        expect = torch.zeros(SC.N_CHANNELS, dtype=torch.bool)
        expect[a] = True
        assert torch.equal(touched[v - 1], expect), f"行 {v} (a={a},k={k}) 动了非源通道"
    # 两臂抽查公式重建：k=0（dose 0.5 全段）与 k=3+2（win 2）
    assert torch.equal(vb[CD.variant_index(9, 0)], CD.apply_noise(x, 9, 0.5)[0])
    assert torch.equal(vb[CD.variant_index(9, 5)], CD.apply_noise_window(x, 9, 1.0, 2)[0])
    print("[② 批] 435 行形；索引/逆映射自洽；变体行只动源；剂量/子窗臂公式重建")


def test_03_crn() -> None:
    e1, h1 = CD.crn_noise(2030, SC.K, SC.N_CHANNELS, 128, "cpu")
    e2, h2 = CD.crn_noise(2030, SC.K, SC.N_CHANNELS, 128, "cpu")
    assert torch.equal(e1, e2) and torch.equal(h1, h2), "同 seed 非逐位可复现"
    e3, _ = CD.crn_noise(2030 + SC.ARM_SEED_OFF["same"], SC.K, SC.N_CHANNELS, 128, "cpu")
    assert not torch.equal(e1, e3), "臂偏置后噪声应不同"
    print("[③ CRN] 同 seed 逐位复现；臂偏置独立")


def test_04_e3_mapping() -> None:
    # 复算 run.py 的置换映射并核对性质（同人在例内、跨人换受试者）
    ds_of = {wi: 100 + wi for wi in range(SC.N_WINDOWS)}
    same = {wi: ds_of[(wi // 8) * 8 + (wi % 8 + 1) % 8] for wi in range(SC.N_WINDOWS)}
    cross = {wi: ds_of[((wi // 8 + 1) % 3) * 8 + wi % 8] for wi in range(SC.N_WINDOWS)}
    for wi in range(SC.N_WINDOWS):
        assert same[wi] // 1 != -1 and (same[wi] - 100) // 8 == wi // 8, "同人置换须留在同例"
        assert (cross[wi] - 100) // 8 != wi // 8, "跨人置换须换受试者"
        assert (cross[wi] - 100) % 8 == wi % 8, "跨人置换须同例内窗"
    print("[④ E3 映射] 同人例内循环 / 跨人换受试者同例内窗")


def test_05_loader() -> None:
    if not torch.cuda.is_available():
        print("[⑤ SKIP] 需 CUDA 装载段")
        return
    ck = torch.load(SC.CKPT_PATH, map_location="cpu", weights_only=False)
    assert ck["config_hash"] == SC.EXPECTED_CONFIG_HASH
    _cfg, _m, _e, prior, sampler, _ck2 = LD.load_frozen_model("cuda")
    assert prior.A.shape == (128, 384) and sampler.gen_lowpass_hz is None
    _lo, lds = LD.load_test_loader(batch_size=2)
    wins = LD.select_windows(lds)
    assert [w[0] for w in wins] == list(range(24))
    assert [w[2] for w in wins] == [s for s in SC.TEST_SUBJECTS for _ in range(8)]
    print("[⑤ loader] hash/prior.A(128×384)/gen_lowpass=None/窗枚举 0..23 ✓")


def test_06_metrics() -> None:
    rng = np.random.default_rng(5)
    x = rng.standard_normal((16, 128))
    y0 = rng.standard_normal((16, 128))          # 同分布独立样本 → 能量距离 ≈ 0（采样涨落内）
    assert _energy_dist(x, y0) < 0.2, "同分布两组能量距离应近 0"
    y = x + 1.0
    assert _energy_dist(x, y) > 1.0, "平移 1.0 组距离应 ≈1"
    c = np.exp(-np.arange(128) / 6.0)          # 6·ln2≈4.16 → 首个 ≤0.5 的整数格点 f=5
    assert _t50(c) == 5.0, f"T50 解析失配 {_t50(c)}（应 5）"
    c2 = np.concatenate([np.zeros(100), np.array([1.0, 0.5])])   # 峰在 100，半衰 101
    assert _t50(c2) == 101.0
    print("[⑥ 指标] 能量距离同分布近 0/平移≈1；T50 解析值")
