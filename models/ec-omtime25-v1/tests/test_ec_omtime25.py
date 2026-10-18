#!/usr/bin/env python3
"""test_ec_ompsd.py — ec-omtime25-v1 B0 契约单测（①–⑦；O 算子 + M-PSD）。

① N：r=0 逐位恒等 + 只动源通道 + 公式重建 + 波形带限/RMS=1/全局共享
② 批 497 行：行 0 逐位 baseline、变体行只动源、索引/逆映射、格点公式抽查
③ CRN：同 eps/eta 两次 sample() 逐位相同（GPU 段）
④ d_time：合成解析值（恒等 0 / 脉冲 1.0 / full/peak）
⑤ readout 合成全套：线性 dbar→slope 解析、AUC 梯形、D̃ 对角≡1 与 0 槽、r* 命中档/删失、
   flip、有效路径
⑥ loader 口径：ckpt hash / gen_lowpass None / 窗枚举 0..79 × 10 例 × 8
⑦ 输出级 r=0 恒等（GPU 段，同批形内）

运行：cd models/ec-omtime25-v1 && python3 tests/run_tests.py
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

_TESTS = Path(__file__).resolve().parent
_PKG = _TESTS.parent
_MODELS = _PKG.parent
_V1 = _MODELS / "gpfm-mamba-v1"

for _p in (str(_PKG), str(_V1)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import candidates as CD   # noqa: E402
import loader as LD      # noqa: E402
import metrics as MT     # noqa: E402
import readout as RO     # noqa: E402

SC = CD.load_config()
_CACHE: dict = {}


def _frozen():
    assert torch.cuda.is_available(), "GPU 段需 CUDA（mamba_ssm kernel 必须 CUDA）"
    if "frozen" not in _CACHE:
        _CACHE["frozen"] = LD.load_frozen_model("cuda")
    return _CACHE["frozen"]


def _test_ds():
    if "test_ds" not in _CACHE:
        _CACHE["test_ds"] = LD.load_test_loader(batch_size=2)[1]
    return _CACHE["test_ds"]


def _synth_window(seed: int = 2026) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, SC.N_CHANNELS, SC.PAST_LEN, generator=g)


# ① O 算子契约
def test_01_bias_operator() -> None:
    x = _synth_window()
    a = 17
    out = CD.apply_paradigm(x, a, 0.0)
    assert out is not x and torch.equal(out, x), "r=0 非逐位恒等（短路 clone 契约）"
    r = SC.R_SAMPLED[3]
    out = CD.apply_paradigm(x, a, r)
    others = [c for c in range(SC.N_CHANNELS) if c != a]
    assert torch.equal(out[0, others], x[0, others]), "O 改动了非源通道"
    xa = x[0, a]
    assert torch.allclose(out[0, a], xa + r * xa.std(), rtol=1e-6, atol=1e-7), "O 公式重建失配"
    print(f"[① O] r=0 恒等；只动通道 {a}；公式重建 x_A + r·σ_A")


# ② 批 497 行
def test_02_batch() -> None:
    x = _synth_window(seed=2027)
    n_r = len(SC.R_SAMPLED)
    assert SC.BATCH_ROWS == 497 == 1 + SC.N_SRC * n_r
    vb = CD.build_variant_batch(x)
    assert vb.shape == (SC.BATCH_ROWS, SC.N_CHANNELS, SC.PAST_LEN)
    assert torch.equal(vb[0], x[0]), "行 0 须逐位 baseline"
    touched = (vb[1:] != x[0]).any(dim=-1)
    for v in range(1, SC.BATCH_ROWS):
        a, j = CD.variant_of(v, n_r)
        assert CD.variant_index(a, j, n_r) == v, f"索引不自洽 variant_of({v})"
        expect = torch.zeros(SC.N_CHANNELS, dtype=torch.bool)
        expect[a] = True
        assert torch.equal(touched[v - 1], expect), f"变体 {v} (a={a}, j={j}) 动了非源通道"
        assert torch.equal(vb[v], CD.apply_paradigm(x, a, SC.R_SAMPLED[j])[0]), \
            f"变体 {v} 公式重建失配"
    print("[② 批] 497 行形；行 0 逐位 baseline；变体行只动源；索引/逆映射自洽；全格点公式重建")


# ③ CRN 确定性（GPU）
def test_03_crn_determinism() -> None:
    if not torch.cuda.is_available():
        print("[③ SKIP] 无 CUDA")
        return
    _cfg, _m, ema, _p, sampler, _ck = _frozen()
    x = _synth_window().to("cuda")
    eps, eta = CD.crn_noise(7, SC.K, SC.N_CHANNELS, 128, "cuda")
    with ema.swap(_m):
        o1 = sampler.sample(x, SC.K, eps=eps, eta=eta)
        o2 = sampler.sample(x, SC.K, eps=eps, eta=eta)
    assert torch.equal(o1, o2), "同 eps/eta 两次调用非逐位相同"
    e2, h2 = CD.crn_noise(7, SC.K, SC.N_CHANNELS, 128, "cuda")
    assert torch.equal(eps, e2) and torch.equal(eta, h2), "crn_noise 同 win_idx 非可复现"
    print(f"[③ CRN] 两次 sample() 逐位相同；crn_noise 可复现（批形 {tuple(o1.shape)}）")


# ④ M-time@25 合成解析（恒等 0；前 25 点脉冲 → 1.0；full/peak）
def test_04_m_time25() -> None:
    rng = np.random.default_rng(7)
    K, C, F = 16, 5, 128
    base = rng.standard_normal((K, C, F))
    res = MT.compute(np.stack([base, base.copy()]))
    assert (res["D"] == 0).all() and (res["mtime_full"] == 0).all()
    pulse = np.zeros((1, 1, F))
    pulse[:, :, :SC.MTIME_EARLY_SAMPLES] = 1.0
    pert = base.copy()
    pert[:, 3:4] += pulse
    res = MT.compute(np.stack([base, pert]))
    assert np.isclose(res["D"][0, 3], 1.0, rtol=1e-12), "主标量（前 25 点均值）解析失配"
    assert np.isclose(res["mtime_full"][0, 3], SC.MTIME_EARLY_SAMPLES / F, rtol=1e-12)
    assert np.isclose(res["mtime_peak"][0, 3], 1.0, rtol=1e-12)
    assert (res["D"][0, [0, 1, 2, 4]] == 0).all()
    assert SC.MTIME_EARLY_SAMPLES == 25
    print("[④ M-time@25] 恒等→0；前 25 点脉冲均值/full/peak 解析；窗常量 25")


# ⑤ readout 合成全套
def test_05_readout() -> None:
    g = len(SC.R_GRID)
    # slope：线性 dbar D̄ = 3r（任意末两轴广播）→ slope 解析 = 3
    dbar = np.asarray(SC.R_GRID) * 3.0
    sl = RO.slope_readout(np.broadcast_to(dbar, (4, 5, g)).copy())
    assert np.allclose(sl, 3.0, rtol=1e-10), f"线性 slope 解析失配 {sl}"
    sl4 = RO.slope_readout(np.broadcast_to(dbar, (4, 5, g)).copy(), seg=SC.SLOPE_SEG_SENS)
    assert np.allclose(sl4, 3.0, rtol=1e-10), "敏感性段 slope 失配"
    # AUC：梯形解析（线性 3r 的 ∫₀^{R_MAX} = 3·R_MAX²/2，除 R_MAX ⇒ 1.5·R_MAX）
    assert np.allclose(RO.auc_readout(np.broadcast_to(dbar, (2, g)).copy()),
                       1.5 * SC.R_MAX, rtol=1e-10), "AUC 梯形失配"
    # D̃：构造 dbar9 (n,n,g)：自响应 = r，A→B = 0.5r ⇒ D̃ = 0.5（r>0 槽）、对角 ≡1、0 槽 0
    n = 4
    d9 = np.zeros((n, g, n))
    for j, r in enumerate(SC.R_GRID):
        d9[:, j, :] = 0.5 * r
        d9[np.arange(n), j, np.arange(n)] = r
    dtil = RO.dtil_curves(d9)
    assert (dtil[:, :, 0] == 0).all(), "D̃ 0 槽须 0"
    assert np.allclose(dtil[np.arange(n), np.arange(n), 1:], 1.0), "D̃ 对角须 ≡1"
    off = ~np.eye(n, dtype=bool)
    assert np.allclose(dtil[off][:, 1:], 0.5), "D̃ offdiag 须 0.5"
    # r*：D̃=0.5 ⇒ τ̃=0.25 在首个 r>0 档命中；τ̃=0.6 全网格不达 → 删失 R_NEXT
    rs = RO.rstar_readout(dtil, 0.25)
    assert (rs[off] == SC.R_GRID[1]).all() and (rs[np.arange(n), np.arange(n)] == SC.R_GRID[1]).all()
    rs2 = RO.rstar_readout(dtil, 0.6)
    assert (rs2[off] == SC.R_NEXT).all(), "未达阈须右删失 R_NEXT"
    # flip：R_NEXT − r*；删失 → 0
    fl = RO.flip_rstar(rs2)
    assert (fl[off] == 0).all() and (fl >= 0).all()
    # 有效路径：r* = R_GRID[1] ≤ R_MAX ⇒ offdiag 全部有效
    paths = RO.effective_paths(rs)
    assert len(paths) == n * (n - 1) and all(a != b for a, b in paths)
    assert SC.R_NEXT == 5.0 and SC.R_MAX == 4.0 and SC.BATCH_ROWS == 497
    print("[⑤ readout] slope/AUC 解析、敏感性段、D̃ 对角≡1/0 槽、r* 命中/删失、flip、有效路径、常量")


# ⑥ loader 口径 + 窗枚举
def test_06_loader_calibers() -> None:
    ck = torch.load(SC.CKPT_PATH, map_location="cpu", weights_only=False)
    assert ck["config_hash"] == SC.EXPECTED_CONFIG_HASH == "0dc0d5249c68"
    cfg = LD.load_v1fmt_config().build_config()
    assert cfg.config_hash() == ck["config_hash"], "装载 config_hash 与 ckpt 失配"
    lp = "sampler 断言随 GPU 段跳过"
    if torch.cuda.is_available():
        _cfg, _m, _e, _p, sampler, _ck2 = _frozen()
        assert sampler.gen_lowpass_hz is None, "gen_lowpass_hz 须 None"
        lp = "sampler.gen_lowpass_hz is None（断言过）"
    ds = _test_ds()
    wins = LD.select_windows(ds)
    assert [w[0] for w in wins] == list(range(SC.N_WINDOWS)), "win_idx 须 0..79"
    assert [w[2] for w in wins] == [s for s in SC.TEST_SUBJECTS
                                    for _ in range(SC.N_WINDOWS_PER_SUBJECT)], \
        "win_idx 受试者序须按 manifest 顺序 × 8"
    assert [w[3] for w in wins] == list(range(SC.N_WINDOWS_PER_SUBJECT)) * len(SC.TEST_SUBJECTS), \
        "例内窗序须为 0..7 × 10 轮"
    print(f"[⑥ 口径] ckpt hash ✓；{lp}；窗枚举 0..79 = manifest 顺序 {SC.TEST_SUBJECTS} × 8")


# ⑦ 输出级 r=0 恒等（GPU，同批形内）
def test_07_output_level_identity() -> None:
    if not torch.cuda.is_available():
        print("[⑦ SKIP] 无 CUDA")
        return
    _cfg, model, ema, _p, sampler, _ck = _frozen()
    x = _synth_window().to("cuda")
    eps, eta = CD.crn_noise(0, SC.K, SC.N_CHANNELS, 128, "cuda")
    x2 = torch.cat([x, CD.apply_paradigm(x, 5, 0.0)], dim=0)
    with ema.swap(model):
        out = sampler.sample(x2, SC.K, eps=eps, eta=eta)
    assert torch.equal(out[0], out[1]), "r=0 变体与 baseline 同批内非逐位相等"
    print("[⑦ 输出级] r=0 变体与 baseline 单次 sample() 同批内逐位相等")
