#!/usr/bin/env python3
"""test_perturb_search.py — perturb-search-v1 B0 契约单测（implement.md B0 ①–⑧）。

① r=0 六算子全部逐位 == baseline（短路 clone 路径）
② 扰动仅改源通道（其余 61 通道逐位断言）+ 波形事实（N 带限/RMS、P 支集、B 零相位带通）
③ 显式 eps/eta 同输入两次调用逐位相同（GPU 段，无 CUDA 跳过）
④ 度量合成数据 sanity（构造已知差 → D>0 含解析值；恒等输入 → D=0；null 同构）
⑤ 批构造行索引/逆映射（559/1737 两批形，r 网格化）
⑥ `sampler.gen_lowpass_hz is None`（fs=500 brickwall 坑防重开）+ 装载后 config_hash 校验
⑦ 输出级恒等：测试批（baseline 行 + 一行 r=0 变体）单次 sample() 后两行逐位相等
   （AC2 第一款的采样器级读法，同批形内成立）
⑧ Gate D 单调性：r 网格推导复算（倍率梯×r0 → 钳制 → 升序去重）+ 判定器合成数据
   （升势过 / 注入下降抓到该步 / 常数曲线过——「不减」含等号）

运行：cd models/perturb-search-v1 && python3 -m pytest tests/ -q（GPU 段需 CUDA）
"""

from __future__ import annotations

import os

# BLAS 护栏（environment.md 惯例：numpy/torch 导入前；有效 16 核 cgroup 限 8 线程，可覆盖）
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "8")

import sys
from pathlib import Path

import numpy as np
import torch

_TESTS = Path(__file__).resolve().parent
_PKG = _TESTS.parent                       # models/perturb-search-v1
_MODELS = _PKG.parent
_V1 = _MODELS / "gpfm-mamba-v1"

for _p in (str(_PKG), str(_V1)):           # 先插包目录再插 v1 → v1 在最前（`config` 名留给 v1）
    if _p not in sys.path:
        sys.path.insert(0, _p)

import candidates as CD   # noqa: E402  本目录（算子 + crn_noise + search_config 唯一装载）
import loader as LD       # noqa: E402  触发 v1 sample/data/model/gpr_prior 只读装配
import metrics as MT      # noqa: E402
import scorecard as SG    # noqa: E402

SC = CD.load_config()
_CACHE: dict = {}


def _frozen():
    """GPU 冻结装载缓存（load_eval_model + prior + sampler；多次断言复用同实例）。"""
    assert torch.cuda.is_available(), "GPU 段需 CUDA（mamba_ssm kernel 必须 CUDA）"
    if "frozen" not in _CACHE:
        _CACHE["frozen"] = LD.load_frozen_model("cuda")
    return _CACHE["frozen"]


def _test_ds():
    """fold0 test 数据集缓存（iterate_test_batches 内重算训练折统计，加载一次复用）。"""
    if "test_ds" not in _CACHE:
        _CACHE["test_ds"] = LD.load_test_loader(batch_size=2)[1]
    return _CACHE["test_ds"]


def _synth_window(seed: int = 2026) -> torch.Tensor:
    """合成窗 (1, 62, 384)（CRN 契约与数据无关）。"""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, SC.N_CHANNELS, SC.PAST_LEN, generator=g)


# ---------------------------------------------------------------------------
# ① r=0 六算子逐位恒等（短路 clone）
# ---------------------------------------------------------------------------

def test_01_r0_bitwise_identity() -> None:
    x = _synth_window()
    x3 = x.expand(3, -1, -1).contiguous()
    for key in SC.PARADIGMS:
        out = CD.apply_paradigm(key, x, 5, 0.0)
        assert out is not x and torch.equal(out, x), f"{key} r=0 非逐位恒等（短路 clone 契约）"
        out3 = CD.apply_paradigm(key, x3, 5, 0.0)
        assert torch.equal(out3, x3), f"{key} r=0 批形非逐位恒等"
    print("[① r=0] 六算子 r=0 短路 clone 全部逐位恒等（单窗 + B=3 批形）")


# ---------------------------------------------------------------------------
# ② 只动源通道 + 波形事实
# ---------------------------------------------------------------------------

def test_02_only_source_channel_touched() -> None:
    x = _synth_window()
    a = 17
    others = [c for c in range(SC.N_CHANNELS) if c != a]
    for key in SC.PARADIGMS:
        out = CD.apply_paradigm(key, x, a, SC.R0[key])
        assert torch.equal(out[0, others], x[0, others]), \
            f"{key} 改动了非源通道（须只动通道 {a}）"
        assert not torch.equal(out[0, a], x[0, a]), f"{key} 未改动源通道（r0={SC.R0[key]} 应非零）"
    # 波形事实：N = 1–40 Hz 带限 + RMS=1 + 全局共享
    g = CD.noise_waveform()
    assert CD.noise_waveform() is g, "N 波形须全局共享单实例"
    gn = g.numpy().astype(np.float64)
    assert np.isclose(np.sqrt((gn ** 2).mean()), 1.0, rtol=1e-6), "N 波形 RMS != 1"
    spec = np.abs(np.fft.rfft(gn)) ** 2
    fr = np.fft.rfftfreq(SC.NOISE_LEN, d=1.0 / SC.FS)
    band_pow = spec[(fr >= 1.0) & (fr <= 40.0)].sum() / spec.sum()
    assert band_pow > 1 - 1e-9, f"N 波形带外（1–40 Hz 外）能量占比 {1 - band_pow:.2e} 应≈0"
    # P = 汉宁脉冲支集 [334, 354)，端点为零
    h = CD.pulse_waveform()
    start = SC.PAST_LEN - SC.PULSE_CENTER_OFFSET - SC.PULSE_LEN // 2
    nz = torch.nonzero(h).flatten()
    assert nz.min() >= start and nz.max() < start + SC.PULSE_LEN and h[0] == 0 and h[-1] == 0, \
        "P 脉冲支集/端点契约破坏"
    # L 代数形式精确重建：x_A − r·(x_A − μ_A)
    out = CD.apply_paradigm("L", x, a, SC.R0["L"])
    xa = x[0, a]
    expect = xa - SC.R0["L"] * (xa - xa.mean())
    assert torch.allclose(out[0, a], expect, rtol=1e-6, atol=1e-7), "L 代数形式失配"
    # B 零相位带通：带内 10 Hz 正弦增益≈(1+r)，DC 与 30 Hz 近零
    t = torch.arange(SC.PAST_LEN, dtype=torch.float64) / SC.FS
    for freq, sig, keep in ((10.0, torch.sin(2 * np.pi * 10.0 * t).float(), True),
                            (0.0, torch.ones(SC.PAST_LEN), False),
                            (30.0, torch.sin(2 * np.pi * 30.0 * t).float(), False)):
        bp = CD.alpha_bandpass(sig.unsqueeze(0))
        mid = bp[0, SC.PAST_LEN // 4: 3 * SC.PAST_LEN // 4]      # 去边缘
        ref = sig[SC.PAST_LEN // 4: 3 * SC.PAST_LEN // 4]
        if keep:
            assert torch.allclose(mid, ref, atol=0.05), f"{freq} Hz 带内信号被衰减（零相位带通失配）"
        else:
            assert mid.abs().max() < 0.05, f"{freq} Hz 带外信号泄漏（α 带通失配）"
    print(f"[② 源通道] 六算子只动通道 {a}（其余 61 通道逐位不动）；N 波形带限/RMS=1/全局共享；"
          f"P 支集 [{start},{start + SC.PULSE_LEN})；L 代数形式；B 零相位带通全中")


# ---------------------------------------------------------------------------
# ③ 显式 eps/eta 两次调用逐位相同（GPU）
# ---------------------------------------------------------------------------

def test_03_crn_determinism() -> None:
    if not torch.cuda.is_available():
        print("[③ SKIP] 无 CUDA——CRN 确定性断言跳过")
        return
    _cfg, _m, ema, _p, sampler, _ck = _frozen()
    x = _synth_window().to("cuda")
    eps, eta = CD.crn_noise(7, SC.K, SC.N_CHANNELS, 128, "cuda")
    assert eps.shape == eta.shape == (1, SC.K, SC.N_CHANNELS, 128)
    with ema.swap(_m):
        o1 = sampler.sample(x, SC.K, eps=eps, eta=eta)
        o2 = sampler.sample(x, SC.K, eps=eps, eta=eta)
    assert torch.equal(o1, o2), "同 eps/eta 两次调用非逐位相同（确定性破坏）"
    # crn_noise 跨调用确定性（同 win_idx 同张量；跨阶段对照的根基）
    e2, h2 = CD.crn_noise(7, SC.K, SC.N_CHANNELS, 128, "cuda")
    assert torch.equal(eps, e2) and torch.equal(eta, h2), "crn_noise 同 win_idx 非逐位可复现"
    print(f"[③ CRN] 同 eps/eta 两次 sample() 逐位相同；crn_noise 同 win_idx 逐位可复现"
          f"（批形 {tuple(o1.shape)}）")


# ---------------------------------------------------------------------------
# ④ 度量合成 sanity（已知差 → 解析值；恒等 → 0；null 同构）
# ---------------------------------------------------------------------------

def test_04_metrics_synthetic() -> None:
    rng = np.random.default_rng(SC.SEED)
    K, C, F = SC.K, 5, 128
    base = rng.standard_normal((K, C, F))

    def _run(pert):
        samples = np.stack([base, np.broadcast_to(pert, base.shape).copy()])
        return MT.compute_metrics(samples, SC.METRICS)

    # 恒等输入 → 全部 D 精确为 0；null（K=16 两半相同构造下 h1==h2 恒等情形）
    res = _run(base)
    for m in SC.METRICS:
        assert (res[m]["D"] == 0).all(), f"{m} 恒等输入 D != 0"
    # 已知差 → D>0（M-env 对纯 DC 不敏感、M-FC 对等比缩放不变 ⇒ 逐度量构造）
    amp = 0.5
    s3 = MT.baseline_scale(base)[3]
    res = _run(base + np.eye(C)[3][None, :, None] * amp)        # 通道 3 加常数
    assert np.isclose(res["M-RMS"]["D"][0, 3], amp / s3, rtol=1e-10), \
        f"M-RMS 解析值失配 {res['M-RMS']['D'][0, 3]} != {amp / s3}"
    assert (res["M-RMS"]["D"][0, [0, 1, 2, 4]] == 0).all()
    # DC 平移对谱族不可见（Welch detrend='constant' 去均值）——数值零，登记为 O 范式解释注记
    assert res["M-PSD"]["D"][0].max() < 1e-6 and res["M-band"]["D"][0].max() < 1e-6, \
        "谱族对纯 DC 平移应数值零（detrend 去均值性质）"
    res = _run(base * 1.2)                                      # 全通道等比放大
    assert np.isclose(res["M-env"]["D"][0].mean(), 0.2, rtol=1e-8), \
        f"M-env 等比放大解析值失配 {res['M-env']['D'][0].mean()}"
    nb = int(np.sum((MT._welch_freqs() >= 1.0) & (MT._welch_freqs() <= 40.0)))
    expect = np.sqrt(nb) * np.log10(1.2 ** 2)
    assert np.allclose(res["M-PSD"]["D"][0], expect, rtol=1e-8), "M-PSD 等比放大解析值失配"
    assert np.allclose(res["M-band"]["D"][0], np.log10(1.2 ** 2), rtol=1e-8), \
        "M-band 等比放大解析值失配（五带同 |Δ|=log10(1.44)）"
    noise = rng.standard_normal((K, 1, F))
    pert = base.copy()
    pert[:, 4:5] += noise                                       # 仅通道 4 加不相关噪声
    res = _run(pert)
    d_fc = res["M-FC"]["D"][0]
    assert (d_fc >= 0).all() and d_fc[4] > 0, "M-FC 应非负且扰动通道行变化"
    assert d_fc[4] == d_fc.max(), \
        "M-FC 扰动通道行（全元素变）应大于非扰动行（仅含扰动通道的 1 元素变——网络级 caveat）"
    pulse = np.zeros((1, 1, F))
    pulse[:, :, :SC.MTIME_EARLY_SAMPLES] = 1.0
    res = _run(base + pulse * 1.0)                              # 前 32 样本加 1.0
    assert np.isclose(res["M-time"]["D"][0, 0], 1.0, rtol=1e-12), "M-time 早期均值解析值失配"
    assert np.isclose(res["M-time"]["mtime_full"][0, 0], SC.MTIME_EARLY_SAMPLES / F, rtol=1e-12)
    assert np.isclose(res["M-time"]["mtime_peak"][0, 0], 1.0, rtol=1e-12)
    # null：h1==h2（构造 K=16、后 8 样本复制前 8）→ 全度量 null == 0；随机半 → 0<null 有限
    b16 = rng.standard_normal((SC.K, 3, F))
    b16[SC.K // 2:] = b16[:SC.K // 2]
    resn = MT.compute_metrics(np.stack([b16, b16.copy()]), SC.METRICS)
    for m in SC.METRICS:
        assert (resn[m]["null"] == 0).all(), f"{m} null 恒等半 != 0"
    b16[SC.K // 2:] = rng.standard_normal((SC.K // 2, 3, F))
    resn = MT.compute_metrics(np.stack([b16, b16.copy()]), SC.METRICS)
    for m in SC.METRICS:
        assert (resn[m]["null"] > 0).all() and np.isfinite(resn[m]["null"]).all(), \
            f"{m} null 非正/非有限"
    print("[④ 度量] 恒等→0；M-RMS/M-env/M-PSD/M-band/M-time 解析值；M-FC 纯靶行选择；"
          "null 恒等半 0 / 随机半正有限（六度量全检）")


# ---------------------------------------------------------------------------
# ⑤ 批构造行索引/逆映射（559 / 1737 两批形，r 网格化）
# ---------------------------------------------------------------------------

def test_05_batch_index_map() -> None:
    x = _synth_window(seed=2027)
    for paradigms, n_rows in ((SC.PHASE1_PROBES, SC.PHASE1_BATCH_ROWS),
                              (SC.PHASE2_PARADIGMS, SC.PHASE2_BATCH_ROWS)):
        specs = CD.variant_specs(paradigms)
        n_rt = len(specs)
        assert n_rows == 1 + SC.N_SRC * n_rt
        vb = CD.build_variant_batch(x, paradigms)
        assert vb.shape == (n_rows, SC.N_CHANNELS, SC.PAST_LEN)
        assert torch.equal(vb[0], x[0]), "baseline 行须逐位复制输入窗"
        touched = (vb[1:] != x[0]).any(dim=-1)                   # (n_rows-1, 62)
        for v in range(1, n_rows):
            a, j = CD.variant_of(v, n_rt)
            assert CD.variant_index(a, j, n_rt) == v, \
                f"索引表不自洽: variant_of({v}) → (a={a}, j={j})"
            expect = torch.zeros(SC.N_CHANNELS, dtype=torch.bool)
            expect[a] = True
            assert torch.equal(touched[v - 1], expect), \
                f"变体 {v} ({specs[j]}, a={a}) 改动了非源通道"
        # G r0 格点行 == r0 单强度口径施加（Gate A/B/C r0 切片的批构造侧根基）；最低格点行同检
        for r in (SC.R0["G"], SC.R_GRID["G"][0]):
            j = specs.index((paradigms.index("G"), "G", r))
            v = CD.variant_index(9, j, n_rt)
            assert torch.equal(vb[v], CD.apply_gain(x, 9, r)[0]), f"G r={r} 格点行失配"
        # N r0 格点行公式重建
        jn = specs.index((paradigms.index("N"), "N", SC.R0["N"]))
        v = CD.variant_index(9, jn, n_rt)
        xa = x[0, 9]
        sig = xa.std()
        expect = xa + SC.R0["N"] * sig * CD.noise_waveform()
        assert torch.allclose(vb[v, 9], expect, rtol=1e-6, atol=1e-7), "N 行公式重建失配"
    print(f"[⑤ 批构造] 559/1737 行形（62 源 × 9/28 格点）；行 0 逐位 baseline；变体行只动源通道；"
          f"索引/逆映射自洽；G r0/最低格点行与 N r0 行公式重建")


# ---------------------------------------------------------------------------
# ⑥ gen_lowpass_hz is None + 装载 config_hash 校验
# ---------------------------------------------------------------------------

def test_06_loader_calibers() -> None:
    ck = torch.load(SC.CKPT_PATH, map_location="cpu", weights_only=False)
    assert ck["config_hash"] == SC.EXPECTED_CONFIG_HASH == "0dc0d5249c68", \
        f"ckpt config_hash {ck['config_hash']!r} != pin（错拿 checkpoint / hash 漂移）"
    cfg_mod = LD.load_v1fmt_config()
    assert Path(sys.modules["v1fmt_config"].__file__) == Path(SC.V1FMT_CONFIG_PATH)
    cfg = cfg_mod.build_config()
    assert cfg.config_hash() == ck["config_hash"], "装载 v1fmt config_hash 与 ckpt 失配"
    for name in ("sample", "data", "gpr_prior", "model"):        # 模块归属（遮蔽纪律）
        mod = sys.modules[name]
        assert Path(mod.__file__).resolve().parent == _V1, f"{name} 归属漂移: {mod.__file__}"
    assert "search_config" in sys.modules and sys.modules["search_config"] is SC
    lp_note = "sampler 断言随 GPU 段跳过"
    if torch.cuda.is_available():
        _cfg, _m, _e, _p, sampler, _ck2 = _frozen()
        assert sampler.gen_lowpass_hz is None, \
            "gen_lowpass_hz 须 None（fmt 口径；构造器默认 40.0 + fs=500 静默削波陷阱）"
        lp_note = "sampler.gen_lowpass_hz is None（断言过）"
    ds = _test_ds()
    ids = [int(e["subject_id"]) for e in ds.subjects]
    assert all(s in ids for s in SC.TEST_SUBJECTS_HEAD), \
        f"manifest 前 3 例缺失于 test_ds.subjects: {ids}"
    wins1 = LD.select_windows(ds)
    wins2 = LD.select_windows(ds)
    assert [tuple(w) for w in wins1] == [tuple(w) for w in wins2], "窗枚举非确定性"
    assert [w[0] for w in wins1] == list(range(SC.N_WINDOWS)), "win_idx 须为 0..23 固定枚举"
    assert [w[2] for w in wins1] == [sid for sid in SC.TEST_SUBJECTS_HEAD
                                     for _ in range(SC.N_WINDOWS_PER_SUBJECT)], \
        "win_idx 受试者序须按 manifest 顺序（10→0–7、11→8–15、12→16–23）"
    print(f"[⑥ 口径] ckpt hash {ck['config_hash']} == 装载 config hash；sample/data/gpr_prior/"
          f"model 归属 v1；search_config 唯一名；{lp_note}；窗枚举 0..23 确定（manifest 顺序 "
          f"{SC.TEST_SUBJECTS_HEAD}，数据集内部序 {ids[:3]}… 仅定位用）")


# ---------------------------------------------------------------------------
# ⑦ 输出级 r=0 恒等（同批形内）
# ---------------------------------------------------------------------------

def test_07_output_level_identity() -> None:
    if not torch.cuda.is_available():
        print("[⑦ SKIP] 无 CUDA——输出级恒等断言跳过")
        return
    _cfg, model, ema, _p, sampler, _ck = _frozen()
    x = _synth_window().to("cuda")
    eps, eta = CD.crn_noise(0, SC.K, SC.N_CHANNELS, 128, "cuda")
    for key in ("G", "N"):                                       # 抽两算子（短路路径同构）
        x2 = torch.cat([x, CD.apply_paradigm(key, x, 5, 0.0)], dim=0)   # baseline + r=0 变体
        with ema.swap(model):
            out = sampler.sample(x2, SC.K, eps=eps, eta=eta)
        assert out.shape == (2, SC.K, SC.N_CHANNELS, 128)
        assert torch.equal(out[0], out[1]), \
            f"{key} r=0 变体与 baseline 同批内非逐位相等（CRN 契约破坏）"
    print("[⑦ 输出级] r=0 变体与 baseline 单次 sample() 同批内逐位相等（G/N 抽查）")


# ---------------------------------------------------------------------------
# ⑧ Gate D 单调性（r 网格推导复算 + 判定器合成数据）
# ---------------------------------------------------------------------------

def test_08_gate_d_monotonicity() -> None:
    # 网格推导规则复算（倍率梯 × r0 → 逐算子钳制 → 升序去重）与 R_GRID 字面值逐位一致；
    # r_2 恒等（用户 2026-10-18 口径确认）：r0 必居格点序 2 = 网格中位格点（A/B/C 评估点）；
    # 批形状常量
    assert SC.R_GRID_MULT.index(1.0) == 2
    for p in SC.PARADIGMS:
        clamp = SC.R_CLAMP.get(p)
        grid = sorted({min(r, clamp) if clamp is not None else r
                       for r in (m * SC.R0[p] for m in SC.R_GRID_MULT)})
        assert tuple(grid) == SC.R_GRID[p], f"{p} R_GRID 与推导规则失配"
        assert SC.R0[p] in SC.R_GRID[p], f"{p} r0 不在网格上（A/B/C r0 切片不存在）"
        assert SC.R_GRID[p].index(SC.R0[p]) == 2, \
            f"{p} r0 不在格点序 2（r_2 恒等破坏——A/B/C 评估点须为网格中位格点）"
    assert SC.PHASE1_BATCH_ROWS == 559 and SC.PHASE2_BATCH_ROWS == 1737
    # 判定器：升势曲线过；注入下降恰好抓到该步；常数曲线过（「不减」含等号）；R≤1 不判
    rng = np.random.default_rng(7)
    R, W, A, B = 5, 6, 8, 8
    D = np.stack([(k + 1) * 0.1 + rng.standard_normal((W, A, B)) * 1e-3 for k in range(R)])
    ev = SG.gate_d(D)
    assert ev["pass"] and ev["viol_steps"] == [] and ev["n_steps"] == R - 1
    assert ev["edge_mono_frac"] > 0.99 and ev["curve"][0] < ev["curve"][-1]
    D[3] = D[3] * 0.5                       # 曲线 ≈ [.1,.2,.3,.2,.5] → 恰在步 2→3 下降
    ev = SG.gate_d(D)
    assert not ev["pass"] and ev["viol_steps"] == [2] and ev["edge_mono_frac"] < 1.0
    ev = SG.gate_d(np.ones((R, W, A, B)))   # 常数曲线：等号 = 不减 → 过
    assert ev["pass"] and ev["viol_steps"] == [] and ev["edge_mono_frac"] == 1.0
    assert SG.gate_d(D[:1])["pass"] is False   # R ≤ 1 无法判定（n_steps=0）
    print("[⑧ Gate D] 网格推导复算一致（钳制/去重/r0 居格点序 2 = r_2 恒等/批形 559·1737）；"
          "升势过；注入下降抓到步 2；常数曲线过（不减含等号）；R≤1 不判")


# ---------------------------------------------------------------------------
# ⑨ Gate B 合并边池配对回归（W≥3 + P=2 全路径）
# ---------------------------------------------------------------------------

def test_09_gate_b_spearman_merged_pool() -> None:
    """evaluate_candidate W≥3 双探针全路径：Gate B Spearman 的 dist 须随探针平铺——
    2026-10-18 phase1 主 Run 抓获的潜伏 bug（merged_d 7564 vs dist 3782 ValueError），
    本测以合成数据锁定 W≥3 + P=2 路径不再抛维度错且统计量有限。W=24 对齐 Gate C
    预注册划分（例 #1+#2 的 16 窗 vs 例 #3 的 8 窗，W≤16 会后半空池 NaN）。"""
    rng = np.random.default_rng(11)
    W, A = 24, 6
    D1 = rng.random((W, A, A))
    D2 = rng.random((W, A, A))
    null = rng.random((W, A)) + 0.1
    coords = rng.random((A, 3))
    ev = SG.evaluate_candidate([D1, D2], null, coords)
    assert np.isfinite(ev["gateB"]["spearman_dist"]), "Gate B Spearman 非有限（合并池配对破坏）"
    assert np.isfinite(ev["gateB"]["ratio"]) and np.isfinite(ev["gateC"]["rho"])
    assert ev["gateB"]["ratio"] > 0
    # 阴性诊断路径回归：cr 形 cand_rows（gateA_median_z 为列表 + 逐门 pass 键）不抛 KeyError
    # ——2026-10-18 phase1 第二次跑抓获（阴性路径首次被走到）。
    rows = [{"metric": "M1", "gateA_median_z": [1.0, 0.5], "gateB_ratio": 1.2,
             "gateB_pass": False, "gateC_rho": 0.3, "gateC_pass": True,
             "gateD_pass": False, "gateD_viol_steps": 1, "gateD_n_steps": 3,
             "all_gates_pass": False},
            {"metric": "M2", "gateA_median_z": [2.5, 0.1], "gateB_ratio": 0.9,
             "gateB_pass": False, "gateC_rho": 0.6, "gateC_pass": True,
             "gateD_pass": True, "gateD_viol_steps": 0, "gateD_n_steps": 3,
             "all_gates_pass": False}]
    txt = SG.negative_diagnosis(rows, "metric")
    assert "最佳可得 = M1" in txt and "maximin 达成率 0.50" in txt and "D 违反步" in txt, \
        "阴性诊断输出不符（最佳可得 = maximin 达成率最大者）"
    print("[⑨ 合并池] W≥3 双探针 evaluate_candidate 全路径过（Spearman dist 平铺配对；"
          "7564-vs-3782 回归锁定）；阴性诊断 cr 形行回归过")
