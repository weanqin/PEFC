#!/usr/bin/env python3
"""test_perturb_ec.py — perturb-ec-v1 B0 单测（implement.md B0；①②③断言惯例，rng 2026）。

① CRN 契约（GPU 段，无 CUDA 自动跳过）：r=0 变体与 baseline **同一 sample() 调用同批内**
   逐位相等；扰动仅改源通道输入行（变体批 497 行全量校验）；同 eps/eta 两次调用逐位相同。
② 口径：importlib 装载的 v4afmt config `config_hash()` == ckpt 内 hash（先 torch.load 读
   ckpt 比对）；sys.modules 中 sample/data/gpr_prior/model `__file__` 归属 gpfm-mamba-v1、
   `config`（若存在）亦归属 v1；`posterior_mean` 扰动差 B≠A 通道**严格为零**（先验逐通道
   事实，design §2.1 A-M1；GPRPrior CPU 构造，coords=chan_coords_62.npy，p/f/period_p 取自
   v4afmt cfg）；`sampler.gen_lowpass_hz is None`（B-Major1，GPU 段）。
③ readout/response 数学（合成数据，不需模型）：D̄=2r → slope=2、AUC=解析值 R_MAX；永不达 τ
   → r*=R_NEXT 右删失 → strength=0；翻转/R_NEXT 由 R_GRID 推导（pin 2.56）；有效路径集；
   load_ec_pert 契约 npz 往返（**不 import baseline-ec-v1**，字段按 design §5 抄写断言）；
   D 曲线 / null 距离解析值（构造已知幅度合成张量）。

运行：python3 models/perturb-ec-v1/tests/run_tests.py（或本文件直跑；GPU 段需 CUDA——
mamba_ssm kernel 必须 CUDA；末尾附 ckpt 装载 GPU 冒烟：1 batch 采样 shape/finite + 497
变体并批 + 逐被试窗子采样确定性）。
"""

from __future__ import annotations

import os

# BLAS 护栏（environment.md 惯例：numpy/torch 导入前；有效 16 核 cgroup 限 8 线程，可覆盖）
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "8")

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

_TESTS = Path(__file__).resolve().parent
_PKG = _TESTS.parent                       # models/perturb-ec-v1
_MODELS = _PKG.parent
_V1 = _MODELS / "gpfm-mamba-v1"

for _p in (str(_PKG), str(_V1)):           # 先插包目录再插 v1 → v1 在最前（`config` 名留给 v1）
    if _p not in sys.path:
        sys.path.insert(0, _p)

import perturb as PB       # noqa: E402  本目录（A 案算子 + config 唯一装载）
import response as RS      # noqa: E402
import readout as RO       # noqa: E402
import loader as LD        # noqa: E402  触发 v1 sample/data/model/gpr_prior 导入
import gpr_prior as GP     # noqa: E402  v1（posterior_mean 逐通道事实断言用）

PC = PB.load_config()
_CACHE: dict = {}


def _frozen():
    """GPU 冻结装载缓存（load_eval_model + prior + sampler；多次断言复用同实例）。"""
    assert torch.cuda.is_available(), "GPU 段需 CUDA（mamba_ssm kernel 必须 CUDA）"
    if "frozen" not in _CACHE:
        _CACHE["frozen"] = LD.load_frozen_model("cuda")
    return _CACHE["frozen"]


def _test_data():
    """fold0 test loader 缓存（iterate_test_batches 内重算训练折统计，加载一次复用）。"""
    if "test_data" not in _CACHE:
        _CACHE["test_data"] = LD.load_test_loader(batch_size=2)
    return _CACHE["test_data"]


def test_crn_contract() -> None:
    """① CRN：r=0 同批逐位恒等 / 扰动仅动源通道行 / 同 eps/eta 两次调用逐位相同。"""
    # ---- 纯算子段（CPU）：r=0 数值恒等 + 变体批形状/索引表/只动源通道 ----
    g = torch.Generator().manual_seed(PC.SEED)
    x = torch.randn(1, PC.N_SRC, 384, generator=g)          # 合成窗（CRN 契约与数据无关）
    assert torch.equal(PB.apply_gain(x, 5, 0.0), x), "r=0 数值恒等失败（×1.0 位保持）"
    vb = PB.build_variant_batch(x)
    assert vb.shape == (PC.N_VARIANTS, PC.N_SRC, 384) and PC.N_VARIANTS == 497
    assert torch.equal(vb[0], x[0]), "baseline 行须逐位复制输入窗"
    touched = (vb[1:] != x[0]).any(dim=-1)                   # (496, 62) 每变体改了哪些通道
    for v in range(1, PC.N_VARIANTS):
        a, j = PB.variant_of(v)
        assert PB.variant_index(a, j) == v, f"索引表不自洽: variant_of({v}) → (a={a}, j={j})"
        expect = torch.zeros(PC.N_SRC, dtype=torch.bool)
        expect[a] = True
        assert torch.equal(touched[v - 1], expect), \
            f"变体 {v} (a={a}, j={j}) 改动了非源通道（须只动通道 {a}）"
        assert torch.allclose(vb[v, a], x[0, a] * (1.0 + PC.PERT_R_GRID[j]),
                              rtol=1e-6, atol=1e-7), \
            f"变体 {v} 通道 {a} 增益失配（应 ×(1+r)，r={PC.PERT_R_GRID[j]}；" \
            "直接对乘积比较避免差值抵消）"

    # ---- 采样段（GPU）：同批恒等 + 确定性（无 CUDA 跳过，打印注明）----
    if not torch.cuda.is_available():
        print("[① SKIP] 无 CUDA——CRN 采样断言（同批恒等/两次调用逐位相同）跳过")
        return
    cfg, model, ema, prior, sampler, _ck = _frozen()
    K, C, F = PC.K, PC.N_SRC, int(cfg.future_len)
    xg = x.to("cuda")
    eps, eta = RS.crn_noise(0, K, C, F, "cuda")
    assert eps.shape == eta.shape == (1, K, C, F)
    x2 = torch.cat([xg, PB.apply_gain(xg, 5, 0.0)], dim=0)   # 同批 baseline + r=0 变体（B=2）
    with ema.swap(model):
        out = sampler.sample(x2, K, eps=eps, eta=eta)
        out_again = sampler.sample(x2, K, eps=eps, eta=eta)  # 同输入两次调用
    assert out.shape == (2, K, C, F) and torch.isfinite(out).all()
    assert torch.equal(out[0], out[1]), "r=0 变体与 baseline 同批内非逐位相等（CRN 契约破坏）"
    assert torch.equal(out, out_again), "同 eps/eta 两次调用非逐位相同（确定性破坏）"
    print(f"[① CRN] r=0 同批逐位恒等；496 变体只动各自源通道；同 eps/eta 两次调用逐位相同"
          f"（B=2, K={K}, 批形 {tuple(out.shape)}）")


def test_calibers() -> None:
    """② 口径：hash / 模块归属 / config 名消遮蔽 / posterior_mean 逐通道零差 / 低通口径。"""
    # ---- ckpt hash 比对（先 torch.load 读 ckpt，再对 importlib 装载 config）----
    ck = torch.load(PC.CKPT_PATH, map_location="cpu", weights_only=False)
    assert ck["config_hash"] == PC.EXPECTED_CONFIG_HASH == "05d9bb875a2c", \
        f"ckpt config_hash {ck['config_hash']!r} != pin（错拿 checkpoint / hash 漂移）"
    cfg_mod = LD.load_v4afmt_config()
    assert Path(sys.modules["v4afmt_config"].__file__) == Path(PC.V4AFMT_CONFIG_PATH)
    cfg = cfg_mod.build_config()
    assert cfg.config_hash() == ck["config_hash"], \
        "装载 v4afmt config_hash 与 ckpt 失配（v4a-fmt config.py 被改动？——config_hash 逐位不可变）"
    assert ck["epoch"] == 44, f"ckpt epoch {ck['epoch']} != 44（best ep45 0-based）"
    # ---- 模块归属（遮蔽纪律）：v1 四模块 + config 名 ----
    for name in ("sample", "data", "gpr_prior", "model"):
        mod = sys.modules[name]
        assert Path(mod.__file__).resolve().parent == _V1, f"{name} 归属漂移: {mod.__file__}"
    assert "perturb_config" in sys.modules and "v4afmt_config" in sys.modules
    if "config" in sys.modules:                              # sample.py 顶层 import 消费者
        assert Path(sys.modules["config"].__file__).resolve().parent == _V1, \
            f"`config` 名被非 v1 模块占据: {sys.modules['config'].__file__}（消遮蔽纪律破坏）"
    # ---- r 网格 / 档位推导（caliber 锁）----
    rg = np.asarray(PC.R_GRID)
    assert rg[0] == 0.0 and (np.diff(rg) > 0).all() and PC.PERT_R_GRID == tuple(rg[1:])
    assert np.allclose(np.asarray(PC.PERT_R_GRID[1:]) / PC.PERT_R_GRID[:-1], 2.0), \
        "采样档非对数等比 ×2（低剂量加密网格契约）"
    assert PC.R_NEXT == PC.R_MAX * (PC.R_MAX / PC.R_GRID[-2]) == 2.56   # 推导 + pin
    assert PC.SLOPE_SEG == (0.0, 0.01, 0.02, 0.04, 0.08) and PC.SLOPE_SEG_SENS == (0.0, 0.01, 0.02, 0.04)
    assert PC.TAU_MAIN == 0.95 and PC.TAU_MAIN in PC.TAU_QUANTILES and PC.TAU_QUANTILES == (0.90, 0.95, 0.99)
    assert (PC.K, PC.N_W, PC.SEED, PC.N_SRC) == (8, 32, 2026, 62)
    # ---- posterior_mean 逐通道事实（design §2.1 A-M1；CPU 构造 GPRPrior）----
    coords = np.load(f"{cfg.data_root}/chan_coords_62.npy")
    prior = GP.GPRPrior(
        coords, past_len=cfg.past_len, future_len=cfg.future_len,
        kernel_t=cfg.kernel_t, gamma=cfg.gamma, period_p=cfg.period_p,
        iso=cfg.iso, pp_jitter=cfg.pp_jitter, sigma_s=cfg.sigma_s,
        jitter_s=cfg.jitter_s, bandlimit_hz=getattr(cfg, "prior_bandlimit_hz", None),
    )
    gt = torch.Generator().manual_seed(PC.SEED)
    xa = torch.randn(3, cfg.n_channels, cfg.past_len, generator=gt)
    A_CH, R = 17, 0.37
    mu0 = prior.posterior_mean(xa)
    mu1 = prior.posterior_mean(PB.apply_gain(xa, A_CH, R))
    diff = mu1 - mu0
    others = [b for b in range(cfg.n_channels) if b != A_CH]
    assert (diff[:, others] == 0).all(), "B≠A 通道先验差非严格为零（posterior_mean 逐通道事实破坏）"
    assert torch.allclose(mu1[:, A_CH], (1.0 + R) * mu0[:, A_CH], rtol=1e-5), \
        "源通道先验均值非线性外推失配（应 ×(1+r)）"
    # ---- 采样器低通口径（B-Major1；GPU 段）----
    if torch.cuda.is_available():
        _, _, _, _, sampler, _ = _frozen()
        assert sampler.gen_lowpass_hz is None, \
            "gen_lowpass_hz 须 None（fmt 口径；构造器默认 40.0 + fs=500 静默削波陷阱）"
        lp_note = "sampler.gen_lowpass_hz is None（B-Major1 断言过）"
    else:
        lp_note = "sampler 低通断言随 GPU 段跳过"
    print(f"[② 口径] ckpt hash {ck['config_hash']} == 装载 config hash；epoch={ck['epoch']}；"
          f"sample/data/gpr_prior/model/config 全归属 v1；perturb_config/v4afmt_config 唯一名装载；"
          f"R_NEXT={PC.R_NEXT} 由网格推导；posterior_mean B≠A 差严格为零；{lp_note}")


def test_readout_math() -> None:
    """③ readout/response 合成数学（不需模型；D̄=2r 解析对照 + 契约 npz 往返）。"""
    rg = np.asarray(PC.R_GRID, dtype=np.float64)
    dbar = 2.0 * rg                                          # 索引 0 ≡ 0（构造 D̄(0)=0）
    # slope（主段/敏感性段）与 AUC 解析值：∫2r dr / r_max = r_max
    assert np.isclose(RO.slope_readout(dbar), 2.0, rtol=1e-10)
    assert np.isclose(RO.slope_readout(dbar, seg=PC.SLOPE_SEG_SENS), 2.0, rtol=1e-10)
    assert np.isclose(RO.auc_readout(dbar), PC.R_MAX, rtol=1e-10)
    D = np.broadcast_to(2.0 * rg, (PC.N_SRC, PC.N_SRC, 9)).copy()   # 受试者级曲线形状
    assert RO.slope_readout(D).shape == (62, 62) and np.allclose(RO.slope_readout(D), 2.0)
    assert RO.rstar_readout(D, 0.05).shape == (62, 62)
    # r*：首达档 / 永不达 τ → R_NEXT 右删失；翻转 = R_NEXT − r*，删失 → 0
    assert RO.rstar_readout(dbar, tau=0.05) == 0.04, "首达档失配（2r≥0.05 应首达 r=0.04）"
    assert RO.rstar_readout(dbar, tau=3.0) == PC.R_NEXT, "右删失编码失配（应 R_NEXT=2.56）"
    assert RO.flip_rstar(PC.R_NEXT) == 0.0, "删失翻转应 = 0 最弱"
    assert np.isclose(RO.flip_rstar(0.04), PC.R_NEXT - 0.04) and np.isclose(RO.flip_rstar(0.04), 2.52)
    # 有效路径集：A≠B、r* ≤ R_MAX（恰上界含、对角不含、删失不含）
    rstar = np.full((62, 62), PC.R_NEXT)
    rstar[3, 40] = 0.08
    rstar[7, 9] = PC.R_MAX
    rstar[5, 5] = 0.04                                       # 对角（A=B）不算路径
    assert RO.effective_paths(rstar) == [(3, 40), (7, 9)]
    # load_ec_pert 契约 npz 往返（字段按 design §5 抄写；不 import baseline-ec-v1）
    rng = np.random.default_rng(PC.SEED)
    ec = rng.random((62, 62))
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "ec_pert_slope_test.npz"
        RO.save_ec_pert_npz(p, ec, "slope", np.array([10, 11]))
        d = np.load(p, allow_pickle=True)
        assert {"ec", "readout", "subject_ids", "orientation"} <= set(d.files), \
            f"契约字段缺失: {sorted(d.files)}"
        assert np.array_equal(d["ec"], ec) and (d["ec"] >= 0).all()
        assert str(d["readout"]) == "slope" and str(d["orientation"]) == "larger_is_stronger"
        assert list(d["subject_ids"]) == [10, 11]
        for ro in ("auc", "rstar"):                          # readout 合法域 + global 口径
            RO.save_ec_pert_npz(Path(td) / f"ec_{ro}.npz", ec, ro, "global")
        ok = False
        try:
            RO.save_ec_pert_npz(Path(td) / "bad.npz", ec, "nope", "global")
        except AssertionError:
            ok = True
        assert ok, "非法 readout 未被拦截"
    # ---- response 度量解析值（构造已知幅度合成张量，K=2/C=2/F=4）----
    amp = torch.tensor([0.5, 2.0]).view(-1, 1)               # base = ±amp → s_B = amp（解析）
    base = torch.zeros(2, 2, 4)
    base[0], base[1] = amp.expand(2, 4), -amp.expand(2, 4)
    assert torch.allclose(RS.baseline_scale(base), torch.tensor([0.5, 2.0]), rtol=1e-6)
    nulls = RS.null_distances(base)                          # 唯一对 (0,1)：|2·amp|/amp = 2
    assert nulls.shape == (1, 2) and torch.allclose(nulls, torch.full((1, 2), 2.0), rtol=1e-6)
    samples = base.unsqueeze(0).repeat(PC.N_VARIANTS, 1, 1, 1)
    C_syn = base.shape[1]
    for v in range(1, PC.N_VARIANTS):                        # 每变体源通道 a%C 加常量 0.5
        a, _j = PB.variant_of(v)                             # （合成张量仅 2 通道，a 映射 a%C）
        ac = a % C_syn
        samples[v, :, ac, :] = samples[v, :, ac, :] + 0.5
    Dc = RS.d_curves(samples)                                # D[a, j, a%C] = 0.5/amp[a%C]，其余通道 = 0
    assert Dc.shape == (62, 8, C_syn)
    for a in (0, 1, 2, 30, 61):
        ac, other = a % C_syn, (a + 1) % C_syn
        expect = torch.full((8,), 0.5 / float(amp[ac, 0]))
        assert torch.allclose(Dc[a, :, ac], expect, rtol=1e-6), f"D[{a},:,{ac}] 解析值失配"
        assert (Dc[a, :, other] == 0).all(), f"D[{a},:,{other}] 非源通道响应非零"
    print(f"[③ readout] D̄=2r → slope=2（主/敏感段）、AUC={PC.R_MAX}（解析）；τ=3.0 → r*={PC.R_NEXT}"
          f" → strength=0；翻转偏移 {PC.R_NEXT} 由 R_GRID 推导；有效路径对角/删失排除；"
          f"契约 npz 四字段往返；D/null 解析值（1.0/0.25/2.0）全中")


def test_gpu_smoke() -> None:
    """ckpt 装载 GPU 冒烟（implement.md B0 验证）：1 batch 采样 shape/finite + 497 变体
    并批（主 Run 批形）+ test 折 ID / 逐被试窗子采样确定性。"""
    if not torch.cuda.is_available():
        print("[smoke SKIP] 无 CUDA——ckpt 装载冒烟跳过")
        return
    cfg, model, ema, prior, sampler, ck = _frozen()
    assert sampler.gen_lowpass_hz is None                    # ② 复验（冒烟路径上再锁一次）
    loader_obj, test_ds = _test_data()
    ids = sorted(int(e["subject_id"]) for e in test_ds.subjects)
    assert ids == list(PC.EXPECTED_TEST_IDS), f"test 折 ID 漂移: {ids}"
    assert len(test_ds) > 0
    # 逐被试窗子采样：确定性 / 升序 / 越界检查
    sub1 = LD.subsample_windows_per_subject(test_ds)
    sub2 = LD.subsample_windows_per_subject(test_ds)
    assert sorted(sub1) == ids
    for sid, idx in sub1.items():
        assert len(idx) == PC.N_W == 32, f"被试 {sid} 窗数 {len(idx)} != N_w=32"
        assert np.array_equal(idx, sub2[sid]), f"被试 {sid} 子采样非确定性"
        assert (np.diff(idx) > 0).all() and 0 <= idx[0] and idx[-1] < len(test_ds)
    # 1 batch CRN 采样（真实 z-score 窗，显式 eps/eta）
    xp, _xf = next(iter(loader_obj))
    xp = xp.to("cuda")
    assert tuple(xp.shape[1:]) == (cfg.n_channels, cfg.past_len)
    eps, eta = RS.crn_noise(0, PC.K, cfg.n_channels, cfg.future_len, "cuda")
    with ema.swap(model):
        out = sampler.sample(xp, PC.K, eps=eps, eta=eta)
    assert out.shape == (xp.shape[0], PC.K, cfg.n_channels, cfg.future_len)
    assert torch.isfinite(out).all(), "采样输出含非有限值"
    # 主 Run 批形：单窗 497 变体一次调用 → D 曲线可算且有限非负
    with ema.swap(model):
        sv = RS.sample_window_variants(sampler, xp[:1], win_idx=0)
    assert sv.shape == (PC.N_VARIANTS, PC.K, cfg.n_channels, cfg.future_len)
    assert torch.isfinite(sv).all()
    D = RS.d_curves(sv)
    assert D.shape == (PC.N_SRC, PC.N_R_PERT, cfg.n_channels)
    assert torch.isfinite(D).all() and (D >= 0).all()
    # 对角自响应 vs 跨通道（量级观察，B1 检查点；本阶段只打印不断言）
    r_top = PC.PERT_R_GRID[-1]
    print(f"[smoke] fold0 test {len(ids)} 例 ID 全中；1 batch ({xp.shape[0]} 窗) K={PC.K} 采样 "
          f"{tuple(out.shape)} 有限；497 变体并批 {tuple(sv.shape)} 有限 → D {tuple(D.shape)} 有限非负；"
          f"r={r_top} 自响应 D[5,{PC.N_R_PERT - 1},5]={D[5, -1, 5]:.4f} vs "
          f"跨通道 D[5,{PC.N_R_PERT - 1},0]={D[5, -1, 0]:.4f}（量级观察，B1 判定）")


def main() -> None:
    test_crn_contract()   # ①
    test_calibers()       # ②
    test_readout_math()   # ③
    test_gpu_smoke()      # ckpt 装载 GPU 冒烟
    print("[PASS] perturb-ec-v1 B0 单测全部通过（跳过段已注明）")


if __name__ == "__main__":
    main()
