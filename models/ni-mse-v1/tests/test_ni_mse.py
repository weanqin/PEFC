#!/usr/bin/env python3
"""test_ni_mse.py — ni-mse-v1 七项单测（design.md §2 清单；implement.md S2.3）。

① P(X,a,0) ≡ X 逐位恒等（纯张量；只动源通道；禁原位）
② 窗子集确定性（同种子两次抽取逐位一致；W_pert ⊂ W_null；原序保持；spawn_key 合法形式）
③ τ_b 分位数单点（合成 N 池化数组 vs 手算 Q90/Q95/split-half——池化口径，非中位链）
④ refs 映射（10 test sid 全命中 refs_s{sid}.npz；ec (62,62)；行=源契约键）
⑤ R2 GPU 前向强形式（1 窗最小规模、真 θ*：P(X,a,0) 扰动路径输出 vs 干净基线路径输出，
   并入同一 sample() 调用同批（行 0/1 同 chunk）同 eps/eta，逐位恒等）
⑥ 装载遮蔽契约（七项 `__file__`：config/model/graph→v5-fmt；sample/data/gpr_prior/mamba→v1）
⑦ `sampler.gen_lowpass_hz is None`（已登记陷阱锚点：默认 40.0 静默削波）
⑧ perturb per-例续跑跳过（Run 002 前补丁：伪造已存在产物 → 二次调用整例跳过、产物逐字节
   不变、零数据集/零模型访问；缺臂/NaN/r2_checked 不完整/k_resp 漂移/窗子集漂移 ⇒ 重算）

运行：cd /root/npi/models/ni-mse-v1 && python3 -m pytest tests/ -q
（⑤⑦ 需 CUDA——mamba_ssm kernel 必须 CUDA，无 CUDA 自动跳过；GPU 段用真 θ* 但小规模）
"""

from __future__ import annotations

import _blas_guard  # noqa: F401  必须最先：BLAS 五变量先于 numpy 装载

import hashlib
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

_TESTS = Path(__file__).resolve().parent
_PKG = _TESTS.parent                       # models/ni-mse-v1
_MODELS = _PKG.parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

import noise as NZ      # noqa: E402
import stages as ST     # noqa: E402
import thostar as TH    # noqa: E402
import windows as WD    # noqa: E402

_V5F = _MODELS / "gpfm-mamba-v5-fmt"
_V1 = _MODELS / "gpfm-mamba-v1"


@lru_cache(maxsize=1)
def _gpu_thostar():
    """真 θ* 装载（CUDA；进程内一次，⑤⑦ 共享）。"""
    return TH.load_thostar("cuda")


@lru_cache(maxsize=1)
def _first_test_window():
    """fold0 test 第一例（s10）第 0 窗真实 X_past（经 v1 确定性 loader/z-score 管道）。"""
    _, test_ds = TH.load_test_data()
    xp, _ = test_ds[0]
    return xp


# ---- ① 扰动算子恒等 ----
def test_perturb_operator_identity() -> None:
    g = torch.Generator().manual_seed(2026)
    X = torch.randn(3, 62, 384, generator=g)
    X0 = X.clone()
    xi = torch.randn(3, 384, generator=g)
    Y0 = NZ.apply_noise(X, 5, 0.0, xi)
    assert torch.equal(Y0, X), "P(X,a,0) ≡ X 逐位恒等失败（c=0 无扰动对照）"
    Yc = NZ.apply_noise(X, 5, 0.5, xi)
    keep = [i for i in range(62) if i != 5]
    assert torch.equal(Yc[:, keep, :], X[:, keep, :]), "扰动越界：非源通道被改动"
    assert torch.allclose(Yc[:, 5, :], X[:, 5, :] + 0.5 * xi, rtol=0, atol=0), \
        "源通道加性注入 X[a]+c·ξ 失败"
    assert torch.equal(X, X0), "输入被原位修改（禁原位契约）"
    # (C,P) 形状同契约
    x1 = X[0]
    Y1 = NZ.apply_noise(x1, 5, 0.0, xi[0])
    assert torch.equal(Y1, x1)


# ---- ② 窗子集确定性 ----
def test_window_subset_determinism() -> None:
    n_total = 930                                   # fmt 60001 样本 @384/128/stride64
    sid = 10
    # spawn_key 合法形式：整数元组（字符串 spawn_key 本机非法——§1b-2）
    key = NZ.spawn_key_for(NZ.BRANCH["win_null"], sid)
    assert key == (20, sid) and all(isinstance(x, int) for x in key)
    ss = np.random.SeedSequence(entropy=NZ.MASTER_ENTROPY, spawn_key=key)  # 构造合法即测
    assert ss.spawn_key == (20, sid)
    w1 = WD.draw_subset(n_total, 200, NZ.BRANCH["win_null"], sid)
    w2 = WD.draw_subset(n_total, 200, NZ.BRANCH["win_null"], sid)
    assert np.array_equal(w1, w2), "同种子两次抽取不一致（确定性失败）"
    assert (np.diff(w1) > 0).all(), "窗子集未按原序（升序）排列"
    assert len(set(w1.tolist())) == len(w1), "无放回抽取出现重复"
    # W_pert ⊂ W_null（经 build_window_plan 同一断言口径）
    pos = WD.draw_subset(len(w1), 50, NZ.BRANCH["win_pert"], sid)
    wpert = np.sort(w1[pos])
    assert set(wpert.tolist()) <= set(w1.tolist()), "W_pert ⊄ W_null"
    assert (np.diff(wpert) > 0).all()
    # 窗数不足取全部（冒烟口径）
    small = WD.draw_subset(7, 200, NZ.BRANCH["win_null"], sid)
    assert np.array_equal(small, np.arange(7))


# ---- ③ τ_b 池化分位数单点 ----
def test_tau_quantile_pooling() -> None:
    rng = np.random.default_rng(7)
    N = (rng.standard_normal((64, 3, 200)) ** 2) * np.array([1.0, 4.0, 9.0])[None, :, None]
    got = ST.tau_stats(N)
    # 池化口径：N (K,C,W) → 逐通道 (W*K) 展平（w-major）后 Q90/Q95——非中位链
    flat = N.transpose(1, 2, 0).reshape(3, -1)
    assert np.allclose(got["tau_b"], np.quantile(flat, 0.90, axis=1), rtol=0, atol=0)
    assert np.allclose(got["tau_q95"], np.quantile(flat, 0.95, axis=1), rtol=0, atol=0)
    h1 = np.quantile(flat[:, ::2], 0.90, axis=1)
    h2 = np.quantile(flat[:, 1::2], 0.90, axis=1)
    rel = np.abs(h1 - h2) / np.quantile(flat, 0.90, axis=1)
    assert np.allclose(got["sh_rel"], rel, rtol=0, atol=0)
    assert got["sh_median"] == float(np.median(rel)) and got["sh_max"] == float(np.max(rel))
    # R4 排除窗（NaN 列）不进池
    N2 = N.copy()
    N2[:, :, [3, 100]] = np.nan
    got2 = ST.tau_stats(N2)
    flat2 = N2.transpose(1, 2, 0)[:, [w for w in range(200) if w not in (3, 100)], :].reshape(3, -1)
    assert np.allclose(got2["tau_b"], np.quantile(flat2, 0.90, axis=1), rtol=0, atol=0)


# ---- ④ refs 被试映射 ----
def test_refs_mapping() -> None:
    import re
    from pathlib import Path
    refs_dir = Path("/root/npi/refs/granger_bivariate_spectral/"
                    "edrafmt48-cont-zscore-fs250-fixed8-f1-125-nf65")
    all_ids = sorted(int(m.group(1)) for f in refs_dir.glob("refs_s*.npz")
                     if (m := re.fullmatch(r"refs_s(\d+)\.npz", f.name)))
    assert len(all_ids) == 48, f"refs 例数 {len(all_ids)} != 48"
    for sid in TH.EXPECTED_TEST_IDS:
        p = refs_dir / f"refs_s{sid}.npz"
        assert p.exists(), f"test sid {sid} 未命中 {p}"
        z = np.load(p)
        assert "ec" in z, f"refs_s{sid}.npz 缺 ec 键（实际键：{list(z.keys())}）"
        ec = z["ec"]
        assert ec.shape == (62, 62), f"ec 形状 {ec.shape} != (62,62)"
        assert np.isfinite(ec).all() and (np.diag(ec) == 0).all()
        assert float(ec[~np.eye(62, dtype=bool)].min()) >= 0.0, "offdiag 负值 = 实现错误"


# ---- ⑤ R2 GPU 前向强形式（真 θ*，1 窗 K=2）----
@pytest.mark.skipif(not torch.cuda.is_available(), reason="mamba_ssm kernel 需要 CUDA")
def test_r2_gpu_forward_strong() -> None:
    th = _gpu_thostar()
    xp = _first_test_window().to(th.device)            # (C,P) 真实窗
    C, P = xp.shape
    F = th.cfg.future_len
    n0 = ST.n0_from_cfg(th.cfg)
    assert n0 == 25, f"fmt@250 n₀ 应为 25，got {n0}"
    sid, w, a, K = 10, 0, 3, 2
    xi = np.stack([NZ.draw_xi(sid, w, a, k, P) for k in range(K)])               # (K,P)
    eps = np.stack([NZ.draw_eps_pert(sid, w, a, k, (C, F)) for k in range(K)])   # (K,C,F)
    eta = np.stack([NZ.draw_eta(sid, w, a, k, (C, F)) for k in range(K)])
    xi_t = torch.from_numpy(xi.astype(np.float32)).to(th.device)
    eps_t = torch.from_numpy(eps.astype(np.float32)).to(th.device)
    eta_t = torch.from_numpy(eta.astype(np.float32)).to(th.device)
    # 同一 sample() 调用同批：行 2k = P(X,a,0,ξ_k) 扰动路径；行 2k+1 = 干净基线 X；
    # 每 k 对共享同 eps/eta（行 0/1 同 chunk_bk=1024 分块内）
    pert0 = NZ.apply_noise(xp, a, 0.0, xi_t[0]).unsqueeze(0)
    pert1 = NZ.apply_noise(xp, a, 0.0, xi_t[1]).unsqueeze(0)
    batch = torch.cat([pert0, xp.unsqueeze(0), pert1, xp.unsqueeze(0)], dim=0)   # (4,C,P)
    eps_b = torch.stack([eps_t[0], eps_t[0], eps_t[1], eps_t[1]]).unsqueeze(1)   # (4,1,C,F)
    eta_b = torch.stack([eta_t[0], eta_t[0], eta_t[1], eta_t[1]]).unsqueeze(1)
    with th.ema.swap(th.model):
        out = th.sampler.sample(batch, 1, eps=eps_b, eta=eta_b)[:, 0]            # (4,C,F)
    assert out.shape == (4, C, F)
    assert torch.isfinite(out).all(), "R1：采样输出含 NaN/Inf"
    assert torch.equal(out[0], out[1]), "R2：P(X,a,0) 扰动路径 vs 干净基线非逐位恒等"
    assert torch.equal(out[2], out[3]), "R2：P(X,a,0) 扰动路径 vs 干净基线非逐位恒等（k=1）"


# ---- ⑥ 装载遮蔽契约（七项 __file__）----
def test_loader_shadow_contract() -> None:
    TH.shadow_modules()
    expect = {"config": _V5F, "model": _V5F, "graph": _V5F,
              "sample": _V1, "data": _V1, "gpr_prior": _V1, "mamba": _V1}
    for name, parent in expect.items():
        mod = sys.modules[name]
        got = Path(mod.__file__).resolve().parent
        assert got == parent, f"{name} 解析到 {mod.__file__}（应为 {parent / (name + '.py')}）"
    cfg = TH.load_cfg()
    assert cfg.config_hash() == TH.EXPECTED_CONFIG_HASH == "a64c0d416756"


# ---- ⑦ sampler.gen_lowpass_hz is None ----
@pytest.mark.skipif(not torch.cuda.is_available(), reason="装载需 CUDA（mamba_ssm）")
def test_sampler_gen_lowpass_none() -> None:
    th = _gpu_thostar()
    assert th.sampler.gen_lowpass_hz is None, \
        "fmt 口径必须 gen_lowpass_hz=None（构造默认 40.0 brickwall fs=500 = 静默削波陷阱）"
    assert th.sampler.steps == th.cfg.euler_steps and th.sampler.sigma_min == th.cfg.sigma_min


# ---- ⑧ perturb per-例续跑跳过（Run 002 前补丁，2026-10-20）----
def _forge_perturb_products(out_root: Path, sid: int, c_arms: tuple, C=3, W=2, K=2,
                            r2=None, k_resp=None, finite=True, wpert=None,
                            drop_arm=None) -> None:
    """伪造一例已完成 perturb（+null 前置）产物（字段契约对齐 perturb_stage 落盘）。"""
    F, P = 8, 16
    wpert = np.arange(W) if wpert is None else np.asarray(wpert, dtype=np.int64)
    (out_root / "null").mkdir(parents=True, exist_ok=True)
    np.savez(out_root / "null" / f"{sid}.npz",
             y_ref_wpert=np.random.rand(W, C, F).astype(np.float32),
             wpert_global=wpert)
    (out_root / "perturb").mkdir(parents=True, exist_ok=True)
    for c in c_arms:
        if c == drop_arm:
            continue
        rbar = np.random.rand(C, C).astype(np.float32)
        if not finite:
            rbar[0, 1] = np.nan
        np.savez(out_root / "perturb" / f"{sid}_c{c:g}.npz",
                 R_bar=rbar, k_resp=(K if k_resp is None else k_resp),
                 r2_checked=(len(c_arms) if r2 is None else r2),
                 r4_count=0, wpert_global=wpert)


class _NoDatasetAccess:
    """任何索引访问 = 续跑失败（已存在例不得触数据集）。"""

    def __getitem__(self, i):        # pragma: no cover - 断言即失败
        raise AssertionError("续跑失败：已存在例不应访问 test_ds")


def test_perturb_resume_skip_stage_level(tmp_path) -> None:
    """二次调用整例跳过：零数据集/零模型访问，产物逐字节不变，summary 标记 skipped。"""
    from types import SimpleNamespace

    sid, c_arms, C, W, K = 10, (0.05, 0.5), 3, 2, 2
    p = ST.Params(c_arms=c_arms, k_resp=K)
    _forge_perturb_products(tmp_path, sid, c_arms, C=C)
    plan = {sid: {"wnull": np.arange(930), "wpert": np.arange(W)}}
    th = SimpleNamespace(                              # 无 ema/sampler ⇒ 跳过路径零 GPU 依赖
        device="cpu", cfg=SimpleNamespace(n_channels=C, future_len=8, past_len=16, fs=250.0))
    before = {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
              for f in sorted((tmp_path / "perturb").glob("*.npz"))}
    summaries = ST.perturb_stage(th, _NoDatasetAccess(), plan, p, tmp_path, [sid])
    s = summaries[str(sid)] if str(sid) in summaries else summaries[sid]
    assert s.get("skipped") is True and s["r2_checked"] == len(c_arms)
    after = {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
             for f in sorted((tmp_path / "perturb").glob("*.npz"))}
    assert before == after, "续跑跳过不得改写既有产物（结果不变契约）"


def test_perturb_resume_guard_conditions(tmp_path) -> None:
    """可跳过判定的守卫面：完整 ⇒ 跳过摘要；缺臂/NaN/r2 不完整/k_resp 漂移/窗子集漂移 ⇒ None。"""
    sid, c_arms, C, W, K = 10, (0.05, 0.5), 3, 2, 2
    wpert = np.arange(W, dtype=np.int64)

    def cached(**kw):
        root = tmp_path / f"case{len(list(tmp_path.iterdir()))}"
        kw.setdefault("wpert", wpert)
        _forge_perturb_products(root, sid, c_arms, C=C, W=W, K=K, **kw)
        return ST.perturb_subject_cached(root / "perturb", sid, ST.Params(c_arms=c_arms), K, wpert)

    ok = cached()
    assert ok is not None and ok["skipped"] is True and ok["r2_checked"] == len(c_arms)
    assert cached(drop_arm=0.5) is None, "缺臂 ⇒ 重算"
    assert cached(finite=False) is None, "R_bar 含 NaN ⇒ 重算"
    assert cached(r2=1) is None, "r2_checked != |c 臂|（R2 硬门不完整）⇒ 重算"
    assert cached(k_resp=7) is None, "k_resp 漂移 ⇒ 重算"
    assert cached(wpert=np.array([5, 9])) is None, "窗子集漂移 ⇒ 重算"
