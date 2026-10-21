#!/usr/bin/env python3
"""test_v4b_gnn.py — v4bg（v4b d_ch=1 基座 + LearnableMexHatMixer 图头）单测十二项
（任务 10-19-gpfm-mamba-v4b-gnn-f0，design §7，implement.md S1.2）。

① GraphHead 前向 shape (B,T,62,1)→(B,T,62) 与 dtype
② 因果性（v4b 惯例三段 + future 128 截取守卫）：future 扰动不改 past 输出；future 后段扰动
   不改前段；past 前缀条件生效
③ 梯度有限性：反传一步全参数 finite（含 W / log_sigma 梯度非零的测试有效性）+ GPU B=256
   fwd+bwd 冒烟
④ 参数量精确断言：head 12,358 / 全模型 12,864（design §2.4 预算 12,358 对照，±0）+ 逐模块明细
⑤ config↔model 一致（d_model/graph_* 双处）+ config_hash 稳定（两次 resolve 同 hash）+ 敏感性 +
   v1/v2/v3lp/v4/v4a config_hash 零漂移（子进程口径）
⑥ AC2 等价测：同 coords/σ/γ/W 下 A_eff 与 latent-flow 源 LearnableMexHatMixer.
   effective_adjacency allclose rtol=1e-6（ricker + dog 两核，σ 取非默认值；源经 sys.path
   导入只读）
⑦ 方向单测（A-1/B-3）：γ_m=0 + w_init 单边注入（仅 W[i,j]=1）——扰动输入通道 j ⇒ 仅通道 i
   响应（k≠i 逐位不变）；反向注入（W[j,i]=1）⇒ 扰动 j 全通道零响应；读出转置约定断言
   （A_eff.T[j,i]=γ 即行=源/列=靶口径）
⑧ 双遮蔽 + 前缀守卫：config/model 命中本目录、mamba→v1；v4bg- 前缀守卫 train/evaluate 正反例；
   graph_* 运行时断言正反例；W 范数仪表（mod.EMA 子类）记录双口径
⑨ 正则旋钮：λ=0 时 graph_regularization 返回精确 0（损失逐位一致）；λ>0 时含惩罚项且与
   λ_u·uniformity+λ_s·sparsity 手工复算一致；两惩罚方法与源数值一致
⑩ 通道名断言门禁：真数据根 46/46 顺序全等（扫描数恰 46）；正反例（mismatch 报错 / 例数
   不符报错）
⑪ split_manifest_override 生效（指向 v4 的 fold0 val4 manifest，与 v4a 逐字节相同）+ 32/4/10
   划分 + v1/splits 与 v4 manifest 零改动（hash+mtime 双检）+ default-off 钩子正反例
⑫ 钩子 kwargs 契约（§2.5）：默认值（None/"rmsnorm"/"off"）可构造；非默认逐项报错

运行：python3 -m pytest models/gpfm-mamba-v4b-gnn/tests/test_v4b_gnn.py -q（需 CUDA——mamba_ssm）
      或 python3 models/gpfm-mamba-v4b-gnn/tests/test_v4b_gnn.py
"""

from __future__ import annotations

import hashlib
import json
import os

# BLAS 线程上限（environment.md 惯例：有效 16 核 cgroup，须在 numpy 导入前设置；可覆盖）
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "8")

import dataclasses  # noqa: E402
import importlib.util  # noqa: E402
import inspect  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

_HERE = Path(__file__).resolve().parent          # models/gpfm-mamba-v4b-gnn/tests
_VBG = _HERE.parent                               # models/gpfm-mamba-v4b-gnn
_V1 = _VBG.parent / "gpfm-mamba-v1"              # 只读依赖的 v1 目录
_V4 = _VBG.parent / "gpfm-mamba-v4"
_V4A = _VBG.parent / "gpfm-mamba-v4a"
_LF1 = _VBG.parent / "latent-flow-v1"            # 融合源（只读；等价测 sys.path 导入）

sys.path.insert(0, str(_V1))
sys.path.insert(0, str(_VBG))

import config as C  # noqa: E402
import data as D  # noqa: E402  # v1 data.py（只读复用；split_manifest_override 钩子所在）
import graph_head as GH  # noqa: E402
import model as M  # noqa: E402

if not torch.cuda.is_available():
    raise RuntimeError("mamba_ssm 官方 kernel 需要 CUDA")
torch.manual_seed(1234)  # 测试进程定 seed（默认 torch 种子逐进程随机——缓存模型 init 须可复现；
                         # 见 ② 注记：d1 的 RMSNorm(1) 边界为符号量化，敏感性断言须用强扰动）

# ---- 登记值（实测精确数 / hash pin；后续任何口径变更须重算登记）----
EXPECTED_HEAD_PARAMS = 12_358     # in_proj 128 + msg 2×4,160 + W 3,844 + σ 1 + out_proj 65
EXPECTED_TOTAL_PARAMS = 12_864    # = 基座 506（seg 2 + in_proj 19 + trunk 4×121 + RMSNorm 1）+ head
                                 #（design §2.4 预算 head 12,358 / 总量 ≈12.9k——精确吻合）
V1_HASH_DEFAULT = "c6ff80ac1c43"  # 基准：v1 main-01 落盘 resolved config hash（零漂移守护）
V2_HASH_DEFAULT = "fc2a9ce14948"  # 基准：v2d62-main-01 落盘 resolved config hash
V3_HASH_DEFAULT = "0c2de30a4868"  # 基准：v3lp 落盘 resolved config hash
V4_HASH_DEFAULT = "91dfd211dbc3"  # v4（v4b d_ch=8）默认 config hash（d_ch 1→8 方案 B 后登记值）
V4A_HASH_DEFAULT = "2a456201dc90" # v4a（r=16）默认 config hash
VBG_HASH_DEFAULT = "5920e339c28c" # v4bg 默认 config hash（2026-10-19 首建登记）
VBG_HASH_EPOCHS1 = "30c4a8adae57" # epochs=1 覆写口径（bench 用）

_MODEL_CACHE: dict = {}


def build_model(dev: str = "cuda") -> M.VelocityNetwork:
    """按 v1 train.py:273-281 的显式 kwargs 构建（运行时契约四件套口径，含三钩子 kwargs）。"""
    if "m" not in _MODEL_CACHE:
        cfg = C.Config()
        _MODEL_CACHE["m"] = M.VelocityNetwork(
            n_channels=cfg.n_channels, past_len=cfg.past_len, future_len=cfg.future_len,
            d_model=cfg.d_model, n_layers=cfg.n_layers, d_state=cfg.d_state,
            d_conv=cfg.d_conv, expand=cfg.expand, time_emb_dim=cfg.time_emb_dim,
            use_fast_path=cfg.use_fast_path,
            bottleneck_rank=getattr(cfg, "bottleneck_rank", None),
            norm_type=str(getattr(cfg, "norm_type", "rmsnorm")),
            cond_adaln=str(getattr(cfg, "cond_adaln", "off")),
        ).to(dev)
    return _MODEL_CACHE["m"]


def _inputs(cfg: C.Config, dev: str, B: int = 4, seed: int = 7):
    g = torch.Generator(device=dev).manual_seed(seed)
    xp = torch.randn(B, cfg.n_channels, cfg.past_len, device=dev, generator=g)
    xf = torch.randn(B, cfg.n_channels, cfg.future_len, device=dev, generator=g)
    t = torch.rand(B, device=dev, generator=g).clamp(0.05, 0.95)
    return xp, xf, t


def _load_source_mixer(coords: torch.Tensor, **kw):
    """latent-flow-v1 源 LearnableMexHatMixer（只读导入；独立 sys.path，不改源一字节）。"""
    sys.path.insert(0, str(_LF1))
    from latent_flow.models.oc_latent_flow import LearnableMexHatMixer  # noqa: PLC0415
    return LearnableMexHatMixer(n_channels=62, latent_dim=8, hidden_ch=16,
                                coords=coords, **kw)


def _geometry_coords() -> torch.Tensor:
    g = torch.load(C.Config().geometry_path, map_location="cpu", weights_only=False)
    return g["coords"].float()


# ① ------------------------------------------------------------------------
def test_graph_head_forward_shape() -> None:
    cfg = C.Config()
    head = GH.GraphHead().to("cuda")
    for (B, T) in ((3, 128), (2, 512), (1, 1)):
        x = torch.randn(B, T, cfg.n_channels, cfg.d_model, device="cuda")
        v = head(x)
        assert v.shape == (B, T, cfg.n_channels), f"{(B, T)}: {tuple(v.shape)}"
        assert v.dtype == torch.float32
        assert torch.isfinite(v).all()
    # 位置逐点：head 对时间轴无交互（打乱时间序不改变逐位置输出）
    x = torch.randn(2, 7, cfg.n_channels, cfg.d_model, device="cuda")
    perm = torch.randperm(7)
    assert torch.allclose(head(x)[:, perm], head(x[:, perm]), atol=1e-6), "head 时间轴存在交互"
    print("[① shape] (B,T,62,1)→(B,T,62) float32 全有限；时间轴逐点（打乱不变）")


# ② ------------------------------------------------------------------------
def test_causality_future_slice() -> None:
    """因果性（v4b 惯例三段 + future 128 截取守卫）。

    d1 结构注记（实测）：head 前 RMSNorm(1) 把池化特征量化为符号（|输出|≈1，x/√(x²+eps)），
    弱扰动若不翻转任何 trunk 输出符号则 head 输入逐位不变——「输出应改变」的两处**测试有效性**
    断言须用强随机扰动（10σ；31k 边界元素零翻转概率≈0）；因果性断言（past 不受扰动）对任意
    扰动成立，保持 +1.0 平移口径与 v4b 测试同款。
    """
    cfg = C.Config()
    model = build_model()
    model.eval()
    xp, xf, t = _inputs(cfg, "cuda")
    gstr = torch.Generator(device="cuda").manual_seed(11)  # 强扰动 RNG（固定 seed 可复现）
    with torch.no_grad():
        ref = model(xp, xf, t, return_all=True)          # (B, 62, 512)
        # future 截取守卫：默认前向输出 == return_all 的 future 切片（head 只读 future 128 位置）
        v = model(xp, xf, t)
        assert v.shape == (4, cfg.n_channels, cfg.future_len)
        assert torch.allclose(v, ref[:, :, cfg.past_len:], atol=1e-6), "future 截取守卫破坏"
        # future 扰动不改 past 位置输出（因果性；对任意扰动成立）
        xf2 = xf.clone(); xf2[:, :, :] += 1.0
        out2 = model(xp, xf2, t, return_all=True)
        assert torch.allclose(out2[:, :, : cfg.past_len], ref[:, :, : cfg.past_len], atol=1e-5), \
            "扰动 future 输入改变了 past 位置输出（因果性破坏）"
        # 测试有效性：future 输入对 future 输出敏感（强随机扰动穿透符号量化边界）
        xf2s = xf + 10.0 * torch.randn(xf.shape, device="cuda", generator=gstr)
        out2s = model(xp, xf2s, t, return_all=True)
        assert not torch.allclose(out2s[:, :, cfg.past_len:], ref[:, :, cfg.past_len:], atol=1e-5), \
            "强扰动 future 输入未改变 future 输出（前缀条件/敏感性失效？）"
        # future 后段扰动不改前段
        xf3 = xf.clone(); xf3[:, :, 100:] += 1.0
        out3 = model(xp, xf3, t, return_all=True)
        assert torch.allclose(out3[:, :, cfg.past_len: cfg.past_len + 50],
                              ref[:, :, cfg.past_len: cfg.past_len + 50], atol=1e-5), \
            "future 后段扰动影响了前段输出（序列内因果性破坏）"
        # past 前缀条件生效（强扰动口径）
        xp4 = xp.clone()
        xp4[:, :, -50:] += 10.0 * torch.randn(xp4[:, :, -50:].shape, device="cuda", generator=gstr)
        out4 = model(xp4, xf, t, return_all=True)
        assert not torch.allclose(out4[:, :, cfg.past_len:], ref[:, :, cfg.past_len:], atol=1e-5), \
            "past 强扰动未影响 future 输出（前缀条件未生效）"
    print("[② 因果性] future 截取守卫 + 三段因果（past 不受 future 扰动/后段不前传/前缀生效"
          "（强扰动口径））全过；d1 边界符号量化注记见 docstring")


# ③ ------------------------------------------------------------------------
def test_gradients_finite() -> None:
    cfg = C.Config()
    model = build_model()
    model.train()
    model.zero_grad(set_to_none=True)
    xp, xf, t = _inputs(cfg, "cuda")
    v = model(xp, xf, t)
    (v ** 2).mean().backward()
    n_grad = 0
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), "存在非有限梯度"
            n_grad += 1
    gh = dict(model.named_parameters())
    assert gh["graph_head.W"].grad.abs().sum() > 0, "W 梯度为零（零初始化下不可达——测试无效）"
    assert gh["graph_head.log_sigma"].grad.abs().sum() > 0, "σ 梯度为零（核先验不可学——测试无效）"
    # GPU 训练形状冒烟（B=256 = 训练 batch）
    model.zero_grad(set_to_none=True)
    g = torch.Generator(device="cuda").manual_seed(1)
    xpB = torch.randn(cfg.batch_size, cfg.n_channels, cfg.past_len, device="cuda", generator=g)
    xfB = torch.randn(cfg.batch_size, cfg.n_channels, cfg.future_len, device="cuda", generator=g)
    tB = torch.rand(cfg.batch_size, device="cuda", generator=g)
    torch.cuda.synchronize(); t0 = time.time()
    vB = model(xpB, xfB, tB)
    (vB ** 2).mean().backward()
    torch.cuda.synchronize(); dt = time.time() - t0
    assert vB.shape == (cfg.batch_size, cfg.n_channels, cfg.future_len)
    assert torch.isfinite(vB).all()
    print(f"[③ 梯度] {n_grad} 参数张量梯度全有限（W/σ 梯度非零）；"
          f"GPU 冒烟 B={cfg.batch_size} fwd+bwd {dt * 1000:.0f} ms，"
          f"peak {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB")


# ④ ------------------------------------------------------------------------
def test_param_count_exact() -> None:
    model = build_model()
    n = model.num_params()
    head_n = model.graph_head.num_params()
    assert head_n == EXPECTED_HEAD_PARAMS, f"GraphHead 参数量 {head_n} != {EXPECTED_HEAD_PARAMS}"
    assert n == EXPECTED_TOTAL_PARAMS, f"全模型参数量 {n} != {EXPECTED_TOTAL_PARAMS}"
    d = {k: p.numel() for k, p in model.named_parameters()}
    seg = d["seg_past"] + d["seg_future"]
    in_proj = d["in_proj.weight"] + d["in_proj.bias"]
    blocks = sum(v for k, v in d.items() if k.startswith("blocks."))
    head_norm = d["head_norm.weight"]
    gh = {k[len("graph_head."):]: v for k, v in d.items() if k.startswith("graph_head.")}
    msg = sum(gh[k] for k in gh if k.startswith("msg_layers."))
    assert seg == 2 and in_proj == 19 and blocks == 4 * 121 and head_norm == 1, \
        f"基座明细漂移：seg {seg} in_proj {in_proj} blocks {blocks} head_norm {head_norm}"
    assert gh["in_proj.weight"] + gh["in_proj.bias"] == 128, "head in_proj != 128"
    assert msg == 2 * 4_160, f"head msg != 2×4,160（got {msg}）"
    assert gh["W"] == 62 * 62, "head W != 3,844"
    assert gh["log_sigma"] == 1, "head σ != 1"
    assert gh["out_proj.weight"] + gh["out_proj.bias"] == 65, "head out_proj != 65"
    print(f"[④ 参数量] 全模型 {n} == {EXPECTED_TOTAL_PARAMS}；GraphHead {head_n} == "
          f"{EXPECTED_HEAD_PARAMS}（design §2.4 预算精确吻合；明细：seg 2 + in_proj 19 + "
          f"trunk 4×121 + RMSNorm 1 + [in_proj 128 + msg 2×4,160 + W 3,844 + σ 1 + out_proj 65]）")


# ⑤ ------------------------------------------------------------------------
def test_config_model_consistency() -> None:
    cfg = C.Config()
    fields = ("d_model", "graph_kernel", "graph_k", "graph_hidden", "graph_gamma",
              "graph_gamma_m", "graph_w_init", "geometry_path", "data_root")
    sig = inspect.signature(M.VelocityNetwork.__init__)
    for f in fields:
        assert sig.parameters[f].default == getattr(cfg, f), \
            f"model 类默认 {f}={sig.parameters[f].default!r} != cfg.{f}={getattr(cfg, f)!r}"
    assert cfg.d_model == 1 and cfg.graph_kernel == "ricker" and cfg.graph_k == 2 \
        and cfg.graph_hidden == 64 and cfg.graph_gamma == 0.1 and cfg.graph_gamma_m == 1.0 \
        and cfg.graph_lambda_uniform == 0.0 and cfg.graph_lambda_sparse == 0.0, \
        "cfg 默认超参漂移（design §3 口径）"
    assert cfg.variant_name == "gpfm-mamba-v4b-gnn"
    h1, h2 = C.Config().config_hash(), C.build_config().config_hash()
    assert h1 == h2, "config_hash 不稳定（同配置两次构建不一致）"
    assert h1 == VBG_HASH_DEFAULT, \
        f"v4bg config_hash {h1} != 登记值 {VBG_HASH_DEFAULT}（口径变更后须显式重算登记）"
    assert C.build_config(epochs=1).config_hash() == VBG_HASH_EPOCHS1, "epochs=1 bench 覆写口径漂移"
    h_mut = dataclasses.replace(cfg, graph_gamma_m=0.0).config_hash()
    assert h_mut != h1, "config_hash 对 graph_gamma_m 不敏感（防线失效）"
    h_k = dataclasses.replace(cfg, graph_kernel="dog").config_hash()
    assert h_k != h1, "config_hash 对 graph_kernel 不敏感（防线失效）"
    h_split = dataclasses.replace(cfg, n_val_subjects=2).config_hash()
    assert h_split != h1, "config_hash 对 n_val_subjects 不敏感（划分口径未进 hash）"
    # 既有变体 config_hash 零漂移（子进程隔离导入；基准取自已落盘 run / 各自测试登记值）
    for path, base, name in ((str(_V1), V1_HASH_DEFAULT, "v1"),
                             (str(_V1.parent / "gpfm-mamba-v2"), V2_HASH_DEFAULT, "v2"),
                             (str(_V1.parent / "gpfm-mamba-v3"), V3_HASH_DEFAULT, "v3lp"),
                             (str(_V4), V4_HASH_DEFAULT, "v4"),
                             (str(_V4A), V4A_HASH_DEFAULT, "v4a")):
        code = (f"import sys; sys.path.insert(0, {path!r}); import config as C; "
                f"print(C.Config().config_hash())")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        got = out.stdout.strip()
        assert got == base, f"{name} config_hash 漂移：{got} != 基准 {base}（既有变体须零改动）"
    print(f"[⑤ config↔model] graph_* 双处一致；hash {h1} 稳定且对 graph_gamma_m/graph_kernel/"
          f"划分敏感；epochs=1={VBG_HASH_EPOCHS1}；v1/v2/v3lp/v4/v4a 零漂移"
          f"（{V1_HASH_DEFAULT}/{V2_HASH_DEFAULT}/{V3_HASH_DEFAULT}/{V4_HASH_DEFAULT}/{V4A_HASH_DEFAULT}）")


# ⑥ ------------------------------------------------------------------------
def test_ac2_equivalence_ricker_dog() -> None:
    coords = _geometry_coords()
    torch.manual_seed(23)
    w = torch.randn(62, 62)

    # ---- ricker：σ 取非默认值（验证公式而非初值巧合）----
    src = _load_source_mixer(coords, k_layers=2, gamma=0.13, gamma_m=0.7,
                             kernel="ricker", w_init=w)
    ours = GH.GraphHead(gamma=0.13, gamma_m=0.7, kernel="ricker", w_init=w)
    with torch.no_grad():
        src.log_sigma.fill_(float(np.log(0.05)))       # 非默认 σ（median_d≈0.121）
        ours.log_sigma.fill_(float(np.log(0.05)))
    a_src, a_ours = src.effective_adjacency(), ours.effective_adjacency()
    assert a_src.shape == a_ours.shape == (62, 62)
    assert torch.allclose(a_ours, a_src, rtol=1e-6, atol=0.0), \
        f"ricker A_eff 等价性破坏：max|Δ|={float((a_ours - a_src).abs().max()):.3e}"
    # pairwise_d buffer 同构（距离阵逐位一致）
    assert torch.equal(ours.pairwise_d, src.pairwise_d), "pairwise_d 不一致"

    # ---- dog：4 参数全部取非默认值 ----
    dog_init = {"sigma_e": 0.031, "sigma_i": 0.097, "Ae": 1.3, "Ai": 0.4}
    src_d = _load_source_mixer(coords, k_layers=2, gamma=0.11, gamma_m=0.6,
                               kernel="dog", w_init=w, dog_init=dog_init)
    ours_d = GH.GraphHead(gamma=0.11, gamma_m=0.6, kernel="dog", w_init=w, dog_init=dog_init)
    with torch.no_grad():  # 扰动全部 4 参数到非默认值（重参数化恒等式下两侧同值）
        for mod in (src_d, ours_d):
            mod.log_sigma_e.fill_(float(np.log(0.027)))
            mod.log_delta_sigma.fill_(float(np.log(0.081)))
            mod.log_Ae.fill_(float(np.log(1.7)))
            mod.log_delta_a.fill_(float(np.log(np.expm1(1.1))))
    ad_src, ad_ours = src_d.effective_adjacency(), ours_d.effective_adjacency()
    assert torch.allclose(ad_ours, ad_src, rtol=1e-6, atol=0.0), \
        f"dog A_eff 等价性破坏：max|Δ|={float((ad_ours - ad_src).abs().max()):.3e}"
    assert ours_d.dog_params() == src_d.dog_params(), "dog_params 重建不一致"
    dr = float((a_ours - a_src).abs().max().detach())
    dd = float((ad_ours - ad_src).abs().max().detach())
    print(f"[⑥ AC2 等价] ricker（σ=0.05）与 dog（4 参数非默认）两核下 A_eff 与源 "
          f"allclose rtol=1e-6（max|Δ| ricker {dr:.2e} / dog {dd:.2e}）；pairwise_d 逐位一致")


# ⑦ ------------------------------------------------------------------------
def test_direction_semantics() -> None:
    i_tgt, j_src = 17, 5
    head = GH.GraphHead(gamma_m=0.0, gamma=0.1).to("cuda")  # 纯 W 消融核（γ_m=0）
    g = torch.Generator(device="cuda").manual_seed(3)
    x = torch.randn(2, 5, 62, 1, device="cuda", generator=g)

    # 单边注入 W[i,j]=1（源语义：j 驱动 i）⇒ 扰动输入通道 j 仅通道 i 响应
    with torch.no_grad():
        head.W.zero_(); head.W[i_tgt, j_src] = 1.0
    A = head.effective_adjacency().detach()
    assert A[i_tgt, j_src] == 0.1 and A.abs().sum() == 0.1, "A_eff 构造与注入不符"
    # 读出转置约定（A-1）：对外读出 = A_eff.T（行=源、列=靶）——[j,i] 位承载 j→i 边权
    assert A.T[j_src, i_tgt] == 0.1, "转置读出位不一致（行=源/列=靶约定破坏）"

    x2 = x.clone(); x2[:, :, j_src, :] += 1.0
    head.eval()
    with torch.no_grad():
        v1, v2 = head(x), head(x2)
    # 响应集恰为 {i_tgt, j_src}：j 经自身残差流（in_proj(x_j) 恒等通路，平凡响应）；
    # i 经注入图边 W[i,j]（**唯一跨通道响应**——方向语义的检验对象）；其余 60 通道逐位不变
    changed = [k for k in range(62) if not torch.equal(v2[:, :, k], v1[:, :, k])]
    assert changed == sorted({i_tgt, j_src}), \
        f"单边注入响应集 {changed} != {{{i_tgt}, {j_src}}}（方向语义破坏）"
    assert not torch.allclose(v2[:, :, i_tgt], v1[:, :, i_tgt], atol=1e-6), \
        f"注入边 W[{i_tgt},{j_src}] 下目标通道 {i_tgt} 未响应（测试无效）"

    # 反向注入 W[j,i]=1（i 驱动 j）⇒ 扰动输入通道 j 仅自身平凡响应——通道 i 零响应
    #（A[·,j] 全零：j 不驱动任何通道；读出方向 j→i 不存在）
    head_r = GH.GraphHead(gamma_m=0.0, gamma=0.1).to("cuda")
    with torch.no_grad():
        head_r.W.zero_(); head_r.W[j_src, i_tgt] = 1.0
    head_r.eval()
    with torch.no_grad():
        vr1, vr2 = head_r(x), head_r(x2)
    changed_r = [k for k in range(62) if not torch.equal(vr2[:, :, k], vr1[:, :, k])]
    assert changed_r == [j_src], \
        f"反向注入响应集 {changed_r} != [{j_src}]（j→i 方向不应存在——方向语义破坏）"
    assert torch.equal(vr2[:, :, i_tgt], vr1[:, :, i_tgt]), "反向注入下通道 i 出现响应（方向反转泄漏）"

    # K=3 多轮下方向隔离仍成立（上方 head 即 k_layers=2 默认）
    head3 = GH.GraphHead(gamma_m=0.0, gamma=0.1, k_layers=3).to("cuda")
    with torch.no_grad():
        head3.W.zero_(); head3.W[i_tgt, j_src] = 1.0
    head3.eval()
    with torch.no_grad():
        w1, w2 = head3(x), head3(x2)
    changed3 = [k for k in range(62) if not torch.equal(w2[:, :, k], w1[:, :, k])]
    assert changed3 == sorted({i_tgt, j_src}), f"K=3 方向隔离破坏：响应集 {changed3}"
    print(f"[⑦ 方向] W[{i_tgt},{j_src}]=1：扰动输入通道 {j_src} 响应集恰 {{{i_tgt},{j_src}}}"
          f"（i = 唯一跨通道响应，其余 60 逐位不变，K=2/K=3 双验）；反向注入响应集 [{j_src}]"
          f"（i 零响应）；A_eff.T[{j_src},{i_tgt}]=γ 转置读出约定（行=源/列=靶）一致")


# ⑧ ------------------------------------------------------------------------
def test_masking_and_prefix_guard() -> None:
    def load_wrapper(mod_name: str, filename: str):
        spec = importlib.util.spec_from_file_location(mod_name, str(_VBG / filename))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    tr = load_wrapper("v4bg_train_wrapper_under_test", "train.py")
    tr._shadow_v4bg()
    import config as _C
    import model as _M
    assert Path(_C.__file__).resolve().parent == _VBG, f"config 遮蔽未命中 v4bg：{_C.__file__}"
    assert Path(_M.__file__).resolve().parent == _VBG, f"model 遮蔽未命中 v4bg：{_M.__file__}"
    assert _M.VelocityNetwork is M.VelocityNetwork, "遮蔽后的 model 与测试直接导入的不一致"
    import mamba as _mb
    assert Path(_mb.__file__).resolve().parent == _V1, f"mamba 应解析到 v1：{_mb.__file__}"
    import graph_head as _gh
    assert Path(_gh.__file__).resolve().parent == _VBG, "graph_head 应随 model 命中 v4bg"

    # graph_* 运行时断言：正例（真实 cfg）通过；反例（shim graph_gamma_m=0.5）SystemExit
    tr._assert_model_config_consistency()

    class _BadCfg:
        d_model, graph_kernel, graph_k = 1, "ricker", 2
        graph_hidden, graph_gamma, graph_gamma_m = 64, 0.1, 0.5
        graph_w_init = ""
        geometry_path = C.Config().geometry_path
        data_root = C.Config().data_root

    class _BadConfMod:
        Config = staticmethod(lambda: _BadCfg())

    real_conf = sys.modules["config"]
    try:
        sys.modules["config"] = _BadConfMod()
        try:
            tr._assert_model_config_consistency()
            raise AssertionError("graph_gamma_m 漂移未被运行时断言拦截")
        except SystemExit as e:
            assert "graph_gamma_m" in str(e), f"退出信息未指明 graph_gamma_m：{e}"
    finally:
        sys.modules["config"] = real_conf

    # 前缀守卫（train wrapper）：正例 + 反例
    tr._guard_run_id(["prog", "--run-id", "v4bg-smoke-a"])
    tr._guard_run_id(["prog", "--resume",
                      "/x/v4bg-bench-1ep/checkpoints/gpfm-mamba-v4b-gnn-last.pt"])
    for bad in (["prog", "--run-id", "v4b-smoke-a"],
                ["prog", "--run-id", "v4a-r16-main-01"],
                ["prog", "--run-id", "main-01"],
                ["prog", "--resume", "/x/v2d62-main-01/checkpoints/last.pt"]):
        try:
            tr._guard_run_id(bad)
            raise AssertionError(f"前缀守卫未拦截：{bad}")
        except SystemExit:
            pass
    argv_auto = ["prog"]
    tr._guard_run_id(argv_auto)
    assert argv_auto[-2] == "--run-id" and argv_auto[-1].startswith("v4bg-"), argv_auto

    # evaluate wrapper 守卫：缺 --run-id / 错前缀退出，正例通过
    ev = load_wrapper("v4bg_evaluate_wrapper_under_test", "evaluate.py")
    ev._guard_run_id(["prog", "--run-id", "v4bg-main-01"])
    for bad in (["prog"], ["prog", "--run-id", "v4b-main-01"]):
        try:
            ev._guard_run_id(bad)
            raise AssertionError(f"evaluate 守卫未拦截：{bad}")
        except SystemExit:
            pass

    # W 范数仪表（B-5）：mod.EMA 被替换为子类，update 后记录 raw/EMA 双口径
    mod = type("FakeV1Train", (), {})()  # 占位命名空间对象
    tr._instrument_wnorm(mod)
    assert issubclass(mod.EMA, M.EMA), "仪表未替换 EMA"
    head = GH.GraphHead(gamma_m=0.0, gamma=0.1)
    with torch.no_grad():
        head.W.normal_(0.0, 0.01)
    ema = mod.EMA(head, decay=0.999)
    n0 = len(tr._W_TRAJ)
    ema.update(head)
    with torch.no_grad():
        head.W.add_(0.001)
    ema.update(head)
    assert len(tr._W_TRAJ) == n0 + 2, "W 范数轨迹未按 step 记录"
    rec = tr._W_TRAJ[-1]
    assert set(rec) == {"step", "w_norm_raw", "w_norm_ema"}
    assert rec["w_norm_raw"] != rec["w_norm_ema"], "raw/EMA 双口径未区分"
    assert abs(rec["w_norm_raw"] - float(head.W.detach().norm())) < 1e-6, "raw 口径与 W 范数不符"
    print("[⑧ 遮蔽/守卫] config+model 双遮蔽命中 v4bg（mamba/graph_head 归属正确）；graph_* "
          "运行行断言正反例；v4bg- 前缀守卫 train 5 例 + evaluate 3 例；W 范数仪表双口径记录正常")


# ⑨ ------------------------------------------------------------------------
def test_regularization_knobs() -> None:
    coords = _geometry_coords()
    torch.manual_seed(31)
    w = torch.randn(62, 62) * 0.05
    ours = GH.GraphHead(gamma=0.13, gamma_m=0.7, w_init=w)
    src = _load_source_mixer(coords, k_layers=2, gamma=0.13, gamma_m=0.7, w_init=w)
    # 两惩罚方法与源数值一致（同参数下）
    assert torch.allclose(ours.uniformity_penalty(), src.uniformity_penalty(), rtol=1e-6, atol=0.0), \
        "uniformity_penalty 与源不一致"
    assert torch.allclose(ours.sparsity_penalty(), src.sparsity_penalty(), rtol=1e-6, atol=0.0), \
        "sparsity_penalty 与源不一致"
    # λ=0：精确 0 张量，基线损失逐位一致（+0.0 不改正常浮点位）
    base = torch.tensor(1.2345)
    reg0 = GH.graph_regularization(ours, 0.0, 0.0)
    assert reg0.item() == 0.0 and torch.equal(base + reg0, base), "λ=0 未返回精确 0（逐位一致破坏）"
    # λ>0：含惩罚项，与手工复算一致
    lam_u, lam_s = 0.01, 0.02
    reg = GH.graph_regularization(ours, lam_u, lam_s)
    manual = lam_u * ours.uniformity_penalty() + lam_s * ours.sparsity_penalty()
    assert torch.allclose(reg, manual, rtol=1e-7, atol=0.0), "λ>0 组合器与手工复算不一致"
    assert not torch.equal(base + reg, base) and float(reg.detach()) > 0, "λ>0 惩罚项未生效"
    reg_u = GH.graph_regularization(ours, lam_u, 0.0)
    assert torch.allclose(reg_u, lam_u * ours.uniformity_penalty(), rtol=1e-7, atol=0.0), \
        "单旋钮（仅 λ_u）组合不一致"
    print(f"[⑨ 正则] 两惩罚方法与源数值一致；λ=0 精确 0（损失逐位一致）；λ>0 生效且与手工复算一致"
          f"（示例 reg={float(reg.detach()):.6f}）——默认口径 λ_u=λ_s=0 不进损失（10-19 对齐）")


# ⑩ ------------------------------------------------------------------------
def test_channel_names_gate() -> None:
    cfg = C.Config()
    # 正例：真数据根全 46 例顺序全等（扫描数恰 46）
    head = GH.GraphHead(geometry_path=cfg.geometry_path, data_root=cfg.data_root,
                        expect_n_subjects=46)
    assert head.n_scanned == 46, f"扫描 npz 数 {head.n_scanned} != 46"
    assert len(head.channel_names) == 62
    coords, names, n = GH._geometry_gate(cfg.geometry_path, cfg.data_root, 46)
    assert n == 46 and len(names) == 62 and coords.shape == (62, 3)

    # 反例 1：channel_names 顺序不一致 → ValueError
    real = list(names)  # gate 返回的 geometry 62 名单（正例已证与全 46 例 npz 全等）
    tmp = tempfile.mkdtemp(prefix="v4bg_gate_neg_")
    try:
        bad = list(real); bad[1], bad[2] = bad[2], bad[1]  # 交换两通道（顺序破坏）
        os.makedirs(os.path.join(tmp, "HC"))
        np.savez(os.path.join(tmp, "HC", "1_rest_v4.npz"),
                 channel_names=np.array(bad), data=np.zeros((62, 512), dtype=np.float32))
        try:
            GH.GraphHead(geometry_path=cfg.geometry_path, data_root=tmp, expect_n_subjects=1)
            raise AssertionError("通道名 mismatch 未被门禁拦截")
        except ValueError as e:
            assert "不全等" in str(e), f"报错未指明顺序全等断言：{e}"
        # 反例 2：例数不符（匹配但仅 1 例，期望 46）→ ValueError
        np.savez(os.path.join(tmp, "HC", "2_rest_v4.npz"),
                 channel_names=np.array(real), data=np.zeros((62, 512), dtype=np.float32))
        os.remove(os.path.join(tmp, "HC", "1_rest_v4.npz"))
        try:
            GH.GraphHead(geometry_path=cfg.geometry_path, data_root=tmp, expect_n_subjects=46)
            raise AssertionError("例数门禁未拦截")
        except ValueError as e:
            assert "npz 数" in str(e), f"报错未指明例数断言：{e}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # lru_cache 无污染：真数据根再次构造仍正常
    GH.GraphHead()
    print("[⑩ 通道门禁] 真数据根 46/46 顺序全等（geometry_edra_fmt.pt ↔ 全部 npz channel_names）；"
          "mismatch / 例数不符正反例拦截；缓存无污染")


# ⑪ ------------------------------------------------------------------------
def test_split_manifest_override() -> None:
    cfg = C.Config()
    # default-off 钩子正反例：无字段 → v1 原路径；v4bg 有 override → 指向 v4 的 manifest
    class _NoOverride:
        fold, subject_filter, split_seed = cfg.fold, cfg.subject_filter, cfg.split_seed

    v1_manifest = _V1 / "splits" / "fold0_status_ok_seed42.json"
    assert D.split_manifest_path(_NoOverride()) == str(v1_manifest), \
        "default-off 钩子破坏：无字段时未回到 v1 原 manifest 路径（v1/v2/v3 零漂移守护）"
    assert D.split_manifest_path(cfg) == cfg.split_manifest_override
    assert Path(cfg.split_manifest_override).parent == _V4 / "splits", \
        "override 未指向 v4 的 splits/（design §3：显式指向 v4 的 fold0 val4 manifest）"
    assert Path(cfg.split_manifest_override).exists(), "v4 manifest 不存在（首建会写 v1 侧——违规）"

    # 与 v4a 的 manifest 逐字节相同（design §3 拍板事实，独立断言）
    v4a_manifest = _V4A / "splits" / "fold0_status_ok_seed42_val4.json"
    assert Path(cfg.split_manifest_override).read_bytes() == v4a_manifest.read_bytes(), \
        "v4 与 v4a 的 val4 manifest 不再逐字节相同（口径漂移）"

    # 快照：v1/splits 目录与 v4 manifest（hash+mtime 双检，fixate 后须零改动）
    def _snap(p: Path):
        return {f.name: (hashlib.sha256(f.read_bytes()).hexdigest(), f.stat().st_mtime_ns)
                for f in sorted(p.iterdir()) if f.is_file()}
    v1_splits_before = _snap(_V1 / "splits")
    v4_manifest = Path(cfg.split_manifest_override)
    v4_before = (hashlib.sha256(v4_manifest.read_bytes()).hexdigest(),
                 v4_manifest.stat().st_mtime_ns)

    entries = D.load_manifest(cfg.data_root, subject_filter=cfg.subject_filter)
    assert len(entries) == 46, f"status_ok 过滤后应 46 例，got {len(entries)}"
    train, val, test = D.split_fold(
        entries, n_folds=cfg.n_folds, fold=cfg.fold, split_seed=cfg.split_seed,
        n_val_subjects=cfg.n_val_subjects, val_subject_seed=cfg.val_subject_seed,
    )
    assert (len(train), len(val), len(test)) == (32, 4, 10), \
        f"划分计数漂移：{len(train)}/{len(val)}/{len(test)}（应 32/4/10）"
    ids = lambda xs: sorted(e["subject_id"] for e in xs)  # noqa: E731
    saved = json.loads(v4_manifest.read_text(encoding="utf-8"))
    assert ids(train) == saved["train"] and ids(val) == saved["val"] and ids(test) == saved["test"], \
        "本次划分与 v4 固化 manifest 不一致（override 未生效或划分漂移）"
    # fixate 两次（幂等校验，verify-only 不写）+ v1/splits 与 v4 manifest 零改动
    p1 = D.fixate_split(cfg, train, val, test)
    p2 = D.fixate_split(cfg, train, val, test)
    assert Path(p1) == Path(cfg.split_manifest_override) and p1 == p2
    assert _snap(_V1 / "splits") == v1_splits_before, "v1/splits 出现新文件/改动（零改动违规）"
    after = (hashlib.sha256(v4_manifest.read_bytes()).hexdigest(),
             v4_manifest.stat().st_mtime_ns)
    assert after == v4_before, "v4 manifest 被改动（只读违规）"
    print(f"[⑪ manifest] override 生效（v4/splits/fold0_..._val4.json，与 v4a 逐字节相同）；"
          f"划分 32/4/10 与固化清单逐 ID 一致；fixate 幂等 verify-only；"
          f"v1/splits 零新文件、v4 manifest hash+mtime 未动；default-off 钩子正反例通过")


# ⑫ ------------------------------------------------------------------------
def test_hook_kwargs_contract() -> None:
    cfg = C.Config()
    kw = dict(
        n_channels=cfg.n_channels, past_len=cfg.past_len, future_len=cfg.future_len,
        d_model=cfg.d_model, n_layers=cfg.n_layers, d_state=cfg.d_state,
        d_conv=cfg.d_conv, expand=cfg.expand, time_emb_dim=cfg.time_emb_dim,
        use_fast_path=cfg.use_fast_path,
    )
    # 正例：v1 train.py:278-280 / sample.py:52-54 无条件透传的三 kwargs 默认值可构造
    m = M.VelocityNetwork(**kw, bottleneck_rank=None, norm_type="rmsnorm", cond_adaln="off")
    assert m.num_params() == EXPECTED_TOTAL_PARAMS
    # 反例：非默认逐项报错（防隐性口径漂移——本变体无这些分支）
    for bad in ({"bottleneck_rank": 16}, {"norm_type": "layernorm"}, {"cond_adaln": "scale_shift"}):
        try:
            M.VelocityNetwork(**kw, **bad)
            raise AssertionError(f"非默认钩子 kwargs 未被拦截：{bad}")
        except ValueError as e:
            assert next(iter(bad)) in str(e), f"报错未指明 {next(iter(bad))}：{e}"
    # 类默认即契约值（v1 train.py getattr 透传路径：v4bg config 无三字段 → 默认值进入）
    sig = inspect.signature(M.VelocityNetwork.__init__)
    assert sig.parameters["bottleneck_rank"].default is None
    assert sig.parameters["norm_type"].default == "rmsnorm"
    assert sig.parameters["cond_adaln"].default == "off"
    assert not hasattr(cfg, "bottleneck_rank") and not hasattr(cfg, "norm_type") \
        and not hasattr(cfg, "cond_adaln"), "v4bg config 不应携带三钩子字段（默认分支即本变体口径）"
    print("[⑫ 钩子契约] 默认值（None/rmsnorm/off）可构造；非默认逐项 ValueError；"
          "config 无三字段（v1 train/sample 的 getattr 透传→类默认→断言通过）")


def main() -> None:
    test_graph_head_forward_shape()
    test_causality_future_slice()
    test_gradients_finite()
    test_param_count_exact()
    test_config_model_consistency()
    test_ac2_equivalence_ricker_dog()
    test_direction_semantics()
    test_masking_and_prefix_guard()
    test_regularization_knobs()
    test_channel_names_gate()
    test_split_manifest_override()
    test_hook_kwargs_contract()
    print("[PASS] test_v4b_gnn 十二项全部断言通过")


if __name__ == "__main__":
    main()
