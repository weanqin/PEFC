#!/usr/bin/env python3
"""test_v5.py — v5（v2 基线 × 4×[Mamba→GNN 空间块→LayerNorm] 轴分解交替）单测十项
（任务 10-19-gpfm-mamba-v4b-gnn-f0 二次转向，design §4，implement.md S1'.2）。

① 构建 + 前向 shape：(4,62,384)+(4,62,128)+t → (4,62,128)；return_all → (4,62,512)；
   dtype/设备；future 切片一致；梯度有限 + EMA swap/restore + GPU B=256 冒烟
② **交替结构断言**：trunk 结构清单（4×[MambaBlock / GraphSpatialStep(msg_i) / LayerNorm_i]，
   LN/msg 逐块独立实例、A_eff 跨块共享单实例）+ 前向钩子序 = [mamba_i, spatial_i, ln_i] ×4
   （每时间块后恰一空间块一 LN，顺序严格 Mamba→空间块→LN）
③ 因果性：return_all 口径三段（future 扰动不改 past / future 后段扰动不改前段 / past 前缀
   条件生效）+ **空间块位置逐点**（扰动时间位 l 只影响位 l——空间轴不串时间，组件级）
④ **方向单测（B-3 沿用）**：γ_m=0 + w_init 单边注入（W[i,j]=1）——A_eff = γ·W 精确；
   扰动输入通道 j ⇒ 响应集恰为 {j（残差自响应，平凡）, i（唯一跨通道响应）}；
   反向注入（W[j,i]=1）⇒ 仅 {j}（跨通道零响应）；读出转置口径 A_eff.T[j,i]=γ（行=源/列=靶）
⑤ 参数量精确断言（±0）：全模型 152,647 = v2 基线 132,682 + 增量 19,965
   （= 4×(msg 3,906 + LN 124) + W 3,844 + σ 1，design §2.5 预算）+ 逐模块明细 + v2 基线并列实测
⑥ config↔model 一致（d_model/graph_* 双处）+ config_hash 稳定 + 敏感性 +
   **d_model==n_channels 断言正反例** + v1/v2 config_hash 零漂移（子进程口径）
⑦ **A_eff 等价测（AC2）**：同 coords/σ/γ/W 下与 latent-flow 源 LearnableMexHatMixer.
   effective_adjacency allclose rtol=1e-6（ricker + dog 两核，参数取非默认值；源经
   sys.path 导入只读）；pairwise_d 逐位一致；dog_params 数值一致
⑧ 双遮蔽（config+model+graph 命中本目录、mamba→v1）+ `v5-` 前缀守卫 train/evaluate
   正反例 + graph_*/d_model 运行时断言正反例 + 钩子三参默认接受/非默认报错
⑨ 几何门禁正反例（真数据根 46/46 顺序全等 + 名序不一致报错 + 例数不符报错 + lru_cache）+
   λ 旋钮（λ=0 损失逐位不变 + λ>0 与手工复算一致 + 两惩罚与源数值一致）
⑩ 回归：v1 七测 + v4 test_v4b + v4a test_v4a 全绿（子进程）+ v1 `c6ff80ac1c43` /
   v2 `fc2a9ce14948` hash 零漂移

运行：python3 -m pytest models/gpfm-mamba-v5/tests/test_v5.py -q（需 CUDA——mamba_ssm）
      或 python3 models/gpfm-mamba-v5/tests/test_v5.py
"""

from __future__ import annotations

import math
import os

# BLAS 线程上限（environment.md 惯例：有效 16 核 cgroup，须在 numpy 导入前设置；可覆盖）
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "8")

import dataclasses  # noqa: E402
import importlib.util  # noqa: E402
import inspect  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

_HERE = Path(__file__).resolve().parent          # models/gpfm-mamba-v5/tests
_V5 = _HERE.parent                                # models/gpfm-mamba-v5
_V1 = _V5.parent / "gpfm-mamba-v1"                # 只读依赖的 v1 目录
_V4 = _V5.parent / "gpfm-mamba-v4"
_V4A = _V5.parent / "gpfm-mamba-v4a"
_V2 = _V5.parent / "gpfm-mamba-v2"
_LF1 = _V5.parent / "latent-flow-v1"              # 融合公式源（只读；等价测 sys.path 导入）

sys.path.insert(0, str(_V1))
sys.path.insert(0, str(_V5))

import config as C  # noqa: E402
import graph as G  # noqa: E402
import model as M  # noqa: E402
from mamba import MambaBlock  # noqa: E402  # v1 mamba.py（只读复用；v5 目录无此文件）

if not torch.cuda.is_available():
    raise RuntimeError("mamba_ssm 官方 kernel 需要 CUDA")
torch.manual_seed(1234)  # 测试进程定 seed（模型 init/随机张量可复现）

# ---- 登记值（实测精确数 / hash pin；后续任何口径变更须重算登记）----
EXPECTED_TOTAL_PARAMS = 152_647    # = v2 132,682 + 19,965（design §2.5 预算精确吻合）
EXPECTED_V2_PARAMS = 132_682       # v2 基线（v1 VelocityNetwork d_model=62）实测精确值
EXPECTED_INCREMENT = 19_965        # 4×(msg 3,906 + LN 124) + W 3,844 + σ 1
V1_HASH_DEFAULT = "c6ff80ac1c43"   # 基准：v1 main-01 落盘 resolved config hash（零漂移守护）
V2_HASH_DEFAULT = "fc2a9ce14948"   # 基准：v2d62-main-01 落盘 resolved config hash
V5_HASH_DEFAULT = "1ac433fdd3d0"   # v5 默认 config hash（2026-10-19 首建登记）

_MODEL_CACHE: dict = {}


def build_model(dev: str = "cuda") -> M.VelocityNetwork:
    """按 v1 train.py:273-281 的显式 kwargs 构建（运行时契约四件套口径，含三钩子 kwargs
    ——v1 train.py 无条件透传 bottleneck_rank/norm_type/cond_adaln，v5 config 无此三字段
    ⇒ getattr 默认值，即类默认；graph_* 不透传，靠类默认 == config 双处一致防线）。"""
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


# ---------------------------------------------------------------- ① 前向 shape
def test_forward_shape() -> None:
    cfg = C.Config()
    model = build_model()
    xp, xf, t = _inputs(cfg, model.graph.pairwise_d.device)
    model.eval()
    with torch.no_grad():
        v = model(xp, xf, t)
        assert v.shape == (4, cfg.n_channels, cfg.future_len), f"v 形状错误 {tuple(v.shape)}"
        assert torch.isfinite(v).all()
        assert v.dtype == torch.float32 and v.device == xp.device
        out_all = model(xp, xf, t, return_all=True)
        assert out_all.shape == (4, cfg.n_channels, cfg.past_len + cfg.future_len), \
            f"return_all 形状错误 {tuple(out_all.shape)}"
        assert torch.allclose(out_all[:, :, cfg.past_len:], v, atol=1e-6), "future 切片与 return_all 不一致"
    # 梯度有限（全参数张量）
    model.train()
    model.zero_grad(set_to_none=True)
    v = model(xp, xf, t)
    (v ** 2).mean().backward()
    n_grad = 0
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), "存在非有限梯度"
            n_grad += 1
    # W / log_sigma 梯度非零（测试有效性：图参数在损失路径上）
    assert model.graph.W.grad is not None and model.graph.W.grad.abs().sum() > 0, "W 梯度为零（图算子不在损失路径）"
    assert model.graph.log_sigma.grad is not None and model.graph.log_sigma.grad.abs().sum() > 0, \
        "log_sigma 梯度为零（几何先验不在损失路径）"
    # EMA（v1 复制类）update/swap/restore——备份-恢复包裹（缓存模型不得携带 +1 污染，后续测试复用）
    backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
    ema = M.EMA(model, decay=0.999)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    v_mod = model(xp, xf, t)
    ema.update(model)
    with ema.swap(model):
        v_ema = model(xp, xf, t)
    v_restored = model(xp, xf, t)
    assert not torch.allclose(v_ema, v_mod, atol=1e-6), "EMA swap 未生效"
    assert torch.allclose(v_restored, v_mod, atol=1e-6), "swap 恢复后权重与原权重不一致"
    model.load_state_dict(backup)
    # GPU 训练形状冒烟（B=256 forward+backward，v1/v4 惯例）
    model.zero_grad(set_to_none=True)
    xpB = torch.randn(cfg.batch_size, cfg.n_channels, cfg.past_len, device=xp.device)
    xfB = torch.randn(cfg.batch_size, cfg.n_channels, cfg.future_len, device=xp.device)
    tB = torch.rand(cfg.batch_size, device=xp.device)
    torch.cuda.synchronize()
    t0 = time.time()
    vB = model(xpB, xfB, tB)
    (vB ** 2).mean().backward()
    torch.cuda.synchronize()
    dt = time.time() - t0
    assert vB.shape == (cfg.batch_size, cfg.n_channels, cfg.future_len)
    assert torch.isfinite(vB).all()
    print(f"[前向] v {tuple(v.shape)} / return_all {tuple(out_all.shape)} / dtype {v.dtype}；"
          f"{n_grad} 参数张量梯度有限（W/log_sigma 非零）；EMA swap 正常；"
          f"GPU 冒烟 B={cfg.batch_size} fwd+bwd {dt * 1000:.0f} ms，"
          f"peak {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB")


# ---------------------------------------------------- ② 交替结构断言（named + 钩子序）
def test_alternating_structure() -> None:
    cfg = C.Config()
    model = build_model()
    n = cfg.n_layers  # 4
    # named 结构清单：4×[MambaBlock / GraphSpatialStep(msg Linear(62,62)) / LayerNorm(62)]
    assert isinstance(model.blocks, nn.Sequential) and len(model.blocks) == n
    for i in range(n):
        assert isinstance(model.blocks[i], MambaBlock), f"blocks[{i}] 非 MambaBlock"
        assert isinstance(model.spatial_blocks[i], G.GraphSpatialStep), f"spatial_blocks[{i}] 类型错"
        assert tuple(model.spatial_blocks[i].msg.weight.shape) == (cfg.d_model, cfg.d_model) \
            and model.spatial_blocks[i].msg.bias is not None, "msg_i 应为 Linear(62,62)（含 bias）"
        assert isinstance(model.block_lns[i], nn.LayerNorm) \
            and tuple(model.block_lns[i].normalized_shape) == (cfg.d_model,), "LN_i 应为 LayerNorm(62)"
    # 逐块独立实例（msg 4 个互异、LN 4 个互异）
    assert len({id(s.msg) for s in model.spatial_blocks}) == n, "msg_i 须逐块独立实例"
    assert len({id(ln) for ln in model.block_lns}) == n, "LN_i 须逐块独立实例"
    # A_eff 跨块共享单实例：全模型恰一个 MexHatAdjacency；spatial step 不持有 adjacency 子模块
    adj_modules = [mm for mm in model.modules() if isinstance(mm, G.MexHatAdjacency)]
    assert len(adj_modules) == 1 and adj_modules[0] is model.graph, "A_eff 须跨块共享单实例"
    for s in model.spatial_blocks:
        assert not any(isinstance(mm, G.MexHatAdjacency) for mm in s.children()), \
            "GraphSpatialStep 不应注册 adjacency 子模块（共享由 forward 形参传递）"
    # 前向钩子序：每时间块后恰一空间块一 LN，顺序严格 [mamba_i, spatial_i, ln_i] ×4
    order: list[str] = []
    hooks = []

    def _rec(tag: str):
        def hook(mod, inp, out):
            order.append(tag)
        return hook

    for i in range(n):
        hooks.append(model.blocks[i].register_forward_hook(_rec(f"mamba{i}")))
        hooks.append(model.spatial_blocks[i].register_forward_hook(_rec(f"spatial{i}")))
        hooks.append(model.block_lns[i].register_forward_hook(_rec(f"ln{i}")))
    xp, xf, t = _inputs(cfg, model.graph.pairwise_d.device)
    model.eval()
    with torch.no_grad():
        model(xp, xf, t)
    for h in hooks:
        h.remove()
    expected = [tag for i in range(n) for tag in (f"mamba{i}", f"spatial{i}", f"ln{i}")]
    assert order == expected, f"交替执行序破坏：{order}（应 {expected}）"
    # head 前置链在交替主体之后（head_norm/head 不在钩子序内——只挂在 trunk 12 模块上）
    print(f"[交替结构] 4×[Mamba→空间块→LayerNorm] 子模块清单与执行序逐项一致（钩子序 {order}）；"
          f"msg/LN 逐块独立；A_eff 单实例共享")


# ---------------------------------------------------------------- ③ 因果性
def test_causality() -> None:
    cfg = C.Config()
    model = build_model()
    xp, xf, t = _inputs(cfg, model.graph.pairwise_d.device)
    model.eval()
    with torch.no_grad():
        ref = model(xp, xf, t, return_all=True)
        xf2 = xf.clone()
        xf2[:, :, :] += 1.0
        out2 = model(xp, xf2, t, return_all=True)
        assert torch.allclose(out2[:, :, : cfg.past_len], ref[:, :, : cfg.past_len], atol=1e-5), \
            "扰动 future 输入改变了 past 位置输出（因果性破坏）"
        assert not torch.allclose(out2[:, :, cfg.past_len:], ref[:, :, cfg.past_len:], atol=1e-5), \
            "扰动 future 输入应改变 future 输出"
        xf3 = xf.clone()
        xf3[:, :, 100:] += 1.0
        out3 = model(xp, xf3, t, return_all=True)
        assert torch.allclose(out3[:, :, cfg.past_len : cfg.past_len + 50],
                              ref[:, :, cfg.past_len : cfg.past_len + 50], atol=1e-5), \
            "future 后段扰动影响了前段输出（序列内因果性破坏）"
        xp4 = xp.clone()
        xp4[:, :, -50:] += 1.0
        out4 = model(xp4, xf, t, return_all=True)
        assert not torch.allclose(out4[:, :, cfg.past_len:], ref[:, :, cfg.past_len:], atol=1e-5), \
            "past 扰动未影响 future 输出（前缀条件未生效）"
    # 空间块位置逐点（组件级）：扰动时间位 l 只影响位 l——空间轴聚合不串时间
    torch.manual_seed(11)
    A = torch.randn(62, 62)
    step = G.GraphSpatialStep(62)
    x = torch.randn(3, 33, 62)
    y1 = step(x, A)
    x2 = x.clone()
    x2[:, 17, :] += 1.0
    y2 = step(x2, A)
    assert torch.equal(y2[:, 17], y1[:, 17]) is False, "扰动位自身应变化（测试有效性）"
    assert torch.equal(y2[:, :17], y1[:, :17]) and torch.equal(y2[:, 18:], y1[:, 18:]), \
        "空间块位置非逐点（时间位 l 的影响泄漏到其他时间位）"
    print("[因果性] future 扰动不改 past；future 后段扰动不改前段；past 前缀条件生效；"
          "空间块位置逐点（扰动 l 位只影响 l 位）")


# ---------------------------------------------------------------- ④ 方向单测
def test_direction() -> None:
    torch.manual_seed(13)
    coords = torch.randn(62, 3)
    i0, j0 = 7, 5
    gam = 0.1
    # 正向注入：W[i0,j0]=1（A[i,j]=j 驱动 i ⇒ j0 驱动 i0）
    W = torch.zeros(62, 62)
    W[i0, j0] = 1.0
    adj = G.MexHatAdjacency(62, coords=coords, gamma=gam, gamma_m=0.0, w_init=W)
    A = adj.effective_adjacency()
    assert torch.allclose(A, gam * W, rtol=1e-7, atol=1e-8), "γ_m=0 ⇒ A_eff 应精确等于 γ·W"
    assert abs(A[i0, j0].item() - gam) < 1e-7 and A[j0, i0].item() == 0.0, "源语义 A[i,j]=j 驱动 i 破坏"
    assert abs(A.t()[j0, i0].item() - gam) < 1e-7, "读出转置口径（行=源/列=靶）破坏：A_eff.T[j,i] 应为 γ"
    # 隔离 msg（恒等化）后经 GraphSpatialStep 验证响应集
    step = G.GraphSpatialStep(62)
    with torch.no_grad():
        step.msg.weight.copy_(torch.eye(62))
        step.msg.bias.zero_()
    x = torch.randn(2, 21, 62)
    x2 = x.clone()
    x2[:, :, j0] += 1.0
    y1, y2 = step(x, A), step(x2, A)
    diff = (y1 - y2).abs().amax(dim=(0, 1))
    responded = set(torch.nonzero(diff > 1e-7).flatten().tolist())
    assert responded == {j0, i0}, \
        f"正向注入响应集应恰为 {{j0(残差自响应), i0(唯一跨通道)}}，got {sorted(responded)}"
    # 反向注入：W[j0,i0]=1 ⇒ 扰动 j0 跨通道零响应（仅残差自响应）
    Wb = torch.zeros(62, 62)
    Wb[j0, i0] = 1.0
    adj_b = G.MexHatAdjacency(62, coords=coords, gamma=gam, gamma_m=0.0, w_init=Wb)
    Ab = adj_b.effective_adjacency()
    y1b, y2b = step(x, Ab), step(x2, Ab)
    diffb = (y1b - y2b).abs().amax(dim=(0, 1))
    responded_b = set(torch.nonzero(diffb > 1e-7).flatten().tolist())
    assert responded_b == {j0}, f"反向注入跨通道零响应破坏：响应集 {sorted(responded_b)}（应仅 {{j0}}）"
    print(f"[方向] A_eff=γ·W 精确；源语义 A[i,j]=j 驱动 i（正向注入响应集 {{{j0}, {i0}}}、"
          f"反向注入 {{{j0}}}）；读出转置 A_eff.T[{j0},{i0}]={gam}")


# ---------------------------------------------------------------- ⑤ 参数量
def test_param_count() -> None:
    model = build_model()
    cfg = C.Config()
    n = model.num_params()
    assert n == EXPECTED_TOTAL_PARAMS, f"参数量精确值 {n} != {EXPECTED_TOTAL_PARAMS}"
    d = {k: p.numel() for k, p in model.named_parameters()}
    seg = d["seg_past"] + d["seg_future"]
    in_proj = d["in_proj.weight"] + d["in_proj.bias"]
    blocks = sum(v for k, v in d.items() if k.startswith("blocks."))
    graph = sum(v for k, v in d.items() if k.startswith("graph."))
    spatial = sum(v for k, v in d.items() if k.startswith("spatial_blocks."))
    lns = sum(v for k, v in d.items() if k.startswith("block_lns."))
    head_norm = d["head_norm.weight"]
    head = d["head.weight"] + d["head.bias"]
    dm = cfg.d_model
    assert seg == 2 and in_proj == (dm + 1 + cfg.time_emb_dim) * dm + dm, "seg/in_proj 明细漂移"
    assert blocks == 4 * 30_938, f"时间块明细漂移：{blocks}（应 4×30,938——Mamba(62) x_proj/out_proj 无 bias 实测口径）"
    assert graph == 62 * 62 + 1, f"图算子明细漂移：{graph}（应 W 3,844 + σ 1）"
    assert spatial == 4 * (dm * dm + dm), f"msg 明细漂移：{spatial}（应 4×3,906）"
    assert lns == 4 * (dm * 2), f"LN 明细漂移：{lns}（应 4×124）"
    assert head_norm == dm and head == dm * dm + dm, "head 明细漂移"
    # v2 基线并列（同进程独立装载 v1 model.py，d_model=62 构建——薄变体基线口径）
    spec = importlib.util.spec_from_file_location("gpfm_mamba_v1_model_ref", str(_V1 / "model.py"))
    v1m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v1m)
    v2_model = v1m.VelocityNetwork(
        n_channels=cfg.n_channels, past_len=cfg.past_len, future_len=cfg.future_len,
        d_model=62, n_layers=cfg.n_layers, d_state=cfg.d_state, d_conv=cfg.d_conv,
        expand=cfg.expand, time_emb_dim=cfg.time_emb_dim, use_fast_path=cfg.use_fast_path,
    )
    n_v2 = v2_model.num_params()
    assert n_v2 == EXPECTED_V2_PARAMS, f"v2 基线参数量 {n_v2} != {EXPECTED_V2_PARAMS}"
    assert n - n_v2 == EXPECTED_INCREMENT, \
        f"增量 {n - n_v2} != {EXPECTED_INCREMENT}（design §2.5 预算 4×(3,906+124)+3,844+1）"
    print(f"[参数量] v5 {n} = v2 {n_v2} + {EXPECTED_INCREMENT}（4×(msg 3,906+LN 124)+W 3,844+σ 1，"
          f"design §2.5 预算精确吻合）；明细：seg 2 + in_proj {in_proj} + blocks {blocks} + "
          f"graph {graph} + spatial {spatial} + lns {lns} + head_norm {head_norm} + head {head}")


# ------------------------------------------------- ⑥ config↔model 一致 + hash 稳定
def test_config_model_consistency() -> None:
    cfg = C.Config()
    assert cfg.d_model == 62 == cfg.n_channels, f"轴分解不变量破坏：d_model {cfg.d_model} / C {cfg.n_channels}"
    assert cfg.variant_name == "gpfm-mamba-v5"
    sig = inspect.signature(M.VelocityNetwork.__init__)
    for name in ("d_model", "graph_kernel", "graph_gamma", "graph_gamma_m",
                 "graph_w_init", "geometry_path"):
        assert sig.parameters[name].default == getattr(cfg, name), \
            f"model 类默认 {name} != cfg.{name}（双处一致性断裂）"
    assert cfg.graph_kernel == "ricker" and cfg.graph_gamma == 0.1 and cfg.graph_gamma_m == 1.0 \
        and cfg.graph_w_init == "" and cfg.graph_lambda_uniform == 0.0 and cfg.graph_lambda_sparse == 0.0, \
        "graph 字段口径漂移（ricker/γ=0.1/γ_m=1.0/零初始化 W/λ 关闭）"
    h1, h2 = C.Config().config_hash(), C.build_config().config_hash()
    assert h1 == h2, "config_hash 不稳定（同配置两次构建不一致）"
    assert h1 == V5_HASH_DEFAULT, \
        f"v5 config_hash {h1} != 登记值 {V5_HASH_DEFAULT}（口径变更后须显式重算登记）"
    for field, val in (("graph_gamma", 0.5), ("graph_kernel", "dog"), ("d_model", 16)):
        h_mut = dataclasses.replace(cfg, **{field: val}).config_hash()
        assert h_mut != h1, f"config_hash 对 {field} 不敏感（防线失效）"
    # d_model == n_channels 断言正反例（要求③机器可读防线）
    with pytest.raises(ValueError, match="d_model"):
        M.VelocityNetwork(n_channels=62, d_model=16)  # 反例：模型维 != 通道轴
    M.VelocityNetwork(n_channels=62, d_model=62)      # 正例：62==62 构建通过（构建即抛/过，不经前向）
    # v1/v2 config_hash 零漂移（子进程隔离导入，基准 = 已落盘 run 的 resolved hash）
    for path, base, name in ((str(_V1), V1_HASH_DEFAULT, "v1"),
                             (str(_V2), V2_HASH_DEFAULT, "v2")):
        code = (f"import sys; sys.path.insert(0, {path!r}); import config as C; "
                f"print(C.Config().config_hash())")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        got = out.stdout.strip()
        assert got == base, f"{name} config_hash 漂移：{got} != 基准 {base}（v1/v2 须零改动）"
    print(f"[config↔model] d_model/graph_* 六字段双处一致；config_hash {h1} 稳定且对 "
          f"graph_gamma/graph_kernel/d_model 敏感；d_model==C 断言正反例；"
          f"v1 {V1_HASH_DEFAULT} / v2 {V2_HASH_DEFAULT} 零漂移")


# ---------------------------------------------------------------- ⑦ A_eff 等价测
def test_aeff_equivalence() -> None:
    torch.manual_seed(3)
    coords = torch.randn(62, 3) * 0.1
    W = torch.randn(62, 62) * 0.3
    # ---- ricker（σ 取非默认 0.05）----
    src = _load_source_mixer(coords)
    ours = G.MexHatAdjacency(62, coords=coords)
    assert ours.kernel == "ricker" and src.kernel == "ricker"
    with torch.no_grad():
        ours.log_sigma.fill_(math.log(0.05))
        src.log_sigma.fill_(math.log(0.05))
        ours.W.copy_(W)
        src.W.copy_(W)
        src.gamma_buf.fill_(0.1)
        src.gamma_m.fill_(1.0)  # ours 构造默认即 γ=0.1/γ_m=1.0
    assert torch.equal(ours.pairwise_d, src.pairwise_d), "pairwise_d 不逐位一致（coords 同源）"
    A_o, A_s = ours.effective_adjacency(), src.effective_adjacency()
    assert torch.allclose(A_o, A_s, rtol=1e-6, atol=0.0), \
        f"ricker A_eff 不等价：max|Δ| = {(A_o - A_s).abs().max().item():.3e}"
    max_r = (A_o - A_s).abs().max().item()
    # ---- dog（4 参数全非默认）----
    dog_init = {"sigma_e": 0.03, "sigma_i": 0.09, "Ae": 1.3, "Ai": 0.45}
    src_d = _load_source_mixer(coords, kernel="dog", dog_init=dog_init, gamma=0.1, gamma_m=1.0)
    ours_d = G.MexHatAdjacency(62, coords=coords, kernel="dog", dog_init=dog_init)
    assert ours_d.dog_params().keys() == src_d.dog_params().keys()
    for k in ("sigma_e", "sigma_i", "Ae", "Ai"):
        assert abs(ours_d.dog_params()[k] - src_d.dog_params()[k]) < 1e-12, f"dog_params[{k}] 不一致"
    with torch.no_grad():
        for p_o, p_s in ((ours_d.log_sigma_e, src_d.log_sigma_e),
                         (ours_d.log_delta_sigma, src_d.log_delta_sigma),
                         (ours_d.log_Ae, src_d.log_Ae),
                         (ours_d.log_delta_a, src_d.log_delta_a)):
            p_o.fill_(p_s)  # 参数逐位对齐后只剩公式等价性
        ours_d.W.copy_(W)
        src_d.W.copy_(W)
    A_od, A_sd = ours_d.effective_adjacency(), src_d.effective_adjacency()
    assert torch.allclose(A_od, A_sd, rtol=1e-6, atol=0.0), \
        f"dog A_eff 不等价：max|Δ| = {(A_od - A_sd).abs().max().item():.3e}"
    max_d = (A_od - A_sd).abs().max().item()
    assert ours.dog_params() == {} and src.dog_params() == {}, "ricker 核 dog_params 应为空 dict（源语义）"
    print(f"[A_eff 等价] ricker max|Δ|={max_r:.3e} / dog max|Δ|={max_d:.3e}（allclose rtol=1e-6）；"
          f"pairwise_d 逐位一致；dog_params 四参数一致（σe 0.03/σi 0.09/Ae 1.3/Ai 0.45）")


# ------------------------------------------------- ⑧ 双遮蔽 + 前缀守卫 + 钩子三参
def test_masking_and_guard() -> None:
    def load_wrapper(mod_name: str, filename: str):
        spec = importlib.util.spec_from_file_location(mod_name, str(_V5 / filename))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    tr = load_wrapper("v5_train_wrapper_under_test", "train.py")
    # 双遮蔽正例：config/model/graph 均命中 v5 目录，mamba 命中 v1
    tr._shadow_v5()
    import config as _C
    import model as _M
    assert Path(_C.__file__).resolve().parent == _V5, f"config 遮蔽未命中 v5：{_C.__file__}"
    assert Path(_M.__file__).resolve().parent == _V5, f"model 遮蔽未命中 v5：{_M.__file__}"
    assert _M.VelocityNetwork is M.VelocityNetwork, "遮蔽后的 model 与测试直接导入的不一致"
    import graph as _Gg
    assert Path(_Gg.__file__).resolve().parent == _V5 and _Gg.MexHatAdjacency is G.MexHatAdjacency, \
        f"graph 解析未命中 v5：{_Gg.__file__}"
    import mamba as _mb
    assert Path(_mb.__file__).resolve().parent == _V1, f"mamba 应解析到 v1：{_mb.__file__}"

    # graph_*/d_model 运行时断言：正例（真实 cfg）通过；反例（shim 漂移）SystemExit
    tr._assert_model_config_consistency()

    class _BadCfg:
        d_model = 62
        n_channels = 62
        graph_kernel = "ricker"
        graph_gamma = 0.5          # 漂移
        graph_gamma_m = 1.0
        graph_w_init = ""
        geometry_path = "x"

    class _BadConfMod:
        Config = staticmethod(lambda: _BadCfg())

    real_conf = sys.modules["config"]
    try:
        sys.modules["config"] = _BadConfMod()
        try:
            tr._assert_model_config_consistency()
            raise AssertionError("graph_gamma 漂移未被运行时断言拦截")
        except SystemExit as e:
            assert "graph_gamma" in str(e), f"退出信息未指明 graph_gamma：{e}"
        # d_model != n_channels 反例
        _BadCfg.graph_gamma = 0.1
        _BadCfg.d_model = 16
        try:
            tr._assert_model_config_consistency()
            raise AssertionError("d_model 漂移未被运行时断言拦截")
        except SystemExit as e:
            assert "d_model" in str(e), f"退出信息未指明 d_model：{e}"
    finally:
        sys.modules["config"] = real_conf

    # 前缀守卫（train wrapper）：正例 + 反例 + 缺省自动派生
    tr._guard_run_id(["prog", "--run-id", "v5-smoke-a"])                              # 正例
    argv_resume_ok = ["prog", "--resume", "/x/v5-bench-1ep/checkpoints/gpfm-mamba-v5-last.pt"]
    tr._guard_run_id(argv_resume_ok)                                                  # resume 派生正例
    for bad in (["prog", "--run-id", "v2d62-xxx"],
                ["prog", "--run-id", "v4bg-xxx"],
                ["prog", "--run-id", "main-01"],
                ["prog", "--resume", "/x/v2d62-main-01/checkpoints/last.pt"]):
        try:
            tr._guard_run_id(bad)
            raise AssertionError(f"前缀守卫未拦截：{bad}")
        except SystemExit:
            pass
    argv_auto = ["prog"]                                                              # 缺省自动派生 v5- 前缀
    tr._guard_run_id(argv_auto)
    assert argv_auto[-2] == "--run-id" and argv_auto[-1].startswith("v5-"), argv_auto

    # evaluate wrapper 守卫：缺 --run-id / 错前缀退出，正例通过
    ev = load_wrapper("v5_evaluate_wrapper_under_test", "evaluate.py")
    ev._guard_run_id(["prog", "--run-id", "v5-main-01"])
    for bad in (["prog"], ["prog", "--run-id", "main-01"], ["prog", "--run-id", "v2d62-main-01"]):
        try:
            ev._guard_run_id(bad)
            raise AssertionError(f"evaluate 守卫未拦截：{bad}")
        except SystemExit:
            pass

    # 钩子三参（§2.3）：默认值可构造（build_model 已证）；非默认逐项报错
    for kwargs, field in ((dict(bottleneck_rank=16), "bottleneck_rank"),
                          (dict(norm_type="layernorm"), "norm_type"),
                          (dict(cond_adaln="scale_shift"), "cond_adaln")):
        with pytest.raises(ValueError, match=field):
            M.VelocityNetwork(n_channels=62, d_model=62, **kwargs)
    M.VelocityNetwork(n_channels=62, d_model=62,
                      bottleneck_rank=None, norm_type="rmsnorm", cond_adaln="off")  # 默认三参显式接受
    print("[遮蔽/守卫] config+model+graph 三模块遮蔽命中 v5（mamba→v1）；graph_*/d_model 运行时"
          "断言正反例；v5- 前缀守卫 train 5 例 + evaluate 4 例全过；钩子三参默认接受/非默认报错")


# ------------------------------------------------- ⑨ 几何门禁 + λ 旋钮
def test_geometry_gate_and_lambda() -> None:
    cfg = C.Config()
    # 正例：真数据根全 46 例顺序全等（choke point 首扫；lru_cache 二次调用同对象）
    coords, names = G.load_geometry(cfg.geometry_path, cfg.data_root)
    assert coords.shape == (62, 3) and coords.dtype == torch.float32
    assert len(names) == 62 and isinstance(names, tuple)
    coords2, names2 = G.load_geometry(cfg.geometry_path, cfg.data_root)
    assert coords2 is coords and names2 is names, "lru_cache 未按 (geometry_path, data_root) 命中"
    # 反例 1：通道名序不一致
    bad_names = list(names)
    bad_names[0], bad_names[1] = bad_names[1], bad_names[0]
    with tempfile.TemporaryDirectory() as td:
        sub = Path(td) / "HC"
        sub.mkdir()
        np.savez(sub / "x_rest_v4.npz", channel_names=np.array(bad_names),
                 data=np.zeros((62, 4), dtype=np.float32))
        with pytest.raises(ValueError, match="channel_names"):
            G.load_geometry(cfg.geometry_path, td)
    # 反例 2：例数不符（名序正确，仅 1 例 ≠ 46）
    with tempfile.TemporaryDirectory() as td2:
        sub = Path(td2) / "HC"
        sub.mkdir()
        np.savez(sub / "x_rest_v4.npz", channel_names=np.array(names),
                 data=np.zeros((62, 4), dtype=np.float32))
        with pytest.raises(ValueError, match="例数"):
            G.load_geometry(cfg.geometry_path, td2)

    # λ 旋钮（组件级）：λ=0 损失逐位不变；λ>0 与手工复算一致；两惩罚与源数值一致
    torch.manual_seed(5)
    coords_r = torch.randn(62, 3)
    adj = G.MexHatAdjacency(62, coords=coords_r)
    x = torch.randn(2, 9, 62)
    loss = (x ** 2).mean()
    reg0 = G.graph_regularization(adj, 0.0, 0.0)
    assert reg0.item() == 0.0 and torch.equal(loss, loss + reg0), "λ=0 须损失逐位不变"
    lam_u, lam_s = 0.3, 0.02
    reg = G.graph_regularization(adj, lam_u, lam_s)
    manual = lam_u * adj.uniformity_penalty() + lam_s * adj.sparsity_penalty()
    assert torch.equal(reg, manual), "λ>0 组合器与手工复算不一致"
    # 惩罚与 latent-flow 源数值一致（同 coords/σ/W 对齐）
    src = _load_source_mixer(coords_r)
    with torch.no_grad():
        src.log_sigma.fill_(adj.log_sigma.detach())
        src.W.copy_(adj.W.detach())
        src.gamma_buf.fill_(float(adj.gamma_buf))
        src.gamma_m.fill_(float(adj.gamma_m))
    assert torch.allclose(adj.uniformity_penalty(), src.uniformity_penalty(), rtol=1e-6)
    assert torch.allclose(adj.sparsity_penalty(), src.sparsity_penalty(), rtol=1e-6)
    assert cfg.graph_lambda_uniform == 0.0 and cfg.graph_lambda_sparse == 0.0, \
        "config 双 λ 默认须为 0（关闭——不进损失）"
    print("[几何门禁] 真数据根 46/46 顺序全等（lru_cache 生效）；名序不一致/例数不符反例均报错；"
          "λ=0 损失逐位不变 + λ>0 手工复算一致 + uniformity/sparsity 与源数值一致（rtol=1e-6）")


# ---------------------------------------------------------------- ⑩ 回归
def test_regressions() -> None:
    # v1 七测（script 式，子进程直接执行；exit 0 = 全部断言通过）
    v1_tests = ["test_bandlimit.py", "test_data.py", "test_gc_psd.py", "test_gpr.py",
                "test_losses.py", "test_model.py", "test_sample.py"]
    for f in v1_tests:
        r = subprocess.run([sys.executable, str(_V1 / "tests" / f)],
                           capture_output=True, text=True)
        assert r.returncode == 0, f"v1 {f} 回归失败：\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}"
    # v4 test_v4b / v4a test_v4a
    for path, name in ((str(_V4 / "tests" / "test_v4b.py"), "v4 test_v4b"),
                       (str(_V4A / "tests" / "test_v4a.py"), "v4a test_v4a")):
        r = subprocess.run([sys.executable, path], capture_output=True, text=True)
        assert r.returncode == 0, f"{name} 回归失败：\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}"
    # v1/v2 config_hash 零漂移（与 ⑥ 同款子进程口径，此处为 ⑩ 的独立断言）
    for path, base, name in ((str(_V1), V1_HASH_DEFAULT, "v1"),
                             (str(_V2), V2_HASH_DEFAULT, "v2")):
        code = (f"import sys; sys.path.insert(0, {path!r}); import config as C; "
                f"print(C.Config().config_hash())")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        assert out.stdout.strip() == base, f"{name} hash 漂移：{out.stdout.strip()} != {base}"
    print(f"[回归] v1 七测 + v4 test_v4b + v4a test_v4a 全绿（exit 0）；"
          f"v1 {V1_HASH_DEFAULT} / v2 {V2_HASH_DEFAULT} 零漂移")


def main() -> None:
    test_forward_shape()             # ①
    test_alternating_structure()     # ②
    test_causality()                 # ③
    test_direction()                 # ④
    test_param_count()               # ⑤
    test_config_model_consistency()  # ⑥
    test_aeff_equivalence()          # ⑦
    test_masking_and_guard()         # ⑧
    test_geometry_gate_and_lambda()  # ⑨
    test_regressions()               # ⑩
    print("[PASS] test_v5 十项全部断言通过")


if __name__ == "__main__":
    main()
