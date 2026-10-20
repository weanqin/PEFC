#!/usr/bin/env python3
"""test_v5_fmt.py — v5-fmt（v5 轴分解交替 × edra_fmt/250 Hz fmt 口径）单测十一项
（任务 10-20-gpfm-mamba-v5-fmt-f0，implement.md S1.3；v5 十项的 fmt 实例 + fmt 差异项）。

① 构建 + 前向 shape（fmt 口径 LayerNorm 构建）：(4,62,384)+(4,62,128)+t → (4,62,128)；
   return_all → (4,62,512)；dtype/设备；future 切片一致；梯度有限 + EMA swap/restore +
   GPU B=256 冒烟
② **交替结构断言 + fmt norm 位**：trunk 4×[MambaBlock / GraphSpatialStep(msg_i) /
   LayerNorm_i]（LN/msg 逐块独立、A_eff 跨块共享单实例）+ 前向钩子序 [mamba_i, spatial_i,
   ln_i]×4 + **fmt 口径 5 norm 位（4×MambaBlock pre-norm + head_norm）全 LayerNorm(62)**
③ 因果性：return_all 三段 + 空间块位置逐点（组件级）
④ **方向单测**：γ_m=0 + w_init 单边注入——A_eff=γ·W 精确；正向注入响应集恰 {j, i}、
   反向注入仅 {j}；读出转置口径 A_eff.T[j,i]=γ
⑤ 参数量精确断言（±0）：**152,957** = v5 rms 分支 152,647 + **310**（fmt 5 norm 位
   rmsnorm→layernorm 各 +62 bias；实测写死）+ 逐模块明细（blocks 4×31,000 / head_norm 124）+
   v5 并列实测（子进程隔离装载 v5 model.py，state_dict 键集与 rms 分支逐位一致）
⑥ config↔model 一致（d_model/graph_* 双处）+ **fmt 口径字段**（data_root/fs=250/
   norm_type="layernorm"/period_p=25/gen_lowpass_hz=None/n_val_subjects=4/
   split_manifest_override→v1-fmt splits）+ config_hash 稳定 + pin + 敏感性 +
   d_model==n_channels 断言正反例 + v1/v2/v5/v1-fmt config_hash 零漂移（子进程口径）
⑦ **A_eff 等价测**：同 coords/σ/γ/W 下与 latent-flow 源 LearnableMexHatMixer.
   effective_adjacency allclose rtol=1e-6（ricker + dog 两核非默认参数；源只读导入）
⑧ 双遮蔽（config+model+graph 命中本目录、mamba→v1）+ `v5-` 前缀守卫 train/evaluate
   正反例 + graph_*/d_model 运行时断言正反例 + **钩子（fmt 口径）：norm_type="layernorm"
   接受、"rmsnorm" 接受（默认分支）、其余非法值报错**；bottleneck_rank/cond_adaln 非默认报错
⑨ 几何门禁正反例（**fmt 数据根全 48/48 例**通道名与 geometry_edra_fmt.pt 顺序全等 +
   名序不一致报错 + 例数不符报错（1≠48）+ lru_cache）+ λ 旋钮（λ=0 逐位不变 + λ>0 手工
   复算一致 + 两惩罚与源数值一致）
⑩ 回归：v1 七测 + v4 test_v4b + v4a test_v4a + **v5 test_v5** 全绿（子进程）+ pin 复验
   v1 `c6ff80ac1c43` / v2 `fc2a9ce14948` / v5 `1ac433fdd3d0` / v1-fmt `0dc0d5249c68`
⑪ **划分固化只读复用（S1.2）**：fmt 48 例（HC24/MDD24）→ 34/4/10 + val 分层 2HC+2MDD +
   三折互斥 + fixate_split 幂等校验通过 **v1-fmt manifest（零重新划分，文件逐字节不变）**

运行：python3 -m pytest models/gpfm-mamba-v5-fmt/tests/test_v5_fmt.py -q（需 CUDA——mamba_ssm）
      或 python3 models/gpfm-mamba-v5-fmt/tests/test_v5_fmt.py
"""

from __future__ import annotations

import os

# BLAS 线程上限（environment.md 惯例：有效 16 核 cgroup，须在 numpy 导入前设置；可覆盖）
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "8")

import dataclasses  # noqa: E402
import importlib.util  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

_HERE = Path(__file__).resolve().parent          # models/gpfm-mamba-v5-fmt/tests
_V5F = _HERE.parent                               # models/gpfm-mamba-v5-fmt
_MODELS = _V5F.parent
_V1 = _MODELS / "gpfm-mamba-v1"                   # 只读依赖的 v1 目录
_V4 = _MODELS / "gpfm-mamba-v4"
_V4A = _MODELS / "gpfm-mamba-v4a"
_V2 = _MODELS / "gpfm-mamba-v2"
_V5 = _MODELS / "gpfm-mamba-v5"                   # 结构源（只读；回归 + rms 分支并列实测）
_V1FMT = _MODELS / "gpfm-mamba-v1-fmt"            # fmt 口径先例（只读；manifest 复用源）
_LF1 = _MODELS / "latent-flow-v1"                 # A_eff 公式源（只读；等价测 sys.path 导入）

sys.path.insert(0, str(_V1))
sys.path.insert(0, str(_V5F))

import config as C  # noqa: E402
import data as D  # noqa: E402  # v1 data.py（split_manifest_override/fixate_split 钩子所在）
import graph as G  # noqa: E402
import model as M  # noqa: E402
from mamba import MambaBlock  # noqa: E402  # v1 mamba.py（只读复用；v5-fmt 目录无此文件）

if not torch.cuda.is_available():
    raise RuntimeError("mamba_ssm 官方 kernel 需要 CUDA")
torch.manual_seed(1234)  # 测试进程定 seed（模型 init/随机张量可复现）

# ---- 登记值（实测精确数 / hash pin；后续任何口径变更须重算登记）----
EXPECTED_TOTAL_PARAMS = 152_957   # = v5 rms 分支 152,647 + 310（fmt 5 norm 位 ×62 bias，实测）
EXPECTED_V5_PARAMS = 152_647      # v5 全模型实测精确值（rms 分支并列复测 + 子进程 v5 model.py）
EXPECTED_NORM_INCREMENT = 310     # 5×62：4×MambaBlock pre-norm + head_norm 各 +62（layernorm bias）
V1_HASH_DEFAULT = "c6ff80ac1c43"  # 基准：v1 main-01 落盘 resolved config hash（零漂移守护）
V2_HASH_DEFAULT = "fc2a9ce14948"  # v2d62-main-01 落盘 resolved config hash
V5_HASH_DEFAULT = "1ac433fdd3d0"  # v5 默认 config hash（2026-10-19 登记）
V1FMT_HASH_DEFAULT = "0dc0d5249c68"  # v1-fmt 默认 config hash（2026-10-16 首建 pin，复验）
V5FMT_HASH_DEFAULT = "a64c0d416756"  # v5-fmt 默认 config hash（2026-10-20 用户拍板启用 early_stop_patience=10 后的现行 pin；无早停字段旧版 82acf24340e3 留档 experiment.md）
V5FMT_HASH_EPOCHS1 = "603188ccc0a1"  # v5-fmt epochs=1 覆写 pin（同拍板后；旧版 f4be07288e9b 留档——smoke/bench 两 run 实跑于旧 pin 口径，登记自洽）
FMT_DATA_ROOT = "/root/autodl-tmp/edra_processed_fmt"
FMT_OUTPUTS_ROOT = "/root/npi/outputs/gpfm-mamba-f0"

_MODEL_CACHE: dict = {}


def build_model(dev: str = "cuda") -> M.VelocityNetwork:
    """按 v1 train.py:273-281 的显式 kwargs 构建（运行时契约四件套口径，含三钩子 kwargs
    ——v1 train.py 无条件透传 bottleneck_rank/norm_type/cond_adaln，v5-fmt config 的
    norm_type="layernorm" 经 getattr 透传 ⇒ fmt 结构；graph_* 不透传，靠类默认 == config
    双处一致防线）。"""
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
    # GPU 训练形状冒烟（B=256 forward+backward，v5 惯例）
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


# ------------------------------------------- ② 交替结构断言 + fmt norm 位（named + 钩子序）
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
    # fmt 口径 norm 位（D4）：5 位全 LayerNorm（4×MambaBlock pre-norm + head_norm）
    norm_mods = [b.norm for b in model.blocks] + [model.head_norm]
    assert len(norm_mods) == 5 and all(isinstance(nm, nn.LayerNorm) for nm in norm_mods), \
        "fmt 口径 5 norm 位应全为 LayerNorm（4 pre-norm + head_norm）"
    assert all(tuple(nm.normalized_shape) == (cfg.d_model,) for nm in norm_mods), "LayerNorm 形状应 (62,)"
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
    print(f"[交替结构] 4×[Mamba→空间块→LayerNorm] 子模块清单与执行序逐项一致（钩子序 {order}）；"
          f"msg/LN 逐块独立；A_eff 单实例共享；fmt 5 norm 位全 LayerNorm(62)")


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
    head_norm = sum(v for k, v in d.items() if k.startswith("head_norm."))
    head = d["head.weight"] + d["head.bias"]
    dm = cfg.d_model
    assert seg == 2 and in_proj == (dm + 1 + cfg.time_emb_dim) * dm + dm, "seg/in_proj 明细漂移"
    assert blocks == 4 * 31_000, f"时间块明细漂移：{blocks}（应 4×31,000——Mamba(62) + LayerNorm pre-norm 实测口径）"
    assert graph == 62 * 62 + 1, f"图算子明细漂移：{graph}（应 W 3,844 + σ 1）"
    assert spatial == 4 * (dm * dm + dm), f"msg 明细漂移：{spatial}（应 4×3,906）"
    assert lns == 4 * (dm * 2), f"LN 明细漂移：{lns}（应 4×124）"
    assert head_norm == dm * 2 and d.get("head_norm.bias") == dm, \
        f"head_norm 明细漂移：{head_norm}（fmt 口径 LayerNorm 应 weight+bias=124）"
    assert head == dm * dm + dm, "head 明细漂移"
    # v5 并列实测（子进程隔离装载 v5 model.py——rmsnorm 默认分支；sys.path 只含 v5+v1）
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(_V1)!r})\n"
        f"sys.path.insert(0, {str(_V5)!r})\n"
        "import model as M\n"
        "m = M.VelocityNetwork(n_channels=62, d_model=62)\n"
        "print(m.num_params())\n"
        "print('\\n'.join(m.state_dict().keys()))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    lines = out.stdout.strip().splitlines()
    n_v5 = int(lines[0])
    v5_keys = set(lines[1:])
    assert n_v5 == EXPECTED_V5_PARAMS, f"v5 model.py 参数量 {n_v5} != {EXPECTED_V5_PARAMS}（v5 侧漂移？）"
    # v5-fmt rms 分支（同结构）：参数量 + state_dict 键集与 v5 逐位一致（复制零结构改动证据）
    m_rms = M.VelocityNetwork(n_channels=62, d_model=62, norm_type="rmsnorm")
    assert m_rms.num_params() == EXPECTED_V5_PARAMS, \
        f"v5-fmt rms 分支参数量 {m_rms.num_params()} != v5 {EXPECTED_V5_PARAMS}（复制引入结构差异）"
    assert set(m_rms.state_dict().keys()) == v5_keys, \
        "v5-fmt rms 分支 state_dict 键集与 v5 不一致（复制引入结构差异）"
    # fmt 增量 = 恰 5 个 norm bias 键（4 pre-norm + head_norm）
    fmt_keys = set(model.state_dict().keys())
    extra = fmt_keys - v5_keys
    assert extra == {f"blocks.{i}.norm.bias" for i in range(4)} | {"head_norm.bias"}, \
        f"fmt 键增量应恰 5 个 norm bias，got {sorted(extra)}"
    assert n - n_v5 == EXPECTED_NORM_INCREMENT == 5 * dm, \
        f"增量 {n - n_v5} != {EXPECTED_NORM_INCREMENT}（5×62 = fmt 5 norm 位 layernorm bias）"
    print(f"[参数量] v5-fmt {n} = v5 {n_v5} + {EXPECTED_NORM_INCREMENT}（5 norm 位 ×62 bias）；"
          f"明细：seg 2 + in_proj {in_proj} + blocks {blocks} + graph {graph} + spatial {spatial} + "
          f"lns {lns} + head_norm {head_norm} + head {head}；rms 分支键集与 v5 逐位一致"
          f"（复制零结构改动），fmt 增量键恰 5 个 norm bias")


# ------------------------------------------------- ⑥ config↔model 一致 + fmt 口径字段 + hash 稳定
def test_config_model_consistency() -> None:
    cfg = C.Config()
    assert cfg.d_model == 62 == cfg.n_channels, f"轴分解不变量破坏：d_model {cfg.d_model} / C {cfg.n_channels}"
    assert cfg.variant_name == "gpfm-mamba-v5-fmt"
    # fmt 口径字段（v1-fmt 先例逐字段）
    assert cfg.data_root == FMT_DATA_ROOT, f"data_root 应 fmt 数据根，got {cfg.data_root}"
    assert cfg.fs == 250.0, f"fs 应 250.0，got {cfg.fs}"
    assert cfg.norm_type == "layernorm", f"norm_type 应 layernorm（fmt D4），got {cfg.norm_type}"
    assert cfg.period_p == 25, f"period_p 应 25（fmt D3），got {cfg.period_p}"
    assert cfg.gen_lowpass_hz is None, "gen_lowpass_hz 应 None（fmt D2 无低通）"
    assert cfg.n_val_subjects == 4, "n_val_subjects 应 4（fmt 34/4/10 口径）"
    assert cfg.split_manifest_override == str(_V1FMT / "splits" / "fold0_status_ok_seed42_val4.json"), \
        "split_manifest_override 须只读复用 v1-fmt 固化 manifest"
    assert Path(cfg.split_manifest_override).parent == _V1FMT / "splits", "override 须落 v1-fmt splits/（只读）"
    assert cfg.outputs_root == FMT_OUTPUTS_ROOT, "outputs_root 与 v5 同根（PRD R1 并列口径）"
    # graph 七字段与 d_model 不变（与 v5 逐字段一致）
    assert cfg.graph_kernel == "ricker" and cfg.graph_gamma == 0.1 and cfg.graph_gamma_m == 1.0 \
        and cfg.graph_w_init == "" and cfg.graph_lambda_uniform == 0.0 and cfg.graph_lambda_sparse == 0.0, \
        "graph 字段口径漂移（ricker/γ=0.1/γ_m=1.0/零初始化 W/λ 关闭）"
    assert cfg.geometry_path == "/root/autodl-tmp/edra_processed_fmt/geometry_edra_fmt.pt"
    assert (cfg.n_channels, cfg.past_len, cfg.future_len, cfg.stride) == (62, 384, 128, 64)
    assert (cfg.n_layers, cfg.d_state, cfg.d_conv, cfg.expand, cfg.time_emb_dim) == (4, 16, 4, 2, 16)
    assert (cfg.batch_size, cfg.epochs, cfg.lr, cfg.seed, cfg.ema_decay) == (256, 100, 1e-3, 1234, 0.999)
    assert not hasattr(cfg, "lr_schedule"), \
        "v5-fmt 不得含 lr_schedule（v5 家族默认 cosine；v1-fmt 侧 constant 不移植）"
    assert getattr(cfg, "early_stop_patience", None) == 10, \
        "early_stop_patience 须 == 10（2026-10-20 用户拍板「cosine+早停10开启」后启用；" \
        "v1 train.py getattr 透传，< 字段即禁用）"
    # config↔model 双处一致（graph_* + d_model；类默认与 cfg 字段）
    sig = inspect.signature(M.VelocityNetwork.__init__)
    for name in ("d_model", "graph_kernel", "graph_gamma", "graph_gamma_m",
                 "graph_w_init", "geometry_path"):
        assert sig.parameters[name].default == getattr(cfg, name), \
            f"model 类默认 {name} != cfg.{name}（双处一致性断裂）"
    # hash 稳定 + pin + 敏感性
    h1, h2 = C.Config().config_hash(), C.build_config().config_hash()
    assert h1 == h2, "config_hash 不稳定（同配置两次构建不一致）"
    assert h1 == V5FMT_HASH_DEFAULT, \
        f"v5-fmt config_hash {h1} != 登记值 {V5FMT_HASH_DEFAULT}（口径变更后须显式重算登记）"
    assert C.build_config(epochs=1).config_hash() == V5FMT_HASH_EPOCHS1, "epochs=1 覆写 hash != pin"
    for field, val in (("graph_gamma", 0.5), ("graph_kernel", "dog"), ("d_model", 16),
                       ("norm_type", "rmsnorm"), ("period_p", 50), ("fs", 500.0),
                       ("data_root", "/root/autodl-tmp/edra_process_v4"),
                       ("gen_lowpass_hz", 40.0), ("n_val_subjects", 2)):
        h_mut = dataclasses.replace(cfg, **{field: val}).config_hash()
        assert h_mut != h1, f"config_hash 对 {field} 不敏感（防线失效）"
    # d_model == n_channels 断言正反例（要求③机器可读防线）
    with pytest.raises(ValueError, match="d_model"):
        M.VelocityNetwork(n_channels=62, d_model=16)  # 反例：模型维 != 通道轴
    M.VelocityNetwork(n_channels=62, d_model=62)      # 正例：62==62 构建通过（构建即抛/过，不经前向）
    # v1/v2/v5/v1-fmt config_hash 零漂移（子进程隔离导入，基准 = 已登记 pin）
    for path, base, name in ((str(_V1), V1_HASH_DEFAULT, "v1"),
                             (str(_V2), V2_HASH_DEFAULT, "v2"),
                             (str(_V5), V5_HASH_DEFAULT, "v5"),
                             (str(_V1FMT), V1FMT_HASH_DEFAULT, "v1-fmt")):
        code = (f"import sys; sys.path.insert(0, {path!r}); import config as C; "
                f"print(C.Config().config_hash())")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        got = out.stdout.strip()
        assert got == base, f"{name} config_hash 漂移：{got} != 基准 {base}（v1/v2/v5/v1-fmt 须零改动）"
    print(f"[config↔model] d_model/graph_* 六字段双处一致；fmt 口径字段 9 项全命中；"
          f"config_hash {h1} 稳定（epochs=1 {V5FMT_HASH_EPOCHS1}）且对 9 字段敏感；d_model==C 断言"
          f"正反例；v1 {V1_HASH_DEFAULT} / v2 {V2_HASH_DEFAULT} / v5 {V5_HASH_DEFAULT} / "
          f"v1-fmt {V1FMT_HASH_DEFAULT} 零漂移")


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


# ------------------------------------------- ⑧ 双遮蔽 + 前缀守卫 + 钩子（fmt 口径 norm_type）
def test_masking_and_guard() -> None:
    def load_wrapper(mod_name: str, filename: str):
        spec = importlib.util.spec_from_file_location(mod_name, str(_V5F / filename))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    tr = load_wrapper("v5fmt_train_wrapper_under_test", "train.py")
    # 双遮蔽正例：config/model/graph 均命中 v5-fmt 目录，mamba 命中 v1
    tr._shadow_v5_fmt()
    import config as _C
    import model as _M
    assert Path(_C.__file__).resolve().parent == _V5F, f"config 遮蔽未命中 v5-fmt：{_C.__file__}"
    assert Path(_M.__file__).resolve().parent == _V5F, f"model 遮蔽未命中 v5-fmt：{_M.__file__}"
    assert _M.VelocityNetwork is M.VelocityNetwork, "遮蔽后的 model 与测试直接导入的不一致"
    import graph as _Gg
    assert Path(_Gg.__file__).resolve().parent == _V5F and _Gg.MexHatAdjacency is G.MexHatAdjacency, \
        f"graph 解析未命中 v5-fmt：{_Gg.__file__}"
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

    # 前缀守卫（train wrapper）：正例 + 反例 + 缺省自动派生（前缀 v5- 沿用，PRD R1）
    tr._guard_run_id(["prog", "--run-id", "v5-fmt-smoke-a"])                            # 正例
    tr._guard_run_id(["prog", "--run-id", "v5-fmt-01"])                                # 主 Run 名正例
    argv_resume_ok = ["prog", "--resume", "/x/v5-fmt-bench-1ep/checkpoints/gpfm-mamba-v5-fmt-last.pt"]
    tr._guard_run_id(argv_resume_ok)                                                   # resume 派生正例
    for bad in (["prog", "--run-id", "v2d62-xxx"],
                ["prog", "--run-id", "v4bg-xxx"],
                ["prog", "--run-id", "v4afmt-main-01"],
                ["prog", "--run-id", "main-01"],
                ["prog", "--resume", "/x/v2d62-main-01/checkpoints/last.pt"]):
        try:
            tr._guard_run_id(bad)
            raise AssertionError(f"前缀守卫未拦截：{bad}")
        except SystemExit:
            pass
    argv_auto = ["prog"]                                                               # 缺省自动派生 v5- 前缀
    tr._guard_run_id(argv_auto)
    assert argv_auto[-2] == "--run-id" and argv_auto[-1].startswith("v5-"), argv_auto

    # evaluate wrapper 守卫：缺 --run-id / 错前缀退出，正例通过
    ev = load_wrapper("v5fmt_evaluate_wrapper_under_test", "evaluate.py")
    ev._guard_run_id(["prog", "--run-id", "v5-fmt-01"])
    for bad in (["prog"], ["prog", "--run-id", "main-01"], ["prog", "--run-id", "v2d62-main-01"],
                ["prog", "--run-id", "v4afmt-main-01"]):
        try:
            ev._guard_run_id(bad)
            raise AssertionError(f"evaluate 守卫未拦截：{bad}")
        except SystemExit:
            pass

    # 钩子（fmt 口径）：norm_type="layernorm" 接受（build_model 已证 fmt 构建）+ "rmsnorm" 接受
    # （默认分支 = v5 结构）+ 其余非法值报错；bottleneck_rank/cond_adaln 非默认报错
    M.VelocityNetwork(n_channels=62, d_model=62, norm_type="layernorm")   # fmt 口径接受
    M.VelocityNetwork(n_channels=62, d_model=62, norm_type="rmsnorm")     # 默认分支接受
    for kwargs, field in ((dict(norm_type="bogus"), "norm_type"),
                          (dict(norm_type="rms-norm"), "norm_type"),
                          (dict(bottleneck_rank=16), "bottleneck_rank"),
                          (dict(cond_adaln="scale_shift"), "cond_adaln")):
        with pytest.raises(ValueError, match=field):
            M.VelocityNetwork(n_channels=62, d_model=62, **kwargs)
    M.VelocityNetwork(n_channels=62, d_model=62,
                      bottleneck_rank=None, norm_type="rmsnorm", cond_adaln="off")  # 默认三参显式接受
    print("[遮蔽/守卫] config+model+graph 三模块遮蔽命中 v5-fmt（mamba→v1）；graph_*/d_model 运行时"
          "断言正反例；v5- 前缀守卫 train 5 反例 + evaluate 4 反例全过（v5-fmt-01 正例；v5-main-01"
          " 前缀合法属口径内——隔离层为 checkpoint 主名，登记见 experiment.md）；钩子 fmt 口径"
          " layernorm/rmsnorm 双接受、非法值与其余钩子非默认报错")


# ------------------------------------------------- ⑨ 几何门禁（fmt 数据根）+ λ 旋钮
def test_geometry_gate_and_lambda() -> None:
    cfg = C.Config()
    # 正例：fmt 数据根全 48 例顺序全等（choke point 首扫；lru_cache 二次调用同对象）
    coords, names = G.load_geometry(cfg.geometry_path, cfg.data_root)
    assert coords.shape == (62, 3) and coords.dtype == torch.float32
    assert len(names) == 62 and isinstance(names, tuple)
    coords2, names2 = G.load_geometry(cfg.geometry_path, cfg.data_root)
    assert coords2 is coords and names2 is names, "lru_cache 未按 (geometry_path, data_root) 命中"
    n_npz = len(list(Path(cfg.data_root).glob("*/*.npz")))
    assert n_npz == 48, f"fmt 数据根 npz 例数 {n_npz} != 48（门禁口径漂移）"
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
    # 反例 2：例数不符（名序正确，仅 1 例 ≠ 48）
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
    print("[几何门禁] fmt 数据根 48/48 通道名与 geometry_edra_fmt.pt 顺序全等（lru_cache 生效）；"
          "名序不一致/例数不符（1≠48）反例均报错；λ=0 损失逐位不变 + λ>0 手工复算一致 + "
          "uniformity/sparsity 与源数值一致（rtol=1e-6）")


# ---------------------------------------------------------------- ⑩ 回归
def test_regressions() -> None:
    # v1 七测（script 式，子进程直接执行；exit 0 = 全部断言通过）
    v1_tests = ["test_bandlimit.py", "test_data.py", "test_gc_psd.py", "test_gpr.py",
                "test_losses.py", "test_model.py", "test_sample.py"]
    for f in v1_tests:
        r = subprocess.run([sys.executable, str(_V1 / "tests" / f)],
                           capture_output=True, text=True)
        assert r.returncode == 0, f"v1 {f} 回归失败：\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}"
    # v4 test_v4b / v4a test_v4a / v5 test_v5（结构源回归）
    for path, name in ((str(_V4 / "tests" / "test_v4b.py"), "v4 test_v4b"),
                       (str(_V4A / "tests" / "test_v4a.py"), "v4a test_v4a"),
                       (str(_V5 / "tests" / "test_v5.py"), "v5 test_v5")):
        r = subprocess.run([sys.executable, path], capture_output=True, text=True)
        assert r.returncode == 0, f"{name} 回归失败：\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}"
    # config_hash 零漂移 pin 复验（与 ⑥ 同款子进程口径，此处为 ⑩ 的独立断言）
    for path, base, name in ((str(_V1), V1_HASH_DEFAULT, "v1"),
                             (str(_V2), V2_HASH_DEFAULT, "v2"),
                             (str(_V5), V5_HASH_DEFAULT, "v5"),
                             (str(_V1FMT), V1FMT_HASH_DEFAULT, "v1-fmt")):
        code = (f"import sys; sys.path.insert(0, {path!r}); import config as C; "
                f"print(C.Config().config_hash())")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        assert out.stdout.strip() == base, f"{name} hash 漂移：{out.stdout.strip()} != {base}"
    print(f"[回归] v1 七测 + v4 test_v4b + v4a test_v4a + v5 test_v5 全绿（exit 0）；"
          f"v1 {V1_HASH_DEFAULT} / v2 {V2_HASH_DEFAULT} / v5 {V5_HASH_DEFAULT} / "
          f"v1-fmt {V1FMT_HASH_DEFAULT} 零漂移")


# ------------------------------------------- ⑪ 划分固化只读复用（S1.2：v1-fmt manifest 34/4/10）
def test_split_fixation_readonly() -> None:
    cfg = C.Config()

    class _NoOverride:  # 模拟 v1/v2/v3/v5 config（无 split_manifest_override 字段 → 原路径，零漂移）
        fold, subject_filter, split_seed = cfg.fold, cfg.subject_filter, cfg.split_seed

    assert D.split_manifest_path(_NoOverride()) == str(_V1 / "splits" / "fold0_status_ok_seed42.json"), \
        "default-off 钩子破坏：无字段时未回到 v1 原 manifest 路径"
    assert D.split_manifest_path(cfg) == cfg.split_manifest_override, \
        "v5-fmt 应经 override 指向 v1-fmt 固化 manifest"

    # fmt 数据 48 例 → 与 v1-fmt manifest 同参数划分 → 34/4/10（校验，不重写）
    entries = D.load_manifest(cfg.data_root, subject_filter=cfg.subject_filter)
    assert len(entries) == 48, f"status_ok 过滤后应 48 例，got {len(entries)}"
    groups_count = {}
    for e in entries:
        groups_count[e["group"]] = groups_count.get(e["group"], 0) + 1
    assert groups_count == {"HC": 24, "MDD": 24}, f"分组计数异常：{groups_count}"
    train, val, test = D.split_fold(
        entries, n_folds=cfg.n_folds, fold=cfg.fold, split_seed=cfg.split_seed,
        n_val_subjects=cfg.n_val_subjects, val_subject_seed=cfg.val_subject_seed,
    )
    ids = lambda xs: sorted(e["subject_id"] for e in xs)  # noqa: E731
    assert (len(train), len(val), len(test)) == (34, 4, 10), \
        f"划分整数分配漂移：train {len(train)} / val {len(val)} / test {len(test)}（应 34/4/10）"
    vg = sorted(e["group"] for e in val)
    assert vg == ["HC", "HC", "MDD", "MDD"], f"val 分层破坏（应 2 HC + 2 MDD）：{vg}"
    assert not (set(ids(train)) & set(ids(val))) and not (set(ids(train)) & set(ids(test))) \
        and not (set(ids(val)) & set(ids(test))), "三折受试者重叠（泄漏）"
    assert len(set(ids(train)) | set(ids(val)) | set(ids(test))) == 48
    # 只读复用核验：fixate_split 幂等校验通过 + v1-fmt manifest 文件逐字节不变（零重新划分）
    manifest_bytes = Path(cfg.split_manifest_override).read_bytes()
    p1 = D.fixate_split(cfg, train, val, test)   # 校验（文件已存在 → 只读比对，不写入）
    p2 = D.fixate_split(cfg, train, val, test)   # 二次校验一致（幂等）
    assert Path(p1) == Path(cfg.split_manifest_override) and p1 == p2
    assert Path(cfg.split_manifest_override).read_bytes() == manifest_bytes, \
        "v1-fmt manifest 被改写（S1.2 只读契约破坏）"
    saved = json.loads(Path(p1).read_text(encoding="utf-8"))
    assert saved["train"] == ids(train) and saved["val"] == ids(val) and saved["test"] == ids(test)
    assert len(saved["groups"]) == 48
    print(f"[划分固化] fmt 48 例（HC24/MDD24）→ train 34 / val 4 / test 10，与 v1-fmt 固化 manifest"
          f" 逐字段一致（val={ids(val)}）；fixate 幂等校验通过且 manifest 逐字节不变（只读复用）：{p1}")


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
    test_split_fixation_readonly()   # ⑪
    print("[PASS] test_v5_fmt 十一项全部断言通过")


if __name__ == "__main__":
    main()
