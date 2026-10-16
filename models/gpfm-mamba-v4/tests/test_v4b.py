#!/usr/bin/env python3
"""test_v4b.py — v4-B（CI 骨干 + cMLP 组件式读出）单测十项（design §4.4，implement.md §1）。

① CI 隔离性：扰动输入通道 j，trunk 表征 h_{t,k}（k≠j）逐位不变；输出 v 经 head 全通道
   可变（合法跨通道影响——两段验证：trunk 不变 + 输出可变，head 混合是唯一通道间通路）
② 因果性：扰动 future 输入不改 past 位置输出；future 后段扰动不改前段；past 前缀条件生效
③ head 结构：W¹ 形状/列组 reshape 视图正确、einsum == 62 个独立子 MLP、子 MLP 梯度独立
④ 前向形状 (B,62,128) / return_all (B,62,512) / 梯度有限 + EMA swap + GPU B=256 冒烟
⑤ 参数量断言：499,456（d_ch=8, d_z=16 口径，±1%；2026-10-16 方案 B 口径变更 1→8 后重算）
⑥ config↔model 一致性：cfg.d_z/head_act/d_model 与类默认同值；config_hash 稳定 + 对 d_z 敏感；
   v1/v2/v3 config_hash 零漂移（子进程口径，基准值取自已落盘 run）；v4 hash 因口径变更登记新值
⑦ 双遮蔽命中 v4 + `v4b-` 前缀守卫正反例 + d_z/head_act 运行时断言正反例
⑧ 划分口径 32/4/10（用户 2026-10-16 拍板长期口径 7/1/2）：train 32 / val 4（分层 2HC+2MDD）/
   test 10 与 v1 固化 manifest 逐 ID 一致（保 test 可比）；v4 固化 manifest 落 v4 splits/
   （首建写入、二建校验一致）；v1 旧 manifest 未被改动（hash+mtime）；default-off 钩子正反例
⑨ 早停口径（用户 2026-10-16 拍板 patience=10）：v1 train.py 纯函数逻辑（patience<=0 恒 False
   /1/9/10 边界/恰好第 patience 轮触发）+ epoch 循环计数模拟 + resume 回放恢复 +
   v1/v2/v3 config 无字段 getattr 默认 0（禁用）断言；v4=10
⑩ lr 调度口径（用户 2026-10-16 拍板：取消 cosine 退火，lr 恒定 1e-3）：v1 train.py 纯函数
   build_lr_scheduler——constant 构建 LambdaLR 且 10 步 step 后 lr 仍精确 1e-3（含
   state_dict/load_state_dict round-trip，ckpt/resume 状态契约）；cosine 构建 T_max=epochs 且
   step 后 lr 严格下降；未知值 ValueError；v1/v2/v3 config 无字段 getattr→"cosine"（零漂移）；
   v4="constant"（config_hash 新 pin 见 ⑥：91dfd211dbc3；v1/v2/v3 hash 逐位不变亦在 ⑥ 断言）

口径变更史（d_ch，均经 config+model 双处同改 + 本测试重跑）：1（用户初指定）→ 8（2026-10-16
方案 B：Run 006/007 证 d_ch=1 trunk 容量塌缩、H3 不可判定；显存探针 d_ch=8@batch256=24.01 GiB）。

运行：python3 models/gpfm-mamba-v4/tests/test_v4b.py（需 CUDA——mamba_ssm 官方 kernel）
"""

from __future__ import annotations

import hashlib
import json
import os

# BLAS 线程上限（environment.md 惯例：有效 16 核 cgroup，须在 numpy 导入前设置；可覆盖）
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "8")

import dataclasses
import importlib.util
import inspect
import subprocess
import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent          # models/gpfm-mamba-v4/tests
_V4 = _HERE.parent                               # models/gpfm-mamba-v4
_V1 = _V4.parent / "gpfm-mamba-v1"

sys.path.insert(0, str(_V1))
sys.path.insert(0, str(_V4))

import config as C  # noqa: E402
import data as D  # noqa: E402  # v1 data.py（只读复用；split_manifest_override 钩子所在）
import model as M  # noqa: E402

EXPECTED_PARAMS = 499_456         # 2026-10-16 方案 B（d_ch=8）：2+152+4×1,304+8+494,078
V1_HASH_DEFAULT = "c6ff80ac1c43"  # 基准：v1 main-01 落盘 resolved config hash（零漂移守护）
V2_HASH_DEFAULT = "fc2a9ce14948"  # 基准：v2d62-main-01 落盘 resolved config hash
V3_HASH_DEFAULT = "0c2de30a4868"  # 基准：v3lp 落盘 resolved config hash
V4_HASH_VAL4 = "91dfd211dbc3"     # v4 默认 config hash——2026-10-16 口径变更链：251c7be664ea →
                                  # 划分 32/4/10（f088186dd386）→ 早停 patience=10（26cd2bb2e43f）
                                  # → lr 恒定（a965f24e580a）→ **d_ch 1→8 方案 B（本值）**；
                                  # epochs=1 覆写口径 02f0fb271e0f（bench 用，随 Run 008 登记）。
                                  # 后续任何口径变更（含 d_z/d_ch/划分/早停/lr）须重算登记此值


def build_model(cfg: C.Config, dev: str) -> M.VelocityNetwork:
    """按 v1 train.py 的显式 kwargs 构建（不含 d_z/head_act——类默认须独立成立，design §4.3）。"""
    return M.VelocityNetwork(
        n_channels=cfg.n_channels,
        past_len=cfg.past_len,
        future_len=cfg.future_len,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        d_state=cfg.d_state,
        d_conv=cfg.d_conv,
        expand=cfg.expand,
        time_emb_dim=cfg.time_emb_dim,
        use_fast_path=cfg.use_fast_path,
    ).to(dev)


def _inputs(cfg: C.Config, dev: str, B: int = 4, seed: int = 7):
    g = torch.Generator(device=dev).manual_seed(seed)
    xp = torch.randn(B, cfg.n_channels, cfg.past_len, device=dev, generator=g)
    xf = torch.randn(B, cfg.n_channels, cfg.future_len, device=dev, generator=g)
    t = torch.rand(B, device=dev, generator=g).clamp(0.05, 0.95)
    return xp, xf, t


def test_ci_isolation(model, cfg, xp, xf, t) -> None:
    """① CI 隔离性（两段验证）：trunk 通道 k≠j 逐位不变 + 输出经 head 可变（head 是唯一混合层）。

    直接调用 model.trunk_forward 取逐通道表征 h（模型公开暴露，无需钩子）。
    """
    j = 5
    h1 = model.trunk_forward(xp, xf, t)                    # (B, C, L, d_ch)
    xp2, xf2 = xp.clone(), xf.clone()
    xp2[:, j, :] += 1.0
    xf2[:, j, :] += 1.0
    h2 = model.trunk_forward(xp2, xf2, t)
    assert not torch.equal(h2[:, j], h1[:, j]), "扰动通道 j 未改变通道 j 自身的 trunk 表征（测试无效）"
    for k in range(cfg.n_channels):
        if k == j:
            continue
        assert torch.equal(h2[:, k], h1[:, k]), \
            f"CI 隔离性破坏：扰动通道 {j} 改变了通道 {k} 的 trunk 表征"
    # 输出侧：head 从全部 62 通道读出，通道 j 的扰动经 W¹ 列组影响各目标通道输出（合法影响）
    v1 = model(xp, xf, t)
    v2 = model(xp2, xf2, t)
    assert not torch.allclose(v2, v1, atol=1e-6), "扰动通道 j 输出完全不变（head 未生效？）"
    n_changed = sum(1 for i in range(cfg.n_channels) if not torch.allclose(v2[:, i], v1[:, i], atol=1e-6))
    print(f"[CI 隔离性] trunk：k≠{j} 全部 {cfg.n_channels - 1} 通道逐位不变；"
          f"head：{n_changed}/62 输出通道经混合改变（合法跨通道通路）")


def test_causality(model, cfg, xp, xf, t) -> None:
    """② 因果性（return_all 口径，与 v1 test_model 同款三段）。"""
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
    print("[因果性] future 扰动不改 past 输出；future 后段扰动不改前段；past 前缀条件生效")


def test_head_structure(model, cfg, xp, xf, t) -> None:
    """③ head 结构：形状/列组视图/einsum==62 子 MLP/梯度独立。"""
    Cn, d_z, d_ch = cfg.n_channels, cfg.d_z, cfg.d_model
    K = Cn * d_ch
    assert tuple(model.W1.shape) == (Cn, d_z, K), f"W1 形状 {tuple(model.W1.shape)} != {(Cn, d_z, K)}"
    assert tuple(model.b1.shape) == (Cn, d_z)
    assert tuple(model.W2.shape) == (Cn, d_z)
    assert tuple(model.b2.shape) == (Cn,)

    # 列组视图：W1.view(C, d_z, C, d_ch)[i, :, j, :] == W1[i][:, j*d_ch:(j+1)*d_ch]（源通道 j 列块）
    W1v = model.W1.view(Cn, d_z, Cn, d_ch)
    for (i, j) in ((0, 0), (7, 5), (61, 61), (30, 13)):
        assert torch.equal(W1v[i, :, j, :], model.W1[i, :, j * d_ch : (j + 1) * d_ch]), \
            f"W1 列组视图错误 (i={i}, j={j})"

    # einsum == 62 个独立子 MLP（测试内逐通道循环仅作参照，模型实现无循环）
    torch.manual_seed(11)
    g = torch.randn(3, 17, K, device=model.W1.device)
    v_fast = model._head(g)                                # (B, T, C)
    v_man = torch.empty_like(v_fast)
    for i in range(Cn):
        z_i = model.act(g @ model.W1[i].T + model.b1[i])   # (B, T, d_z)
        v_man[:, :, i] = z_i @ model.W2[i] + model.b2[i]
        # 列组口径复算：z_i = σ(Σ_j Σ_c W1v[i,:,j,c]·g_{...,j,c} + b1_i)（源通道分块求和）
        g_grp = g.view(*g.shape[:-1], Cn, d_ch)              # (B, T, C, d_ch)
        z_grp = model.act(torch.einsum("zjc,btjc->btz", W1v[i], g_grp) + model.b1[i])
        assert torch.allclose(z_grp, z_i, atol=1e-5), f"子 MLP {i} 列组分块求和与整块不一致"
    assert torch.allclose(v_fast, v_man, atol=1e-5), "_head einsum 与逐通道子 MLP 不一致"

    # forward 组合口径：trunk → _pool → _head == forward
    h = model.trunk_forward(xp, xf, t)
    v_pool = model._head(model._pool(h[:, :, cfg.past_len :]))
    assert torch.allclose(v_pool.permute(0, 2, 1), model(xp, xf, t), atol=1e-5), \
        "forward 组合（trunk→pool→head）与直接 forward 不一致"

    # 62 子 MLP 梯度独立：仅目标通道 i_t 的损失 → 其他子 MLP 参数梯度恒零
    model.zero_grad(set_to_none=True)
    out = model(xp, xf, t)
    i_t = 9
    out[:, i_t, :].sum().backward()
    assert model.W1.grad is not None
    assert model.W1.grad[i_t].abs().sum() > 0, "目标子 MLP W1 梯度为零（测试无效）"
    mask = torch.ones(Cn, dtype=torch.bool, device=model.W1.device)
    mask[i_t] = False
    assert model.W1.grad[mask].abs().sum() == 0, "W1 梯度串到其他子 MLP"
    assert model.b1.grad[mask].abs().sum() == 0, "b1 梯度串到其他子 MLP"
    assert model.W2.grad[mask].abs().sum() == 0, "W2 梯度串到其他子 MLP"
    assert model.b2.grad[mask].abs().sum() == 0, "b2 梯度串到其他子 MLP"
    print(f"[head 结构] W1 {tuple(model.W1.shape)} 列组视图正确；einsum==62 子 MLP；"
          f"子 MLP {i_t} 梯度独立（其余 61 个恒零）")


def test_forward_grad(model, cfg, xp, xf, t) -> None:
    """④ 前向形状/梯度有限/EMA + GPU 训练形状冒烟（B=256 forward+backward）。"""
    dev = xp.device
    B = xp.shape[0]
    model.eval()
    with torch.no_grad():
        v = model(xp, xf, t)
        assert v.shape == (B, cfg.n_channels, cfg.future_len), f"v 形状错误 {tuple(v.shape)}"
        assert torch.isfinite(v).all()
        out_all = model(xp, xf, t, return_all=True)
        assert out_all.shape == (B, cfg.n_channels, cfg.past_len + cfg.future_len)
        assert torch.allclose(out_all[:, :, cfg.past_len:], v, atol=1e-6), "future 切片与 return_all 不一致"
    model.train()
    model.zero_grad(set_to_none=True)
    v = model(xp, xf, t)
    (v ** 2).mean().backward()
    n_grad = 0
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), "存在非有限梯度"
            n_grad += 1
    # EMA（v1 复制类）：update/swap/restore
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
    # GPU 训练形状冒烟
    model.zero_grad(set_to_none=True)
    xpB = torch.randn(cfg.batch_size, cfg.n_channels, cfg.past_len, device=dev)
    xfB = torch.randn(cfg.batch_size, cfg.n_channels, cfg.future_len, device=dev)
    tB = torch.rand(cfg.batch_size, device=dev)
    torch.cuda.synchronize()
    t0 = time.time()
    vB = model(xpB, xfB, tB)
    (vB ** 2).mean().backward()
    torch.cuda.synchronize()
    dt = time.time() - t0
    assert vB.shape == (cfg.batch_size, cfg.n_channels, cfg.future_len)
    assert torch.isfinite(vB).all()
    print(f"[前向/梯度] v {tuple(v.shape)}；return_all {tuple(out_all.shape)}；{n_grad} 参数张量梯度有限；"
          f"EMA swap 正常；GPU 冒烟 B={cfg.batch_size} fwd+bwd {dt * 1000:.0f} ms，"
          f"peak {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB")


def test_param_count(model) -> None:
    """⑤ 参数量断言：499,456 ±1%（d_ch=8, d_z=16，2026-10-16 方案 B 口径，明细逐项核对）。"""
    n = model.num_params()
    assert abs(n - EXPECTED_PARAMS) <= EXPECTED_PARAMS * 0.01, f"参数量 {n} 偏离 {EXPECTED_PARAMS} 超过 ±1%"
    # 明细核对（逐模块）
    d = {k: p.numel() for k, p in model.named_parameters()}
    seg = d["seg_past"] + d["seg_future"]
    in_proj = d["in_proj.weight"] + d["in_proj.bias"]
    blocks = sum(v for k, v in d.items() if k.startswith("blocks."))
    head_norm = d["head_norm.weight"]
    head = d["W1"] + d["b1"] + d["W2"] + d["b2"]
    assert seg == 2 and in_proj == 152 and blocks == 4 * 1_304 and head_norm == 8 and head == 494_078, \
        f"参数明细漂移：seg {seg}（=2） in_proj {in_proj}（=152） blocks {blocks}（=5,216） " \
        f"head_norm {head_norm}（=8） head {head}（=494,078）"
    assert n == EXPECTED_PARAMS, f"参数量精确值 {n} != {EXPECTED_PARAMS}"
    print(f"[参数量] {n} == {EXPECTED_PARAMS}（明细：seg 2 + in_proj 152 + trunk 4×1,304 + "
          f"RMSNorm 8 + head 494,078；head 占 {head / n:.1%}）")


def test_config_model_consistency(cfg) -> None:
    """⑥ config↔model 一致性 + config_hash 稳定性/敏感性 + v1/v2/v3 基线 hash 零漂移 + v4 新值登记。"""
    assert cfg.d_model == 8 and cfg.d_z == 16 and cfg.head_act == "silu", \
        f"cfg 超参漂移：d_model {cfg.d_model} d_z {cfg.d_z} head_act {cfg.head_act}"
    sig = inspect.signature(M.VelocityNetwork.__init__)
    assert sig.parameters["d_z"].default == cfg.d_z, "model 类默认 d_z != cfg.d_z"
    assert sig.parameters["head_act"].default == cfg.head_act, "model 类默认 head_act != cfg.head_act"
    assert sig.parameters["d_model"].default == cfg.d_model, "model 类默认 d_model(d_ch) != cfg.d_model"
    assert cfg.variant_name == "gpfm-mamba-v4b"
    h1, h2 = C.Config().config_hash(), C.build_config().config_hash()
    assert h1 == h2, "config_hash 不稳定（同配置两次构建不一致）"
    assert h1 == V4_HASH_VAL4, \
        f"v4 config_hash {h1} != 登记值 {V4_HASH_VAL4}（口径变更后须显式重算登记 V4_HASH_VAL4）"
    h_mut = dataclasses.replace(cfg, d_z=8).config_hash()
    assert h_mut != h1, "config_hash 对 d_z 不敏感（防线失效）"
    h_split = dataclasses.replace(cfg, n_val_subjects=2).config_hash()
    assert h_split != h1, "config_hash 对 n_val_subjects 不敏感（划分口径未进 hash）"
    # v1/v2/v3 config_hash 零漂移（子进程隔离导入，基准 = 已落盘 run 的 resolved hash）
    for path, base, name in ((str(_V1), V1_HASH_DEFAULT, "v1"),
                             (str(_V1.parent / "gpfm-mamba-v2"), V2_HASH_DEFAULT, "v2"),
                             (str(_V1.parent / "gpfm-mamba-v3"), V3_HASH_DEFAULT, "v3lp")):
        code = (f"import sys; sys.path.insert(0, {path!r}); import config as C; "
                f"print(C.Config().config_hash())")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        got = out.stdout.strip()
        assert got == base, f"{name} config_hash 漂移：{got} != 基准 {base}（v1/v2/v3 须零改动）"
    print(f"[config↔model] d_model/d_z/head_act 双处一致；config_hash {h1}（口径变更后登记值）稳定"
          f"且对 d_z/n_val_subjects 敏感；v1 {V1_HASH_DEFAULT} / v2 {V2_HASH_DEFAULT} / "
          f"v3lp {V3_HASH_DEFAULT} 零漂移")


def test_split_caliber_val4() -> None:
    """⑧ 划分口径 32/4/10（用户 2026-10-16 拍板 7/1/2，长期口径）：

    - train 32 / val 4（分层 2 HC + 2 MDD）/ test 10；三折互斥、合计 46
    - test 10 例与 v1 固化 manifest 逐 ID 一致（fold0 外层不动，保 v1/v2/v4 test 可比——关键断言）
    - v4 固化 manifest 落 v4 自己的 splits/（经 v1 data.py default-off 钩子；首建写入、二建校验一致）
    - v1 旧 manifest 未被改动（内容 hash + mtime 双检）
    - default-off 钩子正反例：无 split_manifest_override 字段 → 原 v1 路径（v1/v2/v3 零漂移）
    """
    cfg = C.Config()
    # default-off 钩子：无字段 → 原 v1 路径；v4 有 override → 指向自身 splits/
    class _NoOverride:  # 模拟 v1/v2/v3 config（无 split_manifest_override 字段）
        fold, subject_filter, split_seed = cfg.fold, cfg.subject_filter, cfg.split_seed

    v1_manifest = _V1 / "splits" / "fold0_status_ok_seed42.json"
    assert D.split_manifest_path(_NoOverride()) == str(v1_manifest), \
        "default-off 钩子破坏：无字段时未回到 v1 原 manifest 路径（v1/v2/v3 零漂移守护）"
    assert D.split_manifest_path(cfg) == cfg.split_manifest_override
    assert Path(cfg.split_manifest_override).parent == _V4 / "splits", "override 未落在 v4 自己的 splits/"

    v1_before = hashlib.sha256(v1_manifest.read_bytes()).hexdigest()
    v1_mtime = v1_manifest.stat().st_mtime_ns

    entries = D.load_manifest(cfg.data_root, subject_filter=cfg.subject_filter)
    assert len(entries) == 46, f"status_ok 过滤后应 46 例，got {len(entries)}"
    train, val, test = D.split_fold(
        entries, n_folds=cfg.n_folds, fold=cfg.fold, split_seed=cfg.split_seed,
        n_val_subjects=cfg.n_val_subjects, val_subject_seed=cfg.val_subject_seed,
    )
    assert (len(train), len(val), len(test)) == (32, 4, 10), \
        f"划分计数漂移：train {len(train)} / val {len(val)} / test {len(test)}（应 32/4/10）"
    vg = sorted(e["group"] for e in val)
    assert vg == ["HC", "HC", "MDD", "MDD"], f"val 分层破坏（应 2 HC + 2 MDD）：{vg}"
    ids = lambda xs: sorted(e["subject_id"] for e in xs)  # noqa: E731
    assert not (set(ids(train)) & set(ids(val))) and not (set(ids(train)) & set(ids(test))) \
        and not (set(ids(val)) & set(ids(test))), "三折受试者重叠（泄漏）"
    assert len(ids(train) + ids(val) + ids(test)) == 46
    # 关键可比性断言：test 与 v1 固化 manifest 逐 ID 一致
    saved = json.loads(v1_manifest.read_text(encoding="utf-8"))
    assert ids(test) == sorted(saved["test"]), \
        f"test 折与 v1 固化 manifest 不一致（可比性破坏）：{ids(test)} vs {saved['test']}"
    # v4 固化 manifest：首建写入（若已存在则校验一致）+ 二建校验一致（fixate_split 幂等）
    p1 = D.fixate_split(cfg, train, val, test)
    p2 = D.fixate_split(cfg, train, val, test)
    assert Path(p1) == Path(cfg.split_manifest_override) and p1 == p2
    m4 = json.loads(Path(p1).read_text(encoding="utf-8"))
    assert m4["train"] == ids(train) and m4["val"] == ids(val) and m4["test"] == ids(test)
    assert all(m4["groups"][str(s)] == g for s, g in
               [(e["subject_id"], e["group"]) for e in val]), "manifest 分组记录与划分不一致"
    # v1 旧 manifest 未被改动（内容 hash + mtime 双检）
    assert hashlib.sha256(v1_manifest.read_bytes()).hexdigest() == v1_before, "v1 manifest 内容被改动"
    assert v1_manifest.stat().st_mtime_ns == v1_mtime, "v1 manifest mtime 被改动"
    print(f"[划分口径] train {len(train)} / val {len(val)} / test {len(test)}（32/4/10，7/1/2 长期口径）；"
          f"val={ids(val)} 分层 {vg}；test 与 v1 manifest 逐 ID 一致（fold0 外层不动）；"
          f"v4 manifest {_V4.name}/splits/{Path(p1).name} 首建+二建一致；"
          f"v1 manifest hash+mtime 未动；default-off 钩子正反例通过")


def test_masking_and_guard() -> None:
    """⑦ 双遮蔽命中 v4 + v4b- 前缀守卫正反例 + d_z/head_act 运行时断言正反例。"""

    def load_wrapper(mod_name: str, filename: str):
        spec = importlib.util.spec_from_file_location(mod_name, str(_V4 / filename))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    tr = load_wrapper("v4b_train_wrapper_under_test", "train.py")
    # 双遮蔽正例：config/model 均命中 v4 目录，mamba 命中 v1
    tr._shadow_v4()
    import config as _C
    import model as _M
    assert Path(_C.__file__).resolve().parent == _V4, f"config 遮蔽未命中 v4：{_C.__file__}"
    assert Path(_M.__file__).resolve().parent == _V4, f"model 遮蔽未命中 v4：{_M.__file__}"
    assert _M.VelocityNetwork is M.VelocityNetwork, "遮蔽后的 model 与测试直接导入的不一致"
    import mamba as _mb
    assert Path(_mb.__file__).resolve().parent == _V1, f"mamba 应解析到 v1：{_mb.__file__}"

    # d_z/head_act 运行时断言：正例（真实 cfg）通过；反例（shim config d_z=8）SystemExit
    tr._assert_model_config_consistency()

    class _BadCfg:
        d_z, head_act, d_model = 8, "silu", 8

    class _BadConfMod:
        Config = staticmethod(lambda: _BadCfg())

    real_conf = sys.modules["config"]
    try:
        sys.modules["config"] = _BadConfMod()
        try:
            tr._assert_model_config_consistency()
            raise AssertionError("d_z 漂移未被运行时断言拦截")
        except SystemExit as e:
            assert "d_z" in str(e), f"退出信息未指明 d_z：{e}"
    finally:
        sys.modules["config"] = real_conf

    # 前缀守卫（train wrapper）：正例 + 三反例
    tr._guard_run_id(["prog", "--run-id", "v4b-smoke-a"])                       # 正例
    argv_resume_ok = ["prog", "--resume", "/x/v4b-bench-1ep/checkpoints/gpfm-mamba-v4b-last.pt"]
    tr._guard_run_id(argv_resume_ok)                                            # resume 派生正例
    for bad in (["prog", "--run-id", "v2d62-xxx"],
                ["prog", "--run-id", "main-01"],
                ["prog", "--resume", "/x/v2d62-main-01/checkpoints/last.pt"]):
        try:
            tr._guard_run_id(bad)
            raise AssertionError(f"前缀守卫未拦截：{bad}")
        except SystemExit:
            pass
    argv_auto = ["prog"]                                                        # 缺省自动派生 v4b- 前缀
    tr._guard_run_id(argv_auto)
    assert argv_auto[-2] == "--run-id" and argv_auto[-1].startswith("v4b-"), argv_auto

    # evaluate wrapper 守卫：缺 --run-id / 错前缀退出，正例通过
    ev = load_wrapper("v4b_evaluate_wrapper_under_test", "evaluate.py")
    ev._guard_run_id(["prog", "--run-id", "v4b-main-01"])
    for bad in (["prog"], ["prog", "--run-id", "main-01"]):
        try:
            ev._guard_run_id(bad)
            raise AssertionError(f"evaluate 守卫未拦截：{bad}")
        except SystemExit:
            pass
    print("[遮蔽/守卫] config+model 双遮蔽命中 v4（mamba→v1）；d_z 运行时断言正反例；"
          "v4b- 前缀守卫 train 4 例 + evaluate 3 例全过")


def test_early_stop_hook() -> None:
    """⑨ 早停口径（用户 2026-10-16 拍板 patience=10，prd R3/design §5）：

    - v1 train.py 纯函数 should_early_stop：patience<=0 恒 False（禁用语义）；1/9/10 边界；
      恰好第 patience 轮无改善触发
    - epoch 循环计数模拟（与 train.py 每 epoch 末「新 best 清零否则 +1→判定」同口径）：
      patience=10 恰在连续第 10 轮无改善的 epoch 末触发；patience=0（v1/v2/v3 语义）100 轮恒不触发
    - replay_epochs_since_best（resume 兼容）：metrics.jsonl 回放计数 == 循环计数连续
    - v1/v2/v3 config 无 early_stop_patience 字段 → getattr 默认 0（禁用，子进程口径）；
      v4 cfg.early_stop_patience == 10
    """
    # 载入 v1 train.py（importlib 独立模块名；其 import config/model 命中 v4 缓存——纯函数与 cfg 无关）
    spec = importlib.util.spec_from_file_location("gpfm_mamba_v1_train_es_test", str(_V1 / "train.py"))
    tr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tr)
    f = tr.should_early_stop

    # patience<=0 恒 False（禁用——v1/v2/v3 config 无字段 → getattr 默认 0）
    for p in (0, -1, -10):
        for es in (0, 1, 5, 10, 99, 100):
            assert f(es, p) is False, f"patience {p} 须恒不触发（es={es}）"
    # 边界：恰在第 patience 轮无改善触发（es 达到 patience），差一轮不触发
    assert f(0, 1) is False and f(1, 1) is True
    assert f(8, 9) is False and f(9, 9) is True
    assert f(9, 10) is False and f(10, 10) is True and f(11, 10) is True
    # patience=10（v4 拍板口径）逐轮扫描
    assert [f(es, 10) for es in range(12)] == [False] * 10 + [True, True]

    # epoch 循环计数模拟（同 train.py 每 epoch 末三行钩子的口径）
    def _sim(best_flags, patience):
        es, fired_at = 0, None
        for e, b in enumerate(best_flags):
            es = 0 if b else es + 1
            if f(es, patience):
                fired_at = e
                break
        return fired_at, es

    flags = [True] * 6 + [False] * 20     # 最近 best 在 epoch 5（0 基），此后连续无改善
    fired_at, es = _sim(flags, 10)
    assert fired_at == 15 and es == 10, "patience=10 应恰在 best 后第 10 轮（epoch 15 末）触发"
    fired_at, es = _sim([False] * 20, 10)
    assert fired_at == 9 and es == 10, "全程无 best（fresh start 不发生——best_val=inf 首轮必 best）仅口径参照"
    fired_at, es = _sim([True] * 100, 10)
    assert fired_at is None, "每轮新 best 恒不触发"
    fired_at, es = _sim([False] * 100, 0)  # v1/v2/v3 语义（patience=0）：100 轮无 best 恒不触发
    assert fired_at is None and es == 100, "patience=0（v1/v2/v3）不得改变固定 100 epoch 行为"
    # 中途穿插 best 打断计数
    fired_at, _ = _sim([True, False, False, True, False, False, False], 3)
    assert fired_at == 6, "计数应自最近一次 best 重新累计（epoch 3 best → epoch 6 末触发）"

    # replay_epochs_since_best（resume 回放）== 循环计数连续
    for flags2, want in (([True, False, False, True, False, False], 2),
                         ([True, True], 0),
                         ([True] + [False] * 9, 9),
                         ([], 0)):
        lines = [json.dumps({"epoch": i, "best": b}) for i, b in enumerate(flags2)]
        assert tr.replay_epochs_since_best(lines) == want, f"回放计数错：{flags2}"
    # 空行/空白行跳过（metrics.jsonl 允许尾随空行）；极旧行缺 "best" 键按非 best 计
    assert tr.replay_epochs_since_best(["", "   "]) == 0
    assert tr.replay_epochs_since_best([json.dumps({"epoch": 0}), json.dumps({"epoch": 1, "best": True})]) == 0
    assert tr.replay_epochs_since_best([json.dumps({"epoch": 0, "best": True}), json.dumps({"epoch": 1})]) == 1
    # 截齐语义（resume 块先截到断点 epoch 再回放）——断点在 epoch 3：只回放前 4 行 → 计数 0
    lines = [json.dumps({"epoch": i, "best": b}) for i, b in enumerate(flags)]
    assert tr.replay_epochs_since_best(lines[:4]) == 0, "截齐后回放应只统计断点前行"

    # v1/v2/v3 config 无字段 → getattr 默认 0（子进程隔离导入）；v4 = 10
    for path, name in ((str(_V1), "v1"),
                       (str(_V1.parent / "gpfm-mamba-v2"), "v2"),
                       (str(_V1.parent / "gpfm-mamba-v3"), "v3lp")):
        code = (f"import sys; sys.path.insert(0, {path!r}); import config as C; "
                f"print(getattr(C.Config(), 'early_stop_patience', 0))")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        assert out.stdout.strip() == "0", \
            f"{name} config 应无 early_stop_patience 字段（getattr 默认 0=禁用，零漂移守护）"
    v4_cfg = C.Config()
    assert v4_cfg.early_stop_patience == 10, f"v4 early_stop_patience 应为 10，got {v4_cfg.early_stop_patience}"
    print(f"[早停] should_early_stop 边界（0/负恒 False；恰第 patience 轮触发）+ 循环计数模拟"
          f"（patience=10 epoch15 末触发 / patience=0 百轮不触发 / best 打断重累计）+ 回放恢复 + "
          f"v1/v2/v3 无字段 getattr→0、v4=10 全过")


def test_lr_schedule_hook() -> None:
    """⑩ lr 调度口径（用户 2026-10-16 拍板：取消 cosine 退火，lr 恒定 1e-3，prd R3/design §5）：

    - v1 train.py 纯函数 build_lr_scheduler："constant" → LambdaLR 恒 1.0，10 步 step 后 lr 仍
      精确 = cfg.lr（get_last_lr 与 param_groups 双检）；LambdaLR state_dict/load_state_dict
      round-trip（checkpoint 存 scheduler 状态的既有契约，--resume 全量恢复不变）
    - cosine 口径：无字段 shim（v1/v2/v3 语义）→ getattr 默认 "cosine" → CosineAnnealingLR
      T_max=epochs，step 后 lr 严格下降（与原内联构建同构，零漂移）
    - 未知 lr_schedule 值 → ValueError（拼写错误构建期暴露）
    - v1/v2/v3 config 无字段 → getattr 默认 "cosine"（子进程口径）；v4 = "constant"、lr=1e-3
      （v4 config_hash 新 pin 与 v1/v2/v3 hash 零漂移断言在 ⑥）
    """
    # 载入 v1 train.py（importlib 独立模块名；其 import config/model 命中 v4 缓存——纯函数与 cfg 无关）
    spec = importlib.util.spec_from_file_location("gpfm_mamba_v1_train_lrs_test", str(_V1 / "train.py"))
    tr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tr)
    cfg = C.Config()
    param = torch.nn.Parameter(torch.zeros(1))

    def _opt():
        return torch.optim.AdamW([param], lr=cfg.lr, weight_decay=cfg.weight_decay)

    # constant（v4 拍板口径）：LambdaLR 恒 1.0，10 步 step 后 lr 仍精确 1e-3
    assert cfg.lr_schedule == "constant" and cfg.lr == 1e-3, \
        f"v4 lr 口径漂移：lr_schedule {cfg.lr_schedule!r} lr {cfg.lr}"
    sched_c = tr.build_lr_scheduler(_opt(), cfg)
    assert isinstance(sched_c, torch.optim.lr_scheduler.LambdaLR)
    for _ in range(10):
        sched_c.step()
    assert sched_c.get_last_lr()[0] == cfg.lr, \
        f"constant 口径 lr 漂移：get_last_lr {sched_c.get_last_lr()[0]} != {cfg.lr}"
    assert sched_c.optimizer.param_groups[0]["lr"] == cfg.lr, "param_groups lr 与 get_last_lr 不一致"

    # state_dict/load_state_dict round-trip（resume 契约；普通函数 lambda 的 lr_lambdas 存 None）
    sd = sched_c.state_dict()
    sched_r = tr.build_lr_scheduler(_opt(), cfg)
    sched_r.load_state_dict(sd)
    sched_r.step()
    assert sched_r.get_last_lr()[0] == cfg.lr, "round-trip 后 constant lr 漂移"

    # cosine（v1/v2 语义）：无字段 shim → 默认 cosine；T_max=epochs；step 后严格下降
    class _NoSched:  # 模拟 v1/v2/v3 config（无 lr_schedule 字段）
        epochs, lr, weight_decay = cfg.epochs, cfg.lr, cfg.weight_decay

    sched_k = tr.build_lr_scheduler(_opt(), _NoSched())
    assert isinstance(sched_k, torch.optim.lr_scheduler.CosineAnnealingLR)
    assert sched_k.T_max == cfg.epochs, f"cosine T_max {sched_k.T_max} != epochs {cfg.epochs}"
    lr0 = sched_k.get_last_lr()[0]
    for _ in range(10):
        sched_k.step()
    lr10 = sched_k.get_last_lr()[0]
    assert lr0 == cfg.lr and lr10 < lr0, f"cosine 口径 lr 未按 T_max=epochs 下降：{lr0} -> {lr10}"

    # 未知值 → ValueError
    class _BadSched:
        epochs, lr, weight_decay = cfg.epochs, cfg.lr, cfg.weight_decay
        lr_schedule = "linear"

    try:
        tr.build_lr_scheduler(_opt(), _BadSched())
        raise AssertionError("未知 lr_schedule 未被拦截")
    except ValueError as e:
        assert "lr_schedule" in str(e), f"报错未指明 lr_schedule：{e}"

    # v1/v2/v3 config 无字段 → getattr 默认 "cosine"（子进程隔离导入，零漂移守护）
    for path, name in ((str(_V1), "v1"),
                       (str(_V1.parent / "gpfm-mamba-v2"), "v2"),
                       (str(_V1.parent / "gpfm-mamba-v3"), "v3lp")):
        code = (f"import sys; sys.path.insert(0, {path!r}); import config as C; "
                f"print(getattr(C.Config(), 'lr_schedule', 'cosine'))")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        assert out.stdout.strip() == "cosine", \
            f"{name} config 应无 lr_schedule 字段（getattr 默认 cosine，零漂移守护）"
    print(f"[lr 调度] constant：LambdaLR 10 步后 lr 仍精确 {cfg.lr}（round-trip 同）；"
          f"cosine：T_max={cfg.epochs}，10 步 {lr0:.2e}->{lr10:.2e} 严格下降；未知值 ValueError；"
          f"v1/v2/v3 无字段 getattr→cosine、v4=constant 全过")


def main() -> None:
    assert torch.cuda.is_available(), "mamba_ssm 官方 kernel 需要 CUDA"
    cfg = C.Config()
    dev = "cuda"
    torch.manual_seed(cfg.seed)
    # 用 v1 train.py 的显式 kwargs 构建（不传 d_z/head_act）——运行时契约四件套口径
    model = build_model(cfg, dev)
    xp, xf, t = _inputs(cfg, dev)

    test_ci_isolation(model, cfg, xp, xf, t)          # ①
    test_causality(model, cfg, xp, xf, t)             # ②
    test_head_structure(model, cfg, xp, xf, t)        # ③
    test_forward_grad(model, cfg, xp, xf, t)          # ④
    test_param_count(model)                           # ⑤
    test_config_model_consistency(cfg)                # ⑥
    test_masking_and_guard()                          # ⑦
    test_split_caliber_val4()                         # ⑧（32/4/10 划分口径，2026-10-16 用户拍板）
    test_early_stop_hook()                            # ⑨（早停 patience=10，2026-10-16 用户拍板）
    test_lr_schedule_hook()                           # ⑩（lr 恒定取消 cosine，2026-10-16 用户拍板）
    print("[PASS] test_v4b 十项全部断言通过")


if __name__ == "__main__":
    main()
