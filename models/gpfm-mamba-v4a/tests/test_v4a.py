#!/usr/bin/env python3
"""test_v4a.py — v4-A（低秩输入瓶颈 r=16，v2 基础单变量）单测八项（design §4.4，implement.md §1）。

① None 分支等价：v1 model.py 钩子 default-off——与改动前代码（git ref 0c8d0ee 的 v1 model.py，
   同进程 importlib 加载）同 seed 构建后前向逐位一致 + state_dict 键集与逐值不变 + 不传 kwargs
   （默认 None）同构；显式 r 分支输出不同于 None 分支（测试有效性）
② r=16 两层结构：in_proj = Sequential(Linear(79→16), Linear(16→62)) 纯线性无激活；键
   in_proj.{0,1}.{weight,bias}；形状断言；Sequential 调用 == 手工复合；复合矩阵 W1@W0 的
   rank ≤ 16 构造断言（rank≤r 线性映射——接口强制跨通道混合的数学实质）
③ 前向 (B,62,128)/return_all (B,62,512)/因果性（三段，v1 test_model 同款）/梯度有限/EMA swap/
   GPU B=256 fwd+bwd 冒烟
④ 参数量断言：130,056（= v2 None 分支 132,682 − 2626；0.9802× v2，容量不变口径）精确值 +
   明细逐项（seg 2 + in_proj 2334 + trunk 123,752 + head_norm 62 + head 3,906）
⑤ config_hash：v1/v2/v3 三变体零漂移（子进程口径，基准取自已落盘 run）+ v4a 新 hash pin
   （默认 r16=2a456201dc90、--rank 8=a1960979c695、--rank 32=f949f6c63531、epochs=1 bench
   覆写=eacb763a176a）+ 敏感性（rank/划分进 hash）+ v1/v2/v3 无 bottleneck_rank 字段
   getattr→None（透传零漂移）
⑥ v1 既有七测逐一回归 PASS（子进程：test_model/test_sample/test_gpr/test_losses/test_data/
   test_gc_psd/test_bandlimit——default-off 钩子对 v1/v2/v3 行为零漂移的行为级守护）
⑦ 单遮蔽命中 v4a（config→v4a、model→v1——与 v4b 双遮蔽的差异点）+ `v4a-` 前缀守卫正反例 +
   --rank 正反例（choices {8,16,32}、argv 剥离、build_config 实例改写、rXX 标识校验、
   resume 派生 run 名同检）
⑧ val4 划分（32/4/10，用户 2026-10-16 拍板 7/1/2 长期口径）：train 32 / val 4（分层 2HC+2MDD
   = [25,37HC,40,49MDD]）/ test 10 逐 ID 同 v1 固化 manifest；v4a 自备 manifest 首建写入 +
   二建校验一致 + **内容与 v4b 的 val4 manifest 逐字节一致**；v1 与 v4b manifest 均未被改动
   （hash+mtime 双检）；default-off 钩子正反例

运行：python3 models/gpfm-mamba-v4a/tests/test_v4a.py（需 CUDA——mamba_ssm 官方 kernel；
⑥ 含 v1 七测子进程回归，全程约数分钟）
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
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent          # models/gpfm-mamba-v4a/tests
_V4A = _HERE.parent                               # models/gpfm-mamba-v4a
_V1 = _V4A.parent / "gpfm-mamba-v1"
_V4B = _V4A.parent / "gpfm-mamba-v4"

sys.path.insert(0, str(_V1))
sys.path.insert(0, str(_V4A))

import config as C  # noqa: E402
import data as D  # noqa: E402  # v1 data.py（只读复用；split_manifest_override 钩子所在）
import model as M  # noqa: E402  # v1 model.py（bottleneck_rank 钩子所在——v4a 单遮蔽）

# ---- 登记值（实测精确数 / hash pin；后续任何口径变更须重算登记）----
EXPECTED_PARAMS = 130_056          # seg 2 + in_proj 2,334 + trunk 123,752 + head_norm 62 + head 3,906
V2_NONE_PARAMS = 132_682           # v2 同构（in_proj 满秩 4,960）；差 = 2,626（4960→2334）
V1_HASH_DEFAULT = "c6ff80ac1c43"   # 基准：v1 main-01 落盘 resolved config hash（零漂移守护）
V2_HASH_DEFAULT = "fc2a9ce14948"   # 基准：v2d62-main-01 落盘 resolved config hash
V3_HASH_DEFAULT = "0c2de30a4868"   # 基准：v3lp 落盘 resolved config hash
V4A_HASH_DEFAULT = "2a456201dc90"  # v4a 默认（r=16）config hash（2026-10-16 首建）
V4A_HASH_RANK8 = "a1960979c695"    # --rank 8 覆写口径（r 扫描条件扩展）
V4A_HASH_RANK32 = "f949f6c63531"   # --rank 32 覆写口径
V4A_HASH_EPOCHS1 = "eacb763a176a"  # epochs=1 bench 覆写口径（Run 011 bench 用）
PREREF_GIT = "0c8d0ee"             # v1 model.py 改动前 commit（None 分支等价性的对照代码源）
V1_TESTS = ("test_model", "test_sample", "test_gpr", "test_losses",
            "test_data", "test_gc_psd", "test_bandlimit")


def build_model(cfg: C.Config, dev: str, rank=None) -> M.VelocityNetwork:
    """按 v1 train.py 的显式 kwargs 构建（rank=None → 不传该 kwarg，即 default-off 语义）。"""
    kw = dict(
        n_channels=cfg.n_channels, past_len=cfg.past_len, future_len=cfg.future_len,
        d_model=cfg.d_model, n_layers=cfg.n_layers, d_state=cfg.d_state,
        d_conv=cfg.d_conv, expand=cfg.expand, time_emb_dim=cfg.time_emb_dim,
        use_fast_path=cfg.use_fast_path,
    )
    if rank is not None:
        kw["bottleneck_rank"] = rank
    return M.VelocityNetwork(**kw).to(dev)


def _inputs(cfg: C.Config, dev: str, B: int = 4, seed: int = 7):
    g = torch.Generator(device=dev).manual_seed(seed)
    xp = torch.randn(B, cfg.n_channels, cfg.past_len, device=dev, generator=g)
    xf = torch.randn(B, cfg.n_channels, cfg.future_len, device=dev, generator=g)
    t = torch.tensor([0.1, 0.3, 0.5, 0.9], device=dev)
    return xp, xf, t


def _load_preref_model_class():
    """从 git ref 加载改动前的 v1 model.py（None 分支等价性的对照代码，同进程 importlib）。"""
    try:
        src = subprocess.check_output(
            ["git", "-C", str(_V1.parents[1]), "show", f"{PREREF_GIT}:models/gpfm-mamba-v1/model.py"],
            text=True)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"无法读取 git ref {PREREF_GIT} 的 v1 model.py（等价性对照源缺失）：{e}")
    f = tempfile.NamedTemporaryFile("w", suffix="_v1_model_preref.py", delete=False)
    f.write(src)
    f.close()
    spec = importlib.util.spec_from_file_location("v1_model_preref", f.name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.VelocityNetwork


def test_none_branch_equivalence(cfg, xp, xf, t) -> None:
    """① None 分支等价：vs 改动前代码前向逐位一致 + state_dict 键集/逐值不变；显式 None 与不传同构。"""
    PreVN = _load_preref_model_class()
    kw = dict(
        n_channels=cfg.n_channels, past_len=cfg.past_len, future_len=cfg.future_len,
        d_model=cfg.d_model, n_layers=cfg.n_layers, d_state=cfg.d_state,
        d_conv=cfg.d_conv, expand=cfg.expand, time_emb_dim=cfg.time_emb_dim,
        use_fast_path=cfg.use_fast_path,
    )
    dev = xp.device
    torch.manual_seed(cfg.seed); m_pre = PreVN(**kw).to(dev)                 # 改动前（无该参数）
    torch.manual_seed(cfg.seed); m_none = M.VelocityNetwork(**{**kw, "bottleneck_rank": None}).to(dev)
    torch.manual_seed(cfg.seed); m_def = M.VelocityNetwork(**kw).to(dev)     # 不传（默认 None=v1/v2/v3 语义）
    torch.manual_seed(cfg.seed); m_r16 = M.VelocityNetwork(**{**kw, "bottleneck_rank": 16}).to(dev)
    with torch.no_grad():
        o_pre, o_none, o_def, o_r16 = m_pre(xp, xf, t), m_none(xp, xf, t), m_def(xp, xf, t), m_r16(xp, xf, t)
    assert torch.equal(o_pre, o_none), "显式 None 与改动前代码前向非逐位一致（default-off 破坏）"
    assert torch.equal(o_pre, o_def), "不传 kwargs（默认 None）与改动前代码前向非逐位一致"
    assert not torch.equal(o_pre, o_r16), "r=16 输出与 None 分支逐位相同（测试无效——r 分支未生效）"
    sd_pre, sd_none = m_pre.state_dict(), m_none.state_dict()
    assert sorted(sd_pre) == sorted(sd_none) == sorted(m_def.state_dict()), \
        f"state_dict 键集漂移：{sorted(set(sd_pre) ^ set(sd_none))}"
    assert all(torch.equal(sd_pre[k], sd_none[k]) for k in sd_pre), "state_dict 逐值漂移（初始化路径变化）"
    assert isinstance(m_none.in_proj, nn.Linear) and tuple(m_none.in_proj.weight.shape) == (62, 79), \
        "None 分支 in_proj 非 Linear(79→62) 原结构"
    print(f"[None 等价] vs git ref {PREREF_GIT} 改动前 v1 model.py：前向逐位一致（显式 None 与"
          f" 不传两口径）；state_dict 键集与逐值不变；r=16 输出确实不同（钩子生效）")


def test_rank16_structure(model, cfg, xp, xf, t) -> None:
    """② r=16 两层结构：Sequential 纯线性无激活 + 键/形状断言 + 复合 == 手工两步 + rank≤16 构造。"""
    r = cfg.bottleneck_rank
    ip = model.in_proj
    assert isinstance(ip, nn.Sequential) and len(ip) == 2, "in_proj 非 Sequential(两层)"
    assert all(isinstance(m, nn.Linear) for m in ip), \
        f"in_proj 含非 Linear 模块（违反纯线性无激活设计）：{[type(m).__name__ for m in ip]}"
    sd = model.state_dict()
    for k in ("in_proj.0.weight", "in_proj.0.bias", "in_proj.1.weight", "in_proj.1.bias"):
        assert k in sd, f"state_dict 缺键 {k}"
    assert tuple(sd["in_proj.0.weight"].shape) == (r, 79), f"in_proj.0.weight 形状 {tuple(sd['in_proj.0.weight'].shape)} != {(r, 79)}"
    assert tuple(sd["in_proj.0.bias"].shape) == (r,)
    assert tuple(sd["in_proj.1.weight"].shape) == (62, r), f"in_proj.1.weight 形状 {tuple(sd['in_proj.1.weight'].shape)} != {(62, r)}"
    assert tuple(sd["in_proj.1.bias"].shape) == (62,)
    assert not hasattr(ip, "activation") and not any(hasattr(m, "activation") for m in ip), "结构面无激活属性"
    # Sequential 调用 == 手工复合（纯线性两层）
    with torch.no_grad():
        x_in = torch.randn(3, 17, 79, device=sd["in_proj.0.weight"].device)
        manual = ip[1](ip[0](x_in))
    assert torch.allclose(ip(x_in), manual, atol=1e-6), "Sequential 前向与手工两步复合不一致"
    # rank≤r 构造断言：两层复合矩阵 W = W1 @ W0 的秩 ≤ r（默认初始化满秩 → == r）
    W = ip[1].weight @ ip[0].weight
    rank = int(torch.linalg.matrix_rank(W))
    assert tuple(W.shape) == (62, 79)
    assert rank <= r, f"复合映射秩 {rank} > r={r}（低秩构造破坏）"
    assert rank == r, f"默认初始化复合秩 {rank} != {r}（初始化异常——谱窄风险项监控）"
    # 前向输出与 None 分支同形且有限（结构可运行）
    v = model(xp, xf, t)
    assert v.shape == (xp.shape[0], 62, cfg.future_len) and torch.isfinite(v).all()
    print(f"[r=16 结构] Sequential(Linear(79→{r}), Linear({r}→62)) 纯线性无激活；键 in_proj.{{0,1}}.*；"
          f"复合 == 手工两步；复合矩阵 (62,79) rank={rank} ≤ {r}（接口强制混合构造成立）")


def test_forward_grad(model, cfg, xp, xf, t) -> None:
    """③ 前向形状/return_all/因果性/梯度有限/EMA + GPU 训练形状冒烟（B=256 forward+backward）。"""
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
        # 因果性（v1 test_model 同款三段）
        ref = out_all
        xf2 = xf.clone(); xf2[:, :, :] += 1.0
        out2 = model(xp, xf2, t, return_all=True)
        assert torch.allclose(out2[:, :, : cfg.past_len], ref[:, :, : cfg.past_len], atol=1e-5), \
            "扰动 future 输入改变了 past 位置输出（因果性破坏）"
        assert not torch.allclose(out2[:, :, cfg.past_len:], ref[:, :, cfg.past_len:], atol=1e-5), \
            "扰动 future 输入应改变 future 输出"
        xf3 = xf.clone(); xf3[:, :, 100:] += 1.0
        out3 = model(xp, xf3, t, return_all=True)
        assert torch.allclose(out3[:, :, cfg.past_len : cfg.past_len + 50],
                              ref[:, :, cfg.past_len : cfg.past_len + 50], atol=1e-5), \
            "future 后段扰动影响了前段输出（序列内因果性破坏）"
        xp4 = xp.clone(); xp4[:, :, -50:] += 1.0
        out4 = model(xp4, xf, t, return_all=True)
        assert not torch.allclose(out4[:, :, cfg.past_len:], ref[:, :, cfg.past_len:], atol=1e-5), \
            "past 扰动未影响 future 输出（前缀条件未生效）"
    model.train()
    model.zero_grad(set_to_none=True)
    v = model(xp, xf, t)
    (v ** 2).mean().backward()
    n_grad = 0
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), "存在非有限梯度"
            n_grad += 1
    # EMA（v1 复用类）：update/swap/restore
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
    # GPU 训练形状冒烟（B=256 fwd+bwd；r=16 低秩结构训练形态）
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
    print(f"[前向/梯度] v {tuple(v.shape)}；return_all {tuple(out_all.shape)}；因果三段通过；{n_grad} 参数"
          f"张量梯度有限；EMA swap 正常；GPU 冒烟 B={cfg.batch_size} fwd+bwd {dt * 1000:.0f} ms，"
          f"peak {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB")


def test_param_count(cfg) -> None:
    """④ 参数量断言：130,056 精确（= v2 None 132,682 − 2626；0.98× v2 容量不变）+ 明细逐项。"""
    dev = "cuda"
    m_r = build_model(cfg, dev, rank=cfg.bottleneck_rank)
    m_none = build_model(cfg, dev)  # rank=None（v2 同构）
    n, n2 = m_r.num_params(), m_none.num_params()
    assert n2 == V2_NONE_PARAMS, f"v2 同构（None 分支）参数量 {n2} != 登记值 {V2_NONE_PARAMS}"
    assert n == n2 - 2_626, f"r=16 参数量 {n} != None 分支 − 2626（in_proj 4960→2334）"
    assert n == EXPECTED_PARAMS, f"参数量精确值 {n} != {EXPECTED_PARAMS}"
    d = {k: p.numel() for k, p in m_r.named_parameters()}
    seg = d["seg_past"] + d["seg_future"]
    ip0 = d["in_proj.0.weight"] + d["in_proj.0.bias"]      # 79×16+16 = 1,280
    ip1 = d["in_proj.1.weight"] + d["in_proj.1.bias"]      # 16×62+62 = 1,054
    trunk = sum(v for k, v in d.items() if k.startswith("blocks."))
    head_norm = d["head_norm.weight"]
    head = d["head.weight"] + d["head.bias"]
    assert seg == 2 and ip0 == 1_280 and ip1 == 1_054 and trunk == 123_752 \
        and head_norm == 62 and head == 3_906, \
        (f"参数明细漂移：seg {seg}（=2） in_proj.0 {ip0}（=1,280） in_proj.1 {ip1}（=1,054） "
         f"trunk {trunk}（=123,752） head_norm {head_norm}（=62） head {head}（=3,906）")
    assert seg + ip0 + ip1 + trunk + head_norm + head == n
    print(f"[参数量] {n} == {EXPECTED_PARAMS}（明细：seg 2 + in_proj 2,334（1,280+1,054） + "
          f"trunk 123,752 + head_norm 62 + head 3,906）；v2 同构 {n2} − 2,626；{n / n2:.4f}× v2（容量不变）")


def test_config_hash(cfg) -> None:
    """⑤ config_hash：三变体零漂移 + v4a 四口径 pin + 敏感性 + v1/v2/v3 无 bottleneck_rank 字段。"""
    assert cfg.d_model == 62 and cfg.bottleneck_rank == 16, \
        f"cfg 结构字段漂移：d_model {cfg.d_model} bottleneck_rank {cfg.bottleneck_rank}"
    assert not hasattr(cfg, "d_z") and not hasattr(cfg, "head_act"), \
        "v4a config 不得含 d_z/head_act（v4-B 专属字段，v4-A 无 cMLP head）"
    assert cfg.variant_name == "gpfm-mamba-v4a"
    h1 = C.Config().config_hash()
    assert h1 == C.build_config().config_hash() == V4A_HASH_DEFAULT, \
        f"v4a 默认 config_hash {h1} != pin {V4A_HASH_DEFAULT}（口径变更后须显式重算登记）"
    assert dataclasses.replace(cfg, bottleneck_rank=16).config_hash() == V4A_HASH_DEFAULT, \
        "--rank 16 覆写口径应与默认逐位同 hash（同一 resolved config）"
    assert dataclasses.replace(cfg, bottleneck_rank=8).config_hash() == V4A_HASH_RANK8, \
        f"--rank 8 覆写 hash != pin {V4A_HASH_RANK8}"
    assert dataclasses.replace(cfg, bottleneck_rank=32).config_hash() == V4A_HASH_RANK32, \
        f"--rank 32 覆写 hash != pin {V4A_HASH_RANK32}"
    assert C.build_config(epochs=1).config_hash() == V4A_HASH_EPOCHS1, \
        f"epochs=1 bench 覆写 hash != pin {V4A_HASH_EPOCHS1}"
    hs = {V4A_HASH_DEFAULT, V4A_HASH_RANK8, V4A_HASH_RANK32, V4A_HASH_EPOCHS1}
    assert len(hs) == 4, "四口径 hash 出现碰撞（pin 失效）"
    assert dataclasses.replace(cfg, bottleneck_rank=8).config_hash() != h1, \
        "config_hash 对 bottleneck_rank 不敏感（r 串用防线失效）"
    assert dataclasses.replace(cfg, n_val_subjects=2).config_hash() != h1, \
        "config_hash 对 n_val_subjects 不敏感（划分口径未进 hash）"
    # v1/v2/v3 零漂移：config_hash 逐位 + 无 bottleneck_rank 字段（getattr 透传默认 None）
    for path, base, name in ((str(_V1), V1_HASH_DEFAULT, "v1"),
                             (str(_V1.parent / "gpfm-mamba-v2"), V2_HASH_DEFAULT, "v2"),
                             (str(_V1.parent / "gpfm-mamba-v3"), V3_HASH_DEFAULT, "v3lp")):
        code = (f"import sys; sys.path.insert(0, {path!r}); import config as C; c = C.Config(); "
                f"print(c.config_hash(), getattr(c, 'bottleneck_rank', None))")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        got_hash, got_field = out.stdout.split()
        assert got_hash == base, f"{name} config_hash 漂移：{got_hash} != 基准 {base}（v1/v2/v3 须零改动）"
        assert got_field == "None", f"{name} config 出现 bottleneck_rank 字段（透传零漂移破坏）"
    print(f"[config_hash] v4a 默认 {V4A_HASH_DEFAULT} / r8 {V4A_HASH_RANK8} / r32 {V4A_HASH_RANK32} / "
          f"epochs=1 {V4A_HASH_EPOCHS1}（四 pin 就位，对 rank/n_val_subjects 敏感）；"
          f"v1 {V1_HASH_DEFAULT} / v2 {V2_HASH_DEFAULT} / v3lp {V3_HASH_DEFAULT} 零漂移且无 "
          f"bottleneck_rank 字段")


def test_v1_regression() -> None:
    """⑥ v1 既有七测逐一回归 PASS（子进程；default-off 钩子行为级零漂移守护）。"""
    for name in V1_TESTS:
        path = _V1 / "tests" / f"{name}.py"
        r = subprocess.run([sys.executable, str(path)], capture_output=True, text=True)
        assert r.returncode == 0, f"v1 {name} 回归失败（exit {r.returncode}）：\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}"
        assert "PASS" in r.stdout, f"v1 {name} 未打出 PASS 终态：\n{r.stdout[-2000:]}"
        print(f"[v1 回归] {name}.py PASS")
    print(f"[v1 回归] 七测全 PASS：{'/'.join(V1_TESTS)}")


def test_masking_guard_rank() -> None:
    """⑦ 单遮蔽命中 v4a（config→v4a、model→v1）+ v4a- 前缀守卫 + --rank 正反例（rXX 标识校验）。"""

    def load_wrapper(mod_name: str, filename: str):
        spec = importlib.util.spec_from_file_location(mod_name, str(_V4A / filename))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    tr = load_wrapper("v4a_train_wrapper_under_test", "train.py")
    # 单遮蔽正例：config 命中 v4a；model/mamba 均命中 v1（v4a 目录无 model.py——低秩钩子在 v1）
    tr._shadow_v4a_config()
    import config as _C
    import model as _M
    assert Path(_C.__file__).resolve().parent == _V4A, f"config 遮蔽未命中 v4a：{_C.__file__}"
    assert Path(_M.__file__).resolve().parent == _V1, f"model 须解析到 v1（单遮蔽契约）：{_M.__file__}"
    assert _M.VelocityNetwork is M.VelocityNetwork, "遮蔽后的 model 与测试直接导入的不一致"
    import mamba as _mb
    assert Path(_mb.__file__).resolve().parent == _V1, f"mamba 应解析到 v1：{_mb.__file__}"

    # --rank 解析/剥离（v1 argparse 不识别）：两种传法等价；非法值构建期退出；未传返回 None
    argv = ["prog", "--rank", "8", "--run-id", "v4a-r8-x", "--epochs", "1"]
    assert tr._pop_rank_arg(argv) == 8 and argv == ["prog", "--run-id", "v4a-r8-x", "--epochs", "1"]
    argv2 = ["prog", "--rank=16", "--run-id", "v4a-r16-x"]
    assert tr._pop_rank_arg(argv2) == 16 and argv2 == ["prog", "--run-id", "v4a-r16-x"]
    assert tr._pop_rank_arg(["prog", "--run-id", "v4a-smoke-a"]) is None
    try:
        tr._pop_rank_arg(["prog", "--rank", "5", "--run-id", "v4a-r5-x"])
        raise AssertionError("--rank 5（非法网格值）未被拦截")
    except SystemExit:
        pass

    # 前缀守卫 + rXX 标识校验（train wrapper）：正例 + 反例（前缀错 / r 标识不匹配 / resume 派生同检）
    tr._guard_run_id(["prog", "--run-id", "v4a-smoke-a"], None)                    # 正例（无 --rank 不强制 rXX）
    tr._guard_run_id(["prog", "--run-id", "v4a-r16-bench-1ep"], 16)                # 正例
    tr._guard_run_id(["prog", "--resume", "/x/v4a-r8-main-01/checkpoints/gpfm-mamba-v4a-last.pt"], 8)
    for bad, rank in ((["prog", "--run-id", "v2d62-xxx"], None),
                      (["prog", "--run-id", "main-01"], None),
                      (["prog", "--resume", "/x/v2d62-main-01/checkpoints/last.pt"], None),
                      (["prog", "--run-id", "v4a-r16-x"], 8),                      # r 标识不匹配
                      (["prog", "--run-id", "v4a-smoke-a"], 8),                    # 缺 r8 标识
                      (["prog", "--resume", "/x/v4a-r16-main-01/checkpoints/x.pt"], 8)):
        try:
            tr._guard_run_id(bad, rank)
            raise AssertionError(f"前缀/rXX 守卫未拦截：{bad}（rank={rank}）")
        except SystemExit:
            pass
    argv_auto = ["prog"]                                                            # 缺省自动派生 v4a- 前缀
    tr._guard_run_id(argv_auto, None)
    assert argv_auto[-2] == "--run-id" and argv_auto[-1].startswith("v4a-"), argv_auto

    # --rank 覆写：build_config 实例改写 + hash 变化（事后恢复原函数，避免污染后续构建）
    orig_build = _C.build_config
    try:
        tr._apply_rank_override(8)
        cfg8 = _C.build_config()
        assert cfg8.bottleneck_rank == 8 and cfg8.config_hash() == V4A_HASH_RANK8
        cfg8e = _C.build_config(epochs=1)
        assert cfg8e.bottleneck_rank == 8 and cfg8e.epochs == 1, "--rank 与 epochs 覆写不兼容"
        try:
            _C.build_config(bottleneck_rank=16)
            raise AssertionError("bottleneck_rank 直传覆写未被拦截（r 只应经 --rank 进入）")
        except KeyError:
            pass
    finally:
        _C.build_config = orig_build
    assert C.build_config().config_hash() == V4A_HASH_DEFAULT, "恢复后默认构建漂移"

    # evaluate wrapper 守卫：缺 --run-id / 错前缀 / rXX 不匹配退出；--rank 剥离同款
    ev = load_wrapper("v4a_evaluate_wrapper_under_test", "evaluate.py")
    ev._guard_run_id(["prog", "--run-id", "v4a-r16-main-01"], None)
    ev._guard_run_id(["prog", "--run-id", "v4a-r8-main-01"], 8)
    ev_argv = ["prog", "--rank", "32", "--run-id", "v4a-r32-main-01"]
    assert ev._pop_rank_arg(ev_argv) == 32 and ev_argv == ["prog", "--run-id", "v4a-r32-main-01"]
    for bad, rank in ((["prog"], None),
                      (["prog", "--run-id", "main-01"], None),
                      (["prog", "--run-id", "v4b-main-01"], None),
                      (["prog", "--run-id", "v4a-r16-main-01"], 32)):
        try:
            ev._guard_run_id(bad, rank)
            raise AssertionError(f"evaluate 守卫未拦截：{bad}（rank={rank}）")
        except SystemExit:
            pass
    print("[遮蔽/守卫/--rank] config 单遮蔽命中 v4a（model/mamba→v1）；前缀守卫 train 6 反例 + "
          "evaluate 4 反例；--rank 剥离两传法/非法值拦截/实例改写（hash=pin）/rXX 标识正反例全过")


def test_split_caliber_val4() -> None:
    """⑧ val4 划分口径 32/4/10：结构断言 + test 逐 ID 同 v1 + v4a manifest 首建与 v4b 逐字节一致。

    - train 32 / val 4（分层 2 HC + 2 MDD = [25,37 HC + 40,49 MDD]）/ test 10；三折互斥、合计 46
    - test 10 例与 v1 固化 manifest 逐 ID 一致（fold0 外层不动，保 v1/v2/v4b/v4a test 可比——关键断言）
    - v4a 自备固化 manifest：经 v1 data.py default-off 钩子首建写入本目录 splits/、二建校验一致；
      **内容与 v4b 的 val4 manifest 逐字节一致**（同 split_fold 参数决定性输出）
    - v1 与 v4b manifest 均未被改动（内容 hash + mtime 双检）
    - default-off 钩子正反例：无 split_manifest_override 字段 → 原 v1 路径（v1/v2/v3 零漂移）
    """
    cfg = C.Config()

    class _NoOverride:  # 模拟 v1/v2/v3 config（无 split_manifest_override 字段）
        fold, subject_filter, split_seed = cfg.fold, cfg.subject_filter, cfg.split_seed

    v1_manifest = _V1 / "splits" / "fold0_status_ok_seed42.json"
    v4b_manifest = _V4B / "splits" / "fold0_status_ok_seed42_val4.json"
    assert D.split_manifest_path(_NoOverride()) == str(v1_manifest), \
        "default-off 钩子破坏：无字段时未回到 v1 原 manifest 路径（v1/v2/v3 零漂移守护）"
    assert D.split_manifest_path(cfg) == cfg.split_manifest_override
    assert Path(cfg.split_manifest_override).parent == _V4A / "splits", "override 未落在 v4a 自己的 splits/"

    v1_before = hashlib.sha256(v1_manifest.read_bytes()).hexdigest()
    v1_mtime = v1_manifest.stat().st_mtime_ns
    v4b_before = hashlib.sha256(v4b_manifest.read_bytes()).hexdigest()
    v4b_mtime = v4b_manifest.stat().st_mtime_ns

    entries = D.load_manifest(cfg.data_root, subject_filter=cfg.subject_filter)
    assert len(entries) == 46, f"status_ok 过滤后应 46 例，got {len(entries)}"
    train, val, test = D.split_fold(
        entries, n_folds=cfg.n_folds, fold=cfg.fold, split_seed=cfg.split_seed,
        n_val_subjects=cfg.n_val_subjects, val_subject_seed=cfg.val_subject_seed,
    )
    assert (len(train), len(val), len(test)) == (32, 4, 10), \
        f"划分计数漂移：train {len(train)} / val {len(val)} / test {len(test)}（应 32/4/10）"
    ids = lambda xs: sorted(e["subject_id"] for e in xs)  # noqa: E731
    assert ids(val) == [25, 37, 40, 49], f"val 受试者漂移：{ids(val)}（应 = v4b 的 [25,37HC,40,49MDD]）"
    vg = sorted(e["group"] for e in val)
    assert vg == ["HC", "HC", "MDD", "MDD"], f"val 分层破坏（应 2 HC + 2 MDD）：{vg}"
    groups = {e["subject_id"]: e["group"] for e in val}
    assert groups[25] == "HC" and groups[37] == "HC" and groups[40] == "MDD" and groups[49] == "MDD", \
        f"val 分组与 v4b 口径不符：{groups}"
    assert not (set(ids(train)) & set(ids(val))) and not (set(ids(train)) & set(ids(test))) \
        and not (set(ids(val)) & set(ids(test))), "三折受试者重叠（泄漏）"
    assert len(ids(train) + ids(val) + ids(test)) == 46
    saved = json.loads(v1_manifest.read_text(encoding="utf-8"))
    assert ids(test) == sorted(saved["test"]), \
        f"test 折与 v1 固化 manifest 不一致（可比性破坏）：{ids(test)} vs {saved['test']}"
    # v4a 固化 manifest：首建写入（若已存在则校验一致）+ 二建校验一致（fixate_split 幂等）
    p1 = D.fixate_split(cfg, train, val, test)
    p2 = D.fixate_split(cfg, train, val, test)
    assert Path(p1) == Path(cfg.split_manifest_override) and p1 == p2
    # 与 v4b 的 val4 manifest 逐字节一致（同参数决定性划分）
    assert Path(p1).read_bytes() == v4b_manifest.read_bytes(), \
        "v4a manifest 与 v4b 的 val4 manifest 非逐字节一致（划分确定性破坏）"
    # v1 与 v4b manifest 均未被改动（内容 hash + mtime 双检）
    assert hashlib.sha256(v1_manifest.read_bytes()).hexdigest() == v1_before, "v1 manifest 内容被改动"
    assert v1_manifest.stat().st_mtime_ns == v1_mtime, "v1 manifest mtime 被改动"
    assert hashlib.sha256(v4b_manifest.read_bytes()).hexdigest() == v4b_before, "v4b manifest 内容被改动"
    assert v4b_manifest.stat().st_mtime_ns == v4b_mtime, "v4b manifest mtime 被改动"
    print(f"[划分口径] train {len(train)} / val {len(val)} / test {len(test)}（32/4/10，7/1/2 长期口径）；"
          f"val={ids(val)}（{groups[25]},{groups[37]} HC + {groups[40]},{groups[49]} MDD）；test 与 v1 "
          f"manifest 逐 ID 一致；v4a manifest 首建+二建一致且与 v4b 的 val4 manifest **逐字节一致**；"
          f"v1/v4b manifest hash+mtime 未动；default-off 钩子正反例通过")


def main() -> None:
    assert torch.cuda.is_available(), "mamba_ssm 官方 kernel 需要 CUDA"
    cfg = C.Config()
    dev = "cuda"
    torch.manual_seed(cfg.seed)
    model = build_model(cfg, dev, rank=cfg.bottleneck_rank)
    xp, xf, t = _inputs(cfg, dev)

    test_none_branch_equivalence(cfg, xp, xf, t)   # ①
    test_rank16_structure(model, cfg, xp, xf, t)   # ②
    test_forward_grad(model, cfg, xp, xf, t)       # ③
    test_param_count(cfg)                          # ④
    test_config_hash(cfg)                          # ⑤
    test_v1_regression()                           # ⑥
    test_masking_guard_rank()                      # ⑦
    test_split_caliber_val4()                      # ⑧（32/4/10 划分口径 + manifest 逐字节一致）
    print("[PASS] test_v4a 八项全部断言通过")


if __name__ == "__main__":
    main()
