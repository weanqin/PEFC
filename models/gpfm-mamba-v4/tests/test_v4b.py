#!/usr/bin/env python3
"""test_v4b.py — v4-B（CI 骨干 + cMLP 组件式读出）单测七项（design §4.4，implement.md §1）。

① CI 隔离性：扰动输入通道 j，trunk 表征 h_{t,k}（k≠j）逐位不变；输出 v 经 head 全通道
   可变（合法跨通道影响——两段验证：trunk 不变 + 输出可变，head 混合是唯一通道间通路）
② 因果性：扰动 future 输入不改 past 位置输出；future 后段扰动不改前段；past 前缀条件生效
③ head 结构：W¹ 形状/列组 reshape 视图正确、einsum == 62 个独立子 MLP、子 MLP 梯度独立
④ 前向形状 (B,62,128) / return_all (B,62,512) / 梯度有限 + EMA swap + GPU B=256 冒烟
⑤ 参数量断言：64,056（d_ch=1, d_z=16 口径，±1%；design §3.2）
⑥ config↔model 一致性：cfg.d_z/head_act/d_model 与类默认同值；config_hash 稳定 + 对 d_z 敏感；
   v1/v2 config_hash 零漂移（子进程口径，基准值取自 v1/v2 已落盘 run）
⑦ 双遮蔽命中 v4 + `v4b-` 前缀守卫正反例 + d_z/head_act 运行时断言正反例

运行：python3 models/gpfm-mamba-v4/tests/test_v4b.py（需 CUDA——mamba_ssm 官方 kernel）
"""

from __future__ import annotations

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
import model as M  # noqa: E402

EXPECTED_PARAMS = 64_056          # design §3.2：2+19+484+1+63,550
V1_HASH_DEFAULT = "c6ff80ac1c43"  # 基准：v1 main-01 落盘 resolved config hash（零漂移守护）
V2_HASH_DEFAULT = "fc2a9ce14948"  # 基准：v2d62-main-01 落盘 resolved config hash


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
    """⑤ 参数量断言：64,056 ±1%（d_ch=1, d_z=16，design §3.2 明细逐项核对）。"""
    n = model.num_params()
    assert abs(n - EXPECTED_PARAMS) <= EXPECTED_PARAMS * 0.01, f"参数量 {n} 偏离 {EXPECTED_PARAMS} 超过 ±1%"
    # 明细核对（逐模块）
    d = {k: p.numel() for k, p in model.named_parameters()}
    seg = d["seg_past"] + d["seg_future"]
    in_proj = d["in_proj.weight"] + d["in_proj.bias"]
    blocks = sum(v for k, v in d.items() if k.startswith("blocks."))
    head_norm = d["head_norm.weight"]
    head = d["W1"] + d["b1"] + d["W2"] + d["b2"]
    assert seg == 2 and in_proj == 19 and blocks == 4 * 121 and head_norm == 1 and head == 63_550, \
        f"参数明细漂移：seg {seg}（=2） in_proj {in_proj}（=19） blocks {blocks}（=484） " \
        f"head_norm {head_norm}（=1） head {head}（=63,550）"
    assert n == EXPECTED_PARAMS, f"参数量精确值 {n} != {EXPECTED_PARAMS}"
    print(f"[参数量] {n} == {EXPECTED_PARAMS}（明细：seg 2 + in_proj 19 + trunk 4×121 + "
          f"RMSNorm 1 + head 63,550；head 占 {head / n:.1%}）")


def test_config_model_consistency(cfg) -> None:
    """⑥ config↔model 一致性 + config_hash 稳定性/敏感性 + v1/v2 基线 hash 零漂移。"""
    assert cfg.d_model == 1 and cfg.d_z == 16 and cfg.head_act == "silu", \
        f"cfg 超参漂移：d_model {cfg.d_model} d_z {cfg.d_z} head_act {cfg.head_act}"
    sig = inspect.signature(M.VelocityNetwork.__init__)
    assert sig.parameters["d_z"].default == cfg.d_z, "model 类默认 d_z != cfg.d_z"
    assert sig.parameters["head_act"].default == cfg.head_act, "model 类默认 head_act != cfg.head_act"
    assert sig.parameters["d_model"].default == cfg.d_model, "model 类默认 d_model(d_ch) != cfg.d_model"
    assert cfg.variant_name == "gpfm-mamba-v4b"
    h1, h2 = C.Config().config_hash(), C.build_config().config_hash()
    assert h1 == h2, "config_hash 不稳定（同配置两次构建不一致）"
    h_mut = dataclasses.replace(cfg, d_z=8).config_hash()
    assert h_mut != h1, "config_hash 对 d_z 不敏感（防线失效）"
    # v1/v2 config_hash 零漂移（子进程隔离导入，基准 = 已落盘 run 的 resolved hash）
    for path, base, name in ((str(_V1), V1_HASH_DEFAULT, "v1"),
                             (str(_V1.parent / "gpfm-mamba-v2"), V2_HASH_DEFAULT, "v2")):
        code = (f"import sys; sys.path.insert(0, {path!r}); import config as C; "
                f"print(C.Config().config_hash())")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        got = out.stdout.strip()
        assert got == base, f"{name} config_hash 漂移：{got} != 基准 {base}（v1/v2 须零改动）"
    print(f"[config↔model] d_model/d_z/head_act 双处一致；config_hash {h1} 稳定且对 d_z 敏感；"
          f"v1 {V1_HASH_DEFAULT} / v2 {V2_HASH_DEFAULT} 零漂移")


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
        d_z, head_act, d_model = 8, "silu", 1

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
    print("[PASS] test_v4b 七项全部断言通过")


if __name__ == "__main__":
    main()
