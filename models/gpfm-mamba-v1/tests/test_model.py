"""test_model.py — Mamba 速度网络单测（implement.md 3.3）。

① 前向形状（v (B,62,128)；return_all (B,62,512)）
② 因果性：扰动 future 输入不改 past 位置输出；future 内部后段扰动不改前段输出；past 扰动改变 future 输出（前缀条件生效）
③ 梯度有限；参数量登记
④ GPU forward+backward 冒烟（训练形状 B=256）
运行：python3 models/gpfm-mamba-v1/tests/test_model.py（需 CUDA——mamba_ssm 官方 kernel）
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as C  # noqa: E402
from model import EMA, VelocityNetwork, time_embedding  # noqa: E402


def main() -> None:
    assert torch.cuda.is_available(), "mamba_ssm 官方 kernel 需要 CUDA"
    cfg = C.Config()
    dev = "cuda"
    torch.manual_seed(cfg.seed)
    model = VelocityNetwork(
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
    n_par = model.num_params()
    print(f"[参数量] {n_par}（{n_par/1e3:.1f}k，登记：4×Mamba(d16/s16/conv4/expand2)+头）")

    B = 4
    xp = torch.randn(B, 62, cfg.past_len, device=dev)
    xf = torch.randn(B, 62, cfg.future_len, device=dev)
    t = torch.tensor([0.1, 0.3, 0.5, 0.9], device=dev)

    # ① 形状
    v = model(xp, xf, t)
    assert v.shape == (B, 62, cfg.future_len), f"v 形状错误 {v.shape}"
    assert torch.isfinite(v).all()
    out_all = model(xp, xf, t, return_all=True)
    assert out_all.shape == (B, 62, cfg.past_len + cfg.future_len)
    assert torch.allclose(out_all[:, :, cfg.past_len:], v, atol=1e-6), "future 切片与 return_all 不一致"
    # t 嵌入单调覆盖检查（dim16，非全零、随 t 变化）
    te = time_embedding(t, cfg.time_emb_dim)
    assert te.shape == (B, cfg.time_emb_dim) and not torch.allclose(te[0], te[-1])
    print(f"[形状] v {tuple(v.shape)}；return_all {tuple(out_all.shape)}；time_emb {tuple(te.shape)}")

    # ② 因果性
    model.eval()
    with torch.no_grad():
        ref = model(xp, xf, t, return_all=True)
        xf2 = xf.clone()
        xf2[:, :, :] += 1.0  # 扰动全部 future 输入
        out2 = model(xp, xf2, t, return_all=True)
        assert torch.allclose(out2[:, :, : cfg.past_len], ref[:, :, : cfg.past_len], atol=1e-5), \
            "扰动 future 输入改变了 past 位置输出（因果性破坏）"
        assert not torch.allclose(out2[:, :, cfg.past_len:], ref[:, :, cfg.past_len:], atol=1e-5), \
            "扰动 future 输入应改变 future 输出"
        # future 内部：扰动第 100 位之后，前 50 位输出不变
        xf3 = xf.clone()
        xf3[:, :, 100:] += 1.0
        out3 = model(xp, xf3, t, return_all=True)
        assert torch.allclose(out3[:, :, cfg.past_len : cfg.past_len + 50], ref[:, :, cfg.past_len : cfg.past_len + 50], atol=1e-5), \
            "future 后段扰动影响了前段输出（序列内因果性破坏）"
        # past 前缀条件必须影响 future 输出
        xp4 = xp.clone()
        xp4[:, :, -50:] += 1.0
        out4 = model(xp4, xf, t, return_all=True)
        assert not torch.allclose(out4[:, :, cfg.past_len:], ref[:, :, cfg.past_len:], atol=1e-5), \
            "past 扰动未影响 future 输出（前缀条件未生效）"
    print("[因果性] future 扰动不改 past 输出；future 后段扰动不改前段；past 前缀条件生效")

    # ③ 梯度有限
    model.train()
    v = model(xp, xf, t)
    loss = (v ** 2).mean()
    loss.backward()
    n_grad = 0
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), "存在非有限梯度"
            n_grad += 1
    print(f"[梯度] {n_grad} 个参数张量梯度全部有限")

    # EMA 更新与 swap
    ema = EMA(model, decay=0.9999)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    v_mod = model(xp, xf, t)  # 当前（已修改）权重下的输出
    ema.update(model)
    with ema.swap(model):
        v_ema = model(xp, xf, t)
    v_restored = model(xp, xf, t)
    assert not torch.allclose(v_ema, v_mod, atol=1e-6), "EMA swap 未生效"
    assert torch.allclose(v_restored, v_mod, atol=1e-6), "swap 恢复后权重与原权重不一致"
    print("[EMA] update/swap/restore 正常")

    # ④ GPU 训练形状冒烟（B=256，forward+backward+计时）
    model.zero_grad(set_to_none=True)
    xpB = torch.randn(cfg.batch_size, 62, cfg.past_len, device=dev)
    xfB = torch.randn(cfg.batch_size, 62, cfg.future_len, device=dev)
    tB = torch.rand(cfg.batch_size, device=dev)
    torch.cuda.synchronize()
    import time

    t0 = time.time()
    vB = model(xpB, xfB, tB)
    (vB ** 2).mean().backward()
    torch.cuda.synchronize()
    dt = time.time() - t0
    assert vB.shape == (cfg.batch_size, 62, cfg.future_len)
    assert torch.isfinite(vB).all()
    print(f"[GPU 冒烟] B={cfg.batch_size} forward+backward {dt*1000:.0f} ms，输出有限")
    print(f"[显存] peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
    print("[PASS] test_model 全部断言通过")


if __name__ == "__main__":
    main()
