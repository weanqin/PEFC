"""test_losses.py — 复合损失四项手工小张量单测（implement.md 4.2）。

- 手工小张量对照独立 numpy 参考实现（逐项，rtol 1e-6）
- 常数序列边界：corr 的近平坦通道置 r=0；tv/cons 手工核对
运行：python3 models/gpfm-mamba-v1/tests/test_losses.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from losses import composite_loss, cons_loss, corr_loss, mae_loss, per_window_mae, tv_loss  # noqa: E402


def ref_all(xh: np.ndarray, x1: np.ndarray, eps: float = 1e-6):
    """独立 numpy 参考实现（不共用任何中间量）。"""
    B, C, T = xh.shape
    mae = np.abs(xh - x1).mean()
    cons_terms = []
    corr_terms = []
    for b in range(B):
        for c in range(C):
            a, g = xh[b, c], x1[b, c]
            cons_terms.append(abs(a.std(ddof=1) - g.std(ddof=1)) + abs(a.mean() - g.mean()))
            ae, ge = a - a.mean(), g - g.mean()
            sxy, sxx, syy = (ae * ge).sum(), (ae * ae).sum(), (ge * ge).sum()
            if sxx < eps or syy < eps:
                r = 0.0
            else:
                r = sxy / (np.sqrt(sxx * syy) + eps)
                r = min(1.0, max(-1.0, r))
            corr_terms.append(1.0 - r)
    tv = np.abs(np.diff(xh, axis=-1)).mean()
    return mae, float(np.mean(cons_terms)), float(tv), float(np.mean(corr_terms))


def main() -> None:
    rng = np.random.default_rng(123)
    # 手工小张量（含常数通道边界）
    xh = rng.normal(size=(3, 4, 7))
    x1 = rng.normal(size=(3, 4, 7))
    xh[1, 2, :] = 0.5          # x̂1 常数通道（S_xx=0 → r=0）
    x1[2, 0, :] = -1.25        # x1 常数通道（S_yy=0 → r=0）
    xh_t = torch.tensor(xh)
    x1_t = torch.tensor(x1)

    r_mae, r_cons, r_tv, r_corr = ref_all(xh, x1)
    g_mae = float(mae_loss(xh_t, x1_t))
    g_cons = float(cons_loss(xh_t, x1_t))
    g_tv = float(tv_loss(xh_t))
    g_corr = float(corr_loss(xh_t, x1_t))
    for name, r, g in [("mae", r_mae, g_mae), ("cons", r_cons, g_cons), ("tv", r_tv, g_tv), ("corr", r_corr, g_corr)]:
        assert abs(r - g) <= 1e-6 * max(1.0, abs(r)), f"{name} 与 numpy 参考不一致: ref={r:.8f} got={g:.8f}"
        print(f"[{name}] ref={r:.8f} got={g:.8f} OK")
    # 分母口径手工核对（设计逐字）
    assert abs(g_mae - np.abs(xh - x1).sum() / (3 * 4 * 7)) < 1e-12
    assert abs(g_tv - np.abs(np.diff(xh, axis=-1)).sum() / (3 * 4 * 6)) < 1e-12  # B·C·(T−1)

    comp = composite_loss(xh_t, x1_t)
    exp_total = 1.0 * g_mae + 1.0 * g_cons + 0.1 * g_tv + 0.1 * g_corr
    assert abs(float(comp["total"]) - exp_total) < 1e-6, "总损失加权口径错误"
    assert abs(float(comp["mae"]) - g_mae) < 1e-9

    # 常数序列边界显式断言
    c1 = torch.full((1, 1, 5), 2.0)
    c2 = torch.arange(5.0).reshape(1, 1, 5)
    assert float(corr_loss(c1, c2)) == 1.0, "x̂1 恒定 → r=0 → corr=1"
    assert float(corr_loss(c2, c1)) == 1.0, "x1 恒定 → r=0 → corr=1"
    assert float(tv_loss(c1)) == 0.0, "常数序列 tv=0"
    assert float(cons_loss(c1, c2)) == abs(0.0 - c2.std(unbiased=True)) + abs(2.0 - 2.0)
    # NaN guard
    bad = torch.tensor([[[float("nan"), 1.0, 2.0]]])
    ok = torch.zeros(1, 1, 3)
    for fn in (lambda a, b: mae_loss(a, b), lambda a, b: cons_loss(a, b), lambda a, b: corr_loss(a, b)):
        assert torch.isfinite(fn(bad, ok)).all(), "NaN guard 失效"
    # 每窗 mae（t 分桶监控）
    pm = per_window_mae(xh_t, x1_t)
    assert pm.shape == (3,) and torch.allclose(pm[1], torch.tensor(xh[1] - x1[1]).abs().mean())
    print("[PASS] test_losses 全部断言通过")


if __name__ == "__main__":
    main()
