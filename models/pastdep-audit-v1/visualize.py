#!/usr/bin/env python3
"""visualize.py — pastdep-audit-f0 诊断图（3 面板，nature-figure 契约）。

a E1 对拍：解析 env(f) vs 采样自响应 |δ̄|(t)（半对数）；b E2 注入位置（解析子窗质量 vs
采样响应成对棒）；c E3 置换 ρ 棒 + IQR（判读带阴影）。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), "/root/npi/.zcode/skills/nature-figure/scripts"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib as mpl         # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402

import candidates as CD   # noqa: E402

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7,
    "axes.spines.right": False, "axes.spines.top": False,
    "axes.linewidth": 0.8, "legend.frameon": False,
})
C_ANA, C_SAMP = "#009E73", "#D55E00"
RUN01 = Path(CD.load_config().OUTPUTS_ROOT) / "run_01"
FIGS = RUN01 / "figs"
FIGS.mkdir(parents=True, exist_ok=True)


def save_pub_py(fig, filename, dpi=600):
    require_matplotlib_panel_alignment(
        fig, json_out=f"{filename}.alignment.json", overlay_svg=f"{filename}.alignment.svg",
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    for ext in ("svg", "pdf", "tiff"):
        fig.savefig(f"{filename}.{ext}", bbox_inches="tight",
                    dpi=dpi if ext == "tiff" else None)
    fig.savefig(f"{filename}.png", dpi=300, bbox_inches="tight")


def main_audit() -> None:
    SC = CD.load_config()
    ap = np.load(RUN01 / "A_prior.npz")
    cu = np.load(RUN01 / "curves.npz")
    e3 = np.load(RUN01 / "e3.npz")

    fig = plt.figure(figsize=(7.2, 2.5))
    gs = fig.add_gridspec(1, 3)
    ax0 = fig.add_subplot(gs[0, 0]); ax1 = fig.add_subplot(gs[0, 1]); ax2 = fig.add_subplot(gs[0, 2])

    # (a) E1 对拍（半对数）
    env = ap["env"] / ap["env"].max()
    samp = np.median(cu["e1_self"][:, 2].reshape(-1, 128), axis=0)     # r=2.0 全窗全源自响应
    samp = samp / samp.max()
    f = np.arange(128)
    ax0.semilogy(f[f < 60], env[f < 60], color=C_ANA, lw=0.9,
                 label=f"prior path env(f)  T50={ap['t50_prior']:.0f}")
    ax0.semilogy(f[f < 60], samp[f < 60], color=C_SAMP, lw=0.9,
                 label="sampled |δ̄(t)| (self, r=2)")
    ax0.axvline(float(ap["t50_prior"]), color="#9e9e9e", lw=0.5, ls=":")
    ax0.set_xlabel("future time (samples)"); ax0.set_ylabel("normalized influence")
    ax0.set_ylim(3e-4, 2)
    ax0.legend(fontsize=5.2, loc="lower left")

    # (b) E2 子窗（解析质量 vs 采样响应，成对棒）
    x = np.arange(4)
    sm = cu["e2_targetmed"].mean(axis=(0, 2))
    ax1.bar(x - 0.19, ap["submass"], width=0.36, color=C_ANA, label="prior mass (analytic)")
    ax1.bar(x + 0.19, sm / sm.max(), width=0.36, color=C_SAMP,
            label="sampled response (norm.)")
    ax1.set_xticks(x); ax1.set_xticklabels(["[0,96)", "[96,192)", "[192,288)", "[288,384)"],
                                            fontsize=5.5)
    ax1.set_xlabel("past injection subwindow"); ax1.set_ylabel("normalized response")
    ax1.legend(fontsize=5.2, loc="upper left")

    # (c) E3 ρ 棒 + IQR
    d0 = np.median(e3["dist"][:, 0, :], axis=1)
    names = ("same", "cross", "ablate")
    meds, iqrs = [], []
    for ai in (1, 2, 3):
        r = np.median(e3["dist"][:, ai, :], axis=1) / d0
        meds.append(np.median(r))
        iqrs.append(np.percentile(r, 75) - np.percentile(r, 25))
    ax2.bar(np.arange(3), meds, yerr=iqrs, width=0.55, color="#0072B2",
            error_kw=dict(lw=0.7, ecolor="#444444", capsize=2))
    ax2.axhline(1.0, color="#9e9e9e", lw=0.6, ls="--")
    ax2.axhspan(1.2, 2.0, color="#f0e442", alpha=0.3, lw=0)
    ax2.set_xticks(np.arange(3)); ax2.set_xticklabels(names)
    ax2.set_xlabel("past replacement arm"); ax2.set_ylabel("ρ = d(arm)/d(noise floor)")
    ax2.text(0.02, 0.95, f"medians: {', '.join(f'{m:.1f}' for m in meds)}",
             transform=ax2.transAxes, fontsize=5.2, va="top")

    for i, ax in enumerate((ax0, ax1, ax2)):
        ax.text(-0.24, 1.06, chr(97 + i), transform=ax.transAxes,
                fontsize=8, fontweight="bold", va="top")
    fig.tight_layout()
    save_pub_py(fig, FIGS / "pastdep_audit")
    plt.close(fig)
    print(f"[fig] pastdep_audit → {FIGS}/pastdep_audit.pdf")


def fig_waveforms_three() -> None:
    """future 三曲线：real（数据真值）/ gen（CRN 集群均值±SD）/ OU 先验 μ = A·x_past。

    win0（sid=10），通道 ch36（A，自响应最强）与 ch2（B，最强出边靶）——与 ec-nmtime
    波形图同对象，跨任务可比。"""
    import loader as LD
    import torch
    SC = CD.load_config()
    cfg, model, ema, prior, sampler, ck = LD.load_frozen_model("cuda")
    _lo, lds = LD.load_test_loader()
    win_idx, ds_idx, sid, _w = LD.select_windows(lds)[0]
    xp, xf = lds[ds_idx]
    xp3 = (xp if xp.dim() == 3 else xp.unsqueeze(0)).to("cuda")
    xf = xf if xf.dim() == 2 else xf[0]
    with torch.no_grad():
        mu = prior.posterior_mean(xp3)[0].float().cpu().numpy()          # (62,128)
        eps, eta = CD.crn_noise(SC.SEED + win_idx, SC.K, SC.N_CHANNELS, 128, "cuda")
        with ema.swap(model):
            gen = sampler.sample(xp3, SC.K, eps=eps, eta=eta)[0].float().cpu().numpy()
    past = xp3[0].float().cpu().numpy()                                   # (62,384)
    real = xf.float().numpy()                                             # (62,128)
    np.savez(FIGS / f"wave3_win0_ch36_ch2.npz", past=past, real=real, gen=gen, mu=mu, sid=sid)

    t_past = np.arange(-SC.PAST_LEN, 0) / SC.FS
    t_fut = np.arange(128) / SC.FS
    C_REAL, C_GEN, C_MU, C_PAST = "#000000", "#0072B2", "#D55E00", "#9e9e9e"

    fig = plt.figure(figsize=(7.2, 2.55))
    gs = fig.add_gridspec(1, 2)
    axs = [fig.add_subplot(gs[0, i]) for i in range(2)]
    for ax, ch, name in zip(axs, (36, 2), ("A = ch36", "B = ch2")):
        ax.plot(t_past, past[ch], color=C_PAST, lw=0.5)
        ax.axvline(0, color="#bbbbbb", lw=0.6, ls="--")
        gm, gs_ = gen[:, ch, :].mean(0), gen[:, ch, :].std(0)
        ax.plot(t_fut, real[ch], color=C_REAL, lw=0.7, label="Real (data)")
        ax.plot(t_fut, gm, color=C_GEN, lw=0.9, label="Generated (K=16 mean)")
        ax.fill_between(t_fut, gm - gs_, gm + gs_, color=C_GEN, alpha=0.2, lw=0)
        ax.plot(t_fut, mu[ch], color=C_MU, lw=1.1, ls="--", label="OU prior μ = A·x_past")
        ax.margins(y=0.15)
        ax.set_xlabel("time (s)"); ax.set_ylabel("amplitude (z)")
        ax.set_title(f"{name}", fontsize=7)
    _h, _l = axs[0].get_legend_handles_labels()
    fig.legend(_h, _l, fontsize=5.5, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.12))
    for i, ax in enumerate(axs):
        ax.text(-0.20, 1.10, chr(97 + i), transform=ax.transAxes,
                fontsize=8, fontweight="bold", va="top")
    fig.tight_layout()
    save_pub_py(fig, FIGS / "pastdep_waveforms_3curves")
    plt.close(fig)
    print(f"[fig] 三曲线波形 → {FIGS}/pastdep_waveforms_3curves.pdf（win0 sid={sid}）")


def fig_inference_window() -> None:
    """512-token 推理窗可视化（用户口径更正 2026-10-18）：模型每次 Euler 前向吃
    384 past 数据前缀 + 128 future 流变量位；128 位由 x0=μ+Lε 初始化、16 步 Euler 流
    逐步计算（real 不参与推理，仅对照）。快照 {0,4,8,12,16} 步。"""
    import loader as LD
    import torch
    SC = CD.load_config()
    cfg, model, ema, prior, sampler, ck = LD.load_frozen_model("cuda")
    _lo, lds = LD.load_test_loader()
    win_idx, ds_idx, sid, _w = LD.select_windows(lds)[0]
    xp, xf = lds[ds_idx]
    xp3 = (xp if xp.dim() == 3 else xp.unsqueeze(0)).to("cuda")
    inp = xp3.clone()                                  # 384 数据前缀逐位见证
    with torch.no_grad():
        mu = prior.posterior_mean(xp3)
        eps, eta = CD.crn_noise(SC.SEED + win_idx, SC.K, SC.N_CHANNELS, 128, "cuda")
        x0 = prior.bandlimit(mu.unsqueeze(1) + torch.einsum(
            "ij,bqjk,lk->bqil", prior.L_s, eps, prior.L_t))
        x0 = x0 + sampler.sigma_min * eta                       # (1,K,C,F) 流初值
        snaps = {0: x0[0].float().cpu().numpy()}
        x = x0.clone(); dt = 1.0 / sampler.steps
        with ema.swap(model):
            for i in range(sampler.steps):
                tt = torch.full((x.shape[1],), i * dt, device=x.device, dtype=x.dtype)
                # 每步前向 = 512-token 序列（384 前缀 + 128 当前流状态）
                v = sampler.model(xp3.expand(x.shape[1], -1, -1), x[0], tt)
                x = x + dt * v
                if (i + 1) in (4, 8, 12):
                    snaps[i + 1] = x[0].float().cpu().numpy()
        snaps[sampler.steps] = x[0].float().cpu().numpy()       # 终态 = gen
    assert torch.equal(inp, xp3)
    past = inp[0].float().cpu().numpy()
    real = (xf if xf.dim() == 2 else xf[0]).float().numpy()
    np.savez(FIGS / "inference_window512_win0.npz", past=past, real=real,
             snaps=np.stack([snaps[k] for k in (0, 4, 8, 12, 16)]), sid=sid)

    t_past = np.arange(-SC.PAST_LEN, 0) / SC.FS
    t_fut = np.arange(128) / SC.FS
    C_REAL, C_GEN, C_PAST = "#000000", "#0072B2", "#9e9e9e"
    fig = plt.figure(figsize=(7.2, 2.6))
    gs = fig.add_gridspec(1, 2)
    axs = [fig.add_subplot(gs[0, i]) for i in range(2)]
    for ax, ch in zip(axs, (36, 2)):
        ax.plot(t_past, past[ch], color=C_PAST, lw=0.5)
        ax.axvline(0, color="#bbbbbb", lw=0.6, ls="--")
        ax.plot(t_fut, real[ch], color=C_REAL, lw=0.6, alpha=0.85)
        for si, k in enumerate((4, 8, 12)):
            ax.plot(t_fut, np.mean(snaps[k][:, ch, :], axis=0), color=C_GEN,
                    lw=0.5, alpha=0.18 + 0.10 * si)
        ax.plot(t_fut, np.mean(snaps[0][:, ch, :], axis=0), color="#56B4E9",
                lw=0.8, ls=":")
        gm = np.mean(snaps[16][:, ch, :], axis=0)
        gs_ = np.std(snaps[16][:, ch, :], axis=0)
        ax.plot(t_fut, gm, color=C_GEN, lw=1.0)
        ax.fill_between(t_fut, gm - gs_, gm + gs_, color=C_GEN, alpha=0.15, lw=0)
        ax.set_xlabel("time (s)"); ax.set_ylabel("amplitude (z)")
        ax.set_title(f"ch{ch} — 512-token window (384 data + 128 computed)", fontsize=7)
    import matplotlib.lines as mlines
    handles = [mlines.Line2D([], [], color=C_PAST, lw=1.0, label="384 past prefix (data)"),
               mlines.Line2D([], [], color="#56B4E9", lw=1.0, ls=":",
                             label="x₀ = μ+Lε (flow init)"),
               mlines.Line2D([], [], color=C_GEN, lw=0.5, alpha=0.5,
                             label="Euler steps 4/8/12 (K=16 mean)"),
               mlines.Line2D([], [], color=C_GEN, lw=1.2, label="Generated (step 16, ±SD)"),
               mlines.Line2D([], [], color=C_REAL, lw=0.8, label="Real (data, not used in inference)")]
    fig.legend(handles=handles, fontsize=5.2, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.14))
    for i, ax in enumerate(axs):
        ax.text(-0.20, 1.10, chr(97 + i), transform=ax.transAxes,
                fontsize=8, fontweight="bold", va="top")
    fig.tight_layout()
    save_pub_py(fig, FIGS / "pastdep_inference_window512")
    plt.close(fig)
    print(f"[fig] 512 推理窗 → {FIGS}/pastdep_inference_window512.pdf（win0 sid={sid}）")


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "audit"
    if which in ("audit", "all"):
        main_audit()
    if which in ("wave3", "all"):
        fig_waveforms_three()
    if which in ("win", "all"):
        fig_inference_window()


if __name__ == "__main__":
    main()
