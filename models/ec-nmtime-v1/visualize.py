#!/usr/bin/env python3
"""visualize.py — ec-nmtime-f0 出图（用户 2026-10-18 请求；nature-figure 契约）。

图 1（ec_nmtime_ec）：EC 矩阵证据图，三面板——
  a slope EC（主 readout，连续）；b r\\*-strength EC（τ̃=0.25 翻转，删失=0）；
  c 跨受试者 split-half 稳定性（#1–#5 vs #6–#10 offdiag）。
  核心结论：N×M-time 在 v1-fmt θ* 上给出特异（自跨比 2.11）且跨受试者稳定（ρ=0.90）的 EC。
图 2（ec_nmtime_waveforms）：扰动波形证据图，2×2——
  a 源通道 past 段 baseline vs N@r0 注入；b future 段源通道（自响应）；c future 段最强跨边
  靶通道（跨响应）；d |δ̄(t)| 与 M-time 前 32 样本窗。
  核心结论：N 注入足迹限于源通道 past 段；CRN 配对下 future 呈自/跨响应（A≫B）。
采样复现：win 0 同 CRN seed、同 497 行整批一次 sample()（行独立已证）⇒ 与主 Run 逐位一致。
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "8")

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
_SKILL = Path("/root/npi/.zcode/skills/nature-figure/scripts")
for _p in (str(_HERE), str(_SKILL)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib as mpl         # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402

import candidates as CD   # noqa: E402

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7,
    "axes.spines.right": False, "axes.spines.top": False,
    "axes.linewidth": 0.8, "legend.frameon": False,
})

C_BASE, C_PERT, C_A, C_B, C_WIN = "#9e9e9e", "#0072B2", "#D55E00", "#0072B2", "#f0e442"
RUN01 = Path(CD.load_config().OUTPUTS_ROOT) / "run_01"
FIGS = RUN01 / "figs"
FIGS.mkdir(parents=True, exist_ok=True)


def save_pub_py(fig, filename, dpi=600):
    require_matplotlib_panel_alignment(
        fig, json_out=f"{filename}.alignment.json", overlay_svg=f"{filename}.alignment.svg",
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    fig.savefig(f"{filename}.svg", bbox_inches="tight")
    fig.savefig(f"{filename}.pdf", bbox_inches="tight")
    fig.savefig(f"{filename}.tiff", dpi=dpi, bbox_inches="tight")
    fig.savefig(f"{filename}.png", dpi=300, bbox_inches="tight")


def _slope(dbar_arb):   # (…, 62, 9, 62) [A, r, B] → slope (62, 62)
    import readout as RO
    return RO.slope_readout(np.transpose(dbar_arb, (0, 2, 1)))


def fig_ec() -> None:
    import readout as RO
    SC = CD.load_config()
    sl = np.load(RUN01 / "ec_slope_global.npz")["ec"]
    rs = np.load(RUN01 / "ec_rstar_global.npz")["ec"]
    d = np.load(RUN01 / "dbar_curves.npz")
    dbar_subj = np.stack([np.median(d["dbar"][i * 8:(i + 1) * 8], axis=0)
                          for i in range(len(SC.TEST_SUBJECTS))])
    h1 = _slope(np.median(dbar_subj[:5], axis=0)).ravel()
    h2 = _slope(np.median(dbar_subj[5:], axis=0)).ravel()
    off = ~np.eye(SC.N_SRC, dtype=bool)
    rho = float(spearmanr(h1[off.ravel()], h2[off.ravel()]).statistic)
    n_cens = int((rs[~np.eye(62, dtype=bool)] == 0).sum())

    fig = plt.figure(figsize=(7.2, 2.45))
    gs = fig.add_gridspec(1, 3)                     # 三面板同一 gridspec（对齐审计可比组根基）
    ax0 = fig.add_subplot(gs[0, 0]); ax1 = fig.add_subplot(gs[0, 1]); ax2 = fig.add_subplot(gs[0, 2])
    axes = (ax0, ax1, ax2)
    for ax in axes[:2]:
        ax.set_xticks([0, 31, 61]); ax.set_yticks([0, 31, 61])
        ax.spines[:].set_visible(False)
    im0 = ax0.imshow(sl, cmap="viridis", interpolation="nearest", aspect="equal")
    ax0.set_title("slope EC (main readout)", fontsize=7)
    cb0 = fig.colorbar(im0, cax=ax0.inset_axes([1.03, 0.12, 0.035, 0.76]))
    cb0.ax.tick_params(labelsize=5.5)
    rs_m = np.ma.masked_where(rs == 0, rs)
    cmap = plt.get_cmap("viridis").copy(); cmap.set_bad("#dddddd")
    im1 = ax1.imshow(rs_m, cmap=cmap, interpolation="nearest", aspect="equal")
    ax1.set_title(f"r* strength EC (τ̃=0.25; {n_cens} censored→0)", fontsize=7)
    cb1 = fig.colorbar(im1, cax=ax1.inset_axes([1.03, 0.12, 0.035, 0.76]))
    cb1.ax.tick_params(labelsize=5.5)
    for ax in axes[:2]:
        ax.set_xlabel("target channel B", fontsize=7)
    ax0.set_ylabel("source channel A", fontsize=7)
    ax = ax2
    ax.scatter(h1[off.ravel()], h2[off.ravel()], s=1.2, c="#0072B2", alpha=0.25,
               rasterized=True, linewidths=0)
    lim = [0, max(h1.max(), h2.max()) * 1.02]
    ax.plot(lim, lim, color="#9e9e9e", lw=0.6, ls="--")
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_xlabel("slope EC, subjects #1–#5 (median)", fontsize=7)
    ax.set_ylabel("slope EC, subjects #6–#10 (median)", fontsize=7)
    ax.set_title(f"split-half stability  ρ = {rho:.3f}", fontsize=7)
    import matplotlib.patheffects as pe
    for i, ax in enumerate(axes):
        ax.text(0.03, 0.97, chr(97 + i), transform=ax.transAxes, fontsize=8,
                fontweight="bold", va="top",
                path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])
    fig.tight_layout()
    save_pub_py(fig, FIGS / "ec_nmtime_ec")
    plt.close(fig)
    print(f"[fig1] EC 三面板 → {FIGS}/ec_nmtime_ec.pdf（split-half ρ={rho:.3f}）")


def fig_waveforms(r_val: float = 0.5) -> None:
    import loader as LD
    import torch
    SC = CD.load_config()
    sl = np.load(RUN01 / "ec_slope_global.npz")["ec"]
    off = ~np.eye(SC.N_SRC, dtype=bool)
    a_star = int(np.argmax(np.diag(sl)))
    row = sl[a_star] * off[a_star]                       # a_star 的最强出边（对角排除；slope ≥ 0）
    b_star = int(np.argmax(row))
    in_grid = r_val in SC.R_SAMPLED

    cfg, model, ema, prior, sampler, ck = LD.load_frozen_model("cuda")
    _lo, lds = LD.load_test_loader()
    win_idx, ds_idx, sid, _w = LD.select_windows(lds)[0]
    xp, _xf = lds[ds_idx]
    xp = xp.to("cuda", non_blocking=True)
    eps, eta = CD.crn_noise(win_idx, SC.K, SC.N_CHANNELS, 128, "cuda")
    if in_grid:                                          # 主 Run 整批同构行（逐位一致）
        batch = CD.build_variant_batch(xp)
        with ema.swap(model):
            out = sampler.sample(batch, SC.K, eps=eps, eta=eta)
        v = CD.variant_index(a_star, SC.R_SAMPLED.index(r_val), len(SC.R_SAMPLED))
    else:                                                # 网格外剂量：2 行批（行独立已位级证明——
        xp3 = xp if xp.dim() == 3 else xp.unsqueeze(0)  # r=0 同批恒等 +
        batch = torch.cat([xp3, CD.apply_paradigm(xp3, a_star, r_val)], dim=0)  # 跨批形 baseline
        with ema.swap(model):                            # 逐位相等两证）
            out = sampler.sample(batch, SC.K, eps=eps, eta=eta)
        v = 1
    assert torch.isfinite(out).all(), "采样输出含非有限值（网格外剂量数值不稳？）"
    base = out[0].float().cpu().numpy()                  # (K, 62, 128)
    pert = out[v].float().cpu().numpy()
    past_b = batch[0].float().cpu().numpy()              # (62, 384)
    past_p = batch[v].float().cpu().numpy()
    np.savez(FIGS / f"waveforms_win0_a{a_star}_b{b_star}_r{r_val:g}.npz",
             base=base, pert=pert, past_base=past_b, past_pert=past_p,
             a_star=a_star, b_star=b_star, r=r_val, sid=sid, in_grid=in_grid)

    t_past = (np.arange(-SC.PAST_LEN, 0)) / SC.FS
    t_fut = np.arange(SC.FUTURE_LEN) / SC.FS
    dbar_t = np.abs((pert - base).mean(axis=0))         # (62, 128) |δ̄(t)|

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.3))
    extra = "" if in_grid else "  [extrapolated: > R_MAX]"
    ax = axes[0, 0]
    ax.plot(t_past, past_b[a_star], color=C_BASE, lw=0.5, label="Baseline")
    ax.plot(t_past, past_p[a_star], color=C_PERT, lw=0.5, label=f"N @ r = {r_val:g}", alpha=0.85)
    ax.fill_between(t_past, past_b[a_star], past_p[a_star], color=C_PERT, alpha=0.18, lw=0)
    ax.set_xlabel("time (s)"); ax.set_ylabel("amplitude (z)")
    ax.set_title(f"past segment, source A = ch{a_star} (injection footpoint){extra}", fontsize=7)
    ax.margins(y=0.24)                    # 包络上方预留图例空带（整幅宽度无数据）
    ax.legend(fontsize=5.5, loc="upper center", frameon=True, facecolor="white",
              edgecolor="#bbbbbb", framealpha=0.92)
    for row, (ch, tag, col) in enumerate(((a_star, "source A (self-response)", C_A),
                                          (b_star, "target B (cross-response)", C_B))):
        ax = axes[0 if row == 0 else 1, 1]
        bm, pm = base[:, ch, :].mean(0), pert[:, ch, :].mean(0)
        bs, ps = base[:, ch, :].std(0), pert[:, ch, :].std(0)
        ax.plot(t_fut, bm, color=C_BASE, lw=0.8, label="Baseline (K=16 mean)")
        ax.plot(t_fut, pm, color=col, lw=0.8, label="Perturbed (K=16 mean)")
        ax.fill_between(t_fut, bm - bs, bm + bs, color=C_BASE, alpha=0.2, lw=0)
        ax.fill_between(t_fut, pm - ps, pm + ps, color=col, alpha=0.2, lw=0)
        ax.set_xlabel("time (s)"); ax.set_ylabel("amplitude (z)")
        ax.set_title(f"generated future, {tag} = ch{ch}", fontsize=7)
        if row == 0:
            ax.legend(fontsize=5.5, loc="upper left", frameon=True, facecolor="white",
                      edgecolor="#bbbbbb", framealpha=0.92)
    ax = axes[1, 0]
    ax.plot(t_fut, dbar_t[a_star], color=C_A, lw=0.9, label=f"Source A |δ̄(t)| (ch{a_star})")
    ax.plot(t_fut, dbar_t[b_star], color=C_B, lw=0.9, label=f"Target B |δ̄(t)| (ch{b_star})")
    ax.axvspan(0, SC.MTIME_EARLY_SAMPLES / SC.FS, color=C_WIN, alpha=0.35, lw=0)
    ax.set_xlabel("time (s)"); ax.set_ylabel("|ensemble-mean difference|")
    ax.set_title("response readout |δ̄(t)| (CRN-paired; shaded = M-time 128 ms window)",
                 fontsize=7)
    ax.legend(fontsize=5.5, loc="upper right")
    for i, ax in enumerate(axes.ravel()):
        ax.text(-0.24, 1.24, chr(97 + i), transform=ax.transAxes,
                fontsize=8, fontweight="bold", va="top", zorder=10)
    fig.tight_layout()
    name = "ec_nmtime_waveforms" if in_grid and r_val == 0.5 else \
        f"ec_nmtime_waveforms_r{r_val:g}"
    save_pub_py(fig, FIGS / name)
    plt.close(fig)
    print(f"[fig2] 波形四面板 → {FIGS}/{name}.pdf"
          f"（win0 sid={sid}, A=ch{a_star}, B=ch{b_star}, r={r_val:g}"
          f"{'，网格外外推剂量' if not in_grid else ''}）")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("ec", "all"):
        fig_ec()
    if which in ("wave", "all"):
        fig_waveforms()
    if which == "wave_r":
        fig_waveforms(float(sys.argv[2]))
