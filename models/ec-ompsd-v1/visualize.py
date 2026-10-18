#!/usr/bin/env python3
"""visualize.py — ec-ompsd-f0：最大扰动强度 r=R_MAX=4.0 的 past+future 波形图（用户请求）。

四面板：(a) past 源通道 baseline vs O@4.0（DC 偏置足迹）；(b) future 源通道 gen 对比
（K=16 均值±SD）；(c) future 最强出边靶通道；(d) **M-PSD 读出面**——log10 PSD（gen-pert
vs gen-base，1–40 Hz）。win0 同 CRN 整批 → 与主 Run 逐位一致。"""

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
import metrics as MT      # noqa: E402

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7,
    "axes.spines.right": False, "axes.spines.top": False,
    "axes.linewidth": 0.8, "legend.frameon": False,
})
C_BASE, C_PERT, C_A, C_B = "#9e9e9e", "#0072B2", "#D55E00", "#0072B2"
RUN01 = Path(CD.load_config().OUTPUTS_ROOT) / "run_01"
FIGS = RUN01 / "figs"
FIGS.mkdir(parents=True, exist_ok=True)


def save_pub_py(fig, filename, dpi=600):
    require_matplotlib_panel_alignment(
        fig, json_out=f"{filename}.alignment.json", overlay_svg=f"{filename}.alignment.svg",
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    for ext in ("svg", "pdf", "tiff"):
        fig.savefig(f"{filename}.{ext}", bbox_inches="tight", dpi=dpi if ext == "tiff" else None)
    fig.savefig(f"{filename}.png", dpi=300, bbox_inches="tight")


def main() -> None:
    import loader as LD
    import torch
    SC = CD.load_config()
    sl = np.load(RUN01 / "ec_slope_global.npz")["ec"]
    off = ~np.eye(SC.N_SRC, dtype=bool)
    a_star = int(np.argmax(np.diag(sl)))
    row = sl[a_star] * off[a_star]
    b_star = int(np.argmax(row))
    j_max = len(SC.R_SAMPLED) - 1                    # r = R_MAX = 4.0（最大采样强度）
    r_max = SC.R_SAMPLED[j_max]

    cfg, model, ema, prior, sampler, ck = LD.load_frozen_model("cuda")
    _lo, lds = LD.load_test_loader()
    win_idx, ds_idx, sid, _w = LD.select_windows(lds)[0]
    xp, _xf = lds[ds_idx]
    xp3 = (xp if xp.dim() == 3 else xp.unsqueeze(0)).to("cuda")
    eps, eta = CD.crn_noise(win_idx, SC.K, SC.N_CHANNELS, 128, "cuda")
    batch = CD.build_variant_batch(xp3)
    with ema.swap(model):
        out = sampler.sample(batch, SC.K, eps=eps, eta=eta)
    v = CD.variant_index(a_star, j_max, len(SC.R_SAMPLED))
    base = out[0].float().cpu().numpy()              # (K,62,128)
    pert = out[v].float().cpu().numpy()
    past_b = batch[0].float().cpu().numpy()
    past_p = batch[v].float().cpu().numpy()
    np.savez(FIGS / f"waveforms_win0_a{a_star}_b{b_star}_r{r_max:g}.npz",
             base=base, pert=pert, past_base=past_b, past_pert=past_p,
             a_star=a_star, b_star=b_star, r=r_max, sid=sid)

    t_past = np.arange(-SC.PAST_LEN, 0) / SC.FS
    t_fut = np.arange(SC.FUTURE_LEN) / SC.FS
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.3))
    ax = axes[0, 0]
    ax.plot(t_past, past_b[a_star], color=C_BASE, lw=0.5, label="Baseline")
    ax.plot(t_past, past_p[a_star], color=C_PERT, lw=0.5, alpha=0.85,
            label=f"O @ r = {r_max:g}")
    ax.set_xlabel("time (s)"); ax.set_ylabel("amplitude (z)")
    ax.set_title(f"past, source A = ch{a_star} (DC offset = {r_max:g}σ)", fontsize=7)
    ax.margins(y=0.24)
    ax.legend(fontsize=5.5, loc="upper center", frameon=True, facecolor="white",
              edgecolor="#bbbbbb", framealpha=0.92)
    for ax_, (ch, tag, col) in zip((axes[0, 1], axes[1, 1]),
                                   ((a_star, "source A (self-response)", C_A),
                                    (b_star, "target B (cross-response)", C_B))):
        bm, pm = base[:, ch, :].mean(0), pert[:, ch, :].mean(0)
        bs, ps = base[:, ch, :].std(0), pert[:, ch, :].std(0)
        ax_.plot(t_fut, bm, color=C_BASE, lw=0.8, label="Baseline (K=16 mean)")
        ax_.plot(t_fut, pm, color=col, lw=0.8, label="Perturbed (K=16 mean)")
        ax_.fill_between(t_fut, bm - bs, bm + bs, color=C_BASE, alpha=0.2, lw=0)
        ax_.fill_between(t_fut, pm - ps, pm + ps, color=col, alpha=0.2, lw=0)
        ax_.set_xlabel("time (s)"); ax_.set_ylabel("amplitude (z)")
        ax_.set_title(f"generated future, {tag} = ch{ch}", fontsize=7)
    axes[0, 1].legend(fontsize=5.5, loc="upper left", frameon=True, facecolor="white",
                      edgecolor="#bbbbbb", framealpha=0.92)
    ax = axes[1, 0]
    lp = MT.compute(np.stack([base, pert]))["logpsd"]        # (2,K,C,nf) 全频段
    _f = np.linspace(0, SC.FS / 2, lp.shape[-1])
    band = (_f >= 1.0) & (_f <= 40.0)
    ax.plot(_f[band], lp[0][:, a_star, :].mean(0)[band], color=C_BASE, lw=0.9,
            label="Baseline PSD (K=16 mean)")
    ax.plot(_f[band], lp[1][:, a_star, :].mean(0)[band], color=C_A, lw=0.9,
            label="Perturbed PSD")
    ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("log$_{10}$ PSD")
    ax.set_title(f"M-PSD readout, ch{a_star} (1–40 Hz)", fontsize=7)
    ax.legend(fontsize=5.5, loc="upper right")
    for i, ax_ in enumerate(axes.ravel()):
        ax_.text(-0.24, 1.24, chr(97 + i), transform=ax_.transAxes,
                 fontsize=8, fontweight="bold", va="top", zorder=10)
    fig.tight_layout()
    save_pub_py(fig, FIGS / "ec_ompsd_waveforms_r4")
    plt.close(fig)
    print(f"[fig] 最大强度波形 → {FIGS}/ec_ompsd_waveforms_r4.pdf"
          f"（win0 sid={sid}, A=ch{a_star}, B=ch{b_star}, r={r_max:g}）")


if __name__ == "__main__":
    main()
