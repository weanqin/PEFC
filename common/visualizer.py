"""common/visualizer.py —— 仓库统一 visualizer（visualization.md 契约；共享只读区起点）。

- 契约：以「数据 + 图类型 + 配置」为输入，图文件输出到调用方变体的 run 目录 figs/；
  实验/评估/训练代码内不得散落临时绘图代码（spec 纪律），新增图类型在本模块扩展。
- 默认风格遵循 nature-figure figure contract（字体/字号/配色/版式）：
  * sans-serif（Arial -> Helvetica -> DejaVu Sans），基础字号 7 pt（5 pt 下限之上），
    可编辑文本（pdf.fonttype=42 / svg.fonttype=none），去上/右 spine，轴线 0.8 pt，图例无框；
  * 克制配色：中性灰（上下文/先验）+ 蓝（真实侧）+ 红（生成侧/告警）一个信号族；
  * 多面板图带 a/b/c 面板标号，可直接标注（direct labeling）优先于图例。
- 输出：300 dpi PNG（评估/诊断产物口径；论文投稿级导出另走 nature-figure 流程）。

图类型（FIG_TYPES）：
  trajectories     多窗「真实 vs 生成轨迹」（medoid 代表样本 + 90% 逐点预测区间；人工视觉检验素材）
  psd              逐通道真实/生成 log-log 叠加 + 全通道平均
  gc_matrix        GC 有向矩阵对比（真实/生成热图 + 散点）
  training_curves  训练曲线（四项损失+total / 按 t 分桶 mae / EMA 幅度监控）
  spatial_corr     零滞后空间相关三方对比热图（R̂ / K_s / 真实 R）
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np  # noqa: E402

# ---- nature-figure 契约配色（克制三族：中性 / 真实蓝 / 生成红 + 辅助）----
PALETTE = {
    "blue_main": "#0F4D92",     # 真实侧主色
    "blue_secondary": "#3775BA",
    "red_strong": "#B64342",    # 生成侧主色
    "red_soft": "#E9A6A1",
    "neutral_light": "#CFCECE", # 上下文/先验
    "neutral_mid": "#767676",
    "neutral_dark": "#4D4D4D",
    "teal": "#42949E",
    "violet": "#9A4D8E",
    "gold": "#FFD700",
}

DEFAULT_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",       # SVG 内可编辑文本
    "pdf.fonttype": 42,           # PDF 内可编辑 TrueType
    "font.size": 7,               # 基础字号 7 pt（>5 pt 渲染下限）
    "axes.titlesize": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "legend.frameon": False,
    "figure.dpi": 120,
    "savefig.dpi": 300,           # 产物口径：300 dpi PNG
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
}


def apply_style() -> None:
    """应用 nature-figure 契约 rcParams（幂等）。"""
    mpl.rcParams.update(DEFAULT_RC)


def add_panel_label(ax, label: str, dx: float = -0.12, dy: float = 1.04) -> None:
    """面板标号 a/b/c（bold，7pt 契约下限之上）。"""
    ax.text(dx, dy, label, transform=ax.transAxes, fontsize=8,
            fontweight="bold", va="top", ha="left")


class Visualizer:
    """统一 visualizer：render(fig_type, data, filename, **cfg) -> 图文件路径。

    data 为纯 numpy 字典（不接收 torch 张量——评估侧先 .cpu().numpy()）。
    各图类型的数据契约见对应 _draw_* 方法的 docstring。
    """

    FIG_TYPES = ("trajectories", "psd", "gc_matrix", "training_curves", "spatial_corr")

    def __init__(self, out_dir: str, dpi: int = 300):
        self.out_dir = out_dir
        self.dpi = dpi
        os.makedirs(out_dir, exist_ok=True)
        apply_style()

    # ---- 统一入口 ----
    def render(self, fig_type: str, data: Dict, filename: str, **cfg) -> str:
        if fig_type not in self.FIG_TYPES:
            raise KeyError(f"未知图类型 {fig_type!r}；注册表: {self.FIG_TYPES}")
        fig = getattr(self, f"_draw_{fig_type}")(data, **cfg)
        if fig is None:
            raise RuntimeError(f"{fig_type} 未返回 figure")
        path = os.path.join(self.out_dir, filename)
        fig.savefig(path, dpi=self.dpi)
        plt.close(fig)
        return path

    # ---- ① 真实 vs 生成轨迹（medoid 代表 + 90% 预测区间；人工视觉检验）----
    def _draw_trajectories(self, data: Dict, **cfg) -> mpl.figure.Figure:
        """data: {
            past   (W, C, P) 真实 past（上下文，浅灰），
            real   (W, C, F) 真实 future（蓝粗线，独立参考曲线），
            gen_medoid (W, C, F) 代表生成轨迹（红）——K 个实际生成样本中按完整多通道轨迹
              pairwise RMSE 选出的 medoid 实际样本（非逐时间点统计量重构的虚拟波形），
            gen_q05, gen_q95 (W, C, F) 全部 K 样本的逐时间点 5/95 分位
              （fill_between 阴影 = 90% predictive interval；不用 min-max envelope），
            window_ids (W,), channel_ids (C,)（标题用）,
            fs 采样率（默认 500）, context 点数（默认 64）, K 生成样本数（图例标注）
        }"""
        past, real = data["past"], data["real"]
        medoid, q05, q95 = data["gen_medoid"], data["gen_q05"], data["gen_q95"]
        W, C, P = past.shape
        F = real.shape[-1]
        fs = data.get("fs", 500)
        ctx = min(data.get("context", 64), P)
        K = data.get("K")
        fig, axes = plt.subplots(W, C, figsize=(1.5 * C, 1.1 * W), squeeze=False,
                                 sharex=True)
        t_ctx = np.arange(-ctx, 0) / fs
        t_fut = np.arange(F) / fs
        for w in range(W):
            for c in range(C):
                ax = axes[w][c]
                ax.plot(t_ctx, past[w, c, -ctx:], color=PALETTE["neutral_light"],
                        lw=0.6, zorder=1)
                ax.fill_between(t_fut, q05[w, c], q95[w, c],
                                color=PALETTE["red_strong"], alpha=0.18, lw=0, zorder=2)
                ax.plot(t_fut, medoid[w, c], color=PALETTE["red_strong"], lw=0.8,
                        zorder=3)
                ax.plot(t_fut, real[w, c], color=PALETTE["blue_main"], lw=1.0,
                        zorder=4)
                if w == 0:
                    ax.set_title(f"ch{data['channel_ids'][c]}", fontsize=6)
                if w == W - 1:
                    ax.set_xlabel("time (s)")
                if c == 0:
                    ax.set_ylabel(f"win {data['window_ids'][w]}", fontsize=6)
        if W * C >= 2:
            add_panel_label(axes[0][0], "a")
        handles = [
            plt.Line2D([], [], color=PALETTE["neutral_light"], lw=1, label="past (context)"),
            plt.Line2D([], [], color=PALETTE["blue_main"], lw=1, label="real future"),
            plt.Line2D([], [], color=PALETTE["red_strong"], lw=1,
                       label="generated medoid" + (f" (K={K})" if K is not None else "")),
            Patch(facecolor=PALETTE["red_strong"], alpha=0.25, lw=0,
                  label="90% predictive interval"),
        ]
        fig.legend(handles=handles, loc="upper right", ncol=4, bbox_to_anchor=(1.0, 1.02))
        fig.tight_layout(h_pad=0.4, w_pad=0.4)
        return fig

    # ---- ② PSD 对比 ----
    def _draw_psd(self, data: Dict, **cfg) -> mpl.figure.Figure:
        """data: {
            freqs (nfreq,), psd_real (C, nfreq), psd_gen (C, nfreq)（功率谱密度）
        }；面板 a = 逐通道叠加（两组同色族淡线），面板 b = 全通道平均（粗线直接标注）。"""
        freqs = data["freqs"]
        pr, pg = data["psd_real"], data["psd_gen"]
        pos = freqs > 0
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.6))
        for c in range(pr.shape[0]):
            ax1.plot(freqs[pos], pr[c][pos], color=PALETTE["blue_main"], lw=0.3, alpha=0.15)
            ax1.plot(freqs[pos], pg[c][pos], color=PALETTE["red_strong"], lw=0.3, alpha=0.15)
        ax1.plot([], [], color=PALETTE["blue_main"], lw=1, label="real (per channel)")
        ax1.plot([], [], color=PALETTE["red_strong"], lw=1, label="generated (per channel)")
        ax1.set_xscale("log"); ax1.set_yscale("log")
        ax1.set_xlabel("frequency (Hz)"); ax1.set_ylabel("PSD"); ax1.legend()
        ax1.set_title("per-channel PSD", fontsize=6)
        add_panel_label(ax1, "a", dx=-0.18)
        mr, mg = pr.mean(0), pg.mean(0)
        ax2.plot(freqs[pos], mr[pos], color=PALETTE["blue_main"], lw=1.0, label="real")
        ax2.plot(freqs[pos], mg[pos], color=PALETTE["red_strong"], lw=1.0, label="generated")
        ax2.set_xscale("log"); ax2.set_yscale("log")
        ax2.set_xlabel("frequency (Hz)"); ax2.set_ylabel("mean PSD")
        ax2.text(0.03, 0.05, f"n={pr.shape[0]} channels", transform=ax2.transAxes,
                 fontsize=6, color=PALETTE["neutral_mid"])
        ax2.set_title("all-channel mean", fontsize=6)
        add_panel_label(ax2, "b", dx=-0.18)
        fig.tight_layout()
        return fig

    # ---- ③ GC 矩阵对比 ----
    def _draw_gc_matrix(self, data: Dict, **cfg) -> mpl.figure.Figure:
        """data: {
            gc_real (C,C), gc_gen (C,C) 有向 GC 矩阵（ln SSR 比，对角 0），
            stats dict（r/spearman/CI 等，散点面板标注用，可缺省）
        }；面板 a/b 热图（RdBu_r 对称于 0），面板 c 去对角散点 + r/ρ 标注。"""
        gr, gg = np.asarray(data["gc_real"]), np.asarray(data["gc_gen"])
        stats = data.get("stats", {})
        C = gr.shape[0]
        vmax = cfg.get("vmax", max(np.abs(gr).max(), np.abs(gg).max()))
        fig = plt.figure(figsize=(7.0, 2.7))
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.15], wspace=0.35)
        ax1 = fig.add_subplot(gs[0]); ax2 = fig.add_subplot(gs[1]); ax3 = fig.add_subplot(gs[2])
        for ax, m, lab in ((ax1, gr, "real"), (ax2, gg, "generated")):
            im = ax.imshow(m, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
            ax.set_title(f"conditional GC ({lab})", fontsize=6)
            ax.set_xlabel("target j"); ax.set_ylabel("source i")
        cb = fig.colorbar(im, ax=[ax1, ax2], shrink=0.85, pad=0.02)
        cb.set_label("ln SSR ratio", fontsize=6); cb.ax.tick_params(labelsize=5)
        add_panel_label(ax1, "a", dx=-0.28)
        add_panel_label(ax2, "b", dx=-0.28)
        iu = ~np.eye(C, dtype=bool)
        x, y = gr[iu], gg[iu]
        ax3.scatter(x, y, s=1.5, alpha=0.35, color=PALETTE["blue_main"], rasterized=True)
        lim = max(np.abs(x).max(), np.abs(y).max()) * 1.05
        ax3.plot([-lim, lim], [-lim, lim], ls="--", lw=0.6, color=PALETTE["neutral_mid"])
        ax3.set_xlim(-lim, lim); ax3.set_ylim(-lim, lim)
        ax3.set_xlabel("real GC"); ax3.set_ylabel("generated GC")
        txt = ""
        if "pearson_r" in stats:
            txt += f"Pearson r={stats['pearson_r']:.3f}"
        if "spearman_rho" in stats:
            txt += f"\nSpearman rho={stats['spearman_rho']:.3f}"
        if "pearson_ci95" in stats:
            lo, hi = stats["pearson_ci95"]
            txt += f"\n95% CI [{lo:.3f}, {hi:.3f}]"
        ax3.text(0.03, 0.97, txt.strip(), transform=ax3.transAxes, fontsize=6,
                 va="top", color=PALETTE["neutral_dark"])
        add_panel_label(ax3, "c", dx=-0.22)
        return fig

    # ---- ④ 训练曲线 ----
    def _draw_training_curves(self, data: Dict, **cfg) -> mpl.figure.Figure:
        """data: {records: [metrics.jsonl 逐行 dict]}。
        面板 a = train 四项损失+total；面板 b = 按 t 分桶 mae（b0/b2/b4/b6/b8）；
        面板 c = EMA 幅度监控（xhat1_std_ema vs x1_std）+ val total（右轴省略，直接双线）。"""
        recs = data["records"]
        ep = np.array([r["epoch"] + 1 for r in recs])
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7.0, 2.4))
        items = [("mae", "mae"), ("cons", "cons"), ("tv", "tv"), ("corr", "corr"),
                 ("total", "total")]
        colors = [PALETTE["blue_main"], PALETTE["teal"], PALETTE["violet"],
                  PALETTE["gold"], PALETTE["neutral_dark"]]
        for (k, _), col in zip(items, colors):
            ax1.plot(ep, [r["train"][k] for r in recs], lw=0.8, color=col, label=k)
        ax1.set_xlabel("epoch"); ax1.set_ylabel("train loss"); ax1.legend(ncol=2)
        add_panel_label(ax1, "a", dx=-0.22)
        for b, col in zip([0, 2, 4, 6, 8], colors):
            ys = [r["t_bucket_mae"][f"b{b}"] if r["t_bucket_mae"][f"b{b}"] is not None else np.nan
                  for r in recs]
            ax2.plot(ep, ys, lw=0.8, color=col, label=f"t$\\in$[{b/10:.1f},{b/10+0.1:.1f})")
        ax2.set_xlabel("epoch"); ax2.set_ylabel("one-step mae"); ax2.legend(ncol=2)
        add_panel_label(ax2, "b", dx=-0.22)
        ax3.plot(ep, [r["val"]["xhat1_std_ema"] for r in recs], lw=0.8,
                 color=PALETTE["red_strong"], label="$\\hat x_1$ std (EMA)")
        ax3.plot(ep, [r["val"]["x1_std"] for r in recs], lw=0.8, ls="--",
                 color=PALETTE["blue_main"], label="$x_1$ std")
        ax3.set_xlabel("epoch"); ax3.set_ylabel("temporal std"); ax3.legend()
        add_panel_label(ax3, "c", dx=-0.22)
        fig.tight_layout()
        return fig

    # ---- ⑤ 零滞后空间结构三方对比 ----
    def _draw_spatial_corr(self, data: Dict, **cfg) -> mpl.figure.Figure:
        """data: {
            R_hat (C,C) 生成样本零滞后相关, K_s (C,C) 先验空间核, R_real (C,C) 真实零滞后相关,
            stats dict（Frobenius/上三角相关数值，可缺省）
        }；三面板热图（RdBu_r，[-1,1]）。"""
        mats = [("generated  $\\hat R$", data["R_hat"]),
                ("prior  $K_s$", data["K_s"]),
                ("real  $R$", data["R_real"])]
        stats = data.get("stats", {})
        fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5))
        for ax, (title, m) in zip(axes, mats):
            im = ax.imshow(np.asarray(m), cmap="RdBu_r", vmin=-1, vmax=1,
                           interpolation="nearest")
            ax.set_title(title, fontsize=6)
            ax.set_xlabel("channel"); ax.set_ylabel("channel")
        cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
        cb.set_label("zero-lag corr", fontsize=6); cb.ax.tick_params(labelsize=5)
        txt = ""
        for k in ("fro_Rhat_Ks", "fro_Rhat_R", "fro_Ks_R"):
            if k in stats:
                txt += f"{k}={stats[k]:.3f}\n"
        for k in ("tri_r_Rhat_Ks", "tri_r_Rhat_R"):
            if k in stats:
                txt += f"{k}={stats[k]:.3f}\n"
        if txt:
            axes[0].text(0.02, 0.02, txt.strip(), transform=axes[0].transAxes,
                         fontsize=4.6, va="bottom", color=PALETTE["neutral_dark"])
        for ax, lab in zip(axes, "abc"):
            add_panel_label(ax, lab, dx=-0.28)
        return fig
