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
  dose_response    扰动剂量-响应曲线 D̄(r)（多 (A→B) 边 + τ 阈值线；perturb-ec-v1 增量）
  qq_compare       子采样 vs 全集分位数 QQ 对比（代表性行抽查；perturb-ec-v1 增量）
  tau_preview      τ 口径决策预览（地板分离 / 相对口径 D̃ 箱线 / slope 分布；预览非 readout）
  mono_r2          剂量-响应形态诊断：Spearman(D̄,r) 单调性 + 低剂量 OLS R² 分布（B4 增量）
  floor_separation 噪声地板分离：各 r 档 D 分布 vs null 池 + τ 线 + 删失率柱（B4 增量）
  spatial_decay    空间衰减 sanity：D̄(r_max) vs 通道对欧氏距离（散点+分箱 median；B4 增量）
  diag_vs_cross    对角自响应 vs 跨通道量级 + 先验自响应分量解析对照（B4 增量）
  ec_heatmap_grid  EC 矩阵热图网格（多 readout × 多例/组均；B4 增量）
  compare_forest   比较器汇总 forest：readout × 参考法 ρ̄ + bootstrap CI（主比较/上下文分色；B5 增量）
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

    FIG_TYPES = ("trajectories", "psd", "gc_matrix", "ec_matrix", "training_curves",
                 "spatial_corr", "dose_response", "qq_compare", "tau_preview",
                 "mono_r2", "floor_separation", "spatial_decay", "diag_vs_cross",
                 "ec_heatmap_grid", "compare_forest")

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
                       label="generated medoid"
                             + (f" (K={K})" if K is not None else "")
                             + (f", {data['note']}" if data.get("note") else "")),
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

    # ---- ③b EC 参考矩阵对比（baseline-ec-v1；gc_matrix 的标签参数化泛化）----
    def _draw_ec_matrix(self, data: Dict, **cfg) -> mpl.figure.Figure:
        """data: {
            ec_a/ec_b (C,C) 两种观测参考矩阵（行=源/列=靶，非负幅值），
            ec_c (C,C) 可选第三面板（EC_pert 等），
            label_a/label_b/label_c 面板标题，stats dict（散点/文本标注用，可缺省）
        }；热图 Blues（vmin=0），各面板独立色标（量纲不可比，设计如此）。"""
        mats = [(data.get("label_a", "EC a"), data["ec_a"]),
                (data.get("label_b", "EC b"), data["ec_b"])]
        if "ec_c" in data:
            mats.append((data.get("label_c", "EC c"), data["ec_c"]))
        stats = data.get("stats", {})
        fig, axes = plt.subplots(1, len(mats), figsize=(2.3 * len(mats), 2.5))
        for ax, (title, m) in zip(axes, mats):
            m = np.clip(np.asarray(m, dtype=float), 0, None)   # 展示口径：非负幅值
            im = ax.imshow(m, cmap="Blues", vmin=0, vmax=m.max() if m.max() > 0 else 1,
                           interpolation="nearest")
            ax.set_title(title, fontsize=6)
            ax.set_xlabel("target"); ax.set_ylabel("source")
            cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.03)
            cb.ax.tick_params(labelsize=5)
        if stats:
            txt = "  ".join(f"{k}={v:.3f}" for k, v in stats.items() if isinstance(v, (int, float)))
            if txt:
                axes[-1].text(0.02, 0.02, txt, transform=axes[-1].transAxes, fontsize=4.6,
                              va="bottom", color=PALETTE["neutral_dark"])
        for ax, lab in zip(axes, "abc"[:len(axes)]):
            add_panel_label(ax, lab, dx=-0.28)
        fig.tight_layout()
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

    # ---- ⑥ 扰动剂量-响应曲线（perturb-ec-v1 Study II 增量，2026-10-17）----
    def _draw_dose_response(self, data: Dict, **cfg) -> mpl.figure.Figure:
        """data: {
            r_grid (G,) 剂量网格（含 0 点，线性 x 轴），
            curves (n, G) D̄(r) 响应曲线（首列 ≡ r=0 ≡ 0 按 CRN 恒等构造；note 请用
              ASCII——DejaVu 字体无 CJK 字形，中文会渲染为缺字框），
            labels (n,) 曲线标签（如 "A=5→B=17 (strong)"），
            kinds (n,) 可选曲线类别：'self' 用生成红族，其余跨通道路径走蓝/青/紫/灰族
              （缺省全按跨通道配色），
            tau float 主阈值线（虚线，右端直接标注），
            tau_label str（默认 'τ'），
            tau_sens dict {label: value} 敏感性阈值细线（点线，浅灰），
            note str 图内注记（如 smoke 小样本口径，右下角灰字）
        }；单面板，x = 扰动增益 r，y = 归一化 RMS 响应 D̄。"""
        r = np.asarray(data["r_grid"], dtype=float)
        curves = np.asarray(data["curves"], dtype=float)
        n, G = curves.shape
        assert r.shape == (G,), f"r_grid {r.shape} 与曲线 {curves.shape} 末维不符"
        kinds = data.get("kinds") or ["cross"] * n
        cross_cycle = [PALETTE["blue_main"], PALETTE["teal"], PALETTE["violet"],
                       PALETTE["neutral_mid"], PALETTE["blue_secondary"]]
        fig, ax = plt.subplots(figsize=(3.2, 2.4))
        ci = 0
        for i in range(n):
            if kinds[i] == "self":
                col, lw, z = PALETTE["red_strong"], 1.2, 4
            else:
                col, lw, z = cross_cycle[ci % len(cross_cycle)], 0.8, 3
                ci += 1
            ax.plot(r, curves[i], color=col, lw=lw, marker="o", ms=2.2, zorder=z,
                    label=data["labels"][i])
        if "tau" in data and data["tau"] is not None:
            tau = float(data["tau"])
            ax.axhline(tau, color=PALETTE["neutral_dark"], lw=0.8, ls="--", zorder=2)
            ax.text(0.99, tau, f" {data.get('tau_label', 'τ')}",
                    transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                    fontsize=6, color=PALETTE["neutral_dark"])
            for k, v in (data.get("tau_sens") or {}).items():
                ax.axhline(float(v), color=PALETTE["neutral_light"], lw=0.6, ls=":",
                           zorder=1)
                ax.text(0.99, float(v), f" {k}", transform=ax.get_yaxis_transform(),
                        ha="right", va="bottom", fontsize=5,
                        color=PALETTE["neutral_mid"])
        ax.set_xlabel("perturbation gain r")
        ax.set_ylabel(r"response $\bar{D}(r)$  (norm. RMS)")
        ax.legend(loc="upper left")
        if data.get("note"):
            ax.text(0.98, 0.02, data["note"], transform=ax.transAxes, fontsize=4.6,
                    ha="right", va="bottom", color=PALETTE["neutral_mid"])
        fig.tight_layout()
        return fig

    # ---- ⑦ 子采样 vs 全集 QQ 对比（perturb-ec-v1 Study II 增量，2026-10-17 B2）----
    def _draw_qq_compare(self, data: Dict, **cfg) -> mpl.figure.Figure:
        """data: {
            panels: [ {"label": str（轴标签量名）, "title": str 可选,
                       "sub": (n,) 子样本值, "full": (m,) 全集值,
                       "stat_lines": [str, ...] 可选左上注记行}, ... ]（1–3 面板）,
            qgrid (Q,) 分位数网格（缺省 linspace(0.005, 0.995, 199)）,
            note str 可选右下注记
        }；每面板 QQ 散点（x = 全集分位数、y = 子样本分位数）+ y=x 参考虚线。
        代表性行抽查：子样本分位数贴近 y=x ⇒ 同分布。"""
        panels = data["panels"]
        qgrid = np.asarray(data.get("qgrid", np.linspace(0.005, 0.995, 199)), dtype=float)
        fig, axes = plt.subplots(1, len(panels), figsize=(2.9 * len(panels), 2.7),
                                 squeeze=False)
        for ax, p in zip(axes[0], panels):
            qf = np.quantile(np.asarray(p["full"], dtype=float), qgrid)
            qs = np.quantile(np.asarray(p["sub"], dtype=float), qgrid)
            ax.scatter(qf, qs, s=3, alpha=0.5, color=PALETTE["blue_main"], rasterized=True)
            lim = max(float(qf.max()), float(qs.max())) * 1.05
            ax.plot([0, lim], [0, lim], ls="--", lw=0.7, color=PALETTE["neutral_mid"])
            ax.set_xlim(0, lim)
            ax.set_ylim(0, lim)
            ax.set_title(p.get("title", p["label"]), fontsize=6)
            ax.set_xlabel(f"{p['label']}  (full-set quantile)")
            ax.set_ylabel("subsample quantile")
            stats = p.get("stat_lines") or []
            if stats:
                ax.text(0.04, 0.96, "\n".join(stats), transform=ax.transAxes,
                        fontsize=4.6, va="top", color=PALETTE["neutral_dark"])
        for ax, lab in zip(axes[0], "abc"[:len(panels)]):
            add_panel_label(ax, lab, dx=-0.28)
        if data.get("note"):
            fig.text(0.99, 0.005, data["note"], ha="right", va="bottom", fontsize=4.6,
                     color=PALETTE["neutral_mid"])
        fig.tight_layout()
        return fig

    # ---- ⑧ τ 口径决策预览（perturb-ec-v1 Study II 增量，2026-10-17 B2；预览非 readout）----
    def _draw_tau_preview(self, data: Dict, **cfg) -> mpl.figure.Figure:
        """data: {
            null_pool (n,) null 距离池（面板 a 直方图，中性灰）,
            dbar_cross_rmax (m,) 跨通道 D̄(r_max)（面板 a 直方图，蓝）,
            taus dict {label: value} 预注册 τ 横线（面板 a，右端标注）,
            dtil {"r_labels": [str], "boxes": [(m,) ...], "lines": {label: value} τ̃ 横线,
                   "censor_lines": [str]} 面板 b 素材（D̃(r) 分布 + 相对口径删失率注记）,
            slope {"values": (m,), "stats": dict（median/q1/q3 数值线标注用）} 面板 c 素材,
            note str 预览口径注记（右下）
        }；面板 a = 噪声地板分离（密度直方图 log-y + τ 线）；面板 b = 相对口径 D̃(r) 箱线 +
        τ̃ 线；面板 c = 低剂量 slope 分布直方图。**预览图：非 readout 承诺，τ 口径待拍板。**"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7.0, 2.5))
        # (a) 地板分离
        null_pool = np.asarray(data["null_pool"], dtype=float)
        dmax = np.asarray(data["dbar_cross_rmax"], dtype=float)
        bins = np.linspace(0, max(null_pool.max(), dmax.max()) * 1.02, 80)
        ax1.hist(null_pool, bins=bins, density=True, color=PALETTE["neutral_light"],
                 alpha=0.85, label="null pool (indep. pairs)")
        ax1.hist(dmax, bins=bins, density=True, color=PALETTE["blue_main"], alpha=0.6,
                 label=r"cross $\bar{D}(r_{max})$")
        ax1.set_yscale("log")
        ax1.set_xlim(0, bins[-1])                      # 非负量：压掉自动负 tick 余量
        ymin, ymax = ax1.get_ylim()
        for k, v in (data.get("taus") or {}).items():
            ax1.axvline(float(v), color=PALETTE["neutral_dark"], lw=0.7, ls="--")
            ax1.text(float(v), ymin, f"{k}={float(v):.2f}", rotation=90, fontsize=4.6,
                     va="bottom", ha="right", color=PALETTE["neutral_dark"])
        ax1.set_xlabel("normalized distance / response")
        ax1.set_ylabel("density (log)")
        ax1.set_title("noise floor separation", fontsize=6)
        ax1.legend(loc="upper right", fontsize=4.6)
        add_panel_label(ax1, "a", dx=-0.28)
        # (b) 相对口径 D̃(r) 箱线
        dtil = data["dtil"]
        boxes = [np.asarray(b, dtype=float) for b in dtil["boxes"]]
        ax2.boxplot(boxes, widths=0.5, patch_artist=True, showfliers=False,
                    medianprops=dict(color=PALETTE["red_strong"], lw=1.0),
                    boxprops=dict(facecolor=PALETTE["red_soft"], alpha=0.5,
                                  edgecolor=PALETTE["neutral_dark"], lw=0.6),
                    whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))
        ax2.set_xticks(np.arange(1, len(boxes) + 1))
        ax2.set_xticklabels(dtil["r_labels"])
        ax2.set_ylim(0, None)                          # D̃ 非负：下界贴 0
        for k, v in (dtil.get("lines") or {}).items():
            ax2.axhline(float(v), color=PALETTE["neutral_dark"], lw=0.7, ls="--")
            ax2.text(0.99, float(v), f" {k}", transform=ax2.get_yaxis_transform(),
                     ha="right", va="bottom", fontsize=5, color=PALETTE["neutral_dark"])
        cens = dtil.get("censor_lines") or []
        if cens:
            ax2.text(0.03, 0.97, "\n".join(cens), transform=ax2.transAxes, fontsize=4.6,
                     va="top", color=PALETTE["neutral_dark"])
        ax2.set_ylabel(r"relative response $\tilde{D}(r)$")
        ax2.set_xlabel("perturbation gain r")
        ax2.set_title("relative caliber (vs self-response)", fontsize=6)
        add_panel_label(ax2, "b", dx=-0.22)
        # (c) slope 分布
        sl = data["slope"]
        vals = np.asarray(sl["values"], dtype=float)
        ax3.hist(vals, bins=60, density=True, color=PALETTE["teal"], alpha=0.7)
        st = sl.get("stats", {})
        for key, col in (("median", PALETTE["neutral_dark"]),
                         ("q1", PALETTE["neutral_light"]), ("q3", PALETTE["neutral_light"])):
            if key in st:
                ax3.axvline(float(st[key]), color=col, lw=0.8 if key == "median" else 0.6,
                            ls="-" if key == "median" else ":")
        if "median" in st:
            ax3.text(0.97, 0.95, f"median={st['median']:.4f}", transform=ax3.transAxes,
                     fontsize=4.6, ha="right", va="top", color=PALETTE["neutral_dark"])
        ax3.set_xlabel("low-dose slope (OLS, r$\\leq$0.08)")
        ax3.set_ylabel("density")
        ax3.set_title("slope readout (tau-free)", fontsize=6)
        add_panel_label(ax3, "c", dx=-0.22)
        if data.get("note"):
            fig.text(0.99, 0.005, data["note"], ha="right", va="bottom", fontsize=4.6,
                     color=PALETTE["neutral_mid"])
        fig.tight_layout()
        return fig

    # ---- ⑨ 剂量-响应形态诊断（perturb-ec-v1 B4 增量，2026-10-17，design §6-2）----
    def _draw_mono_r2(self, data: Dict, **cfg) -> mpl.figure.Figure:
        """data: {
            spearman (n,) 跨例跨边 Spearman(D̄, r)（9 点含 r=0；预期全 1.0——照登）,
            r2 (n,) 低剂量段 OLS R²（r ∈ {0,0.01,0.02,0.04,0.08}）,
            mono_frac float 完全单调（8 档非降）边比例（面板 a 注记）,
            n_subjects int（注记）, note str 可选
        }；面板 a = Spearman 分布（0.9–1.0 限幅直方图）；面板 b = 低剂量 R² 分布。"""
        sp = np.asarray(data["spearman"], dtype=float)
        r2 = np.asarray(data["r2"], dtype=float)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.6, 2.4))
        ax1.hist(np.clip(sp, 0.9, 1.0), bins=np.linspace(0.9, 1.0, 41), density=True,
                 color=PALETTE["teal"], alpha=0.75)
        ax1.axvline(float(np.median(sp)), color=PALETTE["neutral_dark"], lw=0.8, ls="--")
        ax1.set_xlim(0.9, 1.0)
        ax1.set_xlabel(r"Spearman($\bar{D}$, r)  (clipped to [0.9, 1])")
        ax1.set_ylabel("density")
        ax1.set_title("monotonicity (all edges, all subjects)", fontsize=6)
        mf = data.get("mono_frac")
        txt = f"median={np.median(sp):.3f}"
        if mf is not None:
            txt += f"\nfully non-decreasing: {mf * 100:.1f}%"
        if data.get("n_subjects") is not None:
            txt += f"\nn={sp.size} edges ({data['n_subjects']} subjects)"
        ax1.text(0.03, 0.95, txt, transform=ax1.transAxes, fontsize=4.6, va="top",
                 color=PALETTE["neutral_dark"])
        add_panel_label(ax1, "a", dx=-0.24)
        ax2.hist(r2, bins=60, density=True, color=PALETTE["blue_main"], alpha=0.7)
        ax2.axvline(float(np.median(r2)), color=PALETTE["neutral_dark"], lw=0.8, ls="--")
        ax2.set_xlabel("low-dose OLS R$^2$  (r$\\leq$0.08)")
        ax2.set_ylabel("density")
        ax2.set_title("low-dose linearity (all edges)", fontsize=6)
        ax2.text(0.03, 0.95, f"median={np.median(r2):.4f}\nq10={np.quantile(r2, 0.10):.4f}",
                 transform=ax2.transAxes, fontsize=4.6, va="top",
                 color=PALETTE["neutral_dark"])
        add_panel_label(ax2, "b", dx=-0.24)
        if data.get("note"):
            fig.text(0.99, 0.005, data["note"], ha="right", va="bottom", fontsize=4.6,
                     color=PALETTE["neutral_mid"])
        fig.tight_layout()
        return fig

    # ---- ⑩ 噪声地板分离（perturb-ec-v1 B4 增量，2026-10-17，design §6-3）----
    def _draw_floor_separation(self, data: Dict, **cfg) -> mpl.figure.Figure:
        """data: {
            r_labels (G-1,) 扰动档标签（r=0 不采样）,
            d_boxes [(m,) ...] 各 r 档跨通道 D̄ 分布（跨例池）,
            null_pool (n,) null 距离池（面板 a 首列箱线；自动子采样至 ~4 万点）,
            taus dict {label: value} 绝对 τ 横线（面板 a 虚线，右端标注）,
            censor_abs dict {label: rate}（面板 b 绝对口径删失率，中性灰族）,
            censor_rel dict {label: rate}（面板 b 相对口径 τ̃ 删失率，蓝族）,
            ylog bool 面板 a 对数 y（默认 True）, note str 可选
        }；面板 a = null vs 各 r 档 D 分布（箱线，log-y）+ 绝对 τ 三线；面板 b = 删失率柱
        （绝对 q90/95/99 与相对 τ̃ 0.25/0.50 两族并列）。"""
        r_labels = data["r_labels"]
        null_pool = np.asarray(data["null_pool"], dtype=float)
        step = max(1, null_pool.size // 40000)                    # 箱线样本上限（分位数级无失真）
        boxes = [null_pool[::step]] + [np.asarray(b, dtype=float) for b in data["d_boxes"]]
        labels = ["null"] + list(r_labels)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.2, 2.6),
                                       gridspec_kw={"width_ratios": [1.6, 1.0]})
        bp = ax1.boxplot(boxes, widths=0.55, patch_artist=True, showfliers=False,
                         medianprops=dict(color=PALETTE["red_strong"], lw=0.9),
                         whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(PALETTE["neutral_light"] if i == 0 else PALETTE["blue_main"])
            patch.set_alpha(0.85 if i == 0 else 0.55)
            patch.set_edgecolor(PALETTE["neutral_dark"])
            patch.set_lw(0.6)
        if data.get("ylog", True):
            ax1.set_yscale("log")
        for k, v in (data.get("taus") or {}).items():
            ax1.axhline(float(v), color=PALETTE["neutral_dark"], lw=0.7, ls="--", zorder=1)
        if data.get("taus"):
            tau_txt = "abs tau:  " + "  ".join(f"{k}={float(v):.2f}"
                                               for k, v in data["taus"].items())
            ax1.text(0.02, 0.96, tau_txt, transform=ax1.transAxes, fontsize=4.6, va="top",
                     color=PALETTE["neutral_dark"])
        ax1.set_xticks(np.arange(1, len(boxes) + 1))
        ax1.set_xticklabels(labels, rotation=45, ha="right")
        ax1.set_ylabel(r"normalized distance / $\bar{D}(r)$  (log)")
        ax1.set_title("noise floor vs response distributions", fontsize=6)
        add_panel_label(ax1, "a", dx=-0.30)
        bars, cols, labs = [], [], []
        for k, v in (data.get("censor_abs") or {}).items():
            bars.append(float(v)); cols.append(PALETTE["neutral_mid"]); labs.append(k)
        for k, v in (data.get("censor_rel") or {}).items():
            bars.append(float(v)); cols.append(PALETTE["blue_main"]); labs.append(k)
        xs = np.arange(len(bars))
        ax2.bar(xs, bars, width=0.62, color=cols, alpha=0.85)
        for x, v in zip(xs, bars):
            ax2.text(x, min(v, 1.0) + 0.015, f"{v * 100:.1f}%", ha="center", va="bottom",
                     fontsize=4.6, color=PALETTE["neutral_dark"])
        ax2.set_xticks(xs)
        ax2.set_xticklabels(labs, rotation=45, ha="right")
        ax2.set_ylim(0, 1.12)
        ax2.set_ylabel("right-censoring rate")
        ax2.set_title("censoring by caliber (grey=abs, blue=rel)", fontsize=6)
        add_panel_label(ax2, "b", dx=-0.42)
        if data.get("note"):
            fig.text(0.99, 0.005, data["note"], ha="right", va="bottom", fontsize=4.6,
                     color=PALETTE["neutral_mid"])
        fig.tight_layout()
        return fig

    # ---- ⑪ 空间衰减 sanity（perturb-ec-v1 B4 增量，2026-10-17，design §6-4）----
    def _draw_spatial_decay(self, data: Dict, **cfg) -> mpl.figure.Figure:
        """data: {
            dist (m,) 通道对欧氏距离 mm（跨例池，与 dmax 对齐；散点请先子采样）,
            dmax (m,) 对应 D̄(r_max),
            bins (B,) 分箱边界（缺省 12 箱线性）,
            per_subject [{"sid": int, "dist": (m,), "dmax": (m,)} ...] 面板 b 细线（可缺省）,
            rho float 全池 Spearman ρ（注记；定性 sanity 不作判据）, note str 可选
        }；面板 a = 散点 + 分箱 median（粗线）+ IQR 带；面板 b = 逐例分箱 median 细线 + 组池粗线。"""
        dist = np.asarray(data["dist"], dtype=float)
        dmax = np.asarray(data["dmax"], dtype=float)
        bins = np.asarray(data.get("bins", np.linspace(dist.min(), dist.max(), 13)), dtype=float)
        centers = 0.5 * (bins[:-1] + bins[1:])

        def _binned_q(vals, q, d_ref):
            w_ = np.digitize(d_ref, bins) - 1
            ok = (d_ref >= bins[0]) & (d_ref < bins[-1])
            return np.array([np.quantile(vals[(w_ == b) & ok], q)
                             if ((w_ == b) & ok).sum() >= 5 else np.nan
                             for b in range(len(centers))])
        med = _binned_q(dmax, 0.50, dist)
        q1 = _binned_q(dmax, 0.25, dist)
        q3 = _binned_q(dmax, 0.75, dist)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.8, 2.5))
        ax1.scatter(dist, dmax, s=1.2, alpha=0.12, color=PALETTE["blue_main"],
                    rasterized=True, edgecolors="none")
        ax1.plot(centers, med, color=PALETTE["red_strong"], lw=1.2, zorder=5,
                 label="binned median")
        ax1.fill_between(centers, q1, q3, color=PALETTE["red_strong"], alpha=0.15, lw=0,
                         zorder=4, label="IQR")
        rho = data.get("rho")
        if rho is not None:
            ax1.text(0.03, 0.95, f"Spearman rho={rho:.3f}", transform=ax1.transAxes,
                     fontsize=4.6, va="top", color=PALETTE["neutral_dark"])
        ax1.set_xlabel("channel-pair Euclidean distance (mm)")
        ax1.set_ylabel(r"$\bar{D}(r_{max})$")
        ax1.set_title("spatial decay (pooled, qualitative)", fontsize=6)
        ax1.legend(loc="upper right", fontsize=4.6)
        add_panel_label(ax1, "a", dx=-0.24)
        for ps in data.get("per_subject", []):
            m_ = _binned_q(np.asarray(ps["dmax"], float), 0.50, np.asarray(ps["dist"], float))
            ax2.plot(centers, m_, lw=0.6, alpha=0.6, color=PALETTE["neutral_mid"])
        ax2.plot(centers, med, lw=1.4, color=PALETTE["red_strong"], label="pooled median")
        ax2.set_xlabel("channel-pair Euclidean distance (mm)")
        ax2.set_ylabel(r"binned median $\bar{D}(r_{max})$")
        ax2.set_title("per-subject binned medians", fontsize=6)
        ax2.legend(loc="upper right", fontsize=4.6)
        add_panel_label(ax2, "b", dx=-0.24)
        if data.get("note"):
            fig.text(0.99, 0.005, data["note"], ha="right", va="bottom", fontsize=4.6,
                     color=PALETTE["neutral_mid"])
        fig.tight_layout()
        return fig

    # ---- ⑫ 对角 vs 跨通道 + 先验分量对照（perturb-ec-v1 B4 增量，2026-10-17，design §6-5）----
    def _draw_diag_vs_cross(self, data: Dict, **cfg) -> mpl.figure.Figure:
        """data: {
            r_grid (G,) 剂量网格（含 0）,
            self_med/self_q1/self_q3 (G,) 对角自响应 D̄ 中位/IQR（跨例池）,
            cross_med/cross_q1/cross_q3 (G,) 跨通道 D̄ 中位/IQR,
            prior_med/prior_q1/prior_q3 (G,) 先验自响应分量（解析重建）中位/IQR（可缺省）,
            share {"r_labels": [str], "boxes": [(n,)...]} 先验占比分布素材
              （面板 b；share = D_prior/D_obs-self，1.0 虚线 = 先验全额解释）,
            note str 可选
        }；面板 a = 三族中位曲线 + IQR 带（self 红 / cross 蓝 / prior 灰虚线）；面板 b =
        先验分量占比箱线 + 1.0 参考线 + 逐档 median 标注。"""
        r = np.asarray(data["r_grid"], dtype=float)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.0, 2.5))
        for key, col, lab in (("cross", PALETTE["blue_main"], "cross (A$\\neq$B)"),
                              ("self", PALETTE["red_strong"], "self (A=B)")):
            med = np.asarray(data[f"{key}_med"], dtype=float)
            ax1.plot(r, med, color=col, lw=1.1, marker="o", ms=2.2, label=lab)
            ax1.fill_between(r, np.asarray(data[f"{key}_q1"], dtype=float),
                             np.asarray(data[f"{key}_q3"], dtype=float),
                             color=col, alpha=0.15, lw=0)
        if "prior_med" in data:
            ax1.plot(r, np.asarray(data["prior_med"], dtype=float), lw=1.0, ls="--",
                     color=PALETTE["neutral_dark"], marker="s", ms=1.8,
                     label="prior self component (analytic)")
            ax1.fill_between(r, np.asarray(data["prior_q1"], dtype=float),
                             np.asarray(data["prior_q3"], dtype=float),
                             color=PALETTE["neutral_dark"], alpha=0.10, lw=0)
        ax1.set_xlabel("perturbation gain r")
        ax1.set_ylabel(r"$\bar{D}(r)$  (norm. RMS, median [IQR])")
        ax1.set_title("diagonal vs cross-channel magnitude", fontsize=6)
        ax1.legend(loc="upper left", fontsize=4.6)
        add_panel_label(ax1, "a", dx=-0.24)
        sh = data.get("share")
        if sh:
            boxes = [np.asarray(b, dtype=float) for b in sh["boxes"]]
            ax2.boxplot(boxes, widths=0.5, patch_artist=True, showfliers=False,
                        medianprops=dict(color=PALETTE["red_strong"], lw=1.0),
                        boxprops=dict(facecolor=PALETTE["red_soft"], alpha=0.5,
                                      edgecolor=PALETTE["neutral_dark"], lw=0.6),
                        whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))
            ax2.set_xticks(np.arange(1, len(boxes) + 1))
            ax2.set_xticklabels(sh["r_labels"])
            ax2.axhline(1.0, color=PALETTE["neutral_dark"], lw=0.7, ls="--")
            ax2.text(0.99, 1.0, " prior explains all", transform=ax2.get_yaxis_transform(),
                     ha="right", va="bottom", fontsize=4.6, color=PALETTE["neutral_dark"])
            med = np.array([np.median(b) for b in boxes])
            ax2.plot(np.arange(1, len(boxes) + 1), med, lw=0.9,
                     color=PALETTE["red_strong"], marker="o", ms=2.0)
            for x, v in zip(np.arange(1, len(boxes) + 1), med):
                ax2.text(x, v, f" {v:.2f}", ha="left", va="bottom", fontsize=4.6,
                         color=PALETTE["neutral_dark"])
            ax2.set_ylabel("prior share  $\\bar{D}_{prior}/\\bar{D}_{self}$")
            ax2.set_xlabel("perturbation gain r")
            ax2.set_title("prior self-response share", fontsize=6)
        add_panel_label(ax2, "b", dx=-0.30)
        if data.get("note"):
            fig.text(0.99, 0.005, data["note"], ha="right", va="bottom", fontsize=4.6,
                     color=PALETTE["neutral_mid"])
        fig.tight_layout()
        return fig

    # ---- ⑬ EC 矩阵热图网格（perturb-ec-v1 B4 增量，2026-10-17，design §6-6）----
    def _draw_ec_heatmap_grid(self, data: Dict, **cfg) -> mpl.figure.Figure:
        """data: {
            rows: [{"label": str（行标题，readout 名）, "mats": [(C,C) ...]} ...],
            col_labels [str ...]（列标题，如 group-mean / subject id）,
            unit str 色标名（缺省 'strength'）, note str 可选
        }；每行共享 vmax（行内例间可比），Blues 非负色标，行级 colorbar；行=源、列=靶。"""
        rows, col_labels = data["rows"], data["col_labels"]
        n_r, n_c = len(rows), len(col_labels)
        fig, axes = plt.subplots(n_r, n_c, figsize=(2.1 * n_c, 1.9 * n_r + 0.3),
                                 squeeze=False)
        for ri, row in enumerate(rows):
            mats = [np.clip(np.asarray(m, dtype=float), 0, None) for m in row["mats"]]
            vmax = max((float(m.max()) if m.size else 0.0) for m in mats) or 1.0
            for ci, m in enumerate(mats):
                ax = axes[ri][ci]
                im = ax.imshow(m, cmap="Blues", vmin=0, vmax=vmax, interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                if ri == 0:
                    ax.set_title(col_labels[ci], fontsize=5)
                if ci == 0:
                    ax.set_ylabel(f"{row['label']}\n(source rows)", fontsize=5)
                if ci == n_c - 1:
                    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.03)
                    cb.ax.tick_params(labelsize=4)
                    cb.set_label(data.get("unit", "strength"), fontsize=4.6)
        for ci in range(n_c):
            axes[0][ci].set_xlabel("target cols", fontsize=5)
        if data.get("note"):
            fig.text(0.99, 0.005, data["note"], ha="right", va="bottom", fontsize=4.6,
                     color=PALETTE["neutral_mid"])
        fig.tight_layout()
        return fig

    # ---- ⑭ 比较器汇总 forest（perturb-ec-v1 B5 增量，2026-10-17，design §5）----
    def _draw_compare_forest(self, data: Dict, **cfg) -> mpl.figure.Figure:
        """data: {
            rows: [{"readout": str, "method": str, "rho": float, "ci": (lo, hi),
                    "rho_raw": float 可选（未偏 ρ 上下文，空心方块）,
                    "main": bool（主比较 Holm 族行 = True）,
                    "holm_p": float 可选（仅主族行标注）,
                    "ceiling_ratio": float 可选（行尾注记）} ...]
              （y 轴自上而下按给定序；建议主 readout 置顶）,
            n_subjects int, ceilings dict {method: value} 可选（注记）,
            note str 可选（右下，ASCII——DejaVu 无 CJK）
        }；水平 forest：x = 逐被试 ρ̄（点）+ 受试者 bootstrap 95% CI（线），ρ=0 参考虚线；
        主比较行蓝、上下文行中性灰；行尾注记 ρ̄/CI/Holm p（主族）/ceiling 比值。"""
        rows = data["rows"]
        n = len(rows)
        fig, ax = plt.subplots(figsize=(6.4, 0.62 * n + 1.1))
        ys = np.arange(n)[::-1]                       # 首行置顶
        xall = [0.0]
        for row, y in zip(rows, ys):
            lo, hi = row["ci"]
            main = bool(row.get("main"))
            col = PALETTE["blue_main"] if main else PALETTE["neutral_mid"]
            ax.plot([lo, hi], [y, y], color=col, lw=1.2, solid_capstyle="butt", zorder=3)
            ax.plot([lo, lo], [y - 0.14, y + 0.14], color=col, lw=1.0, zorder=3)
            ax.plot([hi, hi], [y - 0.14, y + 0.14], color=col, lw=1.0, zorder=3)
            ax.plot(row["rho"], y, marker="o", ms=4.2, color=col, zorder=4,
                    markeredgecolor="white", markeredgewidth=0.5)
            if row.get("rho_raw") is not None:
                ax.plot(row["rho_raw"], y, marker="s", ms=3.0, linestyle="none",
                        markerfacecolor="none", markeredgecolor=col, markeredgewidth=0.8,
                        zorder=4)
            xall += [row["rho"], lo, hi]
        ax.axvline(0.0, color=PALETTE["neutral_dark"], lw=0.7, ls="--", zorder=1)
        pad = max(0.05, 0.08 * (max(xall) - min(xall)))
        ax.set_xlim(min(xall) - pad, max(xall) + pad)
        ax.set_ylim(-0.7, n - 0.3)
        labels = []
        for row in rows:
            lab = f"{row['readout']} x {row['method'].upper()}"
            lab += "  [MAIN]" if row.get("main") else "  [context]"
            labels.append(lab)
        ax.set_yticks(ys)
        ax.set_yticklabels(labels)
        for row, y in zip(rows, ys):
            lo, hi = row["ci"]
            txt = f"rho={row['rho']:+.3f}  CI[{lo:+.3f},{hi:+.3f}]"
            if row.get("holm_p") is not None:
                txt += f"  Holm p={row['holm_p']:.3f}"
            if row.get("ceiling_ratio") is not None:
                txt += f"  ceil_ratio={row['ceiling_ratio']:.3f}"
            ax.text(max(xall) + pad, y, txt, va="center", ha="left", fontsize=4.6,
                    color=PALETTE["neutral_dark"])
        ax.set_xlabel(r"per-subject distance-partial Spearman $\bar\rho$  (mean + bootstrap 95% CI)")
        ax.set_title(f"EC_pert vs observational refs "
                     f"(n={data.get('n_subjects', '?')} subjects)", fontsize=6)
        handles = [
            plt.Line2D([], [], color=PALETTE["blue_main"], marker="o", ms=4, lw=1.2,
                       label="main comparison (Holm family)"),
            plt.Line2D([], [], color=PALETTE["neutral_mid"], marker="o", ms=4, lw=1.2,
                       label="context readout (not in family)"),
            plt.Line2D([], [], color=PALETTE["neutral_dark"], marker="s", ms=3, lw=0,
                       markerfacecolor="none", label="unpartial rho (context)"),
        ]
        ax.legend(handles=handles, loc="lower left", fontsize=4.6)
        if data.get("note"):
            fig.text(0.99, 0.005, data["note"], ha="right", va="bottom", fontsize=4.6,
                     color=PALETTE["neutral_mid"])
        fig.tight_layout()
        return fig
