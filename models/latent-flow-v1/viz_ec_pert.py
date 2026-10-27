#!/usr/bin/env python3
"""viz_ec_pert.py — EC_pert v4 三图可视化（任务 10-24-exitgnn-ec-viz，nature-figure 契约）.

数据只读：归档任务 10-24-exitgnn-ec-unbiased 的 recompute_v4（conn v4 =
−(ln σ*+0.5·ln2πe)+5/7，8 例 128×128，行=源/列=靶，全正）+ refs GC v1 档
（8 例折叠组均值；2026-10-27 线配置固定：v2 1-125hz 对比项已取消）。零数值推断
重算（组均值/差值/相关仅为图面注记，且对 Run 003 在档组均值 ρ 做 assert 对拍）。

图契约（figure contract）：
  图 1 ec_subjects_v4 — 逐被试 2×4 热图（行 1 MDD / 行 2 HC，共享稳健色标 =
    offdiag P1–P99；对角自响应钳在 σ_min ⇒ 饱和，注记随图）。
    结论：「EC_pert 逐被试呈靶主效应主导的列条纹结构，个体差异在强度非拓扑；
    MDD/HC 无肉眼分组差异」。
  图 2 ec_group_vs_gc — 组均值二联：EC_pert v4 × GC v1（量纲不同，
    各自色标显式注明；面板题注入档组均值 Spearman ρ）。
    结论：「EC_pert 组均值结构与观测 GC v1 参考结构不对应（描述性）」。
  图 3 ec_mdd_hc — MDD 均值 × HC 均值（共享稳健色标）+ MDD−HC 差异（发散
    色标，0 居中）；n=4/组描述性注记。
    结论：「组间形态高度相似，差异面板无系统结构」。

后端 = Python matplotlib（已存偏好，backend 独占）；导出 PNG(300dpi)+PDF，
多面板 render-time alignment gate（audit_panel_alignment）+ 源码 preflight
（validate_figure.py）+ PDF collision audit（audit_figure_collisions.py，
QA 脚本侧执行）。

Run:
  /root/miniconda3/bin/python models/latent-flow-v1/viz_ec_pert.py
"""
from __future__ import annotations

import os

# --- BLAS 守卫：numpy 导入前（npi 环境契约）---
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy import stats as sps  # noqa: E402

import matplotlib as mpl  # noqa: E402

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.axes_grid1 import make_axes_locatable  # noqa: E402

SKILL_SCRIPTS = "/root/npi/.zcode/skills/nature-figure/scripts"
sys.path.insert(0, SKILL_SCRIPTS)
from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402

RECOMPUTE_DIR = Path("/root/npi/outputs/10-24-exitgnn-ec-unbiased/recompute_v4")
# 2026-10-27 线配置固定：v2 1-125hz 对比项取消（refs 物理文件已删），仅存 v1 档
REFS_DIRS = [
    ("GC v1 · 1-45 Hz, order 8 fixed",
     Path("/root/npi/refs/granger_bivariate_spectral/"
          "mild40-trial500-zscore-fs250-fixed8-f1-45-favg")),
]
OUT_DIR = Path("/root/npi/outputs/10-24-exitgnn-ec-viz")
# Run 003 在档组均值矩阵 Spearman（gccorr_dual_summary.group_mean）——图面注记对拍锚
ANCHOR_GROUPMEAN_RHO = {"v1": +0.1934}
OFF = ~np.eye(128, dtype=bool)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
})


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--main-dir", type=str, default=str(RECOMPUTE_DIR))
    p.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    return p.parse_args(argv)


def load_subjects(main_dir: Path) -> list[dict]:
    rows = []
    for f in sorted(main_dir.glob("subject_*.json")):
        d = json.load(open(f))
        rows.append({
            "sid": d["subject_file"], "group": d["group"],
            "conn": np.array(d["connectivity_matrix"], dtype=np.float64)})
    assert len(rows) == 8, f"expected 8 subjects, got {len(rows)}"
    assert [r["group"] for r in rows] == ["MDD"] * 4 + ["HC"] * 4, \
        "行序契约：前 4 MDD / 后 4 HC（fold0 test 顺序）"
    return rows


def load_refs_means(rows: list[dict]) -> list[np.ndarray]:
    """refs 档（v1）按本 8 例 sid 折叠组均值。"""
    means = []
    for _name, d in REFS_DIRS:
        stack = []
        for r in rows:
            sid = int(r["sid"][1:])
            z = np.load(d / f"refs_s{sid}.npz", allow_pickle=True)
            stack.append(z["ec"].astype(np.float64))
        means.append(np.stack(stack).mean(axis=0))
    return means


def robust_range(mats: list[np.ndarray], q_lo=1.0, q_hi=99.0) -> tuple[float, float]:
    """offdiag 合并值的百分位色标域（对角钳边饱和，防其主导色标）。"""
    v = np.concatenate([m[OFF] for m in mats])
    return float(np.percentile(v, q_lo)), float(np.percentile(v, q_hi))


def _heatmap(ax, m, *, vmin, vmax, cmap="viridis"):
    im = ax.imshow(m, cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest", aspect="equal", origin="upper")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(0.6)
    return im


def _save(fig, name: str, out_dir: Path) -> None:
    """alignment gate + PNG(300dpi)+PDF 导出（内部研究图：PNG 交付 + PDF 供 collision audit）。"""
    require_matplotlib_panel_alignment(
        fig, json_out=str(out_dir / f"{name}.alignment.json"),
        overlay_svg=str(out_dir / f"{name}.alignment.svg"),
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    fig.savefig(out_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_subjects(rows, out_dir) -> str:
    vmin, vmax = robust_range([r["conn"] for r in rows])
    fig, axes = plt.subplots(2, 4, figsize=(9.2, 5.2), constrained_layout=True)
    for i, (r, ax) in enumerate(zip(rows, axes.ravel())):
        im = _heatmap(ax, r["conn"], vmin=vmin, vmax=vmax)
        ax.set_title(f"{r['sid']} · {r['group']}", fontsize=7, pad=2.5)
        if i >= 4:
            ax.set_xlabel("Target E1–E128 (col)", fontsize=6)
        if i % 4 == 0:
            ax.set_ylabel("Source E1–E128 (row)", fontsize=6)
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.015)
    cb.set_label("EC_pert v4  (shared scale: offdiag P1–P99 = "
                 f"{vmin:.2f}–{vmax:.2f}; diagonal self-response clamped at "
                 "σ_min ⇒ saturated)", fontsize=6)
    cb.ax.tick_params(labelsize=6)
    fig.suptitle("EC_pert v4 per subject — latent-flow exit-gnn surrogate, unbiased "
                 "global-τ (q=0.5); row = source, col = target", fontsize=8)
    name = "ec_subjects_v4"
    _save(fig, name, out_dir)
    return name


def _panel_with_cbar(fig, ax, im, letter, title, sub=6):
    """多联图统一面板构造：主轴 + 右侧比例色标（等宽 ⇒ gutter 逐位一致）。
    面板字母并入标题串（mathtext 粗体）——独立 text 对象会与长标题 bbox 重叠
    （collision audit FAIL 实测）。"""
    ax.set_title(f"$\\bf{{{letter}}}$  {title}", fontsize=7, pad=3)
    ax.set_xlabel("Target E1–E128 (col)", fontsize=sub)
    ax.set_ylabel("Source E1–E128 (row)", fontsize=sub)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4.5%", pad=0.12)
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_linewidth(0.6)


def fig_group_vs_gc(rows, ref_means, out_dir) -> str:
    ec_mean = np.stack([r["conn"] for r in rows]).mean(axis=0)
    # 图面注记的组均值 ρ 与 Run 003 在档值对拍（防图档漂移；仅 v1——2026-10-27
    # 线配置固定取消 v2 对比项）
    rhos = [float(sps.spearmanr(ec_mean[OFF], m[OFF]).statistic) for m in ref_means]
    assert len(rhos) == 1 and abs(rhos[0] - ANCHOR_GROUPMEAN_RHO["v1"]) < 5e-3, rhos

    panels = ([("a", "EC_pert v4 · group mean (n=8)", ec_mean,
                robust_range([ec_mean]), None)]
              + [(chr(ord("a") + 1 + k), f"{title} · group mean (n=8)", m,
                  robust_range([m]), rhos[k])
                 for k, (title, _d) in enumerate(REFS_DIRS) for m in [ref_means[k]]])
    fig, axes = plt.subplots(1, len(panels), figsize=(7.4, 3.9))
    fig.subplots_adjust(left=0.065, right=0.985, top=0.84, bottom=0.11, wspace=0.55)
    for (letter, title, m, (vmin, vmax), rho), ax in zip(panels, axes):
        im = _heatmap(ax, m, vmin=vmin, vmax=vmax)
        _panel_with_cbar(fig, ax, im, letter,
                         title + (f"  |  ρ={rho:+.2f}" if rho is not None else ""))
    fig.suptitle("Group-mean structure: EC_pert vs observational GC references "
                 "(ρ = Spearman offdiag vs EC; scales NOT shared across panels)",
                 fontsize=8)
    name = "ec_group_vs_gc"
    _save(fig, name, out_dir)
    return name


def fig_mdd_hc(rows, out_dir) -> str:
    mdd = np.stack([r["conn"] for r in rows if r["group"] == "MDD"]).mean(axis=0)
    hc = np.stack([r["conn"] for r in rows if r["group"] == "HC"]).mean(axis=0)
    diff = mdd - hc
    vmin, vmax = robust_range([mdd, hc])
    dmax = float(np.percentile(np.abs(diff), 99))
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.9))
    fig.subplots_adjust(left=0.065, right=0.985, top=0.84, bottom=0.11, wspace=0.55)
    for (letter, title, m, vv, cm) in (
            ("a", "MDD mean (n=4, descriptive)", mdd, (vmin, vmax), "viridis"),
            ("b", "HC mean (n=4, descriptive)", hc, (vmin, vmax), "viridis"),
            ("c", "MDD − HC", diff, (-dmax, dmax), "RdBu_r")):
        ax = axes[ord(letter) - ord("a")]
        im = _heatmap(ax, m, vmin=vv[0], vmax=vv[1], cmap=cm)
        _panel_with_cbar(fig, ax, im, letter, title)
    fig.suptitle("Group structure MDD vs HC — a/b share scale "
                 f"(offdiag P1–P99 = {vmin:.2f}–{vmax:.2f}); c diverging, "
                 f"±{dmax:.2f} (P99 |diff|), 0-centred. c column structure = "
                 "target-sensitivity sampling noise at n=4/group, not group "
                 "signal (cross-group pairwise ρ 0.66 ≈ within-group 0.65)",
                 fontsize=8)
    name = "ec_mdd_hc"
    _save(fig, name, out_dir)
    return name


def main() -> int:
    a = parse_args()
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = load_subjects(Path(a.main_dir))
    ref_means = load_refs_means(rows)
    for make in (lambda: fig_subjects(rows, out_dir),
                 lambda: fig_group_vs_gc(rows, ref_means, out_dir),
                 lambda: fig_mdd_hc(rows, out_dir)):
        name = make()
        print(f"[viz] wrote {name}.png/.pdf (+alignment.json)", flush=True)
    print(f"[viz] done in {time.time() - t0:.1f}s -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
