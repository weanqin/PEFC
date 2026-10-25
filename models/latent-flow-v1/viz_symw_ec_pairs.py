#!/usr/bin/env python3
"""viz_symw_ec_pairs.py — EC_pert(conn v4) × GC-EC(v1 refs) 逐被试并排热力图 + 组均值两版.

任务 10-25-symw-ec-viz（nature-figure 契约，后端 = Python matplotlib 已存偏好独占）。
数据只读：任务 10-25-symw-ec-unbiased 归档 recompute_v4 的 EC_pert（conn v4，8 例
128×128，行=源/列=靶）+ refs GC v1（mild40 1–45 Hz order 8 fixed faverage）；
ρ 注记直接读 gccorr_dual_summary.json per-subject spearman_raw_aligned（图面值
与矩阵重算值 assert 对拍防档漂移）。零数值推断重算。

图契约（figure contract）：
  个体图 ×8 subj_s{id}_{group} — 1×2 并排（左 EC_pert / 右 GC-EC）：
    颜色 = offdiag 矩阵内百分位秩（rankdata/(N-1) → [0,1]，两面板同 cmap 同色标），
    对角双侧掩蔽（白 + 细纹 hatch）；单一共享 colorbar；面板标题带原始值域；
    图内角注该被试 Spearman ρ(aligned)；suptitle 含 128×128 与任务名。
    结论：「同一被试内两法结构不对应（ρ 注记），秩归一使两法纹理可直接对读」。
  group_means_rank — 8 例秩图逐元素均值的同规格 1×2 + group ρ̄(±SD) 注记。
    结论：「组水平上 EC_pert 的列条纹结构 vs GC-EC 的稀疏尖峰结构仍不对应」。
  group_means_abs（补充，量级诚实）— 8 例原始值均值 1×2，各测自身
    vmax = pooled offdiag P99.5、独立 colorbar 标真实数值。
    结论：「两法绝对量级差 ~13×（EC [0.03, 3.90] vs GC [0, 0.32]）秩版刻意抹平
    量级，此版照登」。

Run:
  /root/miniconda3/bin/python models/latent-flow-v1/viz_symw_ec_pairs.py            # 全量
  /root/miniconda3/bin/python models/latent-flow-v1/viz_symw_ec_pairs.py --subjects 1  # 冒烟
"""
from __future__ import annotations

import os

# --- BLAS 守卫：numpy 导入前（npi 环境契约；纯 CPU）---
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

RECOMPUTE_DIR = Path("/root/npi/outputs/10-25-symw-ec-unbiased/recompute_v4")
GC_REF_DIR = Path("/root/npi/refs/granger_bivariate_spectral/"
                  "mild40-trial500-zscore-fs250-fixed8-f1-45-favg")
GEOM_PT = Path("/root/autodl-tmp/Mild Depression/geometry_mild.pt")
RHO_SUMMARY = RECOMPUTE_DIR / "gccorr_dual_summary.json"
OUT_DIR = Path("/root/npi/outputs/10-25-symw-ec-viz")
TASK_NAME = "10-25-symw-ec-unbiased"
ALL_SIDS = [1, 5, 7, 10, 21, 25, 27, 30]
N = 128
OFF = ~np.eye(N, dtype=bool)
TICK_IDX = list(range(0, N, 16))  # 0,16,...,112 → E1,E17,...,E113（每 16 通道稀疏刻度）

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
})

CMAP = mpl.colormaps["viridis"].copy()  # 感知均匀序（克制剂配色）
CMAP.set_bad(color="white")             # 对角 NaN 掩蔽 → 白

# 对角掩蔽纹理：白 + 细纹。raster RGBA overlay（fill-only，零 stroke path）——
# patch+hatch 方案实测被 collision audit 判 text-stroke FAIL：hatch 线按未裁剪
# 平铺几何进 PDF（PyMuPDF 读到跨页面 bbox）。单 imshow 栅格叠加无 stroke，
# 审计 PASS 且视觉等价（2026-10-25 Run 002 修复回环）。
_DIAG_TEX: np.ndarray | None = None


def _diag_texture(sub: int = 8, period: int = 3) -> np.ndarray:
    """对角带纹理 RGBA（缓存）：对角格白底 + 平行细灰纹，非对角全透明."""
    global _DIAG_TEX
    if _DIAG_TEX is not None:
        return _DIAG_TEX
    r = N * sub
    tex = np.zeros((r, r, 4), dtype=np.float32)
    tex[..., :3] = 1.0                                   # 白底
    rr, cc = np.meshgrid(np.arange(r), np.arange(r), indexing="ij")
    diag = (rr // sub) == (cc // sub)
    stripe = ((cc % sub) - (rr % sub)) % period == 0     # 格内平行细纹
    tex[diag & stripe, :3] = 0.62                        # 细纹灰
    tex[..., 3] = 0.0                                    # 缺省透明
    tex[diag, 3] = 1.0                                   # 仅对角格不透明
    _DIAG_TEX = tex
    return tex


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--subjects", type=str, default=",".join(map(str, ALL_SIDS)),
                   help="逗号分隔 sid 子集（冒烟）；缺省全量 8 例")
    p.add_argument("--recompute-dir", type=str, default=str(RECOMPUTE_DIR))
    p.add_argument("--gc-dir", type=str, default=str(GC_REF_DIR))
    p.add_argument("--geom-pt", type=str, default=str(GEOM_PT))
    p.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    return p.parse_args(argv)


def load_channel_names(geom_pt: Path) -> list[str]:
    import torch  # 纯 CPU load（map_location）；仅读 channel_names
    d = torch.load(str(geom_pt), map_location="cpu", weights_only=False)
    names = [str(c) for c in d["channel_names"]]
    assert len(names) == N, f"expected {N} channel names, got {len(names)}"
    for i, c in enumerate(names):
        assert c == f"E{i + 1}", f"channel name mismatch at {i}: {c}"
    return names


def load_data(main_dir: Path, gc_dir: Path, sids: list[int]) -> list[dict]:
    """EC_pert（conn v4 json）+ GC-EC（refs npz）+ ρ(aligned) 注记值（只读）。"""
    summary = json.load(open(RHO_SUMMARY))
    v1 = summary["refs_blocks"]["v1_1_45_fixed8"]
    rows = []
    for k, sid in enumerate(sids):
        matches = sorted(main_dir.glob("subject_*.json"))
        hit = None
        for f in matches:
            d = json.load(open(f))
            if d["subject_file"] == f"s{sid}":
                hit = (f, d)
                break
        assert hit is not None, f"subject s{sid} not found in {main_dir}"
        _f, d = hit
        ec = np.array(d["connectivity_matrix"], dtype=np.float64)
        gc = np.load(gc_dir / f"refs_s{sid}.npz")["ec"].astype(np.float64)
        assert ec.shape == (N, N) and gc.shape == (N, N)
        rho = float(v1["per_subject"][str(sid)]["spearman_raw_aligned"])
        # 漂移对拍：图面注记 ρ 必须等于本图两矩阵 offdiag Spearman（aligned）
        rho_chk = float(sps.spearmanr(ec[OFF], gc[OFF]).statistic)
        assert abs(rho - rho_chk) < 1e-9, f"s{sid}: json rho {rho} != recomputed {rho_chk}"
        rows.append({"sid": f"s{sid}", "sid_num": sid, "group": d["group"],
                     "ec": ec, "gc": gc, "rho": rho})
    groups = [r["group"] for r in rows]
    if sids == ALL_SIDS:
        assert groups == ["MDD"] * 4 + ["HC"] * 4, f"行序契约破坏: {groups}"
    return rows


def rank_map(m: np.ndarray) -> np.ndarray:
    """offdiag 矩阵内百分位秩 → [0,1]；对角 NaN（双侧统一掩蔽口径）。"""
    r = np.full_like(m, np.nan, dtype=np.float64)
    ranks = sps.rankdata(m[OFF])                       # 1..16256（无并列损失）
    r[OFF] = (ranks - 1.0) / (ranks.size - 1.0)
    return r


def _heatmap(ax, m, *, vmin, vmax, tick_labels) -> object:
    im = ax.imshow(m, cmap=CMAP, vmin=vmin, vmax=vmax,
                   interpolation="nearest", aspect="equal", origin="upper",
                   zorder=1)
    # 对角掩蔽视觉：NaN→set_bad 白之上叠「白+细纹」栅格纹理（fill-only，零 stroke）
    ax.imshow(_diag_texture(), extent=(-0.5, N - 0.5, N - 0.5, -0.5),
              origin="upper", interpolation="nearest", aspect="equal",
              zorder=2)
    ax.set_xticks(TICK_IDX, tick_labels, fontsize=5.5)
    ax.set_yticks(TICK_IDX, tick_labels, fontsize=5.5)
    ax.tick_params(length=2, pad=1.5)
    ax.set_xlabel("target (column)", fontsize=6)
    ax.set_ylabel("source (row)", fontsize=6)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
    return im


def _save(fig, name: str, fig_dir: Path, qa_dir: Path) -> Path:
    """alignment gate（1×2 等宽面板，容差 1.5pt）→ PNG(300dpi)+PDF；QA JSON 统一随 qa/ 档."""
    qa_dir.mkdir(parents=True, exist_ok=True)
    require_matplotlib_panel_alignment(
        fig, json_out=str(qa_dir / f"{name}.alignment.json"),
        overlay_svg=str(qa_dir / f"{name}.alignment.svg"),
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    png = fig_dir / f"{name}.png"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    return png


def _corner_note(ax, text: str) -> None:
    """图内角注（面板内右下角，完全含于热图栅格内 → collision 语义 contained）."""
    ax.text(0.975, 0.045, text, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6, zorder=5,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      alpha=0.85, edgecolor="none"))


def fig_subject_pair(row: dict, fig_dir: Path, qa_dir: Path,
                     tick_labels: list[str]) -> str:
    r_ec, r_gc = rank_map(row["ec"]), rank_map(row["gc"])
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.5), constrained_layout=True)
    specs = (("a", f"{row['sid']} ({row['group']}) — EC_pert (conn v4) "
               f"[{row['ec'].min():.2f}–{row['ec'].max():.2f}]", r_ec, axes[0]),
             ("b", f"{row['sid']} ({row['group']}) — GC-EC (1–45Hz fixed8) "
               f"[{row['gc'].min():.3f}–{row['gc'].max():.3f}]", r_gc, axes[1]))
    im = None
    for letter, title, m, ax in specs:
        im = _heatmap(ax, m, vmin=0.0, vmax=1.0, tick_labels=tick_labels)
        ax.set_title(f"$\\bf{{{letter}}}$  {title}", fontsize=7, pad=3)
    cb = fig.colorbar(im, ax=list(axes), shrink=0.9, pad=0.025,
                      ticks=[0, 0.25, 0.5, 0.75, 1.0])
    cb.set_label("percentile rank (offdiag, within matrix)", fontsize=6)
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_linewidth(0.6)
    _corner_note(axes[1], f"Spearman ρ(aligned) = {row['rho']:+.3f}")
    fig.suptitle(f"{row['sid']} ({row['group']}) — EC_pert (conn v4) vs GC-EC "
                 f"(v1 refs, 1–45 Hz, order 8 fixed) — 128×128 (E1–E128), "
                 f"row = source, col = target · task {TASK_NAME}", fontsize=8)
    name = f"subj_{row['sid']}_{row['group']}"
    _save(fig, name, fig_dir, qa_dir)
    return name


def fig_group_rank(rows: list[dict], fig_dir: Path, qa_dir: Path,
                   tick_labels: list[str], rho_group: dict) -> str:
    ec_rm = np.stack([rank_map(r["ec"]) for r in rows]).mean(axis=0)
    gc_rm = np.stack([rank_map(r["gc"]) for r in rows]).mean(axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.5), constrained_layout=True)
    specs = (("a", "Group mean (n=8) — EC_pert (conn v4) rank-mean "
               f"[{ec_rm[OFF].min():.2f}–{ec_rm[OFF].max():.2f}]", ec_rm, axes[0]),
             ("b", "Group mean (n=8) — GC-EC (1–45Hz fixed8) rank-mean "
               f"[{gc_rm[OFF].min():.2f}–{gc_rm[OFF].max():.2f}]", gc_rm, axes[1]))
    im = None
    for letter, title, m, ax in specs:
        im = _heatmap(ax, m, vmin=0.0, vmax=1.0, tick_labels=tick_labels)
        ax.set_title(f"$\\bf{{{letter}}}$  {title}", fontsize=7, pad=3)
    cb = fig.colorbar(im, ax=list(axes), shrink=0.9, pad=0.025,
                      ticks=[0, 0.25, 0.5, 0.75, 1.0])
    cb.set_label("mean percentile rank (offdiag, within matrix)", fontsize=6)
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_linewidth(0.6)
    _corner_note(axes[1], f"per-subject ρ(aligned): mean {rho_group['mean']:+.3f}, "
                          f"SD {rho_group['std']:.3f} (n=8)")
    fig.suptitle(f"Group mean (n=8, rank-normalized) — EC_pert (conn v4) vs GC-EC "
                 f"(v1 refs, 1–45 Hz, order 8 fixed) — 128×128 (E1–E128), "
                 f"row = source, col = target · task {TASK_NAME}", fontsize=8)
    name = "group_means_rank"
    _save(fig, name, fig_dir, qa_dir)
    return name


def fig_group_abs(rows: list[dict], fig_dir: Path, qa_dir: Path,
                  tick_labels: list[str]) -> str:
    ec_m = np.stack([r["ec"] for r in rows]).mean(axis=0)
    gc_m = np.stack([r["gc"] for r in rows]).mean(axis=0)
    # 各测自身 vmax = 8 例 pooled offdiag P99.5（量级诚实：色标刻度 = 真实数值）
    vmax_ec = float(np.percentile(np.concatenate([r["ec"][OFF] for r in rows]), 99.5))
    vmax_gc = float(np.percentile(np.concatenate([r["gc"][OFF] for r in rows]), 99.5))
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.2))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.85, bottom=0.13, wspace=0.5)
    specs = (("a", f"Group mean (n=8) — EC_pert (conn v4) "
               f"[{ec_m.min():.2f}–{ec_m.max():.2f}]", ec_m, vmax_ec, "%.2f",
             "EC_pert v4 — 0 to pooled offdiag P99.5", axes[0]),
             ("b", f"Group mean (n=8) — GC-EC (1–45Hz fixed8) "
               f"[{gc_m.min():.3f}–{gc_m.max():.3f}]", gc_m, vmax_gc, "%.3f",
             "GC-EC v1 — 0 to pooled offdiag P99.5", axes[1]))
    for letter, title, m, vmax, fmt, cb_label, ax in specs:
        im = _heatmap(ax, m, vmin=0.0, vmax=vmax, tick_labels=tick_labels)
        ax.set_title(f"$\\bf{{{letter}}}$  {title}", fontsize=7, pad=3)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4.5%", pad=0.12)
        cb = fig.colorbar(im, cax=cax)
        cb.set_label(f"{cb_label} ({fmt % vmax})", fontsize=6)
        cb.ax.tick_params(labelsize=6)
        cb.outline.set_linewidth(0.6)
    fig.suptitle(f"Group mean (n=8, raw values) — EC_pert (conn v4) vs GC-EC "
                 f"(v1 refs) — scales NOT shared (independent colorbars, true "
                 f"units) · 128×128, row = source, col = target · task {TASK_NAME}",
                 fontsize=8)
    name = "group_means_abs"
    _save(fig, name, fig_dir, qa_dir)
    return name


def print_memo(rows: list[dict]) -> None:
    """数据侧备忘（供 README / 交付总结引用）。"""
    E = np.stack([r["ec"] for r in rows]); G = np.stack([r["gc"] for r in rows])
    ec_m, gc_m = E.mean(0), G.mean(0)
    p995_ec = float(np.percentile(E[:, OFF], 99.5))
    p995_gc = float(np.percentile(G[:, OFF], 99.5))
    print(f"[memo] EC_pert pooled offdiag min/max = {E[:,OFF].min():.3f}/{E[:,OFF].max():.3f}; "
          f"pooled offdiag P99.5 = {p995_ec:.3f} (= clamp ceiling, all 8 subjects "
          f"share exact min/max {E.min():.3f}/{E.max():.3f} from σ-clamp)")
    print(f"[memo] GC-EC   pooled offdiag min/max = {G[:,OFF].min():.5f}/{G[:,OFF].max():.3f}; "
          f"pooled offdiag P99.5 = {p995_gc:.4f}")
    print(f"[memo] group-mean EC_pert range = [{ec_m.min():.3f}, {ec_m.max():.3f}]; "
          f"group-mean GC-EC range = [{gc_m.min():.3f}, {gc_m.max():.3f}]")
    for r in rows:
        print(f"[memo] {r['sid']} ({r['group']}): EC [{r['ec'].min():.3f}, {r['ec'].max():.3f}] "
              f"GC [{r['gc'].min():.3f}, {r['gc'].max():.3f}] rho={r['rho']:+.3f}")


def main() -> int:
    a = parse_args()
    t0 = time.time()
    sids = [int(s) for s in a.subjects.split(",")]
    out_dir = Path(a.out_dir)
    fig_dir_subj = out_dir / "subject_pairs"
    qa_dir = out_dir / "qa"
    fig_dir_subj.mkdir(parents=True, exist_ok=True)
    rows = load_data(Path(a.recompute_dir), Path(a.gc_dir), sids)
    tick_labels = load_channel_names(Path(a.geom_pt))
    tick_labels = [tick_labels[i] for i in TICK_IDX]
    print(f"[viz] subjects: {[r['sid'] for r in rows]}", flush=True)
    print_memo(rows)
    for r in rows:
        name = fig_subject_pair(r, fig_dir_subj, qa_dir, tick_labels)
        print(f"[viz] wrote subject_pairs/{name}.png/.pdf", flush=True)
    if sids == ALL_SIDS:
        summary = json.load(open(RHO_SUMMARY))
        rho_group = summary["refs_blocks"]["v1_1_45_fixed8"]["group"]["spearman_raw_aligned"]
        for make in (lambda: fig_group_rank(rows, out_dir, qa_dir, tick_labels, rho_group),
                     lambda: fig_group_abs(rows, out_dir, qa_dir, tick_labels)):
            name = make()
            print(f"[viz] wrote {name}.png/.pdf", flush=True)
    else:
        print(f"[viz] subset {sids} != full 8 -> group figures skipped", flush=True)
    print(f"[viz] done in {time.time() - t0:.1f}s -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
