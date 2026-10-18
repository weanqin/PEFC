"""汇总消融实验对比表 + 图（task 07-15-graph-module-ablation, 多 fold）。

两种模式：
  --fold N          : 单 fold 汇总 → results/ablation/fold{N}/summary.{csv,json}
  --cross-fold 0,1,2,3,4 : 跨 fold 汇总 → results/ablation/summary_all_folds.{csv,json}
                           + outputs/ablation/compare_all_folds.png

收集每变体：预测性能(history.json best epoch) + GC/TE 相关性 + A_eff 结构 + DE 结构。
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
OUTPUTS_DIR = ROOT / "outputs"
LOG_DIR = ROOT / "logs" / "ablation"

VARIANTS = [
    ("baseline_screg", "baseline (M+SC-reg·W+W)"),
    ("nosc", "no-SC (M+W)"),
    ("nomexhat", "no-mexhat (SC-reg·W+W)"),
    ("now", "no-W (M+SC-reg)"),
]
N_SUBJECTS = 10


def _load_json(path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _matrix_stats(matrix):
    m = np.asarray(matrix, dtype=np.float64)
    m[~np.isfinite(m)] = 0.0
    n = m.shape[0]
    off = m[~np.eye(n, dtype=bool)]
    sparsity = float(np.mean(np.abs(off) < 1e-6))
    denom = max(np.mean(np.abs(off)), 1e-12)
    directedness = float(np.mean(np.abs(off - off.T)) / denom) if n > 1 else float("nan")
    cv = float(np.std(off) / (np.mean(np.abs(off)) + 1e-12))
    return {"sparsity": sparsity, "directedness": directedness, "cv": cv}


def _ckpt_stem(fold, name):
    base = f"fold{fold}_oc_latent_flow_flow_best_graph_mexhat_screg_l0.01_w0.1"
    return base if name == "baseline_screg" else f"{base}_{name}"


def collect_variant(fold, name):
    """Collect all metrics for one variant in one fold."""
    row = {"fold": fold, "variant": name, "label": dict(VARIANTS).get(name, name)}

    # 预测性能：history.json
    stem = _ckpt_stem(fold, name)
    # fold0 baseline 重建的 history 在 logs/ablation/ 根；其余在 logs/ablation/fold{N}/
    candidates = [
        LOG_DIR / f"fold{fold}" / f"{stem}_history.json",
        LOG_DIR / f"{stem}_history.json",  # fold0 legacy root-level
    ]
    hist = None
    for c in candidates:
        hist = _load_json(c)
        if hist:
            break
    if hist and hist.get("val"):
        best = min(hist["val"], key=lambda e: e.get("loss", float("inf")))
        row["val_loss_best"] = best.get("loss")
        row["val_corr2"] = best.get("corr2")
        row["val_recon2"] = best.get("recon2")
        row["best_epoch"] = best.get("epoch")
    else:
        row["val_loss_best"] = None
        row["val_corr2"] = None
        row["val_recon2"] = None
        row["best_epoch"] = None

    # GC/TE 对比
    cmp = _load_json(RESULTS_DIR / "compare_params_ablation" / f"fold{fold}" / name
                     / f"compare_fold{fold}.json")
    if cmp and "results" in cmp:
        for m in ("gc", "te"):
            agg = cmp["results"].get(m, {})
            row[f"{m}_r_abs"] = agg.get("mean_r_abs")
            row[f"{m}_r_signed"] = agg.get("mean_r_signed")
    else:
        for m in ("gc", "te"):
            row[f"{m}_r_abs"] = None
            row[f"{m}_r_signed"] = None

    # A_eff 结构
    aeff = _load_json(RESULTS_DIR / "graph_connectivity_ablation" / f"fold{fold}" / name
                      / f"graph_connectivity_fold{fold}.json")
    if aeff and "connectivity_matrix" in aeff:
        for k, v in _matrix_stats(aeff["connectivity_matrix"]).items():
            row[f"aeff_{k}"] = v
    else:
        for k in ("sparsity", "directedness", "cv"):
            row[f"aeff_{k}"] = None

    # DE 结构（fold-mean）
    de_dir = RESULTS_DIR / "oc_connectivity_relative_ablation" / f"fold{fold}" / name
    de_mats = []
    i = 0
    while (de_dir / f"subject_{i}.json").exists():
        payload = _load_json(de_dir / f"subject_{i}.json")
        if payload and "connectivity_matrix" in payload:
            de_mats.append(np.asarray(payload["connectivity_matrix"], dtype=np.float64))
        i += 1
    if de_mats:
        stack = np.stack(de_mats, axis=0)
        mean_cm = np.nanmean(stack, axis=0)
        for k, v in _matrix_stats(mean_cm).items():
            row[f"de_{k}"] = v
        row["de_n_subjects"] = len(de_mats)
    else:
        for k in ("sparsity", "directedness", "cv"):
            row[f"de_{k}"] = None
        row["de_n_subjects"] = 0

    return row


def _fmt(v, fmt=".4f"):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "NA"
    try:
        return format(float(v), fmt)
    except (TypeError, ValueError):
        return str(v)


FIELDS = ["fold", "variant", "label", "best_epoch", "val_loss_best", "val_corr2",
          "val_recon2", "gc_r_abs", "gc_r_signed", "te_r_abs", "te_r_signed",
          "aeff_sparsity", "aeff_directedness", "aeff_cv",
          "de_sparsity", "de_directedness", "de_cv", "de_n_subjects"]


def write_csv(rows, path):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(FIELDS)
        for r in rows:
            w.writerow([_fmt(r.get(f), fmt=".4f") if f not in (
                "variant", "label", "best_epoch", "de_n_subjects", "fold")
                        else r.get(f) for f in FIELDS])
    print(f"  CSV: {path}")


def write_json(rows, path):
    payload = {"variants": rows}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=lambda o: None if (
            isinstance(o, float) and not np.isfinite(o)) else o)
    print(f"  JSON: {path}")


def plot_compare(rows, path, title):
    labels = [r["label"] for r in rows]
    gc = [r.get("gc_r_abs") for r in rows]
    te = [r.get("te_r_abs") for r in rows]
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.bar(x - w / 2, [0.0 if g is None else g for g in gc], w, label="GC r_abs", color="#4C72B0")
    ax.bar(x + w / 2, [0.0 if t is None else t for t in te], w, label="TE r_abs", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Pearson $r$ (A_eff vs baseline)", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  PNG: {path}")


def print_table(rows, fold_label):
    print(f"\n=== comparison table ({fold_label}) ===")
    print(f"{'variant':<16} {'val_corr2':>10} {'gc_r_abs':>10} {'te_r_abs':>10} "
          f"{'aeff_cv':>10} {'de_cv':>10}")
    for r in rows:
        print(f"{r['variant']:<16} {_fmt(r.get('val_corr2')):>10} "
              f"{_fmt(r.get('gc_r_abs')):>10} {_fmt(r.get('te_r_abs')):>10} "
              f"{_fmt(r.get('aeff_cv')):>10} {_fmt(r.get('de_cv')):>10}")


def summarize_single_fold(fold):
    print(f"=== ablation summary (fold {fold}) ===")
    rows = [collect_variant(fold, name) for name, _ in VARIANTS]
    out_dir = RESULTS_DIR / "ablation" / f"fold{fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, out_dir / "summary.csv")
    write_json(rows, out_dir / "summary.json")
    print_table(rows, f"fold {fold}")
    print("=== done ===")


def summarize_cross_fold(folds):
    """Aggregate across folds: per (variant, metric) mean ± std over folds."""
    print(f"=== cross-fold summary (folds {folds}) ===")
    # collect all rows across folds
    all_rows = []
    per_fold_ok = {}
    for f in folds:
        sj = _load_json(RESULTS_DIR / "ablation" / f"fold{f}" / "summary.json")
        if sj and "variants" in sj:
            all_rows.extend(sj["variants"])
            per_fold_ok[f] = True
        else:
            print(f"  [warn] fold {f} summary missing, skipping")
            per_fold_ok[f] = False

    if not all_rows:
        print("  no fold summaries found, abort")
        return

    # aggregate: for each variant, mean/std of numeric metrics across folds
    numeric_keys = ["val_loss_best", "val_corr2", "val_recon2",
                    "gc_r_abs", "gc_r_signed", "te_r_abs", "te_r_signed",
                    "aeff_sparsity", "aeff_directedness", "aeff_cv",
                    "de_sparsity", "de_directedness", "de_cv"]
    agg_rows = []
    for name, label in VARIANTS:
        vr = [r for r in all_rows if r.get("variant") == name]
        if not vr:
            continue
        row = {"variant": name, "label": label, "n_folds": len(vr)}
        for k in numeric_keys:
            vals = [r.get(k) for r in vr
                    if r.get(k) is not None and np.isfinite(r.get(k, float("nan")))]
            if vals:
                arr = np.array(vals, dtype=float)
                row[f"{k}_mean"] = float(arr.mean())
                row[f"{k}_std"] = float(arr.std())
            else:
                row[f"{k}_mean"] = None
                row[f"{k}_std"] = None
        agg_rows.append(row)

    out_dir = RESULTS_DIR / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV: one row per variant, mean/std columns
    agg_fields = ["variant", "label", "n_folds"]
    for k in numeric_keys:
        agg_fields += [f"{k}_mean", f"{k}_std"]
    with open(out_dir / "summary_all_folds.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(agg_fields)
        for r in agg_rows:
            w.writerow([_fmt(r.get(f), ".4f") if f not in ("variant", "label", "n_folds")
                        else r.get(f) for f in agg_fields])
    print(f"  CSV: {out_dir / 'summary_all_folds.csv'}")

    payload = {"folds": [f for f in folds if per_fold_ok.get(f)],
               "aggregates": agg_rows, "all_rows": all_rows}
    with open(out_dir / "summary_all_folds.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=lambda o: None if (
            isinstance(o, float) and not np.isfinite(o)) else o)
    print(f"  JSON: {out_dir / 'summary_all_folds.json'}")

    # plot: grouped bars of GC/TE r_abs mean across folds, error bars = std
    labels = [r["label"] for r in agg_rows]
    gc_mean = [r.get("gc_r_abs_mean") or 0.0 for r in agg_rows]
    gc_std = [r.get("gc_r_abs_std") or 0.0 for r in agg_rows]
    te_mean = [r.get("te_r_abs_mean") or 0.0 for r in agg_rows]
    te_std = [r.get("te_r_abs_std") or 0.0 for r in agg_rows]
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.bar(x - w / 2, gc_mean, w, yerr=gc_std, label="GC r_abs", color="#4C72B0", capsize=3)
    ax.bar(x + w / 2, te_mean, w, yerr=te_std, label="TE r_abs", color="#DD8452", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Pearson $r$ (mean ± std over folds)", fontsize=9)
    ax.set_title(f"Graph ablation: A_eff vs GC/TE ({len([f for f in folds if per_fold_ok.get(f)])} folds)", fontsize=10)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "ablation" / "compare_all_folds.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  PNG: {OUTPUTS_DIR / 'ablation' / 'compare_all_folds.png'}")

    # print table
    print(f"\n{'variant':<16} {'val_corr2':>14} {'gc_r_abs':>14} {'te_r_abs':>14}")
    for r in agg_rows:
        def ms(k):
            m, s = r.get(f"{k}_mean"), r.get(f"{k}_std")
            if m is None:
                return "NA"
            return f"{m:.3f}±{s:.3f}"
        print(f"{r['variant']:<16} {ms('val_corr2'):>14} {ms('gc_r_abs'):>14} {ms('te_r_abs'):>14}")
    print("=== done ===")


def parse_args():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--fold", type=int, choices=range(5),
                   help="Single-fold summary mode.")
    g.add_argument("--cross-fold", type=str,
                   help="Cross-fold aggregate mode, e.g. --cross-fold 0,1,2,3,4")
    return p.parse_args()


def main():
    args = parse_args()
    if args.fold is not None:
        summarize_single_fold(args.fold)
    else:
        folds = [int(f) for f in args.cross_fold.split(",") if f.strip() != ""]
        summarize_cross_fold(folds)


if __name__ == "__main__":
    main()
