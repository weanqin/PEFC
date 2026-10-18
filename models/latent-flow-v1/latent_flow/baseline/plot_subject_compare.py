"""Side-by-side directed-connectivity comparison plot for one subject.

Renders GC / TE / OC-LatentFlow attribution as a 3-panel heatmap row (shared
128x128 layout, row=source a / col=target b), each with its own colour scale
(the three methods span very different dynamic ranges, so a shared scale
would wash out structure). Below, a 3-panel histogram row shows each
method's off-diagonal value distribution.

This is the "at a glance" structural-contrast figure: GC is dense +
near-symmetric (linear), TE is sparser + more directed (nonlinear), OC is
the most directed + lowest-rank (model attribution).

Output:
    outputs/baseline_connectivity/subject_0_compare.png

Usage (run from repo root D:/102-npi/ude-vae-npi):

    python -u -m latent_flow.baseline.plot_subject_compare \\
        --subject 0
"""

import argparse
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


def _load(path):
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    cm = np.asarray(payload["connectivity_matrix"], dtype=np.float64)
    cm[~np.isfinite(cm)] = np.nan
    return cm


def _offdiag(cm):
    n = cm.shape[0]
    mask = ~np.eye(n, dtype=bool)
    vals = cm[mask]
    return vals[np.isfinite(vals)]


def _directedness(cm):
    asym = np.nanmean(np.abs(cm - cm.T))
    mag = np.nanmean(np.abs(cm))
    return float(asym / mag) if mag > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, default=0)
    ap.add_argument("--baseline-dir", type=str,
                    default="results/baseline_connectivity")
    ap.add_argument("--oc-dir", type=str, default="results/oc_connectivity")
    ap.add_argument("--out", type=str,
                    default="outputs/baseline_connectivity")
    args = ap.parse_args()

    i = args.subject
    paths = {
        "GC (linear)": os.path.join(args.baseline_dir, "gc", f"subject_{i}.json"),
        "TE (nonlinear)": os.path.join(args.baseline_dir, "te",
                                       f"subject_{i}.json"),
        "OC attribution": os.path.join(args.oc_dir, f"subject_{i}.json"),
    }
    mats, present = {}, []
    for label, p in paths.items():
        if os.path.exists(p):
            mats[label] = _load(p)
            present.append(label)
        else:
            print(f"  (missing) {label}: {p}")
    if not present:
        raise SystemExit("no input matrices found")

    fig, axes = plt.subplots(2, len(present), figsize=(5 * len(present), 9))
    if len(present) == 1:
        axes = axes.reshape(2, 1)

    # Row 1: heatmaps.
    for col, label in enumerate(present):
        cm = mats[label]
        ax = axes[0, col]
        finite = cm[np.isfinite(cm)]
        vmin = float(np.min(finite)) if finite.size else 0.0
        vmax = float(np.max(finite)) if finite.size else 1.0
        im = ax.imshow(cm, cmap="hot", aspect="auto", origin="lower",
                       vmin=vmin, vmax=vmax)
        ax.set_xlabel("target channel b", fontsize=9)
        ax.set_ylabel("source channel a", fontsize=9)
        ax.set_title(f"{label}\nmean={np.nanmean(cm):.3g}  "
                     f"dir={_directedness(cm):.2f}", fontsize=10)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=7)

    # Row 2: histograms (off-diagonal values).
    for col, label in enumerate(present):
        cm = mats[label]
        ax = axes[1, col]
        vals = _offdiag(cm)
        if vals.size > 0:
            ax.hist(vals, bins=40, color="tab:orange", alpha=0.75,
                    edgecolor="white")
        ax.set_xlabel("connectivity value", fontsize=9)
        ax.set_ylabel("count", fontsize=9)
        ax.set_title(f"{label} distribution", fontsize=10)
        ax.tick_params(labelsize=8)

    fig.suptitle(f"Directed connectivity — fold-0 subject {i} "
                 f"(rows=source, cols=target; brighter = stronger a->b)",
                 fontsize=12, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"subject_{i}_compare.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
