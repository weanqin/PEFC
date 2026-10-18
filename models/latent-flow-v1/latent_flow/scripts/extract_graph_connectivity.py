"""Extract directed connectivity from a trained graph-mode flow (R4 clean path).

Task 06-29 R4. The graph-mode flow concentrates ALL cross-channel mixing in the
graph mixer. For the Mexican-hat variant (B.5), the model's INFERRED directed
connectivity is read DIRECTLY from ``effective_adjacency()`` = mexhat(d; sigma)
+ gamma * W — this is the cleanest connectivity source (no bisection needed).
For the Gaussian variant (B.1-B.4), it falls back to the K-hop reach formula.

Output (relative to repo root):
    results/graph_connectivity/graph_connectivity_fold{fold}.json
    outputs/graph_connectivity/connectivity_fold{fold}_heatmap.png

Usage (from repo root):

    D:/program/anaconda/envs/npi/python.exe -u -m \\
        latent_flow.scripts.extract_graph_connectivity \\
        --fold 0
"""

import argparse
import json
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from latent_flow.config import CHECKPOINT_DIR
from latent_flow.models.graph_utils import pairwise_distance, load_geometry
from latent_flow.scripts._connectivity_driver import json_default, load_model


DEFAULT_OUTPUT_DIR = "results/graph_connectivity"
DEFAULT_OUTPUTS_DIR = "outputs/graph_connectivity"


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract directed connectivity from a graph-mode flow "
                    "(R4 clean path).")
    p.add_argument("--checkpoint", type=str, default="",
                   help="graph-mode flow checkpoint. Empty auto-selects the "
                        "mexhat variant if present, else gaussian.")
    p.add_argument("--fold", type=int, default=0, choices=range(5))
    p.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--outputs-dir", type=str, default=DEFAULT_OUTPUTS_DIR)
    p.add_argument("--device", type=str, default="")
    return p.parse_args()


def effective_connectivity(model) -> np.ndarray:
    """Read the model's inferred directed connectivity from the graph mixer.

    Mexican-hat variant: A_eff = mexhat(d; sigma) + gamma * W directly.
    Gaussian variant: K-hop reach (symmetric structural).
    """
    flow = model.flow
    mixer = flow.graph_mixer
    C = mixer.n_channels

    if hasattr(mixer, "effective_adjacency"):
        # Mexican-hat variant: A_eff is the connectivity matrix directly.
        return mixer.effective_adjacency().detach().float().cpu().numpy()

    # Gaussian variant: K-hop reach formula (legacy).
    K = mixer.k_layers
    layer_gain = np.array([layer.weight.detach().abs().mean().item()
                           for layer in mixer.msg_layers])
    A_hat = mixer.a_hat.detach().float().cpu().numpy()
    M = np.eye(C)
    reach = np.zeros((C, C))
    total = layer_gain.sum() + 1e-12
    for k in range(K):
        M = M @ A_hat
        reach = reach + (layer_gain[k] / total) * M
    return 0.5 * (reach + reach.T)


def main():
    args = parse_args()
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.checkpoint:
        ckpt = args.checkpoint
    else:
        mexhat = str(CHECKPOINT_DIR / f"fold{args.fold}_oc_latent_flow_flow_best_graph_mexhat.pt")
        gauss = str(CHECKPOINT_DIR / f"fold{args.fold}_oc_latent_flow_flow_best_graph.pt")
        ckpt = mexhat if os.path.exists(mexhat) else gauss
    print(f"Checkpoint: {ckpt}")
    model = load_model(ckpt, device)
    kernel = getattr(model, "flow_graph_kernel", "gaussian")
    print(f"Flow kernel: {kernel}")

    C = model.n_channels
    conn = effective_connectivity(model)              # (C, C)
    print(f"Connectivity matrix: shape {conn.shape}")

    # ----- diagnostics: D1 (sparsity/structure), D2 (distance-decay) -----
    off = conn[~np.eye(C, dtype=bool)]
    cv = off.std() / (np.abs(off).mean() + 1e-12)
    asym = np.abs(conn - conn.T).mean()
    print(f"  D1 offdiag CV: {cv * 100:.2f}%   (GC~99%, TE~48%)")
    print(f"  asymmetry |M-M^T| mean: {asym:.4e}")

    coords, _ = load_geometry()
    d = pairwise_distance(coords).cpu().numpy()
    inv_d = 1.0 / (d + 1e-6)
    off_mask = ~np.eye(C, dtype=bool)
    r = float(np.corrcoef(conn[off_mask], inv_d[off_mask])[0, 1])
    print(f"  D2 Pearson(conn, 1/d): {r:.4f}  (>0 = nearer electrodes connect more)")

    # ----- heatmap (Nature style) -----
    os.makedirs(args.outputs_dir, exist_ok=True)
    from latent_flow.viz.nature_style import (
        apply_nature_style, add_panel_label, save_nature)
    apply_nature_style(font_size=7, axes_linewidth=0.8)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    # Diverging cmap because Mexican-hat A_eff has +/- weights.
    vmax = np.percentile(np.abs(conn), 99)
    im = ax.imshow(conn, aspect="equal", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, origin="upper", interpolation="nearest")
    ax.set_title(f"learned $A_{{\\rm eff}}$  (mexhat + $\\gamma W$)\n"
                 f"fold {args.fold} · kernel={kernel}", fontsize=8, pad=4)
    ax.set_xlabel("target channel $b$", fontsize=7)
    ax.set_ylabel("source channel $a$", fontsize=7)
    ax.set_xticks([0, C // 2, C - 1])
    ax.set_yticks([0, C // 2, C - 1])
    ax.tick_params(labelsize=6)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("$A_{\\rm eff}[a,b]$  (red=excite, blue=inhibit)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    cbar.outline.set_linewidth(0.5)
    fig.tight_layout()
    out_base = os.path.join(args.outputs_dir,
                            f"connectivity_fold{args.fold}_heatmap")
    saved = save_nature(fig, out_base)
    plt.close(fig)
    print(f"  heatmap saved: {saved}")

    # ----- JSON -----
    os.makedirs(args.output_dir, exist_ok=True)
    out_json = os.path.join(args.output_dir, f"graph_connectivity_fold{args.fold}.json")
    payload = {
        "checkpoint": os.path.basename(ckpt),
        "model": f"OC-LatentFlow (stage-2 {kernel}-graph flow) — learned-mixer connectivity",
        "method": ("effective_adjacency (mexhat + gamma*W)"
                   if kernel == "mexhat" else "K-hop graph reach"),
        "n_channels": int(C),
        "connectivity_matrix": conn.tolist(),
        "diagnostics": {
            "offdiag_cv": float(cv),
            "asymmetry_mean_abs": float(asym),
            "distance_decay_pearson": r,
        },
    }
    with open(out_json, "w") as fh:
        json.dump(payload, fh, indent=2, default=json_default)
    print(f"  JSON saved: {out_json}")


if __name__ == "__main__":
    main()
