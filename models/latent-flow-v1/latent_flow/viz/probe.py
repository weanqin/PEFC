"""Perturbation-response visualization helpers for the fixed-sigma probe.

Companion to ``scripts/probe_perturbation_response.py``. The probe skips
sigma training and instead measures the model's per-target MSE response
to a FIXED perturbation amplitude on each source channel, producing a
128 x 128 response matrix per sigma value. These helpers render:

  - single-sigma heatmap (128 x 128, viridis colorbar)
  - side-by-side comparison heatmap across 3 sigma values (1 x 3 subplot)
  - per-source out-degree histogram (3 sigma values overlaid)

All plots use the Agg backend (no GUI) per logging-guidelines.md. Figures
are sized so a 128 x 128 matrix stays readable. Files are saved at
dpi=150 to balance size and legibility. Labels are ASCII-only to avoid
Windows GBK encoding crashes (per project gotcha in CLAUDE.md).

Shape note:
  response_matrix is (n_src, n_tgt) with n_src = n_tgt = 128 in the
  probe's default config. Heatmaps handle non-square inputs directly.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless; safe to import at module load
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------

def plot_response_heatmap(
    response_matrix,
    sigma_value: float,
    save_path: str,
    title: Optional[str] = None,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
):
    """Render one sigma's (n_src, n_tgt) response matrix as a heatmap.

    Args:
        response_matrix: 2-D array-like (numpy / torch / list-of-lists)
            of per-target MSE response values. Rows are source channels,
            columns are target channels.
        sigma_value: the fixed perturbation amplitude this matrix
            corresponds to. Used in the default title and colorbar label
            so a reader can identify the sigma without opening the JSON.
        save_path: output PNG path. Parent dir must exist.
        title: optional plot title. When None, a default title including
            the sigma value is used.
        cmap: matplotlib colormap. Defaults to "viridis" (sequential,
            perceptually uniform, colorblind-friendly) — matches the
            connectivity.py convention.
        vmin / vmax: color scale bounds. When None (default), the scale
            auto-fits to the matrix's finite range so structure is
            visible regardless of the response magnitude.
    """
    arr = _to_numpy_2d(response_matrix)
    n_src, n_tgt = arr.shape
    # Square-ish figsize so a 128x128 matrix is ~8x7 inches. Linear in
    # the larger dimension; never smaller than 4 inches.
    size = max(4.0, min(10.0, 8.0 * max(n_src, n_tgt) / 128.0))
    fig, ax = plt.subplots(figsize=(size, size * 0.9))

    # Auto-scale when not overridden so NaN / zero rows don't blow out
    # the colorbar. Matches the connectivity.py plot behavior.
    finite = arr[np.isfinite(arr)]
    if vmin is None:
        vmin = float(np.min(finite)) if finite.size > 0 else 0.0
    if vmax is None:
        vmax = float(np.max(finite)) if finite.size > 0 else 0.0

    im = ax.imshow(
        arr, cmap=cmap, aspect="auto",
        origin="lower", vmin=vmin, vmax=vmax,
    )
    ax.set_xlabel("target channel b")
    ax.set_ylabel("source channel a")
    if title is None:
        title = f"response matrix (sigma={sigma_value})"
    ax.set_title(title, fontsize=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"MSE response (sigma={sigma_value})")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Comparison heatmap (1 x 3 across sigma values)
# ---------------------------------------------------------------------------

def plot_comparison(
    response_matrices: Dict[float, np.ndarray],
    save_path: str,
    title: Optional[str] = None,
    cmap: str = "viridis",
    shared_scale: bool = True,
):
    """Side-by-side heatmaps for multiple sigma values (1 row x N cols).

    Args:
        response_matrices: dict mapping sigma_value -> (n_src, n_tgt)
            matrix. The dict insertion order is preserved on the plot
            (left -> right). Python 3.7+ dicts are insertion-ordered so
            callers can rely on the order they built the dict in.
        save_path: output PNG path.
        title: optional super-title for the whole figure (printed above
            the subplot row via fig.suptitle).
        cmap: matplotlib colormap. Defaults to "viridis".
        shared_scale: when True (default), all subplots share the same
            color scale (computed from the global finite min / max).
            This makes cross-sigma magnitude comparison visual — the
            whole point of the comparison plot. When False, each subplot
            auto-scales to its own range.
    """
    if not response_matrices:
        raise ValueError("response_matrices must be non-empty")

    # Preserve dict insertion order so the plot columns are predictable.
    items = list(response_matrices.items())
    n_panels = len(items)

    # Shared color scale.
    if shared_scale:
        all_finite = np.concatenate(
            [_to_numpy_2d(m).flatten() for _, m in items]
        )
        all_finite = all_finite[np.isfinite(all_finite)]
        gmin = float(np.min(all_finite)) if all_finite.size > 0 else 0.0
        gmax = float(np.max(all_finite)) if all_finite.size > 0 else 0.0
    else:
        gmin = gmax = None

    # ~6 inches per panel keeps 128x128 legible without blowing past a
    # reasonable figure width. Cap total width at ~24 inches.
    panel_w = min(6.0, max(4.0, 24.0 / n_panels))
    fig, axes = plt.subplots(
        1, n_panels, figsize=(panel_w * n_panels, panel_w * 0.9),
        squeeze=False,
    )
    # Collect the axes into a list (axes.flat is a single-use iterator;
    # passing it directly to fig.colorbar after the for-loop above would
    # hand colorbar an exhausted iterator and crash with IndexError).
    axes_list = list(axes.flat)

    im = None
    for ax, (sigma_val, matrix) in zip(axes_list, items):
        arr = _to_numpy_2d(matrix)
        vmin = gmin if shared_scale else (
            float(np.min(arr[np.isfinite(arr)]))
            if np.isfinite(arr).any() else 0.0
        )
        vmax = gmax if shared_scale else (
            float(np.max(arr[np.isfinite(arr)]))
            if np.isfinite(arr).any() else 0.0
        )
        im = ax.imshow(
            arr, cmap=cmap, aspect="auto",
            origin="lower", vmin=vmin, vmax=vmax,
        )
        ax.set_xlabel("target channel b")
        ax.set_ylabel("source channel a")
        ax.set_title(f"sigma={sigma_val}", fontsize=9)

    # One shared colorbar on the right of the row. ``im`` is the last
    # ScalarMappable produced in the loop; the shared vmin/vmax above
    # guarantees all panels are on the same scale when shared_scale=True.
    if im is not None:
        fig.colorbar(im, ax=axes_list, fraction=0.046, pad=0.04,
                     label="MSE response")
    if title:
        fig.suptitle(title, fontsize=11)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Per-source out-degree histogram
# ---------------------------------------------------------------------------

def plot_out_degree_histogram(
    stats_dict: Dict[float, dict],
    save_path: str,
    title: Optional[str] = None,
    bins: int = 40,
):
    """Per-source out-degree (mean response over targets) histogram.

    For each sigma value, compute ``out_degree[a] = mean_b
    response_matrix[a, b]`` and plot its distribution. Overlays the 3
    sigma values on one axis so cross-sigma shifts are visible.

    Args:
        stats_dict: dict mapping sigma_value -> stats dict. The stats
            dict must contain a ``per_source_out_degree_mean`` key whose
            value is a list/array of length ``n_src`` (one out-degree
            per source channel). Produced by
            ``probe_perturbation_response.compute_stats``.
        save_path: output PNG path.
        title: optional plot title.
        bins: histogram bin count (default 40).
    """
    if not stats_dict:
        raise ValueError("stats_dict must be non-empty")

    fig, ax = plt.subplots(figsize=(10, 6))
    # Tab10 has enough distinct colors for 3-5 sigma values.
    cmap = plt.get_cmap("tab10")

    for i, (sigma_val, stats) in enumerate(stats_dict.items()):
        out_deg = np.asarray(
            stats["per_source_out_degree_mean"], dtype=np.float32,
        )
        out_deg = out_deg[np.isfinite(out_deg)]
        if out_deg.size == 0:
            continue
        ax.hist(
            out_deg, bins=bins, alpha=0.6,
            color=cmap(i % 10), edgecolor="white",
            label=f"sigma={sigma_val}",
        )

    ax.set_xlabel("per-source out-degree (mean response over targets)")
    ax.set_ylabel("source channel count")
    if title is None:
        title = "per-source out-degree distribution"
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_numpy_2d(x) -> np.ndarray:
    """Coerce torch tensor / list-of-lists to a 2-D numpy array.

    Accepts:
      - numpy ndarray (returned as-is, dtype float32 cast)
      - torch.Tensor (detached, moved to CPU, converted)
      - nested Python lists (np.asarray)

    Raises:
        ValueError: if the result is not 2-D.
    """
    if isinstance(x, np.ndarray):
        arr = x
    elif hasattr(x, "detach"):  # torch.Tensor (duck-typed)
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)
    arr = arr.astype(np.float32, copy=False)
    if arr.ndim != 2:
        raise ValueError(
            f"expected 2-D matrix; got shape {tuple(arr.shape)} (ndim={arr.ndim})"
        )
    return arr
