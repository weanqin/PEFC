"""Connectivity-matrix visualization helpers for sigma-based attribution.

Four plot types:
  - sigma heatmap (128 x 128): source-wise sigma matrix, viridis.
  - connectivity heatmap (128 x 128): -(log sigma + 0.5*log(2*pi*e)), hot.
  - sigma histogram: flattened sigma values, clamp edges annotated.
  - connectivity histogram: flattened connectivity values.

All plots use the Agg backend (no GUI) per logging-guidelines.md. Figures
are sized so a 128 x 128 matrix stays readable (~8 x 8 inch). Files are
saved at dpi=150 to balance size and legibility.

Shape note:
  sigma_matrix / connectivity_matrix may be rectangular during partial
  runs (n_sources_done x 128). The heatmaps handle this by reading the
  matrix shape directly rather than assuming square.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless; safe to import at module load
import matplotlib.pyplot as plt

from latent_flow.viz.nature_style import (
    apply_nature_style,
    PALETTE,
    save_nature,
)

# Apply the publication style once at import so every helper below inherits it.
apply_nature_style()

# Shared sigma-range convention for sigma-based attribution (see
# attribution/bisect_sigma.py). Defined locally here so the plotting
# helpers do not depend on any attribution implementation module; keep these
# in sync with the SIGMA_MIN/MAX in attribution/bisect_sigma.py.
SIGMA_MIN = 0.01
SIGMA_MAX = 0.3
SIGMA_INIT = 0.05  # historical JET/SGD init; kept for histogram annotation


# ---------------------------------------------------------------------------
# Heatmaps
# ---------------------------------------------------------------------------

def plot_sigma_matrix_heatmap(
    sigma_matrix,
    save_path: str,
    title: Optional[str] = None,
    cmap: str = "viridis",
    vmin: Optional[float] = SIGMA_MIN,
    vmax: Optional[float] = SIGMA_MAX,
):
    """Render the (n_src_done, 128) sigma matrix as a Nature-style heatmap.

    Args:
        sigma_matrix: 2-D array-like (numpy / torch / list-of-lists) of
            sigma values. Rows are source channels done so far, columns
            are target channels.
        save_path: output path (no extension needed). Saved as svg+pdf+png.
        title: optional plot title.
        cmap: matplotlib colormap. Defaults to "viridis" (sequential,
            perceptually uniform, colorblind-friendly).
        vmin / vmax: color scale bounds. Default to the sigma clamp range
            so heatmaps across subjects are directly comparable.
    """
    arr = _to_numpy_2d(sigma_matrix)
    n_src, n_tgt = arr.shape
    size = max(4.0, min(10.0, 8.0 * max(n_src, n_tgt) / 128.0))
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(
        arr, cmap=cmap, aspect="equal",
        origin="lower", vmin=vmin, vmax=vmax, interpolation="nearest",
    )
    ax.set_xlabel("target channel b")
    ax.set_ylabel("source channel a")
    if title:
        ax.set_title(title, fontsize=8)
    # Thin, unobtrusive colorbar matching the publication palette.
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$\sigma^\ast$", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    cbar.outline.set_linewidth(0.5)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    save_nature(fig, save_path)
    plt.close(fig)


def plot_connectivity_heatmap(
    connectivity_matrix,
    save_path: str,
    title: Optional[str] = None,
    cmap: str = "magma",
):
    """Render the (n_src_done, 128) connectivity matrix as a Nature-style heatmap.

    Connectivity = -(log sigma + 0.5*log(2*pi*e)); large = strong a->b.
    The colormap range auto-fits to the data so structure (bright rows / dark
    rows) is visible regardless of the subject's overall connectivity scale.
    Default cmap switched to "magma" (perceptually uniform, Nature-style).

    Args:
        connectivity_matrix: 2-D array-like of connectivity scores.
        save_path: output path (no extension needed). Saved as svg+pdf+png.
        title: optional plot title.
        cmap: matplotlib colormap. Defaults to "magma".
    """
    arr = _to_numpy_2d(connectivity_matrix)
    n_src, n_tgt = arr.shape
    size = max(4.0, min(10.0, 8.0 * max(n_src, n_tgt) / 128.0))
    fig, ax = plt.subplots(figsize=(size, size))
    # Auto-scale: pin at the matrix's finite range so NaN rows (failed
    # sources) don't blow out the colorbar.
    finite = arr[np.isfinite(arr)]
    if finite.size > 0:
        vmin = float(np.min(finite))
        vmax = float(np.max(finite))
    else:
        vmin = vmax = 0.0
    im = ax.imshow(
        arr, cmap=cmap, aspect="equal",
        origin="lower", vmin=vmin, vmax=vmax, interpolation="nearest",
    )
    ax.set_xlabel("target channel b")
    ax.set_ylabel("source channel a")
    if title:
        ax.set_title(title, fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("connectivity  $-H(a{\\to}b)$", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    cbar.outline.set_linewidth(0.5)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    save_nature(fig, save_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Histograms
# ---------------------------------------------------------------------------

def plot_sigma_histogram(
    sigma_matrix,
    save_path: str,
    title: Optional[str] = None,
    bins: int = 40,
):
    """Flattened sigma matrix histogram with clamp-range annotations.

    Args:
        sigma_matrix: 2-D array-like.
        save_path: output path (no extension needed). Saved as svg+pdf+png.
        title: optional plot title.
        bins: histogram bin count. Defaults to 40 (finer than the 20 used
            for the PR2 single-source validation because a full matrix
            has up to 128 x 128 = 16384 samples).
    """
    arr = _to_numpy_2d(sigma_matrix).flatten()
    # Drop NaN rows from failed sources so the histogram isn't dominated
    # by a single NaN bucket.
    arr = arr[np.isfinite(arr)]
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    if arr.size > 0:
        ax.hist(
            arr, bins=bins, range=(SIGMA_MIN, SIGMA_MAX),
            color=PALETTE["blue_main"], alpha=0.85, edgecolor="white",
            linewidth=0.3,
        )
    ax.axvline(
        SIGMA_MIN, color=PALETTE["red_strong"], linestyle="--", lw=0.8,
        label=rf"$\sigma_{{\min}}={SIGMA_MIN}$",
    )
    ax.axvline(
        SIGMA_MAX, color=PALETTE["red_strong"], linestyle="--", lw=0.8,
        label=rf"$\sigma_{{\max}}={SIGMA_MAX}$",
    )
    ax.axvline(
        SIGMA_INIT, color=PALETTE["green_3"], linestyle=":", lw=0.8,
        label=rf"$\sigma_{{\rm init}}={SIGMA_INIT}$",
    )
    ax.set_xlabel(r"$\sigma^\ast$ value")
    ax.set_ylabel("count")
    ax.set_xlim(SIGMA_MIN, SIGMA_MAX)
    if title:
        ax.set_title(title, fontsize=8)
    ax.legend(fontsize=6, loc="upper right")
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    save_nature(fig, save_path)
    plt.close(fig)


def plot_connectivity_histogram(
    connectivity_matrix,
    save_path: str,
    title: Optional[str] = None,
    bins: int = 40,
):
    """Flattened connectivity matrix histogram.

    Connectivity can include NaN (failed sources). NaNs are dropped
    before plotting. The x-axis auto-fits to the finite range.

    Args:
        connectivity_matrix: 2-D array-like.
        save_path: output path (no extension needed). Saved as svg+pdf+png.
        title: optional plot title.
        bins: histogram bin count.
    """
    arr = _to_numpy_2d(connectivity_matrix).flatten()
    arr = arr[np.isfinite(arr)]
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    if arr.size > 0:
        ax.hist(
            arr, bins=bins,
            color=PALETTE["teal"], alpha=0.85, edgecolor="white",
            linewidth=0.3,
        )
    ax.set_xlabel(r"connectivity  $-H(a{\to}b)$")
    ax.set_ylabel("count")
    if title:
        ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    save_nature(fig, save_path)
    plt.close(fig)


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
