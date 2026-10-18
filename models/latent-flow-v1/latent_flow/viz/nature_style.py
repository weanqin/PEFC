"""Nature-style figure aesthetics for OC-LatentFlow visualizations.

Single source of truth for the publication-style look across the connectivity
and EEG visualizations. Adapted from the nature-figure skill
(static/core/contract.md, static/fragments/backend/python.md, references/api.md):
editable SVG text, restrained palette, hidden top/right spines, Arial sans-serif,
multi-format export (svg + pdf + png).

The palette and helpers below mirror references/api.md so all figures share one
visual language. Import apply_nature_style() once before creating any figure.
"""

from __future__ import annotations

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Palette (verbatim from nature-figure references/api.md, curated subset)
# ---------------------------------------------------------------------------

PALETTE = {
    # signal family — the model / method under study
    "blue_main":      "#0F4D92",
    "blue_secondary": "#3775BA",
    "teal":   "#42949E",
    "violet": "#9A4D8E",
    # directional cues — reserve green/red for gain/drop or excite/inhibit
    "green_3":   "#8BCF8B",
    "red_strong": "#B64342",
    # neutral family — truth baselines, axes, grid
    "neutral_light": "#CFCECE",
    "neutral_mid":   "#767676",
    "neutral_dark":  "#4D4D4D",
    "neutral_black": "#272727",
}

# Truth vs prediction (used by the EEG time-domain + PSD panels).
TRUTH_COLOR = PALETTE["neutral_black"]      # reference / ground truth
PRED_COLOR = PALETTE["blue_main"]           # OC-LatentFlow forecast

# Signed diverging matrix (A_eff: excite red, inhibit blue).
DIVERGING_POS = PALETTE["red_strong"]       # positive = excitation
DIVERGING_NEG = PALETTE["blue_main"]        # negative = inhibition

# Sequential strength matrix (B connectivity, 0..strong).
SEQ_CMAP = "magma"                           # nature-figure default for heatmaps


# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------

def apply_nature_style(font_size: int = 7, axes_linewidth: float = 0.8):
    """Apply Nature-style rcParams. Call once before creating any figures.

    Presets (from references/api.md):
      - dense journal-width multi-panels: font_size=8,  axes_linewidth=1
      - compact figures:                  font_size=7,  axes_linewidth=0.8  (default)
      - large single panels:              font_size=16, axes_linewidth=2.5
    """
    # MANDATORY: editable text in SVG (api.md: non-negotiable, always first).
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans",
                                       "sans-serif"]
    plt.rcParams["svg.fonttype"] = "none"      # text stays <text>, not paths
    plt.rcParams["pdf.fonttype"] = 42          # editable TrueType in PDF
    # Layout & style.
    plt.rcParams["font.size"] = font_size
    plt.rcParams["axes.linewidth"] = axes_linewidth
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["xtick.major.width"] = axes_linewidth
    plt.rcParams["ytick.major.width"] = axes_linewidth
    plt.rcParams["xtick.major.size"] = 3
    plt.rcParams["ytick.major.size"] = 3


# ---------------------------------------------------------------------------
# Reusable helpers (adapted from references/api.md)
# ---------------------------------------------------------------------------

def add_panel_label(ax, label: str, x: float = -0.06, y: float = 1.02,
                    fontsize: int = 9, color: str = "black"):
    """Place a Nature-style bold panel label (a, b, c...) near top-left."""
    ax.text(x, y, label, transform=ax.transAxes, fontsize=fontsize,
            fontweight="bold", color=color, ha="left", va="bottom")


def save_nature(fig, out_path: str, formats=("png",),
                dpi: int = 600):
    """Save a figure as PNG (and optionally other formats).

    out_path is the path WITHOUT extension (or with one that gets stripped).
    Creates parent dirs. Default = png only (per project convention: raster
    preview is the only output). Pass formats=("svg","pdf","png") to also emit
    editable vector formats for manuscript submission.
    """
    base = out_path
    # strip any extension the caller passed
    for ext in ("svg", "pdf", "png", "tiff", "tif", "eps", "jpg"):
        if base.lower().endswith("." + ext):
            base = base[: -(len(ext) + 1)]
            break
    os.makedirs(os.path.dirname(os.path.abspath(base)) or ".", exist_ok=True)
    saved = []
    for fmt in formats:
        p = f"{base}.{fmt}"
        fig.savefig(p, format=fmt, dpi=dpi, bbox_inches="tight")
        saved.append(p)
    return saved


def is_dark(hex_color: str, threshold: float = 128) -> bool:
    """Return True if hex color is dark (use white text on it)."""
    c = hex_color.lstrip("#")
    r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    return (0.299 * r + 0.587 * g + 0.114 * b) < threshold
