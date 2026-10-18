"""Reusable visualization for MODMA EEG experiments.

Replaces copy-pasted plotting logic across training scripts.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch


DEFAULT_CHANNELS = [
    (0, "E1"), (15, "E16"), (30, "E31"), (50, "E51"),
    (70, "E71"), (90, "E91"), (110, "E111"), (127, "E128"),
]

DEFAULT_Z_DIMS = [
    (0, "z0"), (6, "z1"), (12, "z2"), (24, "z4"),
    (48, "z8"), (96, "z16"), (192, "z32"), (384, "z64"),
]

DEFAULT_BAND_SHADES = [(0.5, 4), (4, 8), (8, 13), (13, 30)]


class Visualizer:
    """Parameterizable EEG visualization.

    Args:
        channel_selection: list of (idx, name) tuples for channel plots.
        z_selection: list of (dim, name) tuples for latent plots.
        fs: sampling rate in Hz.
        figsize_per_col: width (inches) per column.
        figsize_per_row: height (inches) per row.
        dpi: output resolution.
        band_shades: list of (low, high) Hz for background shading on PSD plots.
    """

    def __init__(
        self,
        channel_selection=None,
        z_selection=None,
        fs=250.0,
        figsize_per_col=3.0,
        figsize_per_row=2.5,
        dpi=150,
        band_shades=None,
    ):
        self.channel_selection = channel_selection or DEFAULT_CHANNELS
        self.z_selection = z_selection or DEFAULT_Z_DIMS
        self.fs = fs
        self.figsize_per_col = figsize_per_col
        self.figsize_per_row = figsize_per_row
        self.dpi = dpi
        self.band_shades = band_shades or DEFAULT_BAND_SHADES

    def _setup_axes(self, n_rows, n_cols):
        figsize = (self.figsize_per_col * n_cols, self.figsize_per_row * n_rows)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        return fig, axes

    def plot_time_domain(
        self,
        samples,
        save_path,
        forecast_boundary=None,
        labels=("True", "Pred"),
        colors=("b", "r"),
        n_show=3,
    ):
        """Plot time-domain comparison.

        Args:
            samples: list of dicts, each with keys:
                - ``t``: (T,) time array in seconds
                - ``true``: (T, n_channels) true signal
                - ``pred``: (T, n_channels) predicted signal
            save_path: output PNG path.
            forecast_boundary: float or None. If set, draws a vertical dashed line.
            labels: legend labels for (true, pred).
            colors: line colors for (true, pred).
            n_show: max number of samples (rows) to plot.
        """
        n_rows = min(n_show, len(samples))
        n_cols = len(self.channel_selection)
        fig, axes = self._setup_axes(n_rows, n_cols)

        for b_idx, sample in enumerate(samples[:n_rows]):
            t = sample["t"]
            true = sample["true"]
            pred = sample["pred"]

            for c_idx, (ch, name) in enumerate(self.channel_selection):
                ax = axes[b_idx, c_idx]
                ax.plot(t, true[:, ch], color=colors[0], alpha=0.6, lw=1, label=labels[0])
                ax.plot(t, pred[:, ch], color=colors[1], alpha=0.6, lw=1, label=labels[1])
                if forecast_boundary is not None:
                    ax.axvline(forecast_boundary, color="k", ls="--", lw=0.7, alpha=0.5)
                ax.set_title(f"{name} S{b_idx + 1}", fontsize=8)
                if b_idx == n_rows - 1:
                    ax.set_xlabel("Time (s)", fontsize=7)
                if c_idx == 0:
                    ax.set_ylabel("Amp", fontsize=7)
                ax.tick_params(labelsize=6)
                if b_idx == 0 and c_idx == 0:
                    ax.legend(fontsize=6)

        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()

    def plot_psd(
        self,
        samples,
        save_path,
        labels=("True", "Pred"),
        colors=("b", "r"),
        xlim=(0, 50),
        n_show=3,
    ):
        """Plot PSD comparison on forecast segment.

        Args:
            samples: list of dicts, each with keys:
                - ``true_fc``: (T_fc, n_channels) true forecast segment
                - ``pred_fc``: (T_fc, n_channels) predicted forecast segment
            save_path: output PNG path.
            labels, colors: same as ``plot_time_domain``.
            xlim: frequency axis limits.
            n_show: max number of samples (rows) to plot.
        """
        n_rows = min(n_show, len(samples))
        n_cols = len(self.channel_selection)
        fig, axes = self._setup_axes(n_rows, n_cols)

        for b_idx, sample in enumerate(samples[:n_rows]):
            true_fc = sample["true_fc"]
            pred_fc = sample["pred_fc"]

            for c_idx, (ch, name) in enumerate(self.channel_selection):
                ax = axes[b_idx, c_idx]
                f_t, psd_t = welch(true_fc[:, ch], fs=self.fs)
                f_p, psd_p = welch(pred_fc[:, ch], fs=self.fs)
                ax.semilogy(f_t, psd_t, color=colors[0], alpha=0.6, lw=1, label=labels[0])
                ax.semilogy(f_p, psd_p, color=colors[1], alpha=0.6, lw=1, label=labels[1])
                for low, high in self.band_shades:
                    ax.axvspan(low, high, alpha=0.03)
                ax.set_xlim(*xlim)
                ax.set_title(f"{name} S{b_idx + 1}", fontsize=8)
                if b_idx == n_rows - 1:
                    ax.set_xlabel("Freq (Hz)", fontsize=7)
                if c_idx == 0:
                    ax.set_ylabel("PSD", fontsize=7)
                ax.tick_params(labelsize=6)
                if b_idx == 0 and c_idx == 0:
                    ax.legend(fontsize=6)

        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()

    def plot_latent_trajectory(
        self,
        trajectories,
        save_path,
        z_selection=None,
        forecast_boundary=None,
        traj_label="z(t)",
        n_show=3,
    ):
        """Plot latent variable trajectories.

        Args:
            trajectories: list of dicts, each with keys:
                - ``t``: (T,) time array in seconds
                - ``z``: (T, latent_dim) latent trajectory
            save_path: output PNG path.
            z_selection: override ``self.z_selection``.
            forecast_boundary: vertical dashed line position.
            traj_label: legend label (e.g. ``"z(t)"`` or ``"z (constant)"``).
            n_show: max number of samples (rows) to plot.
        """
        z_sel = z_selection or self.z_selection
        n_rows = min(n_show, len(trajectories))
        n_cols = len(z_sel)
        fig, axes = self._setup_axes(n_rows, n_cols)

        for b_idx, traj in enumerate(trajectories[:n_rows]):
            t = traj["t"]
            z = traj["z"]

            for c_idx, (dim, name) in enumerate(z_sel):
                ax = axes[b_idx, c_idx]
                ax.plot(t, z[:, dim], "g-", alpha=0.6, lw=1, label=traj_label)
                if forecast_boundary is not None:
                    ax.axvline(forecast_boundary, color="k", ls="--", lw=0.7, alpha=0.5)
                ax.set_title(f"{name} S{b_idx + 1}", fontsize=8)
                if b_idx == n_rows - 1:
                    ax.set_xlabel("Time (s)", fontsize=7)
                if c_idx == 0:
                    ax.set_ylabel(traj_label, fontsize=7)
                ax.tick_params(labelsize=6)
                if b_idx == 0 and c_idx == 0:
                    ax.legend(fontsize=6)

        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()
