"""Visualize OC-LatentFlow waveforms (task 06-23 stage 1.2 viz).

Plots truth vs prediction on the held-out fold-0 test subjects. The time-domain
plot shows the FULL window (obs 375 + fc 125) so both the reconstruction of the
known obs segment and the forecast of the fc segment are visible, with a marker
at the obs->fc boundary. The PSD plot focuses on the fc segment (it evaluates
forecast quality; the obs segment is a known-input reconstruction).

Run (from repo root):

    D:\\program\\anaconda\\envs\\npi\\python.exe -m \\
        latent_flow.scripts.plot_oc_latent_flow \\
        --stage 1 --fold 0

Outputs:
    results/new_latent_flow/vae_time.png   (or flow_time.png for stage 2)
    results/new_latent_flow/vae_psd.png    (or flow_psd.png for stage 2)
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import glob

import numpy as np
import torch
import torch.utils.data as data
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch

from latent_flow.config import DATA_ROOT, CHECKPOINT_DIR
from latent_flow.data.modma import MODMADataset, collate_fn
from latent_flow.data.splitter import SubjectFoldSplitter
from latent_flow.models.oc_latent_flow import OCLatentFlow
from latent_flow.scripts._connectivity_driver import load_model
from latent_flow.viz.nature_style import (
    apply_nature_style,
    add_panel_label,
    PALETTE,
    save_nature,
)

# Publication style for every figure this script emits.
apply_nature_style(font_size=7, axes_linewidth=0.8)

OBS_LEN = 375
FC_LEN = 125
T_FULL = OBS_LEN + FC_LEN
FS = 250.0
BOUNDARY_S = OBS_LEN / FS  # 1.5s, obs->fc boundary

# Channel subsets (match ablate_z_obs_compare.py convention)
TIME_CHANNEL_SEL = [(0, "E1"), (32, "E33"), (64, "E65")]
PSD_CHANNEL_SEL = [(0, "E1"), (32, "E33"), (64, "E65"), (127, "E128")]
N_SAMPLES = 3


def _load_test_loader(fold, seed, batch_size=8):
    files_mdd = sorted(glob.glob(os.path.join(str(DATA_ROOT), "MDD", "*.mat")))
    files_hc = sorted(glob.glob(os.path.join(str(DATA_ROOT), "HC", "*.mat")))
    splitter = SubjectFoldSplitter(files_mdd, files_hc, n_folds=5, seed=seed)
    _, test_files = splitter.get_fold(fold)
    ds = MODMADataset(subject_files=test_files, group="MIX",
                      normalize_per_channel=True, obs_size=OBS_LEN)
    return data.DataLoader(ds, batch_size=batch_size, shuffle=False,
                           collate_fn=collate_fn)


def _collect_samples(model, loader, device, n_samples=N_SAMPLES, stage=2):
    """Take the first batch, predict, return FULL-window truth + pred.

    Returns arrays of shape (n_samples, T_FULL=500, C). The caller slices
    obs/fc as needed. stage=1 uses proj bootstrap (flow untrained).
    """
    model.eval()
    batch = next(iter(loader))
    x_obs = batch["x_obs"].to(device)
    x_target = batch["x_target"].to(device)
    x_hat = model.sample(x_obs, n_steps=10, use_proj=(stage == 1))
    truth = x_target[:, :T_FULL, :].detach().cpu()
    pred = x_hat[:, :T_FULL, :].detach().cpu()
    return truth[:n_samples], pred[:n_samples]


def plot_time_domain(truth, pred, stage, save_dir, variant=""):
    """Nature-style full-window overlay: obs (reconstruction) + fc (forecast).

    Rows = samples, cols = channels. Truth (neutral-black dashed) and
    prediction (signal-blue solid) over the whole 0..2s window. A shaded band
    + boundary line marks the obs->fc boundary at 1.5s, separating
    reconstruction (left) from forecast (right).
    """
    t = np.arange(T_FULL) / FS
    n_rows = N_SAMPLES
    n_cols = len(TIME_CHANNEL_SEL)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.3 * n_cols, 1.5 * n_rows),
                             sharex=True)

    label = f"OC-LatentFlow ({stage})"
    for r in range(n_rows):
        for c_idx, (ch, name) in enumerate(TIME_CHANNEL_SEL):
            ax = axes[r, c_idx]
            # Shade the fc forecast region (restrained accent).
            ax.axvspan(BOUNDARY_S, T_FULL / FS,
                       color=PALETTE["red_strong"], alpha=0.07)
            # Boundary marker.
            ax.axvline(BOUNDARY_S, color=PALETTE["neutral_mid"],
                       alpha=0.8, lw=0.7, linestyle=":")

            ax.plot(t, truth[r, :, ch].numpy(), color=PALETTE["neutral_black"],
                    alpha=0.8, lw=1.0, linestyle="--", label="Truth")
            ax.plot(t, pred[r, :, ch].numpy(), color=PALETTE["blue_main"],
                    alpha=0.9, lw=0.9, label=label)

            ax.set_title(f"{name} · S{r + 1}", fontsize=7, pad=2)
            if r == n_rows - 1:
                ax.set_xlabel("Time (s)", fontsize=7)
            if c_idx == 0:
                ax.set_ylabel("Amplitude", fontsize=7)
            ax.tick_params(labelsize=6)
            # obs | fc boundary annotation (top row only).
            if r == 0:
                ax.text(BOUNDARY_S / 2, ax.get_ylim()[1], "obs", fontsize=6,
                        color=PALETTE["neutral_mid"], ha="center", va="bottom")
                ax.text((BOUNDARY_S + T_FULL / FS) / 2, ax.get_ylim()[1],
                        "fc", fontsize=6, color=PALETTE["neutral_mid"],
                        ha="center", va="bottom")
            if r == 0 and c_idx == 0:
                ax.legend(fontsize=6, loc="upper right")
                add_panel_label(ax, "a", x=-0.18, y=1.15, fontsize=9)

    var_tag = f"  ({variant})" if variant else ""
    fig.suptitle(
        f"OC-LatentFlow {stage} — obs reconstruction + fc forecast vs truth"
        f"{var_tag}  (fold 0 test)",
        fontsize=8, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    base = os.path.join(save_dir, f"{stage.lower()}_time")
    saved = save_nature(fig, base)
    plt.close(fig)
    print(f"  saved {saved}")


def plot_psd(truth, pred, stage, save_dir, variant=""):
    """Nature-style mean PSD of the fc segment per channel: truth vs prediction.

    PSD evaluates forecast quality, so only the fc segment is plotted (the obs
    segment is a known-input reconstruction). Truth = neutral-black dashed
    reference; prediction = signal-blue solid.
    """
    truth_fc = truth[:, OBS_LEN:, :]
    pred_fc = pred[:, OBS_LEN:, :]
    n = len(PSD_CHANNEL_SEL)
    fig, axes = plt.subplots(1, n, figsize=(2.0 * n, 1.9), sharey=False)
    if n == 1:
        axes = [axes]

    label = f"OC-LatentFlow ({stage})"
    for ax_idx, (ch, name) in enumerate(PSD_CHANNEL_SEL):
        ax = axes[ax_idx]

        truth_psds = []
        for b in range(truth_fc.shape[0]):
            f_t, p_t = welch(truth_fc[b, :, ch].numpy(), fs=FS,
                             nperseg=min(FC_LEN, 64))
            truth_psds.append(p_t)
        ax.semilogy(f_t, np.mean(truth_psds, axis=0),
                    color=PALETTE["neutral_black"], alpha=0.85, lw=1.2,
                    linestyle="--", label="Truth")

        pred_psds = []
        for b in range(pred_fc.shape[0]):
            f_p, p_p = welch(pred_fc[b, :, ch].numpy(), fs=FS,
                             nperseg=min(FC_LEN, 64))
            pred_psds.append(p_p)
        mean_pred = np.clip(np.mean(pred_psds, axis=0), 1e-20, None)
        ax.semilogy(f_p, mean_pred, color=PALETTE["blue_main"], alpha=0.85,
                    lw=0.9, label=label)

        ax.set_xlim(0, 50)
        ax.set_title(f"{name} (fc)", fontsize=7, pad=2)
        ax.set_xlabel("Frequency (Hz)", fontsize=7)
        ax.tick_params(labelsize=6)
        if ax_idx == 0:
            ax.set_ylabel("PSD", fontsize=7)
            ax.legend(fontsize=6, loc="upper right")
            add_panel_label(ax, "a", x=-0.30, y=1.20, fontsize=9)

    var_tag = f"  ({variant})" if variant else ""
    fig.suptitle(f"OC-LatentFlow {stage} — fc PSD vs truth{var_tag}  (fold 0 test)",
                 fontsize=8, y=1.04)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    base = os.path.join(save_dir, f"{stage.lower()}_psd")
    saved = save_nature(fig, base)
    plt.close(fig)
    print(f"  saved {saved}")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--stage", type=int, default=1, choices=(1, 2))
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint", type=str, default="",
                   help="explicit checkpoint path (auto-detects graph/mexhat). "
                        "Empty: stage 1 = vae, stage 2 = dense flow.")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    save_dir = "results/new_latent_flow"
    os.makedirs(save_dir, exist_ok=True)

    # Pick checkpoint. An explicit --checkpoint can be any variant (dense,
    # gaussian-graph, mexhat) — load_model auto-detects and injects the
    # electrode coords the graph-mode mixer needs. Fallback keeps the old
    # stage-based default for backward compatibility.
    if args.checkpoint:
        ckpt_path = args.checkpoint
        model = load_model(ckpt_path, device)
        variant = getattr(model, "flow_graph_kernel", "dense")
    else:
        ckpt_name = "vae" if args.stage == 1 else "flow"
        ckpt_path = str(CHECKPOINT_DIR
                        / f"fold{args.fold}_oc_latent_flow_{ckpt_name}_best.pt")
        model = OCLatentFlow()
        model.load_state_dict(
            torch.load(ckpt_path, map_location=device, weights_only=False))
        model.to(device)
        model.eval()
        variant = "dense"
    print(f"Loaded {ckpt_path}  (variant: {variant})")

    loader = _load_test_loader(args.fold, args.seed)
    stage_label = "VAE" if args.stage == 1 else "Flow"

    print(f"\nGenerating {stage_label} waveform visualizations...")
    truth, pred = _collect_samples(model, loader, device, stage=args.stage)
    plot_time_domain(truth, pred, stage_label, save_dir, variant=variant)
    plot_psd(truth, pred, stage_label, save_dir, variant=variant)

    # std_ratio split by segment (obs reconstruction vs fc forecast)
    print("\nPer-sample std_ratio (pred std / truth std):")
    for r in range(N_SAMPLES):
        sr_obs = (pred[r, :OBS_LEN].std() / (truth[r, :OBS_LEN].std() + 1e-8)).item()
        sr_fc = (pred[r, OBS_LEN:].std() / (truth[r, OBS_LEN:].std() + 1e-8)).item()
        print(f"  S{r+1}: obs={sr_obs:.3f}  fc={sr_fc:.3f}")


if __name__ == "__main__":
    main()
