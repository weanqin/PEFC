"""Ablation: decoder quality from ideal latent vs flow latent (task 06-23 diag).

Isolates whether the fc high-frequency deficit is a DECODER problem or a FLOW
problem. Three reconstructions of the same fc segment, same test batch:

  A. ideal: z_obs = enc(x_obs), z_fc = enc(x_fc)  <- decoder gets the TRUE fc latent
  B. flow:  z_obs = enc(x_obs), z_fc = flow(z_obs) <- the production path
  C. proj:  z_obs = enc(x_obs), z_fc = proj(z_obs) <- stage-1 bootstrap

If A has good high-freq / PSD but B/C don't, the DECODER is fine and the FLOW
(or proj) is the bottleneck (its z_fc lacks the high-freq structure that
enc(x_fc) carries). If A is ALSO smooth/low-freq, the DECODER itself can't
recover high-freq (its ConvTranspose+interpolate upsampling is too coarse, and
adding U-Net skips is the fix).

Run (from repo root):

    D:\\program\\anaconda\\envs\\npi\\python.exe -m \\
        latent_flow.scripts.diag_oc_latent_flow --fold 0
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
from latent_flow.viz.metrics import MetricAggregator

OBS_LEN = 375
FC_LEN = 125
T_FULL = 500
FS = 250.0


def _load_loader(fold, seed):
    files_mdd = sorted(glob.glob(os.path.join(str(DATA_ROOT), "MDD", "*.mat")))
    files_hc = sorted(glob.glob(os.path.join(str(DATA_ROOT), "HC", "*.mat")))
    splitter = SubjectFoldSplitter(files_mdd, files_hc, n_folds=5, seed=seed)
    _, test_files = splitter.get_fold(fold)
    ds = MODMADataset(subject_files=test_files, group="MIX",
                      normalize_per_channel=True, obs_size=OBS_LEN)
    return data.DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate_fn)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = str(CHECKPOINT_DIR / f"fold{args.fold}_oc_latent_flow_flow_best.pt")
    # Route through load_checkpoint: handles both bare state-dicts (legacy)
    # and self-describing format_version=1 payloads (variant rebuild + unwrap).
    model = OCLatentFlow.load_checkpoint(ckpt_path, device)
    model.to(device).eval()

    loader = _load_loader(args.fold, args.seed)

    paths = {"ideal": [], "flow": [], "proj": []}
    truth_list = []
    latent_dists = {"flow_vs_ideal": [], "proj_vs_ideal": []}

    with torch.no_grad():
        for batch in loader:
            x_obs = batch["x_obs"].to(device)
            x_target = batch["x_target"].to(device)
            x_fc = x_target[:, OBS_LEN:OBS_LEN + FC_LEN, :]

            mu_obs, _, skips = model._encode_obs(x_obs, return_skips=True)
            # ideal: encode the true fc (encoder accepts arbitrary length)
            z_fc_ideal, _ = model._encode_obs(x_fc)
            # flow: production path
            z = mu_obs
            for i in range(10):
                t = torch.full((z.shape[0],), i / 10, device=device)
                z = z + 0.1 * model.flow(z, t, mu_obs)
            z_fc_flow = z
            # proj: bootstrap
            z_fc_proj = model.proj(mu_obs)

            # skips come from the obs encoder (obs multi-scale features feed the
            # decoder regardless of which fc latent is used)
            x_ideal = model.decoder(mu_obs, z_fc_ideal, skips=skips)[:, OBS_LEN:, :]
            x_flow = model.decoder(mu_obs, z_fc_flow, skips=skips)[:, OBS_LEN:, :]
            x_proj = model.decoder(mu_obs, z_fc_proj, skips=skips)[:, OBS_LEN:, :]

            paths["ideal"].append(x_ideal.cpu())
            paths["flow"].append(x_flow.cpu())
            paths["proj"].append(x_proj.cpu())
            truth_list.append(x_fc.cpu())

            latent_dists["flow_vs_ideal"].append(
                (z_fc_flow - z_fc_ideal).pow(2).mean().item())
            latent_dists["proj_vs_ideal"].append(
                (z_fc_proj - z_fc_ideal).pow(2).mean().item())

    for k in paths:
        paths[k] = torch.cat(paths[k])
    truth = torch.cat(truth_list)

    print(f"=== Ablation: decoder from ideal vs flow vs proj latent ===")
    print(f"(test set: {truth.shape[0]} windows)\n")

    # Latent-space distance to the ideal z_fc
    print("Latent MSE to ideal z_fc (enc(x_fc)):")
    print(f"  flow: {np.mean(latent_dists['flow_vs_ideal']):.4f}")
    print(f"  proj: {np.mean(latent_dists['proj_vs_ideal']):.4f}")
    print(f"  (smaller = closer to the true fc latent)\n")

    # Signal-space multi-metric per path
    for name, pred in paths.items():
        agg = MetricAggregator()
        # process in chunks to bound memory
        step = 256
        for i in range(0, pred.shape[0], step):
            agg.update(pred[i:i+step], truth[i:i+step])
        s = agg.summary()
        mae = (pred - truth).abs().mean().item()
        print(f"{name:6s}: MAE={mae:.4f}  Pearson={s['pearson']:.4f}  "
              f"PSD-L1={s['psd_l1']:.4f}  std_ratio={s['std_ratio']:.4f}  "
              f"coherence={s['coherence']:.4f}")

    print("\nReading guide:")
    print(" - If IDEAL has good PSD-L1/std but FLOW/PROJ don't: DECODER is fine,")
    print("   FLOW/PROJ latent lacks high-freq -> improve flow, not decoder.")
    print(" - If IDEAL is ALSO smooth/low-freq: DECODER can't recover high-freq")
    print("   (ConvTranspose+interpolate too coarse) -> add U-Net skips to decoder.")

    # PSD plot: truth vs ideal vs flow on a few channels
    PSD_CH = [(0, "E1"), (32, "E33"), (64, "E65")]
    fig, axes = plt.subplots(1, len(PSD_CH), figsize=(4.2 * len(PSD_CH), 3.2))
    if len(PSD_CH) == 1:
        axes = [axes]
    n_plot = min(64, truth.shape[0])
    for ax_i, (ch, name) in enumerate(PSD_CH):
        ax = axes[ax_i]
        for sig, color, label, ls in [
            (truth[:n_plot], "black", "Truth", "--"),
            (paths["ideal"][:n_plot], "tab:green", "ideal z_fc", "-"),
            (paths["flow"][:n_plot], "tab:blue", "flow z_fc", "-"),
            (paths["proj"][:n_plot], "tab:red", "proj z_fc", ":"),
        ]:
            psds = []
            for b in range(sig.shape[0]):
                f, p = welch(sig[b, :, ch].numpy(), fs=FS, nperseg=min(FC_LEN, 64))
                psds.append(p)
            ax.semilogy(f, np.clip(np.mean(psds, 0), 1e-20, None),
                        color=color, alpha=0.8, lw=1.0, linestyle=ls, label=label)
        ax.set_xlim(0, 50); ax.set_title(f"{name} (fc)", fontsize=10)
        ax.set_xlabel("Freq (Hz)", fontsize=8)
        if ax_i == 0:
            ax.set_ylabel("PSD", fontsize=8); ax.legend(fontsize=7)
    plt.suptitle("OC-LatentFlow ablation: ideal vs flow vs proj latent PSD (fc)",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    out = "results/new_latent_flow/ablation_latent_psd.png"
    os.makedirs("results/new_latent_flow", exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"\nPSD plot saved: {out}")


if __name__ == "__main__":
    main()
