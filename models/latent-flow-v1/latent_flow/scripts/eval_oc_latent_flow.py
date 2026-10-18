"""Diagnose OC-LatentFlow checkpoint (task 06-23 stage 1.2).

Two checks:
1. z_obs collapse: PCA on the encoder output across val subjects. The original
   latent_flow collapsed to PC1=98.7% (single effective dimension). Healthy
   per-channel temporal latent should spread variance across many components.
2. Multi-metric eval on held-out test subjects: MAE, Pearson, PSD-L1,
   std_ratio, coherence, diversity. Mirrors scripts/eval_model_compare.py.

Run (from repo root):

    D:\\program\\anaconda\\envs\\npi\\python.exe -m \\
        latent_flow.scripts.eval_oc_latent_flow \\
        --stage 1 --fold 0
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import glob

import numpy as np
import torch
import torch.utils.data as data

from latent_flow.config import DATA_ROOT, CHECKPOINT_DIR
from latent_flow.data.modma import MODMADataset, collate_fn
from latent_flow.data.splitter import SubjectFoldSplitter
from latent_flow.models.oc_latent_flow import OCLatentFlow
from latent_flow.viz.metrics import MetricAggregator

OBS_LEN = 375
FORECAST_LEN = 125


def _load_test_loader(fold, seed, batch_size=32):
    files_mdd = sorted(glob.glob(os.path.join(str(DATA_ROOT), "MDD", "*.mat")))
    files_hc = sorted(glob.glob(os.path.join(str(DATA_ROOT), "HC", "*.mat")))
    splitter = SubjectFoldSplitter(files_mdd, files_hc, n_folds=5, seed=seed)
    _, test_files = splitter.get_fold(fold)
    ds = MODMADataset(subject_files=test_files, group="MIX",
                      normalize_per_channel=True, obs_size=OBS_LEN)
    return data.DataLoader(ds, batch_size=batch_size, shuffle=False,
                           collate_fn=collate_fn)


def z_obs_pca_collapse(model, loader, device, max_samples=2000):
    """Flatten z_obs (B,C,T_lat,D) -> (N, C*T_lat*D) and run PCA.

    Returns the variance ratio of the top components. PC1 near 1.0 means
    collapse (the original latent_flow failure at PC1=98.7%).
    """
    model.eval()
    feats = []
    n = 0
    with torch.no_grad():
        for batch in loader:
            x_obs = batch["x_obs"].to(device)
            mu_obs, _ = model._encode_obs(x_obs)
            feats.append(mu_obs.reshape(mu_obs.shape[0], -1).cpu().numpy())
            n += x_obs.shape[0]
            if n >= max_samples:
                break
    feats = np.concatenate(feats, axis=0)[:max_samples]  # (N, C*T_lat*D)
    feats = feats - feats.mean(0, keepdims=True)
    # SVD-based PCA on the feature covariance
    U, S, Vt = np.linalg.svd(feats, full_matrices=False)
    var = (S ** 2)
    var_ratio = var / var.sum()
    eff_dim = (var_ratio ** 2).sum() ** -1  # effective dimensionality
    return var_ratio, eff_dim


def full_eval(model, loader, device, stage=2):
    """Multi-metric eval on held-out test. stage=1 uses proj bootstrap."""
    model.eval()
    agg = MetricAggregator()
    mae_sum, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            x_obs = batch["x_obs"].to(device)
            x_target = batch["x_target"].to(device)
            # stage 1 (VAE only): use proj bootstrap; flow is untrained.
            x_hat = model.sample(x_obs, n_steps=10, use_proj=(stage == 1))
            x_fc = x_target[:, OBS_LEN:, :]
            x_fc_hat = x_hat[:, OBS_LEN:, :]
            agg.update(x_fc_hat, x_fc)
            mae_sum += (x_fc_hat - x_fc).abs().mean().item() * x_obs.shape[0]
            n += x_obs.shape[0]
    metrics = agg.summary()
    metrics["mae"] = mae_sum / n
    return metrics


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--stage", type=int, default=1, choices=(1, 2))
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = OCLatentFlow()
    ckpt_name = "vae" if args.stage == 1 else "flow"
    ckpt_path = str(CHECKPOINT_DIR / f"fold{args.fold}_oc_latent_flow_{ckpt_name}_best.pt")
    # Route through load_checkpoint: handles both bare state-dicts (legacy)
    # and self-describing format_version=1 payloads (variant rebuild + unwrap),
    # so eval keeps working after a retrain that writes the new format.
    model = OCLatentFlow.load_checkpoint(ckpt_path, device)
    print(f"Loaded {ckpt_path}")

    loader = _load_test_loader(args.fold, args.seed)

    # --- z_obs collapse check ---
    print("\n=== z_obs PCA collapse check ===")
    var_ratio, eff_dim = z_obs_pca_collapse(model, loader, device)
    print(f"  PC1 = {var_ratio[0]*100:.1f}%   "
          f"(original latent_flow collapsed to PC1=98.7%)")
    print(f"  PC2 = {var_ratio[1]*100:.1f}%")
    print(f"  PC3 = {var_ratio[2]*100:.1f}%")
    print(f"  top-10 cumulative = {var_ratio[:10].sum()*100:.1f}%")
    print(f"  effective dimensionality = {eff_dim:.1f} / {model.n_channels*model.t_lat*model.latent_dim}")
    if var_ratio[0] > 0.5:
        print("  [WARN] PC1 > 50%%: possible collapse")
    else:
        print("  [OK] PC1 < 50%%: no single-axis collapse")

    # --- multi-metric eval ---
    print("\n=== Multi-metric eval (held-out test) ===")
    metrics = full_eval(model, loader, device, stage=args.stage)
    print(f"  MAE       = {metrics['mae']:.4f}")
    print(f"  Pearson   = {metrics['pearson']:.4f}   (JET v2 no_revin baseline: 0.0004)")
    print(f"  PSD-L1    = {metrics['psd_l1']:.4f}    (target <= 1.0)")
    print(f"  std_ratio = {metrics['std_ratio']:.4f}   (target >= 0.5)")
    print(f"  coherence = {metrics['coherence']:.4f}")
    print(f"  diversity = {metrics['diversity']:.4f}")


if __name__ == "__main__":
    main()
