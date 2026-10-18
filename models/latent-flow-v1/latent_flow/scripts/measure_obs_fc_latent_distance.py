"""Measure latent-space distance: enc(x_obs) vs enc(x_fc) vs N(0,I).

Task 08-04-fm-similar-distribution-research (Part 2).

Question
--------
OC-LatentFlow's Stage-2 flow transports z_obs = enc(x_obs) -> z_fc = enc(x_fc)
(same frozen encoder, applied to the two adjacent segments of one EEG window).
Is this transport "small" relative to the noise->fc baseline N(0,I) -> enc(x_fc)?

This script is READ-ONLY over a frozen VAE checkpoint: it loads the encoder,
runs it on the obs and fc segments of a fold's held-out test windows, and
reports four distribution distances for two pairs:

    D_obs_fc   : distance(enc(x_obs), enc(x_fc))   # the real Stage-2 transport
    D_noise_fc : distance(N(0,I),        enc(x_fc)) # the noise->fc baseline

Headline number = D_obs_fc / D_noise_fc per metric (<<1 means obs and fc
latents are very close; ~1 means the obs->fc transport is as large as
starting from noise).

Latent layout & aggregation
---------------------------
ConvEncoder output (the VAE posterior mu) has shape (B, C=128, T_lat=16, D=64).
The KL term is applied per-element toward N(0,I), so the "natural" distance unit
is per-element. We report distances in three aggregation modes so the choice is
explicit and reproducible:

  - per-D     : average over (C, T_lat), giving a D=64-dim vector per window.
                Primary mode. Full-covariance Fréchet/Wasserstein are feasible
                at D=64 and this matches the per-latent-dimension KL.
  - per-ch    : average over D, giving a C*T_lat=2048-dim vector per window.
                Captures channel/time structure. Full covariance infeasible,
                so diagonal-Gaussian Fréchet/Wasserstein only.
  - flatten   : C*T_lat*D = 131072-dim vector per window. Diagonal-Gaussian
                Fréchet/Wasserstein only (full cov would be 128 GiB).

Distances are computed on the posterior mean mu (deterministic), which is what
Stage 2 uses (z_obs = mu_obs, z_fc_target = mu_fc at stage 2; see
oc_latent_flow.py:1097,1108).

Metrics
-------
1. per-dimension mean / std of each latent distribution
2. Fréchet distance. Full-covariance for per-D; diagonal-Gaussian
   (||m1-m2||^2 + sum((s1-s2)^2)) for per-ch and flatten.
3. MMD^2 (RBF kernel, median-heuristic bandwidth) — distribution-free.
   Skipped in flatten mode (dim=131072): the RBF bandwidth collapses and
   cdist is ~20s per kernel matrix, so the metric is both meaningless and
   costly at very high D.
4. 2-Wasserstein. Closed-form full-cov Gaussian for per-D; diagonal-Gaussian
   (||m1-m2||^2 + sum((sqrt(s1)-sqrt(s2))^2)) for per-ch and flatten.

Usage (from the PARENT of this repo, i.e. D:\\102-npi\\):

    D:/program/anaconda/envs/npi/python.exe -u -m \\
        latent_flow.scripts.measure_obs_fc_latent_distance --fold 1

Outputs a JSON record under results/ and prints a summary table.
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data
from scipy.linalg import sqrtm
from scipy.spatial.distance import cdist

from latent_flow.config import DATA_ROOT, CHECKPOINT_DIR
from latent_flow.data.modma import MODMADataset, collate_fn
from latent_flow.data.splitter import SubjectFoldSplitter
from latent_flow.models.oc_latent_flow import OCLatentFlow

OBS_LEN = 375


def _load_test_loader(fold, seed, batch_size=64):
    """Subject-level fold test loader (reuses eval_oc_latent_flow pattern)."""
    files_mdd = sorted(glob.glob(os.path.join(str(DATA_ROOT), "MDD", "*.mat")))
    files_hc = sorted(glob.glob(os.path.join(str(DATA_ROOT), "HC", "*.mat")))
    splitter = SubjectFoldSplitter(files_mdd, hc_files=files_hc, n_folds=5, seed=seed)
    _, test_files = splitter.get_fold(fold)
    ds = MODMADataset(subject_files=test_files, group="MIX",
                      normalize_per_channel=True, obs_size=OBS_LEN)
    return data.DataLoader(ds, batch_size=batch_size, shuffle=False,
                           collate_fn=collate_fn)


@torch.no_grad()
def collect_latents(model, loader, device, max_windows):
    """Run the frozen encoder on obs and fc segments of all test windows.

    Returns mu_obs, mu_fc as (N, C, T_lat, D) numpy arrays (posterior means,
    deterministic — matches Stage-2 usage). N = min(test set, max_windows).
    """
    mu_obs_list, mu_fc_list = [], []
    n = 0
    for batch in loader:
        x_obs = batch["x_obs"].to(device)            # (B, 375, 128)
        x_target = batch["x_target"].to(device)      # (B, 500, 128)
        x_fc = x_target[:, OBS_LEN:, :]              # (B, 125, 128)

        mu_obs, _ = model._encode_obs(x_obs)         # (B, 128, 16, 64)
        mu_fc, _ = model._encode_obs(x_fc)           # same encoder on fc segment

        mu_obs_list.append(mu_obs.cpu().numpy())
        mu_fc_list.append(mu_fc.cpu().numpy())
        n += x_obs.shape[0]
        if n >= max_windows:
            break

    mu_obs = np.concatenate(mu_obs_list, axis=0)[:max_windows]
    mu_fc = np.concatenate(mu_fc_list, axis=0)[:max_windows]
    return mu_obs, mu_fc


# ---------------------------------------------------------------------------
# Distance metrics
# ---------------------------------------------------------------------------

def gaussian_frechet(mu1, s1, mu2, s2):
    """Fréchet distance between two Gaussians: ||m1-m2||^2 + Tr(S1+S2-2*sqrt(S1S2))."""
    diff = mu1 - mu2
    covmean, _ = sqrtm(s1 @ s2, disp=False)
    # sqrtm can produce complex due to numerical noise; take real part.
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(s1 + s2 - 2.0 * covmean))


def mmd_rbf(x, y, bandwidth=None, max_samples=1000):
    """Unbiased MMD^2 with RBF kernel. Subsamples for O(N^2) tractability.

    Uses scipy.spatial.distance.cdist (BLAS-backed) for pairwise squared
    distances so the (N,N,D) difference tensor is never materialized — this
    is essential for the flatten mode (D=131072).

    x, y: (N, D). bandwidth=None -> median heuristic on pooled pairwise dists.
    """
    rng = np.random.default_rng(0)
    if x.shape[0] > max_samples:
        x = x[rng.choice(x.shape[0], max_samples, replace=False)]
    if y.shape[0] > max_samples:
        y = y[rng.choice(y.shape[0], max_samples, replace=False)]

    def sqdist(a, b):
        # squared Euclidean distance via cdist (no 3D tensor)
        return cdist(a, b, metric="sqeuclidean")

    if bandwidth is None:
        pooled = np.vstack([x, y])
        sub = pooled[rng.choice(pooled.shape[0], min(500, pooled.shape[0]),
                                replace=False)]
        d2 = sqdist(sub, sub)
        med = np.median(d2[d2 > 0])
        bandwidth = np.sqrt(med) if med > 0 else 1.0

    def k(a, b):
        return np.exp(-sqdist(a, b) / (2.0 * bandwidth ** 2))

    nx, ny = x.shape[0], y.shape[0]
    kxx = k(x, x)
    kyy = k(y, y)
    kxy = k(x, y)
    # unbiased estimator
    np.fill_diagonal(kxx, 0.0)
    np.fill_diagonal(kyy, 0.0)
    mxx = kxx.sum() / (nx * (nx - 1))
    myy = kyy.sum() / (ny * (ny - 1))
    mxy = kxy.mean()
    return float(mxx + myy - 2.0 * mxy)


def wasserstein2_gaussian(mu1, s1, mu2, s2):
    """Closed-form 2-Wasserstein between two Gaussians."""
    diff = mu1 - mu2
    s1_sqrt, _ = sqrtm(s1, disp=False)
    if np.iscomplexobj(s1_sqrt):
        s1_sqrt = s1_sqrt.real
    inner = s1 + s2 - 2.0 * (s1_sqrt @ s2 @ s1_sqrt)
    inner = np.maximum(inner.real if np.iscomplexobj(inner) else inner, 0.0)
    return float(diff @ diff + np.trace(inner))


def fit_gaussian(x, diagonal=False):
    """Mean and (co)variance of (N, D).

    diagonal=True -> return per-dim variance (D,), used when full covariance
    is infeasible (high D) or unwanted. diagonal=False -> full cov (D, D).
    """
    mu = x.mean(axis=0)
    if diagonal:
        sigma = x.var(axis=0)        # (D,) per-dim variance
    else:
        sigma = np.cov(x, rowvar=False)
    return mu, sigma


def frechet_diagonal(mu1, var1, mu2, var2):
    """Diagonal-Gaussian Fréchet: ||m1-m2||^2 + sum((v1-v2)^2)."""
    diff = mu1 - mu2
    return float(diff @ diff + np.sum((var1 - var2) ** 2))


def wasserstein2_diagonal(mu1, var1, mu2, var2):
    """Diagonal-Gaussian 2-Wasserstein: ||m1-m2||^2 + sum((sqrt(v1)-sqrt(v2))^2)."""
    diff = mu1 - mu2
    return float(diff @ diff + np.sum((np.sqrt(np.maximum(var1, 0))
                                       - np.sqrt(np.maximum(var2, 0))) ** 2))


def measure_pair(a, b, label, mode):
    """Compute all four distances between samples a, b in the given mode.

    mode: 'per-D'   -> rows averaged over (C,T_lat), leaving a D=64 vector.
                       Full-covariance Fréchet/Wasserstein (feasible at 64).
          'per-ch'  -> rows averaged over D, leaving a C*T_lat=2048 vector.
                       Diagonal-Gaussian Fréchet/Wasserstein.
          'flatten' -> C*T_lat*D=131072 vector per window.
                       Diagonal-Gaussian Fréchet/Wasserstein (full cov infeasible).
    """
    if mode == "per-D":
        ax = a.mean(axis=(1, 2))         # (N, 64)
        bx = b.mean(axis=(1, 2))
        diagonal = False
    elif mode == "per-ch":
        # average over D, keep (C, T_lat) flattened
        ax = a.mean(axis=3).reshape(a.shape[0], -1)   # (N, 2048)
        bx = b.mean(axis=3).reshape(b.shape[0], -1)
        diagonal = True
    elif mode == "flatten":
        ax = a.reshape(a.shape[0], -1)   # (N, 131072)
        bx = b.reshape(b.shape[0], -1)
        diagonal = True
    else:
        raise ValueError(f"unknown mode {mode!r}")

    mu_a, sig_a = fit_gaussian(ax, diagonal=diagonal)
    mu_b, sig_b = fit_gaussian(bx, diagonal=diagonal)

    if diagonal:
        frechet = frechet_diagonal(mu_a, sig_a, mu_b, sig_b)
        w2 = wasserstein2_diagonal(mu_a, sig_a, mu_b, sig_b)
        var_a_diag, var_b_diag = sig_a, sig_b
    else:
        frechet = gaussian_frechet(mu_a, sig_a, mu_b, sig_b)
        w2 = wasserstein2_gaussian(mu_a, sig_a, mu_b, sig_b)
        var_a_diag, var_b_diag = np.diag(sig_a), np.diag(sig_b)

    # MMD is only meaningful at moderate dim: the RBF kernel bandwidth
    # collapses at very high D (kernel matrix -> constant), giving a
    # meaningless ratio, and cdist over (500,131072) is ~20s each. Skip
    # MMD when dim > 4096 (flatten mode, dim=131072).
    if ax.shape[1] > 4096:
        mmd = float("nan")
    else:
        mmd = mmd_rbf(ax, bx)

    return {
        "label": label,
        "mode": mode,
        "covariance": "diagonal" if diagonal else "full",
        "n_samples_a": int(ax.shape[0]),
        "n_samples_b": int(bx.shape[0]),
        "dim": int(ax.shape[1]),
        "mean_a_norm": float(np.linalg.norm(mu_a)),
        "mean_b_norm": float(np.linalg.norm(mu_b)),
        "mean_diff_norm": float(np.linalg.norm(mu_a - mu_b)),
        "std_a_rms": float(np.sqrt((var_a_diag ** 2).mean())),
        "std_b_rms": float(np.sqrt((var_b_diag ** 2).mean())),
        "frechet": frechet,
        "mmd2": mmd,
        "wasserstein2": w2,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fold", type=int, default=1, choices=range(5),
                   help="which CV fold's test subjects to measure on")
    p.add_argument("--seed", type=int, default=42, help="fold split seed")
    p.add_argument("--checkpoint", type=str, default="",
                   help="VAE checkpoint path; default = fold{N}_oc_latent_flow_vae_best.pt")
    p.add_argument("--max-windows", type=int, default=4000,
                   help="cap on test windows (distance metrics are O(N^2))")
    p.add_argument("--noise-samples", type=int, default=4000,
                   help="number of N(0,I) samples to draw for the noise baseline")
    p.add_argument("--output", type=str, default="",
                   help="JSON output path; default = results/obs_fc_latent_distance/fold{N}.json")
    p.add_argument("--device", type=str, default="")
    args = p.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_path = (args.checkpoint
                 or str(CHECKPOINT_DIR / f"fold{args.fold}_oc_latent_flow_vae_best.pt"))
    print(f"Loading VAE checkpoint: {ckpt_path}")
    model = OCLatentFlow.load_checkpoint(ckpt_path, device)
    latent_shape = (model.n_channels, model.t_lat, model.latent_dim)  # (128,16,64)
    print(f"Latent shape (C, T_lat, D) = {latent_shape}")

    loader = _load_test_loader(args.fold, args.seed)
    print(f"Collecting latents on fold {args.fold} test set "
          f"(cap {args.max_windows} windows)...")
    mu_obs, mu_fc = collect_latents(model, loader, device, args.max_windows)
    print(f"  mu_obs: {mu_obs.shape}   mu_fc: {mu_fc.shape}")

    # N(0,I) baseline samples, same latent shape.
    rng = np.random.default_rng(0)
    noise = rng.standard_normal((args.noise_samples, *latent_shape)).astype(np.float32)
    print(f"  noise : {noise.shape}")

    results = {
        "fold": args.fold,
        "seed": args.seed,
        "checkpoint": ckpt_path,
        "latent_shape": list(latent_shape),
        "n_windows": int(mu_obs.shape[0]),
        "aggregation_modes": ["per-D", "per-ch", "flatten"],
        "note": ("Distances on posterior mean mu (deterministic, matches "
                 "Stage-2 usage z_obs=mu_obs, z_fc=mu_fc). per-D = mean over "
                 "(C,T_lat), full covariance; per-ch = mean over D, diagonal; "
                 "flatten = C*T_lat*D, diagonal."),
    }

    for mode in ["per-D", "per-ch", "flatten"]:
        print(f"\n=== Aggregation mode: {mode} ===")
        d_obs_fc = measure_pair(mu_obs, mu_fc, "obs_vs_fc", mode)
        d_noise_fc = measure_pair(noise, mu_fc, "noise_vs_fc", mode)

        # headline ratios (guard against zero baseline)
        def ratio(num, den):
            return float(num / den) if den > 0 else float("nan")

        ratios = {
            "frechet":     ratio(d_obs_fc["frechet"],     d_noise_fc["frechet"]),
            "mmd2":        ratio(d_obs_fc["mmd2"],        d_noise_fc["mmd2"]),
            "wasserstein2": ratio(d_obs_fc["wasserstein2"], d_noise_fc["wasserstein2"]),
        }

        results[f"obs_vs_fc_{mode}"] = d_obs_fc
        results[f"noise_vs_fc_{mode}"] = d_noise_fc
        results[f"ratio_obs_fc_over_noise_fc_{mode}"] = ratios

        print(f"  {'metric':<14} {'D(obs,fc)':>14} {'D(noise,fc)':>14} {'ratio':>10}")
        for m in ["frechet", "mmd2", "wasserstein2"]:
            r = ratios[m]
            rstr = f"{r:>10.4f}" if not np.isnan(r) else f"{'--':>10}"
            dval = d_obs_fc[m]
            dobs = f"{dval:>14.6g}" if not np.isnan(dval) else f"{'--':>14}"
            nval = d_noise_fc[m]
            dnoi = f"{nval:>14.6g}" if not np.isnan(nval) else f"{'--':>14}"
            print(f"  {m:<14} {dobs} {dnoi} {rstr}")
        print(f"  ratio interpretation: <1 = obs closer to fc than noise is; "
              f"~1 = obs as far from fc as noise")

    out_path = (Path(args.output) if args.output
                else Path("results") / "obs_fc_latent_distance" / f"fold{args.fold}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
