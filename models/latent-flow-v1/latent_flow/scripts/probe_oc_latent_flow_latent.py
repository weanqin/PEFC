"""Latent-space perturbation-response probe for OC-LatentFlow (2026-06-28).

Companion to ``scripts/probe_oc_latent_flow.py``. That probe perturbs the
raw signal ``x_obs[:,:,a]`` and measures the MSE response on the fc
waveform ``x_fc`` — i.e. it walks the full encoder → flow → decoder path.
This probe perturbs the **flow input latent** ``z_obs[:, a, :, :]`` directly
and measures the response on the **flow output latent** ``z_fc_hat`` —
bypassing both the encoder (per-channel gain distortion) and the decoder
(z_obs skip masking). It isolates the flow's own channel-interaction
structure: ``z_obs_a → z_fc_b``.

Motivation
----------
The flow is the architecture's only cross-channel mixing site (encoder is
per-channel; decoder folds channels into batch). The channel mixing happens
at ``FlowVelocityNet``'s reshape+input_proj (a dense Linear mixing the full
128×64 latent) and the per-block 1×1 convs. Whether that mixing produces
*structured* a→b connectivity (not just diffuse coupling) is exactly what
this probe tests — without encoder/decoder confounds.

Perturbation model
------------------
z_obs has shape (B, C=128, T_lat=16, D=64). Channel ``a`` is a clean slice
``z_obs[:, a, :, :]`` (shape (B, 16, 64)). For source channel a:

    xi ~ N(0, 1) shape (B, T_lat, D), seeded per source
    z_obs_pert[:, a, :, :] = z_obs[:, a, :, :] + sigma * xi
    z_fc_pert = euler(flow, start=z_obs_pert, cond=z_obs_pert)   # mode "coupled"
    response[a, b] = MSE(z_fc_pert[:, b, :, :], z_fc_base[:, b, :, :])

Because z_obs is both the ODE start state AND the velocity condition in
``sample``, perturbing it touches both (this mirrors real inference, where
start and condition are the same z_obs). The ``--condition-mode`` flag
exposes the alternative: perturb the start only, keep the condition clean
(``cond=z_obs``), which measures the pure transport response.

Sigma units
-----------
Latent-space sigma is NOT comparable to signal-space sigma (the encoder's
per-channel gain maps one to the other, channel-dependently). To keep the
amplitude meaningful and roughly channel-fair, sigma is interpreted
RELATIVE to each channel's own latent std by default
(``z_obs[:,a].std() * sigma``); pass ``--absolute-sigma`` to perturb in raw
latent units instead. CV (std/mean of response) is sigma-scale-invariant
and the headline metric regardless of the unit.

Output (relative to ):
    results/probe_oc_latent/probe_subject_{idx}.json
    outputs/probe_oc_latent/subject_{idx}_sigma_{sigma}_heatmap.png
    outputs/probe_oc_latent/subject_{idx}_comparison.png
    outputs/probe_oc_latent/subject_{idx}_out_degree_hist.png

Usage (from repo root):

    D:/program/anaconda/envs/npi/python.exe -u -m \\
        latent_flow.scripts.probe_oc_latent_flow_latent \\
        --fold 0 --subjects 1 --sources 128
"""

import argparse
import json
import math
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import numpy as np
import torch

from latent_flow.config import CHECKPOINT_DIR
from latent_flow.scripts._connectivity_driver import (
    N_CHANNELS,
    OBS_LEN,
    compute_stats,
    get_fold_test_subjects,
    json_default,
    load_model,
    load_subject_trials,
)
from latent_flow.viz.probe import (
    plot_comparison,
    plot_out_degree_histogram,
    plot_response_heatmap,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR = "results/probe_oc_latent"
DEFAULT_OUTPUTS_DIR = "outputs/probe_oc_latent"

DEFAULT_SIGMAS = "0.1,0.2,0.3"

# JET v2 baseline CV (from 06-16 probe / domain.md), for the comparison print.
JET_CV_RANGE = (0.095, 0.175)

N_STEPS = 10  # OC flow Euler steps (quality-guidelines)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Latent-space perturbation-response probe for OC-LatentFlow. "
            "Perturbs z_obs[:, a] (the flow's input latent) and measures "
            "MSE response on z_fc_hat (the flow's output latent), "
            "isolating the flow's own channel-interaction structure."
        ),
    )
    p.add_argument("--subjects", type=int, default=1,
                   help="number of fold test subjects to process (default 1).")
    p.add_argument("--sources", type=int, default=N_CHANNELS,
                   help=f"number of source channels to perturb (default {N_CHANNELS}).")
    p.add_argument("--checkpoint", type=str, default="",
                   help="OC-LatentFlow flow checkpoint (stage-2). Empty uses "
                        "checkpoints/fold{fold}_oc_latent_flow_flow_best.pt.")
    p.add_argument("--sigmas", type=str, default=DEFAULT_SIGMAS,
                   help=f"comma-separated sigma values (default '{DEFAULT_SIGMAS}').")
    p.add_argument("--absolute-sigma", action="store_true",
                   help="interpret sigma in raw latent units. Default "
                        "(unset) scales sigma by each channel's own latent "
                        "std (z_obs[:,a].std()*sigma) so amplitude is "
                        "channel-fair and roughly comparable to signal-space.")
    p.add_argument("--condition-mode", type=str, default="coupled",
                   choices=("coupled", "start_only"),
                   help="how the perturbed z_obs enters the flow. 'coupled' "
                        "(default): perturbed z_obs is BOTH the ODE start and "
                        "the velocity condition (mirrors real inference). "
                        "'start_only': perturbed z_obs is the start, but the "
                        "condition stays the clean z_obs (pure transport "
                        "response).")
    p.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                   help="directory for per-subject JSON results.")
    p.add_argument("--outputs-dir", type=str, default=DEFAULT_OUTPUTS_DIR,
                   help="directory for per-subject PNG visualizations.")
    p.add_argument("--subject-idx", type=int, default=0,
                   help="0-based subject index in the fold test (default 0).")
    p.add_argument("--fold", type=int, default=0, choices=range(5),
                   help="fold index for the SubjectFoldSplitter (default 0).")
    p.add_argument("--seed", type=int, default=42,
                   help="random seed for the subject splitter (default 42).")
    p.add_argument("--device", type=str, default="",
                   help="torch device override. Empty uses cuda.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Flow-only forward (encoder once, then latent-space Euler)
# ---------------------------------------------------------------------------

def encode_obs(model, x_obs):
    """Run the frozen encoder once; return z_obs = mu_obs (deterministic)."""
    mu_obs, _, _ = model._encode_obs(x_obs, return_skips=True)
    return mu_obs  # (B, C, T_lat, D)


@torch.no_grad()
def flow_euler(model, z_start, z_cond, n_steps=N_STEPS):
    """Integrate dz/dt = v(z, t, z_cond) from t=0 to t=1 (Euler).

    z_start, z_cond: (B, C, T_lat, D). Returns z_fc_hat (same shape).
    Mirrors OCLatentFlow.sample's flow loop, but takes start and condition
    as separate args so the probe can decouple them (start_only mode).
    """
    z = z_start
    B = z.shape[0]
    for i in range(n_steps):
        t = torch.full((B,), i / n_steps, device=z.device)
        v = model.flow(z, t, z_cond)
        z = z + (1.0 / n_steps) * v
    return z


def forecast_latent(model, x_obs, condition_mode="coupled"):
    """Baseline latent forecast: encode -> flow Euler -> z_fc_base.

    condition_mode selects how start/condition are coupled in the baseline
    (must match the per-source perturbed forward so baseline and perturbed
    paths differ ONLY by the perturbation). For the clean baseline both
    start and condition are the unperturbed z_obs regardless of mode.
    """
    z_obs = encode_obs(model, x_obs)
    return flow_euler(model, z_obs, z_obs)


# ---------------------------------------------------------------------------
# Probe core
# ---------------------------------------------------------------------------

@torch.no_grad()
def probe_single_sigma(model, x_obs, z_fc_base, sigma, n_sources, device,
                       absolute_sigma=False, condition_mode="coupled",
                       seed=42, progress_prefix=""):
    """Perturb every source channel with amplitude ``sigma``; return response.

    response[a, b] = MSE over (batch, T_lat, D) of
        z_fc_pert[:, b] vs z_fc_base[:, b]
    when channel a's latent is perturbed.

    Sigma handling:
      - absolute_sigma=False (default): per-channel relative, the actual
        perturbation on channel a is z_obs[:,a].std()*sigma. Channel-fair.
      - absolute_sigma=True: raw latent units, sigma applied uniformly.

    condition_mode:
      - 'coupled': start = cond = z_obs_pert (mirrors inference).
      - 'start_only': start = z_obs_pert, cond = z_obs (pure transport).

    Returns (n_sources, C) float32 numpy array of per-target MSE response.
    """
    N = x_obs.shape[0]
    response = np.zeros((n_sources, N_CHANNELS), dtype=np.float32)

    # Encode once; reuse for all sources (encoder is deterministic at stage 2).
    z_obs = encode_obs(model, x_obs)  # (B, C, T_lat, D)

    # Per-channel latent std (over batch, T_lat, D) for relative sigma.
    if not absolute_sigma:
        per_ch_std = z_obs.std(dim=(0, 2, 3)).clamp_min(1e-8)  # (C,)

    for src in range(n_sources):
        src_t0 = time.time()
        # Fresh generator per source so noise is independent across sources.
        noise_gen = torch.Generator(device=device).manual_seed(seed * 1000 + src + 1)
        xi = torch.randn(N, 16, model.latent_dim, device=device,
                         generator=noise_gen)  # (B, T_lat, D)

        amp = (sigma if absolute_sigma
               else float(sigma) * per_ch_std[src].item())
        z_obs_pert = z_obs.clone()
        z_obs_pert[:, src, :, :] = z_obs_pert[:, src, :, :] + amp * xi

        if condition_mode == "coupled":
            z_fc_pert = flow_euler(model, z_obs_pert, z_obs_pert)
        else:  # start_only
            z_fc_pert = flow_euler(model, z_obs_pert, z_obs)

        sq = (z_fc_pert.float() - z_fc_base.float()) ** 2  # (B, C, T_lat, D)
        # MSE per target channel: mean over batch, T_lat, D.
        response[src, :] = sq.mean(dim=(0, 2, 3)).cpu().numpy()

        if src % 16 == 0 or src == n_sources - 1:
            dt = time.time() - src_t0
            print(f"{progress_prefix}src={src:3d}/{n_sources} "
                  f"amp={amp:.4e} "
                  f"mean_resp={float(response[src].mean()):.4e} "
                  f"max_resp={float(response[src].max()):.4e} "
                  f"({dt:.2f}s)")

    return response


# ---------------------------------------------------------------------------
# Per-subject driver
# ---------------------------------------------------------------------------

def run_subject(model, x_obs_subject, subject_idx, subject_file, sigmas,
                n_sources, absolute_sigma, condition_mode,
                output_dir, outputs_dir, checkpoint_name, device):
    n_trials = x_obs_subject.shape[0]
    print(f"\n=== subject {subject_idx} "
          f"({os.path.basename(subject_file)}) === "
          f"{n_trials} trials, probing {n_sources} sources x {len(sigmas)} sigmas "
          f"[latent-space, mode={condition_mode}, "
          f"sigma={'absolute' if absolute_sigma else 'relative'}]")

    base_t0 = time.time()
    z_fc_base = forecast_latent(model, x_obs_subject, condition_mode)
    base_dt = time.time() - base_t0
    print(f"baseline latent forward: {base_dt:.2f}s, "
          f"z_fc_base mean={float(z_fc_base.mean()):.4e} "
          f"std={float(z_fc_base.std()):.4e} "
          f"abs_max={float(z_fc_base.abs().max()):.4e}")

    response_matrices = {}
    stats_per_sigma = {}

    for sigma in sigmas:
        sigma_t0 = time.time()
        print(f"\n--- sigma={sigma} --- perturbing {n_sources} source channels")
        response = probe_single_sigma(
            model=model, x_obs=x_obs_subject, z_fc_base=z_fc_base,
            sigma=sigma, n_sources=n_sources, device=device,
            absolute_sigma=absolute_sigma, condition_mode=condition_mode,
            progress_prefix=f"subject {subject_idx} / sigma={sigma} / ",
        )
        response_matrices[sigma] = response
        stats = compute_stats(response)
        stats_per_sigma[sigma] = stats
        sigma_dt = time.time() - sigma_t0
        print(f"sigma={sigma} done in {sigma_dt:.2f}s ({sigma_dt / 60.0:.2f} min) | "
              f"mean={stats['mean']:.4e} std={stats['std']:.4e} "
              f"CV={stats['cv'] * 100:.2f}%")

    print("\n--- channel-sensitivity CV vs JET v2 ceiling ---")
    for sigma in sigmas:
        s = stats_per_sigma[sigma]
        verdict = ("ABOVE ceiling (per-channel latent helps)"
                   if s["cv"] > JET_CV_RANGE[1]
                   else "AT/BELOW ceiling (still limited)")
        print(f"sigma={sigma}: CV={s['cv'] * 100:.2f}%  "
              f"[JET v2 range {JET_CV_RANGE[0] * 100:.1f}%-"
              f"{JET_CV_RANGE[1] * 100:.1f}%] -> {verdict}")

    # ----- save PNGs ---------------------------------------------------
    for sigma in sigmas:
        plot_response_heatmap(
            response_matrix=response_matrices[sigma],
            sigma_value=sigma,
            save_path=os.path.join(outputs_dir,
                                   f"subject_{subject_idx}_sigma_{sigma}_heatmap.png"),
            title=(f"OC-LatentFlow LATENT response  subj {subject_idx}  "
                   f"sigma={sigma} ({'abs' if absolute_sigma else 'rel'}, "
                   f"{condition_mode})"),
        )
    plot_comparison(
        response_matrices=response_matrices,
        save_path=os.path.join(outputs_dir, f"subject_{subject_idx}_comparison.png"),
        title=(f"OC-LatentFlow LATENT response comparison  subj {subject_idx}  "
               f"({condition_mode})"),
    )
    plot_out_degree_histogram(
        stats_dict=stats_per_sigma,
        save_path=os.path.join(outputs_dir, f"subject_{subject_idx}_out_degree_hist.png"),
        title=f"OC-LatentFlow LATENT out-degree  subj {subject_idx}  ({condition_mode})",
    )

    # ----- save JSON ---------------------------------------------------
    out_json = os.path.join(output_dir, f"probe_subject_{subject_idx}.json")
    payload = {
        "subject_idx": int(subject_idx),
        "subject_file": os.path.basename(subject_file),
        "n_trials": int(n_trials),
        "n_sources": int(n_sources),
        "checkpoint": checkpoint_name,
        "model": "OC-LatentFlow (stage-2 flow path) — LATENT-SPACE probe",
        "probe_site": "z_obs[:, a] -> z_fc_hat[:, b] (flow input/output latent; "
                      "encoder run once, decoder bypassed)",
        "condition_mode": condition_mode,
        "sigma_unit": "absolute" if absolute_sigma else "relative (per-channel std)",
        "sigmas": [float(s) for s in sigmas],
        "n_steps": N_STEPS,
        "jet_cv_range": [JET_CV_RANGE[0], JET_CV_RANGE[1]],
        "forecast_sanity": {
            "mean": float(z_fc_base.mean()),
            "std": float(z_fc_base.std()),
            "abs_max": float(z_fc_base.abs().max()),
        },
        "stats_per_sigma": {
            str(sigma): {k: (v if not isinstance(v, list) else v)
                         for k, v in stats_per_sigma[sigma].items()}
            for sigma in sigmas
        },
    }
    with open(out_json, "w") as fh:
        json.dump(payload, fh, indent=2, default=json_default)
    print(f"saved JSON: {out_json}")
    return payload


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")
    print(f"Condition mode: {args.condition_mode} | "
          f"sigma unit: {'absolute' if args.absolute_sigma else 'relative'}")

    sigmas = [float(s) for s in args.sigmas.split(",") if s.strip()]
    print(f"Sigmas: {sigmas}")

    ckpt_path = (
        args.checkpoint
        or str(CHECKPOINT_DIR / f"fold{args.fold}_oc_latent_flow_flow_best.pt")
    )
    model = load_model(ckpt_path, device)
    checkpoint_name = os.path.basename(ckpt_path)

    test_files = get_fold_test_subjects(args.fold, args.seed)
    subjects_to_run = test_files[args.subject_idx:args.subject_idx + args.subjects]

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.outputs_dir, exist_ok=True)

    all_payloads = []
    for i, subject_file in enumerate(subjects_to_run):
        subject_idx = args.subject_idx + i
        x_obs = load_subject_trials(subject_file, device, normalize=True)
        payload = run_subject(
            model=model, x_obs_subject=x_obs, subject_idx=subject_idx,
            subject_file=subject_file, sigmas=sigmas, n_sources=args.sources,
            absolute_sigma=args.absolute_sigma, condition_mode=args.condition_mode,
            output_dir=args.output_dir, outputs_dir=args.outputs_dir,
            checkpoint_name=checkpoint_name, device=device,
        )
        all_payloads.append(payload)

    print("\n=== aggregate CV across subjects ===")
    for sigma in sigmas:
        cvs = [p["stats_per_sigma"][str(sigma)]["cv"]
               for p in all_payloads
               if str(sigma) in p["stats_per_sigma"]]
        cvs = [c for c in cvs if not math.isnan(c)]
        if cvs:
            mean_cv = sum(cvs) / len(cvs)
            print(f"sigma={sigma}: mean CV = {mean_cv * 100:.2f}%  "
                  f"[JET v2 range {JET_CV_RANGE[0] * 100:.1f}%-"
                  f"{JET_CV_RANGE[1] * 100:.1f}%]")

    print("\nDone.")


if __name__ == "__main__":
    main()
