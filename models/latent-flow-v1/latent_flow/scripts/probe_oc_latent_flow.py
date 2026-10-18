"""Fixed-sigma perturbation-response probe for OC-LatentFlow (task 06-25).

Mirrors ``scripts/probe_perturbation_response.py`` (the JET v2 probe) but
targets OC-LatentFlow's stage-2 flow path. The goal is to answer:

    Does the OC-LatentFlow per-channel temporal latent differentiate
    responses when we perturb a single source channel a, across the 128
    target channels b?

JET v2's probe measured channel-sensitivity CV (std/mean) of only 9.5%
(sigma=0.1) -> 17.5% (sigma=0.3), rooted in its channel collapse
(per-channel cosine 0.987, 06-19 diagnosis). OC-LatentFlow's z_obs PC1=4.1%
/ effective dim 84.4 SHOULD yield a much higher CV — this probe verifies
whether the per-channel latent actually translates into channel-specific
forecast sensitivity, which is the prerequisite for the bisection
sigma-matrix attribution to surface real connectivity structure.

Key simplification vs the JET probe: OC-LatentFlow's stage-2 flow path is
**deterministic** (z_obs = mu_obs, no reparameterize, flow Euler from a
fixed start). There is no flow-matching starting noise to pin down, so the
JET ``x0_init`` / ``method`` / fixed-x0 machinery is unnecessary —
perturbed-vs-baseline deltas are clean by construction.

Algorithm (per subject):

    1. Load checkpoints/fold{fold}_oc_latent_flow_flow_best.pt (stage-2).
    2. Baseline: x_fc_base = model.sample(x_obs, n_steps=10, use_proj=False).
    3. For each sigma in {0.1, 0.2, 0.3}:
         For each source channel a in [0, 127):
             xi ~ N(0, 1) shape (N, T_obs), seeded per source
             x_obs_pert[:, :, a] += sigma * xi
             x_fc_pert = model.sample(x_obs_pert, n_steps=10, use_proj=False)
             response[a, b] = MSE(x_fc_pert[:, :, b], x_fc_base[:, :, b])
         CV(sigma) = std(response) / mean(response)   # over all (a, b)
    4. Compare CV vs JET (9.5%-17.5%). CV >> JET => ceiling broken.

Output (relative to ):
    results/probe_oc/probe_subject_{idx}.json
    outputs/probe_oc/subject_{idx}_sigma_{sigma}_heatmap.png
    outputs/probe_oc/subject_{idx}_comparison.png
    outputs/probe_oc/subject_{idx}_out_degree_hist.png

Runtime estimate: ~1-2 s per source forward x 128 source x 3 sigma
~ 10-15 min per subject on RTX 5070 Ti.

Usage (run from repo root D:/102-npi/ude-vae-npi):

    D:/program/anaconda/envs/npi/python.exe -u -m \\
        latent_flow.scripts.probe_oc_latent_flow --fold 0
"""

import argparse
import json
import math
import os
import sys
import time

# KMP_DUPLICATE_LIB_OK avoids a Windows DLL conflict when torch + scipy
# both try to initialize the Intel OpenMP runtime.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# Force UTF-8 stdout so ASCII progress prints survive a GBK pipe
# (logging-guidelines.md).
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import numpy as np
import torch

from latent_flow.config import CHECKPOINT_DIR
from latent_flow.scripts._connectivity_driver import (
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

DEFAULT_OUTPUT_DIR = "results/probe_oc"
DEFAULT_OUTPUTS_DIR = "outputs/probe_oc"

DEFAULT_SIGMAS = "0.1,0.2,0.3"

# JET v2 baseline CV (from 06-16 probe / domain.md), for the comparison print.
JET_CV_RANGE = (0.095, 0.175)

N_STEPS = 10            # OC flow Euler steps (quality-guidelines)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Fixed-sigma perturbation-response probe for OC-LatentFlow "
            "(stage-2 flow path). Measures per-target MSE response to a "
            "fixed perturbation on each source channel; reports the "
            "channel-sensitivity CV to compare against the JET v2 ceiling."
        ),
    )
    p.add_argument(
        "--subjects", type=int, default=1,
        help="number of fold test subjects to process (default 1).",
    )
    p.add_argument(
        "--sources", type=int, default=N_CHANNELS,
        help=f"number of source channels to perturb (default {N_CHANNELS}).",
    )
    p.add_argument(
        "--checkpoint", type=str, default="",
        help=(
            "path to OC-LatentFlow flow checkpoint. Empty (default) uses "
            "checkpoints/fold{fold}_oc_latent_flow_flow_best.pt."
        ),
    )
    p.add_argument(
        "--sigmas", type=str, default=DEFAULT_SIGMAS,
        help=f"comma-separated fixed sigma values (default '{DEFAULT_SIGMAS}').",
    )
    p.add_argument(
        "--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help="directory for per-subject JSON results.",
    )
    p.add_argument(
        "--outputs-dir", type=str, default=DEFAULT_OUTPUTS_DIR,
        help="directory for per-subject PNG visualizations.",
    )
    p.add_argument(
        "--subject-idx", type=int, default=0,
        help="0-based subject index in the fold test to start from (default 0).",
    )
    p.add_argument(
        "--fold", type=int, default=0, choices=range(5),
        help="fold index for the SubjectFoldSplitter (default 0).",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="random seed for the subject splitter (default 42).",
    )
    p.add_argument(
        "--device", type=str, default="",
        help="torch device override, e.g. 'cuda:0' or 'cpu'. Empty uses cuda.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Probe core
# ---------------------------------------------------------------------------

@torch.no_grad()
def forecast_fc(model, x_obs, n_steps=N_STEPS, use_proj=False):
    """Forward + fc slice. OC sample returns the full obs+fc window."""
    amp_dtype = torch.bfloat16 if x_obs.device.type == "cuda" else torch.float32
    with torch.amp.autocast(
        device_type=("cuda" if x_obs.device.type == "cuda" else "cpu"),
        dtype=amp_dtype,
        enabled=(x_obs.device.type == "cuda"),
    ):
        out = model.sample(x_obs, n_steps=n_steps, use_proj=use_proj)
    # sample returns the full window (B, obs_len + fc_len, C); take the fc tail.
    if out.shape[1] > OBS_LEN:
        out = out[:, OBS_LEN:, :]
    return torch.nan_to_num(out.float(), nan=0.0, posinf=0.0, neginf=0.0)


@torch.no_grad()
def compute_baseline_forecast(model, x_obs):
    """Baseline forecast (no perturbation). Deterministic for OC stage-2."""
    return forecast_fc(model, x_obs)


@torch.no_grad()
def probe_single_sigma(
    model,
    x_obs,
    x_fc_base,
    sigma,
    n_sources,
    device,
    seed=42,
    progress_prefix: str = "",
):
    """Perturb every source channel with the SAME amplitude ``sigma``.

    For each source channel a in [0, n_sources):
        xi ~ N(0, 1) shape (N, T_obs), seeded per source, scaled by sigma
        x_obs_pert[:, :, a] += sigma * xi
        x_fc_pert = forecast_fc(model, x_obs_pert)
        response[a, b] = MSE(x_fc_pert[:, :, b], x_fc_base[:, :, b])

    Args:
        model: OCLatentFlow on device, eval mode.
        x_obs: (N, T_obs, C) subject trials on device.
        x_fc_base: (N, T_fc, C) baseline forecast on device.
        sigma: scalar perturbation amplitude.
        n_sources: number of source channels to perturb.
        device: torch device.
        seed: base seed; per-source noise uses seed*1000 + src so sources
            are statistically independent yet reproducible.
        progress_prefix: prepended to per-source progress lines.

    Returns:
        (n_sources, C) float32 numpy array of per-target MSE response.
    """
    N, T_obs, C = x_obs.shape
    response = np.zeros((n_sources, C), dtype=np.float32)

    for src in range(n_sources):
        src_t0 = time.time()
        # Fresh generator per source so noise is independent across sources.
        noise_gen = torch.Generator(device=device).manual_seed(seed * 1000 + src + 1)
        xi = torch.randn(N, T_obs, device=device, generator=noise_gen)

        x_obs_pert = x_obs.clone()
        x_obs_pert[:, :, src] = x_obs_pert[:, :, src] + float(sigma) * xi

        x_fc_pert = forecast_fc(model, x_obs_pert)

        sq = (x_fc_pert.float() - x_fc_base.float()) ** 2
        response[src, :] = sq.mean(dim=(0, 1)).cpu().numpy()

        if src % 16 == 0 or src == n_sources - 1:
            dt = time.time() - src_t0
            print(
                f"{progress_prefix}src={src:3d}/{n_sources} "
                f"mean_resp={float(response[src].mean()):.4e} "
                f"max_resp={float(response[src].max()):.4e} "
                f"({dt:.2f}s)"
            )

    return response


# ---------------------------------------------------------------------------
# Per-subject driver
# ---------------------------------------------------------------------------

def run_subject(model, x_obs_subject, subject_idx, subject_file, sigmas,
                n_sources, output_dir, outputs_dir, checkpoint_name, device):
    """Probe one subject at all sigmas; write JSON + PNGs; return payload."""
    n_trials = x_obs_subject.shape[0]
    print(
        f"\n=== subject {subject_idx} "
        f"({os.path.basename(subject_file)}) === "
        f"{n_trials} trials, probing {n_sources} sources x {len(sigmas)} sigmas"
    )

    base_t0 = time.time()
    x_fc_base = compute_baseline_forecast(model, x_obs_subject)
    base_dt = time.time() - base_t0
    print(
        f"baseline forward: {base_dt:.2f}s, "
        f"x_fc_base mean={float(x_fc_base.mean()):.4e} "
        f"std={float(x_fc_base.std()):.4e} "
        f"abs_max={float(x_fc_base.abs().max()):.4e}"
    )

    response_matrices = {}      # sigma -> (n_src, C) np array
    stats_per_sigma = {}        # sigma -> stats dict

    for sigma in sigmas:
        sigma_t0 = time.time()
        print(f"\n--- sigma={sigma} --- perturbing {n_sources} source channels")
        response = probe_single_sigma(
            model=model,
            x_obs=x_obs_subject,
            x_fc_base=x_fc_base,
            sigma=sigma,
            n_sources=n_sources,
            device=device,
            progress_prefix=f"subject {subject_idx} / sigma={sigma} / ",
        )
        response_matrices[sigma] = response
        stats = compute_stats(response)
        stats_per_sigma[sigma] = stats
        sigma_dt = time.time() - sigma_t0
        print(
            f"sigma={sigma} done in {sigma_dt:.2f}s ({sigma_dt / 60.0:.2f} min) | "
            f"mean={stats['mean']:.4e} std={stats['std']:.4e} "
            f"CV={stats['cv'] * 100:.2f}%"
        )

    # ----- CV comparison vs JET ceiling --------------------------------
    print("\n--- channel-sensitivity CV vs JET v2 ceiling ---")
    for sigma in sigmas:
        s = stats_per_sigma[sigma]
        verdict = (
            "ABOVE ceiling (per-channel latent helps)"
            if s["cv"] > JET_CV_RANGE[1]
            else "AT/BELOW ceiling (still limited)"
        )
        print(
            f"sigma={sigma}: CV={s['cv'] * 100:.2f}%  "
            f"[JET v2 range {JET_CV_RANGE[0] * 100:.1f}%-"
            f"{JET_CV_RANGE[1] * 100:.1f}%] -> {verdict}"
        )

    # ----- save PNGs ---------------------------------------------------
    for sigma in sigmas:
        plot_response_heatmap(
            response_matrix=response_matrices[sigma],
            sigma_value=sigma,
            save_path=os.path.join(
                outputs_dir,
                f"subject_{subject_idx}_sigma_{sigma}_heatmap.png",
            ),
            title=(
                f"OC-LatentFlow (flow) response  subj {subject_idx}  "
                f"sigma={sigma}"
            ),
        )
    plot_comparison(
        response_matrices=response_matrices,
        save_path=os.path.join(outputs_dir, f"subject_{subject_idx}_comparison.png"),
        title=f"OC-LatentFlow (flow) response comparison  subj {subject_idx}",
    )
    plot_out_degree_histogram(
        stats_dict=stats_per_sigma,
        save_path=os.path.join(outputs_dir, f"subject_{subject_idx}_out_degree_hist.png"),
        title=f"OC-LatentFlow (flow) out-degree  subj {subject_idx}",
    )

    # ----- save JSON ---------------------------------------------------
    out_json = os.path.join(output_dir, f"probe_subject_{subject_idx}.json")
    payload = {
        "subject_idx": int(subject_idx),
        "subject_file": os.path.basename(subject_file),
        "n_trials": int(n_trials),
        "n_sources": int(n_sources),
        "checkpoint": checkpoint_name,
        "model": "OC-LatentFlow (stage-2 flow path)",
        "sigmas": [float(s) for s in sigmas],
        "n_steps": N_STEPS,
        "use_proj": False,
        "jet_cv_range": [JET_CV_RANGE[0], JET_CV_RANGE[1]],
        "forecast_sanity": {
            "mean": float(x_fc_base.mean()),
            "std": float(x_fc_base.std()),
            "abs_max": float(x_fc_base.abs().max()),
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
            model=model,
            x_obs_subject=x_obs,
            subject_idx=subject_idx,
            subject_file=subject_file,
            sigmas=sigmas,
            n_sources=args.sources,
            output_dir=args.output_dir,
            outputs_dir=args.outputs_dir,
            checkpoint_name=checkpoint_name,
            device=device,
        )
        all_payloads.append(payload)

    # Aggregate CV across subjects (final headline number).
    print("\n=== aggregate CV across subjects ===")
    for sigma in sigmas:
        cvs = [
            p["stats_per_sigma"][str(sigma)]["cv"]
            for p in all_payloads
            if str(sigma) in p["stats_per_sigma"]
        ]
        cvs = [c for c in cvs if not math.isnan(c)]
        if cvs:
            mean_cv = sum(cvs) / len(cvs)
            print(
                f"sigma={sigma}: mean CV = {mean_cv * 100:.2f}%  "
                f"[JET v2 range {JET_CV_RANGE[0] * 100:.1f}%-"
                f"{JET_CV_RANGE[1] * 100:.1f}%]"
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
