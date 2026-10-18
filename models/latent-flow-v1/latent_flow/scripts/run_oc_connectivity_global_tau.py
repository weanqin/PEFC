"""Global-quantile-tau connectivity runner for OC-LatentFlow (sigma* preserved).

Task 07-04-global-tau-sigma. Preserves the discard-entropy semantics —
sigma*[a,b] = b's tolerance to a's perturbation = a->b connection strength —
while fixing the threshold design that broke the original bisection.

Root cause of the original failure (task 07-04-de-vs-baseline-zero-corr):
``tau = median_b(MSE_b^a(sigma_max))`` was a SHARED PER-SOURCE threshold, so
sigma* encoded each source's global destructiveness (78% row variance), not
the per-edge a->b tolerance.

Fix: tau = a single quantile of the GLOBAL MSE response distribution over
all (a,b) pairs. Every source/target shares the SAME threshold, so sigma*
values are directly comparable across the matrix.

Two-stage algorithm (tau depends on the global distribution):
  Stage 1: one forward per source at sigma_ref -> (S,C) raw response matrix
           -> tau = compute_global_tau(response_matrix, q).
  Stage 2: per source, sweep a log-spaced sigma grid -> per-target MSE curves
           -> sigma_star_with_global_tau -> connectivity = -(log sigma* + ...).

See ``attribution/bisect_sigma.py`` for the algorithm primitives. The original
bisection path (compute_tau_from_median + walk_monotone_curve) and the
median-polish control experiment are untouched.

Output (relative to repo root):
    results/oc_connectivity_global_tau_q{q}/subject_{idx}.json

Usage (run from repo root D:/102-npi/ude-vae-npi, conda npi GPU):

    # smoke: 16 sources x 1 subject
    python -u -m latent_flow.scripts.run_oc_connectivity_global_tau \\
        --subjects 1 --sources 16 --q 0.5 --device cuda

    # full: 128 sources x 10 subjects
    python -u -m latent_flow.scripts.run_oc_connectivity_global_tau \\
        --subjects 10 --sources 128 --q 0.5 --device cuda
"""

import argparse
import json
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

from latent_flow.attribution.bisect_sigma import (
    SIGMA_MAX,
    SIGMA_MIN,
    LatentProbeSite,
    SignalProbeSite,
    _HALF_LOG_2PIE,
    _extract_mse,
    compute_global_tau,
    log_grid,
    sigma_star_with_global_tau,
)
from latent_flow.config import CHECKPOINT_DIR
from latent_flow.data.datasets import get_dataset
from latent_flow.scripts._connectivity_driver import (
    N_CHANNELS,
    get_fold_test_subjects,
    json_default,
    load_subject_trials,
    sanitize_for_json,
    subsample_trials,
)
from latent_flow.scripts.run_oc_connectivity_relative import (
    _load_model_compat,
)

# Channel count is dataset-derived. This module-level name shadows the driver's
# constant and is re-assigned in main() based on --dataset so the array
# allocations and --sources default match the cohort (MODMA 128, EDRA 62).
N_CHANNELS = get_dataset("modma").channels

DEFAULT_OUTPUT_DIR = "results/oc_connectivity_global_tau_signed"
DEFAULT_CHECKPOINT = "fold0_oc_latent_flow_flow_best_graph_mexhat_screg_l0.01_w0.1_nosc.pt"
DEFAULT_SIGMA_REF = 0.3

# EDRA checkpoints are namespaced under checkpoints/edra/. The fold0 mexhat graph
# model is the EDRA connectivity reference (see domain.md EDRA SOTA config).
EDRA_CHECKPOINT_DIR = CHECKPOINT_DIR / "edra"
EDRA_DEFAULT_CHECKPOINT = "fold0_oc_latent_flow_flow_best_graph_mexhat.pt"
DEFAULT_Q = 0.5
DEFAULT_N_GRID = 24


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Global-quantile-tau connectivity runner for OC-LatentFlow "
            "(task 07-04-global-tau-sigma). Preserves sigma* tolerance "
            "semantics; fixes the threshold to a single global quantile."
        ),
    )
    p.add_argument("--fold", type=int, default=0, choices=range(5))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dataset", choices=["modma", "edra"], default="modma",
                   help="cohort (MODMA 128ch / EDRA 62ch). Selects checkpoint dir, "
                        "geometry, subject set, and channel count.")
    p.add_argument("--subjects", type=int, default=10)
    p.add_argument("--subject-idx", type=int, default=0)
    p.add_argument("--sources", type=int, default=0,
                   help="source channels to sweep (default = dataset channel count; "
                        "set lower for smoke).")
    p.add_argument("--n-trials", type=int, default=0,
                   help="trials to subsample per subject (0 = all).")
    p.add_argument("--sigma-ref", type=float, default=DEFAULT_SIGMA_REF,
                   help=f"sigma for stage-1 response scan (default {DEFAULT_SIGMA_REF}).")
    p.add_argument("--q", type=float, default=DEFAULT_Q,
                   help=f"quantile for global tau (default {DEFAULT_Q}; 0.5=median).")
    p.add_argument("--checkpoint", type=str, default="")
    p.add_argument("--output-dir", type=str, default="",
                   help=f"default {DEFAULT_OUTPUT_DIR}_q{{q}}")
    p.add_argument("--sigma-min", type=float, default=SIGMA_MIN)
    p.add_argument("--sigma-max", type=float, default=SIGMA_MAX)
    p.add_argument("--n-grid", type=int, default=DEFAULT_N_GRID,
                   help=f"sigma grid points for stage-2 search (default {DEFAULT_N_GRID}).")
    p.add_argument("--n-steps", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", type=str, default="")
    p.add_argument("--latent", action="store_true",
                   help="use LatentProbeSite (perturb z_obs, measure z_fc_hat; "
                        "bypasses decoder) instead of the default signal-space site.")
    return p.parse_args()


def _stage1_response_scan(site, sources, sigma_ref, device, amp_dtype):
    """Stage 1: one forward per source at sigma_ref -> (S, C) raw response + mean_diff."""
    response = np.full((sources, N_CHANNELS), float("nan"), dtype=np.float32)
    mean_diff = np.full((sources, N_CHANNELS), float("nan"), dtype=np.float32)
    for src in range(sources):
        site.set_source(src)
        # Reuse perturb_and_measure at the single sigma_ref amplitude.
        enabled = device.type == "cuda"
        with torch.amp.autocast(
            device_type=("cuda" if device.type == "cuda" else "cpu"),
            dtype=amp_dtype, enabled=enabled,
        ):
            result = site.perturb_and_measure(float(sigma_ref), src)
            if isinstance(result, tuple):
                # SignalProbeSite returns (mse, mean_diff) tuple
                mse_vec, md_vec = result
            else:
                # LatentProbeSite returns single tensor; fill mean_diff with NaN
                mse_vec = result
                md_vec = torch.full_like(mse_vec, float("nan"))
            mse_vec = mse_vec.detach().cpu()
            md_vec = md_vec.detach().cpu()
        response[src, :] = mse_vec.numpy()
        mean_diff[src, :] = md_vec.numpy()
    return torch.from_numpy(response), torch.from_numpy(mean_diff)


def _stage2_sigma_search(site, sources, grid, tau_global,
                         sigma_min, sigma_max, device, amp_dtype):
    """Stage 2: per source, sweep sigma grid -> sigma* under global tau.

    Mirrors the old ``attribution_engine`` pattern: ONE autocast context wraps
    the whole grid loop per source (re-entering autocast per sigma is slow and
    the GPU->CPU transfer per point adds up). mse_curve stays on GPU until the
    monotone walk, matching the engine's throughput.
    """
    sigma_matrix = np.full((sources, N_CHANNELS), float("nan"), dtype=np.float32)
    enabled = device.type == "cuda"
    for src in range(sources):
        site.set_source(src)
        with torch.amp.autocast(
            device_type=("cuda" if device.type == "cuda" else "cpu"),
            dtype=amp_dtype, enabled=enabled,
        ):
            mse_curve = torch.stack(
                [_extract_mse(site.perturb_and_measure(float(s), src))
                 for s in grid], dim=0
            )  # (n_grid, C) on GPU
        sigma_star = sigma_star_with_global_tau(
            mse_curve, grid, float(tau_global), sigma_min, sigma_max)
        sigma_matrix[src, :] = sigma_star.numpy()
    return sigma_matrix


def main():
    args = parse_args()
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    # Dataset-driven setup: channel count, checkpoint dir, geometry, subject set.
    global N_CHANNELS
    N_CHANNELS = get_dataset(args.dataset).channels
    if args.sources <= 0:
        args.sources = N_CHANNELS
    print(f"Dataset: {args.dataset} | channels: {N_CHANNELS} | sources: {args.sources}")

    if args.checkpoint:
        ckpt_path = args.checkpoint
    elif args.dataset == "edra":
        ckpt_path = str(EDRA_CHECKPOINT_DIR / EDRA_DEFAULT_CHECKPOINT)
    else:
        ckpt_path = str(CHECKPOINT_DIR / DEFAULT_CHECKPOINT)
    model = _load_model_compat(ckpt_path, device)
    checkpoint_name = os.path.basename(ckpt_path)

    test_files = get_fold_test_subjects(args.fold, args.seed, dataset=args.dataset)
    subjects_to_run = test_files[args.subject_idx:args.subject_idx + args.subjects]

    space_tag = "latent" if args.latent else "signal"
    out_dir = args.output_dir or f"{DEFAULT_OUTPUT_DIR}_{space_tag}_q{args.q}"
    os.makedirs(out_dir, exist_ok=True)

    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    grid = log_grid(args.sigma_min, args.sigma_max, args.n_grid)
    print(f"Paradigm: global-quantile-tau (sigma* preserved) | "
          f"q={args.q} sigma_ref={args.sigma_ref} "
          f"grid=log[{args.sigma_min},{args.sigma_max}] n={args.n_grid}")
    print(f"Subjects: {len(subjects_to_run)} | sources: {args.sources} | "
          f"out: {out_dir}")

    t0_global = time.time()
    for i, subject_file in enumerate(subjects_to_run):
        subject_idx = args.subject_idx + i
        print(f"\n=== subject {subject_idx} ({os.path.basename(subject_file)}) ===")

        x_obs_full = load_subject_trials(subject_file, device, normalize=True,
                                         dataset=args.dataset)
        x_obs, n_used = subsample_trials(
            x_obs_full, args.n_trials, args.seed, subject_idx)
        if n_used != x_obs_full.shape[0]:
            print(f"  subsampled {n_used}/{x_obs_full.shape[0]} trials")

        if args.latent:
            site = LatentProbeSite(
                model=model, x_obs_subject=x_obs,
                n_steps=args.n_steps, seed=args.seed,
                absolute_sigma=False, condition_mode="coupled",
                n_grid=args.n_grid,
                sigma_min=args.sigma_min, sigma_max=args.sigma_max,
                batch_size=args.batch_size,
            )
            site_kind = "latent"
        else:
            site = SignalProbeSite(
                model=model, x_obs_subject=x_obs,
                n_steps=args.n_steps, use_proj=False,
                batch_size=args.batch_size, seed=args.seed,
                n_grid=args.n_grid,
                sigma_min=args.sigma_min, sigma_max=args.sigma_max,
                use_biased_noise=True,
            )
            site_kind = "signal"

        subj_t0 = time.time()
        n_failed = 0
        try:
            # Stage 1: global tau from the response distribution.
            print(f"  stage 1: response scan ({args.sources} sources @ sigma_ref={args.sigma_ref})")
            response_matrix, mean_diff_matrix_t = _stage1_response_scan(
                site, args.sources, args.sigma_ref, device, amp_dtype)
            mean_diff_matrix = mean_diff_matrix_t.numpy()
            tau_global = compute_global_tau(response_matrix, args.q)
            print(f"  stage 1: global tau (q={args.q}) = {tau_global:.6f}")

            # Stage 2: sigma* search under the global tau.
            print(f"  stage 2: sigma* search ({args.sources} sources x {args.n_grid} grid)")
            sigma_matrix = _stage2_sigma_search(
                site, args.sources, grid, tau_global,
                args.sigma_min, args.sigma_max, device, amp_dtype)
        except RuntimeError as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            n_failed = args.sources
            sigma_matrix = np.full(
                (args.sources, N_CHANNELS), float("nan"), dtype=np.float32)
            response_matrix = torch.zeros(args.sources, N_CHANNELS)
            mean_diff_matrix = np.full(
                (args.sources, N_CHANNELS), float("nan"), dtype=np.float32)
            tau_global = float("nan")

        # Connectivity = differential entropy form (preserved semantics).
        sigma_t = torch.from_numpy(sigma_matrix.copy())
        connectivity = -(sigma_t.float().log() + _HALF_LOG_2PIE).numpy()

        # Directional sign from the mean shift of FC forecast output
        # (task 07-23-de-directional-noise-sign): +1 = excitatory,
        # -1 = inhibitory, 0 = no significant direction.
        sign_matrix = np.sign(mean_diff_matrix)
        signed_connectivity = sign_matrix * connectivity

        elapsed_min = (time.time() - subj_t0) / 60.0
        n_succeeded = args.sources - n_failed
        # Diagnostics: boundary clamp proportions.
        n_at_min = int(np.sum(np.isfinite(sigma_matrix[:, :]) & (
            sigma_matrix <= args.sigma_min + 1e-6)))
        n_at_max = int(np.sum(np.isfinite(sigma_matrix[:, :]) & (
            sigma_matrix >= args.sigma_max - 1e-6)))
        n_total = sigma_matrix.size
        payload = {
            "subject_idx": int(subject_idx),
            "subject_file": os.path.basename(subject_file),
            "n_trials": int(n_used),
            "n_sources_requested": int(args.sources),
            "n_sources_succeeded": int(n_succeeded),
            "method": f"global_tau_quantile_signed ({site_kind}-space, sigma* preserved, directional sign)",
            "tau_global": float(tau_global),
            "q": float(args.q),
            "sigma_ref": float(args.sigma_ref),
            "sigma_matrix": sigma_matrix.tolist(),
            "connectivity_matrix": connectivity.tolist(),
            "connectivity_formula": "-(log sigma* + 0.5*log(2*pi*e))",
            "raw_response_matrix": response_matrix.numpy().tolist(),
            "sign_matrix": sign_matrix.tolist(),
            "signed_connectivity_matrix": signed_connectivity.tolist(),
            "mean_diff_matrix": mean_diff_matrix.tolist(),
            "mu_bias_per_channel": (
                site._mu_bias.detach().cpu().tolist()
                if hasattr(site, "_mu_bias") else None
            ),
            "noise_model": "biased_gaussian_mu05std",
            "grid": "log_spaced",
            "hparams": {
                "q": float(args.q),
                "sigma_ref": float(args.sigma_ref),
                "sigma_min": float(args.sigma_min),
                "sigma_max": float(args.sigma_max),
                "n_grid": int(args.n_grid),
                "n_steps": int(args.n_steps),
                "batch_size": int(args.batch_size),
                "seed": int(args.seed),
            },
            "checkpoint": checkpoint_name,
            "diagnostics": {
                "n_at_sigma_min": n_at_min,
                "n_at_sigma_max": n_at_max,
                "pct_at_sigma_min": n_at_min / n_total * 100 if n_total else 0.0,
                "pct_at_sigma_max": n_at_max / n_total * 100 if n_total else 0.0,
            },
            "n_source_failures": int(n_failed),
            "elapsed_min": float(elapsed_min),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        payload = sanitize_for_json(payload)
        out_json = os.path.join(out_dir, f"subject_{subject_idx}.json")
        with open(out_json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=json_default, allow_nan=False)
        print(f"  subject {subject_idx} done in {elapsed_min:.2f} min | "
              f"tau={tau_global:.6f} "
              f"@min={n_at_min/n_total*100:.1f}% @max={n_at_max/n_total*100:.1f}% | "
              f"wrote {out_json}")

    print(f"\n=== summary ===")
    print(f"subjects done: {len(subjects_to_run)}")
    print(f"elapsed: {(time.time() - t0_global) / 3600:.2f}h")
    print("Done.")


if __name__ == "__main__":
    main()
