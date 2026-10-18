"""Per-edge relative-response connectivity runner for OC-LatentFlow.

Task 07-04-redesign-bisection-paradigm. Implements the redesigned
perturbation-inference paradigm: instead of the original bisection
(per-source shared tau threshold -> sigma* -> -log sigma*) which produced a
source-main-effect-dominated, subject-invariant matrix (task 07-04-de-vs-
baseline-zero-corr root cause), this runner collects ONE forward per source
at a fixed sigma_ref, then applies median polish (two-way ANOVA residual)
to eliminate both the source and target main effects, leaving the a->b
residual signal.

See ``attribution/bisect_sigma.py``::

    compute_relative_response(site, source, sigma_ref)  -> (C,) raw MSE_b^a
    build_connectivity_relative(response_matrix)        -> (S, C) residual

Output (relative to repo root):
    results/oc_connectivity_relative/subject_{idx}.json

The JSON mirrors the connectivity-runner schema (subject_idx, subject_file,
connectivity_matrix, checkpoint, method, ...) so the existing
``scripts/diag_de_vs_baseline.py`` consumes it unchanged via ``--oc-dir``.

Usage (run from repo root D:/102-npi/ude-vae-npi):

    # smoke: 8 sources x 1 subject
    python -u -m latent_flow.scripts.run_oc_connectivity_relative \\
        --subjects 1 --sources 8

    # full: 128 sources x 10 subjects
    python -u -m latent_flow.scripts.run_oc_connectivity_relative \\
        --subjects 10 --sources 128 --checkpoint \\
        checkpoints/fold0_oc_latent_flow_flow_best_graph_mexhat.pt
"""

import argparse
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
    SignalProbeSite,
    build_connectivity_relative,
    compute_relative_response,
)
from latent_flow.config import CHECKPOINT_DIR
from latent_flow.scripts._connectivity_driver import (
    N_CHANNELS,
    get_fold_test_subjects,
    json_default,
    load_model,
    load_subject_trials,
    sanitize_for_json,
    subsample_trials,
)

import json

DEFAULT_OUTPUT_DIR = "results/oc_connectivity_relative"
DEFAULT_CHECKPOINT = "fold0_oc_latent_flow_flow_best_graph_mexhat.pt"
DEFAULT_SIGMA_REF = 0.3


def _load_model_compat(checkpoint_path, device):
    """Load an OC-LatentFlow checkpoint.

    Historically a wrapper that tolerated the pre-SC mexhat variant. SC-prior
    support has since been removed from the model and ``_connectivity_driver.
    load_model`` now tolerates legacy ``sc_base``/``gamma_sc`` checkpoint keys
    internally, so this is a thin pass-through kept for call-site stability.
    """
    return load_model(checkpoint_path, device)


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Per-edge relative-response connectivity runner for OC-LatentFlow "
            "(task 07-04-redesign). One forward per source at fixed sigma_ref, "
            "then median polish to remove source/target main effects."
        ),
    )
    p.add_argument("--fold", type=int, default=0, choices=range(5))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--subjects", type=int, default=10)
    p.add_argument("--subject-idx", type=int, default=0)
    p.add_argument("--sources", type=int, default=N_CHANNELS,
                   help=f"source channels to sweep (default {N_CHANNELS}; 8 for smoke).")
    p.add_argument("--n-trials", type=int, default=0,
                   help="trials to subsample per subject (0 = all).")
    p.add_argument("--sigma-ref", type=float, default=DEFAULT_SIGMA_REF,
                   help=f"fixed perturbation amplitude (default {DEFAULT_SIGMA_REF}).")
    p.add_argument("--checkpoint", type=str, default="",
                   help="OC-LatentFlow checkpoint. Empty uses DEFAULT_CHECKPOINT.")
    p.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--n-steps", type=int, default=10,
                   help="OC flow Euler steps (default 10).")
    p.add_argument("--batch-size", type=int, default=32,
                   help="trial batch size per forward (default 32).")
    p.add_argument("--no-col-norm", action="store_true",
                   help="ablate: skip target (column) main-effect removal.")
    p.add_argument("--no-row-norm", action="store_true",
                   help="ablate: skip source (row) main-effect removal.")
    p.add_argument("--device", type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    ckpt_path = (
        args.checkpoint
        or str(CHECKPOINT_DIR / DEFAULT_CHECKPOINT)
    )
    model = _load_model_compat(ckpt_path, device)
    checkpoint_name = os.path.basename(ckpt_path)

    test_files = get_fold_test_subjects(args.fold, args.seed)
    subjects_to_run = test_files[args.subject_idx:args.subject_idx + args.subjects]

    os.makedirs(args.output_dir, exist_ok=True)

    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    row_norm = not args.no_row_norm
    col_norm = not args.no_col_norm
    print(f"Paradigm: relative-response (median polish) | "
          f"sigma_ref={args.sigma_ref} row_norm={row_norm} col_norm={col_norm}")
    print(f"Subjects: {len(subjects_to_run)} | sources: {args.sources} | "
          f"n_trials: {args.n_trials} (0=all)")

    t0_global = time.time()
    for i, subject_file in enumerate(subjects_to_run):
        subject_idx = args.subject_idx + i
        print(f"\n=== subject {subject_idx} "
              f"({os.path.basename(subject_file)}) ===")

        x_obs_full = load_subject_trials(subject_file, device, normalize=True)
        x_obs, n_used = subsample_trials(
            x_obs_full, args.n_trials, args.seed, subject_idx)
        if n_used != x_obs_full.shape[0]:
            print(f"  subsampled {n_used}/{x_obs_full.shape[0]} trials")

        site = SignalProbeSite(
            model=model, x_obs_subject=x_obs,
            n_steps=args.n_steps, use_proj=False,
            batch_size=args.batch_size, seed=args.seed,
            sigma_min=args.sigma_ref, sigma_max=args.sigma_ref,
        )

        # Collect ONE forward per source -> (S, C) raw response matrix.
        # OOM-safe per source (skip + record NaN on failure, like SubjectRunner).
        response_matrix = np.full(
            (args.sources, N_CHANNELS), float("nan"), dtype=np.float32)
        n_failed = 0
        subj_t0 = time.time()
        for src in range(args.sources):
            try:
                site.set_source(src)
                vec = compute_relative_response(
                    site=site, source=src, sigma_ref=args.sigma_ref,
                    device=device, amp_dtype=amp_dtype,
                )
                response_matrix[src, :] = vec.numpy()
            except RuntimeError as exc:
                print(f"  [src {src:3d}] FAILED: {type(exc).__name__}: {exc}")
                n_failed += 1
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            except Exception as exc:  # noqa: BLE001
                print(f"  [src {src:3d}] FAILED: {type(exc).__name__}: {exc}")
                n_failed += 1
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            if (src + 1) % 16 == 0 or src == 0 or src == args.sources - 1:
                print(f"  [src {src:3d}/{args.sources}] "
                      f"({(time.time() - subj_t0) / 60:.2f} min)")

        # Median polish: remove source (row) + target (col) main effects,
        # leaving the a->b residual.
        connectivity = build_connectivity_relative(
            torch.from_numpy(response_matrix),
            row_norm=row_norm, col_norm=col_norm,
        ).numpy()

        elapsed_min = (time.time() - subj_t0) / 60.0
        n_succeeded = args.sources - n_failed
        payload = {
            "subject_idx": int(subject_idx),
            "subject_file": os.path.basename(subject_file),
            "n_trials": int(n_used),
            "n_sources_requested": int(args.sources),
            "n_sources_succeeded": int(n_succeeded),
            "method": "relative_response (median polish)",
            "sigma_ref": float(args.sigma_ref),
            "raw_response_matrix": response_matrix.tolist(),
            "connectivity_matrix": connectivity.tolist(),
            "connectivity_formula": (
                "median_polish(MSE_b^a(sigma_ref)) residual, "
                "row+col main effects removed"),
            "normalisation": {"row_norm": row_norm, "col_norm": col_norm},
            "hparams": {
                "sigma_ref": float(args.sigma_ref),
                "n_steps": int(args.n_steps),
                "batch_size": int(args.batch_size),
                "seed": int(args.seed),
            },
            "checkpoint": checkpoint_name,
            "n_source_failures": int(n_failed),
            "elapsed_min": float(elapsed_min),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        payload = sanitize_for_json(payload)
        out_json = os.path.join(args.output_dir, f"subject_{subject_idx}.json")
        with open(out_json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=json_default, allow_nan=False)
        print(f"  subject {subject_idx} done in {elapsed_min:.2f} min "
              f"({n_failed} failures) | wrote {out_json}")

    print(f"\n=== summary ===")
    print(f"subjects done: {len(subjects_to_run)}")
    print(f"elapsed: {(time.time() - t0_global) / 3600:.2f}h")
    print("Done.")


if __name__ == "__main__":
    main()
