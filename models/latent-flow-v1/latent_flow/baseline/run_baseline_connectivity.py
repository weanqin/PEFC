"""Traditional directed-connectivity baseline runner (task 06-25).

Estimates per-subject directed 128x128 connectivity matrices for the
fold-0 test subjects using classical methods (independent of the trained
forecast model), for direct comparison with the OC-LatentFlow attribution
matrices in ``results/oc_connectivity/``.

Methods (``--methods``):
    gc  -- pairwise bivariate spectral Granger Causality (Geweke 1982),
           full-band integral. Kernel: ``baseline.granger``.
    te  -- symbolic Transfer Entropy, Bandt-Pompe m=3 tau=1. Kernel:
           ``baseline.transfer_entropy``.
    gpd -- generalized Partial Directed Coherence (Baccala et al. 2007),
           MVAR-based frequency-domain directed partialised connectivity.
           Kernel: ``baseline.gpdc``.

Output (relative to repo root), one JSON + heatmaps per (subject, method):
    results/baseline_connectivity/{gc,te}/subject_{idx}.json
    outputs/baseline_connectivity/{gc,te}/subject_{idx}_conn_heatmap.png
    outputs/baseline_connectivity/{gc,te}/subject_{idx}_conn_hist.png

JSON schema mirrors ``results/oc_connectivity/subject_*.json`` (same
``connectivity_matrix`` field name + directed row=source/col=target axis
convention) so the two can be diffed/compared directly.

Interrupt-resumability: each subject's matrix is written after EVERY
source completes, so an interrupted long run continues from where it
stopped (``--resume`` skips completed subjects; TE has a single pass per
subject so resumability is per-subject for TE).

DCM was attempted as a third baseline (task 07-19) but is **deprecated**:
pairwise 2-node spectral DCM is statistically under-identified on
realistic EEG, and both MAP and full Variational Laplace inversions
collapsed to the prior mean (output ~0.0078 across all pairs). The
``baseline/dcm.py`` kernel + ``tests/test_dcm_*.py`` are kept for
reference but not wired into this runner. See
``.trellis/tasks/07-19-dcm-ec-baseline/`` for the full failure analysis.

Usage (run from repo root D:/102-npi/ude-vae-npi):

    # smoke: 1 subject, GC + TE
    python -u -m latent_flow.baseline.run_baseline_connectivity \
        --fold 0 --subjects 1 --methods gc,te

    # full fold 0: 10 subjects, GC + TE + gPDC (resumable)
    python -u -m latent_flow.baseline.run_baseline_connectivity \
        --fold 0 --subjects 10 --methods gc,te,gpd --resume
"""

import argparse
import datetime as _dt
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

from latent_flow.baseline.data_utils import (
    get_fold_test_subjects,
    load_subject_timeseries,
)
from latent_flow.baseline.granger import (
    default_params as gc_params,
    granger_connectivity,
)
from latent_flow.baseline.transfer_entropy import (
    default_params as te_params,
    transfer_entropy_connectivity,
)
from latent_flow.baseline.gpdc import (
    default_params as gpdc_params,
    gpdc_connectivity,
)
from latent_flow.config import FS
from latent_flow.viz.connectivity import (
    plot_connectivity_heatmap,
    plot_connectivity_histogram,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Channel count is dataset-derived; N_CHANNELS kept as the MODMA default for
# any back-compat reference (the actual count comes from the loaded timeseries).
from latent_flow.data.datasets import get_dataset as _get_dataset
N_CHANNELS = _get_dataset("modma").channels
DEFAULT_RESULTS_DIR = "results/baseline_connectivity"
DEFAULT_OUTPUTS_DIR = "outputs/baseline_connectivity"


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Traditional directed-connectivity baselines (GC + symbolic TE + gPDC) "
            "for fold test subjects. Model-independent baselines for "
            "comparison with OC-LatentFlow attribution."
        ),
    )
    p.add_argument("--fold", type=int, default=0, choices=range(5))
    p.add_argument("--dataset", type=str, default="modma",
                   choices=("modma", "edra"),
                   help="EEG cohort: 'modma' (128ch .mat) or 'edra' (62ch "
                        "BrainVision). Channel count is derived from the "
                        "dataset. Default modma.")
    p.add_argument("--seed", type=int, default=42,
                   help="seed for the subject splitter (default 42).")
    p.add_argument("--subjects", type=int, default=10,
                   help="number of fold test subjects to process (default 10).")
    p.add_argument("--subject-idx", type=int, default=0,
                   help="0-based subject index to start from (default 0).")
    p.add_argument("--methods", type=str, default="gc,te",
                   help="comma-separated subset of {gc,te,gpd} (default 'gc,te').")
    # GC hyperparameters.
    p.add_argument("--gc-max-order", type=int, default=8,
                   help="GC: max AR order; actual order AIC-selected per pair.")
    p.add_argument("--gc-fixed-order", type=int, default=8,
                   help="GC: fixed AR order for all pairs (skip AIC sweep, "
                        "much faster). EEG GC commonly uses one fixed order "
                        "(default 8).")
    p.add_argument("--gc-n-freq", type=int, default=65,
                   help="GC: frequency bins over the band for the integral.")
    p.add_argument("--gc-fmin", type=float, default=1.0,
                   help="GC: lower integration bound (Hz).")
    p.add_argument("--gc-fmax", type=float, default=125.0,
                   help="GC: upper integration bound (Hz; Nyquist at 250Hz).")
    # TE hyperparameters.
    p.add_argument("--te-m", type=int, default=3,
                   help="TE: Bandt-Pompe embedding dimension (default 3).")
    p.add_argument("--te-tau", type=int, default=1,
                   help="TE: embedding delay in samples (default 1).")
    # gPDC hyperparameters.
    p.add_argument("--gpd-max-order", type=int, default=8,
                   help="gPDC: max AR order; actual order AIC-selected (default 8).")
    p.add_argument("--gpd-fixed-order", type=int, default=8,
                   help="gPDC: fixed AR order for all pairs (skip AIC sweep, "
                        "much faster). Default 8 matches GC's fixed order.")
    p.add_argument("--gpd-n-freq", type=int, default=65,
                   help="gPDC: frequency bins over the band for the integral.")
    p.add_argument("--gpd-fmin", type=float, default=1.0,
                   help="gPDC: lower integration bound (Hz).")
    p.add_argument("--gpd-fmax", type=float, default=125.0,
                   help="gPDC: upper integration bound (Hz).")
    # Parallelism (GC is the bottleneck: ~2s/pair x 128x127 pairs/subject).
    p.add_argument("--n-jobs", type=int, default=-1,
                   help="joblib workers for GC (one per source channel). "
                        "-1 = all cores (default); 1 = serial.")
    # I/O.
    p.add_argument("--results-dir", type=str, default=DEFAULT_RESULTS_DIR)
    p.add_argument("--outputs-dir", type=str, default=DEFAULT_OUTPUTS_DIR)
    p.add_argument("--resume", action="store_true",
                   help="skip subjects whose JSON already exists.")
    p.add_argument("--verbose-kernel", action="store_true",
                   help="pass verbose=True to the per-pair kernels (progress).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# JSON I/O (mirror oc_connectivity schema + NaN-safe)
# ---------------------------------------------------------------------------

def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON serializable: {type(obj)}")


def _sanitize(obj):
    """Convert NaN/Inf to string sentinels so json.dump(allow_nan=False) works."""
    if isinstance(obj, float):
        if math.isnan(obj):
            return "NaN"
        if math.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


def _subject_done(json_path):
    """True iff the subject JSON already exists (resume short-circuit)."""
    return os.path.exists(json_path)


def write_subject_json(json_path, subject_idx, subject_file, n_samples,
                       method, method_params, connectivity_matrix, elapsed_min):
    """Write one (subject, method) JSON. Schema mirrors oc_connectivity."""
    payload = {
        "subject_idx": int(subject_idx),
        "subject_file": os.path.basename(subject_file),
        "n_samples": int(n_samples),
        "method": method,
        "method_params": method_params,
        "connectivity_matrix": np.asarray(connectivity_matrix,
                                          dtype=np.float32).tolist(),
        "fs": float(FS),
        "elapsed_min": float(elapsed_min),
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    payload = _sanitize(payload)
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_json_default, allow_nan=False)


def write_pngs(outputs_dir, method, subject_idx, connectivity_matrix):
    """Render heatmap + histogram for one (subject, method)."""
    out_dir = os.path.join(outputs_dir, method)
    os.makedirs(out_dir, exist_ok=True)
    title = f"{method.upper()} connectivity (subject {subject_idx})"
    plot_connectivity_heatmap(
        connectivity_matrix,
        os.path.join(out_dir, f"subject_{subject_idx}_conn_heatmap.png"),
        title=title,
    )
    plot_connectivity_histogram(
        connectivity_matrix,
        os.path.join(out_dir, f"subject_{subject_idx}_conn_hist.png"),
        title=title,
    )


# ---------------------------------------------------------------------------
# Per-subject drivers
# ---------------------------------------------------------------------------

def run_subject_gc(data, subject_idx, subject_file, args, results_dir,
                   outputs_dir):
    """GC: pairwise bivariate spectral GC for all 128 channels."""
    out_json = os.path.join(results_dir, "gc", f"subject_{subject_idx}.json")
    if args.resume and _subject_done(out_json):
        print(f"  [gc] resume: subject {subject_idx} done, skipping")
        return

    n_ch, n_samp = data.shape
    print(f"  [gc] {n_ch} channels x {n_samp} samples, "
          f"order<= {args.gc_max_order} (AIC), "
          f"band {args.gc_fmin}-{args.gc_fmax} Hz")
    t0 = time.time()
    conn = granger_connectivity(
        data, fs=FS, max_order=args.gc_max_order, n_freq=args.gc_n_freq,
        fmin=args.gc_fmin, fmax=args.gc_fmax, verbose=args.verbose_kernel,
        n_jobs=args.n_jobs, fixed_order=args.gc_fixed_order,
    )
    elapsed = (time.time() - t0) / 60.0
    params = gc_params(args.gc_max_order, args.gc_n_freq,
                       args.gc_fmin, args.gc_fmax)
    write_subject_json(out_json, subject_idx, subject_file, n_samp,
                       params["method"], params, conn, elapsed)
    write_pngs(outputs_dir, "gc", subject_idx, conn)
    finite = conn[np.isfinite(conn)]
    print(f"  [gc] subject {subject_idx} done in {elapsed:.2f} min | "
          f"mean={finite.mean():.4f} max={finite.max():.4f}")


def run_subject_te(data, subject_idx, subject_file, args, results_dir,
                   outputs_dir):
    """TE: symbolic Transfer Entropy (Bandt-Pompe m, tau)."""
    out_json = os.path.join(results_dir, "te", f"subject_{subject_idx}.json")
    if args.resume and _subject_done(out_json):
        print(f"  [te] resume: subject {subject_idx} done, skipping")
        return

    n_ch, n_samp = data.shape
    print(f"  [te] {n_ch} channels x {n_samp} samples, "
          f"m={args.te_m} tau={args.te_tau}")
    t0 = time.time()
    conn = transfer_entropy_connectivity(
        data, m=args.te_m, tau=args.te_tau, verbose=args.verbose_kernel,
    )
    elapsed = (time.time() - t0) / 60.0
    params = te_params(args.te_m, args.te_tau)
    write_subject_json(out_json, subject_idx, subject_file, n_samp,
                       "symbolic_te", params, conn, elapsed)
    write_pngs(outputs_dir, "te", subject_idx, conn)
    finite = conn[np.isfinite(conn)]
    print(f"  [te] subject {subject_idx} done in {elapsed:.2f} min | "
          f"mean={finite.mean():.5f} max={finite.max():.5f}")


def run_subject_gpdc(data, subject_idx, subject_file, args, results_dir,
                      outputs_dir):
    """gPDC: generalized Partial Directed Coherence (MVAR-based)."""
    out_json = os.path.join(results_dir, "gpd", f"subject_{subject_idx}.json")
    if args.resume and _subject_done(out_json):
        print(f"  [gpd] resume: subject {subject_idx} done, skipping")
        return

    n_ch, n_samp = data.shape
    print(f"  [gpd] {n_ch} channels x {n_samp} samples, "
          f"max_order={args.gpd_max_order} (AIC), "
          f"band {args.gpd_fmin}-{args.gpd_fmax} Hz")
    t0 = time.time()
    conn = gpdc_connectivity(
        data, fs=FS, max_order=args.gpd_max_order,
        n_freq=args.gpd_n_freq, fmin=args.gpd_fmin, fmax=args.gpd_fmax,
        verbose=args.verbose_kernel, n_jobs=args.n_jobs,
        fixed_order=args.gpd_fixed_order,
    )
    elapsed = (time.time() - t0) / 60.0
    params = gpdc_params(args.gpd_max_order, args.gpd_n_freq,
                          args.gpd_fmin, args.gpd_fmax)
    write_subject_json(out_json, subject_idx, subject_file, n_samp,
                       "gpdc", params, conn, elapsed)
    write_pngs(outputs_dir, "gpd", subject_idx, conn)
    finite = conn[np.isfinite(conn)]
    print(f"  [gpd] subject {subject_idx} done in {elapsed:.2f} min | "
          f"mean={finite.mean():.5f} max={finite.max():.5f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]
    bad = [m for m in methods if m not in ("gc", "te", "gpd")]
    if bad:
        raise SystemExit(f"unknown methods: {bad} (allowed: gc, te, gpd)")
    print(f"Methods: {methods}")

    test_files = get_fold_test_subjects(args.fold, args.seed, dataset=args.dataset)
    subjects = test_files[args.subject_idx:args.subject_idx + args.subjects]
    print(f"Subjects: {len(subjects)} (indices {args.subject_idx}.."
          f"{args.subject_idx + len(subjects) - 1}, dataset={args.dataset})")

    t_global = time.time()
    for i, subject_file in enumerate(subjects):
        subject_idx = args.subject_idx + i
        print(f"\n=== subject {subject_idx} "
              f"({os.path.basename(subject_file)}) ===")
        data = load_subject_timeseries(subject_file, normalize=True,
                                       dataset=args.dataset)
        if "gc" in methods:
            run_subject_gc(data, subject_idx, subject_file, args,
                           args.results_dir, args.outputs_dir)
        if "te" in methods:
            run_subject_te(data, subject_idx, subject_file, args,
                           args.results_dir, args.outputs_dir)
        if "gpd" in methods:
            run_subject_gpdc(data, subject_idx, subject_file, args,
                             args.results_dir, args.outputs_dir)

    print("\n=== summary ===")
    print(f"subjects processed: {len(subjects)}")
    print(f"methods: {methods}")
    print(f"elapsed: {(time.time() - t_global) / 60.0:.2f} min")
    print("Done.")


if __name__ == "__main__":
    main()
