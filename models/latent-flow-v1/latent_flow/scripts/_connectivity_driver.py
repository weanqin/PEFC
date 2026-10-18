"""Shared driver for OC-LatentFlow connectivity experiments.

The four connectivity scripts (two attribution runners, two probes) used to
each carry their own copy of: model/fold/subject loaders, JSON I/O, NaN
sanitisation, and the per-subject resume loop. Only the *per-source work*
differs between them — attribution does a sigma* search, a probe measures a
fixed-sigma response — everything else is identical.

This module owns the shared scaffold. A connectivity experiment is now:

    site = SignalProbeSite(...) or LatentProbeSite(...)   # the perturbation paradigm
    runner = SubjectRunner(output_dir, outputs_dir, n_sources, ...)
    for subject_file in subjects:
        x_obs = load_subject_trials(subject_file, device)
        site = make_site(model, x_obs, ...)               # paradigm-specific
        runner.run_subject(subject_idx, subject_file, x_obs, site, per_source_fn)

where ``per_source_fn(site, src) -> (sigma_vec (C,), meta dict)`` is the only
experiment-specific piece. For attribution it calls ``attribution_engine``;
for a probe it calls ``site.perturb_and_measure`` at fixed sigmas and
reduces.

Resume policy (unified): a NaN-filled ``(n_sources, n_targets)`` sigma matrix
is pre-allocated; finished sources are skipped, and previously-FAILED sources
are retried on resume. Partial JSON is written after every source so a long
run (7-15h, see domain.md) is interrupt-safe.
"""

from __future__ import annotations

import glob
import json
import os
import time
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.utils.data as data

from latent_flow.attribution.bisect_sigma import compute_connectivity
from latent_flow.config import CHECKPOINT_DIR
from latent_flow.data.datasets import get_dataset, list_hc, list_mdd
from latent_flow.data.modma import MODMADataset
from latent_flow.data.splitter import SubjectFoldSplitter
from latent_flow.models.oc_latent_flow import OCLatentFlow
from latent_flow.viz.connectivity import (
    plot_connectivity_heatmap,
    plot_connectivity_histogram,
    plot_sigma_matrix_heatmap,
    plot_sigma_histogram,
)

# Shared constants (channel count is dataset-derived, not hardcoded).
OBS_LEN = 375
FC_LEN = 125
N_CHANNELS = get_dataset("modma").channels  # MODMA default (back-compat)


# ---------------------------------------------------------------------------
# Loaders (shared verbatim across all four scripts)
# ---------------------------------------------------------------------------

def _detect_dataset_from_state(state):
    """Infer the dataset key from a checkpoint's state-dict buffer shapes.

    Channel-count buffers that vary by cohort:
      - gaussian graph variant: ``flow.graph_mixer.a_hat`` (C, C)
      - mexhat graph variant:   ``flow.graph_mixer.W`` and ``pairwise_d`` (C, C)
    The first matching buffer's dim gives the channel count (128 -> modma,
    62 -> edra). Dense-flow checkpoints have none of these and default to
    MODMA. An explicit caller override still wins.
    """
    for buf_key in ("flow.graph_mixer.a_hat", "flow.graph_mixer.W",
                    "flow.graph_mixer.pairwise_d"):
        buf = state.get(buf_key)
        if isinstance(buf, torch.Tensor) and buf.ndim == 2 and buf.shape[0] == buf.shape[1]:
            from latent_flow.data.datasets import _REGISTRY
            for key, desc in _REGISTRY.items():
                if desc.channels == buf.shape[0]:
                    return key
            break
    return "modma"


def load_model(checkpoint_path, device, dataset=None):
    """Instantiate OCLatentFlow() and load the flow checkpoint.

    The flow checkpoint is stage-2: encoder/proj/decoder + a trained flow.
    ``sample(use_proj=False)`` walks the flow (the production path). The model
    is deterministic at stage 2 (z_obs = mu_obs).

    Auto-detects the flow variant from the checkpoint's state-dict keys: if it
    contains ``flow.graph_mixer.*`` the model is built in graph-mode (R2 beta
    variant), otherwise the default dense flow is used. This keeps the loader
    backward-compatible with both legacy dense and new graph checkpoints.

    The dataset (MODMA 128ch vs EDRA 62ch) is auto-detected from the
    checkpoint's ``a_hat`` buffer shape unless *dataset* is given explicitly;
    it selects the geometry artifact used to rebuild the adjacency / coords.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"checkpoint not found: {checkpoint_path}. "
            "Train OC-LatentFlow stage 2 first "
            "(scripts/train_oc_latent_flow.py --stage 2)."
        )
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # Unwrap self-describing format (format_version=1): {state_dict, variant, format_version}
    if isinstance(payload, dict) and payload.get("format_version") == 1:
        state = payload["state_dict"]
    else:
        state = payload
    if dataset is None:
        dataset = _detect_dataset_from_state(state)
    is_graph = any(k.startswith("flow.graph_mixer.") for k in state.keys())
    is_mexhat = any(k.startswith("flow.graph_mixer.log_sigma") for k in state.keys())
    # DoG variant (task 07-30): stores log_sigma_e (+log_delta_sigma/log_Ae/
    # log_delta_a) instead of the single log_sigma. Detected first so it is not
    # misclassified as a plain gaussian-graph checkpoint.
    is_dog = any(k.startswith("flow.graph_mixer.log_sigma_e") for k in state.keys())

    # Channel count is dataset-derived so the mixer's W / pairwise_d / a_hat
    # buffers match the checkpoint (128 for MODMA, 62 for EDRA).
    n_channels = get_dataset(dataset).channels

    if is_dog:
        # Difference-of-Gaussians band-pass variant (task 07-30): needs coords
        # for the pairwise distance matrix. DoG params load from the state_dict.
        from latent_flow.models.graph_utils import load_geometry
        coords, _ = load_geometry(dataset=dataset)
        gamma_m = float(state.get("flow.graph_mixer.gamma_m", torch.tensor(1.0)))
        gamma_w = float(state.get("flow.graph_mixer.gamma_buf", torch.tensor(0.1)))
        model = OCLatentFlow(n_channels=n_channels, flow_graph_mode=True,
                             flow_graph_kernel="dog", flow_coords=coords,
                             flow_mexhat_gamma=gamma_w, flow_mexhat_gamma_m=gamma_m)
    elif is_mexhat:
        # Mexican-hat band-pass variant (B.5): needs electrode coords.
        from latent_flow.models.graph_utils import load_geometry
        coords, _ = load_geometry(dataset=dataset)
        # mexhat kernel scale gamma_m (task 07-15 ablation): default 1.0 for
        # older checkpoints that predate the gamma_m buffer.
        gamma_m = float(state.get("flow.graph_mixer.gamma_m", torch.tensor(1.0)))
        # W residual weight gamma (stored as gamma_buf buffer in newer ckpts).
        gamma_w = float(state.get("flow.graph_mixer.gamma_buf", torch.tensor(0.1)))
        model = OCLatentFlow(n_channels=n_channels, flow_graph_mode=True,
                             flow_graph_kernel="mexhat", flow_coords=coords,
                             flow_mexhat_gamma=gamma_w, flow_mexhat_gamma_m=gamma_m)
    elif is_graph:
        # Gaussian graph-mode variant: build the geometry adjacency for the mixer buffer.
        from latent_flow.models.graph_utils import build_normalized_adj, load_geometry
        coords_all, _ = load_geometry(dataset=dataset)
        a_hat = build_normalized_adj(coords_all).to(device)
        model = OCLatentFlow(n_channels=n_channels, flow_graph_mode=True, flow_a_hat=a_hat)
    else:
        model = OCLatentFlow()
    # strict=False: tolerate legacy checkpoints missing the gamma_m buffer
    # (defaults to 1.0 via the constructor), and tolerate sc_base/gamma_sc keys
    # in checkpoints trained before the SC prior was deprecated (silently
    # dropped — the SC contribution is no longer applied). Any other mismatch
    # is an error.
    missing, unexpected = model.load_state_dict(state, strict=False)
    allow_missing = {"flow.graph_mixer.gamma_m"}
    allow_unexpected = {"flow.graph_mixer.gamma_sc", "flow.graph_mixer.sc_base"}
    real_missing = set(missing) - allow_missing
    real_unexpected = set(unexpected) - allow_unexpected
    assert not real_missing and not real_unexpected, (
        f"state_dict mismatch loading {checkpoint_path}: "
        f"missing={sorted(real_missing)}, unexpected={sorted(real_unexpected)}")
    model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    variant = ("dog-graph" if is_dog
               else "mexhat-graph" if is_mexhat
               else "gaussian-graph" if is_graph
               else "dense")
    print(
        f"Loaded checkpoint: {os.path.basename(checkpoint_path)} "
        f"({n_params / 1e6:.2f}M params, stage-2 {variant} flow, dataset={dataset})"
    )
    return model


def get_fold_test_subjects(fold, seed, dataset="modma"):
    """Return list of test subject files for the given fold (5-fold subject CV).

    *dataset* selects the cohort ("modma" -> .mat in HC/MDD subdirs,
    "edra" -> .vhdr flat H_*/M_* layout).
    """
    desc = get_dataset(dataset)
    files_mdd = list_mdd(desc)
    files_hc = list_hc(desc)
    splitter = SubjectFoldSplitter(files_mdd, files_hc, n_folds=5, seed=seed)
    _, test_files = splitter.get_fold(fold)
    print(
        f"Fold {fold} test: {len(test_files)} subjects "
        f"({len(files_mdd)} {desc.mdd_label} + {len(files_hc)} {desc.hc_label} available)"
    )
    return test_files


def load_subject_trials(subject_file, device, normalize=True, dataset="modma"):
    """Stack one subject's x_obs windows into (N, T_obs, C).

    Mirrors MODMADataset (per-channel z-score when normalize=True) so the input
    distribution matches what the checkpoint saw during training.
    """
    ds = MODMADataset(
        subject_files=[subject_file],
        group="MIX",
        normalize_per_channel=normalize,
        obs_size=OBS_LEN,
        dataset=dataset,
    )
    return torch.stack([w["x_obs"] for w in ds.windows]).to(device)


def subsample_trials(x_obs_full, n_trials, seed, subject_idx):
    """Optionally cap the number of trials (for faster smoke runs).

    Returns (x_obs, n_used). n_trials<=0 or >= N => use all.
    """
    n = x_obs_full.shape[0]
    if n_trials is None or n_trials <= 0 or n_trials >= n:
        return x_obs_full, n
    rng = np.random.RandomState(seed * 1000 + subject_idx)
    idx = rng.choice(n, size=n_trials, replace=False)
    return x_obs_full[idx], n_trials


def compute_stats(response_matrix):
    """Summary statistics for a (n_sources, n_targets) response matrix.

    Used by the fixed-sigma probes. CV = std/mean is the headline metric
    compared against the JET v2 ceiling (9.5%-17.5% per domain.md).
    """
    arr = np.asarray(response_matrix, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {
            "mean": float("nan"), "std": float("nan"),
            "min": float("nan"), "max": float("nan"),
            "cv": float("nan"), "per_source_out_degree_mean": [],
        }
    per_source_out = arr.mean(axis=1)
    mean = float(finite.mean())
    cv = float(finite.std() / mean) if mean > 0 else float("nan")
    return {
        "mean": mean,
        "std": float(finite.std()),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "cv": cv,
        "per_source_out_degree_mean": per_source_out.tolist(),
    }


# ---------------------------------------------------------------------------
# JSON I/O (shared verbatim)
# ---------------------------------------------------------------------------

def json_default(obj):
    """JSON default handler for numpy/torch scalars."""
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    raise TypeError(f"not JSON serializable: {type(obj)}")


def sanitize_for_json(obj):
    """Recursively convert non-finite floats (NaN/Inf) to None for JSON.

    ``json.dump(..., allow_nan=False)`` rejects NaN/Inf; sources that failed
    produce NaN rows, which must be null in the output.
    """
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, (np.floating,)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def load_partial(out_json, n_sources, n_targets=N_CHANNELS):
    """Load a partial sigma matrix for resume.

    Returns ``(sigma_matrix, source_errors, n_done)`` where sigma_matrix is a
    pre-allocated ``(n_sources, n_targets)`` float32 array filled with NaN for
    unfinished rows, and finished (non-NaN) rows for done sources. source_errors
    maps int source idx -> error message. n_done counts completed sources.

    Failed sources (NaN row recorded previously) are NOT counted as done, so
    they are retried on resume.
    """
    sigma_matrix = np.full((n_sources, n_targets), float("nan"), dtype=np.float32)
    source_errors = {}
    if not os.path.exists(out_json):
        return sigma_matrix, source_errors, 0
    try:
        with open(out_json, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (json.JSONDecodeError, OSError):
        return sigma_matrix, source_errors, 0
    saved = payload.get("sigma_matrix", [])
    errs = payload.get("source_errors", {})
    for i, row in enumerate(saved):
        if i >= n_sources:
            break
        if any(v is not None for v in row):
            sigma_matrix[i, :] = [float("nan") if v is None else float(v) for v in row]
    # Normalise error keys to int.
    for k, v in errs.items():
        source_errors[int(k)] = v
    n_done = int(sum(
        1 for i in range(n_sources)
        if not np.isnan(sigma_matrix[i]).all() and i not in source_errors
    ))
    return sigma_matrix, source_errors, n_done


def write_subject_json(out_json, subject_idx, subject_file, n_trials,
                       n_sources, sigma_matrix, source_errors, extra_payload,
                       checkpoint_name, elapsed_min):
    """Write (overwrite) the per-subject JSON with the current sigma matrix."""
    connectivity_matrix = (
        compute_connectivity(torch.from_numpy(sigma_matrix.copy())).numpy()
    )
    payload = {
        "subject_idx": int(subject_idx),
        "subject_file": os.path.basename(subject_file),
        "n_trials": int(n_trials),
        "n_sources_requested": int(n_sources),
        "n_sources_succeeded": int(sum(
            1 for i in range(n_sources)
            if not np.isnan(sigma_matrix[i]).all() and i not in source_errors
        )),
        "sigma_matrix": sigma_matrix.tolist(),
        "connectivity_matrix": connectivity_matrix.tolist(),
        "connectivity_formula": "-(log sigma* + 0.5*log(2*pi*e))",
        "checkpoint": checkpoint_name,
        "source_errors": {str(k): v for k, v in source_errors.items()},
        "elapsed_min": float(elapsed_min),
    }
    payload.update(extra_payload)
    payload = sanitize_for_json(payload)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=json_default, allow_nan=False)


def write_pngs(outputs_dir, subject_idx, sigma_matrix, connectivity_matrix,
               title_prefix="OC-LatentFlow"):
    """Write the 4 standard plots (sigma heatmap/hist, connectivity heatmap/hist)."""
    os.makedirs(outputs_dir, exist_ok=True)
    base = os.path.join(outputs_dir, f"subject_{subject_idx}")
    plot_sigma_matrix_heatmap(
        sigma_matrix, f"{base}_sigma_heatmap.png",
        title=f"{title_prefix} sigma matrix (subject {subject_idx})",
    )
    plot_connectivity_heatmap(
        connectivity_matrix, f"{base}_conn_heatmap.png",
        title=f"{title_prefix} connectivity (subject {subject_idx})",
    )
    plot_sigma_histogram(
        sigma_matrix, f"{base}_sigma_hist.png",
        title=f"{title_prefix} sigma histogram (subject {subject_idx})",
    )
    plot_connectivity_histogram(
        connectivity_matrix, f"{base}_conn_hist.png",
        title=f"{title_prefix} connectivity histogram (subject {subject_idx})",
    )


# ---------------------------------------------------------------------------
# Per-source work contract
# ---------------------------------------------------------------------------

# A per-source callable: given the (already-set) site and the source index,
# return (sigma_vec (C,), meta dict). For attribution this runs the engine;
# for a probe it measures at fixed sigmas and reduces.
PerSourceFn = Callable[["ProbeSite", int], Tuple[torch.Tensor, Dict]]


# ---------------------------------------------------------------------------
# SubjectRunner — the resumable per-subject driver
# ---------------------------------------------------------------------------

class SubjectRunner:
    """Drives one subject's per-source sweep with interrupt-safe resume.

    Owns: the NaN-matrix sigma accumulation, the resume load, the incremental
    JSON/PNG writes, the per-source try/except (OOM recovery), and the
    progress/ETA. Does NOT own the per-source computation — that is the
    ``per_source_fn`` argument, so attribution and probes share one driver.
    """

    def __init__(self, output_dir, outputs_dir, n_sources,
                 checkpoint_name, device, title_prefix="OC-LatentFlow",
                 resume=True, progress_state=None):
        self.output_dir = output_dir
        self.outputs_dir = outputs_dir
        self.n_sources = n_sources
        self.checkpoint_name = checkpoint_name
        self.device = device
        self.title_prefix = title_prefix
        self.resume = resume
        self.progress_state = progress_state  # optional cross-subject dict

    def run_subject(self, subject_idx, subject_file, x_obs_subject, site,
                    per_source_fn, extra_payload=None):
        """Sweep all sources for one subject; resumable.

        Args:
            subject_idx, subject_file: identifiers (for output paths/logging).
            x_obs_subject: (N, T_obs, C) the subject's observation windows.
            site: a ProbeSite (SignalProbeSite / LatentProbeSite / future).
                ``per_source_fn`` is responsible for calling site.set_source
                if its paradigm needs per-source state.
            per_source_fn: ``(site, src) -> (sigma_vec (C,), meta dict)``.
            extra_payload: dict merged into the subject JSON (method, probe_site,
                hparams, etc. — experiment-specific metadata).
        """
        extra_payload = extra_payload or {}
        n_trials = x_obs_subject.shape[0]
        out_json = os.path.join(self.output_dir, f"subject_{subject_idx}.json")
        print(
            f"\n=== subject {subject_idx} "
            f"({os.path.basename(subject_file)}) === "
            f"{n_trials} trials, sweeping {self.n_sources} sources"
        )

        sigma_matrix, source_errors, n_done = (
            load_partial(out_json, self.n_sources) if self.resume
            else (np.full((self.n_sources, N_CHANNELS), float("nan"),
                          dtype=np.float32), {}, 0)
        )
        if n_done > 0:
            print(f"  resume: {n_done}/{self.n_sources} sources done, skipping")

        subject_t0 = time.time()
        for src in range(self.n_sources):
            # Skip finished sources; retry previously-failed ones.
            if not np.isnan(sigma_matrix[src]).all() and src not in source_errors:
                self._bump_progress()
                continue
            source_errors.pop(src, None)

            src_t0 = time.time()
            try:
                sigma_vec, meta = per_source_fn(site, src)
                sigma_matrix[src, :] = sigma_vec.numpy()
            except RuntimeError as exc:
                msg = f"RuntimeError: {type(exc).__name__}: {exc}"
                print(f"  [src {src:3d}] FAILED: {msg}")
                source_errors[src] = msg
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
            except Exception as exc:  # noqa: BLE001
                import traceback
                tb = traceback.format_exc()
                msg = f"{type(exc).__name__}: {exc}\n" + "\n".join(tb.splitlines()[-4:])
                print(f"  [src {src:3d}] FAILED: {msg}")
                source_errors[src] = msg
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

            write_subject_json(
                out_json, subject_idx, subject_file, n_trials,
                self.n_sources, sigma_matrix, source_errors,
                extra_payload, self.checkpoint_name,
                (time.time() - subject_t0) / 60.0,
            )
            self._log_progress(src, src_t0)

        connectivity_matrix = compute_connectivity(
            torch.from_numpy(sigma_matrix.copy())
        ).numpy()
        write_pngs(self.outputs_dir, subject_idx, sigma_matrix,
                   connectivity_matrix, title_prefix=self.title_prefix)
        subject_dt = (time.time() - subject_t0) / 60.0
        print(
            f"  subject {subject_idx} done in {subject_dt:.2f} min "
            f"({len(source_errors)} failures) | wrote JSON + 4 PNGs"
        )

    def _bump_progress(self):
        if self.progress_state is None:
            return
        self.progress_state["sources_done"] += 1

    def _log_progress(self, src, src_t0):
        src_dt = (time.time() - src_t0) / 60.0
        if self.progress_state is not None:
            self.progress_state["sources_done"] += 1
            self.progress_state["elapsed_h"] = (
                (time.time() - self.progress_state["t0_global"]) / 3600.0
            )
            done = max(self.progress_state["sources_done"], 1)
            mean_h = self.progress_state["elapsed_h"] / done
            remaining = self.progress_state["sources_total"] - done
            eta_h = mean_h * remaining
            if (src + 1) % 5 == 0 or src == 0 or src == self.n_sources - 1:
                print(
                    f"  [{self.progress_state['subjects_done'] + 1}/"
                    f"{self.progress_state['subjects_total']} subjects, "
                    f"{self.progress_state['sources_done']}/"
                    f"{self.progress_state['sources_total']} sources] "
                    f"src={src:3d} took {src_dt:.2f} min | "
                    f"elapsed={self.progress_state['elapsed_h']:.2f}h eta={eta_h:.1f}h"
                )
        else:
            if (src + 1) % 5 == 0 or src == 0 or src == self.n_sources - 1:
                print(f"  [src {src:3d}] took {src_dt:.2f} min")
