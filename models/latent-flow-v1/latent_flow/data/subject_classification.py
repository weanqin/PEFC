"""Subject-level labeled dataset for the fold-0 EC-graph classification task.

One sample = one subject. Each item carries:
    - ``de_features``: (C, 5) mean-aggregated differential-entropy node features.
    - ``ec_matrix``:   (C, C) directed effective-connectivity graph from one of
                       four precomputed sources (DE-EC / GC / TE / gPDC).
    - ``label``:       int (0=HC, 1=MDD), resolved from the ``modma_resting``
                       ``HC/`` / ``MDD/`` subdir layout (the json ``subject_file``
                       names do NOT encode the label — see PRD Background).
    - ``subject_id``:  the subject's ``.mat`` basename, for provenance.

EC sources are all **precomputed** for fold 0 (verified on disk 2026-07-31):
    - DE-EC:  ``results/oc_connectivity_global_tau_signed_signal_q0.5/subject_*.json``
              key ``signed_connectivity_matrix`` (directed, signed).
    - GC/TE/gPDC: ``results/baseline_connectivity_fold0/{gc,te,gpd}/subject_*.json``
              key ``connectivity_matrix`` (directed, unsigned).

Leakage control (PRD R5 / AC5): the DE-EC loader asserts that the json's
``checkpoint`` field names the *same* fold as the dataset's ``fold`` argument
(regex on the word "fold" followed by digits). Baselines are per-subject-EEG
(leakage-free) and the on-disk ``baseline_connectivity_fold{k}/`` dir is already
partitioned by the same ``SubjectFoldSplitter`` fold, so loading fold-k baselines
from fold-k's dir cannot pull a train subject's EC.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from latent_flow.config import DATA_ROOT, FS
from latent_flow.data.de_features import aggregate_subject_de, compute_de_features
from latent_flow.data.modma import extract_windows, load_mat_file

# Graph-source keys accepted by load_ec(). "gpdc" is the public key (matches the
# method's standard name and the json "method" field); the on-disk baseline dir
# is "gpd" (legacy short form), mapped via _BASELINE_DIR.
#   - "mean_diff" : OC DE-probe biased-noise mean-response matrix (signed, full-sample).
#   - "signed_de" : OC DE-probe signed connectivity = sign(mean_diff) * DE-strength
#                   (the canonical signed directed EC; full-sample, nosc variant).
GRAPH_SOURCES = ("mean_diff", "signed_de", "de_ec", "gc", "te", "gpdc")

# Map graph_source → on-disk baseline subdir name.
_BASELINE_DIR = {"gc": "gc", "te": "te", "gpdc": "gpd"}

# Absolute path to the project root (latent_flow/). Derived from this file's
# location (data/subject_classification.py -> parents[1] is the package root).
# NOTE: previously this used DATA_ROOT.parent (the ude-vae-npi repo parent);
# after migration the project is self-contained, so derive from __file__.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]

# DE-EC results dir (relative to the latent_flow package root). Resolved lazily
# so the import works from any cwd.
_DE_EC_DIR = _PROJECT_ROOT / "results" / "oc_connectivity_global_tau_signed_signal_q0.5"

# Full-sample (50 subj, 5-fold) OC DE-probe ablation dir. Each fold uses its own
# OC checkpoint (leakage-safe). "nosc" is the primary variant (no SC injection);
# now/nomexhat are near-duplicates (r≈0.998 vs nosc) and not exposed as separate
# graph sources. The "mean_diff_matrix" key is the signed biased-noise response.
_MEAN_DIFF_ROOT = _PROJECT_ROOT / "results" / "oc_connectivity_global_tau_signed_ablation"
_MEAN_DIFF_VARIANT = "nosc"

# Baseline connectivity roots: one dir per fold. These baseline runs were
# originally under the ude-vae-npi repo parent; migrated as a curated store
# that now lives under archive/results_repo/ (gitignored). New baseline runs
# should write to results/baseline_connectivity_fold{k}/ instead.
_BASELINE_ROOT = _PROJECT_ROOT / "archive" / "results_repo"


def resolve_label(subject_file: str) -> int:
    """Return 1 (MDD) or 0 (HC) by checking which modma_resting subdir holds the file.

    The ``subject_file`` field in the EC jsons (e.g. ``02010013EOG.set_59_.mat``)
    contains no HC/MDD marker, so the label MUST be recovered from the on-disk
    path. Raises if the file is in neither subdir.
    """
    name = os.path.basename(subject_file)
    if (DATA_ROOT / "MDD" / name).exists():
        return 1
    if (DATA_ROOT / "HC" / name).exists():
        return 0
    raise FileNotFoundError(
        f"subject file {name!r} not in {DATA_ROOT}/HC or {DATA_ROOT}/MDD; "
        "cannot resolve HC/MDD label"
    )


def _find_json_by_subject(json_dir: Path, subject_file: str) -> Optional[Path]:
    """Return the json in ``json_dir`` whose ``subject_file`` field matches."""
    name = os.path.basename(subject_file)
    for jp in sorted(json_dir.glob("subject_*.json")):
        try:
            with open(jp) as f:
                d = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if os.path.basename(str(d.get("subject_file", ""))) == name:
            return jp
    return None


# Cache: subject_file basename → json path, built lazily over all 5 fold dirs.
_MEAN_DIFF_INDEX: Optional[dict[str, Path]] = None


def _build_mean_diff_index() -> dict[str, Path]:
    """Build (and cache) a {subject_file basename → json path} map across all folds.

    Each subject appears in exactly one fold dir (where it was the held-out test
    subject). Built once per process; the 50-subject scan is negligible cost.
    """
    global _MEAN_DIFF_INDEX
    if _MEAN_DIFF_INDEX is not None:
        return _MEAN_DIFF_INDEX
    idx: dict[str, Path] = {}
    for fk in range(5):
        jdir = _MEAN_DIFF_ROOT / f"fold{fk}" / _MEAN_DIFF_VARIANT
        for jp in sorted(jdir.glob("subject_*.json")):
            try:
                with open(jp) as f:
                    d = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            sf = os.path.basename(str(d.get("subject_file", "")))
            if sf:
                idx[sf] = jp
    _MEAN_DIFF_INDEX = idx
    return idx


def _locate_mean_diff(subject_file: str) -> Optional[Path]:
    """Return the mean_diff json path for a subject (searched across all 5 folds)."""
    return _build_mean_diff_index().get(os.path.basename(subject_file))


# Cache: (graph_source, subject_file) → json path, for per-fold-organized sources
# (baselines live in baseline_connectivity_fold{k}, one entry per subject in its
# own fold). Built lazily per graph_source.
_BASELINE_INDEX: dict[str, dict[str, Path]] = {}


def _locate_baseline(subject_file: str, graph_source: str) -> Optional[Path]:
    """Return the baseline json path for a subject, searched across all 5 fold dirs.

    Baselines are per-subject-EEG (leakage-free) but stored per-fold (one entry per
    subject, in the fold where it is a test subject). The ``fold`` argument to
    ``load_ec`` is therefore NOT used to resolve the path for baselines.
    """
    sub = _BASELINE_DIR[graph_source]
    if sub not in _BASELINE_INDEX:
        idx: dict[str, Path] = {}
        for fk in range(5):
            jdir = _BASELINE_ROOT / f"baseline_connectivity_fold{fk}" / sub
            for jp in sorted(jdir.glob("subject_*.json")):
                try:
                    with open(jp) as f:
                        d = json.load(f)
                except (OSError, json.JSONDecodeError):
                    continue
                sf = os.path.basename(str(d.get("subject_file", "")))
                if sf:
                    idx[sf] = jp
        _BASELINE_INDEX[sub] = idx
    return _BASELINE_INDEX[sub].get(os.path.basename(subject_file))


def load_ec(subject_file: str, graph_source: str, fold: int = 0) -> torch.Tensor:
    """Load a subject's (C, C) EC matrix from one of the precomputed sources.

    Args:
        subject_file: the ``.mat`` basename (matched against each json's
            ``subject_file`` field).
        graph_source: one of ``GRAPH_SOURCES``.
        fold: expected fold index. Used to locate the per-fold baseline dir, and
            as the leakage-guard assertion for OC-probe sources (de_ec / mean_diff).

    Returns:
        ec: (C, C) float32 tensor. mean_diff / de_ec are signed; baselines unsigned.

    Raises:
        ValueError: unknown graph_source, or OC-probe checkpoint fold mismatch.
        FileNotFoundError: no matching json on disk.
    """
    if graph_source not in GRAPH_SOURCES:
        raise ValueError(f"graph_source {graph_source!r}; expected one of {GRAPH_SOURCES}")

    if graph_source in ("mean_diff", "signed_de"):
        # Full-sample (5-fold) OC DE-probe ablation (nosc variant). Each subject's
        # matrix lives in the fold where it was the held-out test subject (computed
        # with that fold's OC checkpoint). The ``fold`` argument is NOT used for
        # directory resolution here — we locate the subject across all 5 fold dirs.
        # Leakage note (decision A): a train subject's matrix was computed with its
        # OWN fold's checkpoint, whose training set includes other folds' subjects —
        # an indirect (model-level) leak, weaker than direct feature leakage.
        # Baselines (gc/te/gpdc) are per-subject-EEG and have no such leak.
        jp = _locate_mean_diff(subject_file)
        if jp is None:
            raise FileNotFoundError(f"no {graph_source} json for {subject_file!r} in {_MEAN_DIFF_ROOT}")
        with open(jp) as f:
            d = json.load(f)
        key = "mean_diff_matrix" if graph_source == "mean_diff" else "signed_connectivity_matrix"
        mat = np.asarray(d[key], dtype=np.float32)
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
            raise ValueError(f"{graph_source} for {subject_file!r} has bad shape {mat.shape}")
        if not np.all(np.isfinite(mat)):
            raise ValueError(f"{graph_source} for {subject_file!r} contains NaN/inf")
        return torch.from_numpy(mat)
    elif graph_source == "de_ec":
        # fold-0-only signed DE-EC (legacy, 10 subjects).
        jdir = _DE_EC_DIR
        key = "signed_connectivity_matrix"
        leak_guard = True
    else:
        # Baselines (gc/te/gpdc): per-subject-EEG, leakage-free, stored per-fold.
        jp = _locate_baseline(subject_file, graph_source)
        if jp is None:
            raise FileNotFoundError(f"no {graph_source} json for {subject_file!r}")
        with open(jp) as f:
            d = json.load(f)
        key = "connectivity_matrix"
        mat = np.asarray(d[key], dtype=np.float32)
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
            raise ValueError(f"{graph_source} for {subject_file!r} has bad shape {mat.shape}")
        if not np.all(np.isfinite(mat)):
            raise ValueError(f"{graph_source} for {subject_file!r} contains NaN/inf")
        return torch.from_numpy(mat)

    # de_ec branch (legacy fold-0-only) reaches here.
    jp = _find_json_by_subject(jdir, subject_file)
    if jp is None:
        raise FileNotFoundError(f"no {graph_source} json for {subject_file!r} in {jdir}")
    with open(jp) as f:
        d = json.load(f)

    # The OC checkpoint that produced this matrix must be the one trained on
    # this fold's train subjects (no train/test subject leakage).
    ckpt = str(d.get("checkpoint", ""))
    m = re.search(r"fold(\d+)", ckpt)
    ckpt_fold = int(m.group(1)) if m else None
    if ckpt_fold is not None and ckpt_fold != fold:
        raise ValueError(
            f"{graph_source} leakage guard: subject {subject_file!r} checkpoint "
            f"{ckpt!r} is fold {ckpt_fold}, expected fold {fold}"
        )

    mat = np.asarray(d[key], dtype=np.float32)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"EC matrix for {subject_file!r} has bad shape {mat.shape}")
    if not np.all(np.isfinite(mat)):
        raise ValueError(f"EC matrix for {subject_file!r} contains NaN/inf")
    return torch.from_numpy(mat)


def list_fold0_subject_files() -> list[str]:
    """Return the 10 canonical fold-0 ``subject_file`` basenames (from DE-EC jsons).

    DE-EC fold-0 is the source of truth for the fold-0 subject set; the baselines
    were verified to cover the same 10 subjects (overlap 10/10). Sorted by subject_idx.
    """
    out: list[tuple[int, str]] = []
    for jp in sorted(_DE_EC_DIR.glob("subject_*.json")):
        try:
            with open(jp) as f:
                d = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        sf = str(d.get("subject_file", ""))
        if sf:
            out.append((int(d.get("subject_idx", 9999)), os.path.basename(sf)))
    out.sort()
    return [sf for _, sf in out]


def get_full_sample_folds(variant: str = _MEAN_DIFF_VARIANT) -> dict[int, list[str]]:
    """Return the full 5-fold subject partition from the mean_diff ablation dir.

    Maps ``fold → [subject_file basenames]`` for the 50 subjects (26 HC + 24 MDD).
    This partition is the source of truth for full-sample 5-fold CV: each fold's
    mean_diff was computed with that fold's OC checkpoint (leakage-safe), and the
    baseline dirs ``baseline_connectivity_fold{k}`` use the same subject partition,
    so all four graph sources share identical train/test splits per fold.

    Args:
        variant: ablation variant dir ("nosc" primary; "now"/"nomexhat" near-duplicates).
    """
    folds: dict[int, list[str]] = {}
    for fk in range(5):
        jdir = _MEAN_DIFF_ROOT / f"fold{fk}" / variant
        out: list[tuple[int, str]] = []
        for jp in sorted(jdir.glob("subject_*.json")):
            try:
                with open(jp) as f:
                    d = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            sf = str(d.get("subject_file", ""))
            if sf:
                out.append((int(d.get("subject_idx", 9999)), os.path.basename(sf)))
        out.sort()
        folds[fk] = [sf for _, sf in out]
    return folds


# ---------------------------------------------------------------------------
# DE feature cache (computed once per subject from raw .mat, reused across runs)
# ---------------------------------------------------------------------------

def compute_subject_de(subject_file: str, n_windows: Optional[int] = None) -> np.ndarray:
    """Compute (C, 5) mean-DE for a subject from its raw .mat recording.

    Args:
        subject_file: ``.mat`` basename; resolved under ``modma_resting/{HC,MDD}/``.
        n_windows: if set, subsample this many windows (even spacing) before
            averaging — speeds up the cache build with negligible effect on the
            mean (windows are highly redundant at 50% overlap).
    """
    name = os.path.basename(subject_file)
    for grp in ("HC", "MDD"):
        cand = DATA_ROOT / grp / name
        if cand.exists():
            data = load_mat_file(str(cand))  # (C, T)
            windows = extract_windows(data)  # list of dicts (T,C)
            if n_windows is not None and len(windows) > n_windows:
                idx = np.linspace(0, len(windows) - 1, n_windows).round().astype(int)
                windows = [windows[i] for i in idx]
            # Each window's x_target is (T=500, C); DE expects (T, C) or (C, T).
            arrs = [w["x_target"].numpy() for w in windows]  # each (500, C)
            return aggregate_subject_de(arrs, fs=float(FS))
    raise FileNotFoundError(f"subject {name!r} not under {DATA_ROOT}/HC or /MDD")


class SubjectClassificationDataset(Dataset):
    """One sample = one subject. fold-0: 10 subjects (4 MDD + 6 HC).

    DE features are computed once per subject and cached in-memory; the EC matrix
    is loaded per graph_source. Build a fresh dataset per graph_source but share
    the DE cache across them.
    """

    def __init__(
        self,
        subject_files: list[str],
        graph_source: str,
        fold: int = 0,
        de_cache: Optional[dict[str, np.ndarray]] = None,
        n_windows_de: Optional[int] = 20,
    ):
        super().__init__()
        if graph_source not in GRAPH_SOURCES:
            raise ValueError(f"graph_source {graph_source!r}; expected one of {GRAPH_SOURCES}")
        self.subject_files = list(subject_files)
        self.graph_source = graph_source
        self.fold = fold

        # Shared DE cache: build missing entries, reuse the rest.
        if de_cache is None:
            de_cache = {}
        self.de_cache = de_cache
        for sf in self.subject_files:
            if sf not in self.de_cache:
                self.de_cache[sf] = compute_subject_de(sf, n_windows=n_windows_de)

        # Preload labels + EC so __getitem__ is cheap and errors surface early.
        self.labels: list[int] = []
        self.ec: list[torch.Tensor] = []
        for sf in self.subject_files:
            self.labels.append(resolve_label(sf))
            self.ec.append(load_ec(sf, graph_source, fold=fold))

    def __len__(self) -> int:
        return len(self.subject_files)

    def __getitem__(self, idx: int) -> dict:
        sf = self.subject_files[idx]
        return {
            "de_features": torch.from_numpy(self.de_cache[sf]).float(),  # (C, 5)
            "ec_matrix": self.ec[idx],                                   # (C, C)
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "subject_id": sf,
        }

    def label_distribution(self) -> dict[str, int]:
        from collections import Counter
        c = Counter("MDD" if l == 1 else "HC" for l in self.labels)
        return dict(c)


# ---------------------------------------------------------------------------
# Window-level dataset (one sample = one EEG window; subject-level LOO split)
# ---------------------------------------------------------------------------

def compute_subject_window_de(subject_file: str, max_windows: Optional[int] = None) -> np.ndarray:
    """Compute per-window DE for a subject → (n_windows, C, 5).

    Unlike ``compute_subject_de`` (which mean-aggregates across windows to a single
    (C,5) per subject), this keeps each window's DE separate so the window-level
    dataset has genuine inter-sample variance in the node features.

    Args:
        subject_file: ``.mat`` basename.
        max_windows: if set, cap the number of windows (even-spaced subsample) to
            bound memory/runtime. None = use all windows.
    """
    name = os.path.basename(subject_file)
    for grp in ("HC", "MDD"):
        cand = DATA_ROOT / grp / name
        if cand.exists():
            data = load_mat_file(str(cand))  # (C, T)
            windows = extract_windows(data)  # list of dicts
            if max_windows is not None and len(windows) > max_windows:
                idx = np.linspace(0, len(windows) - 1, max_windows).round().astype(int)
                windows = [windows[i] for i in idx]
            arrs = [w["x_target"].numpy() for w in windows]  # each (500, C)
            des = [compute_de_features(a, fs=float(FS)) for a in arrs]  # each (C, 5)
            return np.stack(des, axis=0)  # (n_windows, C, 5)
    raise FileNotFoundError(f"subject {name!r} not under {DATA_ROOT}/HC or /MDD")


class WindowClassificationDataset(Dataset):
    """One sample = one EEG window. Split is subject-level (LOO), not window-level.

    Each window carries:
      - ``de_features``: (C, 5) DE of THAT window (varies across windows → real signal).
      - ``ec_matrix``:   (C, C) the subject's EC graph (shared across the subject's windows).
      - ``label``:       the subject's HC/MDD label.
      - ``subject_id``:  the ``.mat`` basename (for subject-level aggregation at eval).

    Leakage control: subject-level splitting is enforced by the caller passing
    disjoint subject lists; this dataset never splits windows of one subject
    across train/test.
    """

    def __init__(
        self,
        subject_files: list[str],
        graph_source: str,
        fold: int = 0,
        window_de_cache: Optional[dict[str, np.ndarray]] = None,
        max_windows_per_subject: Optional[int] = 40,
        label_override: Optional[dict[str, int]] = None,
    ):
        super().__init__()
        if graph_source not in GRAPH_SOURCES:
            raise ValueError(f"graph_source {graph_source!r}; expected one of {GRAPH_SOURCES}")
        self.graph_source = graph_source
        self.fold = fold

        if window_de_cache is None:
            window_de_cache = {}
        self.window_de_cache = window_de_cache

        # Build a flat index: each entry → (subject_idx_in_list, window_idx_in_subject).
        # Per-subject EC + label are stored once; per-window DE is indexed.
        # label_override (subject_file → 0/1) supports the sham-label control: when
        # set, it replaces the true HC/MDD label so a baseline-accuracy sham run can
        # distinguish real signal from overfit noise.
        self.subject_files = list(subject_files)
        self.ec_per_subject: list[torch.Tensor] = []
        self.label_per_subject: list[int] = []
        self.index: list[tuple[int, int]] = []  # (subject_pos, window_pos)
        for spos, sf in enumerate(self.subject_files):
            if sf not in self.window_de_cache:
                self.window_de_cache[sf] = compute_subject_window_de(
                    sf, max_windows=max_windows_per_subject
                )
            self.ec_per_subject.append(load_ec(sf, graph_source, fold=fold))
            if label_override is not None:
                self.label_per_subject.append(int(label_override[sf]))
            else:
                self.label_per_subject.append(resolve_label(sf))
            n_win = self.window_de_cache[sf].shape[0]
            for w in range(n_win):
                self.index.append((spos, w))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        spos, wpos = self.index[idx]
        sf = self.subject_files[spos]
        return {
            "de_features": torch.from_numpy(self.window_de_cache[sf][wpos]).float(),  # (C, 5)
            "ec_matrix": self.ec_per_subject[spos],                                   # (C, C)
            "label": torch.tensor(self.label_per_subject[spos], dtype=torch.long),
            "subject_id": sf,
            "window_idx": torch.tensor(wpos, dtype=torch.long),
        }

    def label_distribution(self) -> dict[str, int]:
        from collections import Counter
        c = Counter("MDD" if l == 1 else "HC" for l in self.label_per_subject)
        return dict(c)
