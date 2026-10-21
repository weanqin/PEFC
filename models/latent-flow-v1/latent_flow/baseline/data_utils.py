"""Data loading for baseline connectivity estimation.

Loads the full continuous per-subject timeseries (NOT windowed) for the
fold test subjects. Connectivity estimators (GC, TE) need the continuous
record -- windowing would artificially truncate the AR / embedding history.

Subject selection is identical to ``scripts/run_oc_connectivity_attribution.
get_fold_test_subjects`` so the baseline matrices cover the exact same
held-out subjects as the OC-LatentFlow attribution matrices (directly
comparable). Per-channel z-scoring matches ``MODMADataset`` (mean-subtract,
divide by std, std==0 guarded).

Supports both cohorts via the ``dataset`` selector:
    - "modma": .mat files, 128 ch (HC/ / MDD/ subdirs).
    - "edra":  BrainVision .vhdr, 62 ch (flat H_* / M_* layout).
"""

import numpy as np

from latent_flow.config import FS
from latent_flow.data.datasets import get_dataset, list_hc, list_mdd
from latent_flow.data.splitter import SubjectFoldSplitter

N_CHANNELS = get_dataset("modma").channels  # MODMA default (back-compat)


def load_subject_timeseries(subject_file, normalize=True, dataset="modma"):
    """Load one subject's full continuous EEG as (n_channels, n_samples).

    Dispatches by the dataset's file_format (.mat -> scipy, .vhdr ->
    BrainVision reader). Optionally per-channel z-scored (same convention as
    MODMADataset).

    Args:
        subject_file: path to a .mat (MODMA) or .vhdr (EDRA) subject file.
        normalize: if True, per-channel z-score (mean 0, std 1 per channel).
        dataset: dataset key selecting the reader ("modma" | "edra").

    Returns:
        (n_channels, n_samples) float32 array.
    """
    desc = get_dataset(dataset)
    if desc.file_format == "mat":
        from scipy.io import loadmat
        data = loadmat(subject_file)["data"].astype(np.float32)
    elif desc.file_format == "brainvision":
        from latent_flow.data.brainvision import read_brainvision
        data = read_brainvision(subject_file, normalize=False)
    elif desc.file_format == "npz":
        # edra_fmt (npi npz contract): per-subject npz with a `data`
        # (n_channels, n_samples) float32 key. Mirrors data/modma.py's branch.
        data = np.load(subject_file, allow_pickle=True)["data"].astype(np.float32)
    else:
        raise ValueError(f"Unknown file_format {desc.file_format!r}")

    if normalize:
        mean = data.mean(axis=1, keepdims=True)
        std = data.std(axis=1, keepdims=True)
        std[std == 0] = 1.0  # avoid div by zero (mirrors MODMADataset)
        data = (data - mean) / std
    return data.astype(np.float32)


def get_fold_test_subjects(fold=0, seed=42, dataset="modma"):
    """Return the fold's test subject paths (same split as OC runner).

    Mirrors ``scripts/run_oc_connectivity_attribution.get_fold_test_subjects``
    so the baseline covers the exact same held-out subjects.

    Args:
        dataset: "modma" (.mat) or "edra" (.vhdr).
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
