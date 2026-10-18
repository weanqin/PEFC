"""Dataset descriptor registry for multi-cohort EEG experiments.

A single source of truth for per-dataset metadata so that consumers (data
loaders, training scripts, connectivity baselines) never hardcode channel
counts, file globs, or geometry paths. Selecting a dataset returns a frozen
``DatasetDescriptor``; everything else is derived from it.

Adding a new cohort = appending one ``DatasetDescriptor`` to ``_REGISTRY``.
"""

import glob
import os
from dataclasses import dataclass, field
from pathlib import Path

from ..config import DATA_ROOT, GEOMETRY_PATH, EDRA_DATA_ROOT, EDRA_GEOMETRY_PATH, FS


@dataclass(frozen=True)
class DatasetDescriptor:
    """Metadata + subject discovery for one EEG cohort."""

    key: str                       # "modma" | "edra"
    channels: int                  # number of EEG channels
    fs: float                      # sampling rate (Hz)
    file_format: str               # "mat" | "brainvision"
    geometry_path: Path            # per-dataset geometry .pt
    normalize_per_channel: bool    # per-channel z-score (EDRA keeps it on, like MODMA)
    data_root: Path                # resolved dataset root on disk
    # Subject discovery
    group_layout: str              # "subdir" (HC/MDD subdirs) | "prefix" (H_*/M_* flat)
    hc_glob: str                   # glob relative to data_root -> header files
    mdd_glob: str
    hc_label: str                  # display label for the healthy group
    mdd_label: str                 # display label for the patient group


_REGISTRY = {
    "modma": DatasetDescriptor(
        key="modma",
        channels=128,
        fs=FS,
        file_format="mat",
        geometry_path=GEOMETRY_PATH,
        normalize_per_channel=True,
        data_root=DATA_ROOT,
        group_layout="subdir",
        hc_glob="HC/*.mat",
        mdd_glob="MDD/*.mat",
        hc_label="HC",
        mdd_label="MDD",
    ),
    "edra": DatasetDescriptor(
        key="edra",
        channels=62,
        fs=FS,
        file_format="brainvision",
        geometry_path=EDRA_GEOMETRY_PATH,
        normalize_per_channel=True,   # decided: same as MODMA
        data_root=EDRA_DATA_ROOT,
        group_layout="prefix",
        hc_glob="H_*.vhdr",
        mdd_glob="M_*.vhdr",
        hc_label="HC",
        mdd_label="depression",
    ),
}


def get_dataset(key="modma"):
    """Return the ``DatasetDescriptor`` for *key* (default MODMA)."""
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown dataset {key!r}; available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[key]


def list_subjects(desc, group):
    """Return the sorted list of subject header paths for *group*.

    Args:
        desc: a ``DatasetDescriptor``.
        group: ``"hc"`` or ``"mdd"``.

    Returns:
        Sorted list of resolved header paths (``.mat`` for MODMA, ``.vhdr``
        for EDRA). Raises if none found.
    """
    if group == "hc":
        pattern = desc.hc_glob
    elif group == "mdd":
        pattern = desc.mdd_glob
    else:
        raise ValueError(f"group must be 'hc' or 'mdd', got {group!r}")
    files = sorted(glob.glob(os.path.join(str(desc.data_root), pattern)))
    if not files:
        raise FileNotFoundError(
            f"No {group} subjects matching {pattern!r} under {desc.data_root} "
            f"(dataset={desc.key!r}). Check the data root / env var."
        )
    return files


def list_hc(desc):
    """Sorted HC subject header paths."""
    return list_subjects(desc, "hc")


def list_mdd(desc):
    """Sorted MDD subject header paths."""
    return list_subjects(desc, "mdd")
