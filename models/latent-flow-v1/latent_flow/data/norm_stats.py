"""Train-only EEG normalization statistics.

Computes pooled per-channel mean/std over a set of **training** subjects' raw
recordings and (optionally) caches the result to disk. The DE-inference
experiment redesign (``docs/OC_LatentFlow_DE_confirmed_decisions_summary.md``
§2.2.A) requires normalization statistics to be estimated from training data
only and then frozen for validation/test — replacing the legacy per-subject
z-score (``data/modma.py::_load_subject``), which is a form of test-set
adaptation the strict held-out DE protocol forbids.

The stats are pooled (sample-weighted) across train subjects, not averaged
per-subject-then-averaged: a longer recording contributes in proportion to
its sample count. The result is a single ``(C, 1)`` mean and ``(C, 1)`` std
that the dataset applies identically to every subject in train/val/test.

This module is purely additive: existing callers that omit ``norm_stats``
keep the per-subject z-score behaviour unchanged.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np

PathLike = Union[str, Path]


def _train_files_hash(train_files: Sequence[PathLike]) -> str:
    """Stable sha1 over the sorted, normalized train-file list.

    Sorting makes the hash order-independent; resolving to absolute + forward
    slashes makes it robust to how callers constructed the paths.
    """
    norm = [str(Path(f).resolve()).replace("\\", "/") for f in train_files]
    joined = "\n".join(sorted(norm))
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()


def compute_train_norm_stats(
    train_files: Iterable[PathLike],
    file_format: str,
    cache_path: Optional[PathLike] = None,
    channels: Optional[int] = None,
    overwrite: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pooled train-only per-channel ``(mean, std)`` for EEG normalization.

    Loads each train subject's raw ``(C, n_samples)`` recording via
    ``latent_flow.data.modma._load_subject(..., normalize=False)`` and
    accumulates sample-weighted per-channel ``sum``, ``sumsq``, ``count``.
    Returns pooled ``mean = sum / count`` and
    ``std = sqrt(max(sumsq / count - mean**2, 0))`` (std==0 guarded to 1.0),
    each ``(C, 1)`` float32.

    Parameters
    ----------
    train_files
        Paths to training-subject raw recordings (e.g. ``.mat`` for MODMA,
        ``.vhdr`` for EDRA).
    file_format
        ``"mat"`` or ``"brainvision"`` — dispatched by ``_load_subject``.
    cache_path
        If given, read/write an npz cache here. The cache stores the
        ``files_hash`` (sha1 of the sorted train list) so a stale cache
        (different train set) is detected and ignored.
    channels
        Channel count ``C``. Inferred from the first subject if ``None``.
    overwrite
        If True, recompute and overwrite the cache even when a matching one
        exists.

    Returns
    -------
    (mean, std) : tuple of np.ndarray
        Each shape ``(C, 1)``, dtype float32.
    """
    train_files = list(train_files)
    if not train_files:
        raise ValueError("compute_train_norm_stats: train_files is empty")

    files_hash = _train_files_hash(train_files)

    # --- cache read --------------------------------------------------------
    if cache_path is not None and not overwrite:
        cache = Path(cache_path)
        if cache.exists():
            try:
                with np.load(cache) as npz:
                    if str(npz["files_hash"]) == files_hash:
                        return npz["mean"].astype(np.float32), npz["std"].astype(np.float32)
            except Exception:
                # Corrupt / partial cache — fall through to recompute.
                pass

    # --- pooled accumulate -------------------------------------------------
    # Late import avoids a circular ref at module import time.
    from latent_flow.data.modma import _load_subject

    if channels is None:
        first = _load_subject(train_files[0], file_format, normalize=False)
        channels = first.shape[0]

    sum_ = np.zeros((channels,), dtype=np.float64)
    sumsq = np.zeros((channels,), dtype=np.float64)
    count = np.zeros((channels,), dtype=np.float64)
    seen = 0
    for f in train_files:
        raw = _load_subject(f, file_format, normalize=False)  # (C, n_samples)
        if raw.shape[0] != channels:
            raise ValueError(
                f"compute_train_norm_stats: subject {f} has {raw.shape[0]} "
                f"channels, expected {channels}"
            )
        sum_ += raw.sum(axis=1)
        sumsq += (raw.astype(np.float64) ** 2).sum(axis=1)
        count += raw.shape[1]
        seen += 1

    if np.any(count == 0):
        raise RuntimeError("compute_train_norm_stats: zero samples accumulated")

    mean = sum_ / count
    var = sumsq / count - mean ** 2
    var = np.clip(var, a_min=0.0, a_max=None)  # guard tiny negative from rounding
    std = np.sqrt(var)
    std[std == 0] = 1.0

    mean = mean.reshape(-1, 1).astype(np.float32)
    std = std.reshape(-1, 1).astype(np.float32)

    # --- cache write -------------------------------------------------------
    if cache_path is not None:
        cache = Path(cache_path)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, mean=mean, std=std, files_hash=np.array(files_hash))

    return mean, std
