"""
Multi-cohort resting-state EEG dataset loader.

Despite the legacy name, this module now serves both cohorts via a
``dataset`` selector (see ``data/datasets.py``):

    - MODMA: .mat files, (128 channels, ~42501 samples), 250 Hz,
      GSN-HydroCel-129 montage. Groups in HC/ / MDD/ subdirs.
    - EDRA:  BrainVision (.vhdr+.dat), (62 channels, 60000 samples), 250 Hz,
      standard 10-20 montage. Groups are flat H_* / M_* filename prefixes.

Windowing (shared, identical contract for both cohorts):
    - 2s windows = 500 samples
    - 1.5s (375 samples) as observation for encoder
    - Full 2s (500 samples) as reconstruction target
    - 50% overlap between consecutive windows (step=250)
"""

import os
import glob
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader

from latent_flow.config import DATA_ROOT, FS, WINDOW_SEC, WINDOW_SAMPLES, OBS_SEC, OBS_SAMPLES, STEP_SAMPLES
from latent_flow.data.datasets import get_dataset


def load_mat_file(filepath):
    """Load a MODMA .mat file and return the 'data' array as float32."""
    mat = sio.loadmat(filepath)
    data = mat["data"]  # (128, n_samples)
    return data.astype(np.float32)


def _load_subject(filepath, file_format, normalize, norm_stats=None):
    """Load one subject's raw EEG as ``(n_channels, n_samples)`` float32.

    Dispatches by ``file_format`` ("mat" | "brainvision"). Normalization is
    applied in one of two mutually-exclusive modes:

    - ``norm_stats=(mean, std)`` — apply a **fixed** per-channel ``(C, 1)``
      mean/std (e.g. pooled train-only statistics from
      ``data.norm_stats.compute_train_norm_stats``). This is the strict
      held-out path used by the DE-inference experiment redesign: val/test
      subjects are normalized with training statistics, never their own.
      Takes precedence over *normalize*.
    - *normalize* (default True) — per-subject z-score (mean 0, std 1,
      std==0 guarded), the legacy convention both cohorts share.

    Passing neither (``normalize=False`` and ``norm_stats=None``) returns the
    raw recording.
    """
    if file_format == "mat":
        data = load_mat_file(filepath)
    elif file_format == "brainvision":
        from latent_flow.data.brainvision import read_brainvision
        # read_brainvision handles its own normalization; pass normalize=False
        # so we don't double-normalize, then normalize uniformly below.
        data = read_brainvision(filepath, normalize=False)
    else:
        raise ValueError(f"Unknown file_format {file_format!r}")

    if norm_stats is not None:
        mean, std = norm_stats  # each (C, 1), broadcast over (C, n_samples)
        data = (data - mean) / std
    elif normalize:
        mean = data.mean(axis=1, keepdims=True)
        std = data.std(axis=1, keepdims=True)
        std[std == 0] = 1.0  # avoid div by zero
        data = (data - mean) / std
    return data.astype(np.float32)


def extract_windows(data, window_size=WINDOW_SAMPLES, step=STEP_SAMPLES,
                    obs_size=OBS_SAMPLES):
    """
    Extract overlapping windows from a single subject's EEG data.

    Args:
        data: (n_channels, n_samples)
        window_size: total window length in samples
        step: step size between consecutive windows
        obs_size: number of observation samples within each window

    Returns:
        list of dicts, each with:
            x_obs: (obs_size, n_channels)
            x_target: (window_size, n_channels)
            t_obs: (obs_size,)
            t_target: (window_size,)
    """
    n_channels, n_samples = data.shape
    windows = []

    # Time arrays in seconds
    dt = 1.0 / FS
    t_obs = np.arange(obs_size) * dt
    t_target = np.arange(window_size) * dt

    for start in range(0, n_samples - window_size + 1, step):
        window = data[:, start:start + window_size]  # (n_ch, window_size)

        # Transpose to (time, channels)
        x_target = window.T  # (window_size, n_channels)
        x_obs = x_target[:obs_size]  # (obs_size, n_channels)

        windows.append({
            "x_obs": torch.tensor(x_obs, dtype=torch.float32),
            "x_target": torch.tensor(x_target, dtype=torch.float32),
            "t_obs": torch.tensor(t_obs, dtype=torch.float32),
            "t_target": torch.tensor(t_target, dtype=torch.float32),
        })

    return windows


class MODMADataset(Dataset):
    """
    Resting-state EEG dataset (MODMA or EDRA).

    Loads one or more subjects, per-channel z-score normalization. The file
    format and channel count are derived from the ``dataset`` descriptor
    (default MODMA); existing call sites that omit ``dataset`` keep working.
    """

    def __init__(
        self,
        subject_files=None,
        group="MDD",
        normalize_per_channel=True,
        window_size=WINDOW_SAMPLES,
        step=STEP_SAMPLES,
        obs_size=OBS_SAMPLES,
        dataset="modma",
        norm_stats=None,
    ):
        """
        Args:
            subject_files: list of subject header paths (.mat for MODMA,
                .vhdr for EDRA), or None to load all subjects in *group*.
            group: "HC" or "MDD" (display only when subject_files is None).
            normalize_per_channel: whether to z-score normalize each channel.
                Ignored if ``dataset`` descriptor overrides it — callers should
                normally leave this True (both cohorts z-score).
            dataset: dataset key ("modma" | "edra"); resolves file_format +
                channel count + default data root via ``data.datasets``.
            norm_stats: optional ``(mean, std)`` tuple, each ``(C, 1)``
                float32, of fixed normalization statistics (e.g. pooled
                train-only stats from
                ``data.norm_stats.compute_train_norm_stats``). When given,
                every subject is normalized with these stats and
                *normalize_per_channel* is ignored. This is the strict
                held-out DE-inference path (train-only stats applied to
                train/val/test identically). Default ``None`` keeps the
                legacy per-subject z-score.
        """
        super().__init__()

        desc = get_dataset(dataset)
        self.dataset_key = desc.key
        self.file_format = desc.file_format
        self.n_channels = desc.channels

        # normalize default follows the descriptor, but an explicit
        # normalize_per_channel=False from the caller still wins (back-compat).
        normalize = normalize_per_channel and desc.normalize_per_channel

        if subject_files is None:
            grp = "hc" if group.upper() == "HC" else "mdd"
            from latent_flow.data.datasets import list_subjects
            subject_files = list_subjects(desc, grp)

        if len(subject_files) == 0:
            raise ValueError(
                f"No subject files for group={group}, dataset={dataset!r} "
                f"under {desc.data_root}"
            )

        self.subject_files = subject_files
        self.group = group
        self.normalize_per_channel = normalize
        self.norm_stats = norm_stats

        # Load all subjects and extract windows
        self.windows = []
        for filepath in subject_files:
            data = _load_subject(
                filepath, desc.file_format, normalize, norm_stats=norm_stats,
            )
            win_list = extract_windows(data, window_size, step, obs_size)
            self.windows.extend(win_list)

        print(f"{dataset.upper()} {group}: {len(subject_files)} subjects, "
              f"{len(self.windows)} windows (win={window_size}, step={step})")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx]


def collate_fn(batch):
    """Collate variable-length windows (though here they're fixed)."""
    x_obs = torch.stack([b["x_obs"] for b in batch])
    x_target = torch.stack([b["x_target"] for b in batch])
    t_obs = torch.stack([b["t_obs"] for b in batch])
    t_target = torch.stack([b["t_target"] for b in batch])
    return {
        "x_obs": x_obs,
        "x_target": x_target,
        "t_obs": t_obs,
        "t_target": t_target,
    }


def get_single_subject_dataloader(
    filepath,
    batch_size=8,
    shuffle=True,
    normalize=True,
    window_size=WINDOW_SAMPLES,
    step=STEP_SAMPLES,
    obs_size=OBS_SAMPLES,
):
    """Create dataloader from a single subject file."""
    ds = MODMADataset(
        subject_files=[filepath],
        normalize_per_channel=normalize,
        window_size=window_size,
        step=step,
        obs_size=obs_size,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return loader, ds


def get_modma_dataloaders(
    group="MDD",
    train_subjects=None,
    val_subjects=None,
    test_subjects=None,
    batch_size=8,
    normalize=True,
):
    """Create train/val/test dataloaders from subject lists."""
    if train_subjects:
        train_ds = MODMADataset(train_subjects, group, normalize)
    else:
        pattern = os.path.join(DATA_ROOT, group, "*.mat")
        all_files = sorted(glob.glob(pattern))
        n = len(all_files)
        if n == 0:
            raise ValueError(f"No subjects found in {DATA_ROOT}/{group}")
        train_ds = MODMADataset(all_files[:max(1, n-2)], group, normalize)
        val_ds = MODMADataset(all_files[max(1, n-2):max(2, n-1)], group, normalize)
        test_ds = MODMADataset(all_files[max(2, n-1):], group, normalize)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        return train_loader, val_loader, test_loader

    train_ds = MODMADataset(train_subjects, group, normalize)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    loaders = [train_loader]
    for subjects in [val_subjects, test_subjects]:
        if subjects:
            ds = MODMADataset(subjects, group, normalize)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
            loaders.append(loader)
        else:
            loaders.append(None)

    return tuple(loaders)
