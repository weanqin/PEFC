"""Shared configuration for multi-cohort EEG experiments.

Environment variables:
    MODMA_DATA_ROOT: overrides the MODMA dataset directory path.
    EDRA_DATA_ROOT:  overrides the EDRA dataset directory path.
    MODMA_DEVICE: overrides the default torch device (e.g. "cuda:0" or "cpu").
"""

import os
from pathlib import Path

import torch


# Derive project root from this file's location:
# config.py lives at project root -> PROJECT_ROOT is this dir
_PROJECT_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Dataset directory: env var takes precedence, otherwise repo-root/modma_resting
DATA_ROOT = Path(os.environ.get("MODMA_DATA_ROOT", _PROJECT_ROOT / "data" / "modma"))
GEOMETRY_PATH = DATA_ROOT / "geometry.pt"
CHECKPOINT_DIR = _PROJECT_ROOT / "checkpoints"

# EDRA resting-state EEG (62ch 10-20, BrainVision). Default points at the
# in-project copy under data/edra; override via EDRA_DATA_ROOT.
# Unlike MODMA, EDRA geometry is a generated artifact (see
# scripts/build_edra_geometry.py), so it lives alongside the data.
EDRA_DATA_ROOT = Path(os.environ.get("EDRA_DATA_ROOT", _PROJECT_ROOT / "data" / "edra"))
EDRA_GEOMETRY_PATH = EDRA_DATA_ROOT / "geometry_edra.pt"

# EDRA "fmt" preprocessing release (task 10-19-latent-flow-edra-fmt-train).
# Same 62ch 10-20 montage as EDRA but stored as per-subject npz files under
# HC/ / MDD/ subdirs (see .trellis/spec/data/edra-fmt.md). Override via the
# EDRA_FMT_DATA_ROOT env var. Geometry is a generated artifact
# (build_edra_fmt_geometry.py at the variant layer) living alongside the data.
EDRA_FMT_DATA_ROOT = Path(os.environ.get("EDRA_FMT_DATA_ROOT", Path("/root/autodl-tmp/edra_processed_fmt")))
EDRA_FMT_GEOMETRY_PATH = EDRA_FMT_DATA_ROOT / "geometry_edra_fmt.pt"

# Structural-connectivity (SC) prior datasets (DTI tractography). OSF
# (88 HC, 90 AAL) is the primary source; Zenodo (1 subj, 5 resolutions) is
# secondary. Override the SC data root via the SC_DATA_ROOT env var.
SC_DATA_ROOT = Path(os.environ.get("SC_DATA_ROOT", _PROJECT_ROOT / "data" / "sc_osf"))

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

_DEVICE_OVERRIDE = os.environ.get("MODMA_DEVICE", "")
if _DEVICE_OVERRIDE:
    DEVICE = torch.device(_DEVICE_OVERRIDE)
else:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Windowing constants
# ---------------------------------------------------------------------------

FS = 250.0
WINDOW_SEC = 2.0
STEP_RATIO = 0.5  # 50% overlap

WINDOW_SAMPLES = int(WINDOW_SEC * FS)          # 500
OBS_SEC = 1.5
OBS_SAMPLES = int(OBS_SEC * FS)                # 375
STEP_SAMPLES = int(WINDOW_SAMPLES * STEP_RATIO)  # 250
