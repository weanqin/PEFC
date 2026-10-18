"""latent_flow: Latent dynamical models for MODMA / EDRA resting-state EEG.

Self-contained research package. Run modules from the parent directory:
``python -m latent_flow.scripts.<name>``.

Active model: OC-LatentFlow (obs-conditioned latent flow). The JET,
latent_flow (legacy), NCDE, UDE, Mamba, and signal-space flow paths were
archived in ADR-0004 (recoverable via git history). The current stage of
results and key decisions are recorded in ``docs/FOUNDATION.md``.
"""

# Active model
from latent_flow.models.oc_latent_flow import OCLatentFlow

# Data
from latent_flow.data.modma import MODMADataset, collate_fn

# Training
from latent_flow.training.trainer import (
    BaseTrainer,
    TrainStep,
    Callback,
)

# Metrics (multi-metric forecast eval)
from latent_flow.viz.metrics import MetricAggregator

# Viz
from latent_flow.viz.visualizer import Visualizer

# Config
from latent_flow.config import (
    DATA_ROOT,
    DEVICE,
    CHECKPOINT_DIR,
    FS,
    OBS_SAMPLES,
    WINDOW_SAMPLES,
)

__all__ = [
    "OCLatentFlow",
    "MODMADataset",
    "collate_fn",
    "BaseTrainer",
    "TrainStep",
    "Callback",
    "MetricAggregator",
    "Visualizer",
    "DATA_ROOT",
    "DEVICE",
    "CHECKPOINT_DIR",
    "FS",
    "OBS_SAMPLES",
    "WINDOW_SAMPLES",
]
