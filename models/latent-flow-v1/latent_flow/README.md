# latent_flow

Hierarchical Python package for MODMA / EDRA resting-state EEG latent dynamics
experiments. Self-contained research project — code, datasets, checkpoints, and
results all live under one root.

> **Active model: OC-LatentFlow only** (per ADR-0004, 2026-06-29). JET,
> latent_flow (legacy), NCDE, UDE, Mamba, and signal-space flow were deleted;
> recoverable via git history.
>
> **Foundation document**: `docs/FOUNDATION.md` records the current stage
> results and key decisions — all subsequent work builds on it.

## Directory Layout

```
latent_flow/                          # ← this dir IS the importable package
├── __init__.py              # Package root; re-exports OCLatentFlow, BaseTrainer, MetricAggregator, ...
├── config.py                # Central configuration (paths, device, windowing)
├── requirements.txt         # Python dependencies
│
├── models/
│   └── oc_latent_flow.py    # OC-LatentFlow (the only model)
│
├── data/                    # loader code + raw datasets
│   ├── modma.py             # MODMADataset, load_mat_file, extract_windows, collate_fn
│   ├── splitter.py          # SubjectFoldSplitter (5-fold subject-level CV)
│   ├── modma/               # MODMA dataset (.mat + geometry.pt)
│   ├── edra/                # EDRA dataset (BrainVision)
│   ├── sc_osf/              # structural connectivity prior (OSF)
│   └── sc_zendo/            # structural connectivity prior (Zenodo)
│
├── training/
│   ├── trainer.py           # BaseTrainer + TrainStep + Callback + History
│   └── cli.py               # parse_train_args (shared CLI flags)
│
├── viz/
│   ├── visualizer.py        # Visualizer class (time-domain, PSD plots)
│   ├── metrics.py           # MetricAggregator + 6-metric eval helpers
│   ├── connectivity.py      # σ-matrix / connectivity heatmaps (attribution viz)
│   └── probe.py             # perturbation-response heatmap helpers (probe viz)
│
├── attribution/
│   └── bisect_sigma.py      # bisection (forward-only grid-sweep) σ search → connectivity
│
├── baseline/                # classical directed-connectivity baselines (Granger / TE)
│
├── scripts/                 # Executable entry points (all target OC-LatentFlow)
│
├── checkpoints/             # .pt model files (gitignored)
├── outputs/                 # .png and other artifacts (gitignored)
├── results/                 # eval JSON / CSV (gitignored; regeneratable)
├── logs/                    # training logs (gitignored)
│
├── docs/                    # FOUNDATION.md, inventory.csv, model_registry.yaml, path_mapping.csv, adr/
├── archive/                 # non-active files (do not participate in runs)
└── .trellis/                # Trellis workflow + spec
```

## Running

`latent_flow` is itself the importable package (the directory has an
`__init__.py` and the folder name is a valid Python identifier). Run modules
from the **parent** of this directory so `latent_flow` is importable:

```bash
cd D:/102-npi
python -m latent_flow.scripts.train_oc_latent_flow
```

> Note: the package directory uses an underscore (`latent_flow`), not a hyphen —
> a hyphen would make it non-importable.

## Script Status

| Script | Status | Description |
|--------|--------|-------------|
| `train_oc_latent_flow` | **active** | OC-LatentFlow two-stage trainer (`--stage {1,2}`, held-out-subject val, KL annealing) |
| `eval_oc_latent_flow` | **active** | z_obs PCA collapse check + multi-metric eval (MAE / Pearson / PSD-L1 / std_ratio / diversity / coherence) |
| `diag_oc_latent_flow` | **active** | decoder-vs-flow bottleneck ablation (ideal vs flow vs proj latent) |
| `plot_oc_latent_flow` | **active** | full-window obs+fc time-domain + fc PSD viz |
| `plot_stage2_compare` | **active** | old (proj-locked) vs corrected stage-2 training curves |
| `shape_check_oc_latent_flow` | **active** | CNN shape audit + stage loss-dict sanity check |
| `probe_oc_latent_flow` | **active** | signal-space fixed-σ perturbation probe (per-target MSE response, CV vs JET ceiling) |
| `run_oc_connectivity_attribution` | **active** | signal-space bisection σ search → directed connectivity matrix (interrupt-resumable) |

### Baseline scripts (`baseline/`)

Classical directed-connectivity baselines run via
`python -m latent_flow.baseline.<name>`:
`run_baseline_connectivity` (Granger + TE), `compare_baseline_vs_oc` (GC/TE vs
OC-LatentFlow attribution), `plot_subject_compare` (3-panel heatmap).

### Deleted in ADR-0004

All `train_*` / `eval_*` / probe / runner / ablation scripts targeting JET,
latent_flow (legacy), NCDE, UDE, Mamba, or signal-space flow were removed
(26 files). See `docs/adr/0004-archive-jet-and-legacy.md`.

## Configuration

`config.py` resolves all paths relative to the project root
(`Path(__file__).resolve().parent`) and defaults to the in-project `data/`
subdirs. Dataset roots can be overridden via env vars for non-default
locations (e.g. external data on another machine):

| Env var | Default | Purpose |
|---|---|---|
| `MODMA_DATA_ROOT` | `<root>/data/modma` | MODMA dataset directory |
| `EDRA_DATA_ROOT`  | `<root>/data/edra`  | EDRA dataset directory |
| `SC_DATA_ROOT`    | `<root>/data/sc_osf` | structural-connectivity prior |
| `MODMA_DEVICE`    | auto (cuda if available) | torch device override |

## Windows Gotcha

Do **not** use `conda run -n npi python ...` on Windows. It triggers a
`UnicodeEncodeError: 'gbk' codec` on stdout. Use the direct python path
instead, from the parent of `latent_flow/`:

```powershell
cd D:\102-npi
$env:KMP_DUPLICATE_LIB_OK="TRUE"
D:\program\anaconda\envs\npi\python.exe -m latent_flow.scripts.train_oc_latent_flow
```

## Environment

- **Conda env**: `npi` (Python 3.13, PyTorch 2.12 cu128)
- **GPU**: RTX 5070 Ti Laptop (12 GB), CUDA device `cuda:0`
- **Required packages**: `torch`, `numpy`, `scipy`, `matplotlib`, `numba`, `tqdm`
