"""Train OCLatentFlow (task 06-23-new-latent-flow-arch) on MODMA data.

Two-stage obs-conditioned latent flow. See
``.trellis/tasks/06-23-new-latent-flow-arch/design.md``.

Stage 1 (``--stage 1``): train encoder + proj + decoder with VAE loss.
    Encoder sees only x_obs; z_fc = proj(z_obs); decoder reconstructs the
    full window. KL anneals beta from 0 to beta_kl_final.

Stage 2 (``--stage 2``): freeze encoder/proj/decoder, train flow.
    Loads the stage-1 checkpoint, freezes the VAE params, and trains the
    flow to learn z_obs -> z_fc transport with velocity MSE + end-to-end
    fc signal loss.

Validation is subject-level (held-out subjects, never random-window split),
per database-guidelines.md and the 06-15 JET methodology fix.

Run (from repo root):

    D:\\program\\anaconda\\envs\\npi\\python.exe -m \\
        latent_flow.scripts.train_oc_latent_flow \\
        --stage 1 --fold 0 --epochs 1
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")

import numpy as np
import torch
import torch.utils.data as data

from latent_flow.config import CHECKPOINT_DIR
from latent_flow.data.datasets import get_dataset, list_hc, list_mdd
from latent_flow.data.modma import MODMADataset, collate_fn
from latent_flow.data.splitter import SubjectFoldSplitter
from latent_flow.models.oc_latent_flow import OCLatentFlow
from latent_flow.training.cli import parse_train_args
from latent_flow.training.seeding import set_seed
from latent_flow.training.trainer import (
    BaseTrainer,
    Callback,
    PrintingCallback,
    TrainStep,
)

OBS_LEN = 375
FORECAST_LEN = 125
VAL_SUBJECT_COUNT = 5
LATENT_DIM = 64
T_LAT = 16


def _build_scheduler(optimizer, args):
    """Build an optional LR scheduler from CLI flags.

    Returns the scheduler (or None). Supports cosine annealing with linear
    warmup; ``none`` returns None (the historical constant-lr behavior).
    """
    mode = getattr(args, "lr_scheduler", "none")
    if mode == "none":
        return None
    epochs = args.epochs
    warmup = max(0, int(getattr(args, "lr_warmup_epochs", 0)))
    if mode == "cosine":
        import torch.optim.lr_scheduler as lrs
        if warmup > 0:
            # Cosine after warmup: warmup LinearLR (epochs 0..warmup) then
            # CosineAnnealingLR (warmup..epochs). SequentialLR switches at the
            # warmup milestone; this torch build requires milestones explicitly.
            cosine = lrs.CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup))
            warm = lrs.LinearLR(optimizer, start_factor=1.0 / max(1, warmup),
                                total_iters=warmup)
            return lrs.SequentialLR(optimizer, [warm, cosine], milestones=[warmup])
        return lrs.CosineAnnealingLR(optimizer, T_max=epochs)
    raise ValueError(f"unknown --lr-scheduler {mode!r}")


# =============================================================================
# TrainStep subclasses
# =============================================================================

class Stage1Step(TrainStep):
    """Stage 1: VAE forward. Model handles loss internally."""

    def __call__(self, batch, model, criterion, device):
        x_obs = batch["x_obs"].to(device)
        x_target = batch["x_target"].to(device)
        return model.forward_stage1(x_obs, x_target)


class Stage2Step(TrainStep):
    """Stage 2: flow forward with frozen VAE.

    Optionally adds regularizers when the flow is a LearnableMexHatMixer:
      - anti-uniformity (penalizes A_eff row-sums becoming uniform).
      - L1 sparsity on A_eff (off-diagonal).
    Weights are read from bound attributes, set externally before training.
    """

    anti_uniform_lambda = 0.0
    sparse_lambda = 0.0

    def __call__(self, batch, model, criterion, device):
        x_obs = batch["x_obs"].to(device)
        x_target = batch["x_target"].to(device)
        loss, losses = model.forward_stage2(x_obs, x_target)
        # The A_eff regularizers (anti-uniformity, sparsity) are kernel-agnostic
        # — they call effective_adjacency(), which both the ricker ("mexhat")
        # and dog kernels implement. Apply to either.
        mixer = (getattr(model, "flow_graph_kernel", "") in ("mexhat", "dog")
                 and hasattr(model.flow, "graph_mixer"))
        if mixer:
            mixer = model.flow.graph_mixer
            # Anti-uniformity regularizer.
            if self.anti_uniform_lambda > 0 and hasattr(mixer, "uniformity_penalty"):
                reg = mixer.uniformity_penalty()
                loss = loss + self.anti_uniform_lambda * reg
                losses = {**losses, "uni_reg": reg.detach()}
            # L1 sparsity on A_eff (off-diagonal).
            if self.sparse_lambda > 0 and hasattr(mixer, "sparsity_penalty"):
                reg = mixer.sparsity_penalty()
                loss = loss + self.sparse_lambda * reg
                losses = {**losses, "sp_reg": reg.detach()}
        return loss, losses


# =============================================================================
# Data loading (held-out subject val)
# =============================================================================

def _split_train_val_subjects(train_files, val_subject_count, seed):
    """Hold out val_subject_count subjects from train_files as val.

    Independent seed from the fold seed so the val pick is not correlated
    with how the fold was generated (mirrors train_jet_forecast_v2.py).
    """
    rng = np.random.RandomState(seed + 1)
    perm = rng.permutation(len(train_files))
    val_idx = perm[:val_subject_count]
    train_idx = perm[val_subject_count:]
    return [train_files[i] for i in train_idx], [train_files[i] for i in val_idx]


def _build_loaders(args, train_files, test_files):
    train_subj, val_subj = _split_train_val_subjects(
        train_files, VAL_SUBJECT_COUNT, args.seed,
    )
    print(f"Train subjects: {len(train_subj)}  Val subjects: {len(val_subj)}  "
          f"Test subjects: {len(test_files)}")

    train_ds = MODMADataset(
        subject_files=train_subj, group="MIX",
        normalize_per_channel=True, obs_size=OBS_LEN,
        dataset=args.dataset,
    )
    val_ds = MODMADataset(
        subject_files=val_subj, group="MIX",
        normalize_per_channel=True, obs_size=OBS_LEN,
        dataset=args.dataset,
    )
    test_ds = MODMADataset(
        subject_files=test_files, group="MIX",
        normalize_per_channel=True, obs_size=OBS_LEN,
        dataset=args.dataset,
    )
    train_loader = data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader, test_loader


# =============================================================================
# Model + trainer
# =============================================================================

def _build_model(device, n_channels, dataset="modma", graph_mode=False, graph_k=2,
                 graph_out_mix_alpha=0.5, graph_kernel="gaussian", mexhat_gamma=0.1,
                 w_init=None, mexhat_gamma_m=1.0,
                 dog_init=None, beta_kl_final=1e-4, use_fc_decoder=False):
    a_hat = None
    coords = None
    if graph_mode:
        from latent_flow.models.graph_utils import build_normalized_adj, load_geometry
        coords_all, _ = load_geometry(dataset=dataset)
        a_hat = build_normalized_adj(coords_all).to(device)
        if graph_kernel in ("mexhat", "dog"):
            # Both band-pass kernels use the pairwise-distance matrix, so both
            # need coords. (dog decouples excitation/inhibition widths.)
            coords = coords_all.to(device)
        print(f"Graph-mode flow: kernel={graph_kernel}, K={graph_k} layers, "
              f"out mix_alpha={graph_out_mix_alpha}, mexhat_gamma={mexhat_gamma}, "
              f"adjacency {tuple(a_hat.shape)} (dataset={dataset})")
    model = OCLatentFlow(
        n_channels=n_channels, latent_dim=LATENT_DIM, t_lat=T_LAT,
        obs_len=OBS_LEN, forecast_len=FORECAST_LEN,
        flow_graph_mode=graph_mode, flow_graph_k=graph_k, flow_a_hat=a_hat,
        flow_graph_out_mix_alpha=graph_out_mix_alpha,
        flow_graph_kernel=graph_kernel, flow_coords=coords,
        flow_mexhat_gamma=mexhat_gamma, flow_w_init=w_init,
        flow_mexhat_gamma_m=mexhat_gamma_m,
        flow_dog_init=dog_init,
        beta_kl_final=beta_kl_final,
        use_fc_decoder=use_fc_decoder,
    ).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,} "
          f"(flow={'graph-'+graph_kernel if graph_mode else 'dense'})")
    return model


class _KLAnnealCallback(Callback):
    """Anneal model._current_beta from 0 to beta_kl_final over warmup epochs.

    BaseTrainer.kl_anneal_epochs only works for criterion.beta_kl; our loss is
    model-internal, so we drive the beta via model.set_kl_beta each epoch.
    """

    def __init__(self, model, beta_final, warmup_epochs):
        self.model = model
        self.beta_final = beta_final
        self.warmup_epochs = max(1, warmup_epochs)

    def on_epoch_begin(self, trainer, epoch):
        if epoch < self.warmup_epochs:
            beta = self.beta_final * (epoch / self.warmup_epochs)
        else:
            beta = self.beta_final
        self.model.set_kl_beta(beta)


def _train_stage(model, train_loader, val_loader, args, save_path, stage, step):
    """Run BaseTrainer for the given stage."""
    warmup = max(1, int(0.2 * args.epochs))  # 20% of epochs (design.md Q10)
    callbacks = [PrintingCallback(prefix=f"[stage{stage}] ")]
    # KL annealing: stage-1 always (VAE training); stage-2 only under joint
    # training (--unfreeze-vae), where the unfrozen posterior needs regularizing.
    # Under frozen stage-2, beta stays 0 (the __init__ default), so the KL term
    # added in forward_stage2 contributes 0 and the loss matches the historical
    # frozen-VAE behavior exactly.
    if stage == 1 or (stage == 2 and getattr(args, "unfreeze_vae", False)):
        callbacks.append(_KLAnnealCallback(model, model.beta_kl_final, warmup))

    trainer = BaseTrainer(
        model=model, criterion=None, device=next(model.parameters()).device,
        train_step=step, lr=args.lr, patience=args.patience,
        save_path=save_path, use_amp=True, max_grad_norm=1.0,
        callbacks=callbacks,
    )
    # Build the LR scheduler against the trainer's own optimizer (so the
    # scheduler and optimizer share the same param groups).
    if getattr(args, "lr_scheduler", "none") != "none":
        trainer.scheduler = _build_scheduler(trainer.optimizer, args)
    history = trainer.fit(train_loader, val_loader, n_epochs=args.epochs)
    if history.best_epoch is not None:
        print(f"\nBest val loss: {history.val[history.best_epoch]['loss']:.4f}")
    if args.patience > 0:
        print(f"Checkpoint saved to {save_path}")
    return history


# =============================================================================
# Stage runners
# =============================================================================

def _run_stage1(model, train_loader, val_loader, args, save_path):
    print("\n=== Stage 1: VAE (encoder + proj + decoder) ===")
    return _train_stage(model, train_loader, val_loader, args, save_path,
                        stage=1, step=Stage1Step())


def _run_stage2(model, train_loader, val_loader, args, save_path, vae_ckpt):
    print("\n=== Stage 2: Flow (freeze VAE, train flow) ===")
    # Load stage-1 checkpoint. The VAE checkpoint contains the DENSE flow's
    # weights too, but for the graph-mode beta variant the flow architecture
    # differs (graph_mixer/graph_out vs input_proj/out_proj), so we load with
    # strict=False to take encoder/proj/decoder only and leave the flow fresh.
    payload = torch.load(
        vae_ckpt, map_location=next(model.parameters()).device,
        weights_only=False)
    # Unwrap self-describing format (format_version=1): {state_dict, variant, ...}
    # — mirrors _connectivity_driver.load_model's unwrap.
    if isinstance(payload, dict) and payload.get("format_version") == 1:
        vae_state = payload["state_dict"]
    else:
        vae_state = payload
    missing, unexpected = model.load_state_dict(vae_state, strict=False)
    # Report only non-flow mismatches; flow-key mismatches are expected when
    # the beta variant uses a different flow architecture.
    flow_prefixes = ("flow.input_proj", "flow.out_proj",
                     "flow.graph_mixer", "flow.graph_out", "flow.blocks")
    missing_real = [k for k in missing if not k.startswith(flow_prefixes)]
    unexpected_real = [k for k in unexpected if not k.startswith(flow_prefixes)]
    if missing_real or unexpected_real:
        raise RuntimeError(
            f"VAE checkpoint mismatch (non-flow keys): "
            f"missing={missing_real}, unexpected={unexpected_real}")
    if missing or unexpected:
        print(f"  (flow keys re-initialized fresh: "
              f"{len([k for k in missing if k.startswith(flow_prefixes)])} "
              f"missing, {len([k for k in unexpected if k.startswith(flow_prefixes)])} "
              f"unexpected — expected for {'graph' if getattr(model, 'flow_graph_mode', False) else 'dense'} flow)")
    joint = bool(getattr(args, "unfreeze_vae", False))
    if not joint:
        for p in model.encoder.parameters():
            p.requires_grad_(False)
        for p in model.proj.parameters():
            p.requires_grad_(False)
        for p in model.decoder.parameters():
            p.requires_grad_(False)
        # fc-only paradigm: decoder_fc is the active decoder; freeze it too.
        if getattr(model, "decoder_fc", None) is not None:
            for p in model.decoder_fc.parameters():
                p.requires_grad_(False)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params ({'joint' if joint else 'flow only'}): {n_trainable:,}")

    # Optimizer covers whichever params are trainable (flow-only when frozen,
    # all params under joint training).
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr)

    callbacks = [PrintingCallback(prefix="[stage2] ")]
    step = Stage2Step()
    step.anti_uniform_lambda = float(getattr(args, "anti_uniform_lambda", 0.0))
    if step.anti_uniform_lambda > 0:
        print(f"  anti-uniformity regularizer: lambda={step.anti_uniform_lambda}")
    step.sparse_lambda = float(getattr(args, "sparse_lambda", 0.0))
    if step.sparse_lambda > 0:
        print(f"  L1 sparsity regularizer: lambda={step.sparse_lambda} "
              f"(on A_eff off-diag)")
    trainer = BaseTrainer(
        model=model, criterion=None, device=next(model.parameters()).device,
        train_step=step, lr=args.lr, patience=args.patience,
        save_path=save_path, use_amp=True, max_grad_norm=1.0,
        optimizer=optimizer, callbacks=callbacks,
        scheduler=_build_scheduler(optimizer, args),
    )
    history = trainer.fit(train_loader, val_loader, n_epochs=args.epochs)
    if history.best_epoch is not None:
        print(f"\nBest val loss: {history.val[history.best_epoch]['loss']:.4f}")
    if args.patience > 0:
        print(f"Checkpoint saved to {save_path}")

    # Persist the full train/val history to JSON (the ablation pipeline consumes
    # per-variant loss curves). Saved beside logs; named from the checkpoint stem
    # so variants are distinguishable. The directory is dataset-namespaced
    # (logs/<dataset>/) to mirror the checkpoint layout so an EDRA run never
    # overwrites a MODMA history of the same stem.
    if args.patience > 0:
        try:
            from pathlib import Path as _P
            dataset = getattr(args, "dataset", "modma")
            # logs/<dataset>/ beside the repo root (CHECKPOINT_DIR = <root>/checkpoints)
            hist_dir = CHECKPOINT_DIR.parent / "logs" / dataset
            hist_dir.mkdir(parents=True, exist_ok=True)
            save_name = _P(save_path).stem
            hist_path = hist_dir / f"{save_name}_history.json"
            history.to_json(hist_path)
            print(f"Training history saved to {hist_path}")
        except Exception as exc:  # never let logging break training
            print(f"[warn] history save failed: {exc}")
    return history


# =============================================================================
# Smoke eval (full multi-metric eval is a separate script per implement.md 3.x)
# =============================================================================

def _smoke_eval(model, test_loader, device, n_batches=3):
    """Quick MAE + std_ratio sanity check on a few batches (full eval later)."""
    model.eval()
    mae_sum, std_sum, n = 0.0, 0.0, 0
    seen_batches = 0
    with torch.no_grad():
        for batch in test_loader:
            x_obs = batch["x_obs"].to(device)
            x_target = batch["x_target"].to(device)
            x_hat = model.sample(x_obs, n_steps=10)
            x_fc = x_target[:, OBS_LEN:, :]
            # fc-only decoder outputs fc directly (no obs slice); full-window
            # decoder outputs the whole window (slice the fc tail).
            x_fc_hat = x_hat if model.use_fc_decoder else x_hat[:, OBS_LEN:, :]
            mae_sum += (x_fc_hat - x_fc).abs().mean().item() * x_obs.shape[0]
            std_sum += (x_fc_hat.std() / (x_fc.std() + 1e-8)).item() * x_obs.shape[0]
            n += x_obs.shape[0]
            seen_batches += 1
            if seen_batches >= n_batches:
                break  # a few batches is enough for smoke
    print(f"[smoke] test MAE={mae_sum/n:.4f}  std_ratio={std_sum/n:.4f}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = parse_train_args("Train OCLatentFlow (two-stage)")
    parser.add_argument("--beta-kl", type=float, default=1e-4,
                        help="Stage-1 KL annealing final beta (default 1e-4). "
                             "Raised to 1e-3 in the KL-constraint ablation (task "
                             "08-01) after fold1-4 logs showed raw KL diverging "
                             "to ~90 with beta=1e-4 contributing only ~0.009 to loss.")
    parser.add_argument("--unfreeze-vae", action="store_true",
                        help="Stage-2 joint training: unfreeze encoder/proj/decoder "
                             "so they train alongside the flow (default: frozen, "
                             "flow-only). Use with a beta-kl > 0 to regularize the "
                             "posterior against drift during joint optimization.")
    parser.add_argument("--fc-decoder", action="store_true",
                        help="Pure-forecast paradigm (task 08-01): decoder generates "
                             "ONLY the fc segment (125) from z_fc alone — no z_obs "
                             "concatenation, no U-Net skip, no obs reconstruction. "
                             "Forces the decoder to consume z_fc (the full-window "
                             "decoder structurally drowns it). Default: off (full-"
                             "window ConvDecoder with skip).")
    parser.add_argument("--dataset", type=str, default="modma",
                        choices=("modma", "edra"),
                        help="EEG cohort: 'modma' (128ch GSN, .mat) or "
                             "'edra' (62ch 10-20, BrainVision). Default modma.")
    parser.add_argument("--lr-scheduler", type=str, default="none",
                        choices=("none", "cosine"),
                        help="LR scheduler: 'none' (constant, default) or "
                             "'cosine' (CosineAnnealingLR, optional warmup).")
    parser.add_argument("--lr-warmup-epochs", type=int, default=0,
                        help="Linear LR warmup epochs before cosine annealing "
                             "(0 = no warmup). Ignored unless --lr-scheduler cosine.")
    parser.add_argument("--stage", type=int, default=1, choices=(1, 2),
                        help="Training stage: 1=VAE, 2=flow")
    parser.add_argument("--fold", type=int, default=0, choices=range(5))
    parser.add_argument("--vae-ckpt", type=str, default=None,
                        help="Stage-2 only: path to stage-1 VAE checkpoint")
    parser.add_argument("--graph-mode", action="store_true",
                        help="Stage-2 only: use the R2 graph-mixer flow "
                             "(geometry-conditioned channel mixing). The dense "
                             "flow is the default.")
    parser.add_argument("--graph-k", type=int, default=2,
                        help="Graph-mixer message-passing layers (default 2).")
    parser.add_argument("--graph-out-mix-alpha", type=float, default=0.5,
                        help="GraphOutProj neighbour-mix gate (0=pure per-channel "
                             "diagonal-dominant, 1=replace self with neighbours). "
                             "Default 0.5 blends self + neighbours to break diagonal "
                             "dominance while keeping geometry-structured mixing.")
    parser.add_argument("--graph-kernel", type=str, default="gaussian",
                        choices=("gaussian", "mexhat", "dog"),
                        help="Graph-mixer kernel: 'gaussian' (low-pass, B.1-B.4), "
                             "'mexhat' (band-pass Mexican hat / Ricker, B.5, 1-param "
                             "sigma), or 'dog' (Difference-of-Gaussians, 4 learnable "
                             "params Ae/Ai/sigma_e/sigma_i that decouple excitation "
                             "from inhibition width, task 07-30). 'mexhat' and 'dog' "
                             "both have center-surround lateral inhibition and do NOT "
                             "isotropize like the Gaussian.")
    parser.add_argument("--mexhat-gamma", type=float, default=0.1,
                        help="Learnable residual weight gamma for the Mexican-hat "
                             "mixer (A_eff = mexhat + gamma*W). Default 0.1.")
    parser.add_argument("--mexhat-gamma-m", type=float, default=1.0,
                        help="Scale for the Mexican-hat kernel M in A_eff "
                             "(A_eff = gamma_m*M + ...). Default 1.0 = baseline. "
                             "Set 0 to ablate the band-pass prior (task 07-15).")
    parser.add_argument("--dog-sigma-e", type=float, default=None,
                        help="DoG init: excitation width sigma_e. None = "
                             "median_d/2 (default). Requires --graph-kernel dog.")
    parser.add_argument("--dog-sigma-i", type=float, default=None,
                        help="DoG init: inhibition width sigma_i (>sigma_e). "
                             "None = median_d (default). Requires --graph-kernel dog.")
    parser.add_argument("--dog-ae", type=float, default=None,
                        help="DoG init: excitation amplitude Ae (>Ai). "
                             "None = 1.0 (default). Requires --graph-kernel dog.")
    parser.add_argument("--dog-ai", type=float, default=None,
                        help="DoG init: inhibition amplitude Ai (<Ae). "
                             "None = 0.6 (default). Requires --graph-kernel dog.")
    parser.add_argument("--anti-uniform-lambda", type=float, default=0.0,
                        help="Anti-uniformity regularizer weight (penalizes A_eff "
                             "row-sums becoming uniform = isotropization). 0=off. "
                             "Try 0.01-0.1 if offdiag CV is low.")
    parser.add_argument("--sparse-lambda", type=float, default=0.0,
                        help="L1 sparsity regularizer weight on A_eff (off-diag). "
                             "Penalizes mean(|A_eff_ij|) for i!=j, encouraging "
                             "sparse connectivity. 0=off. Try 0.001-0.1.")
    args = parser.parse_args()

    device = torch.device(args.device if args.device
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}  dataset: {args.dataset}")
    set_seed(args.seed)

    # Smoke mode: disable checkpoint saving (patience=0 → no early-stop save)
    # so a quick --epochs 1 run does not clobber a real checkpoint.
    if getattr(args, "smoke", False):
        args.patience = 0
        print("SMOKE mode: checkpoints will NOT be saved (patience=0).")

    # Resolve cohort descriptor once; derive channels + subject file lists.
    desc = get_dataset(args.dataset)
    n_channels = desc.channels
    files_mdd = list_mdd(desc)
    files_hc = list_hc(desc)
    splitter = SubjectFoldSplitter(files_mdd, files_hc, n_folds=5, seed=args.seed)
    train_files, test_files = splitter.get_fold(args.fold)
    print(f"Fold {args.fold}: {len(train_files)} train / {len(test_files)} test subjects "
          f"({desc.key}: {len(files_mdd)} {desc.mdd_label} + {len(files_hc)} {desc.hc_label})")

    train_loader, val_loader, test_loader = _build_loaders(args, train_files, test_files)

    w_init = None

    # DoG init overrides (--graph-kernel dog). None values fall back to the
    # median_d-derived defaults inside the mixer (sigma_e=median_d/2, etc.).
    dog_init = None
    if args.graph_kernel == "dog":
        if not args.graph_mode:
            raise ValueError("--graph-kernel dog requires --graph-mode")
        dog_init = {k: v for k, v in (
            ("sigma_e", args.dog_sigma_e), ("sigma_i", args.dog_sigma_i),
            ("Ae", args.dog_ae), ("Ai", args.dog_ai)) if v is not None}
        if dog_init:
            print(f"DoG init overrides: {dog_init}")

    model = _build_model(device, n_channels, dataset=args.dataset,
                         graph_mode=args.graph_mode, graph_k=args.graph_k,
                         graph_out_mix_alpha=args.graph_out_mix_alpha,
                         graph_kernel=args.graph_kernel,
                         mexhat_gamma=args.mexhat_gamma, w_init=w_init,
                         mexhat_gamma_m=args.mexhat_gamma_m,
                         dog_init=dog_init, beta_kl_final=args.beta_kl,
                         use_fc_decoder=args.fc_decoder)

    suffix = "_graph" if args.graph_mode else ""
    if args.graph_mode and args.graph_kernel == "mexhat":
        suffix = "_graph_mexhat"
    if args.graph_mode and args.graph_kernel == "dog":
        # Namespaced so DoG checkpoints never clobber mexhat ones.
        suffix = "_graph_dog"
    # EDRA checkpoints go to a namespaced subdir so they never clobber MODMA.
    # MODMA keeps its original flat layout (checkpoints/fold*_*.pt) for back-compat
    # with existing stage-1 checkpoints and downstream eval scripts.
    if args.dataset == "edra":
        ckpt_dir = CHECKPOINT_DIR / "edra"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    else:
        ckpt_dir = CHECKPOINT_DIR
    vae_path = str(ckpt_dir / f"fold{args.fold}_oc_latent_flow_vae_best.pt")
    flow_path = str(ckpt_dir / f"fold{args.fold}_oc_latent_flow_flow_best{suffix}.pt")
    # --save-path overrides the suffix-derived default (used for the directed-only
    # run in task 07-02-sc-init-w, which must NOT overwrite the symmetric baseline).
    # Also applied to stage-1 vae_path so KL-ablation runs can save to a new name
    # without clobbering the canonical fold{N}_oc_latent_flow_vae_best.pt.
    if getattr(args, "save_path", None):
        flow_path = args.save_path
        vae_path = args.save_path

    if args.stage == 1:
        _run_stage1(model, train_loader, val_loader, args, vae_path)
    else:
        ckpt = args.vae_ckpt or vae_path
        if not os.path.exists(ckpt):
            raise FileNotFoundError(
                f"Stage 2 needs a stage-1 VAE checkpoint: {ckpt} not found. "
                f"Run --stage 1 first."
            )
        _run_stage2(model, train_loader, val_loader, args, flow_path, ckpt)

    # Smoke eval on test
    print("\n--- Smoke eval on test (a few batches) ---")
    _smoke_eval(model, test_loader, device)


if __name__ == "__main__":
    main()
