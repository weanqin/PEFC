#!/usr/bin/env python3
"""train_modma_symw.py — MODMA resting fold0 × symw-exit-GNN 变体 stage-2 训练（包装层）.

任务 10-28-modma-symw-f0。把 mild 线 10-25-latent-flow-exit-gnn-symw 变体
（run_symw_fold0 口径）移植到 MODMA resting fold0：FlowVelocityNetExitGNN 出口流
+ SymWAdam W 逐步对称化（W←(W+Wᵀ)/2）+ decoder 联训（encoder/proj 冻结）+
anti-uniform λ=0.2；lr 1e-3 / 150 ep / batch 64 / patience 30 / seed 42
（= mild symw fold0 configs/stage2_config.json 逐字段对齐）。

- 数据 = 包内 MODMA 管线（500 点窗 / step 250 / obs 375 / fc 125；逐被试逐通道
  z-score；SubjectFoldSplitter 5 折 seed 42；train 折内 5 例 held-out 为 val——
  旧库 train_oc_latent_flow 同款，_build_loaders 零复制）；
- stage-1 VAE = 旧库原位 fold0_oc_latent_flow_vae_best.pt（同 375/125 口径、同架构
  类，只读；flow 键重初始化白名单含 flow.exit_——mild 本地 runner 同款）；
- stage-2 runner = 复用 train_mild_depress._run_stage2_mild（loader 无关零复制）。

Run:
  OMP_NUM_THREADS=8 MODMA_DATA_ROOT=/root/autodl-tmp/modma python3 \
    models/latent-flow-v1/train_modma_symw.py \
    --run-dir /root/npi/outputs/10-28-modma-symw-f0/run_01
"""
from __future__ import annotations

import os

# --- BLAS 守卫：必须在任何 numpy/scipy/torch 导入之前（npi 环境约束） ---
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

VARIANT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(VARIANT_DIR))

from latent_flow.data.datasets import get_dataset, list_hc, list_mdd  # noqa: E402
from latent_flow.data.splitter import SubjectFoldSplitter  # noqa: E402
from latent_flow.models.graph_utils import (  # noqa: E402
    build_normalized_adj,
    load_geometry,
)
from latent_flow.models.oc_latent_flow import OCLatentFlow  # noqa: E402
from latent_flow.scripts.train_oc_latent_flow import (  # noqa: E402
    _build_loaders,
    _smoke_eval,
)
from latent_flow.training.seeding import set_seed  # noqa: E402
from flow_exit_gnn import FlowVelocityNetExitGNN  # noqa: E402
from train_mild_depress import _run_stage2_mild  # noqa: E402

N_CHANNELS = 128
LATENT_DIM = 64
T_LAT = 16
OBS_LEN = 375
FORECAST_LEN = 125
VAE_CKPT_DEFAULT = ("/root/autodl-tmp/latent_ode/checkpoints/"
                    "fold0_oc_latent_flow_vae_best.pt")


def _sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(VARIANT_DIR.parents[1]), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unavailable"


def build_model_modma_symw(device):
    """graph-mexhat OCLatentFlow（MODMA geometry）+ 出口 GNN 流替换（mild 同款）。

    EC driver（ec_modma_symw_v4.py）复用本函数装载 ckpt——训练与推理同一构建路径。
    """
    coords_all, _ = load_geometry(dataset="modma")
    assert coords_all.shape == (N_CHANNELS, 3), coords_all.shape
    a_hat = build_normalized_adj(coords_all).to(device)
    coords = coords_all.to(device)
    model = OCLatentFlow(
        n_channels=N_CHANNELS, latent_dim=LATENT_DIM, t_lat=T_LAT,
        obs_len=OBS_LEN, forecast_len=FORECAST_LEN,
        flow_graph_mode=True, flow_graph_k=2, flow_a_hat=a_hat,
        flow_graph_out_mix_alpha=0.5, flow_graph_kernel="mexhat",
        flow_coords=coords, flow_mexhat_gamma=0.1, flow_w_init=None,
        flow_mexhat_gamma_m=1.0, flow_dog_init=None,
        beta_kl_final=1e-4, use_fc_decoder=False,
    ).to(device)
    model.flow = FlowVelocityNetExitGNN(
        n_channels=N_CHANNELS, latent_dim=LATENT_DIM, t_lat=T_LAT,
        hidden_ch=model.flow.hidden_ch,
        graph_mode=True, graph_k=2, a_hat=a_hat,
        graph_out_mix_alpha=0.5, graph_kernel="mexhat", coords=coords,
        mexhat_gamma=0.1, w_init=None,
        mexhat_gamma_m=1.0, dog_init=None).to(device)
    print(f"Graph-mode flow: kernel=mexhat, K=2, mix_alpha=0.5, gamma=0.1, "
          f"adjacency {tuple(a_hat.shape)} (dataset=modma) | flow replaced by "
          f"FlowVelocityNetExitGNN (exit msg rounds + zero-init exit_out)")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,} "
          f"(stage2 flow=graph-mexhat-exit-gnn-symw)")
    return model


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--fold", type=int, default=0, choices=range(5))
    p.add_argument("--run-dir", type=str, required=True,
                   help="输出目录（必填，无默认——防误写旧任务树）")
    p.add_argument("--vae-ckpt", type=str, default=VAE_CKPT_DEFAULT,
                   help="stage-1 VAE ckpt（默认旧库原位 fold0）")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--beta-kl", type=float, default=1e-4)
    p.add_argument("--anti-uniform-lambda", type=float, default=0.2,
                   help="mild symw fold0 口径 λ=0.2")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--limit-subjects", type=int, default=None,
                   help="冒烟截断：train 折截前 N 例")
    p.add_argument("--save-path", type=str, default=None,
                   help="覆盖 ckpt 落点（冒烟用，防覆盖正式 ckpt）")
    return p.parse_args(argv)


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    (run_dir / "configs").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}  dataset: modma  fold: {args.fold}  "
          f"stage: 2 (symw exit-gnn variant, decoder joint)")
    set_seed(args.seed)

    desc = get_dataset("modma")
    files_mdd, files_hc = list_mdd(desc), list_hc(desc)
    splitter = SubjectFoldSplitter(files_mdd, files_hc, n_folds=5, seed=args.seed)
    train_files, test_files = splitter.get_fold(args.fold)
    print(f"Fold {args.fold}: {len(train_files)} train / {len(test_files)} test "
          f"subjects ({desc.key}: {len(files_mdd)} MDD + {len(files_hc)} HC)")
    if args.limit_subjects:
        train_files = train_files[: args.limit_subjects]
        print(f"[SMOKE] --limit-subjects {args.limit_subjects}: "
              f"train truncated to {len(train_files)}")

    loader_args = argparse.Namespace(dataset="modma", batch_size=args.batch_size,
                                     seed=args.seed)
    train_loader, val_loader, test_loader = _build_loaders(
        loader_args, train_files, test_files)

    model = build_model_modma_symw(device)

    train_args = argparse.Namespace(
        dataset="modma", unfreeze_vae=False,
        anti_uniform_lambda=args.anti_uniform_lambda, sparse_lambda=0.0,
        sym_w=True, lr_scheduler="none", lr_warmup_epochs=0,
        epochs=args.epochs, lr=args.lr, patience=args.patience, seed=args.seed,
    )

    flow_path = args.save_path or str(
        run_dir / "checkpoints" /
        f"fold{args.fold}_modma_symw_exitgnn_graph_mexhat.pt")

    n_total = sum(p.numel() for p in model.parameters())
    n_enc = sum(p.numel() for p in model.encoder.parameters())
    n_proj = sum(p.numel() for p in model.proj.parameters())
    cfg = {
        "task": "10-28-modma-symw-f0 (mild 10-25-symw-gnn-exit variant ported "
                "to MODMA resting fold0)",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_head": _git_head(),
        "cli": vars(args),
        "pinned_trainer_args": vars(train_args),
        "data": {"dataset": "modma", "fold": args.fold,
                 "windows": {"window": 500, "step": 250, "obs": OBS_LEN,
                             "fc": FORECAST_LEN},
                 "normalization": "per-subject per-channel z-score "
                                  "(MODMADataset legacy convention)",
                 "fold_counts": {"train": len(train_files), "test": len(test_files),
                                 "val_heldout": 5}},
        "model": {"n_channels": N_CHANNELS, "latent_dim": LATENT_DIM,
                  "t_lat": T_LAT, "obs_len": OBS_LEN,
                  "forecast_len": FORECAST_LEN,
                  "flow_variant": "graph-mexhat-exit-gnn-symw",
                  "beta_kl_final": args.beta_kl, "use_fc_decoder": False},
        "stage2": {"runner": "train_mild_depress._run_stage2_mild "
                             "(decoder+flow joint, encoder/proj frozen)",
                   "unfreeze_decoder": True,
                   "frozen_modules": ["encoder", "proj"],
                   "anti_uniform_lambda": args.anti_uniform_lambda,
                   "sym_w": True,
                   "trainable_params": n_total - n_enc - n_proj},
        "vae_ckpt": {"path": args.vae_ckpt, "sha256": _sha256_file(args.vae_ckpt)},
        "checkpoints": {"this_run_save_path": flow_path},
    }
    cfg_path = run_dir / "configs" / "stage2_config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"[config] wrote {cfg_path}")

    t1 = time.time()
    history = _run_stage2_mild(model, train_loader, val_loader, train_args,
                               flow_path, args.vae_ckpt)
    stem = Path(args.save_path).stem if args.save_path else "stage2"
    hist_path = run_dir / "logs" / f"{stem}_history.json"
    history.to_json(hist_path)
    print(f"[history] stage2 written by wrapper: {hist_path}")
    print(f"[time] training wall time: {time.time()-t1:.1f}s")

    print("\n--- Smoke eval on test (a few batches, 375/125 caliber) ---")
    _smoke_eval(model, test_loader, device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
