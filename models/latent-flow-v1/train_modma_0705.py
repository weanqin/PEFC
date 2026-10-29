#!/usr/bin/env python3
"""train_modma_0705.py — MODMA resting × 07-05 架构损失（标准 graph-mexhat）stage-2 训练.

任务 10-29-modma-0705-f1234。07-05 口径（与 fold0 原版 fold0_oc_latent_flow_
flow_best_graph_mexhat.pt 同构）：标准 OCLatentFlow graph-mexhat 构建（无
exit-GNN / 无 symw / 无正则），stage-2 = 包内 _run_stage2 flow-only 路径
（encoder/proj/decoder 全冻结，loss = recon2 + corr2 + cons2，β=0）。

- 数据 = 包内 MODMA 管线（500 窗 / step 250 / obs 375 / fc 125；逐被试逐通道
  z-score；SubjectFoldSplitter 5 折 seed 42；train 折内 5 例 held-out val）；
- stage-1 VAE = 旧库原位 fold{N}_oc_latent_flow_vae_best.pt（只读；07-05 fold0
  当年同构流程）；
- 超参（用户 2026-10-29 拍板）：epochs 50 / batch 64 / patience 10 / lr 1e-3 /
  seed 42。

边界（PRD 已登记）：现行包代码 W 自由非对称（07-05 原始训练时旧库强制 W 对称，
该机制轴现路径不可达）；bf16 AMP 非逐位确定。

Run:
  OMP_NUM_THREADS=8 MODMA_DATA_ROOT=/root/autodl-tmp/modma python3 \
    models/latent-flow-v1/train_modma_0705.py --fold 1 \
    --run-dir /root/npi/outputs/10-29-modma-0705-f1234/fold1/run
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
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

VARIANT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(VARIANT_DIR))

from latent_flow.config import CHECKPOINT_DIR  # noqa: E402
from latent_flow.data.datasets import get_dataset, list_hc, list_mdd  # noqa: E402
from latent_flow.data.splitter import SubjectFoldSplitter  # noqa: E402
from latent_flow.models.graph_utils import (  # noqa: E402
    build_normalized_adj,
    load_geometry,
)
from latent_flow.models.oc_latent_flow import OCLatentFlow  # noqa: E402
from latent_flow.scripts.train_oc_latent_flow import (  # noqa: E402
    _build_loaders,
    _run_stage2,
    _smoke_eval,
)
from latent_flow.training.seeding import set_seed  # noqa: E402

N_CHANNELS = 128
LATENT_DIM = 64
T_LAT = 16
OBS_LEN = 375
FORECAST_LEN = 125
OLD_LIB_CKPT_DIR = Path("/root/autodl-tmp/latent_ode/checkpoints")


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


def build_model_modma_std(device):
    """07-05 口径标准 graph-mexhat 构建（与 load_model mexhat 分支同构）。

    EC driver（ec_modma_symw_v4.py --arch std-mexhat）复用本函数装载 ckpt。
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
    print(f"Graph-mode flow: kernel=mexhat, K=2, mix_alpha=0.5, gamma=0.1, "
          f"adjacency {tuple(a_hat.shape)} (dataset=modma)")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,} "
          f"(stage2 flow=graph-mexhat, 07-05 caliber)")
    return model


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--fold", type=int, required=True, choices=range(5),
                   help="10-29 任务原 choices=(1,2,3,4)；10-29-modma-ec-te-corr 起放宽含 0"
                        "（fold0 同口径补全，向后兼容）")
    p.add_argument("--run-dir", type=str, required=True,
                   help="输出目录（必填，无默认——防误写旧任务树）")
    p.add_argument("--vae-ckpt", type=str, default="",
                   help="默认旧库原位 fold{N}_oc_latent_flow_vae_best.pt")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--beta-kl", type=float, default=1e-4)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--limit-subjects", type=int, default=None,
                   help="冒烟截断：train 折截前 N 例（须 >5，5 例 held-out val）")
    p.add_argument("--save-path", type=str, default=None,
                   help="覆盖 ckpt 落点（冒烟用，防覆盖正式 ckpt）")
    return p.parse_args(argv)


def main() -> int:
    args = parse_args()
    vae_ckpt = args.vae_ckpt or str(
        OLD_LIB_CKPT_DIR / f"fold{args.fold}_oc_latent_flow_vae_best.pt")
    assert Path(vae_ckpt).exists(), f"stage-1 VAE not found: {vae_ckpt}"

    run_dir = Path(args.run_dir)
    (run_dir / "configs").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}  dataset: modma  fold: {args.fold}  "
          f"stage: 2 (07-05 caliber: std graph-mexhat, flow-only)")
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

    model = build_model_modma_std(device)

    train_args = argparse.Namespace(
        dataset="modma", unfreeze_vae=False,
        anti_uniform_lambda=0.0, sparse_lambda=0.0,
        lr_scheduler="none", lr_warmup_epochs=0,
        epochs=args.epochs, lr=args.lr, patience=args.patience, seed=args.seed,
    )

    flow_path = args.save_path or str(
        run_dir / "checkpoints" /
        f"fold{args.fold}_modma_0705_graph_mexhat.pt")

    n_total = sum(p.numel() for p in model.parameters())
    n_flow = sum(p.numel() for p in model.flow.parameters())
    cfg = {
        "task": "10-29-modma-0705-f1234 (07-05 caliber std graph-mexhat, "
                "flow-only stage-2, MODMA fold1-4)",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_head": _git_head(),
        "cli": vars(args),
        "resolved_vae_ckpt": vae_ckpt,
        "pinned_trainer_args": vars(train_args),
        "data": {"dataset": "modma", "fold": args.fold,
                 "windows": {"window": 500, "step": 250, "obs": OBS_LEN,
                             "fc": FORECAST_LEN},
                 "normalization": "per-subject per-channel z-score",
                 "fold_counts": {"train": len(train_files), "test": len(test_files),
                                 "val_heldout": 5}},
        "model": {"n_channels": N_CHANNELS, "latent_dim": LATENT_DIM,
                  "t_lat": T_LAT, "obs_len": OBS_LEN,
                  "forecast_len": FORECAST_LEN,
                  "flow_variant": "graph-mexhat (07-05 caliber)",
                  "beta_kl_final": args.beta_kl, "use_fc_decoder": False},
        "stage2": {"runner": "package _run_stage2 (VAE frozen, flow-only)",
                   "frozen_modules": ["encoder", "proj", "decoder"],
                   "anti_uniform_lambda": 0.0, "sym_w": False,
                   "trainable_params": n_flow},
        "vae_ckpt": {"path": vae_ckpt, "sha256": _sha256_file(vae_ckpt)},
        "checkpoints": {"this_run_save_path": flow_path},
    }
    cfg_path = run_dir / "configs" / "stage2_config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"[config] wrote {cfg_path}")

    t1 = time.time()
    history = _run_stage2(model, train_loader, val_loader, train_args,
                          flow_path, vae_ckpt)
    # 包内 _run_stage2 写 logs/modma/<stem>_history.json —— 复制进 run 目录
    src = CHECKPOINT_DIR.parent / "logs" / "modma" / \
        f"{Path(flow_path).stem}_history.json"
    dst = run_dir / "logs" / \
        (f"{Path(args.save_path).stem}_history.json" if args.save_path
         else "stage2_history.json")
    if src.exists():
        shutil.copy2(src, dst)
        print(f"[history] stage2 copied {src} -> {dst}")
    else:
        print(f"[warn] package history not found (patience=0?): {src}")
    print(f"[time] training wall time: {time.time()-t1:.1f}s")

    print("\n--- Smoke eval on test (a few batches, 375/125 caliber) ---")
    _smoke_eval(model, test_loader, device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
