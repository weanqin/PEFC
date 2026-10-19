#!/usr/bin/env python3
"""train_edra_fmt.py — latent-flow-v1 × edra_fmt 两阶段训练入口（包装层）.

任务 10-19-latent-flow-edra-fmt-train（design.md §4）。Stage 1 = VAE
（encoder+proj+decoder，KL 退火）；Stage 2 = 冻结 VAE、潜空间 CFM 流匹配
（D1：默认 mexhat-graph，--dense-fallback 反切 dense）。

数据口径（R2，全部入口硬断言）：
  - manifest 只读复用 gpfm-mamba-v1-fmt 的 fold0 固化划分（34/4/10，
    test [10,11,12,17,24,28,30,44,61,69]，不调用 SubjectFoldSplitter）；
  - 窗口 512（obs 384 + fc 128）/ stride 64，每例 930 窗；
  - 归一化 = 训练折逐通道 mean/std（norm_stats 严格 held-out 路径）。

训练循环零复制：直接 import 复用包内 Stage1Step/Stage2Step/_KLAnnealCallback/
_run_stage1/_run_stage2。模型构建复刻 _build_model 但 obs_len=384/forecast_len=128
（原函数硬编码 375/125，不得复用）；测试集 smoke 评估为 wrapper 自带 384 版
（包内 _smoke_eval 硬编码 375，不得 import）。

产物全部落 --run-dir（默认 outputs/10-19-latent-flow-edra-fmt-train/run_01/）：
  configs/stage{1,2}_config.json  checkpoints/*.pt  logs/*_history.json
（stage2 history 由包内 logs/edra_fmt/ 副本复制而来——训练器内置行为，已 gitignore）。

Run（正式，nohup 由调用方包）：
  python3 /root/npi/models/latent-flow-v1/train_edra_fmt.py --stage 1
  python3 /root/npi/models/latent-flow-v1/train_edra_fmt.py --stage 2
"""
from __future__ import annotations

import os

# --- BLAS 守卫：必须在任何 numpy/scipy/torch 导入之前（npi 环境约束） ---
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data

VARIANT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(VARIANT_DIR))  # 使 latent_flow 可导入

from latent_flow.config import (  # noqa: E402
    CHECKPOINT_DIR,
    EDRA_FMT_DATA_ROOT,
    EDRA_FMT_GEOMETRY_PATH,
)
from latent_flow.data.modma import MODMADataset, collate_fn  # noqa: E402
from latent_flow.data.norm_stats import (  # noqa: E402
    _train_files_hash,
    compute_train_norm_stats,
)
from latent_flow.models.graph_utils import build_normalized_adj, load_geometry  # noqa: E402
from latent_flow.models.oc_latent_flow import OCLatentFlow  # noqa: E402
from latent_flow.scripts.train_oc_latent_flow import (  # noqa: E402
    Stage1Step,  # noqa: F401  (re-exported for lineage clarity)
    Stage2Step,  # noqa: F401
    _KLAnnealCallback,  # noqa: F401
    _run_stage1,
    _run_stage2,
)
from latent_flow.training.seeding import set_seed  # noqa: E402

# --- fmt 契约常量（.trellis/spec/data/edra-fmt.md） ---
OBS_LEN = 384
FORECAST_LEN = 128
WINDOW_SIZE = 512
STEP = 64
EXPECTED_WINDOWS = 930          # (60001-512)//64+1
LATENT_DIM = 64
T_LAT = 16
MANIFEST_DEFAULT = "/root/npi/models/gpfm-mamba-v1-fmt/splits/fold0_status_ok_seed42_val4.json"
EXPECTED_TEST = [10, 11, 12, 17, 24, 28, 30, 44, 61, 69]
RUN_DIR_DEFAULT = "/root/npi/outputs/10-19-latent-flow-edra-fmt-train/run_01"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--stage", type=int, required=True, choices=(1, 2))
    p.add_argument("--run-dir", type=str, default=RUN_DIR_DEFAULT)
    p.add_argument("--manifest", type=str, default=MANIFEST_DEFAULT)
    # 旧库 CLI 默认超参（design D3）
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--beta-kl", type=float, default=1e-4)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--save-path", type=str, default=None,
                   help="Override checkpoint path (smoke 用，防覆盖正式 ckpt)")
    p.add_argument("--vae-ckpt", type=str, default=None,
                   help="Stage-2 only: stage-1 VAE ckpt (default "
                        "<run-dir>/checkpoints/fold0_edra_fmt_vae_best.pt)")
    p.add_argument("--limit-subjects", type=int, default=None,
                   help="冒烟截断：每折只取前 N 例（正式跑禁用）")
    # --- stage-2 流变体（D1：默认 mexhat-graph；--dense-fallback 反切 dense） ---
    p.add_argument("--dense-fallback", action="store_true",
                   help="Stage-2 改用 dense 流（默认 mexhat-graph 的反向开关）")
    p.add_argument("--graph-kernel", type=str, default="mexhat",
                   choices=("gaussian", "mexhat", "dog"))
    p.add_argument("--graph-k", type=int, default=2)
    p.add_argument("--graph-out-mix-alpha", type=float, default=0.5)
    p.add_argument("--mexhat-gamma", type=float, default=0.1)
    p.add_argument("--mexhat-gamma-m", type=float, default=1.0)
    return p.parse_args(argv)


# =============================================================================
# Manifest 解析 + 硬断言（R2 / AC2；失败即拒跑）
# =============================================================================

def load_manifest(manifest_path: str):
    man = json.loads(Path(manifest_path).read_text())
    groups = {int(k): str(v) for k, v in man["groups"].items()}
    train = [int(x) for x in man["train"]]
    val = [int(x) for x in man["val"]]
    test = [int(x) for x in man["test"]]

    assert len(groups) == 48, f"manifest groups != 48: {len(groups)}"
    assert set(groups.values()) == {"HC", "MDD"}, "unexpected group labels"
    n_hc = sum(1 for v in groups.values() if v == "HC")
    assert n_hc == 24 and len(groups) - n_hc == 24, f"HC/MDD != 24/24: {n_hc}"
    assert len(train) == 34, f"train != 34: {len(train)}"
    assert len(val) == 4, f"val != 4: {len(val)}"
    assert len(test) == 10, f"test != 10: {len(test)}"
    assert sorted(test) == EXPECTED_TEST, f"test list mismatch: {sorted(test)}"
    assert set(train) | set(val) | set(test) == set(groups), "union != groups"
    assert not (set(train) & set(val) or set(train) & set(test)
                or set(val) & set(test)), "fold overlap"

    window_record = {}
    for sid in sorted(groups):
        grp = groups[sid]
        f = EDRA_FMT_DATA_ROOT / grp / f"{sid}_rest_v4.npz"
        assert f.exists(), f"subject file missing: {f}"
        with np.load(f, allow_pickle=True) as z:
            d = z["data"]
            n_win = (d.shape[1] - WINDOW_SIZE) // STEP + 1
            assert d.shape == (62, 60001), f"{f}: data shape {d.shape}"
            assert int(z["fs"]) == 250, f"{f}: fs {int(z['fs'])}"
            assert str(z["group"]) == grp and int(z["subject_id"]) == sid, (
                f"{f}: npz group/subject_id mismatch")
        assert n_win == EXPECTED_WINDOWS, f"{f}: {n_win} windows != {EXPECTED_WINDOWS}"
        window_record[str(sid)] = n_win
    print(f"[manifest] {manifest_path}")
    print(f"[manifest] 48/34/4/10 OK; test={sorted(test)}; "
          f"HC 24 / MDD 24; all 48 npz fields+shape OK; "
          f"windows/subject all == {EXPECTED_WINDOWS}")

    def paths(ids):
        return [str(EDRA_FMT_DATA_ROOT / groups[i] / f"{i}_rest_v4.npz") for i in ids]

    return man, groups, paths(train), paths(val), paths(test), window_record


# =============================================================================
# Model（复刻 _build_model，obs_len=384 / forecast_len=128）
# =============================================================================

def build_model_fmt(device, args, stage: int):
    # stage1 不训练流：dense 构建即可（无需 geometry）；stage2 按 D1 默认 mexhat-graph。
    graph_mode = (stage == 2) and not args.dense_fallback
    a_hat, coords = None, None
    if graph_mode:
        if not EDRA_FMT_GEOMETRY_PATH.exists():
            raise SystemExit(
                f"Stage-2 graph-mode needs {EDRA_FMT_GEOMETRY_PATH} — run "
                f"build_edra_fmt_geometry.py first (S2.2 gate must have passed).")
        coords_all, names = load_geometry(path=str(EDRA_FMT_GEOMETRY_PATH))
        assert coords_all.shape == (62, 3), f"geometry shape {coords_all.shape}"
        a_hat = build_normalized_adj(coords_all).to(device)
        if args.graph_kernel in ("mexhat", "dog"):
            coords = coords_all.to(device)
        print(f"Graph-mode flow: kernel={args.graph_kernel}, K={args.graph_k}, "
              f"out mix_alpha={args.graph_out_mix_alpha}, "
              f"mexhat_gamma={args.mexhat_gamma}, adjacency {tuple(a_hat.shape)} "
              f"(geometry={EDRA_FMT_GEOMETRY_PATH})")
    model = OCLatentFlow(
        n_channels=62, latent_dim=LATENT_DIM, t_lat=T_LAT,
        obs_len=OBS_LEN, forecast_len=FORECAST_LEN,
        flow_graph_mode=graph_mode, flow_graph_k=args.graph_k, flow_a_hat=a_hat,
        flow_graph_out_mix_alpha=args.graph_out_mix_alpha,
        flow_graph_kernel=args.graph_kernel, flow_coords=coords,
        flow_mexhat_gamma=args.mexhat_gamma, flow_w_init=None,
        flow_mexhat_gamma_m=args.mexhat_gamma_m, flow_dog_init=None,
        beta_kl_final=args.beta_kl, use_fc_decoder=False,
    ).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,} "
          f"(stage{stage} flow={'graph-' + args.graph_kernel if graph_mode else 'dense'})")
    return model


def _smoke_eval_fmt(model, test_loader, device, n_batches=3):
    """Wrapper 自带 384 版 smoke 评估（包内 _smoke_eval 硬编码 375，不得 import）。"""
    model.eval()
    mae_sum, std_sum, n, seen = 0.0, 0.0, 0, 0
    with torch.no_grad():
        for batch in test_loader:
            x_obs = batch["x_obs"].to(device)
            x_target = batch["x_target"].to(device)
            x_hat = model.sample(x_obs, n_steps=10)
            x_fc = x_target[:, OBS_LEN:, :]
            x_fc_hat = x_hat if model.use_fc_decoder else x_hat[:, OBS_LEN:, :]
            mae_sum += (x_fc_hat - x_fc).abs().mean().item() * x_obs.shape[0]
            std_sum += (x_fc_hat.std() / (x_fc.std() + 1e-8)).item() * x_obs.shape[0]
            n += x_obs.shape[0]
            seen += 1
            if seen >= n_batches:
                break
    print(f"[smoke] test MAE={mae_sum/n:.4f}  std_ratio={std_sum/n:.4f}")


def _git_head() -> str:
    try:
        return subprocess.run(["git", "-C", str(VARIANT_DIR.parents[1]), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unavailable"


def _sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _history_name(args, stage: int) -> str:
    if args.save_path:
        return f"{Path(args.save_path).stem}_history.json"
    return f"stage{stage}_history.json"


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    (run_dir / "configs").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}  dataset: edra_fmt  stage: {args.stage}")
    set_seed(args.seed)

    t0 = time.time()
    man, groups, train_files, val_files, test_files, window_record = \
        load_manifest(args.manifest)
    if args.limit_subjects:
        assert args.limit_subjects >= 1
        train_files = train_files[: args.limit_subjects]
        val_files = val_files[: args.limit_subjects]
        test_files = test_files[: args.limit_subjects]
        print(f"[SMOKE] --limit-subjects {args.limit_subjects}: "
              f"train/val/test truncated to {len(train_files)}/{len(val_files)}/"
              f"{len(test_files)} (manifest assertions ran on the FULL 48)")

    # 归一化：训练折统计（float64 累加，严格 held-out）
    mean, std = compute_train_norm_stats(train_files, file_format="npz", channels=62)
    norm_stats = (mean, std)
    files_hash = _train_files_hash(train_files)
    norm_hash = hashlib.sha256(
        np.concatenate([mean.ravel(), std.ravel()]).tobytes()).hexdigest()

    def make_ds(files):
        return MODMADataset(
            subject_files=files, group="MIX", normalize_per_channel=True,
            window_size=WINDOW_SIZE, step=STEP, obs_size=OBS_LEN,
            dataset="edra_fmt", norm_stats=norm_stats,
        )

    train_ds, val_ds, test_ds = make_ds(train_files), make_ds(val_files), make_ds(test_files)
    for tag, ds, files in (("train", train_ds, train_files), ("val", val_ds, val_files),
                           ("test", test_ds, test_files)):
        assert len(ds) == EXPECTED_WINDOWS * len(files), (
            f"{tag}: {len(ds)} windows != {EXPECTED_WINDOWS} x {len(files)}")
    train_loader = data.DataLoader(train_ds, batch_size=args.batch_size,
                                   shuffle=True, collate_fn=collate_fn)
    val_loader = data.DataLoader(val_ds, batch_size=args.batch_size,
                                 shuffle=False, collate_fn=collate_fn)
    test_loader = data.DataLoader(test_ds, batch_size=args.batch_size,
                                  shuffle=False, collate_fn=collate_fn)
    print(f"[data] train/val/test subjects = {len(train_files)}/{len(val_files)}/"
          f"{len(test_files)}; windows = {len(train_ds)}/{len(val_ds)}/{len(test_ds)} "
          f"(norm: train-fold stats, files_hash={files_hash[:12]}, "
          f"prep {time.time()-t0:.1f}s)")

    model = build_model_fmt(device, args, stage=args.stage)

    # args 命名空间钉死（design §4.5：dataset=edra_fmt 决定包内 history 落点；
    # 其余为旧库默认行为的显式固定）
    train_args = argparse.Namespace(
        dataset="edra_fmt", unfreeze_vae=False,
        anti_uniform_lambda=0.0, sparse_lambda=0.0,
        lr_scheduler="none", lr_warmup_epochs=0,
        epochs=args.epochs, lr=args.lr, patience=args.patience, seed=args.seed,
    )

    suffix = "" if args.dense_fallback else (
        "_graph_mexhat" if args.graph_kernel == "mexhat" else
        "_graph_dog" if args.graph_kernel == "dog" else "_graph")
    vae_path = args.save_path or str(run_dir / "checkpoints" / "fold0_edra_fmt_vae_best.pt")
    flow_path = args.save_path or str(
        run_dir / "checkpoints" / f"fold0_edra_fmt_flow_best{suffix}.pt")

    # resolved config 落盘（训练前，可复现凭据）
    cfg = {
        "task": "10-19-latent-flow-edra-fmt-train",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_head": _git_head(),
        "stage": args.stage,
        "cli": {k: v for k, v in vars(args).items()},
        "pinned_trainer_args": vars(train_args),
        "windows": {"window_size": WINDOW_SIZE, "obs_len": OBS_LEN,
                    "forecast_len": FORECAST_LEN, "step": STEP,
                    "expected_windows_per_subject": EXPECTED_WINDOWS,
                    "assert_record": window_record,
                    "counts": {"train": len(train_ds), "val": len(val_ds),
                               "test": len(test_ds)}},
        "manifest": {
            "path": args.manifest, "sha256": _sha256_file(args.manifest),
            "counts": {"train": 34, "val": 4, "test": 10, "total": 48},
            "test_sorted": EXPECTED_TEST,
            "groups": {str(k): groups[k] for k in sorted(groups)},
            "subject_files": {"train": train_files, "val": val_files,
                              "test": test_files},
        },
        "norm_stats": {"source": "train fold (compute_train_norm_stats, float64)",
                       "files_hash": files_hash, "mean_std_sha256": norm_hash,
                       "shape": list(mean.shape)},
        "model": {"n_channels": 62, "latent_dim": LATENT_DIM, "t_lat": T_LAT,
                  "obs_len": OBS_LEN, "forecast_len": FORECAST_LEN,
                  "flow_variant": ("dense" if args.dense_fallback else
                                   f"graph-{args.graph_kernel}"),
                  "beta_kl_final": args.beta_kl, "use_fc_decoder": False},
        "geometry": {"path": str(EDRA_FMT_GEOMETRY_PATH),
                     "exists": EDRA_FMT_GEOMETRY_PATH.exists()},
        "checkpoints": {"stage1_vae": str(run_dir / "checkpoints" / "fold0_edra_fmt_vae_best.pt"),
                        "stage2_flow_default": str(run_dir / "checkpoints" /
                                                   f"fold0_edra_fmt_flow_best{suffix}.pt"),
                        "this_run_save_path": (args.save_path or
                                               (vae_path if args.stage == 1 else flow_path))},
        "package_history_dir": str(CHECKPOINT_DIR.parent / "logs" / "edra_fmt"),
    }
    cfg_path = run_dir / "configs" / f"stage{args.stage}_config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"[config] wrote {cfg_path}")

    t1 = time.time()
    if args.stage == 1:
        history = _run_stage1(model, train_loader, val_loader, train_args, vae_path)
        hist_path = run_dir / "logs" / _history_name(args, 1)
        history.to_json(hist_path)          # wrapper 自写（_run_stage1 无内置落盘）
        print(f"[history] stage1 written by wrapper: {hist_path}")
    else:
        ckpt = args.vae_ckpt or str(run_dir / "checkpoints" / "fold0_edra_fmt_vae_best.pt")
        if not Path(ckpt).exists():
            raise FileNotFoundError(
                f"Stage 2 needs a stage-1 VAE checkpoint: {ckpt} not found.")
        history = _run_stage2(model, train_loader, val_loader, train_args,
                              flow_path, ckpt)
        # 训练器内置已写包内 logs/edra_fmt/<stem>_history.json —— 复制进 run_01
        src = CHECKPOINT_DIR.parent / "logs" / "edra_fmt" / \
            f"{Path(flow_path).stem}_history.json"
        dst = run_dir / "logs" / _history_name(args, 2)
        if src.exists():
            shutil.copy2(src, dst)
            print(f"[history] stage2 copied {src} -> {dst}")
        else:
            print(f"[warn] package history not found (patience=0?): {src}")
    print(f"[time] training wall time: {time.time()-t1:.1f}s")

    print("\n--- Smoke eval on test (a few batches, 384/128 caliber) ---")
    _smoke_eval_fmt(model, test_loader, device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
