#!/usr/bin/env python3
"""viz_mild.py — Mild Depression 线 EEG 波形检验可视化（visualizer，10-21-latent-flow-mild-train）.

按 .trellis/spec/research/visualization.md：
  - 统一 visualizer（数据 + 图配置输入契约；本任务变体层模块，包体零改动）；
  - **渲染口径 v2.1（用户 2026-10-21 指定，覆盖 v2 的面板构成）**：每面板仅
    real-future（蓝）+ gen medoid（红实线，来自 K 个实际生成样本）+ 90% 预测
    区间（Q05–Q95 分位带）；past 上下文灰线去掉，时间轴缩放 forecast 段；
  - medoid/分位带计算契约与 v2 一致（实样本代表、分位带、禁 min-max）；
  - nature-figure 默认风格（rcParams 契约）。

输入契约：--fold N --ckpt <stage2 ckpt> --subjects sid,sid,... --trials idx[,idx..]
  --k K（每 trial 生成样本数）--seed S --out-dir DIR。
  口径：--window-len（默认 1500 = 全 trial 旧行为）与 --obs-len/--forecast-len
  须与 ckpt 训练口径一致（obs+forecast == window_len 硬断言；10-24 前缀窗
  500 = 375/125）；归一化与 trial 切片 import 训练侧同一实现
  （compute_mild_train_norm / MildTrialDataset），前 window_len 点统计。
输出：每 (subject, trial) 一张 PNG（2×4 通道面板）+ viz_meta.json（口径与产物登记）。

Run:
  python3 /root/npi/models/latent-flow-v1/viz_mild.py --fold 0 \
    --ckpt /root/npi/outputs/10-21-latent-flow-mild-train/run_fold0/checkpoints/fold0_mild_flow_best_graph_mexhat.pt
  python3 /root/npi/models/latent-flow-v1/viz_mild.py --fold 0 \
    --ckpt .../run_w500_fold0/checkpoints/fold0_mild_flow_best_graph_mexhat.pt \
    --window-len 500 --obs-len 375 --forecast-len 125 --subjects 1,7,21 \
    --out-dir .../run_w500_fold0/viz
"""
from __future__ import annotations

import os

# --- BLAS 守卫：必须在任何 numpy/scipy/torch 导入之前（npi 环境约束） ---
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

VARIANT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(VARIANT_DIR))

import matplotlib as mpl  # noqa: E402

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from train_mild_depress import (  # noqa: E402
    FORECAST_LEN,
    MANIFEST_DEFAULT,
    N_CHANNELS,
    OBS_LEN,
    TRIALS,
    WINDOW,
    build_model_mild,
    compute_mild_train_norm,
    load_manifest,
)
from latent_flow.training.seeding import set_seed  # noqa: E402

FS = 250.0
# 通道抽样：沿 GSN-128 阵列均匀取 8 个（E 序假设，D4 签认；仅展示用途）
CHANNELS_DEFAULT = [1, 18, 36, 54, 62, 75, 100, 128]

# nature-figure 默认风格（visualization.md 契约；检验图不进投稿管线，PNG 交付）
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
})

C_PAST, C_REAL, C_GEN, C_BAND = "#9e9e9e", "#1f77b4", "#d62728", "#d62728"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Mild Depression EEG 波形检验可视化（v2 口径）")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--manifest", type=str, default=str(MANIFEST_DEFAULT))
    p.add_argument("--ckpt", type=str, required=True, help="stage2 flow ckpt 路径")
    p.add_argument("--subjects", type=str, default="",
                   help="逗号分隔 subject_id；默认自动取 fold test 首个 MDD+HC")
    p.add_argument("--trials", type=str, default="0", help="逗号分隔 trial 索引（0–29）")
    p.add_argument("--k", type=int, default=32, help="每 trial 生成样本数")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--window-len", type=int, default=WINDOW,
                   help=f"前缀窗口长度（默认 {WINDOW} = 全 trial 旧行为；"
                        f"0 < window_len <= {WINDOW}；与 ckpt 训练口径一致）")
    p.add_argument("--obs-len", type=int, default=OBS_LEN,
                   help=f"obs 段长度（默认 {OBS_LEN}；与 --forecast-len 之和必须为 --window-len，须与 ckpt 训练口径一致）")
    p.add_argument("--forecast-len", type=int, default=FORECAST_LEN,
                   help=f"forecast 段长度（默认 {FORECAST_LEN}；与 --obs-len 之和必须为 --window-len，须与 ckpt 训练口径一致）")
    p.add_argument("--channels", type=str,
                   default=",".join(str(c) for c in CHANNELS_DEFAULT),
                   help="逗号分隔通道号（E 序，1–128）")
    # --- 流出口变体透传（10-24-latent-flow-exit-gnn；默认关 = 原架构） ---
    p.add_argument("--flow-exit-gnn", action="store_true",
                   help="出口 GNN 变体 ckpt 装载（FlowVelocityNetExitGNN 架构 "
                        "strict load 匹配；须与 ckpt 训练口径一致，不一致即 "
                        "拒跑）")
    p.add_argument("--out-dir", type=str,
                   default="/root/npi/outputs/10-21-latent-flow-mild-train/run_fold0/viz")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args(argv)


def _model_args(flow_exit_gnn: bool = False):
    """build_model_mild 所需 args 命名空间（= 训练时钉死的 stage2 mexhat 默认；
    flow_exit_gnn 透传出口 GNN 变体架构（默认 False = 原架构，旧调用面
    （ec_mild_unbiased 等）零参数沿用不受影响）。"""
    return argparse.Namespace(
        dense_fallback=False, graph_kernel="mexhat", graph_k=2,
        graph_out_mix_alpha=0.5, mexhat_gamma=0.1, mexhat_gamma_m=1.0,
        beta_kl=1e-4, flow_exit_gnn=flow_exit_gnn,
    )


def load_flow_ckpt(model, ckpt_path: str):
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = payload["state_dict"] if isinstance(payload, dict) and \
        payload.get("format_version") == 1 else payload
    missing, unexpected = model.load_state_dict(state, strict=True)
    _ = missing, unexpected  # strict=True 下必然为空；显式调用以强断言架构一致
    model.eval()
    return model


@torch.no_grad()
def sample_forecast(model, x_obs, k: int, device, obs_len: int = OBS_LEN):
    """x_obs (1,obs_len,128) → k 条生成样本的 forecast (k,forecast_len,128) numpy。"""
    xb = x_obs.unsqueeze(0).to(device).repeat(k, 1, 1)
    x_hat = model.sample(xb, n_steps=10)          # (k, obs+forecast, 128)
    return x_hat[:, obs_len:, :].cpu().numpy()


def medoid_index(fc: np.ndarray) -> int:
    """v2 口径 medoid：k* = argmin_k mean_{j≠k} d(x_k,x_j)，d = 完整多通道轨迹
    RMSE（Euclid/√(C·F)，全局常数不改 argmin）。代表曲线取实际样本 fc[k*]。"""
    k = fc.shape[0]
    flat = fc.reshape(k, -1).astype(np.float64)
    gram = (flat * flat).sum(axis=1)
    d2 = gram[:, None] + gram[None, :] - 2.0 * flat @ flat.T
    np.maximum(d2, 0.0, out=d2)
    np.fill_diagonal(d2, np.inf)
    rmse = np.sqrt(d2 / flat.shape[1])
    return int(np.argmin(rmse.mean(axis=1)))


def render_trial(x_target: np.ndarray, fc: np.ndarray, channels, k_used: int,
                 title: str, out_png: Path, obs_len: int = OBS_LEN,
                 forecast_len: int = FORECAST_LEN):
    """v2.1 口径（用户 2026-10-21 指定）：仅可视化 real-future + gen-future——
    蓝真实未来 + 红 medoid（实际样本）+ Q05–Q95 带；past 灰线上下文去掉，
    时间轴缩放到 forecast 段（obs_len/fs – (obs_len+forecast_len)/fs，随
    口径自适应）。medoid/分位带计算口径与 v2 不变。obs_len/forecast_len
    缺省 = 训练默认 1125/375。"""
    q05, q95 = np.quantile(fc, [0.05, 0.95], axis=0)     # (forecast_len,128)
    k_star = medoid_index(fc)
    med = fc[k_star]                                      # (forecast_len,128) 实际样本
    t_fc = np.arange(obs_len, obs_len + forecast_len) / FS

    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.0), sharex=True,
                             constrained_layout=True)
    for ax, ch in zip(axes.ravel(), channels):
        ci = ch - 1
        ax.plot(t_fc, x_target[obs_len:, ci], color=C_REAL, lw=1.0, zorder=3)
        ax.fill_between(t_fc, q05[:, ci], q95[:, ci], color=C_BAND, alpha=0.18,
                        lw=0, zorder=2)
        ax.plot(t_fc, med[:, ci], color=C_GEN, lw=1.0, zorder=4)
        ax.set_title(f"E{ch}", fontsize=7, pad=2)
        ax.set_xlim(obs_len / FS, (obs_len + forecast_len) / FS)
        ax.tick_params(labelsize=6, pad=1.5)
    for ax in axes[-1]:
        ax.set_xlabel("time (s)", fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("amp (z)", fontsize=7)

    handles = [
        plt.Line2D([], [], color=C_REAL, lw=1.4, label="real future"),
        plt.Line2D([], [], color=C_GEN, lw=1.4, label=f"gen medoid (K={k_used})"),
        plt.Line2D([], [], color=C_BAND, alpha=0.3, lw=6,
                   label="gen 90% PI (Q05–Q95)"),
    ]
    fig.legend(handles=handles, loc="outside upper right", ncols=3, fontsize=6.5,
               handlelength=1.6, columnspacing=1.2)
    fig.suptitle(title, fontsize=8)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    assert args.obs_len > 0 and args.forecast_len > 0, \
        f"--obs-len/--forecast-len must be positive: {args.obs_len}/{args.forecast_len}"
    assert 0 < args.window_len <= WINDOW, \
        f"--window-len must be in (0, {WINDOW}]: {args.window_len}"
    assert args.obs_len + args.forecast_len == args.window_len, \
        f"--obs-len + --forecast-len ({args.obs_len}+{args.forecast_len}) " \
        f"!= --window-len {args.window_len}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device
                          else ("cuda" if torch.cuda.is_available() else "cpu"))

    man, subjects, train_f, val_f, test_f, fold_dict, _trial_rec = \
        load_manifest(args.manifest, args.fold)
    groups = {sid: v["group"] for sid, v in subjects.items()}
    sid_to_file = {sid: v["file"] for sid, v in subjects.items()}
    # fold 记录的 test 即 subject_id 列表（文件名 m_/h_ 前缀是组内编号，
    # h_i ↔ subject_id 20+i，不得从 stem 解析——stem 解析在 HC 段错位且
    # 使默认首个 HC 挑选必然 StopIteration）
    test_ids = sorted(fold_dict["test"])
    if args.subjects:
        sids = [int(s) for s in args.subjects.split(",")]
    else:
        first_mdd = next(s for s in test_ids if groups[s] == "MDD")
        first_hc = next(s for s in test_ids if groups[s] == "HC")
        sids = [first_mdd, first_hc]
    trials = [int(t) for t in args.trials.split(",")]
    channels = [int(c) for c in args.channels.split(",")]
    assert all(1 <= c <= N_CHANNELS for c in channels)

    # 数据（train 折归一化统计，严格 held-out，与训练入口同一实现；
    # 累加域 = 前 window_len 点前缀窗，与训练侧 compute_mild_train_norm 一致）
    norm_stats = compute_mild_train_norm(train_f, window_len=args.window_len)

    # 模型：stage2 mexhat-graph 架构 + strict 加载（架构与 ckpt 不一致即拒跑；
    # --flow-exit-gnn 透传出口 GNN 变体架构，与 ckpt 训练口径匹配）
    model = build_model_mild(device,
                             _model_args(flow_exit_gnn=args.flow_exit_gnn),
                             stage=2,
                             obs_len=args.obs_len,
                             forecast_len=args.forecast_len)
    load_flow_ckpt(model, args.ckpt)

    set_seed(args.seed)
    meta = {
        "task": "latent-flow-v1 x mild-depression viz wrapper "
                "(shared by tasks 10-21 / 10-24; run provenance = out_dir "
                "+ owning task experiment.md)",
        "caliber": "trajectories-v2.1 "
        "(user 2026-10-21: real future blue + gen medoid red + Q05-Q95 band; "
        "past context dropped, forecast window only)",
        "fold": args.fold, "ckpt": args.ckpt, "k": args.k, "seed": args.seed,
        "window_len": args.window_len,
        # 装载旗标随 meta 落盘（10-24-latent-flow-exit-gnn check 补齐，与
        # quality_meta 口径登记对齐）：True = 出口 GNN 变体架构装载
        "flow_exit_gnn": bool(args.flow_exit_gnn),
        "obs_len": args.obs_len, "forecast_len": args.forecast_len,
        "channels_E": channels,
        "norm": f"train-fold per-channel stats over first {args.window_len} "
                f"samples (held-out; same implementation as training)",
        "figures": [], "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    sid_to_file = {sid: v["file"] for sid, v in subjects.items()}
    for sid in sids:
        f = sid_to_file[sid]
        ds_files = [f]
        from train_mild_depress import MildTrialDataset
        ds = MildTrialDataset(ds_files, norm_stats,
                              obs_len=args.obs_len,
                              forecast_len=args.forecast_len,
                              window_len=args.window_len)
        for ti in trials:
            assert 0 <= ti < TRIALS
            x_target = ds.trials[ti].numpy()     # (window_len,128) z 域
            fc = sample_forecast(model, ds.trials[ti], args.k, device,
                                 obs_len=args.obs_len)
            title = (f"subj {sid} ({groups[sid]}) · trial {ti} · "
                     f"fold{args.fold} · forecast "
                     f"{args.forecast_len / FS:.1f} s after "
                     f"{args.obs_len / FS:.1f} s obs")
            png = out_dir / f"wave_subj{sid:02d}_trial{ti:02d}.png"
            render_trial(x_target, fc, channels, args.k, title, png,
                         obs_len=args.obs_len,
                         forecast_len=args.forecast_len)
            meta["figures"].append(str(png))
            print(f"[viz] {png}  (medoid of K={args.k})")

    meta_path = out_dir / "viz_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[viz] meta -> {meta_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
