#!/usr/bin/env python3
"""quality_check_mild.py — Mild Depression 生成质量数值抽查（10-24 任务持久化，上轮 Run 004–006 内联脚本落盘修正）.

口径来源 .trellis/spec/research/evaluation-calibers.md「代理脑生成质量数值抽查」
条目 + 10-24 design §5。指标双口径（全程 / 去末端，防 Euler 末步发散伪影）：
  ① 逐时间点 Q05–Q95 coverage：真实 future 各时刻（逐通道）落在 K 条生成
     样本的 90% 分位带内的比例；按 trial 均值后按例聚合（mean/max），另报
     全体总体均值；
  ② medoid/real std 比：medoid 样本（viz_mild.medoid_index 选出的实际生成
     样本，禁统计重构）逐通道 std ÷ 真实 future 逐通道 std，按 trial 对通道
     均值，按例报告 mean/min/max。

去末端口径 = **定点沿用登记的绝对 25 点**（evaluation-calibers 登记口径；
占 fc 窗比例随 forecast_len 变化——上轮 375 点窗 6.7%，本轮 125 点窗 20%，
跨口径数值不可比由窗口变更覆盖）。

**确定性采样登记（判读纪律）**：本线 sample() 全确定性（z_obs=μ + 确定性
Euler 10 步 + 确定性 decoder，无后验采样随机源）⇒ K 条样本逐位相同 ⇒ 分位
带退化为单点、coverage 结构性 ≈ 0，**不携带质量信息**；质量判读以
medoid/real std 比 + 波形人工判为主（用户 2026-10-24 拍板，coverage 照登
并注明结构性）。

数据口径：--window-len 前缀窗（默认 1500 = 旧行为；10-24 口径 500 = obs
375 / fc 125）；归一化与 trial 切片 import 训练侧同一实现
（load_manifest / compute_mild_train_norm / MildTrialDataset）；采样管线
import viz_mild.sample_forecast / medoid_index（与波形渲染同一实现）。

输入契约：--fold N --ckpt <stage2 ckpt> --window-len/--obs-len/--forecast-len
  （须与 ckpt 训练口径一致，obs+forecast == window_len 硬断言）
  --subjects sid,sid,...（默认 1,7,21 = fold0 test 首 MDD+HC 判读 triad，
  与上轮 Run 004–006 同 triad 保持连续性；须属 fold test）--trials all|idx,... --k K
  --seed S --out-dir DIR（必填，无默认——防误写旧任务目录）。
输出：quality_meta.json（口径 + 全部数字 + 凭据）+ stdout 摘要表。

Run:
  python3 /root/npi/models/latent-flow-v1/quality_check_mild.py --fold 0 \
    --ckpt .../run_w500_fold0/checkpoints/fold0_mild_flow_best_graph_mexhat.pt \
    --window-len 500 --obs-len 375 --forecast-len 125 \
    --out-dir .../run_w500_fold0/quality
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
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

VARIANT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(VARIANT_DIR))  # 使 latent_flow 与同目录脚本可导入

from latent_flow.data.norm_stats import _train_files_hash  # noqa: E402
from latent_flow.training.seeding import set_seed  # noqa: E402
from train_mild_depress import (  # noqa: E402
    FORECAST_LEN,
    MANIFEST_DEFAULT,
    OBS_LEN,
    TRIALS,
    WINDOW,
    _sha256_file,
    build_model_mild,
    compute_mild_train_norm,
    load_manifest,
)
from viz_mild import _model_args, load_flow_ckpt, medoid_index, sample_forecast  # noqa: E402

# 去末端口径：定点沿用登记的绝对 25 点（非按 fc 长度缩放）
TRIM_TAIL = 25
# 默认判读 triad（fold0 test 首 MDD 1/7 + 首 HC 21，上轮 Run 004–006 定点
# 沿用；他折或改例须显式传 --subjects）
SUBJECTS_DEFAULT = "1,7,21"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Mild Depression 生成质量数值抽查（coverage/medoid 双口径，"
                    "确定性采样结构性注明）")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--manifest", type=str, default=str(MANIFEST_DEFAULT))
    p.add_argument("--ckpt", type=str, required=True, help="stage2 flow ckpt 路径")
    p.add_argument("--window-len", type=int, default=WINDOW,
                   help=f"前缀窗口长度（默认 {WINDOW} = 全 trial 旧行为；"
                        f"0 < window_len <= {WINDOW}；须与 ckpt 训练口径一致）")
    p.add_argument("--obs-len", type=int, default=OBS_LEN,
                   help=f"obs 段长度（默认 {OBS_LEN}；与 --forecast-len 之和必须为 --window-len）")
    p.add_argument("--forecast-len", type=int, default=FORECAST_LEN,
                   help=f"forecast 段长度（默认 {FORECAST_LEN}；与 --obs-len 之和必须为 --window-len，"
                        f"且 > {TRIM_TAIL}（去末端口径定点 {TRIM_TAIL} 点）")
    p.add_argument("--subjects", type=str, default=SUBJECTS_DEFAULT,
                   help=f"逗号分隔 subject_id，须属 fold test（默认 {SUBJECTS_DEFAULT} = "
                        f"fold0 test 首 MDD+HC 判读 triad，与上轮 Run 004–006 同 triad）")
    p.add_argument("--trials", type=str, default="all",
                   help="逗号分隔 trial 索引（0–29）或 all（默认全部）")
    p.add_argument("--k", type=int, default=32, help="每 trial 生成样本数")
    p.add_argument("--seed", type=int, default=42)
    # --- 流出口变体透传（10-24-latent-flow-exit-gnn；默认关 = 原架构） ---
    p.add_argument("--flow-exit-gnn", action="store_true",
                   help="出口 GNN 变体 ckpt 装载（FlowVelocityNetExitGNN 架构 "
                        "strict load 匹配；须与 ckpt 训练口径一致，不一致即 "
                        "拒跑；旗标随 quality_meta 口径串落盘）")
    p.add_argument("--out-dir", type=str, required=True,
                   help="输出目录（必填，无默认——防误写旧任务输出树）")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args(argv)


def _parse_trials(spec: str):
    """'all' → 0..TRIALS-1；否则逗号分隔索引（越界即拒跑）。"""
    if spec.strip().lower() == "all":
        return list(range(TRIALS))
    ts = [int(t) for t in spec.split(",")]
    assert ts, f"--trials 解析为空: {spec!r}"
    assert all(0 <= t < TRIALS for t in ts), \
        f"trial 索引越界（0–{TRIALS - 1}）: {ts}"
    return ts


def _trim_seg(fc: np.ndarray, real: np.ndarray, trim: int):
    """截去末端 trim 点（trim=0 原样返回）；fc (K,F,C) / real (F,C)。"""
    if trim:
        end = fc.shape[1] - trim
        assert end > 0, f"trim {trim} 吃光 forecast 段"
        return fc[:, :end, :], real[:end, :]
    return fc, real


def _coverage(fc: np.ndarray, real: np.ndarray, trim: int) -> float:
    """逐时间点 Q05–Q95 coverage：real (F,C) 各时刻逐通道是否落 K 样本
    （K,F,C）90% 分位带内（trim>0 时双方去末 trim 点再算）。float64 统计。"""
    f, r = _trim_seg(fc, real, trim)
    q05, q95 = np.quantile(f.astype(np.float64), [0.05, 0.95], axis=0)  # (F',C)
    inside = (r >= q05) & (r <= q95)
    return float(inside.astype(np.float64).mean())


def _std_ratio(fc: np.ndarray, real: np.ndarray, trim: int) -> float:
    """medoid/real std 比：medoid 实样本逐通道 std ÷ real 逐通道 std，
    按通道均值返回（float64；medoid 在该口径段内选取——全程/去末各自
    自洽；real 逐通道 std 下护栏 1e-12，z 域非零数据不触发）。"""
    f, r = _trim_seg(fc, real, trim)
    f64 = f.astype(np.float64)
    k_star = medoid_index(f64)
    med = f64[k_star]                                      # (F',C) 实际样本
    ratio = med.std(axis=0) / np.maximum(
        r.astype(np.float64).std(axis=0), 1e-12)
    return float(ratio.mean())


def main() -> int:
    args = parse_args()
    # --- 口径硬断言（失败即拒跑） ---
    assert args.obs_len > 0 and args.forecast_len > 0, \
        f"--obs-len/--forecast-len must be positive: {args.obs_len}/{args.forecast_len}"
    assert 0 < args.window_len <= WINDOW, \
        f"--window-len must be in (0, {WINDOW}]: {args.window_len}"
    assert args.obs_len + args.forecast_len == args.window_len, \
        f"--obs-len + --forecast-len ({args.obs_len}+{args.forecast_len}) " \
        f"!= --window-len {args.window_len}"
    assert args.forecast_len > TRIM_TAIL, \
        f"--forecast-len ({args.forecast_len}) 须 > 去末端口径定点 {TRIM_TAIL} 点"
    assert args.k >= 1, f"--k must be >= 1: {args.k}"
    assert Path(args.ckpt).exists(), f"ckpt not found: {args.ckpt}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device
                          else ("cuda" if torch.cuda.is_available() else "cpu"))

    man, subjects, train_f, _val_f, test_f, fold_dict, _trial_rec = \
        load_manifest(args.manifest, args.fold)
    groups = {sid: v["group"] for sid, v in subjects.items()}
    sid_to_file = {sid: v["file"] for sid, v in subjects.items()}
    # fold 记录的 test 即 subject_id 列表（文件名 m_/h_ 前缀是组内编号，
    # h_i ↔ subject_id 20+i，不得从 stem 解析——上轮 viz 即此坑）
    test_ids = sorted(fold_dict["test"])
    sids = [int(s) for s in args.subjects.split(",")]
    assert all(s in subjects for s in sids), \
        f"--subjects 不在 manifest 内: {[s for s in sids if s not in subjects]}"
    assert all(s in test_ids for s in sids), \
        (f"--subjects 须属 fold{args.fold} test（{test_ids}），"
         f"越界: {[s for s in sids if s not in test_ids]}（防训练折泄漏）")
    trials = _parse_trials(args.trials)

    # 归一化：train 折前 window_len 点统计（与训练入口同一实现，严格 held-out）
    mean, std = compute_mild_train_norm(train_f, window_len=args.window_len)
    norm_stats = (mean, std)
    files_hash = _train_files_hash(train_f)
    norm_hash = hashlib.sha256(
        np.concatenate([mean.ravel(), std.ravel()]).tobytes()).hexdigest()

    # 模型：stage2 mexhat-graph 架构 + strict 加载（架构与 ckpt 不一致即拒跑；
    # --flow-exit-gnn 透传出口 GNN 变体架构，与 ckpt 训练口径匹配）
    model = build_model_mild(device,
                             _model_args(flow_exit_gnn=args.flow_exit_gnn),
                             stage=2,
                             obs_len=args.obs_len,
                             forecast_len=args.forecast_len)
    load_flow_ckpt(model, args.ckpt)

    set_seed(args.seed)
    print(f"[quality] fold{args.fold} subjects={sids} trials={len(trials)} "
          f"K={args.k} seed={args.seed} window={args.window_len} "
          f"(obs {args.obs_len}/fc {args.forecast_len}; trim_tail={TRIM_TAIL})")

    # --- 逐 (subject, trial) 计算 → 按例/总体聚合（float64 统计） ---
    t0 = time.time()
    per_subject = {}
    all_cov_full, all_cov_trim, all_ratio_full, all_ratio_trim = [], [], [], []
    for sid in sids:
        from train_mild_depress import MildTrialDataset
        ds = MildTrialDataset([sid_to_file[sid]], norm_stats,
                              obs_len=args.obs_len,
                              forecast_len=args.forecast_len,
                              window_len=args.window_len)
        cov_f, cov_t, rat_f, rat_t = [], [], [], []
        for ti in trials:
            x_target = ds.trials[ti].numpy()          # (window_len,128) z 域
            real = x_target[args.obs_len:]            # (forecast_len,128)
            fc = sample_forecast(model, ds.trials[ti], args.k, device,
                                 obs_len=args.obs_len)  # (K,forecast_len,128)
            cov_f.append(_coverage(fc, real, trim=0))
            cov_t.append(_coverage(fc, real, trim=TRIM_TAIL))
            rat_f.append(_std_ratio(fc, real, trim=0))
            rat_t.append(_std_ratio(fc, real, trim=TRIM_TAIL))
        per_subject[str(sid)] = {
            "group": groups[sid],
            "n_trials": len(trials),
            "coverage": {
                "full_mean": float(np.mean(cov_f)),
                "full_max_trial": float(np.max(cov_f)),
                "trimmed_mean": float(np.mean(cov_t)),
                "trimmed_max_trial": float(np.max(cov_t)),
            },
            "std_ratio": {
                "full_mean": float(np.mean(rat_f)),
                "full_min": float(np.min(rat_f)),
                "full_max": float(np.max(rat_f)),
                "trimmed_mean": float(np.mean(rat_t)),
                "trimmed_min": float(np.min(rat_t)),
                "trimmed_max": float(np.max(rat_t)),
            },
        }
        all_cov_full.extend(cov_f)
        all_cov_trim.extend(cov_t)
        all_ratio_full.extend(rat_f)
        all_ratio_trim.extend(rat_t)
        print(f"[quality] subj {sid} ({groups[sid]}) done "
              f"({len(trials)} trials, {time.time() - t0:.1f}s elapsed)")

    overall = {
        "coverage": {
            "full_mean": float(np.mean(all_cov_full)),
            "trimmed_mean": float(np.mean(all_cov_trim)),
        },
        "std_ratio": {
            "full_mean": float(np.mean(all_ratio_full)),
            "full_min": float(np.min(all_ratio_full)),
            "full_max": float(np.max(all_ratio_full)),
            "trimmed_mean": float(np.mean(all_ratio_trim)),
            "trimmed_min": float(np.min(all_ratio_trim)),
            "trimmed_max": float(np.max(all_ratio_trim)),
        },
        "n_trials_total": len(all_cov_full),
    }

    meta = {
        "task": "10-24-latent-flow-mild-w500 (caliber first registered "
                "10-21-latent-flow-mild-train Run 004-006; persisted here)",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "caliber": {
            "coverage": "per-timepoint Q05-Q95 band coverage: real future at "
                        "each timepoint & channel inside the K-sample 90% "
                        "quantile band; per-trial mean over (time,channel), "
                        "aggregated per subject (mean/max over trials) and "
                        "overall (mean over all trials)",
            "std_ratio": "medoid sample (viz_mild.medoid_index, actual "
                         "generated sample, no statistical reconstruction) "
                         "per-channel std / real-future per-channel std; "
                         "per-trial mean over channels, per-subject "
                         "mean/min/max over trials; medoid selected within "
                         "each caliber's segment (full / trimmed) — under "
                         "deterministic sampling all K samples are bitwise "
                         "identical so the choice is immaterial",
            "trim_tail_points": TRIM_TAIL,
            "trim_note": f"trimmed caliber drops the last {TRIM_TAIL} points "
                         "(registered fixed absolute count, NOT scaled by "
                         "forecast_len; share of fc window 6.7% at 375 -> "
                         "20% at 125, cross-caliber numbers not comparable)",
            "windows": {"full": args.forecast_len,
                        "trimmed": args.forecast_len - TRIM_TAIL},
            "deterministic_sampling_note":
                "sample() 全确定性（z_obs=mu + 确定性 Euler 10 步 + 确定性 "
                "decoder，无后验采样随机源）⇒ K 条样本逐位相同 ⇒ 分位带退化"
                "为单点、coverage 结构性 ≈ 0，不携带质量信息；判读以 "
                "medoid/real std 比 + 波形人工判为主（用户 2026-10-24 拍板）",
            "flow_exit_gnn": bool(args.flow_exit_gnn),
            "flow_exit_gnn_note":
                "--flow-exit-gnn 装载旗标（10-24-latent-flow-exit-gnn）：True = "
                "出口 GNN 变体 ckpt（FlowVelocityNetExitGNN，出口 K 轮共享 "
                "A_eff 消息传递 + 零初始化 exit_out）；False = 原架构；与 ckpt "
                "训练口径不一致时 strict load 拒跑",
            "euler_steps": 10,
        },
        "fold": args.fold,
        "ckpt": args.ckpt,
        "k": args.k,
        "seed": args.seed,
        "window_len": args.window_len,
        "obs_len": args.obs_len,
        "forecast_len": args.forecast_len,
        "manifest": {"path": args.manifest,
                     "sha256": _sha256_file(args.manifest),
                     "subjects_used": sids, "trials": trials},
        "norm": {"source": f"train fold first-{args.window_len} samples "
                           f"(same implementation as training entry)",
                 "files_hash": files_hash, "mean_std_sha256": norm_hash},
        "device": str(device),
        "subjects": per_subject,
        "overall": overall,
    }
    meta_path = out_dir / "quality_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    # --- stdout 摘要表（数字回填 experiment.md 用） ---
    print(f"\n[quality] === 摘要（fold{args.fold}, K={args.k}, seed={args.seed}, "
          f"window={args.window_len}, obs/fc={args.obs_len}/{args.forecast_len}, "
          f"trim_tail={TRIM_TAIL}） ===")
    print(f"{'sid':>4} {'grp':>3}  {'cov_full':>9} {'cov_trim':>9}  "
          f"{'std_full(mn/mx)':>17} {'std_trim(mn/mx)':>17}")
    for sid in sids:
        s = per_subject[str(sid)]
        c, r = s["coverage"], s["std_ratio"]
        print(f"{sid:>4} {s['group']:>3}  "
              f"{c['full_mean']:>9.4f} {c['trimmed_mean']:>9.4f}  "
              f"{r['full_mean']:>7.3f}({r['full_min']:.2f}/{r['full_max']:.2f})"
              f" {r['trimmed_mean']:>7.3f}({r['trimmed_min']:.2f}/"
              f"{r['trimmed_max']:.2f})")
    o = overall
    print(f"{'ALL':>4} {'':>3}  "
          f"{o['coverage']['full_mean']:>9.4f} "
          f"{o['coverage']['trimmed_mean']:>9.4f}  "
          f"{o['std_ratio']['full_mean']:>7.3f}({o['std_ratio']['full_min']:.2f}/"
          f"{o['std_ratio']['full_max']:.2f}) "
          f"{o['std_ratio']['trimmed_mean']:>7.3f}"
          f"({o['std_ratio']['trimmed_min']:.2f}/"
          f"{o['std_ratio']['trimmed_max']:.2f})")
    print("[quality] 注意：确定性采样 ⇒ K 条逐位相同，coverage 结构性 ≈ 0 "
          "不携带质量信息；判读以 medoid/real std 比 + 波形为主")
    print(f"[quality] meta -> {meta_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
