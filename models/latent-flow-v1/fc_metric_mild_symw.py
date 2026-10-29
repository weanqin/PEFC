#!/usr/bin/env python3
"""fc_metric_mild_symw.py — symw 五折 test 集 gen-fc vs real-fc Pearson 一致性（指标 1 v2.1 镜像）.

任务 10-29-symw-fc-5fold。口径镜像 models/gpfm-mamba-v1/fc_metric.py（指标 1
v2.1，用户 2026-10-12 指定；通过标准组平均 Pearson r ≥ 0.95 = 2026-10-16 用户
拍板，见 spec/research/surrogate-brain-qualification.md）：
  - gen/real 两侧完全相同流程：逐被试 30 个 fc 段拼接 → (128, 3750) float64
    → np.corrcoef → 128×128 Pearson FC；
  - 点估计 = 组内被试平均矩阵上三角（k=1，8128 边）展平 → Pearson r +
    Spearman ρ；显著性 = 逐被试一致性 r_s/ρ_s 单样本双侧 t 检验（df=S−1）。

数据与模型口径（w500 契约 + result 注册件，全只读）：
  - 每折 test 8 例 × 30 trial，fc 段 = trial 前 500 点的第 375–500 点
    （125 点 @250 Hz，128 ch；数据 x_f 字段不参与）；
  - ckpt = result/checkpoints/mild/latent_flow_symw/fold{k}/stage2 best
    （SHA256 对拍 result/checkpoints/MANIFEST.md 登记值；result 目录只读，
    result-dir-policy）；
  - 采样 = viz_mild.sample_forecast 全确定性（z_obs=μ + 确定性 Euler 10 步 +
    确定性 decoder，无后验采样随机源）⇒ K=1（K 条逐位相同）；
  - 正式口径 device = CPU 单线程（GPU 前向受同卡并发进程 cuBLASLt 核选择
    漂移影响不可逐位复现——CUBLAS_WORKSPACE_CONFIG + deterministic
    algorithms 双开关后仍逐次不同，探针存证 2026-10-29；CPU 跨进程逐位
    一致，指标数值受 device 影响约 1e-5）；
  - FC 计算域 = z 域（train 折逐通道统计）；Pearson 对逐通道仿射不变 ⇒ 与
    原始 μV 域数学等价——每折首个 test 被试双阈值断言守卫（float64 重算
    < 1e-12；float32 管线 < 1e-6）。

四重守卫（任一不符拒跑）：
  ① manifest SHA256 == ec_mild_unbiased.EXPECTED_MANIFEST_SHA；
  ② 每折 norm SHA == ec_mild_unbiased.EXPECTED_NORM_SHA_BY_FOLD[fold]；
  ③ 每折 ckpt SHA == CKPT_SHA256_BY_FOLD（result MANIFEST 登记值誊录）；
  ④ ckpt 文件名 fold{k}_ 前缀与 --fold 锁定（防折-权重错配）。

模式：
  --mode fold --fold k --out-root DIR    单折核算 → {out-root}/fold{k}/
      （fc_metric.json + fc_matrices.npz；gen 样本不落盘，逐被试样本
       SHA256 随档登记供确定性复跑对拍）
  --mode summary --out-root DIR          读五折 JSON → summary_5fold.json + README.md
  --mode determinism --fold 0 --out-root DIR   重跑采样对拍 fc_metric.json
      登记的 gen 样本 SHA 与全部数值字段（AC5；fold 任意可选）

Run（正式）：
  for k in 0 1 2 3 4; do
    python3 /root/npi/models/latent-flow-v1/fc_metric_mild_symw.py --mode fold \
      --fold $k --out-root /root/npi/outputs/10-29-symw-fc-5fold; done
  python3 /root/npi/models/latent-flow-v1/fc_metric_mild_symw.py --mode summary \
    --out-root /root/npi/outputs/10-29-symw-fc-5fold
  python3 /root/npi/models/latent-flow-v1/fc_metric_mild_symw.py --mode determinism \
    --fold 0 --out-root /root/npi/outputs/10-29-symw-fc-5fold
"""
from __future__ import annotations

import os

# --- BLAS/torch 线程 = 1（必须在任何 numpy/scipy/torch 导入之前）---
# 本指标脚本负载微小（corrcoef 128×3750 等），换取跨进程逐位复现（AC5）。
# GPU 前向已实测不可逐位复现（同卡并发训练进程改变显存分配→指针对齐→
# cuBLASLt 核选择漂移；CUBLAS_WORKSPACE_CONFIG + use_deterministic_algorithms
# 双开关后仍逐次不同，2026-10-29 探针存证）⇒ 正式口径 = CPU 单线程。
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)   # CPU 单线程 = 逐位复现（见顶部注释）

VARIANT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(VARIANT_DIR))  # 使 latent_flow 与同目录脚本可导入

from ec_mild_unbiased import (  # noqa: E402
    EXPECTED_MANIFEST_SHA,
    EXPECTED_NORM_SHA_BY_FOLD,
    W500_FC,
    W500_OBS,
    W500_WINDOW,
)
from latent_flow.training.seeding import set_seed  # noqa: E402
from train_mild_depress import (  # noqa: E402
    MANIFEST_DEFAULT,
    TRIALS,
    MildTrialDataset,
    _sha256_file,
    build_model_mild,
    compute_mild_train_norm,
    load_manifest,
)
from viz_mild import _model_args, load_flow_ckpt, sample_forecast  # noqa: E402

# result/checkpoints/MANIFEST.md「合格注册」表 stage2 flow 行誊录（fold{k}_
# mild_flow_best_graph_mexhat.pt；result 目录只读，SHA 不符即拒跑）
CKPT_SHA256_BY_FOLD = {
    0: "43564dd326ad7b209ecba367f37b68362af1514f617b27b71da1c67893d6654d",
    1: "06eb59fe35c53693107cc06d3614cfa5698e9ce0c2cd616bafee3d029d97e4a0",
    2: "4aaad72eab5a92433ac5674a8e9b005ea04428d355d0e6454b46b9e5f7f5303d",
    3: "421edf87a191421a1babe3a8dce3cb7a4d78fff132a9e0fa32804767ddde6c44",
    4: "3d8a4e974dfe40e2a462c71a0bbe1e86446f3fe9ebd6f8883c6dc005da2caa2c",
}
CKPT_ROOT_DEFAULT = Path("/root/npi/result/checkpoints/mild/latent_flow_symw")
SEED, K_SAMPLES = 42, 1          # 采样全确定性 ⇒ K=1（K 条逐位相同）
N_EDGES_UPPER = 128 * 127 // 2   # 8128（上三角 k=1，对角剔除）


# ================= 指标纯函数（镜像 gpfm-mamba-v1/fc_metric.py，逐行对齐） =================
# 布局适配：fc_metric 输入 (N,C,T)，本线数据/生成侧为 (N,T,C)——transpose 相应
# 调整，公式（拼接→float64→corrcoef）不变。

def subject_fc(x_subj: np.ndarray) -> np.ndarray:
    """单个被试全部 fc 段拼接后的 128×128 Pearson 相关（float64）。

    x_subj (TRIALS, W500_FC, 128) → (128, TRIALS*W500_FC) → corrcoef。"""
    x = x_subj.transpose(2, 0, 1).reshape(x_subj.shape[2], -1).astype(np.float64)
    return np.corrcoef(x)


def tri_flat(m: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(m.shape[0], k=1)
    return m[iu]


def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def spearman_rho(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return pearson_r(ra, rb)


def one_sample_ttest(x: np.ndarray):
    """单样本双侧 t 检验（H0: 均值=0）；返回 (t, df, p, mean, sd)。"""
    n = len(x)
    m = float(x.mean())
    sd = float(x.std(ddof=1))
    t = m / (sd / np.sqrt(n))
    return t, n - 1, t_two_sided_p(t, n - 1), m, sd


# ---- p 值：正则化不完全 Beta 连分式（镜像 gpfm-mamba-v1/gc_psd.py 同名实现，
# ---- 避免跨变体 import；数值路径与参考指标实现逐位一致） ----

def _betacf(a: float, b: float, x: float, max_iter: int = 600, eps: float = 3e-12) -> float:
    """正则化不完全 Beta 的连分式（Modified Lentz）。"""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """正则化不完全 Beta I_x(a,b)。"""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lfront = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
              + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lfront) * _betacf(a, b, x) / a
    lcomplement = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                   + b * math.log1p(-x) + a * math.log(x))
    return 1.0 - math.exp(lcomplement) * _betacf(b, a, 1.0 - x) / b


def t_two_sided_p(t: float, nu: float) -> float:
    """双侧 t 检验 p：p = I_{nu/(nu+t^2)}(nu/2, 1/2)。"""
    if t == 0:
        return 1.0
    return betainc(nu / 2.0, 0.5, nu / (nu + t * t))


# ================= 守卫与数据装载 =================

def _guarded_context(fold: int, manifest: str, ckpt_root: Path, device):
    """守卫 ①manifest SHA ②norm SHA ③ckpt SHA ④fold 前缀 → 返回运行上下文。"""
    man_sha = _sha256_file(manifest)
    assert man_sha == EXPECTED_MANIFEST_SHA, \
        f"manifest SHA 不符: {man_sha} != {EXPECTED_MANIFEST_SHA}"
    man, subjects, train_f, _val_f, _test_f, fold_dict, _trial_rec = \
        load_manifest(manifest, fold)
    test_ids = sorted(fold_dict["test"])
    assert len(test_ids) == 8, f"fold{fold} test != 8: {test_ids}"

    mean, std = compute_mild_train_norm(train_f, window_len=W500_WINDOW)
    norm_sha = hashlib.sha256(
        np.concatenate([mean.ravel(), std.ravel()]).tobytes()).hexdigest()
    assert norm_sha == EXPECTED_NORM_SHA_BY_FOLD[fold], \
        (f"fold{fold} norm SHA 不符: {norm_sha} != "
         f"{EXPECTED_NORM_SHA_BY_FOLD[fold]}")

    ckpt = ckpt_root / f"fold{fold}" / f"fold{fold}_mild_flow_best_graph_mexhat.pt"
    assert ckpt.exists(), f"ckpt not found: {ckpt}"
    assert ckpt.name.startswith(f"fold{fold}_"), ckpt.name    # 守卫④（结构+复核）
    ckpt_sha = _sha256_file(str(ckpt))
    assert ckpt_sha == CKPT_SHA256_BY_FOLD[fold], \
        f"fold{fold} ckpt SHA 不符: {ckpt_sha} != {CKPT_SHA256_BY_FOLD[fold]}"

    model = build_model_mild(device, _model_args(flow_exit_gnn=True),
                             stage=2, obs_len=W500_OBS, forecast_len=W500_FC)
    load_flow_ckpt(model, str(ckpt))                          # strict 装载
    return model, subjects, test_ids, (mean, std), str(ckpt), ckpt_sha, norm_sha


# ================= 单折核算（--mode fold / determinism 共用采样路径） =================

def run_fold_sampling(fold: int, manifest: str, ckpt_root: Path, device):
    """守卫 + 全量采样 → 逐被试 FC/SHA + 双域等价断言 + 组统计。

    返回 (payload_dict, fc_matrices_dict)。不落盘（落盘由调用侧决定）。"""
    model, subjects, test_ids, norm_stats, ckpt, ckpt_sha, norm_sha = \
        _guarded_context(fold, manifest, ckpt_root, device)
    sid_to_file = {sid: v["file"] for sid, v in subjects.items()}
    groups = {sid: v["group"] for sid, v in subjects.items()}
    set_seed(SEED)
    print(f"[fc] fold{fold} test={test_ids} window={W500_WINDOW} "
          f"(obs {W500_OBS}/fc {W500_FC}) K={K_SAMPLES} seed={SEED} "
          f"ckpt={ckpt}")

    t0 = time.time()
    fc_gen_s, fc_real_s, subj_detail = [], [], []
    for sid in test_ids:
        ds = MildTrialDataset([sid_to_file[sid]], norm_stats,
                              obs_len=W500_OBS, forecast_len=W500_FC,
                              window_len=W500_WINDOW)
        assert len(ds.trials) == TRIALS, f"sid {sid}: {len(ds.trials)} != {TRIALS}"
        gen_segs, real_segs = [], []
        for ti in range(TRIALS):
            x_target = ds.trials[ti].numpy()                   # (500,128) z 域
            real_segs.append(x_target[W500_OBS:])             # (125,128)
            fc = sample_forecast(model, ds.trials[ti], K_SAMPLES, device,
                                 obs_len=W500_OBS)             # (K,125,128)
            gen_segs.append(fc[0])
        gen_subj = np.stack(gen_segs)                          # (30,125,128)
        real_subj = np.stack(real_segs)
        fc_gen_s.append(subject_fc(gen_subj))
        fc_real_s.append(subject_fc(real_subj))
        subj_detail.append({
            "sid": int(sid), "group": groups[sid],
            "pearson_r_s": None, "spearman_rho_s": None,       # 组统计后回填
            "gen_sample_sha256": hashlib.sha256(gen_subj.tobytes()).hexdigest(),
            "real_sample_sha256": hashlib.sha256(real_subj.tobytes()).hexdigest(),
        })
        print(f"[fc]   subj {sid} ({groups[sid]}) done "
              f"({time.time() - t0:.1f}s elapsed)")
    fc_gen_s = np.stack(fc_gen_s)                              # (8,128,128) f64
    fc_real_s = np.stack(fc_real_s)

    # --- 双域等价断言（AC4；每折首个 test 被试，real 侧独立重算） ---
    import scipy.io as sio
    m = sio.loadmat(sid_to_file[test_ids[0]])
    x_raw = np.concatenate([m["x_p"], m["x_f"]], axis=2)       # (30,128,1500) f64
    seg_raw = x_raw[:, :, W500_OBS:W500_WINDOW].transpose(0, 2, 1)  # (30,125,128)
    mean, std = norm_stats
    mean64 = mean.reshape(1, 1, -1).astype(np.float64)         # (128,1)→(1,1,128)
    std64 = std.reshape(1, 1, -1).astype(np.float64)
    seg_z64 = (seg_raw - mean64) / std64
    fc_raw64 = subject_fc(seg_raw)
    fc_z64 = subject_fc(seg_z64)
    d_math = float(np.abs(fc_z64 - fc_raw64).max())            # 纯 float64 舍入
    d_pipe = float(np.abs(fc_real_s[0] - fc_raw64).max())      # 含 float32 量化
    assert d_math < 1e-12, f"z/原始域数学等价破坏: {d_math}"
    assert d_pipe < 1e-6, f"z/原始域管线等价超容差: {d_pipe}"

    # --- 组统计（点估计 + 逐被试 t 检验，镜像 fc_metric.py v2.1） ---
    FC_gen, FC_real = fc_gen_s.mean(axis=0), fc_real_s.mean(axis=0)
    a, b = tri_flat(FC_gen), tri_flat(FC_real)
    r, rho = pearson_r(a, b), spearman_rho(a, b)
    r_s = np.array([pearson_r(tri_flat(fc_gen_s[i]), tri_flat(fc_real_s[i]))
                    for i in range(len(test_ids))])
    rho_s = np.array([spearman_rho(tri_flat(fc_gen_s[i]), tri_flat(fc_real_s[i]))
                      for i in range(len(test_ids))])
    for i, d in enumerate(subj_detail):
        d["pearson_r_s"] = round(float(r_s[i]), 5)
        d["spearman_rho_s"] = round(float(rho_s[i]), 5)
    t_r, df_r, p_r, r_mean, r_sd = one_sample_ttest(r_s)
    t_rho, df_rho, p_rho, rho_mean, rho_sd = one_sample_ttest(rho_s)

    payload = {
        "metric": ("指标 1 v2.1：Pearson FC 矩阵一致性 — symw 五折 test "
                   "（镜像 gpfm-mamba-v1/fc_metric.py；symw = 10-25-md-lf-"
                   "anti0.2-gnnexit-symw，anti-uniform λ0.2 + exit GNN + 每步 W 对称化）"),
        "task": "10-29-symw-fc-5fold",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "fold": fold,
        "n_subjects": len(test_ids),
        "n_trials_per_subject": TRIALS,
        "fc_len": W500_FC,
        "n_channels": 128,
        "n_edges_upper": N_EDGES_UPPER,
        "pearson_r": round(r, 5),
        "spearman_rho": round(rho, 5),
        "significance": {
            "method": "one-sample two-sided t-test on per-subject consistency (H0: mean=0)",
            "pearson_r_per_subject_mean": round(r_mean, 5),
            "pearson_r_per_subject_sd": round(r_sd, 5),
            "pearson_t": round(t_r, 4),
            "pearson_df": int(df_r),
            "pearson_p": p_r,
            "spearman_rho_per_subject_mean": round(rho_mean, 5),
            "spearman_rho_per_subject_sd": round(rho_sd, 5),
            "spearman_t": round(t_rho, 4),
            "spearman_df": int(df_rho),
            "spearman_p": p_rho,
        },
        "subjects": subj_detail,
        "caliber": {
            "fc_segment": f"per-trial points [{W500_OBS},{W500_WINDOW}) of the "
                          f"{W500_WINDOW}-pt prefix window (x_f field unused), "
                          f"{W500_FC} pts @250 Hz",
            "subject_fc": "concat 30 fc segments -> (128, 3750) float64 -> "
                          "np.corrcoef; identical pipeline for gen/real",
            "point_estimate": "group-mean matrices, upper triangle k=1 "
                              f"({N_EDGES_UPPER} edges) flattened",
            "sampling": "viz_mild.sample_forecast, fully deterministic "
                        "(z_obs=mu + deterministic Euler 10 steps + decoder; "
                        "no posterior sampling randomness) => K=1, all K "
                        "bitwise identical; seed 42 set and registered",
            "compute_domain": "z-domain (train-fold per-channel stats); Pearson "
                              "invariant to per-channel affine => equivalent to "
                              "raw uV domain (asserted)",
            "pass_criterion": "group-mean Pearson r >= 0.95 (user 2026-10-16); "
                              "auxiliary: per-subject t-test p < 0.05",
        },
        "guards": {
            "manifest_sha256": EXPECTED_MANIFEST_SHA,
            "norm_sha256": norm_sha,
            "ckpt_sha256": ckpt_sha,
            "domain_equivalence": {
                "subject": int(test_ids[0]),
                "max_abs_diff_float64_recompute": d_math,
                "max_abs_diff_float32_pipeline": d_pipe,
                "thresholds": "<1e-12 / <1e-6",
            },
        },
        "ckpt": ckpt,
        "ckpt_provenance": "result/checkpoints/MANIFEST.md 合格注册件（只读）",
        "device": str(device),
        "determinism_note": (
            "canonical caliber = CPU single-thread (BLAS/torch threads=1): "
            "bitwise reproducible across processes; GPU forward rejected — "
            "concurrent-process memory layout shifts cuBLASLt kernel choice "
            "even with CUBLAS_WORKSPACE_CONFIG + use_deterministic_algorithms "
            "(probe 2026-10-29); metric impact of device choice ~1e-5"),
        "elapsed_s": round(time.time() - t0, 2),
    }
    matrices = {"FC_gen_group": FC_gen, "FC_real_group": FC_real,
                "FC_gen_subj": fc_gen_s, "FC_real_subj": fc_real_s}
    return payload, matrices


# ================= 三种模式 =================

def mode_fold(args) -> int:
    out_dir = Path(args.out_root) / f"fold{args.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload, matrices = run_fold_sampling(
        args.fold, args.manifest, Path(args.ckpt_root), args.device_obj)
    (out_dir / "fc_metric.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    np.savez(out_dir / "fc_matrices.npz", **matrices)
    s = payload["significance"]
    print(f"\n[fc] === fold{args.fold} 摘要 ===")
    print(f"group r={payload['pearson_r']} rho={payload['spearman_rho']} | "
          f"subj r_s {s['pearson_r_per_subject_mean']}±{s['pearson_r_per_subject_sd']} "
          f"t={s['pearson_t']} p={s['pearson_p']:.3e} | 等价性 "
          f"{payload['guards']['domain_equivalence']['max_abs_diff_float64_recompute']:.2e}/"
          f"{payload['guards']['domain_equivalence']['max_abs_diff_float32_pipeline']:.2e}")
    print(f"[fc] -> {out_dir}/fc_metric.json + fc_matrices.npz")
    return 0


def mode_determinism(args) -> int:
    """AC5：重跑整折采样，对拍 fc_metric.json 登记的 gen 样本 SHA + 全部数值字段。"""
    ref_path = Path(args.out_root) / f"fold{args.fold}" / "fc_metric.json"
    assert ref_path.exists(), f"参照件不存在: {ref_path}"
    ref = json.loads(ref_path.read_text())
    payload, _ = run_fold_sampling(
        args.fold, args.manifest, Path(args.ckpt_root), args.device_obj)

    fails = []
    for i, d in enumerate(payload["subjects"]):
        r = ref["subjects"][i]
        if d["sid"] != r["sid"] or d["gen_sample_sha256"] != r["gen_sample_sha256"]:
            fails.append(f"subj[{i}] sid/gen SHA 不符: {d['sid']} vs {r['sid']}, "
                         f"{d['gen_sample_sha256'][:12]} vs {r['gen_sample_sha256'][:12]}")
        for k in ("pearson_r_s", "spearman_rho_s"):
            if d[k] != r[k]:
                fails.append(f"subj[{i}].{k}: {d[k]} != {r[k]}")
    for k in ("pearson_r", "spearman_rho"):
        if payload[k] != ref[k]:
            fails.append(f"{k}: {payload[k]} != {ref[k]}")
    for k in ("pearson_r_per_subject_mean", "pearson_r_per_subject_sd",
              "pearson_t", "spearman_rho_per_subject_mean",
              "spearman_rho_per_subject_sd", "spearman_t"):
        if payload["significance"][k] != ref["significance"][k]:
            fails.append(f"significance.{k}: {payload['significance'][k]} "
                         f"!= {ref['significance'][k]}")
    for k in ("pearson_p", "spearman_p"):
        if not math.isclose(payload["significance"][k], ref["significance"][k],
                            rel_tol=0, abs_tol=1e-12):
            fails.append(f"significance.{k}: {payload['significance'][k]} "
                         f"!= {ref['significance'][k]}")

    if fails:
        print(f"[determinism] fold{args.fold} 复跑不一致 ✗")
        for f_ in fails:
            print(f"  - {f_}")
        return 1
    print(f"[determinism] fold{args.fold} 复跑逐位一致 ✓ "
          f"（8/8 gen 样本 SHA + 组点估计 + 逐被试 r_s/ρ_s + t/p 全对拍）")
    return 0


def mode_summary(args) -> int:
    """五折并置 + R4 三项汇总 + 判定陈述（对照指标 1 通过标准）。"""
    root = Path(args.out_root)
    folds = []
    for k in range(5):
        p = root / f"fold{k}" / "fc_metric.json"
        assert p.exists(), f"缺 {p}（先跑满五折 --mode fold）"
        folds.append(json.loads(p.read_text()))

    per_fold = []
    all_r_s, all_rho_s = [], []
    for f_ in folds:
        s = f_["significance"]
        per_fold.append({
            "fold": f_["fold"],
            "group_pearson_r": f_["pearson_r"],
            "group_spearman_rho": f_["spearman_rho"],
            "pearson_r_per_subject_mean": s["pearson_r_per_subject_mean"],
            "pearson_r_per_subject_sd": s["pearson_r_per_subject_sd"],
            "pearson_t": s["pearson_t"], "pearson_p": s["pearson_p"],
            "spearman_rho_per_subject_mean": s["spearman_rho_per_subject_mean"],
            "spearman_rho_per_subject_sd": s["spearman_rho_per_subject_sd"],
            "spearman_t": s["spearman_t"], "spearman_p": s["spearman_p"],
            "subjects": f_["subjects"],
        })
        all_r_s.extend(d["pearson_r_s"] for d in f_["subjects"])
        all_rho_s.extend(d["spearman_rho_s"] for d in f_["subjects"])
    all_r_s, all_rho_s = np.array(all_r_s), np.array(all_rho_s)
    t_r, df_r, p_r, r_mean, r_sd = one_sample_ttest(all_r_s)
    t_rho, df_rho, p_rho, rho_mean, rho_sd = one_sample_ttest(all_rho_s)

    avg = {
        "group_point_estimates_mean": {
            "pearson_r": round(float(np.mean([f["pearson_r"] for f in folds])), 5),
            "spearman_rho": round(float(np.mean([f["spearman_rho"] for f in folds])), 5),
        },
        "subjects_pooled_40": {
            "pearson_r_mean": round(r_mean, 5), "pearson_r_sd": round(r_sd, 5),
            "pearson_t": round(t_r, 4), "pearson_df": int(df_r), "pearson_p": p_r,
            "spearman_rho_mean": round(rho_mean, 5),
            "spearman_rho_sd": round(rho_sd, 5),
            "spearman_t": round(t_rho, 4), "spearman_df": int(df_rho),
            "spearman_p": p_rho,
        },
        "per_fold_subject_mean_average": {
            "pearson_r": round(float(np.mean(
                [f["significance"]["pearson_r_per_subject_mean"] for f in folds])), 5),
            "spearman_rho": round(float(np.mean(
                [f["significance"]["spearman_rho_per_subject_mean"] for f in folds])), 5),
        },
    }
    fold_rs = [f["pearson_r"] for f in folds]
    judgment = {
        "pass_criterion": "指标 1 v2.1：组平均 Pearson r ≥ 0.95（辅助：逐被试 t 检验 p<0.05）",
        "folds_passing_r": sum(1 for r in fold_rs if r >= 0.95),
        "max_group_r_across_folds": round(max(fold_rs), 5),
        "statement": None,      # 下方回填
    }
    judgment["statement"] = (
        f"五折组平均 Pearson r = {fold_rs}（均值 {avg['group_point_estimates_mean']['pearson_r']}），"
        f"最高 {judgment['max_group_r_across_folds']} "
        f"{'<' if judgment['max_group_r_across_folds'] < 0.95 else '>='} 0.95 通过线 ⇒ "
        + ("全部折未达指标 1 通过标准" if judgment["folds_passing_r"] == 0
           else f"{judgment['folds_passing_r']}/5 折达标")
        + f"；合并 40 例被试级 r_s = {avg['subjects_pooled_40']['pearson_r_mean']}"
          f"±{avg['subjects_pooled_40']['pearson_r_sd']}，"
          f"t={avg['subjects_pooled_40']['pearson_t']}，"
          f"p={avg['subjects_pooled_40']['pearson_p']:.3e}"
        + ("（p<0.05，辅助口径显著）" if avg["subjects_pooled_40"]["pearson_p"] < 0.05
           else "（p≥0.05，辅助口径不显著）")
        + "。本判定属 Surrogate Brain Validity（Q2），与预测质量（Q1）分开陈述。")

    summary = {
        "metric": folds[0]["metric"],
        "task": "10-29-symw-fc-5fold",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model_variant": "mild-latent-flow-symw-anti0.2-gnn-exit（10-25-md-lf-anti0.2-gnnexit-symw）",
        "caliber": folds[0]["caliber"],
        "per_fold": per_fold,
        "five_fold_average": avg,
        "judgment": judgment,
    }
    (root / "summary_5fold.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    # README 五折并置表
    lines = [
        "# symw 五折 test 集 gen-fc vs real-fc Pearson 一致性（指标 1 v2.1）",
        "",
        f"任务 10-29-symw-fc-5fold · 变体 {summary['model_variant']} · "
        f"ckpt = result/checkpoints/mild/latent_flow_symw/（注册件只读）",
        "",
        "口径：逐被试 30 个 fc 段（w500 口径 trial 第 375–500 点，125 pts @250 Hz）"
        "拼接 → (128,3750) float64 → Pearson FC；gen/real 同流程；"
        "点估计 = 组平均矩阵上三角（8128 边）Pearson r / Spearman ρ；"
        "显著性 = 逐被试 r_s/ρ_s 单样本双侧 t 检验（df=7/折；汇总合并 40 例 df=39）。",
        "",
        "| fold | 组 r | 组 ρ | 被试级 r_s (mean±sd) | t | p | 被试级 ρ_s (mean±sd) | t | p |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for f_, d in zip(per_fold, folds):
        s = d["significance"]
        lines.append(
            f"| {f_['fold']} | {f_['group_pearson_r']} | {f_['group_spearman_rho']} | "
            f"{f_['pearson_r_per_subject_mean']}±{f_['pearson_r_per_subject_sd']} | "
            f"{f_['pearson_t']} | {f_['pearson_p']:.3e} | "
            f"{f_['spearman_rho_per_subject_mean']}±{f_['spearman_rho_per_subject_sd']} | "
            f"{f_['spearman_t']} | {f_['spearman_p']:.3e} |")
    a_ = avg["group_point_estimates_mean"]
    p40 = avg["subjects_pooled_40"]
    lines += [
        f"| **5-fold 平均** | **{a_['pearson_r']}** | **{a_['spearman_rho']}** | "
        f"合并40例 {p40['pearson_r_mean']}±{p40['pearson_r_sd']} | {p40['pearson_t']} | "
        f"{p40['pearson_p']:.3e} | 合并40例 {p40['spearman_rho_mean']}±{p40['spearman_rho_sd']} | "
        f"{p40['spearman_t']} | {p40['spearman_p']:.3e} |",
        "",
        f"**判定**：{judgment['statement']}",
        "",
        "产物：`fold{k}/fc_metric.json`（口径+数值+凭据）、`fold{k}/fc_matrices.npz`"
        "（组平均+逐被试 FC）、`summary_5fold.json`；守卫（manifest/norm/ckpt SHA、"
        "z↔原始域等价）见各折 JSON `guards` 块。",
    ]
    (root / "README.md").write_text("\n".join(lines) + "\n")
    print(f"[summary] -> {root}/summary_5fold.json + README.md")
    print(f"[summary] {judgment['statement']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="symw 五折 test 集 gen-fc vs real-fc Pearson 一致性（指标 1 v2.1 镜像）")
    ap.add_argument("--mode", choices=["fold", "summary", "determinism"],
                    default="fold")
    ap.add_argument("--fold", type=int, default=0, choices=[0, 1, 2, 3, 4])
    ap.add_argument("--manifest", type=str, default=str(MANIFEST_DEFAULT))
    ap.add_argument("--ckpt-root", type=str, default=str(CKPT_ROOT_DEFAULT))
    ap.add_argument("--out-root", type=str, required=True,
                    help="任务输出根（必填，无默认——防误写旧任务输出树），"
                         "如 /root/npi/outputs/10-29-symw-fc-5fold")
    ap.add_argument("--device", type=str, default="cpu",
                    help="默认 cpu（正式口径：GPU 前向受同卡并发进程 cuBLASLt "
                         "核选择漂移影响不可逐位复现；CPU 单线程逐位一致，"
                         "探针存证 2026-10-29）")
    args = ap.parse_args()
    args.device_obj = torch.device(args.device)
    if args.mode == "fold":
        return mode_fold(args)
    if args.mode == "determinism":
        return mode_determinism(args)
    return mode_summary(args)


if __name__ == "__main__":
    sys.exit(main())
