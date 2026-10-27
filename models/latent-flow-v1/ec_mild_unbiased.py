#!/usr/bin/env python3
"""ec_mild_unbiased.py — 无偏噪声扰动归因 EC（global-τ 两阶段）× latent-flow mild-w500.

任务 10-24-mild-w500-ec-unbiased（design.md 契约实现）。口径钉死（用户 2026-10-24
拍板，见 consent.md）：q=0.5 不做其他取值测试；主跑 fold0 test 8 例 × 30 trial
全量；扰动 = σ·ξ（μ=0）逐点加在 z-score 后、入模前 x_obs(375) 源通道；σ_ref=0.3；
网格 log[0.01,0.48]×24；conn = −(ln σ* + 0.5·ln 2πe) + 5/7（v4 交付公式，用户
2026-10-27 拍板升为线默认；原房内 v1 −(ln σ*+ln2πe)+1 退役为历史口径——v4 = v1
+ 常数平移 ≈ +1.1332，网格上全正 min=+0.0293 ⇒ abs≡raw）。

复用（零拷贝，包体零改动）：bisect_sigma.SignalProbeSite（use_biased_noise=False，
OCLatentFlow 原生契约——确定性采样 ⇒ 基线唯一）/ log_grid / compute_global_tau /
sigma_star_with_global_tau；train_mild_depress 数据管线；viz_mild 加载器。

三模式：main（GPU 主跑，逐例 json + 曲线 npz，json 已存在即跳过=断点续跑）；
summary（离线：分布/方差分解/跨被试稳定性/重映射对照/曲线后置诊断）；
gccorr（离线：refs mild40 GC × EC，|EC|/raw × aligned/flipped × Spearman/Pearson，
协议对齐 gen_unbiased_gc_corr.py 先例，描述性登记不构成代理脑资格判定）。

10-24-exitgnn-ec-unbiased 增量：--flow-exit-gnn 出口 GNN 变体代理脑（默认关 = w500
基座，旧调用面零变化；viz_mild 同款旗标先例）；--task-label 非空时 main payload 新增
顶层 task 字段、summary/gccorr 替换任务名段；逐例 json/npz 原子写（tmp+rename）。

10-25-symw-ec-unbiased 增量：--symw SymWAdam 对称 W 变体代理脑（10-25-latent-flow-
exit-gnn-symw；架构侧等价 --flow-exit-gnn、与之互斥；payload symw 分支含 A_eff 无向
caveat + hparams symw 键；σ*/τ/conn 计算路径零改动，默认关 = 旧调用面零变化）。

10-25-symw-ec-f12 增量：--symw + --fold {1,2} + f12 run_dir（10-25-latent-flow-
symw-f12）新调用面——EXPECTED_NORM_SHA 按 fold 键控（{0,1,2}，f12 值 = 训练侧
configs norm_stats 登记值，缺键拒跑）；f12 ckpt 文件名 fold{k}_ 前缀与 --fold 锁定
（折-权重错配防护，exit-gnn-symw 先例 run_dir 收窄为仅 --fold 0）；payload surrogate
f12 分支（task/best-val/variant/caveat）+ hparams.fold 显式化；σ*/τ/conn 计算路径
逐行零改动，fold0 旧调用面行为不变（旗标关等价对拍兜底）。

τ 分位口径含对角自响应（flatten 全 128×128，与 unbiased 先例一致；评审 A 注明防
与 offdiag-only 口径误比）。

Run:
  OMP_NUM_THREADS=8 /root/miniconda3/bin/python models/latent-flow-v1/ec_mild_unbiased.py \
    --mode main --ckpt .../fold0_mild_flow_best_graph_mexhat.pt \
    --out-dir /root/npi/outputs/10-24-mild-w500-ec-unbiased/ec_main
"""
from __future__ import annotations

import os

# --- BLAS 守卫：必须在任何 numpy/scipy/torch 导入之前（npi 环境契约） ---
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

VARIANT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(VARIANT_DIR))  # 使 latent_flow 与同目录脚本可导入

from latent_flow.attribution.bisect_sigma import (  # noqa: E402
    SignalProbeSite,
    compute_global_tau,
    log_grid,
    sigma_star_with_global_tau,
)
from latent_flow.data.norm_stats import _train_files_hash  # noqa: E402
from latent_flow.scripts._connectivity_driver import (  # noqa: E402
    json_default,
    sanitize_for_json,
)
from latent_flow.training.seeding import set_seed  # noqa: E402
from train_mild_depress import (  # noqa: E402
    MANIFEST_DEFAULT,
    _sha256_file,
    build_model_mild,
    compute_mild_train_norm,
    load_manifest,
)
from viz_mild import _model_args, load_flow_ckpt  # noqa: E402

# --- w500 口径硬三元组（评审 B：防 train_mild_depress 模块常量 1500/1125/375 误用；
#     compute_mild_train_norm 缺省 window_len 是静默错统计路径，故此断言 + hash 对拍双保险）---
W500_WINDOW, W500_OBS, W500_FC = 500, 375, 125
# norm mean/std SHA256 按 fold 键控（10-25-symw-ec-f12）：fold0 = w500/symw-fold0 在档
# 钉死值；fold1/2 = f12 训练侧 configs/stage{1,2}_config.json norm_stats.mean_std_sha256
# 登记（run_symw_fold{1,2} 两 stage 一致，2026-10-25 核）。缺 fold 键 = 拒跑。
EXPECTED_NORM_SHA_BY_FOLD = {
    0: "d184e64772b368418b5d816680af8ed030b20e3ef76867f99ab6d354cac82758",
    1: "0f10b15350f69757d883beca19500342fc8279a71dc327252e495be08e1d3383",
    2: "a2c27a2d2577210c2696e0310b0bd1a792d8b2d530107b9bdf8d100446c3d102",
}
EXPECTED_MANIFEST_SHA = "47987909c04348e57d98e77e1668678a7885109336df90bd0a5a635f865d881b"

DEFAULT_Q = 0.5            # 用户拍板钉死：不做其他取值测试
SIGMA_MIN, SIGMA_MAX, DEFAULT_N_GRID = 0.01, 0.48, 24
SIGMA_REF, SEED, N_STEPS, BATCH = 0.3, 42, 10, 32
_LOG_2PIE = math.log(2.0 * math.pi * math.e)
# conn v4 = −(ln σ* + 0.5·ln2πe) + 5/7（线默认，用户 2026-10-27 拍板；命名与
# recompute_conn_v4.py 对齐；房内 v1 −(ln σ*+ln2πe)+1 退役为历史口径）
HALF_LOG_2PIE = 0.5 * _LOG_2PIE
V4_OFFSET = 5.0 / 7.0
EPS_REMAP = 1e-4
DEFAULT_CKPT = ("/root/npi/outputs/10-24-latent-flow-mild-w500/run_w500_fold0/"
                "checkpoints/fold0_mild_flow_best_graph_mexhat.pt")
REFS_DIR = ("/root/npi/refs/granger_bivariate_spectral/"
            "mild40-trial500-zscore-fs250-fixed8-f1-45-favg")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--mode", required=True, choices=["main", "summary", "gccorr"])
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--manifest", type=str, default=str(MANIFEST_DEFAULT))
    p.add_argument("--ckpt", type=str, default=DEFAULT_CKPT,
                   help="w500 stage2 ckpt（main 模式必在）")
    p.add_argument("--window-len", type=int, default=W500_WINDOW)
    p.add_argument("--obs-len", type=int, default=W500_OBS)
    p.add_argument("--forecast-len", type=int, default=W500_FC)
    p.add_argument("--subjects", type=int, default=8,
                   help="主跑被试数（fold0 test 共 8；mini 干跑可传 1）")
    p.add_argument("--subject-idx", type=int, default=0)
    p.add_argument("--sources", type=int, default=0,
                   help="扰动源通道数（默认全 128；mini 干跑可传 8）")
    p.add_argument("--n-trials", type=int, default=0,
                   help="每例子采样 trial 数（默认全部 30）")
    p.add_argument("--q", type=float, default=DEFAULT_Q)
    p.add_argument("--sigma-ref", type=float, default=SIGMA_REF)
    p.add_argument("--sigma-min", type=float, default=SIGMA_MIN)
    p.add_argument("--sigma-max", type=float, default=SIGMA_MAX)
    p.add_argument("--n-grid", type=int, default=DEFAULT_N_GRID)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--n-steps", type=int, default=N_STEPS)
    p.add_argument("--batch-size", type=int, default=BATCH)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out-dir", type=str, required=True,
                   help="输出目录（必填，无默认——防误写旧任务树）")
    p.add_argument("--main-dir", type=str, default="",
                   help="summary/gccorr 模式：subject_*.json 所在目录（默认 = --out-dir）")
    p.add_argument("--overwrite", action="store_true",
                   help="main 模式：json 已存在也重算（默认跳过=断点续跑）")
    p.add_argument("--flow-exit-gnn", action="store_true",
                   help="出口 GNN 变体架构（10-24-latent-flow-exit-gnn；默认关 = "
                        "w500 基座，旧调用面不变）")
    p.add_argument("--symw", action="store_true",
                   help="SymWAdam 对称 W 变体代理脑（10-25-latent-flow-exit-gnn-"
                        "symw；symw 训练直用 FlowVelocityNetExitGNN 零架构差异，"
                        "故模型构建侧等价 --flow-exit-gnn；与 --flow-exit-gnn 互斥；"
                        "默认关 = w500 基座，旧调用面不变）")
    p.add_argument("--task-label", type=str, default="",
                   help="非空时：main payload 新增顶层 task 字段；summary/gccorr "
                        "替换 task 串任务名段（保留 Run 注记）。默认空 = 现行为")
    return p.parse_args(argv)


def _caliber_asserts(a) -> None:
    assert (a.window_len, a.obs_len, a.forecast_len) == (W500_WINDOW, W500_OBS, W500_FC), \
        (f"口径钉死 {W500_WINDOW}/{W500_OBS}/{W500_FC}（w500 ckpt 训练口径），"
         f"got {a.window_len}/{a.obs_len}/{a.forecast_len}")
    assert a.obs_len + a.forecast_len == a.window_len
    assert Path(a.ckpt).exists(), f"ckpt not found: {a.ckpt}"
    assert not (a.flow_exit_gnn and a.symw), \
        "--flow-exit-gnn 与 --symw 互斥（symw 代理脑只走 --symw 单旗标）"
    if a.flow_exit_gnn:
        # 同名 ckpt 防交叉：w500 与 exit-gnn 两 run_dir 的 stage2 best 文件名相同，
        # 凭 run_dir 段区分（strict load 双保险之外的第一道闸）
        assert "10-24-latent-flow-exit-gnn" in a.ckpt, \
            f"--flow-exit-gnn 须配 exit-gnn run_dir 的 ckpt，got {a.ckpt}"
    elif a.symw:
        # symw run_dir 段守卫（同款防交叉；symw 与 exit-gnn 文件名亦相同）。
        # f12 扩展（10-25-symw-ec-f12）：另接受 10-25-latent-flow-symw-f12 run_dir，
        # 并断言 ckpt 文件名 fold{k}_ 前缀与 --fold 一致（防折-权重错配——各折 norm/
        # 划分不同，fold1 命令误配 fold0/fold2 ckpt 须在装载前被拦）。
        if "10-25-latent-flow-symw-f12" in a.ckpt:
            assert (a.fold in (1, 2)
                    and Path(a.ckpt).name.startswith(f"fold{a.fold}_")), \
                (f"--symw f12 ckpt 须配 --fold {{1,2}} 且文件名以 fold{a.fold}_ "
                 f"前缀开头（折-权重错配防护），got fold={a.fold}, "
                 f"{Path(a.ckpt).name}")
        else:
            assert "10-25-latent-flow-exit-gnn-symw" in a.ckpt and a.fold == 0, \
                ("--symw 须配 symw run_dir 的 ckpt（exit-gnn-symw 仅限 --fold 0；"
                 f"--fold {a.fold} 须配 10-25-latent-flow-symw-f12 的 fold{a.fold}_"
                 f" 前缀 ckpt），got {a.ckpt}")


def _atomic_write_json(path: Path, payload: dict) -> None:
    """tmp + rename 原子写：防长跑中断留半截 json 被断点续跑误判为完成。"""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=2, default=json_default, allow_nan=False)
    os.replace(tmp, path)


def _atomic_write_npz(path: Path, **arrays) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:  # 文件句柄式：绕过 np.savez 的 .npz 后缀补全
        np.savez_compressed(fh, **arrays)
    os.replace(tmp, path)


def _git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True,
                              cwd="/root/npi").stdout.strip()
    except Exception:
        return "unavailable"


def _norm_sha(mean: np.ndarray, std: np.ndarray) -> str:
    return hashlib.sha256(
        np.concatenate([mean.ravel(), std.ravel()]).tobytes()).hexdigest()


# =============================================================================
# main 模式：两阶段 global-τ EC（GPU）
# =============================================================================

def run_main(a) -> None:
    _caliber_asserts(a)
    assert _sha256_file(a.manifest) == EXPECTED_MANIFEST_SHA, "manifest SHA256 与钉死值不符"
    device = torch.device(a.device if a.device
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    man, subjects, train_f, _val, _test, fold_dict, _rec = load_manifest(a.manifest, a.fold)
    groups = {sid: v["group"] for sid, v in subjects.items()}
    sid_to_file = {sid: v["file"] for sid, v in subjects.items()}
    test_ids = sorted(fold_dict["test"])
    sids = test_ids[a.subject_idx: a.subject_idx + a.subjects]
    assert sids, "--subjects 切片为空"

    mean, std = compute_mild_train_norm(train_f, window_len=a.window_len)
    norm_sha = _norm_sha(mean, std)
    # fold 键控断言（10-25-symw-ec-f12）：各折 train 集不同 ⇒ norm SHA 各折登记；
    # 缺 fold 键（--fold 越界/未登记）= 拒跑
    assert a.fold in EXPECTED_NORM_SHA_BY_FOLD, \
        f"--fold {a.fold} 无登记 norm SHA（已登记折: {sorted(EXPECTED_NORM_SHA_BY_FOLD)}）"
    expected_norm_sha = EXPECTED_NORM_SHA_BY_FOLD[a.fold]
    assert norm_sha == expected_norm_sha, \
        (f"norm mean_std SHA256 对拍失败 (fold={a.fold}): {norm_sha} != "
         f"训练登记值 {expected_norm_sha}（口径漂移即拒跑）")
    files_hash = _train_files_hash(train_f)

    # symw 训练直用 FlowVelocityNetExitGNN（flow_exit_gnn_symw.py:58 import，零架构
    # 差异）⇒ 架构侧等价 flow_exit_gnn=True（SymWAdam 仅 optimizer 尾部投影，不进 ckpt）
    model = build_model_mild(device, _model_args(a.flow_exit_gnn or a.symw), stage=2,
                             obs_len=a.obs_len, forecast_len=a.forecast_len)
    load_flow_ckpt(model, a.ckpt)
    set_seed(a.seed)

    from train_mild_depress import MildTrialDataset  # 延迟导入（同 quality_check_mild）
    grid = log_grid(a.sigma_min, a.sigma_max, a.n_grid)
    print(f"[main] paradigm: global-tau UNBIASED (mu=0, sigma*xi) | q={a.q} "
          f"sigma_ref={a.sigma_ref} grid=log[{a.sigma_min},{a.sigma_max}]x{a.n_grid} | "
          f"subjects={sids} | device={device}", flush=True)

    t_all = time.time()
    for i, sid in enumerate(sids):
        subj_idx = a.subject_idx + i
        json_path = out_dir / f"subject_{subj_idx}.json"
        if json_path.exists() and not a.overwrite:
            print(f"[main] subject_{subj_idx} (s{sid}) json 已存在，跳过（断点续跑）",
                  flush=True)
            continue
        t0 = time.time()
        ds = MildTrialDataset([sid_to_file[sid]], (mean, std),
                              obs_len=a.obs_len, forecast_len=a.forecast_len,
                              window_len=a.window_len)
        trials = ds.trials if a.n_trials <= 0 else ds.trials[:a.n_trials]
        x_obs = torch.stack([t[:a.obs_len] for t in trials]).contiguous().to(device)
        N, P, C = x_obs.shape
        n_src = a.sources if a.sources > 0 else C
        assert n_src <= C

        site = SignalProbeSite(
            model=model, x_obs_subject=x_obs,
            n_steps=a.n_steps, use_proj=False, batch_size=a.batch_size,
            seed=a.seed, n_grid=a.n_grid,
            sigma_min=a.sigma_min, sigma_max=a.sigma_max,
            use_biased_noise=False)
        assert site._mu_bias is None, "μ_bias 未关闭"

        enabled = device.type == "cuda"
        # stage 1：σ_ref 响应扫描 → τ_global（flatten 含对角自响应，先例同口径）
        response = np.full((n_src, C), np.nan, dtype=np.float32)
        mean_diff = np.full((n_src, C), np.nan, dtype=np.float32)
        for src in range(n_src):
            site.set_source(src)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=enabled):
                mse_vec, md_vec = site.perturb_and_measure(float(a.sigma_ref), src)
            response[src, :] = mse_vec.detach().float().cpu().numpy()
            mean_diff[src, :] = md_vec.detach().float().cpu().numpy()
        tau_global = compute_global_tau(torch.from_numpy(response), a.q)

        # stage 2：log 网格走查 → σ*（每源单 autocast 域，模板 _stage2_sigma_search）
        sigma_matrix = np.full((n_src, C), np.nan, dtype=np.float32)
        mse_curve_all = np.full((n_src, a.n_grid, C), np.nan, dtype=np.float32)
        for src in range(n_src):
            site.set_source(src)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=enabled):
                mse_curve = torch.stack(
                    [site.perturb_and_measure(float(s), src)[0] for s in grid], dim=0)
            mse_curve_all[src] = mse_curve.detach().float().cpu().numpy()
            sigma_matrix[src, :] = sigma_star_with_global_tau(
                mse_curve, grid, float(tau_global),
                a.sigma_min, a.sigma_max).numpy()

        sigma_t = torch.from_numpy(sigma_matrix.copy())
        conn = -(sigma_t.float().log() + HALF_LOG_2PIE).numpy() + V4_OFFSET
        sign_matrix = np.sign(mean_diff)
        v = sigma_matrix.flatten()
        n_min = int(np.sum(v <= a.sigma_min + 1e-6))
        n_max = int(np.sum(v >= a.sigma_max - 1e-6))
        elapsed = (time.time() - t0) / 60.0
        # --- 代理脑元数据条件化（design §2.3：旗标关 = 旧字段原值不动）---
        symw_f12 = bool(a.symw) and "10-25-latent-flow-symw-f12" in str(a.ckpt)
        if a.flow_exit_gnn:
            method_str = ("global_tau_quantile UNBIASED (mu=0, sigma*xi, latent-flow "
                          "mild exit-gnn variant surrogate, signal-space, sigma* "
                          "preserved; sign field saved but not directional by design)")
            surrogate_block = {
                "task": "10-24-latent-flow-exit-gnn run_exitgnn_fold0",
                "ckpt_stage2_best": ("0.8241@E116 (early-stop E146, converged within "
                                     "budget; amplitude under-calibration — see "
                                     "surrogate_caveat)"),
                "variant": ("OCLatentFlow two-stage + flow exit GNN variant (exit K=2 "
                            "shared-A_eff message passing + zero-init exit projection, "
                            "+33,024 params vs w500 base; trained lr 1e-3 — two-factor "
                            "caveat vs w500), mexhat-graph flow (K=2, gamma 0.1, "
                            "gamma_m 1.0), full-window decoder, deterministic sampling"),
                "euler_steps": int(a.n_steps), "amp": "bf16-autocast",
                "obs_len": int(a.obs_len), "forecast_len": int(a.forecast_len),
                "n_channels": int(C), "flow_exit_gnn": True}
            surrogate_caveat = (
                "exit-gnn Run 002 (2026-10-24): medoid/real std 0.645/0.581 vs w500 "
                "0.957/0.996 — amplitude under-calibration regression (exit "
                "neighbor-mixing re-averaging hypothesis); lr 1e-3 and exit structure "
                "co-vary (two-factor, ablation pending user decision); theta-candidacy "
                "currently leans w500 base (user readout pending). EC results are "
                "PROVISIONAL and do not constitute a surrogate qualification verdict.")
        elif a.symw:
            method_str = ("global_tau_quantile UNBIASED (mu=0, sigma*xi, latent-flow "
                          "mild exit-gnn sym-w variant surrogate, signal-space, "
                          "sigma* preserved; sign field saved but not directional "
                          "by design)")
            if symw_f12:
                # f12 双折分支（10-25-symw-ec-f12）：best val/真早停 epoch = f12 训练侧
                # stage2 history 登记（fold1 0.8264@E116/E146；fold2 0.8433@E113/E143）
                surrogate_block = {
                    "task": f"10-25-latent-flow-symw-f12 run_symw_fold{a.fold}",
                    "ckpt_stage2_best": {
                        1: "0.8264@E116 (early-stop E146, converged)",
                        2: "0.8433@E113 (early-stop E143, converged)"}[a.fold],
                    "variant": ("OCLatentFlow two-stage + flow exit GNN variant "
                                "(shared-A_eff message passing + zero-init exit "
                                "projection) trained with SymWAdam per-step "
                                "W←(W+Wᵀ)/2 projection + anti-uniform λ0.2 — f12 "
                                "dual-fold training (10-25-latent-flow-symw-f12, "
                                "symw Run 001 same params across folds; this ckpt "
                                f"= fold{a.fold}), mexhat-graph flow (K=2, gamma "
                                "0.1, gamma_m 1.0), full-window decoder, "
                                "deterministic sampling"),
                    "euler_steps": int(a.n_steps), "amp": "bf16-autocast",
                    "obs_len": int(a.obs_len), "forecast_len": int(a.forecast_len),
                    "n_channels": int(C), "symw": True}
                surrogate_caveat = (
                    "symw-f12 Run 001 (2026-10-25): fold-transfer surrogate — "
                    "theta-candidacy pending user decision; cross-fold "
                    "(fold0/1/2) EC juxtaposition is DESCRIPTIVE only (per-fold "
                    "theta are different surrogate instances — not a "
                    "reproducibility readout); A_eff undirected (W strictly "
                    "symmetrized) — fold0 symmetry diagnosis (10-25-symw-ec-"
                    "unbiased) narrowed this caveat: EC v4 matrices NOT collapsed "
                    "to undirected (group corr(M,Mᵀ) −0.002±0.135, fro_asym_ratio "
                    "≈1.02), the constraint acts on GNN message passing only, "
                    "direction carried by asymmetric per-channel encoding/exit "
                    "projection + data-dependent σ*; λ0.2 and sym-W two-factor "
                    "co-vary (ablation pending); EC results are PROVISIONAL and "
                    "do not constitute a surrogate qualification verdict.")
            else:
                surrogate_block = {
                    "task": "10-25-latent-flow-exit-gnn-symw run_symw_fold0",
                    "ckpt_stage2_best": ("0.8054@E144 (early-stop converged; W strictly "
                                         "symmetrized — see surrogate_caveat)"),
                    "variant": ("OCLatentFlow two-stage + flow exit GNN variant "
                                "(shared-A_eff message passing + zero-init exit "
                                "projection) trained with SymWAdam per-step W←(W+Wᵀ)/2 "
                                "projection + anti-uniform λ0.2; two-factor vs exit-gnn "
                                "Run 001 (λ and sym-W co-vary, ablation pending), "
                                "mexhat-graph flow (K=2, gamma 0.1, gamma_m 1.0), "
                                "full-window decoder, deterministic sampling"),
                    "euler_steps": int(a.n_steps), "amp": "bf16-autocast",
                    "obs_len": int(a.obs_len), "forecast_len": int(a.forecast_len),
                    "n_channels": int(C), "symw": True}
                surrogate_caveat = (
                    "symw Run 000 (2026-10-25): A_eff undirected (W strictly "
                    "symmetric, torch.equal-level) ⇒ EC directionality "
                    "construct-limited — per-direction readout must reset "
                    "expectations to undirected caliber; λ0.2 and sym-W two-factor "
                    "co-vary (ablation pending); medoid/real std 0.858/0.848 "
                    "(partial amplitude recovery vs w500 0.957/0.996); "
                    "theta-candidacy pending user decision; EC results are "
                    "PROVISIONAL and do not constitute a surrogate qualification "
                    "verdict.")
        else:
            method_str = ("global_tau_quantile UNBIASED (mu=0, sigma*xi, latent-flow "
                          "mild-w500 surrogate, signal-space, sigma* preserved; sign "
                          "field saved but not directional by design)")
            surrogate_block = {
                "task": "10-24-latent-flow-mild-w500 run_w500_fold0",
                "ckpt_stage2_best": "0.8356@E150 (cap, NOT converged — EC results "
                                    "provisional pending user readout/budget decision)",
                "variant": "OCLatentFlow two-stage, mexhat-graph flow (K=2, gamma 0.1, "
                           "gamma_m 1.0), full-window decoder, deterministic sampling "
                           "(z_obs=mu + deterministic Euler + deterministic decoder)",
                "euler_steps": int(a.n_steps), "amp": "bf16-autocast",
                "obs_len": int(a.obs_len), "forecast_len": int(a.forecast_len),
                "n_channels": int(C)}
            surrogate_caveat = None
        payload = {
            "subject_idx": int(subj_idx), "subject_file": f"s{sid}",
            "group": groups[sid], "n_trials": int(N),
            "n_sources_requested": int(n_src), "n_sources_succeeded": int(n_src),
            "n_source_failures": 0,
            "method": method_str,
            "tau_global": float(tau_global), "q": float(a.q),
            "sigma_ref": float(a.sigma_ref),
            "tau_caliber_note": ("tau quantile over flattened full (src,tgt) matrix "
                                 "INCLUDING diagonal self-response (house precedent; "
                                 "diag share 1/129 — do not compare against "
                                 "offdiag-only calibers)"),
            "sigma_matrix": sigma_matrix.tolist(),
            "connectivity_matrix": conn.tolist(),
            "connectivity_formula": "-(log sigma* + 0.5*log(2*pi*e)) + 5/7",
            "raw_response_matrix": response.tolist(),
            "sign_matrix": sign_matrix.tolist(),
            "signed_connectivity_matrix": (sign_matrix * conn).tolist(),
            "mean_diff_matrix": mean_diff.tolist(),
            "noise_model": ("unbiased_gaussian_mu0 (sigma*xi additive on z-scored "
                            "x_obs (375), before model; same xi across sigma per source)"),
            "grid": "log_spaced",
            "hparams": {"q": float(a.q), "sigma_ref": float(a.sigma_ref),
                        "sigma_min": float(a.sigma_min), "sigma_max": float(a.sigma_max),
                        "n_grid": int(a.n_grid), "n_steps": int(a.n_steps),
                        "batch_size": int(a.batch_size), "seed": int(a.seed),
                        "use_biased_noise": False,
                        "flow_exit_gnn": bool(a.flow_exit_gnn),
                        "symw": bool(a.symw),
                        # f12 显式化（10-25-symw-ec-f12）：折身份进 hparams；
                        # 旗标关/fold0 先例 payload 键面不变
                        **({"fold": int(a.fold)} if symw_f12 else {})},
            "diagnostics": {
                "n_at_sigma_min": n_min, "n_at_sigma_max": n_max,
                "pct_at_sigma_min": n_min / v.size * 100,
                "pct_at_sigma_max": n_max / v.size * 100,
                "pct_interior": 100 - (n_min + n_max) / v.size * 100,
                "sigma_median": float(np.median(v)),
                "sigma_q25": float(np.quantile(v, 0.25)),
                "sigma_q75": float(np.quantile(v, 0.75)),
                "p_response_above_tau": float(np.mean(response > tau_global)),
            },
            "checkpoint": Path(a.ckpt).name,
            "surrogate": surrogate_block,
            "provenance": {
                "manifest": {"path": a.manifest, "sha256": EXPECTED_MANIFEST_SHA},
                "norm": {"source": f"train fold first-{a.window_len} samples",
                         "files_hash": files_hash,
                         "mean_std_sha256": expected_norm_sha},
                "window": {"window_len": a.window_len, "obs_len": a.obs_len,
                           "forecast_len": a.forecast_len},
                "ckpt_path": str(Path(a.ckpt).resolve()),
                "git_head": _git_head()},
            "elapsed_min": float(elapsed),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        payload = sanitize_for_json(payload)
        if a.task_label:
            payload["task"] = a.task_label
        if surrogate_caveat is not None:
            payload["surrogate_caveat"] = surrogate_caveat
        _atomic_write_json(json_path, payload)
        _atomic_write_npz(
            out_dir / f"subject_{subj_idx}_curves.npz",
            mse_curve=mse_curve_all, resp_ref=response, mean_diff_ref=mean_diff,
            grid=np.array(grid, dtype=np.float64),
            tau_global=np.float64(tau_global))
        d = payload["diagnostics"]
        print(f"[main] subject_{subj_idx} (s{sid},{groups[sid]}) {elapsed:.2f} min | "
              f"tau={tau_global:.3e} interior={d['pct_interior']:.1f}% "
              f"@min={d['pct_at_sigma_min']:.1f}% @max={d['pct_at_sigma_max']:.1f}% "
              f"P(resp>tau)={d['p_response_above_tau']:.3f} | wrote {json_path.name}",
              flush=True)
    print(f"[main] done in {(time.time() - t_all) / 60:.1f} min -> {out_dir}", flush=True)


# =============================================================================
# summary 模式：离线组级汇总（无 GPU）
# =============================================================================

def _variance_decomposition(m: np.ndarray, n_iter: int = 10) -> dict:
    """offdiag 方差的行(源)/列(靶)/残差占比（交替均值消主效应；diag 置 nan 屏蔽）。"""
    a = m.astype(np.float64).copy()
    np.fill_diagonal(a, np.nan)
    grand = np.nanmean(a)
    row_eff = np.nanmean(a, axis=1) - grand
    col_eff = np.nanmean(a, axis=0) - grand
    resid = a - grand - row_eff[:, None] - col_eff[None, :]
    for _ in range(n_iter):  # diag 缺失致非正交，交替消元收敛
        resid -= np.nanmean(resid, axis=1, keepdims=True)
        resid -= np.nanmean(resid, axis=0, keepdims=True)
        row_eff = np.nanmean(a - resid, axis=1) - grand
        col_eff = np.nanmean(a - resid, axis=0) - grand
    v_row = np.nanvar(grand + row_eff[:, None] * np.ones_like(a))
    v_col = np.nanvar(grand + col_eff[None, :] * np.ones_like(a))
    v_res = np.nanvar(resid)
    tot = v_row + v_col + v_res
    return {"row_source_share": v_row / tot, "col_target_share": v_col / tot,
            "residual_edge_share": v_res / tot}


def run_summary(a) -> None:
    main_dir = Path(a.main_dir or a.out_dir)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    task_name = a.task_label or "10-24-mild-w500-ec-unbiased"
    caveat_note = ("EC 结论继承 exit-gnn 代理脑幅度欠校准 + lr/结构二因素 caveat"
                   "（provisional）" if a.flow_exit_gnn else
                   ("EC 结论继承 symw 代理脑 A_eff 无向（方向性构造受限）+ λ/sym-W "
                    "二因素 caveat（provisional）" if a.symw else
                    "EC 结论继承 w500 代理脑 stage2 顶格未收敛 caveat（provisional）"))
    rows = sorted(main_dir.glob("subject_*.json"))
    assert rows, f"no subject_*.json under {main_dir}"
    off = ~np.eye(128, dtype=bool)
    per, conn_stack, sig_stack = [], [], []
    for f in rows:
        d = json.load(open(f))
        sig = np.array(d["sigma_matrix"], dtype=np.float64)
        conn = np.array(d["connectivity_matrix"], dtype=np.float64)
        tau = d["tau_global"]
        z = np.load(f.parent / (f.stem + "_curves.npz"))
        curve = z["mse_curve"].astype(np.float64)
        floor, ceil = curve[:, 0, :], curve[:, -1, :]
        dyn = ceil / np.maximum(floor, 1e-30)
        r = {
            "subject": d["subject_file"], "group": d.get("group"),
            "tau": tau, "pct_interior": d["diagnostics"]["pct_interior"],
            "pct_at_min": d["diagnostics"]["pct_at_sigma_min"],
            "pct_at_max": d["diagnostics"]["pct_at_sigma_max"],
            "p_response_above_tau": d["diagnostics"]["p_response_above_tau"],
            "sigma_median_offdiag": float(np.median(sig[off])),
            "conn_mean_offdiag": float(conn[off].mean()),
            "conn_std_offdiag": float(conn[off].std()),
            "dyn_ratio_q05": float(np.quantile(dyn, 0.05)),
            "dyn_ratio_median": float(np.median(dyn)),
            "nonmono_pair_frac": float(np.mean(
                np.any(np.diff(curve, axis=1) < 0, axis=1))),
            "var_decomp": _variance_decomposition(conn),
        }
        # 重映射对照（严格单调 ⇒ Spearman 逐位一致断言，先例 recompute_inv_conn_gc 同款）
        from scipy.stats import spearmanr
        conn_inv = 1.0 / (sig + EPS_REMAP)
        assert abs(spearmanr(conn[off], conn_inv[off]).statistic - 1.0) < 1e-9, \
            f"{d['subject_file']}: 重映射 Spearman 不变校验失败"
        r["spearman_remap_invariance"] = "PASS"
        per.append(r)
        conn_stack.append(conn)
        sig_stack.append(sig)

    conn_stack = np.stack(conn_stack)
    sig_stack = np.stack(sig_stack)

    # 跨被试稳定性：逐对 offdiag Spearman
    stab = np.eye(len(per), dtype=np.float64)
    for i in range(len(per)):
        for j in range(i + 1, len(per)):
            rho = spearmanr(conn_stack[i][off], conn_stack[j][off]).statistic
            stab[i, j] = stab[j, i] = rho
    iu = np.triu_indices(len(per), 1)

    def agg(key):
        v = np.array([p[key] for p in per], dtype=np.float64)
        return {"mean": float(v.mean()), "std": float(v.std(ddof=1)),
                "min": float(v.min()), "max": float(v.max())}

    summary = {
        "task": f"{task_name} Run 002 (offline group summary)",
        "n_subjects": len(per),
        "caliber": {"q": 0.5, "formula": "-(log sigma*+0.5*ln2pie)+5/7",
                    "remap_check": "1/(sigma*+1e-4) Spearman-invariance asserted",
                    "tau_note": "tau quantile includes diagonal self-response "
                                "(house precedent caliber)"},
        "per_subject": per,
        "group": {k: agg(k) for k in
                  ("pct_interior", "pct_at_min", "pct_at_max",
                   "p_response_above_tau", "dyn_ratio_median", "nonmono_pair_frac",
                   "sigma_median_offdiag", "conn_mean_offdiag", "conn_std_offdiag")},
        "var_decomp_group_mean": {
            k: float(np.mean([p["var_decomp"][k] for p in per]))
            for k in per[0]["var_decomp"]},
        "cross_subject_stability": {
            "mean_offdiag_spearman": float(stab[iu].mean()),
            "min": float(stab[iu].min()), "max": float(stab[iu].max()),
            "matrix": stab.tolist()},
        "reading_notes": [
            "row_source_share 高 = σ* 主要编码源全局破坏性（07-04 教训口径），残差份额才是 a->b 边特异信号",
            "P(response>tau) 区分 τ 位置性钳边（P 偏离 0.5 大）vs 响应动态范围不足（dyn_ratio 低）",
            caveat_note,
        ],
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(out_dir / "group_summary.json", "w") as fh:
        json.dump(summary, fh, indent=1, ensure_ascii=False)
    np.savez_compressed(out_dir / "group_mean.npz",
                        conn_mean=conn_stack.mean(axis=0),
                        sigma_mean=sig_stack.mean(axis=0))
    g = summary["group"]
    print(f"[summary] n={len(per)} | interior {g['pct_interior']['mean']:.1f}% | "
          f"dyn_ratio_med {g['dyn_ratio_median']['mean']:.1f}x | "
          f"P(resp>tau) {g['p_response_above_tau']['mean']:.3f} | "
          f"row_share {summary['var_decomp_group_mean']['row_source_share']:.3f} | "
          f"stab rho {summary['cross_subject_stability']['mean_offdiag_spearman']:.3f} "
          f"-> {out_dir / 'group_summary.json'}", flush=True)


# =============================================================================
# gccorr 模式：EC × mild GC 参考相关（离线，协议对齐 gen_unbiased_gc_corr.py）
# =============================================================================

def run_gccorr(a) -> None:
    from scipy import stats as sps
    main_dir = Path(a.main_dir or a.out_dir)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    task_name = a.task_label or "10-24-mild-w500-ec-unbiased"
    surrogate_name = ("mild exit-gnn variant" if a.flow_exit_gnn else
                      ("mild exit-gnn sym-w variant" if a.symw else "mild-w500"))
    refs = {}
    for f in sorted(Path(REFS_DIR).glob("refs_s*.npz")):
        z = np.load(f, allow_pickle=True)
        refs[int(z["subject_id"])] = z["ec"].astype(np.float64)
    off = ~np.eye(128, dtype=bool)

    per, stack_conn, stack_ec = [], [], []
    for f in sorted(main_dir.glob("subject_*.json")):
        d = json.load(open(f))
        sid = int(d["subject_file"][1:])
        assert sid in refs, f"refs 缺 s{sid}"
        conn = np.array(d["connectivity_matrix"], dtype=np.float64)
        sig = np.array(d["sigma_matrix"], dtype=np.float64)
        conn_inv = 1.0 / (sig + EPS_REMAP)
        ec = refs[sid]
        r = {"subject": d["subject_file"], "group": d.get("group")}
        for tag, mat in (("entropy", conn), ("inv", conn_inv)):
            A, R = np.abs(mat), mat
            r[f"spearman_abs_aligned_{tag}"] = float(sps.spearmanr(A[off], ec[off]).statistic)
            r[f"pearson_abs_aligned_{tag}"] = float(np.corrcoef(A[off], ec[off])[0, 1])
            r[f"spearman_raw_aligned_{tag}"] = float(sps.spearmanr(R[off], ec[off]).statistic)
            r[f"spearman_abs_flipped_{tag}"] = float(sps.spearmanr(A[off], ec.T[off]).statistic)
            r[f"spearman_raw_flipped_{tag}"] = float(sps.spearmanr(R[off], ec.T[off]).statistic)
        per.append(r)
        stack_conn.append(conn)
        stack_ec.append(ec)
        print(f"[gccorr] s{sid} ({d.get('group')}): "
              f"abs rho={r['spearman_abs_aligned_entropy']:+.4f} "
              f"r={r['pearson_abs_aligned_entropy']:+.4f} "
              f"(flip {r['spearman_abs_flipped_entropy']:+.4f}; "
              f"inv rho={r['spearman_abs_aligned_inv']:+.4f})", flush=True)

    def agg(k):
        v = np.array([p[k] for p in per], dtype=np.float64)
        return {"mean": float(v.mean()), "std": float(v.std(ddof=1)),
                "min": float(v.min()), "max": float(v.max())}

    A_mean = np.stack([np.abs(c) for c in stack_conn]).mean(axis=0)
    ec_mean = np.stack(stack_ec).mean(axis=0)
    summary = {
        "task": f"{task_name} Run 003 (descriptive EC x GC)",
        "naming": f"EC_pert = unbiased mu=0 global_tau (q=0.5@sigma_ref=0.3) on "
                  f"latent-flow {surrogate_name} surrogate; GC = mne-connectivity "
                  f"spectral gc, VAR order 8 fixed, 1-45 Hz faverage (refs mild40)",
        "protocol": "offdiag 16256 pairs; aligned = row=source vs ec; flipped = vs "
                    "ec.T; entropy form v4 -(log sigma*+0.5*ln2pie)+5/7 and inverse "
                    "1/(sigma*+1e-4) (Spearman identical, Pearson differs)",
        "descriptive_only": ("描述性并置登记（先例协议），不构成代理脑资格判定；"
                             "EC=代理脑内部量 vs GC=观测数据量，实现口径隔离"),
        "per_subject": per,
        "group": {k: agg(k) for k in per[0] if k.startswith(("spearman", "pearson"))},
        "group_mean": {
            "spearman_abs_aligned": float(sps.spearmanr(A_mean[off], ec_mean[off]).statistic),
            "pearson_abs_aligned": float(np.corrcoef(A_mean[off], ec_mean[off])[0, 1])},
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(out_dir / "gccorr_summary.json", "w") as fh:
        json.dump(summary, fh, indent=1, ensure_ascii=False)
    g = summary["group"]
    print(f"\n[gccorr] GROUP abs aligned (entropy): "
          f"rho {g['spearman_abs_aligned_entropy']['mean']:+.4f}"
          f"±{g['spearman_abs_aligned_entropy']['std']:.4f} | "
          f"r {g['pearson_abs_aligned_entropy']['mean']:+.4f}"
          f"±{g['pearson_abs_aligned_entropy']['std']:.4f} | "
          f"group_mean rho={summary['group_mean']['spearman_abs_aligned']:+.4f} "
          f"-> {out_dir / 'gccorr_summary.json'}", flush=True)


def main() -> int:
    a = parse_args()
    if a.mode == "main":
        run_main(a)
    elif a.mode == "summary":
        run_summary(a)
    else:
        run_gccorr(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
