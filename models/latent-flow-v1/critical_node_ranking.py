#!/usr/bin/env python3
"""critical_node_ranking.py — 关键节点排序实验（噪声注入→全局影响→128通道排序→三σ稳健性）.

任务 10-30-critical-node-ranking（prd.md R1–R5 / design.md 契约实现，用户
2026-10-30 直给四步设计）。与 EC_pert 推理同型的虚拟扰动范式，从边级
（源→靶连接）改为**节点级（源→全局影响）**：

  1. 噪声注入：源通道 c 的历史段 x_obs（z-score 后、入模前）加固定方差
     高斯噪声 σ·ξ（μ=0 iid；ξ~N(0,1)），σ ∈ {0.3, 0.5, 0.7} 三档；
  2. 影响度量：I_c(σ) = Σ_{b≠c} MSE_b(σ)（其余 127 个通道未来段 MSE 之和，
     排除源自响应；MSE = 扰动生成 vs 站内缓存干净基线，跨 trial×fc 时间点平均）；
  3. 关键节点排序：全部 128 个通道逐一执行，按 I_c 从大到小排序
     （rank 1 = 全局影响最大 = 最关键节点；并列按通道序号升序稳定排序）；
  4. 稳健性检验：三档 σ 完整重复，对比排序一致性（Spearman/Kendall τ-b +
     top-K 交叠 + 比例性诊断——区分「σ² 缩放的构造性稳定」与「非线性重排」）。

复用（零拷贝，包体零改动，镜像双线 EC 主跑先例）：
  - bisect_sigma.SignalProbeSite（μ=0 无偏 iid / 确定性采样 z_obs=mu_obs /
    CRN = set_source(c) 单次生成 ξ 后跨三档 σ 复用 ⇒ 三档天然配对）；
  - MODMA 装载链（ec_modma_globaltau_unbiased.py 先例）：
    _connectivity_driver.{load_model, get_fold_test_subjects, load_subject_trials,
    subsample_trials, sanitize_for_json, json_default}；
  - mild 装载链（ec_mild_unbiased.py 先例）：train_mild_depress.{MANIFEST_DEFAULT,
    load_manifest, compute_mild_train_norm, build_model_mild, MildTrialDataset} +
    viz_mild.{_model_args, load_flow_ckpt} + ec_mild_unbiased 常量
    （EXPECTED_NORM_SHA_BY_FOLD / EXPECTED_MANIFEST_SHA / W500_*——只 import
    常量，不复用其 _caliber_asserts；模块 import 有 main 守卫，无副作用）。

autocast 契约（评审 B-1，σ=0 断言成立前提）：站点基线在构造器内经 bf16
autocast 计算，perturb_and_measure 本体不管理 dtype ⇒ **每次调用（含 σ=0）
必须包在 torch.amp.autocast("cuda", bf16) 域内**，每源一个域（镜像
ec_mild_unbiased L424-439 / ec_modma_globaltau_unbiased L210-235 先例）。

守卫（design §2.4，跑前对拍 result/ 注册件，只读，任一不符即拒跑）：
  - modma：ckpt = result/checkpoints/modma/latent_flow_0705/fold{k}/stage2/
    fold{k}_modma_0705_graph_mexhat.pt，SHA256 对拍脚本内钉死常量表（抄录自
    result/checkpoints/modma/latent_flow_0705/sha256.json，2026-10-30）；
    文件名 fold{k}_ 前缀与 --fold 锁定；须 env MODMA_DATA_ROOT。
  - mild：ckpt = result/checkpoints/mild/latent_flow_symw/fold{k}/
    fold{k}_mild_flow_best_graph_mexhat.pt，SHA256 对拍钉死常量表（抄录自
    result/checkpoints/MANIFEST.md stage2 表，2026-10-30）；norm SHA 按 fold
    对拍 + manifest SHA 对拍 + 窗三元组 500/375/125 断言。

三模式：main（GPU 主跑，逐被试 json 原子写 + 已存在即跳过 = 断点续跑）；
summary（离线两级三对稳健性 + 比例性诊断 + 双组级口径）；figs（≥3 张图）。

Run（冒烟；主跑命令见任务 experiment.md）:
  MODMA_DATA_ROOT=/root/autodl-tmp/modma python3 models/latent-flow-v1/critical_node_ranking.py \
    --line modma --mode main --fold 0 --subjects 1 \
    --out-dir /root/npi/outputs/10-30-critical-node-ranking
  python3 models/latent-flow-v1/critical_node_ranking.py \
    --line mild --mode main --fold 0 --subjects 1 \
    --out-dir /root/npi/outputs/10-30-critical-node-ranking
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
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

VARIANT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(VARIANT_DIR))  # 使 latent_flow 与同目录脚本可导入

from latent_flow.attribution.bisect_sigma import SignalProbeSite  # noqa: E402
from latent_flow.scripts._connectivity_driver import (  # noqa: E402
    get_fold_test_subjects,
    json_default,
    load_model,
    load_subject_trials,
    sanitize_for_json,
    subsample_trials,
)
from latent_flow.training.seeding import set_seed  # noqa: E402

# --- 口径常量（design §2.1/§2.2，PRD R1/R3 钉死） ---
SIGMAS = (0.3, 0.5, 0.7)          # 三档噪声方差（固定剂量读出，不走 engine 网格）
SIGMA_PAIRS = ((0, 1), (0, 2), (1, 2))   # 稳健性三对
SEED, N_STEPS, BATCH = 42, 10, 32
N_TRIALS_MODMA = 32               # MODMA 每被试 trial 子采样数（PRD R3）
PROBE_SIGMA_MIN, PROBE_SIGMA_MAX = 0.01, 0.7   # 站点元数据（engine 网格不走）
MONO_REL_TOL = 1e-4               # 单调性诊断 bf16 相对容差（design §2.1）
MAX_FAIL_FRAC = 0.10              # 逐源 RuntimeError 失败率 > 10% ⇒ 被试 FAILED
TOP_K_LIST = (10, 20)             # top-K 一致性 K 档（design §3.2）
RATIO_EPS = 1e-12                 # 比例性诊断分母护栏（I(0.3)≈0 通道剔除）
RATIO_QUAD_REF = (0.7 / 0.3) ** 2  # ≈ 5.444：线性响应 I∝σ² 下的期望比值

# --- ckpt 注册件路径模板 + SHA256 钉死常量表（design §2.4；result/ 只读） ---
MODMA_CKPT_TEMPLATE = ("/root/npi/result/checkpoints/modma/latent_flow_0705/"
                       "fold{k}/stage2/fold{k}_modma_0705_graph_mexhat.pt")
# 抄录来源：result/checkpoints/modma/latent_flow_0705/sha256.json 的 fold0..4
# stage2 五条目（2026-10-30 抄录；该文件为 10-29 注册件，只读）。
MODMA_STAGE2_SHA256 = {
    0: "5c6fbf31f10874474c02a3621af8845f517d8d842592cf8f25ca1372439889cb",
    1: "deb199e054cb348e9ae87b1e0a18e65e0702671292fed1430d894d29272f9f1c",
    2: "fa4a0ba57fe3bbf2a8cfac040c448d4c5ac280d7b0a61410eb608ee3c4a78ef5",
    3: "6bb74a1903d051829042f660eeab4cb246c3a2a57c1c15ce071e9e840881280e",
    4: "6c51a4c265e16048e1dce723a8b02348ce5c68e51ea884e4182298c9eebd3d1c",
}
MILD_CKPT_TEMPLATE = ("/root/npi/result/checkpoints/mild/latent_flow_symw/"
                      "fold{k}/fold{k}_mild_flow_best_graph_mexhat.pt")
# 抄录来源：result/checkpoints/MANIFEST.md 文件清单表 stage2 flow 五条目
# （#1/#3/#5/#7/#9 行，2026-10-30 抄录；注册件只读）。
MILD_STAGE2_SHA256 = {
    0: "43564dd326ad7b209ecba367f37b68362af1514f617b27b71da1c67893d6654d",
    1: "06eb59fe35c53693107cc06d3614cfa5698e9ce0c2cd616bafee3d029d97e4a0",
    2: "4aaad72eab5a92433ac5674a8e9b005ea04428d355d0e6454b46b9e5f7f5303d",
    3: "421edf87a191421a1babe3a8dce3cb7a4d78fff132a9e0fa32804767ddde6c44",
    4: "3d8a4e974dfe40e2a462c71a0bbe1e86446f3fe9ebd6f8883c6dc005da2caa2c",
}

# --- 随档 caveat（design §2.2/§5，评审 A-6/B-4：按线分写，不预设结论） ---
DOSE_CAVEAT = {
    "modma": ("剂量外推 caveat：本线 EC 口径 σ 网格 = log[0.01,0.3]×16，三档 "
              "0.3/0.5/0.7 全部 ≥ 网格顶 0.3——响应可能饱和/排序扁平化，属稳健性"
              "检验对象，如实报告（不中断、不剪裁）"),
    "mild": ("剂量外推 caveat：本线 EC 口径 σ 网格 = log[0.01,0.48]×24，"
             "0.5/0.7 超上限 0.48——响应可能饱和/排序扁平化，属稳健性检验对象，"
             "如实报告（不中断、不剪裁）"),
}
CRN_NOTE = ("CRN 单实现语义：set_source(c) 单次生成 ξ 后跨三档 σ 复用（同源同 ξ，"
            "σ 仅做缩放）⇒ 三档配对、剂量-响应无蒙特卡洛抖动；检验的是同一噪声"
            "实现下的剂量稳定性，不是实现无关稳定性（后者须独立 ξ 重复，未在本档）")
NAN_NOTE = ("nan_to_num 局限：forecast_fc/perturb_and_measure 内部 torch.nan_to_num "
            "在 MSE 计算前静默置零 ⇒ NaN 通道不可事后检出，前向数值异常被零化后 "
            "MSE 可能失真（无告警）；处置 = 逐源 RuntimeError 捕获计数 + 本注记")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--line", required=True, choices=["modma", "mild"],
                   help="实验线：modma = MODMA latent-flow 0705 五折；"
                        "mild = mild latent-flow symw 五折")
    p.add_argument("--mode", required=True, choices=["main", "summary", "figs"],
                   help="main = GPU 主跑逐被试 json；summary = 离线稳健性汇总；"
                        "figs = 离线图件（summary/figs 读 <out-dir>/<line>/fold*/）")
    p.add_argument("--fold", type=int, default=0,
                   help="折号（main 模式用；summary/figs 忽略，读全部折）")
    p.add_argument("--ckpt", type=str, default=None,
                   help="覆盖默认注册件 ckpt（守卫负例用；默认按 line/fold 解析"
                        "result/checkpoints/ 注册件路径，只读）")
    p.add_argument("--subjects", type=int, default=0,
                   help="主跑被试数（默认 0 = 该折 test 全部；冒烟可传 1）")
    p.add_argument("--subject-idx", type=int, default=0,
                   help="起始被试序号（折内 test 切片偏移）")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--n-steps", type=int, default=N_STEPS)
    p.add_argument("--batch-size", type=int, default=BATCH)
    p.add_argument("--device", type=str, default=None,
                   help="torch device（默认 cuda 可用时 cuda）")
    p.add_argument("--out-dir", type=str, required=True,
                   help="产物根目录（必填，无默认——防误写旧任务树；产物落 "
                        "<out-dir>/<line>/fold{k}/ 等子树）")
    p.add_argument("--overwrite", action="store_true",
                   help="main 模式：json 已存在也重算（默认跳过=断点续跑；"
                        "FAILED 被试重试须本旗标）")
    return p.parse_args(argv)


def _sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True,
                              cwd="/root/npi").stdout.strip()
    except Exception:
        return "unavailable"


def _atomic_write_json(path: Path, payload: dict) -> None:
    """tmp + rename 原子写：防长跑中断留半截 json 被断点续跑误判为完成。"""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=2, default=json_default, allow_nan=False)
    os.replace(tmp, path)


def _pair_name(i: int, j: int) -> str:
    return f"{SIGMAS[i]:g}<->{SIGMAS[j]:g}"


# =============================================================================
# 守卫 + 双线装载（design §2.3/§2.4；镜像双线 EC 主跑先例）
# =============================================================================

def _ckpt_guard(a) -> tuple[Path, str]:
    """ckpt 解析 + fold 前缀锁定 + SHA256 对拍（任一不符即拒跑，先于任何 GPU 工作）。"""
    if a.line == "modma":
        table, tmpl, src = (MODMA_STAGE2_SHA256, MODMA_CKPT_TEMPLATE,
                            "result/checkpoints/modma/latent_flow_0705/sha256.json")
    else:
        table, tmpl, src = (MILD_STAGE2_SHA256, MILD_CKPT_TEMPLATE,
                            "result/checkpoints/MANIFEST.md (stage2 表)")
    assert a.fold in table, f"--fold {a.fold} 无登记 ckpt SHA（已登记折: {sorted(table)}）"
    ckpt = Path(a.ckpt) if a.ckpt else Path(tmpl.format(k=a.fold))
    name = ckpt.name
    assert name.startswith(f"fold{a.fold}_"), (
        f"ckpt 文件名 fold 前缀与 --fold 锁定不符（折-权重错配防护）: {name} 不以 "
        f"fold{a.fold}_ 开头（--fold {a.fold}）")
    assert ckpt.exists(), f"ckpt not found: {ckpt}"
    sha = _sha256_file(ckpt)
    expected = table[a.fold]
    assert sha == expected, (
        f"ckpt SHA256 与钉死注册值不符 (fold={a.fold}): {sha} != {expected}（来源 "
        f"{src}；口径漂移即拒跑）")
    return ckpt, sha


def _slice_subjects(seq, a):
    if a.subjects > 0:
        out = seq[a.subject_idx:a.subject_idx + a.subjects]
    else:
        out = seq[a.subject_idx:]
    assert out, "--subjects/--subject-idx 切片为空"
    return out


def _prepare_modma(a, device):
    # env 守卫：latent_flow/config.py 无 env 时解析到包内不存在路径（environment.md）
    root = os.environ.get("MODMA_DATA_ROOT", "")
    assert root and Path(root, "geometry.pt").exists(), (
        "MODMA 线必须带 env MODMA_DATA_ROOT=/root/autodl-tmp/modma（缺省解析到包内"
        "不存在路径，.trellis/research/environment.md 登记）")
    ckpt, sha = _ckpt_guard(a)
    model = load_model(str(ckpt), device)   # _connectivity_driver（mexhat 图 ckpt）
    test_files = get_fold_test_subjects(a.fold, a.seed, dataset="modma")
    files = _slice_subjects(test_files, a)

    def build_x(subj_idx: int, subject_file: str):
        x_full = load_subject_trials(subject_file, device, normalize=True,
                                     dataset="modma")
        x_obs, n_used = subsample_trials(x_full, N_TRIALS_MODMA, a.seed, subj_idx)
        return x_obs, n_used

    items = [(a.subject_idx + i, f, Path(f).name) for i, f in enumerate(files)]
    provenance = {"modma_data_root": root,
                  "sha_registry": "/root/npi/result/checkpoints/modma/"
                                  "latent_flow_0705/sha256.json",
                  "n_trials_subsampled": N_TRIALS_MODMA}
    return model, items, build_x, ckpt, sha, provenance


def _prepare_mild(a, device):
    # 只 import ec_mild_unbiased 的常量（模块 import 有 main 守卫，无副作用；
    # 不复用其 _caliber_asserts——run_dir 白名单会拒 result/ 路径，评审 B-6）
    from ec_mild_unbiased import (EXPECTED_MANIFEST_SHA,
                                  EXPECTED_NORM_SHA_BY_FOLD,
                                  W500_FC,
                                  W500_OBS,
                                  W500_WINDOW)
    from train_mild_depress import (MANIFEST_DEFAULT,
                                    MildTrialDataset,
                                    build_model_mild,
                                    compute_mild_train_norm,
                                    load_manifest)
    from viz_mild import _model_args, load_flow_ckpt

    # 窗三元组断言（防 train_mild_depress 模块常量 1500/1125/375 误用，
    # ec_mild_unbiased L98-100 同款双保险）
    assert (W500_WINDOW, W500_OBS, W500_FC) == (500, 375, 125), \
        f"w500 口径三元组漂移: {(W500_WINDOW, W500_OBS, W500_FC)} != (500,375,125)"

    ckpt, sha = _ckpt_guard(a)
    assert _sha256_file(MANIFEST_DEFAULT) == EXPECTED_MANIFEST_SHA, \
        "manifest SHA256 与钉死值不符（splits/mild_5fold_seed42.json 口径漂移即拒跑）"

    _man, subjects, train_f, _val, _test, fold_dict, _rec = load_manifest(
        str(MANIFEST_DEFAULT), a.fold)
    mean, std = compute_mild_train_norm(train_f, window_len=W500_WINDOW)
    norm_sha = hashlib.sha256(
        np.concatenate([mean.ravel(), std.ravel()]).tobytes()).hexdigest()
    assert a.fold in EXPECTED_NORM_SHA_BY_FOLD, \
        f"--fold {a.fold} 无登记 norm SHA（已登记折: {sorted(EXPECTED_NORM_SHA_BY_FOLD)}）"
    expected_norm_sha = EXPECTED_NORM_SHA_BY_FOLD[a.fold]
    assert norm_sha == expected_norm_sha, \
        (f"norm mean_std SHA256 对拍失败 (fold={a.fold}): {norm_sha} != "
         f"训练登记值 {expected_norm_sha}（口径漂移即拒跑）")

    # symw 架构侧 = exit_gnn（symw 训练直用 FlowVelocityNetExitGNN，零架构差异；
    # exit-gnn ckpt 一律 build_model_mild(_model_args(True)) + load_flow_ckpt
    # strict=True，evaluation-calibers 10-30 装载陷阱契约）
    model = build_model_mild(device, _model_args(True), stage=2,
                             obs_len=W500_OBS, forecast_len=W500_FC)
    load_flow_ckpt(model, str(ckpt))
    set_seed(a.seed)   # ec_mild_unbiased L351 先例（model 构建后、被试循环前）

    sid_to_file = {sid: v["file"] for sid, v in subjects.items()}
    groups = {sid: v["group"] for sid, v in subjects.items()}
    sids = _slice_subjects(sorted(fold_dict["test"]), a)

    def build_x(subj_idx: int, sid: int):
        ds = MildTrialDataset([sid_to_file[sid]], (mean, std),
                              obs_len=W500_OBS, forecast_len=W500_FC,
                              window_len=W500_WINDOW)
        x_obs = torch.stack([t[:W500_OBS] for t in ds.trials]).contiguous().to(device)
        return x_obs, len(ds.trials)

    items = [(a.subject_idx + i, sid, f"s{sid}({groups[sid]})")
             for i, sid in enumerate(sids)]
    provenance = {"manifest": {"path": str(MANIFEST_DEFAULT),
                               "sha256": EXPECTED_MANIFEST_SHA},
                  "norm": {"mean_std_sha256": expected_norm_sha,
                           "window": [W500_WINDOW, W500_OBS, W500_FC]},
                  "n_trials": "30 全量（前缀窗 500 = obs 375 + fc 125）"}
    return model, items, build_x, ckpt, sha, provenance


# =============================================================================
# main 模式：噪声注入 → 全局影响 → 排序（GPU）
# =============================================================================

def run_main(a) -> None:
    device = torch.device(a.device if a.device
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # 守卫（SHA/fold 锁定）先于任何 GPU 工作
    prep = _prepare_modma(a, device) if a.line == "modma" else _prepare_mild(a, device)
    model, items, build_x, ckpt, ckpt_sha, provenance = prep
    out_dir = Path(a.out_dir) / a.line / f"fold{a.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[main] paradigm: critical node ranking (mu=0 unbiased, sigma*xi) | "
          f"line={a.line} fold={a.fold} sigmas={list(SIGMAS)} | subjects="
          f"{len(items)} | device={device}", flush=True)

    t_all = time.time()
    for subj_idx, key, label in items:
        json_path = out_dir / f"subject_{subj_idx}.json"
        if json_path.exists() and not a.overwrite:
            print(f"[main] subject_{subj_idx} ({label}) json 已存在，跳过（断点续跑）",
                  flush=True)
            continue
        t0 = time.time()
        x_obs, n_trials = build_x(subj_idx, key)
        N, P, C = x_obs.shape

        site = SignalProbeSite(
            model=model, x_obs_subject=x_obs,
            n_steps=a.n_steps, use_proj=False, batch_size=a.batch_size,
            seed=a.seed, sigma_min=PROBE_SIGMA_MIN, sigma_max=PROBE_SIGMA_MAX,
            use_biased_noise=False)
        assert site._mu_bias is None, "μ_bias 未关闭（无偏 μ=0 契约破坏）"

        enabled = device.type == "cuda"
        # σ=0 结构性恒等断言（首个源；确定性采样 ⇒ 扰动=0 时逐位回基线——
        # CRN 契约免费检查点。评审 B-1：σ=0 调用同样必须在 autocast 域内）
        site.set_source(0)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=enabled):
            mse0 = site.perturb_and_measure(0.0, 0)[0]
        sigma0_max_abs = float(mse0.abs().max().item())
        assert sigma0_max_abs == 0.0, (
            f"σ=0 CRN 结构性恒等断言失败: mse0 max|x|={sigma0_max_abs:.3e} ≠ 0"
            "（STOP——检查 autocast 契约/基线缓存；本断言不得删除）")
        sigma0_identity_pass = True

        # 逐源三档（每源单 autocast 域，同 ξ 跨 σ 复用 = CRN 配对）
        impact = np.full((C, len(SIGMAS)), np.nan, dtype=np.float32)
        self_resp = np.full((C, len(SIGMAS)), np.nan, dtype=np.float32)
        failed_sources = []
        for c in range(C):
            try:
                site.set_source(c)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                        enabled=enabled):
                    mses = [site.perturb_and_measure(float(s), c)[0]
                            for s in SIGMAS]
                for j, mse in enumerate(mses):
                    v = mse.detach().float().cpu().numpy()
                    impact[c, j] = float(v.sum() - v[c])   # Σ_{b≠c}，排除自响应
                    self_resp[c, j] = float(v[c])
            except RuntimeError as exc:
                failed_sources.append(c)
                print(f"  source {c} FAILED: {type(exc).__name__}: {exc}",
                      flush=True)
        n_fail = len(failed_sources)
        status = "failed" if n_fail > MAX_FAIL_FRAC * C else "ok"

        # 排序（descending、并列按通道序号升序稳定）；NaN 行排尾部（FAILED 例占位）
        ranking = np.stack(
            [np.argsort(-impact[:, j], kind="stable") for j in range(len(SIGMAS))],
            axis=1).astype(int)
        rank_ch = np.empty_like(ranking)
        for j in range(len(SIGMAS)):
            rank_ch[ranking[:, j], j] = np.arange(1, C + 1)

        # 单调性诊断（I(0.7)≥I(0.5)(1-1e-4)≥I(0.3)(1-1e-4)，bf16 容差；
        # 违例如实计数不中断；NaN 行不计入——其数已由 n_source_failures 承载）
        finite = np.isfinite(impact).all(axis=1)
        mono_ok = ((impact[:, 2] >= impact[:, 1] * (1 - MONO_REL_TOL))
                   & (impact[:, 1] >= impact[:, 0] * (1 - MONO_REL_TOL)))
        monotone_violations = int(np.sum(finite & ~mono_ok))

        elapsed = (time.time() - t0) / 60.0
        payload = {
            "task": "10-30-critical-node-ranking",
            "status": status,
            "line": a.line,
            "fold": int(a.fold),
            "subject_idx": int(subj_idx),
            "subject_file": label,
            "n_trials": int(n_trials),
            "n_channels": int(C),
            "method": ("critical node ranking: I_c(sigma) = sum_{b!=c} MSE_b(sigma) "
                       "over 127 non-source channels (self-response excluded), "
                       "descending; rank 1 = most critical node"),
            "sigmas": list(SIGMAS),
            "impact_matrix": impact.tolist(),
            "ranking": ranking.tolist(),
            "impact_rank_channelwise": rank_ch.tolist(),
            "self_response": self_resp.tolist(),
            "self_response_note": "diagnostic only; excluded from I_c sums",
            "tie_break": "descending impact, stable by channel index (ascending)",
            "monotone_violations": monotone_violations,
            "monotone_criterion": (f"I(0.7) >= I(0.5)*(1-{MONO_REL_TOL}) >= "
                                   f"I(0.3)*(1-{MONO_REL_TOL}) row-wise, bf16 rel "
                                   f"tol {MONO_REL_TOL}; NaN rows (failed sources) "
                                   "not counted"),
            "sigma0_identity_pass": bool(sigma0_identity_pass),
            "sigma0_max_abs": sigma0_max_abs,
            "n_source_failures": int(n_fail),
            "failed_sources": failed_sources,
            "hparams": {
                "seed": int(a.seed), "n_steps": int(a.n_steps),
                "batch_size": int(a.batch_size),
                "amp": ("bf16-autocast; every perturb_and_measure call incl "
                        "sigma=0 inside one autocast domain per source"),
                "crn_note": CRN_NOTE,
                "dose_extrapolation_caveat": DOSE_CAVEAT[a.line],
                "nan_to_num_note": NAN_NOTE,
                "probe_sigma_min": PROBE_SIGMA_MIN,
                "probe_sigma_max": PROBE_SIGMA_MAX,
                "probe_sigma_note": ("site sigma_min/sigma_max = metadata only "
                                     "(readability); engine grid unused — fixed-"
                                     "dose readout at sigmas"),
            },
            "checkpoint": str(ckpt.resolve()),
            "ckpt_sha256": ckpt_sha,
            "provenance": provenance,
            "git_head": _git_head(),
            "elapsed_min": float(elapsed),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        payload = sanitize_for_json(payload)
        _atomic_write_json(json_path, payload)
        print(f"[main] subject_{subj_idx} ({label}) {elapsed:.2f} min | "
              f"status={status} | sigma0==0 {'PASS' if sigma0_identity_pass else 'FAIL'}"
              f" (max|x|={sigma0_max_abs:.1e}) | mono_viol={monotone_violations} | "
              f"src_fail={n_fail}/{C} | top5@0.5={ranking[:5, 1].tolist()} | "
              f"wrote {json_path.name}", flush=True)
    print(f"[main] done in {(time.time() - t_all) / 60:.1f} min -> {out_dir}",
          flush=True)


# =============================================================================
# 离线装载 + 稳健性指标公共件（summary / figs 共用，无 GPU）
# =============================================================================

def _iter_subject_jsons(out_root: Path, line: str):
    """读 <out_root>/<line>/fold*/subject_*.json；排除并登记 FAILED 例。"""
    line_dir = out_root / line
    fold_dirs = sorted(line_dir.glob("fold*"), key=lambda p: int(p.name[4:]))
    assert fold_dirs, f"no fold dirs under {line_dir}"
    recs, failed = [], []
    for fd in fold_dirs:
        fold = int(fd.name[4:])
        for f in sorted(fd.glob("subject_*.json")):
            d = json.load(open(f))
            if d.get("status", "ok") != "ok":
                failed.append({"fold": fold, "file": f.name,
                               "n_source_failures": d.get("n_source_failures")})
                continue
            recs.append({"fold": fold, "path": f, "d": d,
                         "impact": np.asarray(d["impact_matrix"],
                                              dtype=np.float64),
                         "ranking": np.asarray(d["ranking"], dtype=int)})
    assert recs, f"no OK subject_*.json under {line_dir}/fold*/"
    return recs, failed


def _pair_metrics(vec_i, vec_j, rank_i, rank_j) -> dict:
    """一对 σ 的 128 维向量一致性：Spearman ρ + Kendall τ-b + top-K 交叠。"""
    from scipy.stats import kendalltau, spearmanr
    out = {"spearman": float(spearmanr(vec_i, vec_j).statistic),
           "kendall_tau_b": float(kendalltau(vec_i, vec_j).statistic)}
    for K in TOP_K_LIST:
        out[f"top{K}_overlap"] = float(
            len(set(rank_i[:K].tolist()) & set(rank_j[:K].tolist())) / K)
    return out


def _subject_pairs(rec) -> dict:
    out = {}
    for i, j in SIGMA_PAIRS:
        out[_pair_name(i, j)] = _pair_metrics(
            rec["impact"][:, i], rec["impact"][:, j],
            rec["ranking"][:, i], rec["ranking"][:, j])
    return out


def _group_vectors(recs):
    """组级两口径向量（n,128,3 → 128×3）：primary = 原始影响均值（主口径）；
    secondary = 逐被试 within-subject 秩均值（副口径，aggregation-robustness）。"""
    from scipy.stats import rankdata
    imp = np.stack([r["impact"] for r in recs])          # (n,128,3)
    primary = imp.mean(axis=0)
    ranks = np.stack([
        np.stack([rankdata(-r["impact"][:, j]) for j in range(len(SIGMAS))],
                 axis=1) for r in recs])                  # rank 1 = 影响最大
    secondary = ranks.mean(axis=0)
    return primary, secondary


def _group_pairs(vec) -> dict:
    """组级向量的三对指标（top-K 按该向量自身降序排序）。"""
    rk = [np.argsort(-vec[:, j], kind="stable") for j in range(len(SIGMAS))]
    return {_pair_name(i, j): _pair_metrics(vec[:, i], vec[:, j], rk[i], rk[j])
            for i, j in SIGMA_PAIRS}


def _agg(vals) -> dict:
    v = np.asarray(vals, dtype=np.float64)
    return {"mean": float(v.mean()),
            "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "min": float(v.min()), "max": float(v.max()), "n": int(len(v))}


def _ratio_stats(imp) -> dict:
    """逐被试比例性诊断：ratio_c = I_c(0.7)/I_c(0.3)（分母 ≤eps 通道剔除并计数）。"""
    num, den = imp[:, 2], imp[:, 0]
    mask = den > RATIO_EPS
    n_ex = int((~mask).sum())
    r = num[mask] / den[mask]
    return {"median": float(np.median(r)), "q25": float(np.quantile(r, 0.25)),
            "q75": float(np.quantile(r, 0.75)), "min": float(r.min()),
            "max": float(r.max()), "n_excluded_den_le_eps": n_ex,
            "n_used": int(mask.sum())}


def _loglog_slope(vec, i, j) -> dict:
    """组平均通道级 log-log 斜率：log I_c(σ_hi) vs log I_c(σ_lo)（线性响应 ⇒ ≈2）。"""
    m = (vec[:, i] > 0) & (vec[:, j] > 0)
    n_ex = int((~m).sum())
    if int(m.sum()) < 2:
        return {"slope": None, "intercept": None, "n_used": int(m.sum()),
                "n_excluded_nonpositive": n_ex}
    slope, intercept = np.polyfit(np.log(vec[m, i]), np.log(vec[m, j]), 1)
    return {"slope": float(slope), "intercept": float(intercept),
            "n_used": int(m.sum()), "n_excluded_nonpositive": n_ex}


# =============================================================================
# summary 模式：两级三对稳健性 + 比例性诊断（离线，无 GPU）
# =============================================================================

def run_summary(a) -> None:
    out_root = Path(a.out_dir)
    recs, failed = _iter_subject_jsons(out_root, a.line)
    folds = sorted({r["fold"] for r in recs})
    by_fold = {f: [r for r in recs if r["fold"] == f] for f in folds}

    # --- 逐被试级：每折 mean±sd + 全线合并分布 ---
    per_subject = [{"fold": r["fold"], "subject": r["d"]["subject_file"],
                    "pairs": _subject_pairs(r),
                    "ratio_I07_over_I03": _ratio_stats(r["impact"]),
                    "monotone_violations": int(r["d"]["monotone_violations"]),
                    "sigma0_identity_pass": bool(r["d"]["sigma0_identity_pass"]),
                    "n_source_failures": int(r["d"]["n_source_failures"])}
                   for r in recs]
    pair_names = [_pair_name(i, j) for i, j in SIGMA_PAIRS]
    metric_keys = list(per_subject[0]["pairs"][pair_names[0]].keys())

    per_fold_block = {}
    for f in folds:
        subs = [p for p in per_subject if p["fold"] == f]
        rs = by_fold[f]
        primary_f, secondary_f = _group_vectors(rs)
        per_fold_block[str(f)] = {
            "n_subjects": len(subs),
            "subject_level": {
                pn: {mk: _agg([p["pairs"][pn][mk] for p in subs])
                     for mk in metric_keys} for pn in pair_names},
            "group": {"primary": _group_pairs(primary_f),
                      "secondary": _group_pairs(secondary_f)},
        }

    pooled_subject_level = {
        pn: {**{mk: _agg([p["pairs"][pn][mk] for p in per_subject])
                for mk in metric_keys}}
        for pn in pair_names}

    # --- 组级两口径（全线池化） ---
    primary, secondary = _group_vectors(recs)
    group_pooled = {"primary": _group_pairs(primary),
                    "secondary": _group_pairs(secondary)}

    # --- 比例性诊断（评审 A-1：区分 σ² 缩放构造性稳定与非线性重排） ---
    ratios = [p["ratio_I07_over_I03"] for p in per_subject]
    proportionality = {
        "reference_ratio": float(RATIO_QUAD_REF),
        "reference_note": "linear low-dose response I_c ~ sigma^2 implies "
                          "I(0.7)/I(0.3) ~ (0.7/0.3)^2 = 5.444; deviations = "
                          "nonlinearity evidence",
        "per_subject_ratio_I07_over_I03": {
            "median": _agg([r["median"] for r in ratios]),
            "q25": _agg([r["q25"] for r in ratios]),
            "q75": _agg([r["q75"] for r in ratios]),
            "n_excluded_den_le_eps_mean": float(
                np.mean([r["n_excluded_den_le_eps"] for r in ratios]))},
        "group_loglog_slope_pooled_primary": {
            _pair_name(i, j): _loglog_slope(primary, i, j)
            for i, j in SIGMA_PAIRS},
        "reading_note": ("slope ~2 / ratio ~5.44 = constructive sigma^2 scaling "
                         "stability; high cross-sigma rank correlation under "
                         "slope~2 alone is NOT nontrivial robustness — separate "
                         "(1) sigma^2 scaling stability, (2) non-proportional "
                         "reordering, (3) substantive ranking change when reading"),
    }

    # --- 诊断 + 描述性跨折（背景信息，不参与稳健性判读主口径） ---
    from scipy.stats import spearmanr
    # 跨被试沿通道维拼接 → (n*128, 3)：列 = σ 档（修错：np.stack 后 imp_all[:, j]
    # 取到的是第 j 个被试的整张 128×3 矩阵（三 σ 混合），sigma 键实为被试号）
    imp_all = np.concatenate([r["impact"] for r in recs], axis=0)
    mono_total = int(sum(p["monotone_violations"] for p in per_subject))
    sigma0_pass_rate = float(np.mean(
        [p["sigma0_identity_pass"] for p in per_subject]))
    fold_primary = {f: _group_vectors(by_fold[f])[0] for f in folds}
    xf = {str(f1): {str(f2): float(
              spearmanr(fold_primary[f1][:, 1], fold_primary[f2][:, 1]).statistic)
          for f2 in folds} for f1 in folds}
    summary = {
        "task": "10-30-critical-node-ranking robustness summary",
        "line": a.line,
        "n_subjects": len(recs),
        "folds": {str(f): len(by_fold[f]) for f in folds},
        "n_failed_excluded": len(failed),
        "failed_subjects": failed,
        "sigmas": list(SIGMAS),
        "caliber": {
            "subject_level": "per-subject 128-dim impact vectors: Spearman + "
                             "Kendall tau-b + top-K overlap (top-K by impact "
                             "ranking); per-fold mean+-sd and line-pooled",
            "group_primary": "mean of raw impact vectors (per fold and "
                             "line-pooled) — REPORT main caliber",
            "group_secondary": "mean of per-subject within-subject rank vectors "
                               "(rank 1 = largest impact) — aggregation-"
                               "robustness secondary caliber",
            "chance_level": "random top-K overlap = K/128 (7.8% @K=10, 15.6% "
                            "@K=20)",
        },
        "per_fold": per_fold_block,
        "pooled_subject_level": pooled_subject_level,
        "group_pooled": group_pooled,
        "proportionality": proportionality,
        "diagnostics": {
            "monotone_violations_total": mono_total,
            "monotone_violations_per_subject_max": int(max(
                p["monotone_violations"] for p in per_subject)),
            "sigma0_identity_pass_rate": sigma0_pass_rate,
            "n_source_failures_total": int(sum(
                p["n_source_failures"] for p in per_subject)),
            "impact_distribution": {
                f"sigma={s:g}": {"min": float(imp_all[:, j].min()),
                                 "median": float(np.median(imp_all[:, j])),
                                 "max": float(imp_all[:, j].max())}
                for j, s in enumerate(SIGMAS)},
        },
        "cross_fold_descriptive": {
            "sigma": 0.5, "matrix": xf,
            "note": "pairwise Spearman between per-fold group primary impact "
                    "vectors at sigma=0.5 — DESCRIPTIVE background only (per-"
                    "fold theta are different surrogate instances), not a "
                    "robustness readout"},
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    summary = sanitize_for_json(summary)
    rob_dir = out_root / "robustness"
    rob_dir.mkdir(parents=True, exist_ok=True)
    out_json = rob_dir / f"{a.line}_robustness.json"
    with open(out_json, "w") as fh:
        json.dump(summary, fh, indent=1, ensure_ascii=False, allow_nan=False)

    # --- top-20 名单并置（组平均主口径，design §3.3） ---
    md = [f"# {a.line} top-20 关键节点（组平均主口径：原始影响向量均值）", "",
          "通道号 = 数据通道 index（0-based，128 通道）；rank 1 = 全局影响最大。",
          "主线 = 全线池化；各折为不同 θ 实例 + test 互斥 ⇒ 折间并置为 "
          "DESCRIPTIVE（evaluation-calibers 跨折并置口径纪律）。", ""]
    top = {j: np.argsort(-primary[:, j], kind="stable")[:20]
           for j in range(len(SIGMAS))}
    md += ["## 全线池化（primary caliber，n={}）".format(len(recs)), "",
           "| rank | sigma=0.3 | sigma=0.5 | sigma=0.7 |",
           "|---:|---:|---:|---:|"]
    for r in range(20):
        md.append(f"| {r + 1} | {top[0][r]} | {top[1][r]} | {top[2][r]} |")
    ov = {pn: len(set(top[i][:20].tolist()) & set(top[j][:20].tolist()))
          for pn, (i, j) in zip(pair_names, SIGMA_PAIRS)}
    md += ["", "三 σ top-20 通道身份重合：" +
           "，".join(f"{pn}={n}/20" for pn, n in ov.items()) +
           f"（机会水平 ≈ 20/128 = {20 / 128:.1%}）", ""]
    for f in folds:
        pf, _ = _group_vectors(by_fold[f])
        tf = {j: np.argsort(-pf[:, j], kind="stable")[:20]
              for j in range(len(SIGMAS))}
        md += [f"## fold{f}（n={len(by_fold[f])}）", "",
               "| rank | sigma=0.3 | sigma=0.5 | sigma=0.7 |",
               "|---:|---:|---:|---:|"]
        for r in range(20):
            md.append(f"| {r + 1} | {tf[0][r]} | {tf[1][r]} | {tf[2][r]} |")
        md.append("")
    out_md = rob_dir / f"{a.line}_top_nodes.md"
    out_md.write_text("\n".join(md), encoding="utf-8")

    # 紧凑摘要行
    pp = group_pooled["primary"]
    r_med = proportionality["per_subject_ratio_I07_over_I03"]["median"]["mean"]
    sl = proportionality["group_loglog_slope_pooled_primary"][
        _pair_name(0, 2)]["slope"]
    fold_counts = " ".join(f"f{f}:{len(by_fold[f])}" for f in folds)
    print(f"[summary] {a.line} n={len(recs)} [{fold_counts}] "
          f"failed_excluded={len(failed)} | subj-rho pooled: " +
          " | ".join(f"{pn} {pooled_subject_level[pn]['spearman']['mean']:+.4f}"
                     f"±{pooled_subject_level[pn]['spearman']['sd']:.4f}"
                     for pn in pair_names) +
          " | group-rho(primary): " +
          " | ".join(f"{pn} {pp[pn]['spearman']:+.4f}" for pn in pair_names) +
          f" | ratio I(0.7)/I(0.3) median-of-medians {r_med:.3f} "
          f"(ref {RATIO_QUAD_REF:.3f}) | loglog slope 0.3->0.7 {sl:.3f} "
          f"(linear=>2) | mono_viol_total={mono_total} "
          f"sigma0_pass={sigma0_pass_rate:.3f} -> {out_json}", flush=True)


# =============================================================================
# figs 模式：≥3 张可读图（matplotlib Agg，dpi>=150，design §5/S5）
# =============================================================================

def run_figs(a) -> None:
    import matplotlib
    matplotlib.use("Agg")          # 后端先于 pyplot 导入设置
    import matplotlib.pyplot as plt
    from scipy.stats import rankdata

    out_root = Path(a.out_dir)
    figs_dir = out_root / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)
    recs, failed = _iter_subject_jsons(out_root, a.line)
    primary, _secondary = _group_vectors(recs)
    C = primary.shape[0]
    pair_names = [_pair_name(i, j) for i, j in SIGMA_PAIRS]

    # --- 图 1：组平均影响谱（128 通道三 σ 并置，按 σ=0.5 降序） ---
    order = np.argsort(-primary[:, 1], kind="stable")
    fig, ax = plt.subplots(figsize=(12, 5))
    for j, s in enumerate(SIGMAS):
        ax.plot(np.arange(C), primary[order, j], lw=1.3,
                label=f"sigma={s:g}", alpha=0.9)
    ax.set_xlabel("channel (ordered by sigma=0.5 group-mean impact, descending)")
    ax.set_ylabel("group-mean impact I_c(sigma)")
    ax.set_title(f"{a.line}: group-mean impact spectrum (primary caliber, "
                 f"n={len(recs)})\nabsolute magnitude grows with sigma; "
                 f"x-axis ordered by sigma=0.5")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(figs_dir / f"{a.line}_impact_spectrum.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # --- 图 2：rank-rank 散点（组平均主口径三对，Spearman 入图例） ---
    from scipy.stats import spearmanr
    ranks_g = np.stack([rankdata(-primary[:, j]) for j in range(len(SIGMAS))],
                       axis=1)   # rank 1 = 影响最大
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True, sharey=True)
    for ax, (i, j), pn in zip(axes, SIGMA_PAIRS, pair_names):
        rho = float(spearmanr(primary[:, i], primary[:, j]).statistic)
        ax.scatter(ranks_g[:, i], ranks_g[:, j], s=10, alpha=0.6, edgecolors="none")
        ax.set_title(pn)
        ax.set_xlabel(f"rank at sigma={SIGMAS[i]:g}")
        ax.set_ylabel(f"rank at sigma={SIGMAS[j]:g}")
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.legend([f"Spearman rho = {rho:+.4f}"], loc="lower right", fontsize=9)
        ax.grid(alpha=0.3)
    fig.suptitle(f"{a.line}: rank-rank scatter, group primary caliber "
                 f"(rank 1 = most critical at origin corner)")
    fig.savefig(figs_dir / f"{a.line}_rank_rank_scatter.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # --- 图 3：逐被试 ρ 分箱（折分组箱线；两对 σ 至少） ---
    folds = sorted({r["fold"] for r in recs})
    subj_pairs = [(r["fold"], _subject_pairs(r)) for r in recs]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, (i, j), pn in zip(axes, ((0, 1), (1, 2)),
                              [_pair_name(0, 1), _pair_name(1, 2)]):
        data = [[sp[pn]["spearman"] for f, sp in subj_pairs if f == f0]
                for f0 in folds]
        ax.boxplot(data, tick_labels=[f"f{f}" for f in folds])
        ax.set_title(f"per-subject Spearman rho, {pn}")
        ax.set_xlabel("fold")
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("per-subject Spearman rho")
    fig.suptitle(f"{a.line}: per-subject cross-sigma rank consistency by fold "
                 f"(n={len(recs)})")
    fig.savefig(figs_dir / f"{a.line}_subject_rho_by_fold.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # --- 图 4：top-K 交叠曲线（K=5..40，三对 + 机会水平 K/128 虚线） ---
    rk = [np.argsort(-primary[:, j], kind="stable")
          for j in range(len(SIGMAS))]
    Ks = np.arange(5, 41)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for (i, j), pn in zip(SIGMA_PAIRS, pair_names):
        ov = [len(set(rk[i][:K].tolist()) & set(rk[j][:K].tolist())) / K
              for K in Ks]
        ax.plot(Ks, ov, marker="o", ms=3, lw=1.3, label=pn)
    ax.plot(Ks, Ks / C, "k--", lw=1.2, label="chance K/128")
    ax.set_xlabel("K (top-K channels)")
    ax.set_ylabel("top-K overlap |A intersect B| / K")
    ax.set_title(f"{a.line}: top-K overlap vs K (group primary caliber)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(figs_dir / f"{a.line}_topk_overlap.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    print(f"[figs] {a.line} n={len(recs)} (failed excluded={len(failed)}) -> "
          f"4 png under {figs_dir}", flush=True)


def main() -> int:
    a = parse_args()
    if a.mode == "main":
        run_main(a)
    elif a.mode == "summary":
        run_summary(a)
    else:
        run_figs(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
