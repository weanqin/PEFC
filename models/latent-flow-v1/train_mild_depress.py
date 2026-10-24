#!/usr/bin/env python3
"""train_mild_depress.py — latent-flow-v1 × Mild Depression 两阶段训练入口（包装层）.

任务 10-21-latent-flow-mild-train（design §5）入口；10-24-latent-flow-mild-w500
起新增前缀窗口（--window-len）与 stage2 训练面（--unfreeze-decoder /
--anti-uniform-lambda）参数，默认值全 = 旧行为（bit 兼容）。Stage 1 = VAE
（encoder+proj+decoder，KL 退火）；Stage 2 = 冻结 VAE、潜空间 CFM 流匹配
（默认 mexhat-graph，geometry = MNE GSN-HydroCel-128 模板；--dense-fallback
反切 dense，通道序 E 序假设 D4 已签认）。

数据口径（R2 + 10-24 prd R1，全部入口硬断言，失败即拒跑）：
  - manifest 只读复用 splits/mild_5fold_seed42.json（生成一次即钉死，
    make_mild_manifest.py 生成；此处全量断言重放，不重新划分）；
  - trial 预切、不重新切窗：每 trial 即 1500 点窗（obs=x_p 1125 +
    forecast=x_f 375，比例 3:1 由数据本身保证）；
  - 前缀窗口可参数化（--window-len，默认 1500 = 全 trial，旧行为不变）：
    每 trial 取拼接后 x_p‖x_f 的前 window_len 点（0 < window_len <= 1500；
    数据文件 x_f 字段不参与、仅契约保证 trial 内 0–1499 连续相接），
    10-24 任务新口径 = 500（obs 375 / forecast 125，与包内 MODMA 原生
    口径同构 23.4 样本/潜步）；window_len=1500 时逐位 = 旧行为（默认兼容）；
  - 窗内划分可参数化（--obs-len/--forecast-len，默认 1125/375 = 数据
    x_p/x_f 原切分；硬断言 obs+forecast=window_len 且均为正——新值是对
    前 window_len 点前缀窗的重新切分（obs=前 obs_len，forecast=后
    forecast_len），数据文件本身的形状断言口径不变）；
  - 归一化 = 训练折 28 例全部 trial 前 window_len 点逐通道 mean/std
    （wrapper 内 float64 sum/sumsq，累加域随窗口切片；std==0→1.0，
    严格 held-out；不调用包内 compute_train_norm_stats——其 reader 面向
    连续 (C,N) 文件，trial 文件不适用，口径等价、代码面在 wrapper）。

stage2 训练面（10-24 任务起新增，默认全 off = 旧路径行为不动）：
  - --unfreeze-decoder：decoder+flow 联合训练（仅冻结 encoder+proj），
    走 wrapper 本地 _run_stage2_mild（design §3.2：不挂 KL 退火——encoder
    冻结 ⇒ KL 无可训梯度，β 恒 0；history 由 wrapper 自写，不触包内
    logs/mild/）；默认关 = 原包内 _run_stage2 全冻结路径原样；
  - --anti-uniform-lambda（默认 0.0）：经包内 Stage2Step 生效，stage2
    mexhat/dog 流 loss 增 uni_reg 项；与 --dense-fallback 互斥（dense 流
    无 graph_mixer，正则会静默失效）；
  - --flow-exit-gnn（10-24-latent-flow-exit-gnn 新增，默认关 = 原架构 bit
    兼容）：stage2 标准构建后以同参 FlowVelocityNetExitGNN 替换 model.flow
    （变体子类 flow_exit_gnn.py：出口 graph_out → K 轮共享 A_eff 消息传递 +
    零初始化 exit_out，初始 v≡0 恒等起步；A_eff 与入口共享、梯度双路累计）；
    仅 stage2 且 mexhat/dog 流有效（构造断言拒建）；口径守卫将其纳入新口径
    特征（须显式 --run-dir）；
  - 口径守卫：新口径特征（window_len != 1500 或 unfreeze-decoder 或
    anti-uniform != 0 或 flow-exit-gnn）而 --run-dir 解析为 OUT_BASE 默认值时
    拒跑（防漏传 --run-dir 误写上轮受保护目录树）。

训练循环零复制：直接 import 复用包内 Stage1Step/Stage2Step/_KLAnnealCallback/
_run_stage1/_run_stage2（unfreeze-decoder 分支除外——包内冻结逻辑只有
全冻结/全解冻两态，均不满足「仅冻结 encoder」，故复刻为本地 runner）。
模型构建复刻 train_edra_fmt.build_model_fmt 但 n_channels=128 /
obs_len/forecast_len 可参数化（缺省 1125/375）；测试集 smoke 评估为
wrapper 自带（包内 _smoke_eval 硬编码 375，不得 import）。

产物默认落 outputs/10-21-latent-flow-mild-train/run_fold{N}/：
  configs/stage{1,2}_config.json  checkpoints/fold{N}_mild_*.pt
  logs/*_history.json（stage1 wrapper 自写；stage2 默认从包内 logs/mild/
  复制——训练器内置行为，变体级 .gitignore 已排除 logs/；--unfreeze-decoder
  时改由 wrapper 自写，不触包内目录）。

Run（正式，nohup 由调用方包）：
  python3 /root/npi/models/latent-flow-v1/train_mild_depress.py --fold 0 --stage 1
  python3 /root/npi/models/latent-flow-v1/train_mild_depress.py --fold 0 --stage 2
  # 10-24 前缀窗 500 新口径（--run-dir 必须显式，口径守卫强制）：
  python3 /root/npi/models/latent-flow-v1/train_mild_depress.py --fold 0 --stage 2 \
    --window-len 500 --obs-len 375 --forecast-len 125 \
    --unfreeze-decoder --anti-uniform-lambda 0.1 --run-dir <本轮目录>
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
import scipy.io as sio
import torch
import torch.utils.data as data

VARIANT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(VARIANT_DIR))  # 使 latent_flow 可导入

from latent_flow.config import CHECKPOINT_DIR  # noqa: E402
from latent_flow.data.modma import collate_fn  # noqa: E402
from latent_flow.data.norm_stats import _train_files_hash  # noqa: E402
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
from latent_flow.training.trainer import (  # noqa: E402
    BaseTrainer,
    PrintingCallback,
)
from flow_exit_gnn import FlowVelocityNetExitGNN  # noqa: E402  (变体子类，同目录)

# --- mild 契约常量（.trellis/spec/data/mild-depression.md） ---
OBS_LEN = 1125
FORECAST_LEN = 375
WINDOW = 1500                     # 每 trial 即一个 1500 点窗
TRIALS = 30                       # 每例 trial 数
N_CHANNELS = 128
FS = 250
LATENT_DIM = 64
T_LAT = 16
MILD_DATA_ROOT = Path("/root/autodl-tmp/Mild Depression")
MILD_GEOMETRY_PATH = MILD_DATA_ROOT / "geometry_mild.pt"
MANIFEST_DEFAULT = VARIANT_DIR / "splits" / "mild_5fold_seed42.json"
OUT_BASE = Path("/root/npi/outputs/10-21-latent-flow-mild-train")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--fold", type=int, required=True, choices=(0, 1, 2, 3, 4))
    p.add_argument("--stage", type=int, required=True, choices=(1, 2))
    p.add_argument("--run-dir", type=str, default=None,
                   help=f"默认 {OUT_BASE}/run_fold<fold>/")
    p.add_argument("--manifest", type=str, default=str(MANIFEST_DEFAULT))
    # --- 窗内划分（默认 = 数据 x_p/x_f 原切分 1125/375；10-21 Run 005 起 5:1 =
    #     1250/250；10-24 前缀窗口径 500 = 375/125；obs+forecast 硬断言 =
    #     --window-len） ---
    p.add_argument("--window-len", type=int, default=WINDOW,
                   help=f"前缀窗口长度（默认 {WINDOW} = 全 trial 旧行为不变；每 trial "
                        f"取拼接后前 window_len 点，归一化累加域同步切片；"
                        f"0 < window_len <= {WINDOW}）")
    p.add_argument("--obs-len", type=int, default=OBS_LEN,
                   help=f"obs 段长度（默认 {OBS_LEN}；与 --forecast-len 之和必须为 --window-len）")
    p.add_argument("--forecast-len", type=int, default=FORECAST_LEN,
                   help=f"forecast 段长度（默认 {FORECAST_LEN}；与 --obs-len 之和必须为 --window-len）")
    # 旧库 CLI 默认超参（D3 拍板：epochs 50/patience 10/lr 1e-3/batch 32/
    # beta_kl 1e-4/seed 42）
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
                        "<run-dir>/checkpoints/fold<fold>_mild_vae_best.pt)")
    p.add_argument("--limit-subjects", type=int, default=None,
                   help="冒烟截断：每集只取前 N 例（正式跑禁用）")
    # --- stage-2 流变体（默认 mexhat-graph；--dense-fallback 反切 dense） ---
    p.add_argument("--dense-fallback", action="store_true",
                   help="Stage-2 改用 dense 流（默认 mexhat-graph 的反向开关）")
    p.add_argument("--graph-kernel", type=str, default="mexhat",
                   choices=("gaussian", "mexhat", "dog"))
    p.add_argument("--graph-k", type=int, default=2)
    p.add_argument("--graph-out-mix-alpha", type=float, default=0.5)
    p.add_argument("--mexhat-gamma", type=float, default=0.1)
    p.add_argument("--mexhat-gamma-m", type=float, default=1.0)
    # --- stage-2 训练面（10-24 任务新增；默认全 off = 旧路径行为不动） ---
    p.add_argument("--anti-uniform-lambda", type=float, default=0.0,
                   help="Stage-2 mexhat/dog 流 A_eff 反均匀正则权重（经包内 "
                        "Stage2Step 生效，loss 增 uni_reg 项；0=off；与 "
                        "--dense-fallback 互斥）")
    p.add_argument("--unfreeze-decoder", action="store_true",
                   help="Stage-2 decoder+flow 联合训练（仅冻结 encoder+proj；走 "
                        "wrapper 本地 _run_stage2_mild，history 由 wrapper 自写；"
                        "默认关 = 包内 _run_stage2 全冻结路径原样）")
    # --- stage-2 流出口变体（10-24-latent-flow-exit-gnn；默认关 = 原架构） ---
    p.add_argument("--flow-exit-gnn", action="store_true",
                   help="Stage-2 flow 出口 GNN 图混合变体：标准构建后以同参 "
                        "FlowVelocityNetExitGNN 替换 model.flow（出口 graph_out "
                        "→ K 轮共享 A_eff 消息传递 + 零初始化 exit_out，初始 "
                        "v≡0；A_eff 与入口共享，梯度双路累计）；仅 --stage 2 且 "
                        "mexhat/dog 流有效（与 --dense-fallback 互斥）；属新口径"
                        "特征，须显式 --run-dir（口径守卫）")
    return p.parse_args(argv)


# =============================================================================
# Manifest 解析 + 硬断言重放（R2 / AC2；与 make_mild_manifest.py 同口径，
# 失败即拒跑）
# =============================================================================

def _check_subject_file(path: Path, group: str, sid: int):
    """逐例字段断言重放（同生成器口径）。"""
    assert path.exists(), f"subject file missing: {path}"
    m = sio.loadmat(path)
    xp, xf = m["x_p"], m["x_f"]
    assert xp.shape == (TRIALS, N_CHANNELS, OBS_LEN), \
        f"{path}: x_p {xp.shape} != ({TRIALS},{N_CHANNELS},{OBS_LEN})"
    assert xf.shape == (TRIALS, N_CHANNELS, FORECAST_LEN), \
        f"{path}: x_f {xf.shape} != ({TRIALS},{N_CHANNELS},{FORECAST_LEN})"
    assert m["fs"].item() == FS, f"{path}: fs {m['fs'].item()} != {FS}"
    label = str(np.asarray(m["label"]).ravel()[0])
    assert label == group, f"{path}: label {label!r} != {group!r}"
    code = 1 if group == "MDD" else 0
    assert m["label_code"].item() == code, \
        f"{path}: label_code {m['label_code'].item()} != {code}"
    assert m["subject_id"].item() == sid, \
        f"{path}: subject_id {m['subject_id'].item()} != {sid}"
    assert m["x_p_start_sample"].item() == 0 and \
        m["x_f_start_sample"].item() == OBS_LEN, \
        f"{path}: x_p/x_f start sample mismatch"
    return TRIALS


def load_manifest(manifest_path: str, fold: int):
    man = json.loads(Path(manifest_path).read_text())
    assert man.get("dataset") == "mild_depress", f"dataset != mild_depress"
    assert man.get("seed") == 42, f"manifest seed != 42"
    assert man.get("n_folds") == 5, f"n_folds != 5"
    subjects = {int(k): v for k, v in man["subjects"].items()}
    folds = {int(f["fold"]): f for f in man["folds"]}
    assert sorted(folds) == [0, 1, 2, 3, 4], f"folds != 0..4: {sorted(folds)}"

    assert len(subjects) == 40, f"manifest subjects != 40: {len(subjects)}"
    groups = {sid: v["group"] for sid, v in subjects.items()}
    assert set(groups.values()) == {"HC", "MDD"}, "unexpected group labels"
    n_mdd = sum(1 for v in groups.values() if v == "MDD")
    assert n_mdd == 20 and len(groups) - n_mdd == 20, f"HC/MDD != 20/20: {n_mdd}"
    for sid, v in subjects.items():
        expected_code = 1 if v["group"] == "MDD" else 0
        assert v["label_code"] == expected_code, f"sid {sid}: label_code mismatch"

    # --- 逐例字段断言重放（40 例全量） ---
    trial_record = {}
    for sid in sorted(subjects):
        v = subjects[sid]
        trial_record[str(sid)] = _check_subject_file(Path(v["file"]), v["group"], sid)

    # --- 折结构断言重放（含跨折 test 互斥覆盖） ---
    all_ids = set(subjects)
    test_union = []
    for i in range(5):
        f = folds[i]
        tr, va, te = set(f["train"]), set(f["val"]), set(f["test"])
        assert len(f["train"]) == 28, f"fold{i} train != 28"
        assert len(f["val"]) == 4, f"fold{i} val != 4"
        assert len(f["test"]) == 8, f"fold{i} test != 8"
        assert not (tr & va or tr & te or va & te), f"fold{i} overlap"
        assert tr | va | te == all_ids, f"fold{i} union != all 40"
        n_mdd_te = sum(1 for s in f["test"] if groups[s] == "MDD")
        assert n_mdd_te == 4 and len(f["test"]) - n_mdd_te == 4, \
            f"fold{i} test not 4+4 stratified"
        test_union.extend(f["test"])
    assert len(test_union) == len(set(test_union)) == 40, \
        "5-fold test sets not mutually exclusive / not covering all 40"

    f = folds[fold]
    print(f"[manifest] {manifest_path}")
    print(f"[manifest] 40 subjects (HC 20 / MDD 20); fold{fold} = "
          f"28/4/8 OK (train/val/test disjoint, union = all 40); "
          f"5-fold test mutually exclusive, covers all 40; "
          f"all 40 per-file asserts OK ({TRIALS} trials, "
          f"x_p {OBS_LEN}/x_f {FORECAST_LEN}, fs {FS})")

    def files_of(ids):
        return [subjects[s]["file"] for s in ids]

    return (man, subjects,
            files_of(f["train"]), files_of(f["val"]), files_of(f["test"]),
            folds[fold], trial_record)


# =============================================================================
# Trial 级 Dataset（design §2.3，wrapper 内嵌；产出契约与包内窗口契约逐键同构）
# =============================================================================

class MildTrialDataset(data.Dataset):
    """预切 trial 数据集：每 trial = 前 window_len 点前缀窗（默认全 1500 点）。

    逐例 loadmat → concat(x_p, x_f, axis=2) → (30,128,1500) → 归一化
    (x-mean)/std（(128,1) 逐通道，train 折统计）→ 每 trial 取前
    window_len 点转 (window_len,128) float32。产出 dict：x_obs=
    (obs_len,128)（x_target 前 obs_len 视图）、x_target=(window_len,128)、
    t_obs/t_target（arange*dt, dt=1/250，t_target 长度 = window_len）。

    obs_len/forecast_len/window_len 缺省 = 模块常量（1125/375/1500，数据
    原切分全窗 = 旧行为）；window_len < 1500 即前缀窗口（10-24 口径 500：
    数据 x_f 字段不参与，仅契约保证 trial 内连续）；obs/fc 对前缀窗再切分
    （obs=前段、forecast=后段），数据文件本身的形状断言不受影响。
    """

    def __init__(self, subject_files, norm_stats,
                 obs_len: int = OBS_LEN, forecast_len: int = FORECAST_LEN,
                 window_len: int = WINDOW):
        super().__init__()
        assert 0 < window_len <= WINDOW, \
            f"window_len must be in (0, {WINDOW}]: {window_len}"
        assert obs_len > 0 and forecast_len > 0, \
            f"obs_len/forecast_len must be positive: {obs_len}/{forecast_len}"
        assert obs_len + forecast_len == window_len, \
            f"obs_len + forecast_len ({obs_len}+{forecast_len}) != {window_len}"
        self.obs_len = obs_len
        self.forecast_len = forecast_len
        self.window_len = window_len
        mean, std = norm_stats                    # each (128, 1) float32
        assert mean.shape == std.shape == (N_CHANNELS, 1)
        dt = 1.0 / FS
        self.t_obs = torch.arange(obs_len, dtype=torch.float32) * dt
        self.t_target = torch.arange(window_len, dtype=torch.float32) * dt
        self.trials = []
        for fp in subject_files:
            m = sio.loadmat(fp)
            x = np.concatenate([m["x_p"], m["x_f"]], axis=2)   # (30,128,1500)
            assert x.shape == (TRIALS, N_CHANNELS, WINDOW), \
                f"{fp}: concat shape {x.shape} != ({TRIALS},{N_CHANNELS},{WINDOW})"
            x = (x - mean[None, :, :]) / std[None, :, :]        # broadcast per ch
            for ti in range(x.shape[0]):
                xt = torch.tensor(
                    np.ascontiguousarray(x[ti, :, :window_len].T),
                    dtype=torch.float32)                        # (window_len,128)
                self.trials.append(xt)
        print(f"MildTrialDataset: {len(subject_files)} subjects, "
              f"{len(self.trials)} trials (window={window_len}, obs={obs_len}, "
              f"forecast={forecast_len}; norm: fixed train-fold stats)")

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, idx):
        x_target = self.trials[idx]                # (window_len,128)
        return {
            "x_obs": x_target[:self.obs_len],      # (obs_len,128) view
            "x_target": x_target,
            "t_obs": self.t_obs,
            "t_target": self.t_target,
        }


# =============================================================================
# 归一化（design §2.4：wrapper 内实现，train 折 28 例前 window_len 点
# float64 累加；window_len=1500 时与全长统计逐位一致 = 默认兼容）
# =============================================================================

def compute_mild_train_norm(train_files, window_len: int = WINDOW):
    """train 折逐通道 mean/std；累加域 = 拼接后 trial 的前 window_len 点。"""
    assert 0 < window_len <= WINDOW, \
        f"window_len must be in (0, {WINDOW}]: {window_len}"
    sum_ = np.zeros((N_CHANNELS,), dtype=np.float64)
    sumsq = np.zeros((N_CHANNELS,), dtype=np.float64)
    count = 0.0
    for fp in train_files:
        m = sio.loadmat(fp)
        x = np.concatenate([m["x_p"], m["x_f"]], axis=2).astype(np.float64)
        assert x.shape == (TRIALS, N_CHANNELS, WINDOW), f"{fp}: shape {x.shape}"
        xw = x[:, :, :window_len]                  # 前缀窗切片（1500 = 全长）
        sum_ += xw.sum(axis=(0, 2))
        sumsq += (xw ** 2).sum(axis=(0, 2))
        count += xw.shape[0] * xw.shape[2]
    mean = sum_ / count
    var = np.clip(sumsq / count - mean ** 2, a_min=0.0, a_max=None)
    std = np.sqrt(var)
    std[std == 0] = 1.0                            # 护栏，对齐包内口径
    return (mean.reshape(-1, 1).astype(np.float32),
            std.reshape(-1, 1).astype(np.float32))


# =============================================================================
# Model（复刻 train_edra_fmt.build_model_fmt，n_channels=128 / 1125 / 375）
# =============================================================================

def build_model_mild(device, args, stage: int,
                     obs_len: int = OBS_LEN, forecast_len: int = FORECAST_LEN):
    """复刻 train_edra_fmt.build_model_fmt；obs_len/forecast_len 缺省 = 模块
    常量（1125/375），可由 CLI --obs-len/--forecast-len 传入新划分。"""
    # stage1 不训练流：dense 构建即可（无需 geometry）；stage2 默认 mexhat-graph
    # （geometry_mild.pt 显式路径加载；E 序假设 D4 签认）。
    graph_mode = (stage == 2) and not args.dense_fallback
    a_hat, coords = None, None
    if graph_mode:
        if not MILD_GEOMETRY_PATH.exists():
            raise SystemExit(
                f"Stage-2 graph-mode needs {MILD_GEOMETRY_PATH} — run "
                f"build_mild_geometry.py first (S2 gate must have passed).")
        coords_all, names = load_geometry(path=str(MILD_GEOMETRY_PATH))
        assert coords_all.shape == (N_CHANNELS, 3), \
            f"geometry shape {coords_all.shape}"
        assert names == [f"E{i}" for i in range(1, N_CHANNELS + 1)], \
            "geometry channel names != E1..E128 canonical order"
        a_hat = build_normalized_adj(coords_all).to(device)
        if args.graph_kernel in ("mexhat", "dog"):
            coords = coords_all.to(device)
        print(f"Graph-mode flow: kernel={args.graph_kernel}, K={args.graph_k}, "
              f"out mix_alpha={args.graph_out_mix_alpha}, "
              f"mexhat_gamma={args.mexhat_gamma}, adjacency {tuple(a_hat.shape)} "
              f"(geometry={MILD_GEOMETRY_PATH}, E-order assumption D4 signed)")
    model = OCLatentFlow(
        n_channels=N_CHANNELS, latent_dim=LATENT_DIM, t_lat=T_LAT,
        obs_len=obs_len, forecast_len=forecast_len,
        flow_graph_mode=graph_mode, flow_graph_k=args.graph_k, flow_a_hat=a_hat,
        flow_graph_out_mix_alpha=args.graph_out_mix_alpha,
        flow_graph_kernel=args.graph_kernel, flow_coords=coords,
        flow_mexhat_gamma=args.mexhat_gamma, flow_w_init=None,
        flow_mexhat_gamma_m=args.mexhat_gamma_m, flow_dog_init=None,
        beta_kl_final=args.beta_kl, use_fc_decoder=False,
    ).to(device)
    flow_tag = ("graph-" + args.graph_kernel) if graph_mode else "dense"
    if getattr(args, "flow_exit_gnn", False):
        # 10-24-latent-flow-exit-gnn：标准 OCLatentFlow 构建后以同参变体子类
        # 替换 model.flow（graph 参数同传；A_eff 与入口共享 = graph_mixer 单份，
        # 梯度双路累计；hidden_ch 取自刚构建的标准流，保证同参）
        assert stage == 2, \
            "--flow-exit-gnn 仅对 --stage 2 有意义（stage1 不训练流）"
        assert graph_mode, \
            "--flow-exit-gnn 与 --dense-fallback 互斥：出口消息传递需要 graph_mixer"
        assert args.graph_kernel in ("mexhat", "dog"), (
            f"--flow-exit-gnn 仅支持 mexhat/dog 流（graph_mixer 须有 "
            f"effective_adjacency），got {args.graph_kernel!r}")
        model.flow = FlowVelocityNetExitGNN(
            n_channels=N_CHANNELS, latent_dim=LATENT_DIM, t_lat=T_LAT,
            hidden_ch=model.flow.hidden_ch,
            graph_mode=True, graph_k=args.graph_k, a_hat=a_hat,
            graph_out_mix_alpha=args.graph_out_mix_alpha,
            graph_kernel=args.graph_kernel, coords=coords,
            mexhat_gamma=args.mexhat_gamma, w_init=None,
            mexhat_gamma_m=args.mexhat_gamma_m, dog_init=None).to(device)
        flow_tag += "-exit-gnn"
        print(f"Flow exit: replaced by FlowVelocityNetExitGNN "
              f"(K={args.graph_k} shared-A_eff msg rounds + zero-init "
              f"exit_out; v≡0 at init)")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,} "
          f"(stage{stage} flow={flow_tag})")
    return model


def _smoke_eval_mild(model, test_loader, device, n_batches=3,
                     obs_len: int = OBS_LEN):
    """Wrapper 自带 smoke（包内 _smoke_eval 硬编码 375，不得 import）；
    obs_len 缺省 = 模块常量，可随 --obs-len 传入新划分。"""
    model.eval()
    mae_sum, std_sum, n, seen = 0.0, 0.0, 0, 0
    with torch.no_grad():
        for batch in test_loader:
            x_obs = batch["x_obs"].to(device)
            x_target = batch["x_target"].to(device)
            x_hat = model.sample(x_obs, n_steps=10)
            x_fc = x_target[:, obs_len:, :]
            x_fc_hat = x_hat if model.use_fc_decoder else x_hat[:, obs_len:, :]
            mae_sum += (x_fc_hat - x_fc).abs().mean().item() * x_obs.shape[0]
            std_sum += (x_fc_hat.std() / (x_fc.std() + 1e-8)).item() * x_obs.shape[0]
            n += x_obs.shape[0]
            seen += 1
            if seen >= n_batches:
                break
    print(f"[smoke] test MAE={mae_sum/n:.4f}  std_ratio={std_sum/n:.4f}")


def _run_stage2_mild(model, train_loader, val_loader, args, save_path, vae_ckpt):
    """Stage-2 本地 runner（10-24 design §3.2）：decoder+flow 联合，仅冻结 encoder+proj。

    为什么不复用包内 _run_stage2：其冻结逻辑只有两态——unfreeze_vae=False
    冻结 encoder+proj+decoder（decoder 不可训），True 全解冻（含 encoder，
    且语义为 08-01 joint ablation）——均不满足「仅冻结 encoder」。本 runner
    复刻包内流程（ckpt 加载 / Stage2Step / BaseTrainer），差异三点：
      1. 冻结集 = encoder+proj（proj 不在 forward_stage2 计算图内、无梯度，
         冻结仅为显式声明，行为与「仅冻结 encoder」无差）；decoder+flow 训练；
      2. 不挂 _KLAnnealCallback：encoder 冻结 ⇒ KL 项对可训练参数无梯度，
         β 恒 0（__init__ 默认），与包内 frozen 分支行为一致；
      3. history 由 wrapper 自写（调用方落 run_dir/logs/，文件名沿用
         _history_name 约定），不触包内 logs/mild/ 目录。
    args = wrapper 钉死的 train_args 命名空间（含 anti_uniform_lambda）。
    """
    print("\n=== Stage 2 (wrapper/mild): decoder + flow joint, "
          "encoder/proj frozen ===")
    # --- stage1 ckpt 加载（复刻包内 _run_stage2：format v1 解包，strict=False，
    #     非 flow 键不匹配即报错，flow 键重初始化属图流预期） ---
    payload = torch.load(
        vae_ckpt, map_location=next(model.parameters()).device,
        weights_only=False)
    if isinstance(payload, dict) and payload.get("format_version") == 1:
        vae_state = payload["state_dict"]
    else:
        vae_state = payload
    missing, unexpected = model.load_state_dict(vae_state, strict=False)
    # flow 键重初始化容忍面（stage1 dense 流 ckpt 装载）：graph 流换装时
    # flow.* 键必然错位——graph 流缺 dense 键 / dense ckpt 多 dense 键；
    # --flow-exit-gnn 变体再缺 exit_* 键（exit_msg_layers/exit_out，前缀
    # flow.exit_ 统一覆盖）——均按「flow 键重初始化」容忍，非 flow 键
    # 不匹配仍即报错
    flow_prefixes = ("flow.input_proj", "flow.out_proj",
                     "flow.graph_mixer", "flow.graph_out", "flow.blocks",
                     "flow.exit_")
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
              f"unexpected — expected for "
              f"{'graph' if getattr(model, 'flow_graph_mode', False) else 'dense'} flow)")

    # --- 差异 1：冻结 encoder+proj（decoder+flow 可训练） ---
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    for p in model.proj.parameters():
        p.requires_grad_(False)
    # 本 wrapper 固定 use_fc_decoder=False ⇒ decoder_fc 恒 None，无额外冻结面
    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params (decoder+flow; encoder/proj frozen): "
          f"{n_trainable:,} (total {n_total:,}, frozen {n_total - n_trainable:,})")

    # --- Stage2Step + anti-uniform（差异 2：无 KL 退火，β 恒 0 默认） ---
    step = Stage2Step()
    step.anti_uniform_lambda = float(getattr(args, "anti_uniform_lambda", 0.0))
    if step.anti_uniform_lambda > 0:
        print(f"  anti-uniformity regularizer: lambda={step.anti_uniform_lambda}")
    # 与包内 _run_stage2 逐项对齐（train_args 钉死 sparse_lambda=0.0，行为不变）
    step.sparse_lambda = float(getattr(args, "sparse_lambda", 0.0))
    if step.sparse_lambda > 0:
        print(f"  L1 sparsity regularizer: lambda={step.sparse_lambda} "
              f"(on A_eff off-diag)")
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr)
    trainer = BaseTrainer(
        model=model, criterion=None, device=next(model.parameters()).device,
        train_step=step, lr=args.lr, patience=args.patience,
        save_path=save_path, use_amp=True, max_grad_norm=1.0,
        optimizer=optimizer,
        callbacks=[PrintingCallback(prefix="[stage2] ")],
    )
    history = trainer.fit(train_loader, val_loader, n_epochs=args.epochs)
    if history.best_epoch is not None:
        print(f"\nBest val loss: {history.val[history.best_epoch]['loss']:.4f}")
    if args.patience > 0:
        print(f"Checkpoint saved to {save_path}")
    return history


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
    # 窗内划分硬断言（R2：数据文件口径不变；此处只管拼接后 1500 点 trial 的
    # 前 window_len 点前缀窗的再切分）
    assert args.obs_len > 0 and args.forecast_len > 0, \
        f"--obs-len/--forecast-len must be positive: {args.obs_len}/{args.forecast_len}"
    assert 0 < args.window_len <= WINDOW, \
        f"--window-len must be in (0, {WINDOW}]: {args.window_len}"
    assert args.obs_len + args.forecast_len == args.window_len, \
        f"--obs-len + --forecast-len ({args.obs_len}+{args.forecast_len}) " \
        f"!= --window-len {args.window_len}"
    # --- 口径守卫（10-24 design §3.1 / review B 采纳） ---
    # dense 流无 graph_mixer，anti-uniform 正则会静默失效 ⇒ 互斥拒跑
    assert not (args.dense_fallback and args.anti_uniform_lambda > 0), \
        ("--dense-fallback 与 --anti-uniform-lambda > 0 互斥：dense 流无 "
         "graph_mixer，反均匀正则会静默失效")
    # 出口 GNN 变体守卫（10-24-latent-flow-exit-gnn）：stage1 不训练流、
    # dense/gaussian 流无 effective_adjacency ⇒ 仅 stage2 mexhat/dog 有效
    assert not (args.flow_exit_gnn and args.stage != 2), \
        "--flow-exit-gnn 仅对 --stage 2 有意义（stage1 不训练流）"
    assert not (args.dense_fallback and args.flow_exit_gnn), \
        ("--flow-exit-gnn 与 --dense-fallback 互斥：出口消息传递需要 "
         "graph_mixer.effective_adjacency")
    # 变体装载面守卫（10-24-latent-flow-exit-gnn check 补）：本地
    # _run_stage2_mild 的 VAE ckpt 白名单已补 flow.exit_ 前缀，但默认路径走
    # 包内 _run_stage2（包体零改动不可扩白名单）——exit_* 键会被误报为
    # non-flow mismatch 在装载处崩溃，须在此 fail-fast
    assert not (args.flow_exit_gnn and not args.unfreeze_decoder), \
        ("--flow-exit-gnn 须与 --unfreeze-decoder 同用：仅本地 runner "
         "_run_stage2_mild 的 VAE ckpt 白名单含 flow.exit_ 前缀；默认包内 "
         "_run_stage2 无该前缀，exit_* 键将按 non-flow mismatch 拒跑")
    # 新口径特征而 run_dir 解析为 OUT_BASE 默认值 ⇒ 拒跑（防漏传 --run-dir
    # 误写上轮受保护目录树；旧口径 + 旧默认目录组合不受影响）
    new_caliber = (args.window_len != WINDOW or args.unfreeze_decoder
                   or args.anti_uniform_lambda != 0.0 or args.flow_exit_gnn)
    default_run_dir = OUT_BASE / f"run_fold{args.fold}"
    run_dir = Path(args.run_dir) if args.run_dir else default_run_dir
    if new_caliber:
        assert args.run_dir is not None and \
            run_dir.resolve() != default_run_dir.resolve(), (
                f"新口径（window_len={args.window_len}, "
                f"unfreeze_decoder={args.unfreeze_decoder}, "
                f"anti_uniform_lambda={args.anti_uniform_lambda}, "
                f"flow_exit_gnn={args.flow_exit_gnn}）不得使用 "
                f"OUT_BASE 默认目录 {default_run_dir}（上轮受保护目录树）——"
                f"须显式传 --run-dir 指向本轮声明目录")
    (run_dir / "configs").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}  dataset: mild  fold: {args.fold}  stage: {args.stage}")
    print(f"Run dir: {run_dir}")
    print(f"[window] obs_len={args.obs_len} forecast_len={args.forecast_len} "
          f"(window={args.window_len}{' = full trial (legacy)' if args.window_len == WINDOW else ' prefix'}; "
          f"data files keep native x_p/x_f "
          f"{OBS_LEN}/{FORECAST_LEN} shapes)")
    set_seed(args.seed)

    t0 = time.time()
    (man, subjects, train_files, val_files, test_files,
     fold_record, trial_record) = load_manifest(args.manifest, args.fold)
    if args.limit_subjects:
        assert args.limit_subjects >= 1
        train_files = train_files[: args.limit_subjects]
        val_files = val_files[: args.limit_subjects]
        test_files = test_files[: args.limit_subjects]
        print(f"[SMOKE] --limit-subjects {args.limit_subjects}: "
              f"train/val/test truncated to {len(train_files)}/{len(val_files)}/"
              f"{len(test_files)} (manifest assertions ran on the FULL 40)")

    # 归一化：训练折统计（float64 累加，累加域 = 前 window_len 点，严格
    # held-out；wrapper 内实现）
    mean, std = compute_mild_train_norm(train_files, window_len=args.window_len)
    norm_stats = (mean, std)
    files_hash = _train_files_hash(train_files)
    norm_hash = hashlib.sha256(
        np.concatenate([mean.ravel(), std.ravel()]).tobytes()).hexdigest()

    train_ds = MildTrialDataset(train_files, norm_stats,
                                obs_len=args.obs_len,
                                forecast_len=args.forecast_len,
                                window_len=args.window_len)
    val_ds = MildTrialDataset(val_files, norm_stats,
                              obs_len=args.obs_len,
                              forecast_len=args.forecast_len,
                              window_len=args.window_len)
    test_ds = MildTrialDataset(test_files, norm_stats,
                               obs_len=args.obs_len,
                               forecast_len=args.forecast_len,
                               window_len=args.window_len)
    for tag, ds, files in (("train", train_ds, train_files),
                           ("val", val_ds, val_files),
                           ("test", test_ds, test_files)):
        assert len(ds) == TRIALS * len(files), (
            f"{tag}: {len(ds)} trials != {TRIALS} x {len(files)}")
    train_loader = data.DataLoader(train_ds, batch_size=args.batch_size,
                                   shuffle=True, collate_fn=collate_fn)
    val_loader = data.DataLoader(val_ds, batch_size=args.batch_size,
                                 shuffle=False, collate_fn=collate_fn)
    test_loader = data.DataLoader(test_ds, batch_size=args.batch_size,
                                  shuffle=False, collate_fn=collate_fn)
    print(f"[data] fold{args.fold} train/val/test subjects = "
          f"{len(train_files)}/{len(val_files)}/{len(test_files)}; "
          f"trials = {len(train_ds)}/{len(val_ds)}/{len(test_ds)} "
          f"(norm: train-fold stats, files_hash={files_hash[:12]}, "
          f"prep {time.time()-t0:.1f}s)")

    model = build_model_mild(device, args, stage=args.stage,
                             obs_len=args.obs_len,
                             forecast_len=args.forecast_len)

    # args 命名空间钉死（design §4：dataset="mild" 决定包内 history 落
    # logs/mild/；unfreeze_vae=False / sparse 0 / scheduler none 为旧库默认的
    # 显式固定；anti_uniform_lambda 由 CLI 透传（10-24 起可非 0）——包内
    # _run_stage2 与本地 _run_stage2_mild 均经 Stage2Step 读该值）
    train_args = argparse.Namespace(
        dataset="mild", unfreeze_vae=False,
        anti_uniform_lambda=args.anti_uniform_lambda, sparse_lambda=0.0,
        lr_scheduler="none", lr_warmup_epochs=0,
        epochs=args.epochs, lr=args.lr, patience=args.patience, seed=args.seed,
    )

    suffix = "" if args.dense_fallback else (
        "_graph_mexhat" if args.graph_kernel == "mexhat" else
        "_graph_dog" if args.graph_kernel == "dog" else "_graph")
    vae_path = args.save_path or str(
        run_dir / "checkpoints" / f"fold{args.fold}_mild_vae_best.pt")
    flow_path = args.save_path or str(
        run_dir / "checkpoints" / f"fold{args.fold}_mild_flow_best{suffix}.pt")
    save_this = vae_path if args.stage == 1 else flow_path
    Path(save_this).parent.mkdir(parents=True, exist_ok=True)  # smoke 子目录 ENOENT 防护

    geometry_sha = (_sha256_file(MILD_GEOMETRY_PATH)
                    if MILD_GEOMETRY_PATH.exists() else None)

    # stage2 训练面凭据（S1.4）：trainable 参数量按冻结面从模块计数推得
    #（与训练时 requires_grad 过滤一致；decoder_fc 恒 None——use_fc_decoder
    # 在本 wrapper 固定 False）
    stage2_cfg = None
    if args.stage == 2:
        total_n = sum(p.numel() for p in model.parameters())
        enc_n = sum(p.numel() for p in model.encoder.parameters())
        proj_n = sum(p.numel() for p in model.proj.parameters())
        if args.unfreeze_decoder:
            stage2_trainable = total_n - enc_n - proj_n
            frozen_modules = ["encoder", "proj"]
        else:
            dec_n = sum(p.numel() for p in model.decoder.parameters())
            stage2_trainable = total_n - enc_n - proj_n - dec_n
            frozen_modules = ["encoder", "proj", "decoder"]
        stage2_cfg = {
            "runner": ("wrapper _run_stage2_mild (decoder+flow joint, "
                       "encoder/proj frozen)" if args.unfreeze_decoder else
                       "package _run_stage2 (VAE frozen, flow-only)"),
            "unfreeze_decoder": args.unfreeze_decoder,
            "frozen_modules": frozen_modules,
            "anti_uniform_lambda": args.anti_uniform_lambda,
            "trainable_params": stage2_trainable,
        }

    # resolved config 落盘（训练前，可复现凭据）
    cfg = {
        "task": "latent-flow-v1 x mild-depression train wrapper "
                "(shared by tasks 10-21 / 10-24; run provenance = run_dir "
                "+ owning task experiment.md)",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_head": _git_head(),
        "fold": args.fold,
        "stage": args.stage,
        "cli": {k: v for k, v in vars(args).items()},
        "pinned_trainer_args": vars(train_args),
        "windows": {"window_size": WINDOW,
                    "window_len": args.window_len,
                    "obs_len": args.obs_len,
                    "forecast_len": args.forecast_len,
                    "data_file_split": {"x_p": OBS_LEN, "x_f": FORECAST_LEN,
                                        "note": "data files keep native "
                                                "x_p/x_f shapes; window_len = "
                                                "prefix length of the "
                                                "concatenated 1500-pt trial "
                                                "(1500 = full-length legacy); "
                                                "obs_len/forecast_len re-split "
                                                "the window_len-pt prefix"},
                    "trials_per_subject": TRIALS, "fs": FS,
                    "trial_assert_record": trial_record,
                    "counts": {"train": len(train_ds), "val": len(val_ds),
                               "test": len(test_ds)}},
        "manifest": {
            "path": args.manifest, "sha256": _sha256_file(args.manifest),
            "fold": args.fold,
            "counts": {"train": 28, "val": 4, "test": 8, "total": 40},
            "fold_record": fold_record,
            "subjects": {str(k): subjects[k] for k in sorted(subjects)},
            "subject_files": {"train": train_files, "val": val_files,
                              "test": test_files},
        },
        "norm_stats": {"source": f"train fold first-{args.window_len} samples "
                                 f"(wrapper float64 sum/sumsq over the "
                                 f"prefix window of concatenated x_p+x_f "
                                 f"trials; window_len=1500 equals "
                                 f"full-length)",
                       "files_hash": files_hash, "mean_std_sha256": norm_hash,
                       "shape": list(mean.shape)},
        "model": {"n_channels": N_CHANNELS, "latent_dim": LATENT_DIM,
                  "t_lat": T_LAT, "obs_len": args.obs_len,
                  "forecast_len": args.forecast_len,
                  "flow_variant": ("dense" if args.dense_fallback else
                                   f"graph-{args.graph_kernel}") +
                                  ("-exit-gnn" if args.flow_exit_gnn else ""),
                  "flow_exit_gnn": args.flow_exit_gnn,
                  "beta_kl_final": args.beta_kl, "use_fc_decoder": False},
        "stage2": stage2_cfg,
        "geometry": {"path": str(MILD_GEOMETRY_PATH),
                     "exists": MILD_GEOMETRY_PATH.exists(),
                     "sha256": geometry_sha,
                     "channel_order_assumption": "data ch i <-> E(i+1) (D4 signed)"},
        "checkpoints": {
            "stage1_vae": str(run_dir / "checkpoints" /
                              f"fold{args.fold}_mild_vae_best.pt"),
            "stage2_flow_default": str(run_dir / "checkpoints" /
                                       f"fold{args.fold}_mild_flow_best{suffix}.pt"),
            "this_run_save_path": save_this},
        "package_history_dir": str(CHECKPOINT_DIR.parent / "logs" / "mild"),
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
        ckpt = args.vae_ckpt or str(
            run_dir / "checkpoints" / f"fold{args.fold}_mild_vae_best.pt")
        if not Path(ckpt).exists():
            raise FileNotFoundError(
                f"Stage 2 needs a stage-1 VAE checkpoint: {ckpt} not found.")
        if args.unfreeze_decoder:
            # 10-24 新口径：decoder+flow 联合（仅冻结 encoder+proj）——本地
            # runner，history 由 wrapper 自写，不触包内 logs/mild/
            history = _run_stage2_mild(model, train_loader, val_loader,
                                       train_args, flow_path, ckpt)
            hist_path = run_dir / "logs" / _history_name(args, 2)
            history.to_json(hist_path)
            print(f"[history] stage2 written by wrapper: {hist_path}")
        else:
            # 默认兼容：原包内 _run_stage2 路径原样（全冻结 VAE、flow-only）
            history = _run_stage2(model, train_loader, val_loader, train_args,
                                  flow_path, ckpt)
            # 训练器内置已写包内 logs/mild/<stem>_history.json —— 复制进 run 目录
            src = CHECKPOINT_DIR.parent / "logs" / "mild" / \
                f"{Path(flow_path).stem}_history.json"
            dst = run_dir / "logs" / _history_name(args, 2)
            if src.exists():
                shutil.copy2(src, dst)
                print(f"[history] stage2 copied {src} -> {dst}")
            else:
                print(f"[warn] package history not found (patience=0?): {src}")
    print(f"[time] training wall time: {time.time()-t1:.1f}s")

    print(f"\n--- Smoke eval on test (a few batches, "
          f"{args.obs_len}/{args.forecast_len} caliber) ---")
    _smoke_eval_mild(model, test_loader, device, obs_len=args.obs_len)
    return 0


if __name__ == "__main__":
    sys.exit(main())
