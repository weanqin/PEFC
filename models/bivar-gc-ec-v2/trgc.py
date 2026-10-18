#!/usr/bin/env python3
"""trgc —— TRGC 净时序方向分析（用户 2026-10-18 追加；bivar-gc-ec-v2）。

从已落盘数组**纯计算**（不重跑拟合）：TRGC_s = GC_fwd_s − GC_rev_s（逐例逐有向边，
行=源/列=靶）。两件事：
  ① 组级拓扑：是否符合生理常识（闭眼静息态枕/顶枕 → 额叶沿前后轴的方向偏置），
     并对照「高信噪比通道主导」的 SNR 假象控制；
  ② 受试者间一致性生存分析：逐边单样本 t 检验（df=47）+ BH-FDR + 受试者级
     bootstrap CI（percentile 95%），数存活的「净时序方向显著边」。

判读框架（预登记，沿 rev-01 / compare_rev 口径）：TRGC = 净时序方向**读数**
（fwd−rev 差分），显著边 ≠ 生理因果确证——双变量 GC 口径 + 头皮体积传导局限
如实注明。

交付（落 OUTPUTS_ROOT/<run_id>/，refs-01 / rev-01 既有产物只读不覆写）：
  trgc_summary.json          生存分析（两口径两方向存活边数）+ 拓扑（置换检验 / 轴向
                             Spearman / Top-30 边表）+ SNR 对照
  data/edges_table.npz       逐边 3782 条全量表（mean/sd/t/p/q/effect/CI/存活标记等）
  figs/trgc_group_mean.png   TRGC̄ 带符号矩阵（signed_matrix 增量图型）
  figs/trgc_top30_topo.png   Top-30 边头皮 2D 投影箭头（topo_arrows 增量图型）
  figs/trgc_effect_q.png     效应量 vs −log10(q) 火山散点（volcano_edges 增量图型）

统计复用（只读 import、不得改 v1）：`models/gpfm-mamba-v1/gc_psd.py` 的
`t_two_sided_p` / `bh_fdr`（numpy-only 手写正则化不完全 Beta / BH 排序实现；
注：gc_psd 顶部 import 经 gpr_prior 传递性引入 torch——仅模块装载成本，本脚本
不触任何 torch 功能）。

用法：python3 models/bivar-gc-ec-v2/trgc.py [--fwd-run refs-01] [--rev-run rev-01]
                                           [--run-id trgc-01] [--n-boot 2000]
                                           [--n-perm 5000] [--top-k 30]
确定性：无时间戳/计时字段；bootstrap / 置换各自独立 default_rng(2026)；
重跑产物逐位一致（sha256 自查）。
"""

from __future__ import annotations

import os

# ---- BLAS 护栏（environment.md 惯例：必须在 numpy 之前；setdefault 允许外部覆盖）----
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse           # noqa: E402
import json               # noqa: E402
import sys                # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np                          # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402

# ---- sys.path 装配（run.py / compare_rev.py 同式）：v2 目录居首 ----
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import data_io            # noqa: E402（v2 版；连带经唯一名配方装载 bgec2_config）
import bgec2_config as C  # noqa: E402（v2 config，唯一名）
import compare_rev as cr  # noqa: E402（同变体复用 _load_refs 装载断言；只读）

# ---- 统计函数只读复用（gpfm-mamba-v1，路径登记 + 归属断言；不得改 v1）----
_GPFM_DIR = _HERE.parent / "gpfm-mamba-v1"
sys.path.append(str(_GPFM_DIR))
import gc_psd             # noqa: E402（t_two_sided_p / bh_fdr；传递性 import torch 仅装载成本）

_COMMON = _HERE.parent.parent / "common"
if str(_COMMON) not in sys.path:
    sys.path.append(str(_COMMON))
from visualizer import Visualizer  # noqa: E402（正典路由）

assert Path(gc_psd.__file__).resolve().parent == _GPFM_DIR.resolve(), \
    f"gc_psd 归属异常：{gc_psd.__file__}"
_t_two_sided_p = gc_psd.t_two_sided_p
_bh_fdr = gc_psd.bh_fdr

# ---- 受保护 run 目录（既有产物禁止覆写）----
_PROTECTED_RUNS = ("smoke-01", "refs-01", "refs-01-reprod", "rev-01", "sh-01")


def _load_channel_names(manifest: dict) -> list:
    """通道名从数据集 npz channel_names 读（EXPECTED_IDS 首例；62 个唯一）。"""
    p = manifest[C.EXPECTED_IDS[0]]["npz_path"]
    d = np.load(p, allow_pickle=True)
    names = [str(s) for s in d["channel_names"]]
    assert len(names) == C.N_CHANNELS and len(set(names)) == C.N_CHANNELS, \
        f"{p}: channel_names 数量/唯一性异常"
    return names


def _load_channel_std(manifest: dict) -> np.ndarray:
    """逐通道 std（**z-score 前原尺度**，ddof=0）48 例平均——SNR 对照代理。"""
    stds = np.empty((len(C.EXPECTED_IDS), C.N_CHANNELS), dtype=np.float64)
    for k, sid in enumerate(C.EXPECTED_IDS):
        d = np.load(manifest[sid]["npz_path"], allow_pickle=True)
        assert str(d["subject_id"]) == str(sid), f"{sid}: npz subject_id 不符"
        x = np.asarray(d["data"], dtype=np.float64)
        assert x.shape == (C.N_CHANNELS, C.N_RAW), f"{sid}: 形状 {x.shape} 异常"
        stds[k] = x.std(axis=1, ddof=0)
    return stds.mean(axis=0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fwd-run", default="refs-01", help="正向基准 run 目录名")
    ap.add_argument("--rev-run", default="rev-01", help="时间反转 run 目录名")
    ap.add_argument("--run-id", default="trgc-01", help="本分析 run 目录名（产物落此处）")
    ap.add_argument("--n-boot", type=int, default=2000, help="受试者级 bootstrap 次数")
    ap.add_argument("--n-perm", type=int, default=5000, help="符号翻转置换次数")
    ap.add_argument("--top-k", type=int, default=30, help="Top-K |TRGC| 边表/箭头图 K")
    ap.add_argument("--q-thresh", type=float, default=0.05, help="FDR q 阈值")
    args = ap.parse_args()
    assert args.run_id not in _PROTECTED_RUNS, f"run_id={args.run_id} 为既有产物目录，禁止覆写"

    # ---- 装载：refs-01（fwd）/ rev-01（rev，含 reversed=True 标记）----
    fwd_path = C.OUTPUTS_ROOT / args.fwd_run / "refs_all.npz"
    rev_path = C.OUTPUTS_ROOT / args.rev_run / "refs_all.npz"
    assert fwd_path.exists(), f"缺失 {fwd_path}"
    assert rev_path.exists(), f"缺失 {rev_path}"
    F, sids = cr._load_refs(fwd_path, expect_reversed=False)
    R, sids_r = cr._load_refs(rev_path, expect_reversed=True)
    assert sids == sids_r, "fwd/rev subject_ids 顺序不一致"
    n_subj = len(sids)

    # ---- 数据契约侧复核（compare_rev 同式）：manifest ok == EXPECTED_IDS、groups 一致 ----
    manifest = data_io.load_manifest()
    ok_sids = sorted(s for s, e in manifest.items() if e["status"] == "ok")
    assert ok_sids == list(C.EXPECTED_IDS), f"manifest ok 清单与 EXPECTED_IDS 不一致：{ok_sids}"
    groups = {str(s): manifest[s]["group"] for s in C.EXPECTED_IDS}
    for run_dir in (C.OUTPUTS_ROOT / args.fwd_run, C.OUTPUTS_ROOT / args.rev_run):
        sj = json.loads((run_dir / "refs_summary.json").read_text())
        assert sj["groups"] == groups, f"{run_dir}: refs_summary groups 与 manifest 不一致"

    # ---- TRGC（净时序方向读数；f64 计算）----
    TRGC = F - R
    assert np.isfinite(TRGC).all(), "TRGC 非有限值"
    eye = ~np.eye(C.N_CHANNELS, dtype=bool)
    ii, jj = np.where(eye)                       # 行主序 offdiag（行=源/列=靶）
    n_edges = int(eye.sum())
    assert n_edges == 3782
    X = TRGC[:, eye]                             # (48, 3782)

    # ---- 通道名 / 3D 坐标 / 前后轴登记 ----
    names = _load_channel_names(manifest)
    coords_path = C.DATA_ROOT / "chan_coords_62.npy"
    coords = np.asarray(np.load(coords_path), dtype=np.float64)
    assert coords.shape == (C.N_CHANNELS, 3), f"坐标形状 {coords.shape} 异常"
    y_ax = coords[:, 1]                          # 候选前后轴：y（+y = 前）
    iname = {nm: k for k, nm in enumerate(names)}
    # 轴向实证：10-20 标签梯度 Fpz(额极) > Cz(中央) > Oz(枕) 沿 +y
    assert y_ax[iname["Fpz"]] > y_ax[iname["Cz"]] > y_ax[iname["Oz"]], \
        "y 轴前后向与 10-20 标签梯度矛盾（Fpz>Cz>Oz 不成立）——轴登记失效，halt"
    # 主成分对照（登记用）：电极冠为盘状点云，PC1=左右（帽宽>帽长实测）、PC2=前后——
    # y 轴须与面内主成分之一高度一致（|r|≥0.95），登记实际对应哪一主成分
    xy_c = coords - coords.mean(axis=0, keepdims=True)
    _vt = np.linalg.svd(xy_c, full_matrices=False)[2]
    _yc = y_ax - y_ax.mean()
    pc_corr = [float(pearsonr(_yc, xy_c @ _vt[k]).statistic) for k in range(2)]
    assert max(abs(pc_corr[0]), abs(pc_corr[1])) >= 0.95, \
        f"y 轴与面内主成分一致性不足（PC1={pc_corr[0]:+.4f}, PC2={pc_corr[1]:+.4f}）——halt 请示"
    axis_reg = {
        "axis": "y", "orientation": "+y = anterior（10-20 标签梯度 Fpz>Cz>Oz 实证）",
        "pc_align": {"pc1_pearson": pc_corr[0], "pc2_pearson": pc_corr[1],
                     "note": "盘状电极冠点云 PC1≈左右轴（帽宽>帽长）、PC2≈前后轴；"
                             "y 轴与 PC2 高度一致，取 y（比主成分轴可解释）"},
        "registration": "先看坐标结构后定：coords z 轴为竖直（顶点最高）、x 为左右；"
                        "y 轴即前后轴（10-20 标签梯度实证 + 面内主成分对照），取 y",
    }

    # ---- 三等分 ROI（rank 三等分：np.array_split → post 21 / mid 21 / ant 20）----
    order = np.argsort(y_ax, kind="stable")      # 升序：后 → 前
    post_idx, mid_idx, ant_idx = (np.sort(b) for b in np.array_split(order, 3))
    assert (len(post_idx), len(mid_idx), len(ant_idx)) == (21, 21, 20)
    roi_lab = np.empty(C.N_CHANNELS, dtype=np.int8)   # 0=post 1=mid 2=ant
    roi_lab[post_idx], roi_lab[mid_idx], roi_lab[ant_idx] = 0, 1, 2
    ROI_NAMES = ("posterior", "middle", "anterior")

    # ================= ① 生存分析：逐边单样本 t 检验 + BH-FDR + bootstrap =================
    mean_e = X.mean(axis=0)
    sd_e = X.std(axis=0, ddof=1)
    assert (sd_e > 0).all(), "存在零方差边"
    t_e = mean_e / (sd_e / np.sqrt(n_subj))
    p_e = np.asarray([_t_two_sided_p(float(t), n_subj - 1) for t in t_e], dtype=np.float64)
    q_e = _bh_fdr(p_e)
    eff_e = mean_e / sd_e                        # 效应量 dz 型 = mean TRGC / sd（逐边）

    surv_q = q_e < args.q_thresh
    n_q = int(surv_q.sum())
    n_q_fwd = int((surv_q & (mean_e > 0)).sum())
    n_q_rev = int((surv_q & (mean_e < 0)).sum())

    # bootstrap：受试者重采样（percentile CI 95%）；独立 rng(seed=2026)、分块控内存
    rng_boot = np.random.default_rng(C.SEED)
    n_boot = args.n_boot
    boot = np.empty((n_boot, n_edges), dtype=np.float64)
    CH = 100
    for b0 in range(0, n_boot, CH):
        idx = rng_boot.integers(0, n_subj, size=(min(CH, n_boot - b0), n_subj))
        boot[b0:b0 + idx.shape[0]] = X[idx].mean(axis=1)
    ci_lo, ci_hi = (np.percentile(boot, p_, axis=0) for p_ in (2.5, 97.5))
    surv_ci = (ci_lo > 0) | (ci_hi < 0)
    n_ci = int(surv_ci.sum())
    n_ci_fwd = int((surv_ci & (mean_e > 0)).sum())
    n_ci_rev = int((surv_ci & (mean_e < 0)).sum())
    both = surv_q & surv_ci
    n_both = int(both.sum())
    n_both_fwd = int((both & (mean_e > 0)).sum())
    n_both_rev = int((both & (mean_e < 0)).sum())

    # ================= ② 组级拓扑：前后轴方向偏置 + SNR 对照 =================
    G = TRGC.mean(axis=0)                        # 组均值矩阵（带符号，对角=0）
    G_edges = G[eye]
    pa = G[post_idx[:, None], ant_idx[None, :]].mean()   # 后→前 边群均值（21×20=420 边）
    ap_ = G[ant_idx[:, None], post_idx[None, :]].mean()  # 前→后 边群均值（20×21=420 边）
    d_obs = float(pa - ap_)

    # 符号翻转置换（逐例 ±1，双侧 p = (1+#{|D_perm|≥|D_obs|})/(n_perm+1)；独立 rng(seed=2026)）
    rng_perm = np.random.default_rng(C.SEED)
    n_perm = args.n_perm
    signs = rng_perm.integers(0, 2, size=(n_perm, n_subj)).astype(np.float64) * 2.0 - 1.0
    # 置换组均值矩阵：除以 n_subj，与观测侧 G = TRGC.mean(0) 同式（检验统计量两侧须
    # 同一定义；漏除 n 会把置换分布放大 n 倍、|D_perm|≥|D_obs| 判定阈值等效缩小 n 倍，
    # p 系统性推高）。einsum numpy 内循环（确定性，不涉 BLAS）。
    Gp = np.einsum("bs,sec->bec", signs, TRGC, optimize=False) / float(n_subj)
    d_perm = (Gp[:, post_idx[:, None], ant_idx[None, :]].mean(axis=(1, 2))
              - Gp[:, ant_idx[:, None], post_idx[None, :]].mean(axis=(1, 2)))
    perm_p = float((1 + int((np.abs(d_perm) >= abs(d_obs)).sum())) / (n_perm + 1))

    # 边级：Spearman(TRGC̄, 轴向位移)，disp = y_tgt − y_src（+ = 靶更靠前）
    disp = y_ax[jj] - y_ax[ii]
    sp_disp = spearmanr(G_edges, disp)

    # Top-K |TRGC| 边表（确定性：|·| 降序 stable，并列按索引序）
    k = args.top_k
    ktop = np.argsort(-np.abs(G_edges), kind="stable")[:k]
    topk_rows = [{
        "rank": r + 1,
        "src": names[ii[e]], "tgt": names[jj[e]],
        "src_idx": int(ii[e]), "tgt_idx": int(jj[e]),
        "src_xyz": [float(v) for v in coords[ii[e]]],
        "tgt_xyz": [float(v) for v in coords[jj[e]]],
        "trgc": float(G_edges[e]),
        "direction": "net_forward (GC_fwd>GC_rev)" if G_edges[e] > 0 else "net_reversed (GC_rev>GC_fwd)",
        "src_roi": ROI_NAMES[int(roi_lab[ii[e]])], "tgt_roi": ROI_NAMES[int(roi_lab[jj[e]])],
        "axial_disp": float(disp[e]),
    } for r, e in enumerate(ktop.tolist())]

    # SNR 对照：逐通道原尺度 std（48 例平均）vs 组级 TRGC 出强度（行和 offdiag）
    ch_std = _load_channel_std(manifest)
    ch_out = G.sum(axis=1)                       # 对角=0，行和即出强度（带符号）
    sp_snr = spearmanr(ch_std, ch_out)
    sp_snr_ap = spearmanr(ch_std, y_ax)          # 辅助：SNR 自身的前后梯度（判读用）

    # ---- 产物：edges_table.npz（np.savez 定长时间戳，字节确定）----
    out_dir = C.OUTPUTS_ROOT / args.run_id
    (out_dir / "data").mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "data" / "edges_table.npz"
    np.savez(npz_path,
             src_idx=ii.astype(np.int64), tgt_idx=jj.astype(np.int64),
             trgc_mean=mean_e, trgc_sd=sd_e, t_stat=t_e, p_raw=p_e, q=q_e,
             effect=eff_e, ci_lo=ci_lo, ci_hi=ci_hi,
             surv_q05=surv_q.astype(np.int8), surv_ci=surv_ci.astype(np.int8),
             axial_disp=disp, roi_label=roi_lab,
             chan_axis_y=y_ax, chan_std=ch_std, chan_out_trgc=ch_out)
    (out_dir / "figs").mkdir(parents=True, exist_ok=True)

    # ---- trgc_summary.json（无时间戳/计时字段——确定性契约）----
    summary = {
        "deliverable": "TRGC 净时序方向分析（用户 2026-10-18 追加）：TRGC_s = GC_fwd_s − "
                       "GC_rev_s（逐例逐边，读已落盘 refs-01/rev-01 refs_all.npz 纯计算）",
        "n_subjects": n_subj, "subject_ids": sids, "groups": groups,
        "inputs": {"fwd": {"run_id": args.fwd_run, "path": str(fwd_path)},
                   "rev": {"run_id": args.rev_run, "path": str(rev_path),
                           "reversed_flag": True},
                   "chan_coords": {"path": str(coords_path), "shape": [62, 3]},
                   "channel_names_source": manifest[C.EXPECTED_IDS[0]]["npz_path"]},
        "caliber": {
            "trgc_definition": "TRGC = GC_fwd − GC_rev（逐例逐有向边；行=源/列=靶；"
                               "TRGC>0 = 净前向 fwd 占优，TRGC<0 = 净反向 rev 占优）",
            "edge_flattening": "offdiag 行主序展平（62×62 对角剔除；3782 有向边）",
            "interpretation_frame": "TRGC = 净时序方向读数（fwd−rev 差分）；显著边 ≠ 生理"
                                    "因果确证——双变量 GC 口径 + 头皮体积传导局限如实注明"
                                    "（判读框架沿 rev-01 / compare_rev 预登记）",
            "axis_registration": axis_reg,
            "roi_rule": "rank 三等分（np.array_split(y 升序)）：post 21 / mid 21 / ant 20",
            "seeds": {"bootstrap": C.SEED, "permutation": C.SEED},
            "n_bootstrap": n_boot, "n_perm": n_perm,
            "bootstrap_ci": "受试者有放回重采样 percentile CI 95%（2.5/97.5）",
            "perm_p_rule": "符号翻转（逐例 ±1）双侧 p = (1+#{|D_perm|≥|D_obs|})/(n_perm+1)",
            "topk_rule": "np.argsort(-|TRGC̄|, kind='stable') 前 K（确定性；并列按索引序）",
        },
        "survival": {
            "ttest": {"df": n_subj - 1, "n_edges": n_edges, "q_threshold": args.q_thresh,
                      "n_surviving": n_q,
                      "n_net_forward": n_q_fwd, "n_net_reversed": n_q_rev},
            "bootstrap_ci": {"n_excl_zero": n_ci, "n_net_forward": n_ci_fwd,
                             "n_net_reversed": n_ci_rev},
            "intersection": {"n_both": n_both, "n_net_forward": n_both_fwd,
                             "n_net_reversed": n_both_rev,
                             "definition": "q<阈值 且 bootstrap CI 不含 0"},
            "effect_size": {"definition": "mean TRGC / sd（逐边，单样本 dz 型）",
                            "median": float(np.median(eff_e)),
                            "min": float(eff_e.min()), "max": float(eff_e.max()),
                            "median_surviving_q": float(np.median(eff_e[surv_q])) if n_q else None},
            "trgc_mean_offdiag": {"min": float(mean_e.min()), "median": float(np.median(mean_e)),
                                  "max": float(mean_e.max())},
        },
        "topology": {
            "posterior_to_anterior": {"n_edges": int(len(post_idx) * len(ant_idx)),
                                      "mean_trgc": float(pa)},
            "anterior_to_posterior": {"n_edges": int(len(ant_idx) * len(post_idx)),
                                      "mean_trgc": float(ap_)},
            "diff_pa_minus_ap": d_obs,
            "perm_p_two_sided": perm_p,
            "direction_reading": ("posterior→anterior 净偏置" if d_obs > 0
                                  else "anterior→posterior 净偏置"),
            "edge_disp_spearman": {"rho": float(sp_disp.statistic),
                                   "p": float(sp_disp.pvalue),
                                   "definition": "disp = y_tgt − y_src（+ = 靶更靠前）；"
                                                 "边级 Spearman(组均值 TRGC, disp)"},
            "top_edges": {"k": k, "rows": topk_rows},
        },
        "snr_control": {
            "definition": "逐通道 std（原尺度、z-score 前、ddof=0，48 例平均）vs 组级 TRGC "
                          "出强度（组均值 offdiag 行和，带符号）Spearman——高幅通道是否"
                          "系统性地当源（SNR 假象控制）",
            "spearman_out": {"rho": float(sp_snr.statistic), "p": float(sp_snr.pvalue)},
            "aux_std_vs_axis": {"rho": float(sp_snr_ap.statistic),
                                "p": float(sp_snr_ap.pvalue),
                                "definition": "辅助：通道 std 自身沿前后轴的梯度（判读用）"},
            "channel_std": {"min": float(ch_std.min()), "median": float(np.median(ch_std)),
                            "max": float(ch_std.max()), "per_channel": [float(v) for v in ch_std]},
        },
        "outputs": {"summary": "trgc_summary.json",
                    "data": "data/edges_table.npz",
                    "figs": ["figs/trgc_group_mean.png", "figs/trgc_top30_topo.png",
                             "figs/trgc_effect_q.png"]},
        "command": (f"python3 models/bivar-gc-ec-v2/trgc.py --fwd-run {args.fwd_run} "
                    f"--rev-run {args.rev_run} --run-id {args.run_id} --n-boot {n_boot} "
                    f"--n-perm {n_perm} --top-k {k} --q-thresh {args.q_thresh}"),
        "entry": "models/bivar-gc-ec-v2/trgc.py",
        "determinism": "纯计算；bootstrap/置换各自独立 default_rng(2026)；无时间戳/计时"
                       "字段；np.savez 定长 zip 时间戳——重跑产物逐位一致（sha256 自查）",
    }
    out_path = out_dir / "trgc_summary.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=1))

    # ---- 落图（common/visualizer 正典路由；三个增量图型 2026-10-18，白名单见 experiment.md）----
    viz = Visualizer(str(out_dir / "figs"))
    p1 = viz.render("signed_matrix", {
        "mats": [{"title": f"TRGC mean (n={n_subj})", "m": G}],
        "note": "TRGC = GC$_{fwd}$ − GC$_{rev}$; red = net forward, blue = net reversed;"
                " diag masked",
    }, "trgc_group_mean.png")
    p2 = viz.render("topo_arrows", {
        "coords_xy": coords[:, :2], "names": names,
        "edges": {"src": ii[ktop], "tgt": jj[ktop], "val": G_edges[ktop]},
        "node_w": np.abs(ch_out), "label_edges_n": 15,
        "title": f"Top-{k} |TRGC| edges (head top view, +y = anterior)",
        "note": "node size ∝ |out-strength|; arrows red = net forward, blue = net reversed",
    }, "trgc_top30_topo.png")
    p3 = viz.render("volcano_edges", {
        "x": eff_e, "y": -np.log10(q_e), "y_thresh": -np.log10(args.q_thresh),
        "x_label": "effect size (mean TRGC / sd)",
        "stat_lines": [f"q<{args.q_thresh:g}: {n_q} edges (fwd {n_q_fwd} / rev {n_q_rev})",
                       f"CI excl. 0: {n_ci} (fwd {n_ci_fwd} / rev {n_ci_rev})",
                       f"both: {n_both} (fwd {n_both_fwd} / rev {n_both_rev})"],
    }, "trgc_effect_q.png")

    print(f"[trgc] summary -> {out_path}")
    print(f"[trgc] figs    -> {p1} | {p2} | {p3}")
    print(f"[trgc] 生存分析: q<{args.q_thresh:g} 存活 {n_q}（净前向 {n_q_fwd} / 净反向 {n_q_rev}）| "
          f"CI 不含 0 {n_ci}（fwd {n_ci_fwd} / rev {n_ci_rev}）| 交集 {n_both}")
    print(f"[trgc] 拓扑: 后→前 {pa:.5f} vs 前→后 {ap_:.5f} 差 {d_obs:+.5f}, "
          f"置换 p={perm_p:.4f} ({n_perm} 次) | 轴向 Spearman ρ={sp_disp.statistic:.4f} "
          f"(p={sp_disp.pvalue:.2e})")
    print(f"[trgc] SNR 对照: ρ(ch_std, TRGC 出强度)={sp_snr.statistic:.4f} "
          f"(p={sp_snr.pvalue:.2e}) | 辅助 ρ(ch_std, y)={sp_snr_ap.statistic:.4f}")
    print(f"[trgc] Top-5: " + "; ".join(
        f"{r['src']}→{r['tgt']}={r['trgc']:+.4f}" for r in topk_rows[:5]))


if __name__ == "__main__":
    main()
