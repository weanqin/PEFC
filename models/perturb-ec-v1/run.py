#!/usr/bin/env python3
"""run.py — perturb-ec-v1 CLI 入口（design §1；B0 脚手架 + B1 smoke + B2 bench）。

用法（implement.md 验证命令速查）：
  python3 models/perturb-ec-v1/run.py --stage smoke --run-id smoke-01
  python3 models/perturb-ec-v1/run.py --stage bench --subjects 1 --run-id bench-01
  python3 models/perturb-ec-v1/run.py --stage readout / refs / split-half / compare / figs   # B4/B5 未实现
  mkdir -p outputs/perturb-ec-f0/main-01/logs && nohup python3 run.py --stage main ...       # B3 未实现

smoke（B1，experiment.md Run 001）：test 第一例（subject 10）子采样前 2 窗 × 497 并批
CRN 采样 → 窗级 D 曲线 + null 池 → τ 三分位 → readout 骨架 → 剂量-响应曲线图
（common/visualizer.py 纯增量扩展 dose_response 类型）。

bench（B2，experiment.md Run 002）：subject 10 全 32 窗扰动链逐窗计时（每窗一次
sample()、497 并批）→ 10 例（320 窗）外推 + GPU 显存峰值；930 窗全集 baseline 侧
代表性抽查（QQ/分位数表/KS）；τ 决策数据预览（预注册 τ 删失率 / 相对自响应口径
D̃(r) / slope 分布——**预览、非 readout 承诺、τ 口径待用户拍板**）。

main（B3，experiment.md Run 003）：fold0 test 10 例全量 × N_w=32 扰动批采样
（例序 EXPECTED_TEST_IDS 升序、win_idx 跨例连续 0..319 → subject 10 ≡ bench-01
逐位断言）；逐例落盘 + 断点续跑（done 标记跳过已完成例、例内 partial 逐窗续）。
只落 D 曲线与 null 池（τ 无关数据）；readout/比较是 B4/B5，本 stage 不做。
"""

from __future__ import annotations

import os

# ---- BLAS 护栏（environment.md / _blas_guard 惯例：必须在 numpy/torch 导入之前；----
# ---- 有效 16 核 cgroup 限 8 线程，setdefault 允许外部环境变量覆盖）            ----
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse  # noqa: E402
import json      # noqa: E402
import subprocess  # noqa: E402
import sys       # noqa: E402
import time      # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch        # noqa: E402

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import loader as LD    # noqa: E402  触发 v1 sample/data/model/gpr_prior 只读装配
import perturb as PB   # noqa: E402
import readout as RO   # noqa: E402
import response as RS  # noqa: E402

_COMMON = _REPO / "common"                       # 统一 visualizer（visualization.md 正典路由）
if str(_COMMON) not in sys.path:
    sys.path.append(str(_COMMON))

STAGES = ("smoke", "bench", "main", "readout", "refs", "split-half", "compare", "figs")


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(_REPO),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _logger(log_path: Path):
    """双写 logger：stdout + logs/run.log（逐行 append）。"""
    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    return log


def stage_smoke(run_id: str = "smoke-01", n_windows: int = 2) -> None:
    """B1 smoke：1 例（test 第一例 subject 10）× 子采样前 2 窗全链（implement.md B1）。

    链路：冻结装载（含 gen_lowpass_hz=None 断言）→ 窗子采样 → 逐窗 497 并批 CRN 采样
    → D 曲线 (62,8,62) + null 池 (28,62) → τ 三分位（逐被试分池 = 全窗全通道全对展平）
    → readout 骨架（slope/AUC/r* + 删失率）→ 剂量-响应曲线图。
    迭代顺序（experiment.md Run 001 登记）：test_ds 自然例序 + 例内子采样索引升序；
    win_idx 从 0 连续（与主 Run 预定迭代顺序一致 → smoke eps/eta = 主 Run 前缀）。
    """
    PC = PB.load_config()
    out = Path(PC.OUTPUTS_ROOT) / run_id
    (out / "logs").mkdir(parents=True, exist_ok=True)          # 跑前 mkdir（硬约束）
    (out / "figs").mkdir(parents=True, exist_ok=True)
    log = _logger(out / "logs" / "run.log")
    t0 = time.time()
    git_head = _git_head()
    log(f"stage=smoke run_id={run_id} outputs={out} git={git_head} "
        f"torch={torch.__version__} gpu={torch.cuda.get_device_name(0)}")

    # ---- 冻结装载 + 口径断言 ----
    cfg, model, ema, prior, sampler, ck = LD.load_frozen_model("cuda")
    torch.backends.cudnn.deterministic = cfg.cudnn_deterministic
    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    assert sampler.gen_lowpass_hz is None, "gen_lowpass_hz 须 None（fmt 口径，B-Major1 陷阱）"
    log(f"θ* 装载：epoch={ck['epoch']} hash={ck['config_hash']}；euler_steps={cfg.euler_steps} "
        f"sigma_min={cfg.sigma_min} gen_lowpass_hz=None（断言过）chunk_bk={sampler.chunk_bk}")

    # ---- test 装载 + 逐被试窗子采样（例序 = EXPECTED_TEST_IDS 升序，与 Run 001 登记一致；
    #      test_ds 自然序首例为 44——非 ID 升序，故显式取 EXPECTED_TEST_IDS[0]=10）----
    _loader_obj, test_ds = LD.load_test_loader(batch_size=4, num_workers=2)
    subs = LD.subsample_windows_per_subject(test_ds)
    sid = int(PC.EXPECTED_TEST_IDS[0])
    assert sid in subs, f"登记首例 subject {sid} 不在 test 折（ID 漂移）"
    win_global = [int(i) for i in subs[sid][:n_windows]]
    log(f"test 例（ID 升序）首例 subject={sid}（子采样 N_w={PC.N_W}/例，smoke 取前 {n_windows} 窗）；"
        f"窗全局索引（升序）={win_global}；eps/eta seed={PC.SEED}+窗序（0..{n_windows - 1}）")

    # ---- 逐窗 497 并批 CRN 采样 → D 曲线 + null 距离 ----
    d_all, null_all = [], []
    for wseq, gidx in enumerate(win_global):
        xp, _xf = test_ds[gidx]
        xp = xp.to("cuda")
        with ema.swap(model):
            sv = RS.sample_window_variants(sampler, xp, win_idx=wseq)
        assert sv.shape == (PC.N_VARIANTS, PC.K, cfg.n_channels, cfg.future_len)
        assert torch.isfinite(sv).all(), f"窗 {wseq}（全局 {gidx}）采样含非有限值"
        D = RS.d_curves(sv)                    # (62, 8, 62)：[源A, r档j, 靶B]
        nd = RS.null_distances(sv[0])          # (28, 62)：K=8 两两对 × 通道
        assert torch.isfinite(D).all() and (D >= 0).all(), f"窗 {wseq} D 曲线含 NaN/负值"
        assert torch.isfinite(nd).all() and (nd >= 0).all(), f"窗 {wseq} null 含 NaN/负值"
        d_all.append(D.float().cpu().numpy())
        null_all.append(nd.float().cpu().numpy())
        log(f"窗 {wseq + 1}/{n_windows}（全局 {gidx}）：sample {tuple(sv.shape)} 有限 → "
            f"D {tuple(D.shape)}、null {tuple(nd.shape)}（累计 {time.time() - t0:.1f}s）")

    d_curves = np.stack(d_all)                 # (W, 62, 8, 62)
    null_pool = np.stack(null_all)             # (W, 28, 62)
    assert np.isfinite(d_curves).all() and np.isfinite(null_pool).all(), \
        "B1 检查点① 失败：D/null 含 NaN"
    log(f"检查点① 通过：D {d_curves.shape} / null {null_pool.shape} 全有限（无 NaN）")

    # ---- 检查点③：CRN 恒等运行时复验（真实窗 0 数据路径，同批 B=2）----
    xp0, _ = test_ds[win_global[0]]
    xp0 = xp0.to("cuda")
    a_chk = 0                                  # r=0 为 ×1.0 位保持，通道选取不影响结论
    eps_chk, eta_chk = RS.crn_noise(0, PC.K, cfg.n_channels, cfg.future_len, "cuda")
    with ema.swap(model):
        x2 = torch.cat([xp0.unsqueeze(0), PB.apply_gain(xp0, a_chk, 0.0).unsqueeze(0)], dim=0)
        out2 = sampler.sample(x2, PC.K, eps=eps_chk, eta=eta_chk)
    assert out2.shape == (2, PC.K, cfg.n_channels, cfg.future_len)
    assert torch.equal(out2[0], out2[1]), \
        "B1 检查点③ 失败：r=0 变体 vs baseline 同批非逐位相等（CRN 契约破坏）"
    log(f"检查点③ 通过：CRN 恒等复验（窗 0 真实数据、a={a_chk}、r=0、同批 B=2 逐位相等）")

    # ---- τ 三分位（逐被试分池：全窗 × 全通道 × 全对展平；smoke 级小样本）----
    pooled = null_pool.reshape(-1).astype(np.float64)
    taus = {f"q{int(q * 100)}": float(np.quantile(pooled, q)) for q in PC.TAU_QUANTILES}
    tau_main = taus[f"q{int(PC.TAU_MAIN * 100)}"]
    log(f"null 池 {pooled.size} 样本（{null_pool.shape} 展平）τ 分位："
        + "、".join(f"{k}={v:.4f}" for k, v in taus.items())
        + "（smoke 级小样本 τ 仅形态参考）")

    # ---- smoke 聚合（2 窗均值，主 Run 才做逐被试 median）+ readout 骨架 ----
    dbar = d_curves.mean(axis=0).astype(np.float64)             # (62, 8, 62) [A, r档, B]
    dbar9 = np.concatenate(
        [np.zeros((PC.N_SRC, 1, PC.N_SRC)), dbar], axis=1)      # (62, 9, 62)，索引0 ≡ r=0 ≡ 0
    dbar9_rb = np.transpose(dbar9, (0, 2, 1))                    # (A, B, r)：readout 契约 r 末维
    slope = RO.slope_readout(dbar9_rb)                           # (62, 62) 主 readout
    auc = RO.auc_readout(dbar9_rb)
    rstar = {k: RO.rstar_readout(dbar9_rb, v) for k, v in taus.items()}
    offdiag = ~np.eye(PC.N_SRC, dtype=bool)
    log(f"readout 骨架：slope median(跨通道)={np.median(slope[offdiag]):.4f} "
        f"max={slope[offdiag].max():.4f}；auc median={np.median(auc[offdiag]):.4f}；"
        f"rstar（τ 主口径 q95）median={np.median(rstar['q95'][offdiag]):.3f}")

    # ---- 检查点④：跨通道边删失率粗估（r* 未达网格 → R_NEXT）----
    censoring = {}
    for k, rs in rstar.items():
        reach = float((rs[offdiag] <= PC.R_MAX).mean())         # r_max（含）前达 τ 比例
        censoring[k] = 1.0 - reach
        log(f"删失率[{k}]（跨通道 {int(offdiag.sum())} 边，r* = R_NEXT={PC.R_NEXT}）："
            f"{censoring[k] * 100:.1f}%（r_max 达 τ 比例 {reach * 100:.1f}%）——smoke 级粗估")

    # ---- 检查点②：A=B 自响应 vs A≠B 跨通道（r_max 档，定性）----
    j_top = PC.N_R_PERT - 1
    M = dbar[:, j_top, :]                                       # (62,62) D̄(r_max)
    self_r = np.diag(M)
    cross_r = M[offdiag]
    ratio_med = float(np.median(self_r) / np.median(cross_r))
    overlap = float((cross_r >= np.median(self_r)).mean())      # 跨通道超过自响应中位的比例
    log(f"检查点②（r={PC.PERT_R_GRID[-1]}，2 窗均值 D̄）：自响应 median={np.median(self_r):.4f}"
        f"（Q1={np.quantile(self_r, 0.25):.4f}，Q3={np.quantile(self_r, 0.75):.4f}，"
        f"min={self_r.min():.4f}）；跨通道 median={np.median(cross_r):.4f} "
        f"max={cross_r.max():.4f} Q95={np.quantile(cross_r, 0.95):.4f}；"
        f"median 比 {ratio_med:.2f}×；跨通道 ≥ 自响应中位比例 {overlap * 100:.1f}%"
        f"——定性判定回填 experiment.md（design §7 Risk#1 语义）")
    log(f"低剂量段补充（slope 主段 {PC.SLOPE_SEG}）：自响应 slope median={np.median(slope):.4f} "
        f"vs 跨通道 slope median={np.median(slope[offdiag]):.4f}")

    # ---- 曲线形态定性（Spearman(D̄, r) 分布 + 完全单调比例；scipy 局部导入仅诊断用）----
    from scipy.stats import rankdata
    ranks = rankdata(dbar9, axis=1)                             # 沿 r 档对每 (A,B) 取秩
    rr = np.arange(1.0, dbar9.shape[1] + 1)
    rr_c = (rr - rr.mean()).reshape(1, -1, 1)                   # 广播于 r 轴（axis=1）
    rc = ranks - ranks.mean(axis=1, keepdims=True)
    rho = (rc * rr_c).sum(axis=1) / np.sqrt((rc ** 2).sum(axis=1) * (rr_c ** 2).sum())
    rho_off = rho[offdiag]
    mono_frac = float((np.diff(dbar, axis=1) >= 0).all(axis=1)[offdiag].mean())
    log(f"形态定性：跨通道 Spearman(D̄, r) median={np.median(rho_off):.3f} "
        f"（Q1={np.quantile(rho_off, 0.25):.3f}，Q3={np.quantile(rho_off, 0.75):.3f}）；"
        f"完全单调（8 档非降）比例 {mono_frac * 100:.1f}%；"
        f"自响应 Spearman median={np.median(rho):.3f}")

    # ---- 落盘：D 曲线 / null 池 / config 快照 / readout 骨架 ----
    np.save(out / "d_curves.npy", d_curves)
    np.save(out / "null_pool.npy", null_pool)
    np.savez(out / "readout_smoke.npz", slope=slope, auc=auc,
             **{f"rstar_{k}": v for k, v in rstar.items()},
             taus=np.array([taus[k] for k in sorted(taus)]),
             tau_keys=np.array(sorted(taus)))
    consts = {k: (list(v) if isinstance(v, tuple) else v)
              for k, v in vars(PC).items()
              if k.isupper() and isinstance(v, (int, float, str, bool, tuple))}
    snap = {
        "task": "10-17-perturb-ec-f0", "stage": "smoke", "run_id": run_id,
        "git_head": git_head,
        "constants": consts,
        "model": {"ckpt": PC.CKPT_PATH, "config_hash": ck["config_hash"],
                  "epoch": ck["epoch"], "weights": "EMA",
                  "euler_steps": cfg.euler_steps, "sigma_min": cfg.sigma_min,
                  "gen_lowpass_hz": sampler.gen_lowpass_hz},
        "sampler": {"K": PC.K, "chunk_bk": sampler.chunk_bk,
                    "n_variants_per_window": PC.N_VARIANTS},
        "data": {"subject_id": sid, "n_windows": n_windows,
                 "window_global_indices": win_global,
                 "window_subsample": "default_rng(2026*1000003+sid) 无放回升序，取前 2 窗",
                 "eps_seed_formula": f"{PC.SEED} + win_idx（win_idx=0..{n_windows - 1}，主 Run 前缀）"},
        "env": {"python": sys.version.split()[0], "torch": torch.__version__,
                "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                "blas_threads": os.environ.get("OMP_NUM_THREADS", "?")},
    }
    (out / "config_snapshot.json").write_text(
        json.dumps(snap, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"落盘：d_curves.npy {d_curves.shape}、null_pool.npy {null_pool.shape}、"
        f"readout_smoke.npz、config_snapshot.json")

    # ---- 剂量-响应曲线图（4 条代表性 (A→B)：1 自响应 + 强/中/弱跨通道）----
    a_self = int(np.argmax(self_r))                              # 最强自响应源
    pairs = np.argwhere(offdiag)
    order = np.argsort(M[offdiag])[::-1]                         # r_max 处降序
    (a_s, b_s), (a_m, b_m), (a_w, b_w) = (tuple(map(int, pairs[order[i]]))
                                          for i in (0, len(order) // 2, -1))
    picks = {"self": (a_self, a_self), "cross_strong": (a_s, b_s),
             "cross_median": (a_m, b_m), "cross_weak": (a_w, b_w)}
    curves, labels, kinds = [], [], []
    for kind, (a, b) in picks.items():
        curves.append(dbar9[a, :, b])
        labels.append((f"A=B={a} (self)" if kind == "self" else f"A={a}->B={b} ({kind.split('_')[1]})"))
        kinds.append(kind)
    import visualizer as VZ
    viz = VZ.Visualizer(str(out / "figs"))
    fig_path = viz.render("dose_response", {
        "r_grid": np.asarray(PC.R_GRID, dtype=np.float64),
        "curves": np.stack(curves), "labels": labels, "kinds": kinds,
        "tau": tau_main, "tau_label": f"tau = null q{int(PC.TAU_MAIN * 100)}",
        "tau_sens": {k: v for k, v in taus.items() if v != tau_main},
        "note": (f"subject {sid}, mean of 2 windows (smoke-scale;\n"
                 f"tau from 2-window null pool, shape reference only)"),
    }, "dose_response_smoke.png")
    log(f"图：{fig_path}（4 条代表曲线：self A={a_self}；cross 强 {a_s}→{b_s} / "
        f"中 {a_m}→{b_m} / 弱 {a_w}→{b_w}）")

    # ---- metrics 汇总（机器可读，check 阶段与回填用）----
    metrics = {
        "git_head": git_head, "subject_id": sid, "n_windows": n_windows,
        "window_global_indices": win_global,
        "taus": taus, "null_pool_size": int(pooled.size),
        "self_vs_cross_r_max": {
            "self_median": float(np.median(self_r)), "self_q1": float(np.quantile(self_r, 0.25)),
            "self_q3": float(np.quantile(self_r, 0.75)), "self_min": float(self_r.min()),
            "cross_median": float(np.median(cross_r)), "cross_max": float(cross_r.max()),
            "cross_q95": float(np.quantile(cross_r, 0.95)),
            "median_ratio": ratio_med, "cross_ge_self_median_frac": overlap},
        "slope": {"self_median": float(np.median(slope)),
                  "cross_median": float(np.median(slope[offdiag])),
                  "cross_max": float(slope[offdiag].max())},
        "censoring": censoring,
        "shape": {"spearman_cross_median": float(np.median(rho_off)),
                  "spearman_cross_q1": float(np.quantile(rho_off, 0.25)),
                  "spearman_cross_q3": float(np.quantile(rho_off, 0.75)),
                  "spearman_self_median": float(np.median(rho)),
                  "monotone_frac_cross": mono_frac},
        "fig_picks": {k: list(v) for k, v in picks.items()},
        "fig_path": str(fig_path), "elapsed_s": round(time.time() - t0, 1),
    }
    (out / "smoke_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"smoke 完成：总耗时 {time.time() - t0:.1f}s；四检查点判定回填 experiment.md Run 001")


def stage_bench(run_id: str = "bench-01", n_subjects: int = 1) -> None:
    """B2 bench（implement.md B2 / experiment.md Run 002）：1 例 N_w=32 计时定档 +
    子采样代表性抽查 + τ 决策数据预览。

    - 主链口径 = Run 001 四要素**全部不变**（每窗一次 sample()、497 并批、eps/eta
      seed=2026+窗序 0..31）→ 逐窗计时 + GPU 峰值 → 10 例（320 窗）外推。
    - 代表性扫描（bench 附加，不进主 Run 成本）：subject 10 全 930 窗 baseline-only
      批采样，逐窗独立 seed=2026+1_000_000+窗内偏移；**子样本 vs 全集在同一扫描内比较**
      （同噪声 → 隔离窗选择效应，design §7-7）。
    - τ 预览（a 预注册删失率 / b 相对自响应口径 D̃ / c slope 分布）**只算不判**：
      预览、非 readout 承诺、τ 口径待用户拍板。
    """
    PC = PB.load_config()
    assert n_subjects == 1, "B2 bench 登记为 1 例（--subjects 1）；10 例主链是 B3 main"
    out = Path(PC.OUTPUTS_ROOT) / run_id
    (out / "logs").mkdir(parents=True, exist_ok=True)
    (out / "figs").mkdir(parents=True, exist_ok=True)
    log = _logger(out / "logs" / "run.log")
    t0 = time.time()
    git_head = _git_head()
    log(f"stage=bench run_id={run_id} outputs={out} git={git_head} "
        f"torch={torch.__version__} gpu={torch.cuda.get_device_name(0)}")

    # ---- 冻结装载 + 口径断言（同 Run 001）----
    t_load0 = time.time()
    cfg, model, ema, prior, sampler, ck = LD.load_frozen_model("cuda")
    torch.backends.cudnn.deterministic = cfg.cudnn_deterministic
    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    assert sampler.gen_lowpass_hz is None, "gen_lowpass_hz 须 None（fmt 口径，B-Major1 陷阱）"
    _loader_obj, test_ds = LD.load_test_loader(batch_size=4, num_workers=2)
    subs = LD.subsample_windows_per_subject(test_ds)
    sid = int(PC.EXPECTED_TEST_IDS[0])
    assert sid in subs and len(subs[sid]) >= PC.N_W, \
        f"subject {sid} 子采样窗数 {len(subs.get(sid, []))} 不足 N_W={PC.N_W}"
    win_global = [int(i) for i in subs[sid][:PC.N_W]]
    t_setup = time.time() - t_load0
    log(f"θ* 装载：epoch={ck['epoch']} hash={ck['config_hash']} gen_lowpass_hz=None（断言过）；"
        f"subject={sid}，N_W={PC.N_W} 全子采样窗（升序，窗 0-1 应= smoke-01 两窗）；"
        f"setup（装载+数据）{t_setup:.1f}s")

    # ---- 主链：32 窗逐窗计时（每窗一次 sample()、497 并批；CUDA synchronize 包夹）----
    d_all, null_all, win_times = [], [], []
    torch.cuda.reset_peak_memory_stats()
    mem_base = torch.cuda.memory_allocated()
    t_chain0 = time.time()
    for wseq, gidx in enumerate(win_global):
        xp, _ = test_ds[gidx]
        xp = xp.to("cuda")
        torch.cuda.synchronize()
        ta = time.time()
        with ema.swap(model):
            sv = RS.sample_window_variants(sampler, xp, win_idx=wseq)
        torch.cuda.synchronize()
        tb = time.time()
        assert sv.shape == (PC.N_VARIANTS, PC.K, cfg.n_channels, cfg.future_len)
        assert torch.isfinite(sv).all(), f"窗 {wseq}（全局 {gidx}）采样含非有限值"
        D = RS.d_curves(sv)
        nd = RS.null_distances(sv[0])
        assert torch.isfinite(D).all() and (D >= 0).all() \
            and torch.isfinite(nd).all() and (nd >= 0).all(), f"窗 {wseq} D/null 含 NaN/负值"
        d_all.append(D.float().cpu().numpy())
        null_all.append(nd.float().cpu().numpy())
        tc = time.time()
        win_times.append({"wseq": wseq, "gidx": gidx,
                          "t_sample_s": round(tb - ta, 3), "t_post_s": round(tc - tb, 3)})
        log(f"窗 {wseq + 1}/{PC.N_W}（全局 {gidx}）：sample {tb - ta:.2f}s + D/null 后处理 "
            f"{tc - tb:.2f}s（累计 {time.time() - t0:.1f}s）")
    t_chain = time.time() - t_chain0
    peak_alloc = torch.cuda.max_memory_allocated()
    peak_rsvd = torch.cuda.max_memory_reserved()
    gpu_total = torch.cuda.get_device_properties(0).total_memory
    log(f"主链 32 窗总耗时 {t_chain:.1f}s；GPU 峰值：allocated {peak_alloc / 2**30:.2f} GiB / "
        f"reserved {peak_rsvd / 2**30:.2f} GiB / 卡总 {gpu_total / 2**30:.1f} GiB"
        f"（循环前基线 {mem_base / 2**30:.2f} GiB）")

    d_curves = np.stack(d_all)                    # (32, 62, 8, 62) [w, A, r档, B]
    null_pool = np.stack(null_all)                # (32, 28, 62)
    assert np.isfinite(d_curves).all() and np.isfinite(null_pool).all(), "主链 D/null 含 NaN"
    np.save(out / "d_curves.npy", d_curves)
    np.save(out / "null_pool.npy", null_pool)

    # ---- 确定性复现断言：bench 窗 0-1 ≡ smoke-01（同 seed 同批形，应逐位一致）----
    smoke_dir = Path(PC.OUTPUTS_ROOT) / "smoke-01"
    if (smoke_dir / "d_curves.npy").exists():
        assert np.allclose(d_curves[:2], np.load(smoke_dir / "d_curves.npy"), atol=1e-6) \
            and np.allclose(null_pool[:2], np.load(smoke_dir / "null_pool.npy"), atol=1e-6), \
            "bench 窗 0-1 与 smoke-01 不复现（seed/批形/代码漂移——排查后再进 B3）"
        log("确定性复现断言通过：bench 窗 0-1 ≡ smoke-01（同 seed 同批形逐位一致）")

    # ---- 受试者级聚合（design §2.2 主口径：逐被试 median_w）+ τ（主链 null 池）----
    dbar = np.median(d_curves, axis=0).astype(np.float64)          # (62, 8, 62) [A, r档, B]
    dbar9 = np.concatenate(
        [np.zeros((PC.N_SRC, 1, PC.N_SRC)), dbar], axis=1)         # (62, 9, 62)，索引0 ≡ r=0 ≡ 0
    np.save(out / "dbar_median.npy", dbar9)
    pooled = null_pool.reshape(-1).astype(np.float64)
    taus = {f"q{int(q * 100)}": float(np.quantile(pooled, q)) for q in PC.TAU_QUANTILES}
    tau_main = taus[f"q{int(PC.TAU_MAIN * 100)}"]
    log(f"主链 null 池 {pooled.size} 样本（32×28×62）τ 分位："
        + "、".join(f"{k}={v:.4f}" for k, v in taus.items())
        + "；per-edge 32 窗 median D̄ 落盘 dbar_median.npy (62,9,62)")

    # ---- 代表性扫描：subject 10 全 930 窗 baseline-only（bench 附加，不进主 Run 外推）----
    seg = LD.subject_window_slice(test_ds, sid)
    n_full = seg.stop - seg.start
    scan_batch = 32
    n_pairs = PC.K * (PC.K - 1) // 2
    s_scan = np.empty((n_full, cfg.n_channels), dtype=np.float64)
    nd_scan = np.empty((n_full, n_pairs, cfg.n_channels), dtype=np.float64)
    n_batches = (n_full + scan_batch - 1) // scan_batch
    t_scan0 = time.time()
    equiv_checked = False
    for bi in range(n_batches):
        g0 = seg.start + bi * scan_batch
        idx = np.arange(g0, min(g0 + scan_batch, seg.stop))
        koff = idx - seg.start                       # 窗内偏移 0..929（seed 锚，与批划分无关）
        xb = torch.stack([test_ds[int(g)][0] for g in idx], dim=0).to("cuda")
        with ema.swap(model):
            base_b = RS.sample_baseline_batch(
                sampler, xb, [PC.SEED + 1_000_000 + int(k) for k in koff])
        ndb = RS.null_distances_batch(base_b)
        if not equiv_checked:                        # 批量版与逐窗版首窗逐元素同构断言
            assert torch.allclose(ndb[0], RS.null_distances(base_b[0]), atol=1e-6), \
                "null_distances_batch 与 null_distances 非同构（批量实现破坏口径）"
            equiv_checked = True
        s_scan[koff] = RS.baseline_scale_batch(base_b).double().cpu().numpy()
        nd_scan[koff] = ndb.double().cpu().numpy()
    t_scan = time.time() - t_scan0
    log(f"代表性扫描：{n_full} 窗 baseline-only（{n_batches} 批 × {scan_batch} 窗/批，每批一次 "
        f"sample()、B×K={scan_batch * PC.K} 展平）{t_scan:.1f}s"
        f"（{t_scan / n_full * 1000:.0f} ms/窗）——附加成本，不进主 Run 外推")
    np.save(out / "scan_s.npy", s_scan)
    np.save(out / "scan_null.npy", nd_scan)

    # ---- 代表性对比：同一扫描内 32 窗子样本 vs 930 窗全集（同噪声 → 隔离窗选择效应）----
    from scipy.stats import ks_2samp
    sub_off = np.asarray(win_global) - seg.start
    QGRID_CMP = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
    repr_stats, qq_panels = {}, []
    for key, title, label, arr in (("s_wB", "baseline scale s(w,B)", "s(w,B)", s_scan),
                                   ("null_d", "null pair distance d(i,j)", "d(i,j)", nd_scan)):
        full_v = arr.reshape(-1).astype(np.float64)
        sub_v = arr[sub_off].reshape(-1).astype(np.float64)
        qt = {f"q{int(q * 100):02d}": {"full": float(np.quantile(full_v, q)),
                                       "sub": float(np.quantile(sub_v, q))}
              for q in QGRID_CMP}
        for v in qt.values():
            v["rel_diff"] = (v["sub"] - v["full"]) / v["full"]
        ks = ks_2samp(sub_v, full_v)
        central = [abs(v["rel_diff"]) for kk, v in qt.items()
                   if kk in ("q05", "q10", "q25", "q50", "q75", "q90", "q95")]
        repr_stats[key] = {"n_full": int(full_v.size), "n_sub": int(sub_v.size),
                           "quantiles": qt, "ks_stat": float(ks.statistic),
                           "ks_p": float(ks.pvalue),
                           "max_abs_rel_diff_q05_q95": float(np.max(central))}
        qq_panels.append({"label": label, "title": title, "sub": sub_v, "full": full_v,
                          "stat_lines": [f"KS D={ks.statistic:.3f}, p={ks.pvalue:.1e}",
                                         f"n_full={full_v.size}, n_sub={sub_v.size}"]})
        log(f"代表性[{key}]：KS D={ks.statistic:.4f}（p={ks.pvalue:.2g}）；分位(5–95%)最大相对"
            f"偏差 {np.max(central) * 100:.2f}%（n_full={full_v.size}，n_sub={sub_v.size}）")
    # τ 三口径对照：主链池（eps=2026+wseq）vs 扫描子样本（同 32 窗、不同 eps=2026+1e6+k）
    # vs 扫描全集（930 窗）——前差 = 噪声稳健性，后差 = 窗选择效应
    nd_full_flat = nd_scan.reshape(-1).astype(np.float64)
    nd_sub_flat = nd_scan[sub_off].reshape(-1).astype(np.float64)
    taus_scan_sub = {f"q{int(q * 100)}": float(np.quantile(nd_sub_flat, q))
                     for q in PC.TAU_QUANTILES}
    taus_scan_full = {f"q{int(q * 100)}": float(np.quantile(nd_full_flat, q))
                      for q in PC.TAU_QUANTILES}
    tau_compare = {"chain_pool_32w": taus, "scan_subsample_32w": taus_scan_sub,
                   "scan_full_930w": taus_scan_full}
    for kk in taus:
        log(f"τ[{kk}] 三口径：主链池 {taus[kk]:.4f} / 扫描子样本(同32窗不同eps) "
            f"{taus_scan_sub[kk]:.4f}（噪声效应 {abs(taus_scan_sub[kk] / taus[kk] - 1) * 100:.1f}%）"
            f"/ 扫描全集(930窗) {taus_scan_full[kk]:.4f}"
            f"（窗选择效应 {abs(taus_scan_sub[kk] / taus_scan_full[kk] - 1) * 100:.1f}%）")

    # ---- τ 决策数据预览（只算不判：预览、非 readout 承诺、τ 口径待用户拍板）----
    offdiag = ~np.eye(PC.N_SRC, dtype=bool)
    dbar9_rb = np.transpose(dbar9, (0, 2, 1))                       # (A, B, r)：readout 契约
    slope = RO.slope_readout(dbar9_rb)
    # (a) 预注册 τ（独立样本对池 q90/95/99）下删失率（跨通道 3782 边）
    censor_prereg = {}
    for kk, tv in taus.items():
        rs = RO.rstar_readout(dbar9_rb, tv)
        censor_prereg[kk] = float((rs[offdiag] > PC.R_MAX).mean())
        log(f"τ 预览(a) 删失率[{kk}]（预注册口径，跨通道 {int(offdiag.sum())} 边，32 窗 median "
            f"D̄）：{censor_prereg[kk] * 100:.1f}%——预览、非 readout 承诺、τ 口径待拍板")
    # (b) 相对自响应口径 D̃_{A→B}(r) = D̄_{A→B}(r)/D̄_{B→B}(r)（同 r、逐边；r=0 置 0）
    dtil9 = np.zeros((PC.N_SRC, PC.N_SRC, len(PC.R_GRID)))
    for j in range(PC.N_R_PERT):
        Mj = dbar[:, j, :]                                          # (62,62) 行A列B
        selfj = np.diag(Mj)                                         # D̄_{B→B}(r_j)
        dtil9[:, :, j + 1] = Mj / selfj[None, :]                    # 列 B 归一到 B 自响应
    r_sel = (0.08, 0.32, 1.28)
    dtil_stats = {}
    for r in r_sel:
        j = PC.PERT_R_GRID.index(r)
        v = dtil9[:, :, j + 1][offdiag]
        dtil_stats[f"r={r:g}"] = {"median": float(np.median(v)),
                                  "q1": float(np.quantile(v, 0.25)),
                                  "q3": float(np.quantile(v, 0.75)),
                                  "max": float(v.max())}
        log(f"τ 预览(b) D̃(r={r:g})：median={dtil_stats[f'r={r:g}']['median']:.3f}"
            f"（Q1={dtil_stats[f'r={r:g}']['q1']:.3f}，Q3={dtil_stats[f'r={r:g}']['q3']:.3f}，"
            f"max={dtil_stats[f'r={r:g}']['max']:.3f}）")
    dtil_censor = {}
    for tt in (0.25, 0.50):
        rs = RO.rstar_readout(dtil9, tt)
        dtil_censor[f"ttil={tt:.2f}"] = float((rs[offdiag] > PC.R_MAX).mean())
        log(f"τ 预览(b) 相对口径 r* 删失率[τ̃={tt:g}]：{dtil_censor[f'ttil={tt:.2f}'] * 100:.1f}%")
    # (c) 低剂量段 slope 跨通道分布
    sv = slope[offdiag].astype(np.float64)
    slope_stats = {"median": float(np.median(sv)), "q1": float(np.quantile(sv, 0.25)),
                   "q3": float(np.quantile(sv, 0.75)), "max": float(sv.max()),
                   "min": float(sv.min())}
    log(f"τ 预览(c) slope（低剂量段 {PC.SLOPE_SEG}，跨通道 3782 边）："
        f"median={slope_stats['median']:.4f}（Q1={slope_stats['q1']:.4f}，"
        f"Q3={slope_stats['q3']:.4f}，max={slope_stats['max']:.4f}，min={slope_stats['min']:.4f}）"
        f"——slope readout 不依赖 τ，主 readout 设计上受保护")
    log("τ 预览（a/b/c）全部为决策数据：预览、非 readout 承诺、τ 口径待用户拍板")

    # ---- 计时外推（10 例 × 32 窗 = 320 窗）----
    t_s = np.array([w["t_sample_s"] for w in win_times], dtype=np.float64)
    t_w = np.array([w["t_sample_s"] + w["t_post_s"] for w in win_times], dtype=np.float64)
    n_main_windows = len(PC.EXPECTED_TEST_IDS) * PC.N_W
    est_samp = float(t_s[1:].mean()) * n_main_windows              # 排除首窗 warmup
    est_total = t_setup + est_samp
    est_total_h = est_total / 3600.0
    tier_keep = (est_total_h <= 4.0) and (peak_rsvd < gpu_total)
    log(f"计时：sample 段 mean={t_s.mean():.2f}s/窗（排除首窗 {t_s[1:].mean():.2f}s；"
        f"std={t_s.std():.2f}）；后处理 {t_w[1:].mean() - t_s[1:].mean():.2f}s/窗")
    log(f"外推主 Run（10 例 × {PC.N_W} 窗 = {n_main_windows} 窗）：约 {est_total / 60:.0f} min"
        f"（{est_total_h:.2f} h；阈值 4 h）+ ~10% 余量 ≈ {est_total * 1.1 / 60:.0f} min"
        f"——{'未超' if est_total_h <= 4.0 else '已超'}阈值")
    log(f"档位判定数据：外推 {est_total_h:.2f} h（阈值 4 h）、GPU 峰值 reserved "
        f"{peak_rsvd / 2**30:.2f}/{gpu_total / 2**30:.1f} GiB → "
        f"{'维持 N_w=32/K=8/网格（原则维持，档位调整权留给 τ/网格决策）' if tier_keep else '建议降档（见实验块理由）'}")

    # ---- 图（common/visualizer 纯增量：dose_response 复用 + qq_compare/tau_preview 新类型）----
    import visualizer as VZ
    viz = VZ.Visualizer(str(out / "figs"))
    j_top = PC.N_R_PERT - 1
    M = dbar[:, j_top, :]
    self_r, cross_r = np.diag(M), M[offdiag]
    a_self = int(np.argmax(self_r))
    pairs = np.argwhere(offdiag)
    order = np.argsort(cross_r)[::-1]
    (a_s, b_s), (a_m, b_m), (a_w, b_w) = (tuple(map(int, pairs[order[i]]))
                                          for i in (0, len(order) // 2, -1))
    picks = {"self": (a_self, a_self), "cross_strong": (a_s, b_s),
             "cross_median": (a_m, b_m), "cross_weak": (a_w, b_w)}
    curves, labels = [], []
    for kind, (a, b) in picks.items():
        curves.append(dbar9[a, :, b])
        labels.append(f"A=B={a} (self)" if kind == "self" else f"A={a}->B={b} ({kind.split('_')[1]})")
    fig_dr = viz.render("dose_response", {
        "r_grid": np.asarray(PC.R_GRID, dtype=np.float64), "curves": np.stack(curves),
        "labels": labels, "kinds": list(picks.keys()), "tau": tau_main,
        "tau_label": f"tau = null q{int(PC.TAU_MAIN * 100)}",
        "tau_sens": {k: v for k, v in taus.items() if v != tau_main},
        "note": (f"subject {sid}, median of {PC.N_W} windows (bench-01);\n"
                 f"tau from 32-window null pool ({pooled.size} samples)"),
    }, "dose_response_bench.png")
    fig_qq = viz.render("qq_compare", {
        "panels": qq_panels,
        "note": ("subsample = the N_w=32 registered windows, taken from the same 930-window "
                 "baseline-only scan\n(same per-window eps seeds -> isolates window-selection "
                 "effect; perturbed side not scanned, design Sec.3)"),
    }, "qq_representativeness.png")
    j_sel_idx = [PC.PERT_R_GRID.index(r) + 1 for r in r_sel]
    fig_tau = viz.render("tau_preview", {
        "null_pool": pooled, "dbar_cross_rmax": cross_r, "taus": taus,
        "dtil": {"r_labels": [f"r={r:g}" for r in r_sel],
                 "boxes": [dtil9[:, :, j][offdiag] for j in j_sel_idx],
                 "lines": {"ttil=0.25": 0.25, "ttil=0.50": 0.50},
                 "censor_lines": [f"censored @0.25: {dtil_censor['ttil=0.25'] * 100:.1f}%",
                                  f"censored @0.50: {dtil_censor['ttil=0.50'] * 100:.1f}%"]},
        "slope": {"values": sv, "stats": {"median": slope_stats["median"],
                                          "q1": slope_stats["q1"], "q3": slope_stats["q3"]}},
        "note": ("PREVIEW ONLY - not a readout commitment; tau caliber pending decision. "
                 f"subject {sid}, median of {PC.N_W} windows"),
    }, "tau_preview.png")
    log(f"图：{fig_dr}、{fig_qq}、{fig_tau}")

    # ---- 落盘：timing.json / config_snapshot.json / bench_metrics.json ----
    timing = {
        "per_window": win_times,
        "t_sample_s": {"mean": float(t_s.mean()), "mean_excl_first": float(t_s[1:].mean()),
                       "median": float(np.median(t_s)), "std": float(t_s.std()),
                       "min": float(t_s.min()), "max": float(t_s.max())},
        "t_window_total_s": {"mean_excl_first": float(t_w[1:].mean())},
        "setup_s": round(t_setup, 1), "chain_total_s": round(t_chain, 1),
        "peak_memory_GiB": {"allocated": peak_alloc / 2**30, "reserved": peak_rsvd / 2**30,
                            "gpu_total": gpu_total / 2**30,
                            "pre_loop_baseline": mem_base / 2**30},
        "extrapolation_main_run": {
            "n_subjects": len(PC.EXPECTED_TEST_IDS), "n_windows": n_main_windows,
            "est_sampling_s": round(est_samp, 1), "est_total_s": round(est_total, 1),
            "est_total_min": round(est_total / 60, 1), "est_total_h": round(est_total_h, 3),
            "est_total_h_with_10pct_margin": round(est_total_h * 1.1, 3),
            "threshold_h": 4.0, "within_threshold": bool(est_total_h <= 4.0)},
        "representativeness_scan": {"n_windows": n_full, "batch": scan_batch,
                                    "n_batches": n_batches, "total_s": round(t_scan, 1),
                                    "per_window_ms": round(t_scan / n_full * 1000, 1),
                                    "note": "bench-only 附加成本，不计入主 Run 外推"},
        "tier_recommendation": ("maintain" if tier_keep else "downgrade-considered"),
    }
    (out / "timing.json").write_text(
        json.dumps(timing, indent=2, ensure_ascii=False), encoding="utf-8")

    consts = {k: (list(v) if isinstance(v, tuple) else v)
              for k, v in vars(PC).items()
              if k.isupper() and isinstance(v, (int, float, str, bool, tuple))}
    snap = {
        "task": "10-17-perturb-ec-f0", "stage": "bench", "run_id": run_id,
        "git_head": git_head,
        "calibers": "四要素口径 = Run 001 全部不变（P/D/τ/网格/K/N_w/seed 公式/方向约定）",
        "constants": consts,
        "model": {"ckpt": PC.CKPT_PATH, "config_hash": ck["config_hash"],
                  "epoch": ck["epoch"], "weights": "EMA",
                  "euler_steps": cfg.euler_steps, "sigma_min": cfg.sigma_min,
                  "gen_lowpass_hz": sampler.gen_lowpass_hz},
        "sampler": {"K": PC.K, "chunk_bk": sampler.chunk_bk,
                    "n_variants_per_window": PC.N_VARIANTS},
        "data": {"subject_id": sid, "n_windows": PC.N_W,
                 "window_global_indices": win_global,
                 "window_subsample": "default_rng(2026*1000003+sid) 无放回升序，全 32 窗",
                 "eps_seed_formula": "2026 + win_idx（win_idx=0..31；窗 0-1 = smoke-01 前缀）",
                 "scan": {"purpose": "N_w=32 子采样代表性抽查（baseline 侧，design §7-7）",
                          "n_windows_full": n_full, "batch_windows": scan_batch,
                          "global_index_range": [seg.start, seg.stop],
                          "eps_seed_formula": "2026 + 1_000_000 + 窗内偏移（0..929，逐窗独立，与批划分无关）"}},
        "env": {"python": sys.version.split()[0], "torch": torch.__version__,
                "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                "blas_threads": os.environ.get("OMP_NUM_THREADS", "?")},
    }
    (out / "config_snapshot.json").write_text(
        json.dumps(snap, indent=2, ensure_ascii=False), encoding="utf-8")

    metrics = {
        "git_head": git_head, "subject_id": sid, "n_windows": PC.N_W,
        "window_global_indices": win_global,
        "DISCLAIMER": "τ 预览（censoring_prereg/dtil/slope 分布）= 决策数据：预览、"
                      "非 readout 承诺、τ 口径待用户拍板",
        "taus_chain_pool": taus, "null_pool_size": int(pooled.size),
        "tau_compare": tau_compare,
        "censoring_prereg": censor_prereg,
        "dtil_relative_caliber": {"r_selected": list(r_sel), "stats": dtil_stats,
                                  "rstar_censoring": dtil_censor},
        "slope_cross": slope_stats,
        "representativeness": repr_stats,
        "timing_summary": {"t_sample_mean_excl_first_s": float(t_s[1:].mean()),
                           "est_total_h": round(est_total_h, 3),
                           "peak_reserved_GiB": round(peak_rsvd / 2**30, 2),
                           "tier_recommendation": timing["tier_recommendation"]},
        "figs": {"dose_response": str(fig_dr), "qq": str(fig_qq), "tau_preview": str(fig_tau)},
        "elapsed_s": round(time.time() - t0, 1),
    }
    (out / "bench_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"落盘：d_curves.npy {d_curves.shape}、null_pool.npy {null_pool.shape}、"
        f"dbar_median.npy {dbar9.shape}、scan_s.npy {s_scan.shape}、scan_null.npy "
        f"{nd_scan.shape}、timing.json、config_snapshot.json、bench_metrics.json")
    log(f"bench 完成：总耗时 {time.time() - t0:.1f}s；结果/结论回填 experiment.md Run 002")


def _env_snapshot_text() -> str:
    """env 快照文本（Run 003 复现三要素补齐）：版本信息 + nvidia-smi + pip freeze。"""
    lines = [
        f"python={sys.version.split()[0]}", f"torch={torch.__version__}",
        f"cuda={torch.version.cuda}", f"gpu={torch.cuda.get_device_name(0)}",
        f"blas_threads={os.environ.get('OMP_NUM_THREADS', '?')}",
    ]
    try:
        nvr = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL).strip()
        lines.append(f"nvidia_smi={nvr}")
    except Exception as e:                                    # noqa: BLE001（快照尽力而为）
        lines.append(f"nvidia_smi=(failed: {e})")
    try:
        freeze = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True, stderr=subprocess.DEVNULL)
    except Exception as e:                                    # noqa: BLE001（快照尽力而为）
        freeze = f"(pip freeze failed: {e})"
    return "\n".join(lines) + "\n\n# pip freeze\n" + freeze


def stage_main(run_id: str = "main-01", n_subjects: int = None) -> None:
    """B3 主 Run（implement.md B3 / experiment.md Run 003）：fold0 test 10 例全量 × N_w=32
    扰动批采样 → 逐例落盘 + 断点续跑。

    - 四要素口径 = Run 001/002 **全部不变**；例序 = EXPECTED_TEST_IDS 升序（首例
      subject 10，非 test_ds 自然序——Run 001 勘误口径）；win_idx = 主 Run 内全局窗序号
      **跨例连续**（例 i 窗 j → N_W·i + j，0..319）→ subject 10 win_idx 0..31 与
      bench-01 同 seed 同批形 → 跑完逐位断言。
    - 断点续跑（design §8 回滚点③）：例完成 = 四文件 + done_s{sid}.json；例内每窗
      partial_s{sid}.npz（窗数据与 eps/eta 仅依赖 win_idx ⇒ 续跑结果不变）；重跑
      跳过已完成例。
    - τ 口径决策进行中（主会话/用户）：本 stage 只落盘 D 曲线与 null 池（τ 无关数据）；
      tau_s{sid}.json 三分位照 bench 同式（null 逐例分池）落盘，标「预注册绝对口径，
      供登记」。readout/比较 = B4/B5，不做。
    """
    PC = PB.load_config()
    out = Path(PC.OUTPUTS_ROOT) / run_id
    (out / "logs").mkdir(parents=True, exist_ok=True)             # nohup 前建（硬约束）
    log = _logger(out / "logs" / "run.log")
    t0 = time.time()
    git_head = _git_head()
    log(f"stage=main run_id={run_id} outputs={out} git={git_head} "
        f"torch={torch.__version__} gpu={torch.cuda.get_device_name(0)}")

    # ---- 冻结装载 + 口径断言（同 Run 001/002）----
    t_load0 = time.time()
    cfg, model, ema, prior, sampler, ck = LD.load_frozen_model("cuda")
    torch.backends.cudnn.deterministic = cfg.cudnn_deterministic
    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    assert sampler.gen_lowpass_hz is None, "gen_lowpass_hz 须 None（fmt 口径，B-Major1 陷阱）"
    _loader_obj, test_ds = LD.load_test_loader(batch_size=4, num_workers=2)
    subs = LD.subsample_windows_per_subject(test_ds)
    sids = [int(s) for s in PC.EXPECTED_TEST_IDS]
    if n_subjects is not None:
        sids = sids[:n_subjects]
    for sid in sids:
        assert sid in subs and len(subs[sid]) >= PC.N_W, \
            f"subject {sid} 子采样窗数 {len(subs.get(sid, []))} 不足 N_W={PC.N_W}（test 折漂移？）"
    n_pairs = PC.K * (PC.K - 1) // 2
    log(f"θ* 装载：epoch={ck['epoch']} hash={ck['config_hash']} gen_lowpass_hz=None（断言过）；"
        f"例序（EXPECTED_TEST_IDS 升序）={sids}；每例 N_W={PC.N_W} 窗、win_idx 跨例连续"
        f"0..{len(sids) * PC.N_W - 1}（例 i 窗 j → {PC.N_W}·i+j）；setup "
        f"{time.time() - t_load0:.1f}s；断点续跑：done 标记跳过已完成例 + 例内 partial 续")

    # ---- 主循环：逐例逐窗 497 并批 CRN 采样（每窗一次 sample()）----
    n_total_win = len(sids) * PC.N_W
    subject_rows, win_times = [], []
    n_done_win = 0                                                # 全 Run 口径完成窗数（含跳过例）
    torch.cuda.reset_peak_memory_stats()
    for si, sid in enumerate(sids):
        win_global = [int(i) for i in subs[sid][:PC.N_W]]
        f_d = out / f"d_curves_s{sid}.npy"
        f_n = out / f"null_s{sid}.npy"
        f_db = out / f"dbar_median_s{sid}.npy"
        f_tau = out / f"tau_s{sid}.json"
        f_done = out / f"done_s{sid}.json"
        if f_done.exists() and all(p.exists() for p in (f_d, f_n, f_db, f_tau)):
            prev = json.loads(f_done.read_text(encoding="utf-8"))
            prev["skipped_this_session"] = True
            subject_rows.append(prev)
            n_done_win += PC.N_W
            log(f"subject {sid}（例 {si + 1}/{len(sids)}）：已完成（done 标记在，断点续跑跳过）")
            continue

        # 例内断点：partial_s{sid}.npz（n_done 计数 + 逐窗数组）
        f_partial = out / f"partial_s{sid}.npz"
        d_all, null_all = [], []
        if f_partial.exists():
            try:
                pz = np.load(f_partial)
                n_prev = int(pz["n_done"])
                d_all = [pz[f"d{u}"] for u in range(n_prev)]
                null_all = [pz[f"n{u}"] for u in range(n_prev)]
                log(f"subject {sid}（例 {si + 1}/{len(sids)}）：例内断点续跑自窗 "
                    f"{n_prev + 1}/{PC.N_W}（partial 读取 {n_prev} 窗）")
            except Exception as e:                                # noqa: BLE001（损坏即重算该例）
                d_all, null_all = [], []
                log(f"subject {sid}：partial 读取失败（{e}），该例从头重算")

        t_sub0 = time.time()
        sub_t_sample = []
        for wseq in range(len(d_all), PC.N_W):
            gidx = win_global[wseq]
            win_idx = si * PC.N_W + wseq                         # 全局窗序号（跨例连续）
            xp, _ = test_ds[gidx]
            xp = xp.to("cuda")
            torch.cuda.synchronize()
            ta = time.time()
            with ema.swap(model):
                sv = RS.sample_window_variants(sampler, xp, win_idx=win_idx)
            torch.cuda.synchronize()
            tb = time.time()
            assert sv.shape == (PC.N_VARIANTS, PC.K, cfg.n_channels, cfg.future_len)
            assert torch.isfinite(sv).all(), f"subject {sid} 窗 {wseq}（全局 {gidx}）采样含非有限值"
            D = RS.d_curves(sv)                                  # (62, 8, 62) [A, r档, B]
            nd_ = RS.null_distances(sv[0])                       # (28, 62)
            assert torch.isfinite(D).all() and (D >= 0).all() \
                and torch.isfinite(nd_).all() and (nd_ >= 0).all(), \
                f"subject {sid} 窗 {wseq} D/null 含 NaN/负值"
            d_all.append(D.float().cpu().numpy())
            null_all.append(nd_.float().cpu().numpy())
            np.savez(f_partial, n_done=len(d_all),                              # 逐窗断点
                     **{f"d{u}": a for u, a in enumerate(d_all)},
                     **{f"n{u}": a for u, a in enumerate(null_all)})
            tc = time.time()
            sub_t_sample.append(tb - ta)
            win_times.append({"subject_id": sid, "si": si, "wseq": wseq, "gidx": gidx,
                              "win_idx": win_idx, "t_sample_s": round(tb - ta, 3),
                              "t_post_s": round(tc - tb, 3)})
            done_all = n_done_win + len(d_all)
            if win_times:
                eta_s = float(np.mean([w["t_sample_s"] + w["t_post_s"] for w in win_times])) \
                    * (n_total_win - done_all)
            else:
                eta_s = 0.0
            log(f"例 {si + 1}/{len(sids)}（subject {sid}）窗 {len(d_all)}/{PC.N_W}"
                f"（全局 {gidx}，win_idx={win_idx}）：sample {tb - ta:.2f}s + 后处理 "
                f"{tc - tb:.2f}s；总进度 {done_all}/{n_total_win}，"
                f"ETA {eta_s / 60:.0f} min（累计 {time.time() - t0:.1f}s）")

        # ---- 例完成：聚合 + 四文件 + done 标记，删 partial ----
        d_curves = np.stack(d_all)                               # (32, 62, 8, 62)
        null_pool = np.stack(null_all)                           # (32, 28, 62)
        assert d_curves.shape == (PC.N_W, PC.N_SRC, PC.N_R_PERT, PC.N_SRC) \
            and null_pool.shape == (PC.N_W, n_pairs, PC.N_SRC), \
            f"subject {sid} 形状异常：{d_curves.shape}/{null_pool.shape}"
        assert np.isfinite(d_curves).all() and np.isfinite(null_pool).all(), \
            f"subject {sid} D/null 含 NaN"
        dbar = np.median(d_curves, axis=0).astype(np.float64)     # (62, 8, 62) [A, r档, B]
        dbar9 = np.concatenate(
            [np.zeros((PC.N_SRC, 1, PC.N_SRC)), dbar], axis=1)    # (62, 9, 62) 索引0 ≡ r=0 ≡ 0
        pooled = null_pool.reshape(-1).astype(np.float64)
        taus = {f"q{int(q * 100)}": float(np.quantile(pooled, q)) for q in PC.TAU_QUANTILES}
        np.save(f_d, d_curves)
        np.save(f_n, null_pool)
        np.save(f_db, dbar9)
        tau_json = {
            "subject_id": sid,
            "caliber": "预注册绝对口径（null 独立样本对逐例分池分位，Run 001 §2.3 登记）——"
                       "供登记；τ 口径决策（绝对 vs 相对自响应）由主会话/用户进行中，"
                       "本文件不构成 readout 承诺",
            "tau_main": f"q{int(PC.TAU_MAIN * 100)}",
            "null_pool_shape": list(null_pool.shape),
            "null_pool_size": int(pooled.size),
            "tau_quantiles": taus,
        }
        f_tau.write_text(json.dumps(tau_json, indent=2, ensure_ascii=False), encoding="utf-8")
        t_sub = time.time() - t_sub0
        done_row = {"subject_id": sid, "n_windows": PC.N_W,
                    "win_idx_range": [si * PC.N_W, si * PC.N_W + PC.N_W - 1],
                    "window_global_indices": win_global,
                    "t_subject_s": round(t_sub, 1),
                    "t_sample_mean_s": round(float(np.mean(sub_t_sample)), 3) if sub_t_sample else None,
                    "taus": taus, "null_pool_size": int(pooled.size),
                    "skipped_this_session": False}
        f_done.write_text(json.dumps(done_row, indent=2, ensure_ascii=False), encoding="utf-8")
        f_partial.unlink(missing_ok=True)
        subject_rows.append(done_row)
        n_done_win += PC.N_W
        tsm = done_row["t_sample_mean_s"]
        tsm_str = f"{tsm:.2f}" if tsm is not None else "n/a(全自 partial 续)"
        log(f"subject {sid}（例 {si + 1}/{len(sids)}）完成：{PC.N_W} 窗 {t_sub:.1f}s"
            f"（sample mean {tsm_str}s/窗）；"
            f"null 池 {pooled.size} 样本 τ："
            + "、".join(f"{k}={v:.4f}" for k, v in taus.items())
            + f"；落盘 {f_d.name}/{f_n.name}/{f_db.name}/{f_tau.name} + done 标记（总进度 "
            f"{n_done_win}/{n_total_win}，累计 {time.time() - t0:.1f}s）")

    peak_alloc = torch.cuda.max_memory_allocated()
    peak_rsvd = torch.cuda.max_memory_reserved()
    gpu_total = torch.cuda.get_device_properties(0).total_memory
    t_sampling = time.time() - t0
    log(f"采样循环结束：{len(sids)} 例 × {PC.N_W} 窗（本会话实采 {len(win_times)} 窗，"
        f"跳过 {n_total_win - len(win_times)} 窗）{t_sampling:.1f}s；GPU 峰值：allocated "
        f"{peak_alloc / 2**30:.2f} GiB / reserved {peak_rsvd / 2**30:.2f} GiB / 卡总 "
        f"{gpu_total / 2**30:.1f} GiB")

    # ---- 完成后汇总断言：例数 / shape / finite / subject 10 ≡ bench-01 逐位 ----
    log(f"汇总断言：{len(sids)} 例全在、shape/finite 齐全、subject {sids[0]} ≡ bench-01 逐位")
    bench_cmp = None
    bench_dir = Path(PC.OUTPUTS_ROOT) / "bench-01"
    for si, sid in enumerate(sids):
        d = np.load(out / f"d_curves_s{sid}.npy")
        nul = np.load(out / f"null_s{sid}.npy")
        db = np.load(out / f"dbar_median_s{sid}.npy")
        assert d.shape == (PC.N_W, PC.N_SRC, PC.N_R_PERT, PC.N_SRC) \
            and nul.shape == (PC.N_W, n_pairs, PC.N_SRC) \
            and db.shape == (PC.N_SRC, len(PC.R_GRID), PC.N_SRC), f"subject {sid} shape 异常"
        assert np.isfinite(d).all() and (d >= 0).all() \
            and np.isfinite(nul).all() and (nul >= 0).all() \
            and np.isfinite(db).all() and (db >= 0).all(), f"subject {sid} 产物含 NaN/负值"
        assert json.loads((out / f"tau_s{sid}.json").read_text(encoding="utf-8"))["subject_id"] == sid
        if si == 0 and (bench_dir / "d_curves.npy").exists():
            bd = np.load(bench_dir / "d_curves.npy")
            bn = np.load(bench_dir / "null_pool.npy")
            exact = bool(np.array_equal(d, bd) and np.array_equal(nul, bn))
            close = bool(np.allclose(d, bd, atol=1e-6, rtol=0)
                         and np.allclose(nul, bn, atol=1e-6, rtol=0))
            assert close, \
                f"subject {sid} 与 bench-01 不复现（seed/批形/代码漂移——排查后再进 B4）"
            bench_cmp = {"d_and_null_array_equal": exact, "allclose_atol_1e-6": close,
                         "bench_dir": str(bench_dir)}
            log(f"确定性复现断言通过：subject {sid} d_curves + null ≡ bench-01"
                f"（exact 逐位相等={exact}，allclose(1e-6)={close}）")
    log(f"汇总断言通过：{len(sids)} 例 × 4 文件齐全、shape (32,62,8,62)/(32,28,62)/(62,9,62)、"
        f"全有限非负、τ json 逐例对齐")

    # ---- 落盘：config_snapshot / env_snapshot / main_metrics ----
    consts = {k: (list(v) if isinstance(v, tuple) else v)
              for k, v in vars(PC).items()
              if k.isupper() and isinstance(v, (int, float, str, bool, tuple))}
    snap = {
        "task": "10-17-perturb-ec-f0", "stage": "main", "run_id": run_id,
        "git_head": git_head,
        "calibers": "四要素口径 = Run 001/002 全部不变（10 例全量；档位维持 Run 002 定版）；"
                    "本 Run 只落 D 曲线与 null 池，readout = B4",
        "constants": consts,
        "model": {"ckpt": PC.CKPT_PATH, "config_hash": ck["config_hash"],
                  "epoch": ck["epoch"], "weights": "EMA",
                  "euler_steps": cfg.euler_steps, "sigma_min": cfg.sigma_min,
                  "gen_lowpass_hz": sampler.gen_lowpass_hz},
        "sampler": {"K": PC.K, "chunk_bk": sampler.chunk_bk,
                    "n_variants_per_window": PC.N_VARIANTS},
        "data": {
            "subject_ids": sids,
            "iteration_order": "EXPECTED_TEST_IDS 升序（Run 001 勘误口径，首例 subject 10）",
            "n_windows_per_subject": PC.N_W,
            "win_idx_formula": "win_idx = N_W*si + wseq（全局窗序号跨例连续 0..319；"
                               f"subject {sids[0]} = 0..{PC.N_W - 1} 与 bench-01 同）",
            "window_subsample": "default_rng(2026*1000003+sid) 无放回升序，全 32 窗（同 bench-01）",
            "eps_seed_formula": "2026 + win_idx（逐窗一组 eps/eta 跨 baseline+全变体共享）",
            "window_global_indices_per_subject": {str(r["subject_id"]): r["window_global_indices"]
                                                  for r in subject_rows},
        },
        "resume": {"done_marker": "done_s{sid}.json（四文件齐则跳过该例）",
                   "partial": "partial_s{sid}.npz（例内逐窗断点，完成后删除）"},
        "tau_note": "tau_s{sid}.json = 预注册绝对口径（供登记）；τ 口径决策（绝对 vs 相对自响应）"
                    "进行中，本 Run 不做 readout 判定",
        "env": {"python": sys.version.split()[0], "torch": torch.__version__,
                "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                "blas_threads": os.environ.get("OMP_NUM_THREADS", "?"),
                "snapshot_file": "env_snapshot.txt"},
    }
    (out / "config_snapshot.json").write_text(
        json.dumps(snap, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "env_snapshot.txt").write_text(_env_snapshot_text(), encoding="utf-8")

    t_s_arr = np.array([w["t_sample_s"] for w in win_times], dtype=np.float64) \
        if win_times else np.array([], dtype=np.float64)
    metrics = {
        "git_head": git_head, "stage": "main", "run_id": run_id,
        "calibers": "四要素口径 = Run 001/002 不变；只落 D 曲线与 null 池（τ 无关数据）",
        "subjects": subject_rows,
        "timing": {
            "n_subjects": len(sids), "n_windows_total": n_total_win,
            "n_windows_sampled_this_session": len(win_times),
            "t_sample_mean_s": float(t_s_arr.mean()) if t_s_arr.size else None,
            "t_sample_std_s": float(t_s_arr.std()) if t_s_arr.size else None,
            "t_window_total_mean_s": (float(np.mean([w["t_sample_s"] + w["t_post_s"]
                                                     for w in win_times]))
                                      if win_times else None),
            "elapsed_s_this_session": round(time.time() - t0, 1)},
        "peak_memory_GiB": {"allocated": round(peak_alloc / 2**30, 2),
                            "reserved": round(peak_rsvd / 2**30, 2),
                            "gpu_total": round(gpu_total / 2**30, 1),
                            "note": "本会话采样循环内峰值（全例跳过的续跑会话无采样负载）"},
        "assertions": {"n_subjects_complete": len(subject_rows),
                       "shapes": "(32,62,8,62)/(32,28,62)/(62,9,62) 全例通过",
                       "finite_nonneg": "D/null/dbar 全例通过",
                       "bench01_subject10": bench_cmp},
        "per_window_times": win_times,
        "artifacts": {
            "per_subject": ["d_curves_s{sid}.npy", "null_s{sid}.npy",
                            "dbar_median_s{sid}.npy", "tau_s{sid}.json", "done_s{sid}.json"],
            "run_level": ["config_snapshot.json", "env_snapshot.txt", "main_metrics.json",
                          "logs/run.log", "logs/nohup.log"]},
    }
    (out / "main_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"落盘：逐例 d_curves_s*/null_s*/dbar_median_s*/tau_s* ×{len(sids)}、"
        f"config_snapshot.json、env_snapshot.txt、main_metrics.json（逐例耗时表）")
    log(f"main 完成：本会话 {time.time() - t0:.1f}s（实采 {len(win_times)} 窗）；"
        f"结果/结论回填 experiment.md Run 003")


def main() -> None:
    p = argparse.ArgumentParser(
        description="fold0 虚拟扰动→EC 有效路径推断（v4a-fmt 冻结代理脑，任务 10-17-perturb-ec-f0）")
    p.add_argument("--stage", required=True, choices=STAGES,
                   help="执行阶段（implement.md：smoke=1 例 2 窗全链 / bench=N_w=32 计时 / "
                        "main=10 例扰动批采样 / readout=三 readout+有效路径 / refs=GC·gPDC 参考 / "
                        "split-half=fmt 天花板 / compare=比较器 / figs=诊断图")
    p.add_argument("--subjects", type=int, default=None,
                   help="例数上限（smoke/bench；main 默认 fold0 test 全 10 例）")
    p.add_argument("--run-id", type=str, default=None,
                   help="outputs/perturb-ec-f0/<run-id>/（experiment.md Run 块先登记后跑）")
    args = p.parse_args()
    print(f"[run] stage={args.stage} subjects={args.subjects} run_id={args.run_id}")
    if args.stage == "smoke":
        stage_smoke(run_id=args.run_id or "smoke-01")
    elif args.stage == "bench":
        stage_bench(run_id=args.run_id or "bench-01", n_subjects=args.subjects or 1)
    elif args.stage == "main":
        stage_main(run_id=args.run_id or "main-01", n_subjects=args.subjects)
    else:
        print(f"[run] stage {args.stage!r} not implemented yet（implement.md B4/B5）")


if __name__ == "__main__":
    main()
