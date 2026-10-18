#!/usr/bin/env python3
"""diag_posthoc —— Run diag-01 两问合一诊断（bivar-gc-ec-v2 post-hoc；2026-10-18 用户追加）。

问 A —— 单受试者内部 TRGC 的分半信度：TRGC_s = GC_fwd_s − GC_rev_s 的奇偶分半
重测信度。fwd 与 rev 各自在奇/偶两半上重拟合（逐例 4 次矩阵拟合）：
  fwd:  build_halves(X_z, seg=500) → half_ec(p_max=48) 两半 → ρ_fwd
  rev:  X_rev = X_z[:, ::-1]（rev-01 同式）→ build_halves(seg=500) → half_ec 两半 → ρ_rev
  TRGC: TRGC_A = ec_fwdA − ec_revA、TRGC_B = ec_fwdB − ec_revB
        → ρ_TRGC = Spearman(offdiag TRGC_A, offdiag TRGC_B)
内置交叉验证：ρ_fwd 逐例对照 sh-01 split_half.json 已登记 ceiling.per_subject
（allclose 断言，同码径同 worker 契约应逐位一致；组级应复现 0.8981±0.1414）。

问 B —— GC_fwd 净出度（出−入）与单通道自相关/幅值的相关（诊断性登记，不做因果表述）：
  净出度 = offdiag 行和 − 列和（对角=0，62 维；组均值矩阵 + 逐例）
  自相关 = z-score 后全记录逐通道 lag-1 ACF 与滞后 1..48（=192 ms，P_MAX 物理窗）
           均值 ACF（有偏分母 T），逐例算后取组均值（逐例分布留 npz）
  幅值   = z-score 前原尺度逐通道 std（ddof=0，trgc-01 SNR 对照同式）组均值
  相关   = Spearman(净出度, lag-1 ACF / 窗口 ACF / 幅值)，组级 + 逐例分布（mean±sd）
  对照   = GC_fwd 出强度（不减入）vs 幅值（预期 trgc-01 SNR 对照 ≈0 带）
           + TRGC̄ 出强度 vs 幅值（trgc-01 登记值 allclose 复现断言）

判读框架（预登记）：TRGC = 净时序方向读数（fwd−rev 差分）；显著信度/相关 ≠ 生理
因果确证——双变量 GC 口径 + 头皮体积传导局限如实注明（沿 trgc-01 / rev-01 框架）。

用法：nohup python3 models/bivar-gc-ec-v2/diag_posthoc.py --run-id diag-01 &
      （确定性验证：同输入重跑入 scratch --run-id diag-01-reprod，产物 sha256 逐位比对）
确定性：问 A 16 进程 spawn（worker 单线程 BLAS，refs-01-reprod 同款逐位复现契约）；
问 B 纯计算无 RNG；summary 无时间戳/计时字段；np.savez 定长 zip 时间戳。
产物确定性范围 = diag_summary.json + data/diag_table.npz（logs/ 含时间戳不计入）。
"""

from __future__ import annotations

import os

# ---- BLAS 护栏（environment.md / run.py 同式：必须在 numpy 之前；setdefault 可覆盖）----
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse           # noqa: E402
import json               # noqa: E402
import multiprocessing    # noqa: E402
import sys                # noqa: E402
import time               # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np                          # noqa: E402
from scipy.stats import spearmanr           # noqa: E402

# ---- sys.path 装配（run.py 同式）：v2 目录居首、v1 仅追加（split_half 解析到 v1 只读复用）----
_HERE = Path(__file__).resolve().parent
_V1 = _HERE.parent / "bivar-gc-ec-v1"
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_V1) not in sys.path:
    sys.path.append(str(_V1))

import data_io            # noqa: E402（v2 版；连带经唯一名配方装载 bgec2_config）
import split_half as sh   # noqa: E402（v1 版，只读复用 build_halves/half_ec/offdiag_flat）
import bgec2_config as C  # noqa: E402（v2 config，唯一名）
import compare_rev as cr  # noqa: E402（同变体复用 _load_refs 装载断言；只读）

_BLAS_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")

# ---- 受保护 run 目录（既有产物禁止覆写）----
_PROTECTED_RUNS = ("smoke-01", "refs-01", "refs-01-reprod", "rev-01", "sh-01", "trgc-01")

# ---- 运行时口径锁常量（与 run.py stage_split 逐字同式；design §2.3）----
_SH_N_SEG = 120
_SH_ROWS_PER_HALF = 27120

# ---- 交叉验证容差（compare_rev._CV 同思路）：ρ_fwd 与 sh-01 登记值同为 f64 计算态、----
# ---- 同码径同 worker 契约（refs-01-reprod 已证逐位复现），期望逐位一致；容差仅兜底 ----
_CV_RTOL = 1e-12
_CV_ATOL = 1e-12


def _log_factory(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log_path, "a", encoding="utf-8")

    def log(msg: str):
        stamp = time.strftime("%H:%M:%S")
        line = f"[{stamp}] {msg}"
        print(line, flush=True)
        fh.write(line + "\n")
        fh.flush()
    return log


def _worker_init() -> None:
    """spawn worker 初始化：显式单线程 BLAS（16 进程 cgroup 上限，防 16×8 超订）。"""
    for v in _BLAS_VARS:
        os.environ[v] = "1"


# ================= 问 A：逐例 4 次半侧矩阵拟合（fwd/rev × A/B） =================

def _assert_half(r: dict, tag: str, sid) -> None:
    """半侧拟合的实现正确性断言（全有限 / 对角=0 / offdiag≥0；OLS 契约）。"""
    ec = r["ec"]
    Cn = C.N_CHANNELS
    assert ec.shape == (Cn, Cn)
    assert np.isfinite(ec).all(), f"s{sid} {tag}: 半侧 EC 非有限值"
    assert float(np.abs(np.diag(ec)).max()) == 0.0, f"s{sid} {tag}: 对角非零"
    assert (ec[~np.eye(Cn, dtype=bool)] >= 0).all(), \
        f"s{sid} {tag}: offdiag GC < 0（OLS 样本内单调性违约=实现错误）"


def _diag_worker(npz_path: str) -> dict:
    """spawn worker（问 A 主体）：fwd/rev 各自奇偶两半重拟合 → ρ_fwd / ρ_rev / ρ_TRGC。

    口径锁与 sh-01 _split_worker 逐字同式（build_halves(x, seg=500)，不调 v1
    ceiling_for_subject；断言 n_seg==120、每半 27120 行）；rev 半侧数据与 rev-01
    同式（X_rev = X_z[:, ::-1]，ascontiguousarray 仅内存布局、数值逐位不变）。
    """
    prep = data_io.prepare_subject(npz_path)
    sid = prep["meta"]["subject_id"]
    assert C.SEG_SAMPLES == 500, f"SEG_SAMPLES={C.SEG_SAMPLES} != 500（口径锁）"

    # fwd 两半（与 sh-01 逐字同式——ρ_fwd 即 sh-01 已登记 sh_rho，确定性互证靶）
    hf = sh.build_halves(prep["x"], seg=C.SEG_SAMPLES)
    assert hf["n_seg"] == _SH_N_SEG, \
        f"s{sid}: n_seg={hf['n_seg']} != {_SH_N_SEG}（60001//500=120；口径锁）"
    fa = sh.half_ec(hf["A"], p_max=C.P_MAX)
    fb = sh.half_ec(hf["B"], p_max=C.P_MAX)

    # rev 两半（X_rev = X_z[:, ::-1] 后再 build_halves；rev-01 同式）
    x_rev = np.ascontiguousarray(prep["x"][:, ::-1])
    hr = sh.build_halves(x_rev, seg=C.SEG_SAMPLES)
    assert hr["n_seg"] == _SH_N_SEG, f"s{sid}(rev): n_seg={hr['n_seg']} != {_SH_N_SEG}"
    ra = sh.half_ec(hr["A"], p_max=C.P_MAX)
    rb = sh.half_ec(hr["B"], p_max=C.P_MAX)

    for r, tag in ((fa, "fwdA"), (fb, "fwdB"), (ra, "revA"), (rb, "revB")):
        assert float(r["n"]) == _SH_ROWS_PER_HALF, \
            f"s{sid} {tag}: 行数={r['n']} != {_SH_ROWS_PER_HALF}（60×452；口径锁）"
        _assert_half(r, tag, sid)

    eye = ~np.eye(C.N_CHANNELS, dtype=bool)
    od_fa, od_fb = fa["ec"][eye], fb["ec"][eye]
    od_ra, od_rb = ra["ec"][eye], rb["ec"][eye]
    rho_fwd = float(spearmanr(od_fa, od_fb).statistic)
    rho_rev = float(spearmanr(od_ra, od_rb).statistic)
    trgc_a = fa["ec"] - ra["ec"]               # 带符号（差分），非负断言不适用
    trgc_b = fb["ec"] - rb["ec"]
    assert np.isfinite(trgc_a).all() and np.isfinite(trgc_b).all()
    rho_trgc = float(spearmanr(trgc_a[eye], trgc_b[eye]).statistic)
    return {"subject_id": int(sid), "group": prep["meta"]["group"],
            "rho_fwd": rho_fwd, "rho_rev": rho_rev, "rho_trgcsh": rho_trgc,
            "n_seg": hf["n_seg"], "n_a": fa["n"], "n_b": fb["n"]}


def _dist(vals) -> dict:
    """逐例分布登记：mean±sd（ddof=1）+ min/max。"""
    a = np.asarray(vals, dtype=np.float64)
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)),
            "min": float(a.min()), "max": float(a.max())}


# ================= 问 B：净出度 / ACF / 幅值 + Spearman =================

def _channel_acf(x: np.ndarray) -> tuple:
    """z-scored (62,T) → (lag-1 ACF, 滞后 1..P_MAX 均值 ACF) 逐通道（有偏分母 T）。

    r_k = (1/T)·Σ_t x[t]·x[t+k]（全记录、z-score 后 σ=1 ⇒ 标准 ACF 有偏估计）；
    滞后窗 1..P_MAX=48 = 192 ms（P_MAX 物理窗，design §2.2 同一滞后范围不变量）。
    """
    Cn, T = x.shape
    r = np.empty((Cn, C.P_MAX), dtype=np.float64)
    for lag in range(1, C.P_MAX + 1):
        r[:, lag - 1] = (x[:, :T - lag] * x[:, lag:]).sum(axis=1) / float(T)
    return r[:, 0].copy(), r.mean(axis=1)


def _channel_std(npz_path: str, sid) -> np.ndarray:
    """逐通道原尺度 std（z-score 前，ddof=0）——trgc-01 _load_channel_std 同式。"""
    d = np.load(npz_path, allow_pickle=True)
    assert str(d["subject_id"]) == str(sid), f"{npz_path}: npz subject_id 不符"
    x = np.asarray(d["data"], dtype=np.float64)
    assert x.shape == (C.N_CHANNELS, C.N_RAW), f"{npz_path}: 形状 {x.shape} 异常"
    return x.std(axis=1, ddof=0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-id", default="diag-01", help="本分析 run 目录名（产物落此处）")
    ap.add_argument("--n-proc", type=int, default=C.N_PROC, help="问 A 并行进程数")
    args = ap.parse_args()
    assert args.run_id not in _PROTECTED_RUNS, \
        f"run_id={args.run_id} 为既有产物目录，禁止覆写"
    run_dir = C.OUTPUTS_ROOT / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log = _log_factory(run_dir / "logs" / "run.log")

    # ---- 数据契约侧复核（compare_rev / trgc 同式）----
    manifest = data_io.load_manifest()
    ok_sids = sorted(s for s, e in manifest.items() if e["status"] == "ok")
    assert ok_sids == list(C.EXPECTED_IDS), \
        f"manifest ok 清单与 EXPECTED_IDS 不一致：{ok_sids}"
    groups = {str(s): manifest[s]["group"] for s in C.EXPECTED_IDS}

    # ---- 归属断言：split_half 解析到 v1（只读复用，v2 不定义同名核心模块）----
    assert Path(sh.__file__).resolve().parent == _V1.resolve(), \
        f"split_half 归属异常：{sh.__file__}"

    # ---- 交叉验证靶预检（缺失即报错，先于重拟合）----
    sh01_path = C.OUTPUTS_ROOT / "sh-01" / "split_half.json"
    sh01 = json.loads(sh01_path.read_text())
    sh01_reg = sh01["ceiling"]["per_subject"]
    assert sorted(int(k) for k in sh01_reg) == list(C.EXPECTED_IDS), \
        f"{sh01_path}: ceiling.per_subject 清单与 EXPECTED_IDS 不一致"
    trgc_dir = C.OUTPUTS_ROOT / "trgc-01"
    trgc_sum = json.loads((trgc_dir / "trgc_summary.json").read_text())
    trgc_tab = np.load(trgc_dir / "data" / "edges_table.npz")
    fwd_path = C.OUTPUTS_ROOT / "refs-01" / "refs_all.npz"
    rev_path = C.OUTPUTS_ROOT / "rev-01" / "refs_all.npz"
    assert fwd_path.exists() and rev_path.exists(), "refs-01 / rev-01 refs_all.npz 缺失"

    # ================= 问 A：48 例 × 4 半侧拟合（16 进程 spawn 并行）=================
    for v in _BLAS_VARS:
        os.environ[v] = "1"
    ctx = multiprocessing.get_context("spawn")
    n_proc = min(args.n_proc, len(C.EXPECTED_IDS))
    log(f"[diag A] 启动：{len(C.EXPECTED_IDS)} 例 × 4 半侧拟合（fwd/rev × A/B，"
        f"seg={C.SEG_SAMPLES}、p_max={C.P_MAX}；ρ_fwd 对照 sh-01 登记值 allclose）；"
        f"{n_proc} 进程 spawn 并行（worker 单线程 BLAS）")
    t0 = time.time()
    results = []
    with ctx.Pool(processes=n_proc, initializer=_worker_init) as pool:
        for r in pool.imap_unordered(_diag_worker,
                                     [manifest[s]["npz_path"] for s in C.EXPECTED_IDS]):
            results.append(r)
            log(f"[diag A {len(results)}/{len(C.EXPECTED_IDS)}] s{r['subject_id']} "
                f"({r['group']}) rho_fwd={r['rho_fwd']:.4f} rho_rev={r['rho_rev']:.4f} "
                f"rho_trgcsh={r['rho_trgcsh']:.4f}")
    results.sort(key=lambda r: r["subject_id"])
    sids = [str(r["subject_id"]) for r in results]
    assert sids == [str(s) for s in C.EXPECTED_IDS]

    # ---- 内置交叉验证：ρ_fwd vs sh-01 已登记逐例值（allclose 断言）----
    rho_fwd = np.array([r["rho_fwd"] for r in results], dtype=np.float64)
    rho_rev = np.array([r["rho_rev"] for r in results], dtype=np.float64)
    rho_trgc = np.array([r["rho_trgcsh"] for r in results], dtype=np.float64)
    reg = np.array([sh01_reg[s] for s in sids], dtype=np.float64)
    diffs = np.abs(rho_fwd - reg)
    assert bool(np.allclose(rho_fwd, reg, rtol=_CV_RTOL, atol=_CV_ATOL)), \
        f"ρ_fwd 与 sh-01 登记值不一致：max|Δ|={diffs.max():.3e}（超出 f64 逐位带）"
    log(f"[diag A] 交叉验证通过：ρ_fwd vs sh-01 ceiling.per_subject "
        f"max|Δ|={diffs.max():.3e}（allclose rtol={_CV_RTOL}, atol={_CV_ATOL}）")

    def _rho_block(vals: np.ndarray) -> dict:
        return {**_dist(vals),
                "per_subject": {s: float(v) for s, v in zip(sids, vals)}}

    qa = {
        "definition": "单受试者内部 TRGC 分半信度：fwd/rev 各自奇偶两半重拟合"
                      "（逐例 4 次矩阵拟合），TRGC_A = ec_fwdA − ec_revA、"
                      "TRGC_B = ec_fwdB − ec_revB，ρ = Spearman(offdiag TRGC_A, "
                      "offdiag TRGC_B)；组级 mean±sd（ddof=1）",
        "caliber_lock": {"seg_samples": C.SEG_SAMPLES, "n_seg": _SH_N_SEG,
                         "rows_per_half": _SH_ROWS_PER_HALF,
                         "note": "与 sh-01 运行时口径锁逐字同式（v2 本地编排，"
                                 "不调 v1 ceiling_for_subject）"},
        "fwd_split_half": _rho_block(rho_fwd),
        "rev_split_half": _rho_block(rho_rev),
        "trgc_split_half": _rho_block(rho_trgc),
        "cross_validation_vs_sh01": {
            "target": "sh-01/split_half.json::ceiling.per_subject",
            "assert": f"np.allclose(rtol={_CV_RTOL}, atol={_CV_ATOL})",
            "passed": True,
            "max_absdiff": float(diffs.max()),
            "sh01_registered_mean": float(sh01["ceiling"]["mean"]),
            "sh01_registered_sd": float(sh01["ceiling"]["sd"]),
            "reproduced_mean": float(rho_fwd.mean()),
            "reproduced_sd": float(rho_fwd.std(ddof=1)),
            "note": "同码径同 worker 契约（单线程 BLAS）期望逐位一致；sh-01 登记值"
                    "为 f64 计算态、本 Run 重拟合同为 f64 计算态",
        },
        "comparison_note": "三者并列（fwd 分半 = 噪声天花板参照 / TRGC 分半 = 净差分"
                           "读数信度 / rev 分半 = 反转侧同构天花板）；TRGC 为 fwd−rev "
                           "差分，误差传播预期使其低于两侧分半——实测以登记值为准",
    }

    # ================= 问 B：净出度 / ACF / 幅值 + Spearman（主进程纯计算）=================
    F, sids_f = cr._load_refs(fwd_path, expect_reversed=False)
    R, sids_r = cr._load_refs(rev_path, expect_reversed=True)
    assert sids_f == sids_r == sids, "fwd/rev/问A subject_ids 顺序不一致"
    n_subj = F.shape[0]

    # 逐例逐通道：lag-1 ACF / 窗口 ACF（z-score 后）/ 原尺度 std（z-score 前）
    acf1 = np.empty((n_subj, C.N_CHANNELS), dtype=np.float64)
    wacf = np.empty_like(acf1)
    cstd = np.empty_like(acf1)
    for k, sid in enumerate(C.EXPECTED_IDS):
        prep = data_io.prepare_subject(manifest[sid]["npz_path"])
        assert prep["x"].shape == (C.N_CHANNELS, C.N_RAW)
        acf1[k], wacf[k] = _channel_acf(prep["x"])
        cstd[k] = _channel_std(manifest[sid]["npz_path"], sid)
        if (k + 1) % 12 == 0:
            log(f"[diag B] 通道统计 {k + 1}/{n_subj} 例完成")
    acf1_g, wacf_g, cstd_g = acf1.mean(axis=0), wacf.mean(axis=0), cstd.mean(axis=0)

    # 通道 std 交叉验证：与 trgc-01 edges_table.npz chan_std（同式 48 例平均）对照
    std_cv_diff = float(np.max(np.abs(cstd_g - trgc_tab["chan_std"])))
    assert bool(np.allclose(cstd_g, trgc_tab["chan_std"], rtol=_CV_RTOL, atol=_CV_ATOL)), \
        f"通道 std 与 trgc-01 chan_std 不一致：max|Δ|={std_cv_diff:.3e}"

    # 净出度（行和−列和；对角=0）/ 出强度（行和，不减入）
    net_g = F.mean(axis=0)
    netout_group = net_g.sum(axis=1) - net_g.sum(axis=0)
    outstr_group = net_g.sum(axis=1)
    netout = F.sum(axis=2) - F.sum(axis=1)          # (48,62) 逐例（对角=0 无需掩膜）
    outstr = F.sum(axis=2)
    assert np.allclose(netout_group, netout.mean(axis=0), rtol=0, atol=1e-12)

    # 组级 Spearman（净出度 vs 三通道属性 + 出强度对照）
    sp_g = {
        "netout_vs_lag1acf": spearmanr(netout_group, acf1_g),
        "netout_vs_windowacf": spearmanr(netout_group, wacf_g),
        "netout_vs_amp": spearmanr(netout_group, cstd_g),
        "outstr_vs_amp": spearmanr(outstr_group, cstd_g),
    }
    # 逐例 Spearman 分布
    per = {k: [] for k in sp_g}
    for s in range(n_subj):
        per["netout_vs_lag1acf"].append(float(spearmanr(netout[s], acf1[s]).statistic))
        per["netout_vs_windowacf"].append(float(spearmanr(netout[s], wacf[s]).statistic))
        per["netout_vs_amp"].append(float(spearmanr(netout[s], cstd[s]).statistic))
        per["outstr_vs_amp"].append(float(spearmanr(outstr[s], cstd[s]).statistic))

    # SNR 对照复现断言：TRGC̄ 出强度 vs 通道 std 应复现 trgc-01 snr_control.spearman_out
    ch_out_trgc = (F - R).mean(axis=0).sum(axis=1)
    snr_repro = float(spearmanr(cstd_g, ch_out_trgc).statistic)
    snr_reg = float(trgc_sum["snr_control"]["spearman_out"]["rho"])
    assert abs(snr_repro - snr_reg) <= 1e-9, \
        f"TRGC̄ 出强度 SNR 对照未复现 trgc-01：{snr_repro} vs 登记值 {snr_reg}"

    qb = {
        "definition": "GC_fwd 净出度（offdiag 行和−列和，62 维）被什么单通道属性预测："
                      "Spearman(净出度, lag-1 ACF / 窗口 ACF / 幅值)，组级 + 逐例分布；"
                      "诊断性登记（预登记框架沿 trgc-01），不做因果表述",
        "caliber": {
            "net_out_degree": "行=源/列=靶、对角=0 ⇒ 净出度 = 全行和 − 全列和"
                              "（>0 净源倾向 / <0 净靶倾向）；组级取 refs-01 组均值矩阵，"
                              "逐例取逐例矩阵",
            "acf": f"z-score 后全记录逐通道 r_k=(1/T)Σx[t]x[t+k]（有偏分母；σ=1）；"
                   f"lag-1 = r_1（4 ms @250 Hz）、窗口 = mean(r_1..r_{C.P_MAX})"
                   f"（=192 ms，P_MAX 物理窗，design §2.2 同一滞后范围不变量）；"
                   "逐例算后取组均值",
            "amp": "z-score 前原尺度逐通道 std（ddof=0），48 例平均"
                   "（trgc-01 SNR 对照同式）",
            "acf_window_ms": [4.0, C.P_MAX * 1000.0 / C.FS],
        },
        "vectors_group": {
            "net_out_degree": {"min": float(netout_group.min()),
                               "median": float(np.median(netout_group)),
                               "max": float(netout_group.max())},
            "lag1_acf": {"min": float(acf1_g.min()), "median": float(np.median(acf1_g)),
                         "max": float(acf1_g.max())},
            "window_acf": {"min": float(wacf_g.min()), "median": float(np.median(wacf_g)),
                           "max": float(wacf_g.max())},
            "amp_std": {"min": float(cstd_g.min()), "median": float(np.median(cstd_g)),
                        "max": float(cstd_g.max())},
        },
        "correlations_group": {k: {"rho": float(v.statistic), "p": float(v.pvalue),
                                   "n_channels": C.N_CHANNELS}
                               for k, v in sp_g.items()},
        "correlations_per_subject": {k: _dist(v) for k, v in per.items()},
        "snr_control": {
            "outstr_vs_amp_group_rho": float(sp_g["outstr_vs_amp"].statistic),
            "trgc_outstr_vs_amp": {"reproduced_rho": snr_repro,
                                   "trgc01_registered_rho": snr_reg,
                                   "absdiff": abs(snr_repro - snr_reg),
                                   "assert": "abs(rho−登记值) ≤ 1e-9（同式复现断言）",
                                   "passed": True},
            "std_cross_validation": {
                "target": "trgc-01/data/edges_table.npz::chan_std",
                "passed": True, "max_absdiff": std_cv_diff,
                "note": "同式（原尺度 ddof=0，48 例平均）应逐位一致"},
        },
        "interpretation_frame": "净出度被通道属性预测的程度 = 诊断性登记：|ρ| 大 ⇒ "
                                "本口径 GC 出入不对称部分可由单通道统计（记忆/幅值）"
                                "解释，拓扑判读须打折；|ρ|≈0 ⇒ 净出度含通道属性之外的"
                                "结构。不做因果表述（沿 trgc-01 预登记框架）",
    }

    # ================= 产物落盘（npz + summary；无时间戳/计时字段）=================
    (run_dir / "data").mkdir(parents=True, exist_ok=True)
    npz_path = run_dir / "data" / "diag_table.npz"
    np.savez(npz_path,
             subject_ids=np.array(C.EXPECTED_IDS, dtype=np.int64),
             rho_fwd=rho_fwd, rho_rev=rho_rev, rho_trgcsh=rho_trgc,
             netout=netout, netout_group=netout_group,
             outstr=outstr, outstr_group=outstr_group,
             acf_lag1=acf1, acf_lag1_group=acf1_g,
             acf_win=wacf, acf_win_group=wacf_g,
             chan_std=cstd, chan_std_group=cstd_g)

    summary = {
        "deliverable": "Run diag-01 两问合一诊断（2026-10-18 用户追加）：问 A = TRGC "
                       "奇偶分半信度（fwd/rev 各自两半重拟合，逐例 4 次矩阵拟合）；"
                       "问 B = GC_fwd 净出度与单通道自相关/幅值的 Spearman（诊断性登记）",
        "n_subjects": len(sids), "subject_ids": sids, "groups": groups,
        "inputs": {
            "dataset": str(C.DATA_ROOT),
            "fwd_refs": {"run_id": "refs-01", "path": str(fwd_path)},
            "rev_refs": {"run_id": "rev-01", "path": str(rev_path)},
            "sh01_registered": {"run_id": "sh-01", "path": str(sh01_path),
                                "target": "ceiling.per_subject"},
            "trgc01": {"run_id": "trgc-01",
                       "targets": ["trgc_summary.json::snr_control.spearman_out.rho",
                                   "data/edges_table.npz::chan_std"]},
        },
        "caliber": {
            "question_a_split": "split_half.build_halves(x, seg=500)（奇偶分半/尾段丢弃/"
                                "段内设计不跨段；sh-01 逐字同式）+ half_ec(p_max=48)"
                                "（每半独立逐对 BIC）；rev 半侧 X_rev = X_z[:, ::-1]"
                                "（rev-01 同式）",
            "edge_flattening": "offdiag 行主序展平（62×62 对角剔除；行=源/列=靶；3782 边）",
            "question_b": "见 question_b.caliber",
            "interpretation_frame": "TRGC = 净时序方向读数（fwd−rev 差分）；信度/相关"
                                    "读数 ≠ 生理因果确证——双变量 GC 口径 + 头皮体积"
                                    "传导局限如实注明（沿 trgc-01 / rev-01 预登记框架）",
        },
        "question_a": qa,
        "question_b": qb,
        "outputs": {"summary": "diag_summary.json", "data": "data/diag_table.npz",
                    "figs": []},
        "figs_policy": {
            "decision": "json-only（不出图）",
            "rationale": "净出度 vs ACF 散点若出图须新增 visualizer 图型：edge_scatter "
                         "展示契约为非负轴域 [0,lim] + 恒等线（docstring 明示），净出度"
                         "为带符号读数（出−入可负），语义冲突；诊断登记数字全量落 "
                         "diag_summary.json + diag_table.npz，共享文件 "
                         "common/visualizer.py 保持零改动（白名单纪律裁量）",
        },
        "command": (f"python3 models/bivar-gc-ec-v2/diag_posthoc.py --run-id {args.run_id} "
                    f"--n-proc {args.n_proc}"),
        "entry": "models/bivar-gc-ec-v2/diag_posthoc.py",
        "determinism": "问 A 16 进程 spawn（worker 单线程 BLAS，refs-01-reprod 同款逐位"
                       "复现契约）+ 问 B 纯计算无 RNG；summary 无时间戳/计时字段；"
                       "np.savez 定长 zip 时间戳——产物确定性范围 = diag_summary.json + "
                       "data/diag_table.npz（logs/ 含时间戳不计入），重跑 sha256 逐位一致",
    }
    out_path = run_dir / "diag_summary.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=1))

    log(f"[diag] summary -> {out_path}")
    log(f"[diag] npz     -> {npz_path}")
    log(f"[diag A] TRGC 分半 = {rho_trgc.mean():.4f}±{rho_trgc.std(ddof=1):.4f}"
        f"（范围 {rho_trgc.min():.4f}–{rho_trgc.max():.4f}）| "
        f"fwd 分半 = {rho_fwd.mean():.4f}±{rho_fwd.std(ddof=1):.4f} | "
        f"rev 分半 = {rho_rev.mean():.4f}±{rho_rev.std(ddof=1):.4f}")
    d_l1, d_w1, d_a1, d_o1 = (_dist(per[k]) for k in
                              ("netout_vs_lag1acf", "netout_vs_windowacf",
                               "netout_vs_amp", "outstr_vs_amp"))
    log(f"[diag B] 组级 Spearman：净出度~lag1ACF ρ={sp_g['netout_vs_lag1acf'].statistic:+.4f} | "
        f"净出度~窗口ACF ρ={sp_g['netout_vs_windowacf'].statistic:+.4f} | "
        f"净出度~幅值 ρ={sp_g['netout_vs_amp'].statistic:+.4f} | "
        f"出强度~幅值（对照）ρ={sp_g['outstr_vs_amp'].statistic:+.4f}")
    log(f"[diag B] 逐例（mean±sd）：净出度~lag1ACF {d_l1['mean']:+.4f}±{d_l1['sd']:.4f} | "
        f"净出度~窗口ACF {d_w1['mean']:+.4f}±{d_w1['sd']:.4f} | "
        f"净出度~幅值 {d_a1['mean']:+.4f}±{d_a1['sd']:.4f} | "
        f"出强度~幅值 {d_o1['mean']:+.4f}±{d_o1['sd']:.4f}")
    log(f"[diag] SNR 对照复现：TRGC̄ 出强度~幅值 ρ={snr_repro:+.6f}"
        f"（trgc-01 登记 {snr_reg:+.6f}，|Δ|={abs(snr_repro - snr_reg):.2e}）| "
        f"通道 std 交叉验证 max|Δ|={std_cv_diff:.2e}")
    log(f"[diag] 完成（总耗时 {time.time() - t0:.0f}s）")


if __name__ == "__main__":
    main()
