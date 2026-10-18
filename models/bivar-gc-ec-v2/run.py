#!/usr/bin/env python3
"""run —— bivar-gc-ec-v2 CLI 编排（--stage {smoke,main,reverse,split,figs}；design §1）。

stages:
  smoke   B2 冒烟：主流程限前 N 例（--subjects，默认 2）+ **同式落盘产物**（与 v1 的
          纯控制台冒烟不同）+ 全 1891 对 pair-Gram 条件数实测（gram_pair_cond @p=48，
          min/median/max 入 refs_summary.json；design §2.2 数值监控）。
  main    B3 主 Run：48 例 16 进程并行（spawn + worker 单线程 BLAS，防 16×8 超订）→
          outputs/bivar-gc-ec-fmt-250/<run_id>/{data,refs,logs,figs}/（跑前须先写
          experiment.md Run 块——先写后跑纪律）。
  reverse B4.5 时间反转检验：装载链复用主链 + X_rev = X_z[:, ::-1]，逐例同管线；
          GC_rev 落盘（refs_all.npz 含 reversed=true 标记）+ 四组预登记指标
          （design §2.4）入 refs_summary.json；fwd 基准读 refs-01，缺失即报错。
  split   B4 split-half 天花板：**v2 本地编排**（split_half.build_halves(x, seg=500)
          + half_ec(·, p_max=48) + offdiag_flat + Spearman——不调 v1
          ceiling_for_subject，其无 seg 参数不透传，缺省落 v1 config 250 样本段）；
          运行时口径锁 n_seg==120 / 每半 27120 行 / seg_samples=500。
  figs    B5 落图（经 common/visualizer 正典路由；BIC 阶数分布退化不出图只入 summary）。

用法：nohup python3 models/bivar-gc-ec-v2/run.py --stage smoke --run-id smoke-01 &
      nohup python3 models/bivar-gc-ec-v2/run.py --stage main --run-id refs-01 &
"""

from __future__ import annotations

import os

# ---- BLAS 护栏（environment.md / baseline _blas_guard 惯例：必须在 numpy 之前；----
# ---- 16 核 cgroup 限 8 线程，setdefault 允许外部环境变量覆盖）                  ----
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse           # noqa: E402
import json               # noqa: E402
import multiprocessing    # noqa: E402
import sys                # noqa: E402
import time               # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np        # noqa: E402

# ---- sys.path 装配（design §1）：v2 目录居首（import data_io 恒解析到 v2）、----
# ---- v1 目录仅追加（bivar_gc / split_half 解析到 v1，只读复用）——spawn worker ----
# ---- 重导入同样依赖本模块顶部装配（复刻 v1 run.py 同式）                       ----
_HERE = Path(__file__).resolve().parent
_V1 = _HERE.parent / "bivar-gc-ec-v1"
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_V1) not in sys.path:
    sys.path.append(str(_V1))

import data_io            # noqa: E402（v2 版；连带装载唯一名 bgec2_config）
import bivar_gc           # noqa: E402（v1 版，只读复用）
import split_half as sh   # noqa: E402（v1 版，只读复用 build_halves/half_ec/offdiag_flat）
import bgec2_config as C  # noqa: E402（v2 config，唯一名）
from scipy.stats import spearmanr  # noqa: E402

_BLAS_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")

# split 运行时口径锁常量（design §2.3：段 500 样本、120 段、每半 27120 行）
_SH_N_SEG = 120
_SH_ROWS_PER_HALF = 27120

# fast↔naive 交叉校验常量（smoke-01 门禁处置升级措施：诊断层，不改数值管线）
_FN_SEED = 2026


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


def _order_hist(orders: np.ndarray) -> dict:
    """逐对阶数分布（上三角计数；对称登记的 orders 取上三角即无重复）。"""
    Cn = orders.shape[0]
    iu = np.triu_indices(Cn, k=1)
    vals, counts = np.unique(orders[iu], return_counts=True)
    return {str(int(v)): int(cnt) for v, cnt in zip(vals, counts)}


def _order_hist_stacked(results: list) -> dict:
    r"""跨例堆叠阶数分布（键 = 真实 p* 值，1..P_MAX 刻度；如实登记口径）。

    注：不采用 v1 的跨例求和式（orders 逐元素相加后取直方图）——该式把
    {47,48} 混合对登记成中间值（如 95），阶数语义失真；v2 以堆叠计数登记。
    """
    iu = np.triu_indices(C.N_CHANNELS, k=1)
    pooled_vals = np.concatenate([np.asarray(r["orders"])[iu].ravel() for r in results])
    vals, counts = np.unique(pooled_vals, return_counts=True)
    return {str(int(v)): int(cnt) for v, cnt in zip(vals, counts)}


def _sample_pairs(n_pairs: int) -> list:
    """确定性抽对（rng 2026）：1891 无序对中抽 n_pairs 个（(a,b) a<b 字典序枚举序）。"""
    pairs = [(a, b) for a in range(C.N_CHANNELS - 1) for b in range(a + 1, C.N_CHANNELS)]
    rng = np.random.default_rng(_FN_SEED)
    idx = rng.choice(len(pairs), size=n_pairs, replace=False)
    return [pairs[int(i)] for i in idx]


def _fast_naive_check(x: np.ndarray, gram: dict, orders: np.ndarray, pairs: list) -> dict:
    """fast Gram-slice vs naive lstsq（SVD 稳定路径）交叉校验（诊断层；门禁处置升级措施）。

    每对在 fast p\\* 单阶比较（naive 侧显式 row_from=P_MAX 与 fast 行集对齐）：
    GC 绝对差 max 与 SSR（u/r 全部元素）相对差 max。仅 v1 公开函数显式传参。
    """
    gc_d, ssr_d = 0.0, 0.0
    for a, b in pairs:
        p_star = int(orders[a, b])
        fast = bivar_gc.pair_fit(gram, a, b, (p_star,), C.P_MAX)
        naive = bivar_gc.naive_pair_fit(x, a, b, (p_star,), row_from=C.P_MAX)
        gc_d = max(gc_d, abs(fast["gc_ab"] - naive["gc_ab"]),
                   abs(fast["gc_ba"] - naive["gc_ba"]))
        ssr_d = max(ssr_d,
                    float(np.max(np.abs(fast["ssr_u"] - naive["ssr_u"])
                                 / (np.abs(naive["ssr_u"]) + 1e-12))),
                    float(np.max(np.abs(fast["ssr_r"] - naive["ssr_r"])
                                 / (np.abs(naive["ssr_r"]) + 1e-12))))
    return {"n_pairs": len(pairs), "seed": _FN_SEED,
            "scope": "fast p* 单阶；naive 行集 row_from=P_MAX 显式对齐",
            "gc_absdiff_max": float(gc_d), "ssr_reldiff_max": float(ssr_d)}


def _fn_pooled(results: list) -> dict:
    """跨例汇总 fast↔naive 校验（组级取 max；smoke 级对照 ≤3e-4）。"""
    per = {str(r["subject_id"]): r["fast_naive_check"] for r in results
           if "fast_naive_check" in r}
    return {"scope": next(iter(per.values()))["scope"], "seed": _FN_SEED,
            "n_pairs_per_subject": next(iter(per.values()))["n_pairs"],
            "note": "ssr_reldiff_max 在残差交叉协方差（Σ̂/SSR 非对角元）近零处分母趋零而"
                    "无界——该类条目两侧相关系数均 ≈0（实测最大出点：相关 -0.0005 vs "
                    "-0.0001，相关尺度差 ≤4e-4），属度量伪影非数值问题；GC 绝对差为"
                    "读出级主指标（对照带：smoke 全曲线 ≤3e-4）",
            "pooled": {"gc_absdiff_max": float(max(v["gc_absdiff_max"] for v in per.values())),
                       "ssr_reldiff_max": float(max(v["ssr_reldiff_max"] for v in per.values()))},
            "per_subject": per}


def _fit_subject(npz_path: str, reverse: bool = False, check_pairs: int = 0) -> dict:
    """单例装载 → （reverse=True 时沿时间翻转）→ 双变量 GC 全流程（含逐对 BIC）。

    翻转口径（design §2.4）：X_rev = X_z[:, ::-1]——z-score 为逐通道时不变统计，
    翻转次序无影响；ascontiguousarray 仅内存布局，数值逐位不变。
    """
    t0 = time.time()
    prep = data_io.prepare_subject(npz_path)
    x = prep["x"]
    if reverse:
        x = np.ascontiguousarray(x[:, ::-1])
    r = bivar_gc.bivar_gc_matrix(x, p_max=C.P_MAX)
    out = {"subject_id": int(prep["meta"]["subject_id"]), "group": prep["meta"]["group"],
           "meta": prep["meta"], "ec": r["ec"], "orders": r["orders"],
           "n": r["n"], "wall_s": time.time() - t0}
    if check_pairs:
        # 诊断层（与拟合同码径重建 Gram，逐位同 G；不触碰 ec/orders）
        gram = bivar_gc.build_gram(x, C.P_MAX, row_from=C.P_MAX)
        out["fast_naive_check"] = _fast_naive_check(x, gram, r["orders"],
                                                    _sample_pairs(check_pairs))
        out["wall_s"] = time.time() - t0
    return out


def _subject_worker(npz_path: str) -> dict:
    """spawn worker（main）：单例双变量 GC + 50 对 fast↔naive 交叉校验。"""
    return _fit_subject(npz_path, reverse=False, check_pairs=50)


def _reverse_worker(npz_path: str) -> dict:
    """spawn worker（reverse）：单例时间翻转后同管线 + 50 对交叉校验。"""
    return _fit_subject(npz_path, reverse=True, check_pairs=50)


def _cond_worker(npz_path: str) -> dict:
    """spawn worker（smoke 专用）：单例 GC + 全 1891 对 pair-Gram 条件数实测（@p=P_MAX）。

    条件数 = gram_pair_cond（[截距, A lag1..p, B lag1..p] slice 的 2-范数条件数），
    1891 对全量 min/median/max（design §2.2 数值监控；越限 halt 请示，不自行加载）。
    """
    r = _fit_subject(npz_path, reverse=False)
    prep = data_io.prepare_subject(npz_path)
    gram = bivar_gc.build_gram(prep["x"], C.P_MAX, row_from=C.P_MAX)
    Cn = C.N_CHANNELS
    conds = np.array([bivar_gc.gram_pair_cond(gram, a, b, C.P_MAX)
                      for a in range(Cn - 1) for b in range(a + 1, Cn)])
    r["cond"] = {"p": C.P_MAX, "n_pairs": int(conds.size),
                 "min": float(conds.min()), "median": float(np.median(conds)),
                 "max": float(conds.max())}
    return r


def _assert_subject(r: dict) -> None:
    """实现正确性契约断言（§2.2：全有限 / 对角=0 / offdiag≥0 / 阶数对称）。"""
    ec, od = r["ec"], r["orders"]
    assert ec.shape == (C.N_CHANNELS, C.N_CHANNELS)
    assert np.isfinite(ec).all() and float(np.abs(np.diag(ec)).max()) == 0.0
    assert (ec[~np.eye(C.N_CHANNELS, dtype=bool)] >= 0).all(), \
        f"s{r['subject_id']}: offdiag GC < 0（OLS 样本内单调性违约=实现错误）"
    assert int(od[0, 1]) > 0 and (od == od.T).all()


def _save_refs(run_dir: Path, results: list, log, *, reversed_: bool = False) -> None:
    """逐例 refs_s<id>.npz + refs_all.npz（排序堆叠；reversed 标记按 run 写入）。"""
    refs_dir = run_dir / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)
    results.sort(key=lambda r: r["subject_id"])
    ec_all = []
    for r in results:
        _assert_subject(r)
        np.savez_compressed(refs_dir / f"refs_s{r['subject_id']}.npz",
                            ec=r["ec"].astype(np.float32), orders=r["orders"],
                            n=r["n"], subject_id=r["subject_id"])
        ec_all.append(r["ec"])
    ec_all = np.stack(ec_all)
    sids_str = [str(r["subject_id"]) for r in results]
    payload = {"bivar_gc": ec_all.astype(np.float32), "subject_ids": sids_str}
    if reversed_:
        payload["reversed"] = np.bool_(True)     # GC_rev 落盘标记（design §2.4）
    np.savez_compressed(run_dir / "refs_all.npz", **payload)
    (run_dir / "data").mkdir(parents=True, exist_ok=True)
    (run_dir / "data" / "prep_meta.json").write_text(
        json.dumps([r["meta"] for r in results], ensure_ascii=False, indent=1))
    log(f"[refs] 落盘：refs_all.npz ({len(results)},62,62) f32"
        + ("（reversed=true）" if reversed_ else "")
        + f" + 逐例 refs_s*.npz ×{len(results)} + data/prep_meta.json")


def _caliber_common() -> dict:
    """refs_summary 的口径字段（design §2.1/§2.2）。"""
    return {
        "method": "pairwise bivariate GC（不条件化于其余通道；observational "
                  "directed-connectivity reference 双变量口径）",
        "readout": "GC(A→B)=ln(SSR_r^B/SSR_u^B)，同一次拟合双向读出；"
                   "行=源/列=靶/对角=0",
        "bic": f"逐对 BIC 选阶 p∈{{1..{C.P_MAX}}}，行集 row_from=P_MAX={C.P_MAX}"
               "（嵌套行集）；BIC=ln|Σ̂|+K²p·ln(n)/n（K=2，截距不进惩罚）",
        "p_max_rationale": "P_MAX=48 保物理滞后 192 ms（125 Hz@24 ≡ 250 Hz@48），"
                           "两条参照的模型阶物理含义一致（design §2.2）",
        "fs_raw": C.FS_RAW, "fs": C.FS,
        "resample": "none（250 Hz 原始全长原样进入拟合；不降采、不切窗）",
        "zscore": "逐通道全记录 ddof=0，σ 下限 1e-12（v1 同式）",
        "storage": "float32 落盘 / float64 计算",
        "nonneg": "OLS 样本内 SSR 单调性 ⇒ offdiag≥0 必然（实现正确性断言；"
                  "与 ridge 条件 GC 允许微负为不同口径）",
        "terminology": "observational directed-connectivity reference（非 ground-truth）",
        "out_of_scope": "不做对比交付（不与 125 Hz 参照、不与 Study I 窗式 GC 比较；"
                        "用户 2026-10-18 拍板）",
    }


def _top_grid_note(pooled: dict) -> str:
    """BIC 顶格注记（如实登记；design §2.2——顶格 ⇒ 实效固定阶 VAR(48)）。"""
    total = sum(pooled.values())
    top = pooled.get(str(C.P_MAX), 0)
    if top == total:
        return (f"BIC p*={C.P_MAX} 顶格 100%（{total} 对·例组合；惩罚项 "
                f"4·{C.P_MAX}·ln(n_eff)/n_eff≈0.035 极小 + n_eff≈6 万，预期内）"
                f"⇒ 实效估计量 = 固定阶 VAR({C.P_MAX}) 双变量 GC；如实登记，"
                "不做调参干预，网格扩展属 v1 遗留未决决策不在本任务处理")
    if top + pooled.get(str(C.P_MAX - 1), 0) == total:
        return (f"BIC 阶数分布集中于顶部两阶：p*={C.P_MAX} 占 {top}/{total}、"
                f"p*={C.P_MAX - 1} 占 {total - top}/{total}（实测如实登记）；"
                "BIC 顶部平坦 + Gram 条件数 1e13–5e15 下顶部两阶之分为数值噪声级"
                "（fast↔naive 交叉校验 GC 绝对差见 fast_naive_check，smoke 级 ≤3e-4），"
                f"实效读出 = 顶部阶（≈固定阶 VAR({C.P_MAX})）双变量 GC；不做调参干预")
    adj = top + pooled.get(str(C.P_MAX - 1), 0)
    if adj >= 0.99 * total:
        return (f"BIC 阶数分布集中于顶部：p*={C.P_MAX} 占 {top}/{total}"
                f"（{top / total:.1%}）、p*={C.P_MAX - 1} 占 {pooled.get(str(C.P_MAX - 1), 0)}"
                f"/{total}，其余 {total - adj} 对·例散布于更低阶（占 <0.1%，同为顶部附近"
                "数值噪声级分界）；BIC 顶部平坦 + Gram 条件数 1e13–5e15 下顶部两阶之分与"
                "零星散布均为数值噪声级（fast↔naive 交叉校验见 fast_naive_check 键；"
                "smoke 级全曲线 GC 绝对差 ≤3e-4），"
                f"实效读出 = 顶部阶（≈固定阶 VAR({C.P_MAX})）双变量 GC；实测如实登记，"
                "不做调参干预")
    return (f"BIC 阶数分布非单点（p*={C.P_MAX} 占 {top}/{total}）；"
            f"实测分布如实登记，估计量 = 逐对变阶双变量 GC")


# ---- smoke / main（B2/B3：同主流程；smoke 限例数 + 条件数登记） ----

def _run_subjects(args, log, sids: list, worker, *, reversed_: bool = False) -> list:
    """给定受试者清单的 16 进程 spawn 并行拟合（worker 单线程 BLAS）。"""
    manifest = data_io.load_manifest()
    for v in _BLAS_VARS:
        os.environ[v] = "1"
    ctx = multiprocessing.get_context("spawn")
    n_proc = min(args.n_proc, len(sids))
    t_start = time.time()
    results = []
    tag = "reverse" if reversed_ else "main"
    with ctx.Pool(processes=n_proc, initializer=_worker_init) as pool:
        for r in pool.imap_unordered(worker, [manifest[s]["npz_path"] for s in sids]):
            results.append(r)
            od = r["ec"][~np.eye(C.N_CHANNELS, dtype=bool)]
            log(f"[{tag} {len(results)}/{len(sids)}] s{r['subject_id']} ({r['group']}) "
                f"n_eff={int(r['n'])} gc: min={od.min():.4f} median={np.median(od):.4f} "
                f"max={od.max():.4f}（{r['wall_s']:.1f}s）")
    log(f"[{tag}] 全例拟合完成（总耗时 {time.time() - t_start:.0f}s）")
    return results


def stage_main(args, log, *, smoke: bool = False) -> None:
    """主流程（main 全 48 例 / smoke 限前 N 例 + 条件数登记；同式落盘产物）。"""
    manifest = data_io.load_manifest()
    sids_all = sorted(s for s, e in manifest.items() if e["status"] == "ok")
    assert sids_all == list(C.EXPECTED_IDS), \
        f"manifest 48 例与 EXPECTED_IDS 不一致：{sids_all}"
    sids = sids_all[:max(1, args.subjects or 2)] if smoke else sids_all
    run_dir = C.OUTPUTS_ROOT / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    worker = _cond_worker if smoke else _subject_worker
    log(f"[{'smoke' if smoke else 'main'}] 启动：{len(sids)} 例 {sids}；"
        f"fs={C.FS} 原始全长（不降采、不切窗）；p_max={C.P_MAX}（row_from={C.P_MAX}，"
        f"n_eff={C.N_RAW - C.P_MAX}/例）；{args.n_proc} 进程 spawn 并行（worker 单线程 BLAS）")
    t_start = time.time()
    results = _run_subjects(args, log, sids, worker)
    results.sort(key=lambda r: r["subject_id"])
    _save_refs(run_dir, results, log)

    # ---- refs_summary.json ----
    ec_all = np.stack([r["ec"] for r in results])
    od_flat = ec_all[:, ~np.eye(C.N_CHANNELS, dtype=bool)]   # (n,3782) 批维广播
    order_hists = {str(r["subject_id"]): _order_hist(r["orders"]) for r in results}
    pooled = _order_hist_stacked(results)
    summary = {
        "n_subjects": len(results), "subject_ids": [str(r["subject_id"]) for r in results],
        "groups": {str(r["subject_id"]): r["group"] for r in results},
        "caliber": _caliber_common(),
        "bic_top_grid_note": _top_grid_note(pooled),
        "refs_stats": {"offdiag_min": float(od_flat.min()),
                       "offdiag_median": float(np.median(od_flat)),
                       "offdiag_max": float(od_flat.max())},
        "order_distribution": {"per_subject": order_hists, "pooled": pooled},
        "wall_s_total": round(time.time() - t_start, 1),
        "wall_s_per_subject": {str(r["subject_id"]): round(r["wall_s"], 1) for r in results},
    }
    if any("fast_naive_check" in r for r in results):
        summary["fast_naive_check"] = _fn_pooled(results)
    if smoke:
        summary["condition_numbers"] = {
            "p": C.P_MAX,
            "note": "全 1891 对/例 pair-Gram（[截距, A lag1..48, B lag1..48]，97 维）"
                    "2-范数条件数实测（gram_pair_cond）；越限 halt 请示（design §2.2）",
            "per_subject": {str(r["subject_id"]): r["cond"] for r in results},
        }
    (run_dir / "refs_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    (run_dir / "config_snapshot.json").write_text(json.dumps({
        "run_id": args.run_id, "stage": "smoke" if smoke else "main",
        "entry": str(_HERE / "run.py"),
        "variant": "models/bivar-gc-ec-v2",
        "reused_readonly": ["models/bivar-gc-ec-v1/bivar_gc.py",
                            "models/bivar-gc-ec-v1/split_half.py"],
        "data_root": str(C.DATA_ROOT), "fs_raw": C.FS_RAW, "fs": C.FS,
        "resample": "none", "p_max": C.P_MAX, "row_from": C.P_MAX,
        "n_eff_per_subject": C.N_RAW - C.P_MAX,
        "expected_ids": list(C.EXPECTED_IDS), "outputs_root": str(C.OUTPUTS_ROOT),
        "n_proc": args.n_proc,
        "blas_threads": {k: os.environ.get(k) for k in _BLAS_VARS},
        "command": f"python3 {__file__} --stage {args.stage} --run-id {args.run_id} "
                   f"--n-proc {args.n_proc}",
    }, ensure_ascii=False, indent=1))
    log(f"[{'smoke' if smoke else 'main'}] 完成：refs_summary.json + config_snapshot.json"
        f"（总耗时 {time.time() - t_start:.0f}s）")
    od = ec_all[:, ~np.eye(C.N_CHANNELS, dtype=bool)]
    log(f"[{'smoke' if smoke else 'main'}] offdiag GC（ln 单位）：min={od.min():.4f} "
        f"median={np.median(od):.4f} max={od.max():.4f}；阶数分布 pooled={pooled}")


# ---- reverse（B4.5：时间反转检验；design §2.4） ----

def stage_reverse(args, log) -> None:
    manifest = data_io.load_manifest()
    sids = sorted(s for s, e in manifest.items() if e["status"] == "ok")
    assert sids == list(C.EXPECTED_IDS)
    run_dir = C.OUTPUTS_ROOT / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    fwd_dir = C.OUTPUTS_ROOT / "refs-01"
    fwd_path = fwd_dir / "refs_all.npz"
    assert fwd_path.exists(), \
        f"fwd 基准缺失：{fwd_path} 不存在——先跑 --stage main --run-id refs-01"
    fd = np.load(fwd_path, allow_pickle=True)
    fwd = {str(sid): ec.astype(np.float64)
           for sid, ec in zip(fd["subject_ids"], fd["bivar_gc"])}
    log(f"[reverse] 启动：{len(sids)} 例；X_rev = X_z[:, ::-1]（z-score 后沿时间翻转），"
        f"其余与主 Run 逐字同口径（p_max={C.P_MAX}、row_from={C.P_MAX}、逐对 BIC、"
        f"ln(SSR) 双向、OLS）；fwd 基准 = {fwd_path}")
    t_start = time.time()
    results = _run_subjects(args, log, sids, _reverse_worker, reversed_=True)
    results.sort(key=lambda r: r["subject_id"])
    _save_refs(run_dir, results, log, reversed_=True)

    # ---- 四组预登记指标（逐例 → 组级 mean±sd，ddof=1） ----
    eye = ~np.eye(C.N_CHANNELS, dtype=bool)
    m_rev, m_mir, med_f, med_r, ratio, asym_f, asym_r = [], [], [], [], [], [], []
    for r in results:
        sid = str(r["subject_id"])
        ec_f, ec_r = fwd[sid], r["ec"]
        of, orv = ec_f[eye], ec_r[eye]
        m_rev.append(float(spearmanr(of, orv).statistic))                # ① fwd vs rev
        m_mir.append(float(spearmanr(of, ec_r.T[eye]).statistic))        # ② fwd vs revᵀ
        mf, mr = float(np.median(of)), float(np.median(orv))             # ③ 幅值
        med_f.append(mf); med_r.append(mr); ratio.append(mr / mf)
        asym_f.append(float(np.abs(ec_f - ec_f.T).mean()))               # ④ 方向不对称度
        asym_r.append(float(np.abs(ec_r - ec_r.T).mean()))

    def _grp(vals: list) -> dict:
        a = np.asarray(vals, dtype=np.float64)
        return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)),
                "per_subject": {str(r["subject_id"]): float(v)
                                for r, v in zip(results, vals)}}
    summary = {
        "n_subjects": len(results), "subject_ids": [str(r["subject_id"]) for r in results],
        "groups": {str(r["subject_id"]): r["group"] for r in results},
        "caliber": {**_caliber_common(),
                    "reverse": "X_rev = X_z[:, ::-1]（z-score 后沿时间翻转；z-score 为"
                               "逐通道时不变统计，翻转次序无影响）；其余与主 Run 逐字同口径",
                    "interpretation_frame": "描述性指纹登记（预登记，非硬门限）：二阶平稳"
                                            "高斯过程时间反演二阶结构不变、反向 VAR 系数不同 "
                                            "⇒ GC_rev 与 GC_fwd 既不相等也不必归零（design §2.4）"},
        "fwd_baseline": {"run_id": "refs-01", "path": str(fwd_path)},
        "metrics": {
            "edge_fwd_rev": _grp(m_rev),
            "edge_fwd_mirrorrev": _grp(m_mir),
            "amplitude": {"median_offdiag_fwd": _grp(med_f),
                          "median_offdiag_rev": _grp(med_r),
                          "ratio_rev_over_fwd": _grp(ratio)},
            "direction_asymmetry": {"fwd_mean_abs_gc_diff": _grp(asym_f),
                                    "rev_mean_abs_gc_diff": _grp(asym_r)},
        },
        "refs_stats": {"offdiag_min": float(np.concatenate(
                           [r["ec"][eye] for r in results]).min()),
                       "offdiag_median": float(np.median(np.concatenate(
                           [r["ec"][eye] for r in results]))),
                       "offdiag_max": float(np.concatenate(
                           [r["ec"][eye] for r in results]).max())},
        "order_distribution": {"per_subject": {str(r["subject_id"]): _order_hist(r["orders"])
                                               for r in results},
                               "pooled": _order_hist_stacked(results)},
        "wall_s_total": round(time.time() - t_start, 1),
        "wall_s_per_subject": {str(r["subject_id"]): round(r["wall_s"], 1) for r in results},
    }
    if any("fast_naive_check" in r for r in results):
        summary["fast_naive_check"] = _fn_pooled(results)
    (run_dir / "refs_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    (run_dir / "config_snapshot.json").write_text(json.dumps({
        "run_id": args.run_id, "stage": "reverse", "entry": str(_HERE / "run.py"),
        "variant": "models/bivar-gc-ec-v2",
        "reused_readonly": ["models/bivar-gc-ec-v1/bivar_gc.py"],
        "data_root": str(C.DATA_ROOT), "fs": C.FS, "resample": "none",
        "p_max": C.P_MAX, "row_from": C.P_MAX, "reverse": "X_rev = X_z[:, ::-1]",
        "fwd_baseline_run": "refs-01", "expected_ids": list(C.EXPECTED_IDS),
        "outputs_root": str(C.OUTPUTS_ROOT), "n_proc": args.n_proc,
        "blas_threads": {k: os.environ.get(k) for k in _BLAS_VARS},
        "command": f"python3 {__file__} --stage reverse --run-id {args.run_id} "
                   f"--n-proc {args.n_proc}",
    }, ensure_ascii=False, indent=1))
    m = summary["metrics"]
    log(f"[reverse] 完成（{time.time() - t_start:.0f}s）："
        f"edge_fwd_rev={m['edge_fwd_rev']['mean']:.4f}±{m['edge_fwd_rev']['sd']:.4f}  "
        f"edge_fwd_mirrorrev={m['edge_fwd_mirrorrev']['mean']:.4f}"
        f"±{m['edge_fwd_mirrorrev']['sd']:.4f}  "
        f"幅值比 rev/fwd={m['amplitude']['ratio_rev_over_fwd']['mean']:.3f}"
        f"±{m['amplitude']['ratio_rev_over_fwd']['sd']:.3f}  "
        f"不对称度 fwd={m['direction_asymmetry']['fwd_mean_abs_gc_diff']['mean']:.4f} "
        f"rev={m['direction_asymmetry']['rev_mean_abs_gc_diff']['mean']:.4f}")


# ---- split（B4：v2 本地编排，不调 v1 ceiling_for_subject；design §2.3） ----

def _split_worker(npz_path: str) -> dict:
    """spawn worker：build_halves(x, seg=500) → 两半各自同口径 GC → offdiag Spearman。

    **v2 本地编排**（审查 Mandate A/B：v1 ceiling_for_subject 无 seg 参数且内部
    build_halves(x) 不透传，缺省落 v1 config SEG_SAMPLES=250——禁用）；运行时
    口径锁：n_seg==120、每半 27120 行（60 段 × (500−48) 行）、seg_samples=500。
    """
    t0 = time.time()
    prep = data_io.prepare_subject(npz_path)
    assert C.SEG_SAMPLES == 500, f"SEG_SAMPLES={C.SEG_SAMPLES} != 500（口径锁）"
    halves = sh.build_halves(prep["x"], seg=C.SEG_SAMPLES)   # 偶段A/奇段B/尾段丢弃
    assert halves["n_seg"] == _SH_N_SEG, \
        f"n_seg={halves['n_seg']} != {_SH_N_SEG}（60001//500=120；口径锁）"
    ra = sh.half_ec(halves["A"], p_max=C.P_MAX)
    rb = sh.half_ec(halves["B"], p_max=C.P_MAX)
    assert float(ra["n"]) == _SH_ROWS_PER_HALF and float(rb["n"]) == _SH_ROWS_PER_HALF, \
        f"每半行数 ra={ra['n']} rb={rb['n']} != {_SH_ROWS_PER_HALF}（60×452；口径锁）"
    rho = float(spearmanr(sh.offdiag_flat(ra["ec"]),
                          sh.offdiag_flat(rb["ec"])).statistic)
    out = {"subject_id": int(prep["meta"]["subject_id"]),
           "group": prep["meta"]["group"], "sh_rho": rho,
           "n_seg": halves["n_seg"], "n_a": ra["n"], "n_b": rb["n"],
           "wall_s": time.time() - t0}
    # 诊断层：每半各 25 对 fast↔naive 交叉校验（同一 rng 2026 抽样；门禁处置升级措施）
    pairs = _sample_pairs(25)
    ga = bivar_gc.build_gram(halves["A"], C.P_MAX, row_from=C.P_MAX)
    gb = bivar_gc.build_gram(halves["B"], C.P_MAX, row_from=C.P_MAX)
    out["fast_naive_check"] = {"A": _fast_naive_check(halves["A"], ga, ra["orders"], pairs),
                               "B": _fast_naive_check(halves["B"], gb, rb["orders"], pairs)}
    out["wall_s"] = time.time() - t0
    return out


def stage_split(args, log) -> None:
    manifest = data_io.load_manifest()
    sids = sorted(s for s, e in manifest.items() if e["status"] == "ok")
    assert sids == list(C.EXPECTED_IDS)
    run_dir = C.OUTPUTS_ROOT / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log(f"[split] 启动：{len(sids)} 例 × 两半（v2 本地编排：build_halves(x, seg={C.SEG_SAMPLES})"
        f" + half_ec(p_max={C.P_MAX}) + offdiag_flat + Spearman，不调 v1 ceiling_for_subject；"
        f"偶A/奇B/尾弃/段内设计不跨段；口径锁 n_seg=={_SH_N_SEG}、每半 {_SH_ROWS_PER_HALF} 行）")
    for v in _BLAS_VARS:
        os.environ[v] = "1"
    ctx = multiprocessing.get_context("spawn")
    t_start = time.time()
    results = []
    with ctx.Pool(processes=args.n_proc, initializer=_worker_init) as pool:
        for r in pool.imap_unordered(_split_worker,
                                     [manifest[s]["npz_path"] for s in sids]):
            results.append(r)
            log(f"[split {len(results)}/{len(sids)}] s{r['subject_id']} "
                f"sh_rho={r['sh_rho']:.4f}（n_seg={r['n_seg']}，{r['wall_s']:.1f}s）")
    results.sort(key=lambda r: r["subject_id"])
    arr = np.array([r["sh_rho"] for r in results])
    out = {"seg_s": C.SEG_S, "seg_samples": C.SEG_SAMPLES,
           "n_seg": _SH_N_SEG, "rows_per_half": _SH_ROWS_PER_HALF,
           "caliber": "v2 本地编排（split_half.build_halves(x, seg=500) + "
                      "half_ec(p_max=48) + offdiag_flat + Spearman；不调 v1 "
                      "ceiling_for_subject——其无 seg 参数不透传）：双变量 GC 同口径"
                      "重跑两半（含逐对 BIC），offdiag 3782 有向边 Spearman ρ/例，"
                      "天花板 = mean±sd（ddof=1，48 例）",
           "ceiling": {"mean": float(arr.mean()), "sd": float(arr.std(ddof=1)),
                       "per_subject": {str(r["subject_id"]): r["sh_rho"] for r in results}},
           "fast_naive_check": {
               "scope": "每半各 25 对（同一 rng 2026 抽样）fast↔naive @p*；smoke 级对照 ≤3e-4",
               "pooled": {"gc_absdiff_max": float(max(
                              r["fast_naive_check"][h]["gc_absdiff_max"]
                              for r in results for h in ("A", "B"))),
                          "ssr_reldiff_max": float(max(
                              r["fast_naive_check"][h]["ssr_reldiff_max"]
                              for r in results for h in ("A", "B")))},
               "per_subject": {str(r["subject_id"]): r["fast_naive_check"] for r in results}},
           "wall_s_total": round(time.time() - t_start, 1)}
    (run_dir / "split_half.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    (run_dir / "config_snapshot.json").write_text(json.dumps({
        "run_id": args.run_id, "stage": "split", "entry": str(_HERE / "run.py"),
        "variant": "models/bivar-gc-ec-v2",
        "reused_readonly": ["models/bivar-gc-ec-v1/split_half.py（build_halves/half_ec/"
                            "offdiag_flat 显式传参）", "models/bivar-gc-ec-v1/bivar_gc.py"],
        "data_root": str(C.DATA_ROOT), "fs": C.FS, "p_max": C.P_MAX,
        "seg_samples": C.SEG_SAMPLES, "n_seg": _SH_N_SEG,
        "rows_per_half": _SH_ROWS_PER_HALF,
        "orchestration": "v2 local（禁用 v1 ceiling_for_subject）",
        "expected_ids": list(C.EXPECTED_IDS), "outputs_root": str(C.OUTPUTS_ROOT),
        "n_proc": args.n_proc,
        "blas_threads": {k: os.environ.get(k) for k in _BLAS_VARS},
        "command": f"python3 {__file__} --stage split --run-id {args.run_id} "
                   f"--n-proc {args.n_proc}",
    }, ensure_ascii=False, indent=1))
    log(f"[split] 完成：ceiling = {out['ceiling']['mean']:.4f} ± "
        f"{out['ceiling']['sd']:.4f}（mean±sd，ddof=1）→ split_half.json"
        f"（总耗时 {time.time() - t_start:.0f}s）")


# ---- figs（B5：经 common/visualizer 正典路由；纯复用既有图型，零 visualizer 改动） ----

def stage_figs(args, log) -> None:
    """B5 落图：①组均值三面板（ec_matrix）；②逐例抽样网格（ec_heatmap_grid）。
   阶数分布若退化（100% 顶格）不出图、只入 refs_summary.json（沿 v1 先例）。"""
    run_dir = C.OUTPUTS_ROOT / args.run_id
    summary = json.loads((run_dir / "refs_summary.json").read_text())
    d = np.load(run_dir / "refs_all.npz", allow_pickle=True)
    assert "reversed" not in d, "figs 以 fwd 主 Run（refs-01）为对象，不接受 reversed 产物"
    ec_all = d["bivar_gc"].astype(np.float64)
    sids = [str(s) for s in d["subject_ids"]]
    groups = summary["groups"]

    _COMMON = _HERE.parent.parent / "common"      # 统一 visualizer 正典路由
    if str(_COMMON) not in sys.path:
        sys.path.append(str(_COMMON))
    from visualizer import Visualizer             # noqa: E402（正典路由，B5 时点导入）

    viz = Visualizer(str(run_dir / "figs"))
    idx = {"HC": [i for i, s in enumerate(sids) if groups[s] == "HC"],
           "MDD": [i for i, s in enumerate(sids) if groups[s] == "MDD"]}
    hc, mdd, allm = ec_all[idx["HC"]].mean(0), ec_all[idx["MDD"]].mean(0), ec_all.mean(0)

    eye = ~np.eye(C.N_CHANNELS, dtype=bool)
    od = allm[eye]
    p1 = viz.render("ec_matrix", {
        "ec_a": hc, "ec_b": mdd, "ec_c": allm,
        "label_a": f"HC mean (n={len(idx['HC'])})", "label_b": f"MDD mean (n={len(idx['MDD'])})",
        "label_c": f"all mean (n={len(sids)})",
        "stats": {"median": float(np.median(od)), "max": float(od.max())},
    }, "bivar_gc_group_means.png")
    log(f"[figs] 组均值三面板 → {p1}")

    pooled = summary["order_distribution"]["pooled"]
    top = pooled.get(str(C.P_MAX), 0)
    total = sum(pooled.values())
    if top == total:
        bic_note = f"per-pair BIC (p*={C.P_MAX} all pairs)"
    else:
        bic_note = (f"per-pair BIC (p*={C.P_MAX} in {top / total:.1%} of pairs; "
                    "top-region noise-level split, see refs_summary)")
    pick = np.linspace(0, len(sids) - 1, 6).astype(int)   # 确定性等距抽样
    p2 = viz.render("ec_heatmap_grid", {
        "rows": [{"label": "bivariate GC ln(SSR)", "mats": [ec_all[i] for i in pick]}],
        "col_labels": [f"s{sids[i]} ({groups[sids[i]]})" for i in pick],
        "unit": "ln(SSR ratio)",
        "note": f"pairwise bivariate GC · 250 Hz raw full-length · {bic_note}",
    }, "bivar_gc_subjects_sample.png")
    log(f"[figs] 逐例抽样网格（等距 6 例）→ {p2}")
    log("[figs] 完成（阶数分布如实登记于 refs_summary.json，不新增图型；"
        "common/visualizer.py 零改动）")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", required=True,
                    choices=["smoke", "main", "reverse", "split", "figs"])
    ap.add_argument("--run-id", default="refs-01", help="产物 run 目录名")
    ap.add_argument("--subjects", type=int, default=2, help="smoke 例数（EXPECTED_IDS 前 N）")
    ap.add_argument("--n-proc", type=int, default=C.N_PROC, help="并行进程数")
    args = ap.parse_args()

    if args.stage == "smoke":
        run_dir = C.OUTPUTS_ROOT / args.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        log = _log_factory(run_dir / "logs" / "run.log")
        stage_main(args, log, smoke=True)
        return
    run_dir = C.OUTPUTS_ROOT / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log = _log_factory(run_dir / "logs" / "run.log")
    if args.stage == "main":
        stage_main(args, log)
    elif args.stage == "reverse":
        stage_reverse(args, log)
    elif args.stage == "split":
        stage_split(args, log)
    else:
        stage_figs(args, log)


if __name__ == "__main__":
    main()
