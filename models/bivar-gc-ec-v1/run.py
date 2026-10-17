#!/usr/bin/env python3
"""run —— bivar-gc-ec-v1 CLI 编排（--stage {smoke,main,split,figs}；design §1）。

stages:
  smoke   B0 装载冒烟：manifest 升序前 N 例（--subjects，默认 1）控制台断言输出
          （字段断言 / 形状 (62,30001) / z-score 复核 / 8ch 子集 pair-Gram 条件数
          预演）——**不落盘、不建 run_id**。
  main    B2 主 Run：48 例 16 进程并行（spawn + worker 单线程 BLAS，防 16×8 超订）
          → outputs/bivar-gc-ec-fmt/<run_id>/{data,refs,logs,figs}/（跑前须先写
          experiment.md Run 块——先写后跑纪律）。
  split   B3 split-half 噪声天花板（接线；每半同口径含逐对 BIC → offdiag Spearman）。
  figs    B4 落图（经 common/visualizer 正典路由；B4 阶段实现）。

用法：python3 models/bivar-gc-ec-v1/run.py --stage smoke --subjects 1
      nohup python3 models/bivar-gc-ec-v1/run.py --stage main --run-id refs-01 &
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

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import data_io            # noqa: E402（连带装载唯一名 bgec_config）
import bivar_gc           # noqa: E402
import split_half as sh   # noqa: E402
import bgec_config as C   # noqa: E402

_BLAS_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


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


# ---- smoke（B0：控制台断言输出，不落盘不建 run_id） ----

def stage_smoke(args) -> None:
    manifest = data_io.load_manifest()
    sids = sorted(manifest)[: max(1, args.subjects or 1)]
    for sid in sids:
        e = manifest[sid]
        assert e["status"] == "ok", f"s{sid} status={e['status']} 非 ok"
        t0 = time.time()
        prep = data_io.prepare_subject(e["npz_path"])
        x, m = prep["x"], prep["meta"]
        assert x.shape == (C.N_CHANNELS, C.N_125), f"s{sid}: 降采后形状 {x.shape} 异常"
        mu_dev = float(np.abs(x.mean(axis=-1)).max())
        sd_dev = float(np.abs(x.std(axis=-1) - 1.0).max())
        assert mu_dev < 1e-9 and sd_dev < 1e-9, \
            f"s{sid}: z-score 复核失败 |mean|max={mu_dev:.3e} |σ-1|max={sd_dev:.3e}"
        print(f"[smoke] s{sid} ({e['group']}) 字段断言通过：pipeline={m['pipeline']!r} "
              f"filter={m['filter']!r} fs {m['fs_raw']}→{m['fs']} "
              f"(resample up/down={C.RESAMPLE_UP}/{C.RESAMPLE_DOWN})"
              f" n_raw={m['n_raw']}")
        print(f"[smoke] s{sid} 形状 {x.shape}（62,30001）✓  "
              f"z-score 复核：max|mean|={mu_dev:.3e}  max|σ−1|={sd_dev:.3e}")
        gram = bivar_gc.build_gram(x, C.P_MAX)
        idx8 = np.linspace(0, C.N_CHANNELS - 1, 8).astype(int)
        cs = bivar_gc.subset_cond_stats(gram, idx8, C.P_MAX)
        print(f"[smoke] s{sid} Gram@p_max={C.P_MAX}: n_eff={int(gram['n'])}  "
              f"8ch 子集 pair-Gram 条件数（{cs['n_pairs']} 对，通道 {idx8.tolist()}）："
              f"min={cs['min']:.2e} median={cs['median']:.2e} max={cs['max']:.2e}"
              f"（B2 条件数监控预演）")
        print(f"[smoke] s{sid} 装载+Gram 用时 {time.time() - t0:.1f}s（控制台冒烟，不落盘）")
    print(f"[smoke] B0 装载冒烟通过：{len(sids)} 例（manifest 升序首 {sids}），断言全过")


# ---- main（B2：48 例 16 进程并行；spawn + worker 单线程 BLAS） ----

def _worker_init() -> None:
    """spawn worker 初始化：显式单线程 BLAS（父进程建池前已置环境；此处复核声明意图）。"""
    for v in _BLAS_VARS:
        os.environ[v] = "1"


def _subject_worker(npz_path: str) -> dict:
    """spawn worker：单例装载（字段断言+z-score）→ 双变量 GC 全流程（含逐对 BIC）。"""
    t0 = time.time()
    prep = data_io.prepare_subject(npz_path)
    r = bivar_gc.bivar_gc_matrix(prep["x"], p_max=C.P_MAX)
    return {"subject_id": int(prep["meta"]["subject_id"]), "group": prep["meta"]["group"],
            "meta": prep["meta"], "ec": r["ec"], "orders": r["orders"],
            "n": r["n"], "wall_s": time.time() - t0}


def _order_hist(orders: np.ndarray) -> dict:
    """逐对阶数分布（上三角计数；对称登记的 orders 取上三角即无重复）。"""
    Cn = orders.shape[0]
    iu = np.triu_indices(Cn, k=1)
    vals, counts = np.unique(orders[iu], return_counts=True)
    return {str(int(v)): int(cnt) for v, cnt in zip(vals, counts)}


def stage_main(args, log) -> None:
    manifest = data_io.load_manifest()
    sids = sorted(s for s, e in manifest.items() if e["status"] == "ok")
    assert sids == list(C.EXPECTED_IDS), \
        f"manifest 48 例与 EXPECTED_IDS 不一致：{sids}"
    run_dir = C.OUTPUTS_ROOT / args.run_id
    refs_dir = run_dir / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)
    log(f"[main] 启动：{len(sids)} 例（manifest 全 status=ok）；p_max={C.P_MAX}；"
        f"{args.n_proc} 进程 spawn 并行（worker 单线程 BLAS）")
    # worker BLAS 单线程：spawn 子进程 re-import 时即生效（父进程仅编排，不做重 numpy 活）
    for v in _BLAS_VARS:
        os.environ[v] = "1"
    ctx = multiprocessing.get_context("spawn")
    t_start = time.time()
    results = []
    with ctx.Pool(processes=args.n_proc, initializer=_worker_init) as pool:
        for r in pool.imap_unordered(_subject_worker,
                                     [manifest[s]["npz_path"] for s in sids]):
            results.append(r)
            od = r["ec"][~np.eye(C.N_CHANNELS, dtype=bool)]
            log(f"[main {len(results)}/{len(sids)}] s{r['subject_id']} ({r['group']}) "
                f"n_eff={int(r['n'])} gc: min={od.min():.4f} median={np.median(od):.4f} "
                f"max={od.max():.4f}（{r['wall_s']:.1f}s）")
    results.sort(key=lambda r: r["subject_id"])
    # ---- 完整性断言（§2.2 实现正确性契约）+ 落盘 ----
    ec_all = []
    for r in results:
        ec, od = r["ec"], r["orders"]
        assert ec.shape == (C.N_CHANNELS, C.N_CHANNELS)
        assert np.isfinite(ec).all() and float(np.abs(np.diag(ec)).max()) == 0.0
        assert (ec[~np.eye(C.N_CHANNELS, dtype=bool)] >= 0).all(), \
            f"s{r['subject_id']}: offdiag GC < 0（实现错误）"
        assert int(od[0, 1]) > 0 and (od == od.T).all()
        np.savez_compressed(refs_dir / f"refs_s{r['subject_id']}.npz",
                            ec=ec.astype(np.float32), orders=od,
                            n=r["n"], subject_id=r["subject_id"])
        ec_all.append(ec)
    ec_all = np.stack(ec_all)
    sids_str = [str(r["subject_id"]) for r in results]
    np.savez_compressed(run_dir / "refs_all.npz",
                        bivar_gc=ec_all.astype(np.float32), subject_ids=sids_str)
    (run_dir / "data").mkdir(parents=True, exist_ok=True)
    (run_dir / "data" / "resample_meta.json").write_text(
        json.dumps([r["meta"] for r in results], ensure_ascii=False, indent=1))
    od_flat = ec_all[:, ~np.eye(C.N_CHANNELS, dtype=bool)]   # 批次维广播（(48,62,62)×(62,62)）
    order_hists = {str(r["subject_id"]): _order_hist(r["orders"]) for r in results}
    orders_pooled = np.sum([r["orders"] for r in results], axis=0)   # 对称逐例求和
    pooled = _order_hist(orders_pooled)
    summary = {
        "n_subjects": len(results), "subject_ids": sids_str,
        "groups": {str(r["subject_id"]): r["group"] for r in results},
        "caliber": {
            "method": "pairwise bivariate GC（不条件化于其余通道；observational "
                      "directed-connectivity reference 双变量口径）",
            "readout": "GC(A→B)=ln(SSR_r^B/SSR_u^B)，同一次拟合双向读出；"
                       "行=源/列=靶/对角=0",
            "bic": f"逐对 BIC 选阶 p∈{{1..{C.P_MAX}}}，行集 row_from=P_MAX={C.P_MAX}"
                   "（嵌套行集）；BIC=ln|Σ̂|+K²p·ln(n)/n（K=2，截距不进惩罚）",
            "fs_raw": C.FS_RAW, "fs": C.FS,
            "resample": "resample_poly(up=1,down=2)（抗混叠 FIR 内置；release 已带限 1–40 Hz 不重复滤波）",
            "zscore": "逐通道全记录 ddof=0，σ 下限 1e-12（refs_fmt 同式）",
            "storage": "float32 落盘 / float64 计算",
            "nonneg": "OLS 样本内 SSR 单调性 ⇒ offdiag≥0 必然（实现正确性断言；"
                      "与 ridge 条件 GC 允许微负为不同口径）",
            "terminology": "observational directed-connectivity reference（非 ground-truth）",
            "out_of_scope": "不做 EC_pert 与 GC 的比较（prd 第四轮拍板）；"
                            "fmt 条件 GC 滞后决策",
        },
        "refs_stats": {"offdiag_min": float(od_flat.min()),
                       "offdiag_median": float(np.median(od_flat)),
                       "offdiag_max": float(od_flat.max())},
        "order_distribution": {"per_subject": order_hists, "pooled": pooled},
        "wall_s_total": round(time.time() - t_start, 1),
        "wall_s_per_subject": {str(r["subject_id"]): round(r["wall_s"], 1) for r in results},
    }
    (run_dir / "refs_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    (run_dir / "config_snapshot.json").write_text(json.dumps({
        "run_id": args.run_id, "entry": str(_HERE / "run.py"),
        "data_root": str(C.DATA_ROOT), "fs_raw": C.FS_RAW, "fs": C.FS,
        "resample": [C.RESAMPLE_UP, C.RESAMPLE_DOWN], "p_max": C.P_MAX,
        "expected_ids": list(C.EXPECTED_IDS), "outputs_root": str(C.OUTPUTS_ROOT),
        "n_proc": args.n_proc,
        "blas_threads": {k: os.environ.get(k) for k in _BLAS_VARS},
        "command": f"python3 {__file__} --stage {args.stage} --run-id {args.run_id} "
                   f"--n-proc {args.n_proc}",
    }, ensure_ascii=False, indent=1))
    log(f"[main] 完成：refs_all.npz ({len(results)},62,62) f32 + 逐例 refs_s*.npz + "
        f"refs_summary.json + config_snapshot.json（总耗时 {time.time() - t_start:.0f}s）")


# ---- split（B3 接线：每半同口径含逐对 BIC → 逐例 offdiag Spearman） ----

def _split_worker(npz_path: str) -> dict:
    t0 = time.time()
    prep = data_io.prepare_subject(npz_path)
    r = sh.ceiling_for_subject(prep["x"], p_max=C.P_MAX)
    return {"subject_id": int(prep["meta"]["subject_id"]),
            "sh_rho": r["sh_rho"], "n_seg": r["n_seg"],
            "n_a": r["n_a"], "n_b": r["n_b"], "wall_s": time.time() - t0}


def stage_split(args, log) -> None:
    manifest = data_io.load_manifest()
    sids = sorted(s for s, e in manifest.items() if e["status"] == "ok")
    assert sids == list(C.EXPECTED_IDS)
    run_dir = C.OUTPUTS_ROOT / args.run_id
    assert (run_dir / "refs_summary.json").exists(), "先跑 --stage main（同 run_id）"
    for v in _BLAS_VARS:
        os.environ[v] = "1"
    ctx = multiprocessing.get_context("spawn")
    log(f"[split] 启动：{len(sids)} 例 × 两半（段 {C.SEG_SAMPLES} 样本 = "
        f"{C.SEG_S}s；偶A/奇B/尾弃/段内设计不跨段；同口径含逐对 BIC）")
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
           "caliber": "split_half.py 原式（双变量 GC 同口径重跑两半，含逐对 BIC；"
                      "offdiag 3782 有向边 Spearman ρ/例）",
           "ceiling": {"mean": float(arr.mean()), "sd": float(arr.std(ddof=1)),
                       "per_subject": {str(r["subject_id"]): r["sh_rho"] for r in results}},
           "wall_s_total": round(time.time() - t_start, 1)}
    (run_dir / "split_half.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    log(f"[split] 完成：ceiling = {out['ceiling']['mean']:.4f} ± "
        f"{out['ceiling']['sd']:.4f}（mean±sd，ddof=1）→ split_half.json")


# ---- figs（B4：经 common/visualizer 正典路由；纯复用既有图型，零 visualizer 改动） ----

def stage_figs(args, log) -> None:
    """B4 落图：①组均值三面板（ec_matrix：HC/MDD/全体均值）；②逐例抽样网格
   （ec_heatmap_grid：确定性等距 6 例）。阶数分布因 100% 顶格 p*=24（Run 002 实测）
    退化为单点，以 refs_summary.json 登记呈现，不为单柱图新增 visualizer 图型。"""
    run_dir = C.OUTPUTS_ROOT / args.run_id
    summary = json.loads((run_dir / "refs_summary.json").read_text())
    d = np.load(run_dir / "refs_all.npz", allow_pickle=True)
    ec_all = d["bivar_gc"].astype(np.float64)
    sids = [str(s) for s in d["subject_ids"]]
    groups = summary["groups"]

    _COMMON = _HERE.parent.parent / "common"      # 统一 visualizer 正典路由
    if str(_COMMON) not in sys.path:
        sys.path.append(str(_COMMON))
    from visualizer import Visualizer             # noqa: E402（正典路由，B4 时点导入）

    viz = Visualizer(str(run_dir / "figs"))
    idx = {"HC": [i for i, s in enumerate(sids) if groups[s] == "HC"],
           "MDD": [i for i, s in enumerate(sids) if groups[s] == "MDD"]}
    hc, mdd, allm = ec_all[idx["HC"]].mean(0), ec_all[idx["MDD"]].mean(0), ec_all.mean(0)

    od = allm[~np.eye(C.N_CHANNELS, dtype=bool)]
    p1 = viz.render("ec_matrix", {
        "ec_a": hc, "ec_b": mdd, "ec_c": allm,
        "label_a": f"HC mean (n={len(idx['HC'])})", "label_b": f"MDD mean (n={len(idx['MDD'])})",
        "label_c": f"all mean (n={len(sids)})",
        "stats": {"median": float(np.median(od)), "max": float(od.max())},
    }, "bivar_gc_group_means.png")
    log(f"[figs] 组均值三面板 → {p1}")

    pick = np.linspace(0, len(sids) - 1, 6).astype(int)   # 确定性等距抽样
    p2 = viz.render("ec_heatmap_grid", {
        "rows": [{"label": "bivariate GC ln(SSR)", "mats": [ec_all[i] for i in pick]}],
        "col_labels": [f"s{sids[i]} ({groups[sids[i]]})" for i in pick],
        "unit": "ln(SSR ratio)",
        "note": "pairwise bivariate GC · 125 Hz · per-pair BIC (p*=24 all pairs)",
    }, "bivar_gc_subjects_sample.png")
    log(f"[figs] 逐例抽样网格（等距 6 例）→ {p2}")
    log("[figs] 完成（阶数分布退化单点 → refs_summary.json 登记，不新增图型）")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", required=True, choices=["smoke", "main", "split", "figs"])
    ap.add_argument("--run-id", default="refs-01", help="main/split 产物 run 目录名")
    ap.add_argument("--subjects", type=int, default=1, help="smoke 例数（manifest 升序前 N）")
    ap.add_argument("--n-proc", type=int, default=C.N_PROC, help="main/split 并行进程数")
    args = ap.parse_args()

    if args.stage == "smoke":
        stage_smoke(args)                      # 控制台冒烟：不落盘不建 run_id
        return
    run_dir = C.OUTPUTS_ROOT / args.run_id
    log = _log_factory(run_dir / "logs" / "run.log")
    if args.stage == "main":
        stage_main(args, log)
    elif args.stage == "split":
        stage_split(args, log)
    else:
        stage_figs(args, log)


if __name__ == "__main__":
    main()
