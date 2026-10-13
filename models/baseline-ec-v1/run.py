"""run —— baseline-ec-v1 CLI 编排。

stages:
  resample   46 例 → data/{sid}.npz（125 Hz z-scored）
  refs       两遍：①逐例 BIC → p_shared（中位）②逐例 GC/gPDC/(mne-c 交叉核对) → eval/refs/
  split-half 逐例奇偶两半 EC → eval/split_half.json（天花板）
  compare    比较器（EC_pert 合成矩阵联调，非科学产物）→ eval/comparison.json + 图
  smoke      第 1 例全链（含迷你 split-half 与 compare 联调）

用法：python3 models/baseline-ec-v1/run.py --stage smoke --run-id smoke-a [--mnec sub|all|off]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _blas_guard  # noqa: F401,E402  # 须在 numpy 之前：BLAS 线程护栏（16 核 cgroup）
import numpy as np  # noqa: E402
import config as C  # noqa: E402
import compare  # noqa: E402
import ec_refs  # noqa: E402
import resample_io  # noqa: E402
import split_half as sh  # noqa: E402

sys.path.insert(0, str(C.REPO_ROOT))
from common.visualizer import Visualizer  # noqa: E402


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


def stage_refs(run_dir: Path, log, mnec: str = "sub") -> Dict:
    data_dir = run_dir / "data"
    subjects = sorted(data_dir.glob("*.npz"))
    assert subjects, "先跑 resample"
    # ---- 遍 1：BIC 存档 + 固定阶/固定 λ（2026-10-13 定标：OLS 条件数 ~1e18、
    #      split-half 天花板≈0 → ridge λ_rel=1e-2；p=8 ASCT 锚点；见 experiment.md）----
    meta_rows = []
    for i, sp in enumerate(subjects):
        d = resample_io.load_cached(sp)
        r_bic = ec_refs.bic_pass(d["x"])
        meta_rows.append({"subject_id": d["meta"]["subject_id"], "group": d["meta"]["group"],
                          "p_bic": r_bic["p_star"]})
        log(f"[order {i+1}/{len(subjects)}] {d['meta']['subject_id']} p_bic={r_bic['p_star']}"
            f"（存档；正式口径 p={ec_refs.P_SHARED} ridge λ_rel={ec_refs.LAM_REL}）")
    p_shared = ec_refs.P_SHARED
    log(f"[order] p_shared={p_shared}（固定，ASCT p/2 谱峰锚点）")
    # ---- 遍 2：读出 ----
    refs_dir = run_dir / "eval" / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)
    gc_all, gp_all = [], []
    mnec_every = {"all": 1, "sub": 4, "off": 10 ** 9}[mnec]
    for i, sp in enumerate(subjects):
        d = resample_io.load_cached(sp)
        t0 = time.time()
        do_mnec = (i % mnec_every == 0) if mnec != "off" else False
        r = ec_refs.refs_for_subject(d["x"], p_shared, mnec=do_mnec)
        np.savez_compressed(refs_dir / f"{sp.stem}.npz",
                            gc=r["gc"].astype(np.float32),
                            gpdc_bb=r["gpdc_bb"].astype(np.float32),
                            gpdc_bands=r["gpdc_bands"].astype(np.float32),
                            mnec=r["mnec"].astype(np.float32) if do_mnec else np.zeros(0),
                            lb_frac=r["lb_frac"], n=r["n"], p=r["p"], lam=r["lam"])
        gc_all.append(r["gc"]); gp_all.append(r["gpdc_bb"])
        log(f"[refs {i+1}/{len(subjects)}] {d['meta']['subject_id']} "
            f"gc_max={r['gc'].max():.4f} gpdc_max={r['gpdc_bb'].max():.4f} "
            f"lb_frac={r['lb_frac']:.3f}{' +mnec' if do_mnec else ''} ({time.time()-t0:.1f}s)")
    gc_all, gp_all = np.stack(gc_all), np.stack(gp_all)
    np.savez_compressed(run_dir / "eval" / "refs_all.npz", gc=gc_all.astype(np.float32),
                        gpdc=gp_all.astype(np.float32),
                        subject_ids=[m["subject_id"] for m in meta_rows])
    summary = {"p_shared": p_shared, "lam_rel": ec_refs.LAM_REL, "bic": meta_rows}
    (run_dir / "eval" / "refs_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1))
    log(f"[refs] 完成：{len(subjects)} 例（p={p_shared}, λ_rel={ec_refs.LAM_REL}）；"
        f"产物 eval/refs_all.npz + refs_summary.json")
    return summary


def stage_split_half(run_dir: Path, log, max_subjects: int | None = None) -> Dict:
    data_dir = run_dir / "data"
    subjects = sorted(data_dir.glob("*.npz"))
    if max_subjects:
        subjects = subjects[:max_subjects]
    p_shared = json.loads((run_dir / "eval" / "refs_summary.json").read_text())["p_shared"]
    xs = {}
    for i, sp in enumerate(subjects):
        d = resample_io.load_cached(sp)
        xs[d["meta"]["subject_id"]] = d["x"]
        if (i + 1) % 10 == 0:
            log(f"[split-half] 载入 {i+1}/{len(subjects)}")
    t0 = time.time()
    out = sh.ceilings_all(xs, p_shared)
    for m, v in out.items():
        log(f"[split-half] ceiling[{m}] = {v['mean']:.4f} ± {v['sd']:.4f}")
    (run_dir / "eval" / "split_half.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    log(f"[split-half] 完成 ({time.time()-t0:.0f}s) → eval/split_half.json")
    return out


def stage_compare(run_dir: Path, log) -> Dict:
    refs_npz = np.load(run_dir / "eval" / "refs_all.npz", allow_pickle=True)
    refs = {"gc": refs_npz["gc"].astype(np.float64),
            "gpdc": refs_npz["gpdc"].astype(np.float64)}
    sids = [str(s) for s in refs_npz["subject_ids"]]
    dist = resample_io.channel_distance()
    # EC_pert 合成联调（非科学产物；真实 EC_pert 待 Study II，接口见 compare.load_ec_pert）
    pert = compare.synthetic_ec_pert_demo(refs["gc"].mean(0))
    np.savez(run_dir / "eval" / "ec_pert_demo.npz", ec=pert, readout="slope",
             subject_ids="global", orientation="larger_is_stronger", synthetic=True)
    ec_pert = {"ec": pert, "readout": "slope", "subject_ids": "global"}
    ceilings = None
    sh_path = run_dir / "eval" / "split_half.json"
    if sh_path.exists():
        sh = json.loads(sh_path.read_text())
        ceilings = {m: v["mean"] for m, v in sh.items()}
    res = compare.run_comparison(refs, sids, ec_pert, dist, ceilings=ceilings)
    res["ec_pert_note"] = "SYNTHETIC demo matrix（真实 EC_pert 待 Study II；readout 登记：slope 主）"
    (run_dir / "eval" / "comparison.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    for m in refs:
        e = res["methods"][m]
        ratio = e.get("ceiling_ratio")
        log(f"[compare] {m}: ρ_partial={e['rho_partial_mean']:.4f} "
            f"CI={e['rho_partial_ci95']} p_holm={res['holm_p'][m]:.3g} "
            f"{f'ratio={ratio:.2f}' if ratio is not None else '(ceiling 未就绪)'}")
    # 图（统一 visualizer，ec_matrix）
    vis = Visualizer(str(run_dir / "figs"))
    stats = {"pearson_r": res["methods"]["gc"]["rho_partial_mean"],
             "spearman_rho": res["methods"]["gpdc"]["rho_partial_mean"]}
    vis.render("ec_matrix", {
        "ec_a": refs["gc"].mean(0), "ec_b": refs["gpdc"].mean(0), "ec_c": pert,
        "label_a": "conditional GC (mean)", "label_b": "|gPDC|$^2$ broadband (mean)",
        "label_c": "EC_pert (SYNTHETIC demo)", "stats": stats,
    }, f"ec_matrix_{run_dir.name}.png")
    log("[compare] 完成 → eval/comparison.json + figs/ec_matrix_*.png")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["resample", "refs", "split-half", "compare", "smoke"])
    ap.add_argument("--run-id", default="main-01")
    ap.add_argument("--mnec", default="all", choices=["off", "sub", "all"],
                    help="mne-connectivity 二元交叉核对（62ch 实测 ~25s/例）：off/sub(每4例)/all")
    ap.add_argument("--subjects", type=int, default=None,
                    help="resample 仅前 N 例（smoke 用）")
    args = ap.parse_args()

    run_dir = C.OUTPUTS_ROOT / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log = _log_factory(run_dir / "logs" / "run.log")

    if args.stage in ("resample", "smoke"):
        subjects = resample_io.load_manifest()
        if args.subjects:
            subjects = subjects[:args.subjects]
        log(f"[resample] {len(subjects)} 例（filter 断言 {C.EXPECTED_FILTER!r}）")
        resample_io.stage_resample(run_dir, subjects, log)
        if args.stage == "resample":
            return
    if args.stage in ("refs", "smoke"):
        stage_refs(run_dir, log, mnec=args.mnec)
        if args.stage == "refs":
            return
    if args.stage in ("split-half", "smoke"):
        stage_split_half(run_dir, log, max_subjects=1 if args.stage == "smoke" else None)
        if args.stage == "split-half":
            return
    if args.stage in ("compare", "smoke"):
        stage_compare(run_dir, log)


if __name__ == "__main__":
    main()
