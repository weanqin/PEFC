#!/usr/bin/env python3
"""compare_wrap —— B5 后半比较器独立 CLI（design §5；experiment.md Run 006）。

EC_pert 三 readout 契约 npz（readout-01，只读）× observational directed-connectivity
references（refs-01 refs_all.npz + split-half 天花板，只读）→ **子进程隔离**调
baseline-ec-v1 compare 叶子函数 → comparison.json（全量数值+口径）/ 
comparison_matrices.npz / figs/compare_forest.png。

入口独立性（任务派发口径，与 refs_fmt.py 同款）：**不经 run.py**——CLI 冲突规避。
baseline-ec-v1 与 run.py 零改写；readout-01 / refs-01 产物零改写。

隔离纪律（design §1）：父进程**零 import** baseline-ec-v1 模块（compare.py 模块级
`import config as C` 进程内必被本变体 config.py 遮蔽击穿）；一切比较计算经 fresh
`python3 tools/_compare_worker.py`（cwd=baseline 目录、PYTHONPATH 仅 baseline 目录）
+ manifest/npz/json 文件面交换。

主比较口径（design §5 / evaluation-calibers「Study II 扰动 EC 对比口径」，预注册唯一）：
主 readout slope × 两参考法 → 逐被试距离偏相关 Spearman ρ → 单样本 t（n=10）×
Holm 跨 2 法 + 1000×受试者 bootstrap CI（seed 2026）+ ρ̄/ceiling 比值；auc/rstar 同
流程数值作上下文（不进 Holm 族）。术语纪律：GC/gPDC 一律 observational
directed-connectivity references，无 ground-truth 表述。

用法：python3 models/perturb-ec-v1/compare_wrap.py --run-id compare-01
"""

from __future__ import annotations

import os

# ---- BLAS 护栏（environment.md / _blas_guard 惯例：必须在 numpy/scipy 之前；----
# ---- 16 核 cgroup 限 8 线程，setdefault 允许外部环境变量覆盖）              ----
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse        # noqa: E402
import importlib.util  # noqa: E402
import json            # noqa: E402
import subprocess      # noqa: E402
import sys             # noqa: E402
import time            # noqa: E402
from datetime import date   # noqa: E402
from pathlib import Path    # noqa: E402

import numpy as np  # noqa: E402

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_COMMON = _HERE.parent.parent / "common"     # 统一 visualizer（visualization.md 正典路由）
if str(_COMMON) not in sys.path:
    sys.path.append(str(_COMMON))

# ---- 本变体 config 唯一名装载（design §1 四行配方；就地复刻 refs_fmt 形态，----
# ---- 避免 import perturb 连带 torch——本入口 CPU-only numpy 路径）          ----
_SPEC = importlib.util.spec_from_file_location("perturb_config", str(_HERE / "config.py"))
_PC = importlib.util.module_from_spec(_SPEC)
sys.modules["perturb_config"] = _PC      # 先注册后 exec（design §1 四行配方）
_SPEC.loader.exec_module(_PC)

# ---- 比较器口径常量（数值计算全在 worker——baseline compare/config 单源）----
BASELINE_DIR = Path("/root/npi/models/baseline-ec-v1")
WORKER = _HERE / "tools" / "_compare_worker.py"
FMT_DATA_ROOT = Path("/root/autodl-tmp/edra_processed_fmt")   # chan_coords_62.npy（与 v4 侧逐位相同）
READOUTS = ("slope", "auc", "rstar")     # 主 / 上下文 / 上下文（Holm 族构成见 comparison.json）
METHODS = ("gc", "gpdc")                 # observational directed-connectivity references
AC6_ALPHA = 0.05                         # 初步证据门槛（ρ̄>0 且 Holm 校正 p<0.05）


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


def build_dist_matrix() -> np.ndarray:
    """chan_coords_62.npy (62,3) → 通道对欧氏距离 (62,62) mm。

    coords 单位米（范围 ±0.095）→ ×1000 mm，与 run.py `_dist_mm_matrix` 同式
    （Run 004 空间衰减口径）；偏相关为秩变换尺度不变，mm 仅为产物口径一致。"""
    coords = np.asarray(np.load(FMT_DATA_ROOT / "chan_coords_62.npy"), dtype=np.float64)
    assert coords.shape == (_PC.N_SRC, 3), f"chan_coords 形状异常: {coords.shape}"
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff * diff).sum(-1)) * 1000.0
    np.fill_diagonal(dist, 0.0)
    assert np.isfinite(dist).all()
    return dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="compare-01")
    ap.add_argument("--refs-run", default="refs-01")
    ap.add_argument("--readout-run", default="readout-01")
    args = ap.parse_args()

    outputs = Path(_PC.OUTPUTS_ROOT)
    run_dir = outputs / args.run_id
    refs_dir = outputs / args.refs_run
    ro_dir = outputs / args.readout_run
    (run_dir / "figs").mkdir(parents=True, exist_ok=True)
    log = _log_factory(run_dir / "logs" / "run.log")

    t_start = time.time()
    sids = [str(s) for s in _PC.EXPECTED_TEST_IDS]
    log(f"[compare] 启动：run={args.run_id}；10 例 {sids}；refs={args.refs_run} "
        f"readout={args.readout_run}（均只读）；主比较 = slope × {{gc, gpdc}} Holm")

    # ---- 输入面就绪断言（refs_all + 天花板 + 30 契约 npz）----
    refs_npz = refs_dir / "refs_all.npz"
    refs_summary_path = refs_dir / "refs_summary.json"
    assert refs_npz.exists() and refs_summary_path.exists(), "refs-01 产物缺失"
    d = np.load(refs_npz, allow_pickle=True)
    sids_ref = [str(s) for s in np.atleast_1d(d["subject_ids"])]
    assert sids_ref == sids, f"refs subject_ids {sids_ref} != EXPECTED_TEST_IDS 升序 {sids}"
    ec_paths = {r: [str(ro_dir / f"ec_pert_{r}_s{sid}.npz") for sid in sids]
                for r in READOUTS}
    for r, paths in ec_paths.items():
        for p in paths:
            assert Path(p).exists(), f"契约 npz 缺失: {p}"
    refs_summary = json.loads(refs_summary_path.read_text())
    ceilings = {m: float(refs_summary["split_half"][m]["mean"]) for m in METHODS}
    log(f"[ceiling] refs-01 fmt 侧 split-half："
        + "  ".join(f"{m}={ceilings[m]:.4f}" for m in METHODS))

    # ---- 距离矩阵（偏出控制变量）----
    dist = build_dist_matrix()
    np.save(run_dir / "dist_matrix.npy", dist)
    offd = dist[~np.eye(_PC.N_SRC, dtype=bool)]
    log(f"[dist] chan_coords_62.npy → (62,62) 欧氏距离；offdiag min/median/max = "
        f"{offd.min():.1f}/{np.median(offd):.1f}/{offd.max():.1f} mm")

    # ---- 子进程 worker（fresh python；cwd/PYTHONPATH 仅 baseline 目录）----
    out_json = run_dir / "worker_results.json"
    out_npz = run_dir / "comparison_matrices.npz"
    manifest = {"refs_npz": str(refs_npz), "dist_npy": str(run_dir / "dist_matrix.npy"),
                "ec_paths": ec_paths, "ceilings": ceilings,
                "out_json": str(out_json), "out_npz": str(out_npz)}
    (run_dir / "worker_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1))
    env = {**os.environ, "PYTHONPATH": str(BASELINE_DIR)}
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, str(WORKER), str(run_dir / "worker_manifest.json")],
        cwd=str(BASELINE_DIR), env=env, capture_output=True, text=True, check=False)
    worker_wall = time.time() - t0
    if proc.returncode != 0:
        log(f"[worker] 失败 rc={proc.returncode}\n--- stderr ---\n{proc.stderr}")
        raise RuntimeError(f"worker 失败（rc={proc.returncode}）：见 run.log")
    summary = json.loads(proc.stdout.strip().splitlines()[-1])
    assert summary.get("ok") is True
    payload = json.loads(out_json.read_text())
    log(f"[worker] 完成 wall {worker_wall:.1f}s；rho_partial_mean = "
        + json.dumps(summary["rho_partial_mean"]))

    # ---- comparison.json 装配（Holm 族 = slope×{gc,gpdc}；auc/rstar 上下文不进族）----
    results = payload["readouts"]
    main_res = results["slope"]
    gate = {m: {"rho_partial_mean": main_res["methods"][m]["rho_partial_mean"],
                "holm_p": main_res["holm_p"][m],
                "pass": bool(main_res["methods"][m]["rho_partial_mean"] > 0
                             and main_res["holm_p"][m] < AC6_ALPHA)}
            for m in METHODS}

    def _ctx_entry(r: str) -> dict:
        e = dict(results[r])
        e.pop("holm_p", None)          # 上下文不进 Holm 族：不落 readout 内跨法 holm 值
        e["holm_family"] = "not_in_holm_family (context readout)"
        return e

    comparison = {
        "run_id": args.run_id, "date": date.today().isoformat(),
        "n_subjects": payload["n_subjects"], "subject_ids": payload["subject_ids"],
        "terminology": (
            "GC/gPDC = observational directed-connectivity references; goal = "
            "structural convergence check, NOT ground-truth recovery. "
            "EC_pert = perturbation-derived connectivity readouts from frozen surrogate "
            "theta* (process-validation grade: single model, single fold)."),
        "caliber": {
            "main_comparison": (
                "primary readout slope x {gc, gpdc}: per-subject distance-partial "
                "Spearman rho (rank-based 3-vector partial correlation controlling "
                "channel-pair 3D Euclidean distance; 3782 directed edges, rows=source/"
                "cols=target) -> one-sample t (n=10, df=9) x Holm across 2 methods "
                "(family = {slope x gc, slope x gpdc}) + 1000x subject bootstrap 95% CI "
                "(seed 2026) + rho_bar/ceiling ratio"),
            "context_not_in_holm_family": [
                "auc and rstar readouts: same pipeline numbers (t, raw p, bootstrap CI, "
                "ceiling ratio), flagged context, NOT part of the Holm family",
                "unpartial Spearman rho (rho_raw) per method as context",
                "rstar enters as pre-flipped strength = R_NEXT - r* "
                "(orientation=larger_is_stronger, censored edges -> 0)"],
            "references": ("fmt-side 125 Hz recompute (Run 005 refs-01): ridge conditional "
                           "GC ln(SSR ratio) and |gPDC|^2 broadband 1-40 Hz, p=8, "
                           "lam_rel=1e-2 (baseline-ec-v1 single source); raw edge values"),
            "bootstrap": {"n": payload["config_sources"]["bootstrap_n"],
                          "seed": payload["config_sources"]["bootstrap_seed"],
                          "source": "baseline-ec-v1 config.BOOTSTRAP_N/BOOTSTRAP_SEED 单源"},
            "ceilings_source": (
                f"refs-01 refs_summary.json split_half fmt-side means: "
                + ", ".join(f"{m}={ceilings[m]:.4f}" for m in METHODS)),
            "power_note": (
                "df=9, Holm-corrected alpha~0.025, typical per-subject rho sd 0.25-0.35 "
                "-> minimum detectable rho_bar ~0.55-0.65; a null result is "
                "insufficient evidence, not falsification of convergence"),
            "ac6_gate": f"rho_bar > 0 AND Holm-adjusted p < {AC6_ALPHA} for BOTH methods",
            "implementation": (
                "baseline-ec-v1 compare leaf functions (load_ec_pert / run_comparison / "
                "partial_spearman / holm / bootstrap_ci_mean) imported only inside a "
                "fresh subprocess worker; parent process imports zero baseline modules; "
                "npz/json file-face exchange"),
        },
        "main_comparison": {**main_res, "gate": gate,
                            "gate_pass_overall": bool(all(g["pass"] for g in gate.values()))},
        "context": {r: _ctx_entry(r) for r in ("auc", "rstar")},
        "negative_value_counts_post_contract_abs":
            payload["negative_value_counts_post_contract_abs"],
        "worker_isolation": {
            "interpreter": sys.executable, "script": str(WORKER),
            "cwd": str(BASELINE_DIR), "pythonpath": str(BASELINE_DIR),
            "config_sources": payload["config_sources"],
            "worker_wall_s": round(worker_wall, 1),
            "asserts": ("worker 内断言 config/compare __file__ 归属 baseline-ec-v1 + "
                        "逐例 subject_ids 与 refs 对齐；父进程零 import baseline 模块"),
        },
        "inputs": {"refs_npz": str(refs_npz), "readout_run": str(ro_dir),
                   "ec_paths": ec_paths, "dist_npy": str(run_dir / "dist_matrix.npy")},
        "wall_s_total": round(time.time() - t_start, 1),
    }
    (run_dir / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=1))

    # ---- 汇总图（common/visualizer 纯增量 compare_forest；ASCII 注记）----
    import visualizer as VZ
    viz = VZ.Visualizer(str(run_dir / "figs"))
    rows = []
    for r in READOUTS:
        for m in METHODS:
            e = results[r]["methods"][m]
            rows.append({"readout": r, "method": m,
                         "rho": e["rho_partial_mean"], "ci": e["rho_partial_ci95"],
                         "rho_raw": e["rho_raw_mean"],
                         "main": r == "slope",
                         "holm_p": results[r]["holm_p"][m] if r == "slope" else None,
                         "ceiling_ratio": e.get("ceiling_ratio")})
    fig_path = viz.render("compare_forest", {
        "rows": rows, "n_subjects": comparison["n_subjects"], "ceilings": ceilings,
        "note": ("GC/gPDC = observational directed-connectivity references; "
                 "structural convergence check only. Holm family = slope x {gc, gpdc}; "
                 "auc/rstar = context (not in family). "
                 "bootstrap 1000x, seed 2026; n=10 subjects")},
        "compare_forest.png")
    log(f"[fig] {fig_path}")

    # ---- 汇总登记行 ----
    for r in READOUTS:
        for m in METHODS:
            e = results[r]["methods"][m]
            holm_txt = (f"Holm p={results[r]['holm_p'][m]:.4f}"
                        if r == "slope" else "context (not in Holm family)")
            lo, hi = e["rho_partial_ci95"]
            log(f"[{r} x {m}] rho_bar={e['rho_partial_mean']:+.4f} "
                f"t={e['rho_partial_t']:+.3f} p={e['rho_partial_p']:.4f} {holm_txt} "
                f"CI95=[{lo:+.4f},{hi:+.4f}] rho_raw={e['rho_raw_mean']:+.4f} "
                f"ceil_ratio={e.get('ceiling_ratio')}")
    log(f"[AC6 gate] slope: "
        + "; ".join(f"{m}: rho>0={gate[m]['rho_partial_mean'] > 0}, "
                    f"Holm p={gate[m]['holm_p']:.4f}<{AC6_ALPHA}={gate[m]['holm_p'] < AC6_ALPHA}"
                    for m in METHODS)
        + f" -> overall_pass={comparison['main_comparison']['gate_pass_overall']}")
    (run_dir / "config_snapshot.json").write_text(json.dumps({
        "run_id": args.run_id, "entry": str(_HERE / "compare_wrap.py"),
        "worker": str(WORKER), "baseline_dir": str(BASELINE_DIR),
        "refs_run": args.refs_run, "readout_run": args.readout_run,
        "chan_coords": str(FMT_DATA_ROOT / "chan_coords_62.npy"),
        "expected_test_ids": list(_PC.EXPECTED_TEST_IDS),
        "outputs_root": _PC.OUTPUTS_ROOT, "ac6_alpha": AC6_ALPHA,
        "readouts": list(READOUTS), "methods": list(METHODS),
        "not_via_runpy": "B5 后半独立 CLI（任务派发口径，与 refs_fmt.py 同款）",
        "blas_threads": {k: os.environ.get(k) for k in
                         ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                          "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
    }, ensure_ascii=False, indent=1))
    log(f"[compare] 完成：comparison.json + comparison_matrices.npz + figs/；"
        f"总耗时 {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
