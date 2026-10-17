#!/usr/bin/env python3
"""refs_fmt —— B5 前半参考侧独立 CLI（design §4；experiment.md Run 005）。

fmt 10 例（fold0 test）→ 125 Hz 重采样 + 逐通道 z-score → **子进程隔离**逐例调
baseline-ec-v1（refs_for_subject + split_half.ceiling_for_subject）→ 汇总
refs_all.npz + refs_summary.json（含 split-half 天花板与数据源偏差登记）。

入口独立性（任务派发口径）：**不经 run.py**——B4 readout 代理并行开发中，CLI 冲突
规避；compare（B5 后半）另派代理，本脚本不触。baseline-ec-v1 与数据集只读。

隔离纪律（design §1）：父进程**零 import** baseline-ec-v1 模块（其模块级
`import config as C` 进程内必被本变体 config.py 遮蔽击穿）；一切 baseline 计算
经 fresh `python3 tools/_refs_worker.py`（cwd=baseline 目录、PYTHONPATH 仅 baseline
目录）+ npz 文件面交换。p/λ 不在父进程复写——由 ec_refs.P_SHARED/LAM_REL 单源
输出、从 worker 产物读回登记。

用法：python3 models/perturb-ec-v1/refs_fmt.py --run-id refs-01
"""

from __future__ import annotations

import os

# ---- BLAS 护栏（environment.md / _blas_guard 惯例：必须在 numpy/scipy 之前；----
# ---- 16 核 cgroup 限 8 线程，setdefault 允许外部环境变量覆盖）              ----
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse       # noqa: E402
import csv            # noqa: E402
import importlib.util # noqa: E402
import json           # noqa: E402
import subprocess     # noqa: E402
import sys            # noqa: E402
import time           # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np                 # noqa: E402
from scipy.signal import resample_poly  # noqa: E402

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# ---- 本变体 config 唯一名装载（design §1 配方；就地复刻 perturb.load_config，----
# ---- 避免 import perturb 连带 torch——本入口 CPU-only numpy/scipy 路径）      ----
_SPEC = importlib.util.spec_from_file_location("perturb_config", str(_HERE / "config.py"))
_PC = importlib.util.module_from_spec(_SPEC)
sys.modules["perturb_config"] = _PC      # 先注册后 exec（design §1 四行配方）
_SPEC.loader.exec_module(_PC)

# ---- 参考侧口径常量（design §4；p/λ 见 worker——baseline ec_refs 单源） ----
BASELINE_DIR = Path("/root/npi/models/baseline-ec-v1")
WORKER = _HERE / "tools" / "_refs_worker.py"
FMT_DATA_ROOT = Path("/root/autodl-tmp/edra_processed_fmt")
FS_RAW = 250                   # fmt 数据契约采样率（逐例断言）
FS = 125                       # 目标采样率（与 baseline-ec-v1 同物理率；λ/p 口径直接适用）
RESAMPLE_UP, RESAMPLE_DOWN = 1, 2   # fmt 250 Hz → 125 Hz（v4 侧为 1/4；design §4 登记）
N_CHANNELS = 62
N_RAW = 60001                  # fmt 契约长度（240.004 s）
EXPECTED_FIELDS = {"filter": "1-40(release)", "pipeline": "EDRA_release_1-40_250Hz"}


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


def _load_fmt_manifest() -> dict:
    """fmt manifest.csv → {sid: {group, status, npz_path}}（数据集只读）。"""
    out = {}
    for r in csv.DictReader(open(FMT_DATA_ROOT / "manifest.csv")):
        sid = int(r["subject_id"])
        out[sid] = {
            "group": r["group"], "status": r["status"],
            "npz_path": str(FMT_DATA_ROOT / r["group"] / f"{sid}_rest_v4.npz"),
        }
    return out


def prepare_subject(npz_path: str) -> dict:
    """fmt npz → 字段断言 → 125 Hz 抗混叠降采 → 逐通道全记录 z-score（ddof=0）。

    与 baseline-ec-v1 resample_io.load_and_prepare 同构（σ 下限 1e-12 逐字同式），
    仅数据源（fmt vs v4）与降采样比（1/2 vs 1/4）不同——design §4。
    返回 {x (62,30001) f64, meta}。
    """
    d = np.load(npz_path, allow_pickle=True)
    for k, want in EXPECTED_FIELDS.items():
        got = str(d[k])
        assert got == want, f"{npz_path}: 字段 {k}={got!r} != 契约 {want!r}"
    assert int(d["fs"]) == FS_RAW, f"{npz_path}: fs={int(d['fs'])} != {FS_RAW}"
    assert not bool(d["truncated"]), f"{npz_path}: truncated=True"
    x = np.asarray(d["data"], dtype=np.float64)
    assert x.shape == (N_CHANNELS, N_RAW), f"{npz_path}: 形状 {x.shape} 异常"
    x125 = resample_poly(x, RESAMPLE_UP, RESAMPLE_DOWN, axis=-1)
    mu = x125.mean(axis=-1, keepdims=True)
    sd = x125.std(axis=-1, keepdims=True)          # ddof=0（全记录统计）
    x125z = (x125 - mu) / np.maximum(sd, 1e-12)
    meta = {
        "subject_id": str(d["subject_id"]), "group": str(d["group"]),
        "filter": str(d["filter"]), "pipeline": str(d["pipeline"]),
        "suffix": str(d["suffix"]), "fs_raw": int(d["fs"]), "fs": FS,
        "n_raw": int(x.shape[-1]), "n_125": int(x125.shape[-1]),
        "resample": f"resample_poly(up={RESAMPLE_UP},down={RESAMPLE_DOWN})",
        "zscore": "per-channel whole-record ddof=0 sigma_floor=1e-12",
    }
    return {"x": x125z, "meta": meta}


def run_worker(x_npy: Path, out_npz: Path, log) -> dict:
    """fresh 子进程调 baseline-ec-v1 worker（cwd/PYTHONPATH 仅 baseline 目录）。"""
    env = {**os.environ, "PYTHONPATH": str(BASELINE_DIR)}
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, str(WORKER), str(x_npy), str(out_npz)],
        cwd=str(BASELINE_DIR), env=env, capture_output=True, text=True, check=False)
    dt = time.time() - t0
    if proc.returncode != 0:
        log(f"[worker] 失败 rc={proc.returncode}\n--- stderr ---\n{proc.stderr}")
        raise RuntimeError(f"worker 失败（rc={proc.returncode}）：见 run.log")
    summary = json.loads(proc.stdout.strip().splitlines()[-1])   # 末行 = JSON 摘要
    assert summary.get("ok") is True
    return {"summary": summary, "wall_s": dt}


def _offdiag(m: np.ndarray) -> np.ndarray:
    return m[~np.eye(m.shape[0], dtype=bool)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="refs-01")
    args = ap.parse_args()

    run_dir = Path(_PC.OUTPUTS_ROOT) / args.run_id
    data_dir = run_dir / "data"
    refs_dir = run_dir / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)
    log = _log_factory(run_dir / "logs" / "run.log")

    t_start = time.time()
    sids = list(_PC.EXPECTED_TEST_IDS)          # 升序 [10,...,69]
    manifest = _load_fmt_manifest()
    for sid in sids:
        assert sid in manifest and manifest[sid]["status"] == "ok", \
            f"subject {sid} 不在 fmt manifest 或 status 非 ok"
    log(f"[refs] 启动：{len(sids)} 例 {sids}；fmt {FS_RAW}→{FS} Hz "
        f"(down={RESAMPLE_DOWN})；p/λ 由 baseline ec_refs 单源（worker 默认）")

    # ---- 阶段 A：重采样 + z-score → data/x_s{sid}.npy（float32 落盘） ----
    metas = []
    for i, sid in enumerate(sids):
        e = manifest[sid]
        prep = prepare_subject(e["npz_path"])
        data_dir.mkdir(parents=True, exist_ok=True)
        np.save(data_dir / f"x_s{sid}.npy", prep["x"].astype(np.float32))
        metas.append(prep["meta"])
        log(f"[resample {i+1}/{len(sids)}] s{sid} ({e['group']}) "
            f"fs {FS_RAW}→{FS} n={prep['meta']['n_125']} "
            f"filter={prep['meta']['filter']!r} suffix={prep['meta']['suffix']!r}")
    (data_dir / "resample_meta.json").write_text(
        json.dumps(metas, ensure_ascii=False, indent=1))

    # ---- 阶段 B：逐例子进程 worker（refs + split-half） ----
    per_subject, worker_walls = {}, []
    for i, sid in enumerate(sids):
        out_npz = refs_dir / f"refs_s{sid}.npz"
        r = run_worker(data_dir / f"x_s{sid}.npy", out_npz, log)
        s = r["summary"]
        worker_walls.append(r["wall_s"])
        per_subject[sid] = s
        log(f"[worker {i+1}/{len(sids)}] s{sid} p={s['p']} λ={s['lam']:.4g} "
            f"gc_max={s['gc_max']:.4f} gpdc_max={s['gpdc_max']:.4f} lb_frac={s['lb_frac']:.3f} "
            f"| split-half gc={s['sh_rho_gc']:.4f} gpdc={s['sh_rho_gpdc']:.4f} "
            f"(n_seg={s['n_seg']}，refs {s['t_refs_s']}s + sh {s['t_sh_s']}s，wall {r['wall_s']:.1f}s)")

    # ---- 阶段 C：聚合 + 完整性断言 + 统计 ----
    gc_all, gp_all = [], []
    for sid in sids:
        d = np.load(refs_dir / f"refs_s{sid}.npz")
        gc, gp = d["gc"].astype(np.float64), d["gpdc_bb"].astype(np.float64)
        assert gc.shape == (N_CHANNELS, N_CHANNELS) and gp.shape == (N_CHANNELS, N_CHANNELS)
        assert np.isfinite(gc).all() and np.isfinite(gp).all()
        assert float(np.abs(np.diag(gc)).max()) == 0.0, "gc 对角非零（口径违约）"
        assert gp.min() >= 0.0 and gp.max() <= 1.0 + 1e-6, "gpdc 超出 [0,1]（口径违约）"
        gc_all.append(gc); gp_all.append(gp)
    gc_all, gp_all = np.stack(gc_all), np.stack(gp_all)
    np.savez_compressed(run_dir / "refs_all.npz",
                        gc=gc_all.astype(np.float32), gpdc=gp_all.astype(np.float32),
                        subject_ids=[str(s) for s in sids])

    # split-half 天花板（ddof=1，与 split_half.ceilings_all 同式）
    ceilings = {}
    for m, key in (("gc", "sh_rho_gc"), ("gpdc", "sh_rho_gpdc")):
        arr = np.array([per_subject[s][key] for s in sids])
        ceilings[m] = {"mean": float(arr.mean()), "sd": float(arr.std(ddof=1)),
                       "per_subject": {str(s): per_subject[s][key] for s in sids}}

    # refs 基本统计（offdiag 3782 边/例；登记用）
    def _stats(mats):
        od = np.concatenate([_offdiag(m) for m in mats])
        return {"offdiag_min": float(od.min()), "offdiag_median": float(np.median(od)),
                "offdiag_max": float(od.max()),
                "frac_le_0": float((od <= 0).mean()),
                "frac_le_1e-3": float((od <= 1e-3).mean())}
    diag_gp = np.concatenate([np.diag(m) for m in gp_all])
    refs_stats = {
        "gc": {**_stats(gc_all), "diag": "0（ridge_conditional_gc 按构造只填非对角）"},
        "gpdc": {**_stats(gp_all),
                 "diag_median": float(np.median(diag_gp)),
                 "diag": "非零自连接照存（比较器 offdiag_flat 展平自然剔除）"},
        "lb_frac_mean": float(np.mean([per_subject[s]["lb_frac"] for s in sids])),
    }

    p_used = int(per_subject[sids[0]]["p"])
    lam_used = float(np.load(refs_dir / f"refs_s{sids[0]}.npz")["lam"])
    n_seg = int(per_subject[sids[0]]["n_seg"])
    summary = {
        "n_subjects": len(sids), "subject_ids": [str(s) for s in sids],
        "groups": {str(s): manifest[s]["group"] for s in sids},
        "caliber": {
            "p_shared": p_used, "lam_rel": lam_used, "fs_raw": FS_RAW, "fs": FS,
            "resample": f"resample_poly(up={RESAMPLE_UP},down={RESAMPLE_DOWN})（抗混叠 FIR 内置）",
            "zscore": "逐通道全记录 ddof=0，σ 下限 1e-12（与 baseline resample_io 同式）",
            "storage": "float32 落盘 / float64 复原（与 resample_io cache 往返一致）",
            "gc": "ridge 条件 ln(SSR比)，行=源/列=靶/对角=0（微负可为收缩口径）",
            "gpdc": "|gPDC|² 宽带 1–40 Hz，[0,1] 列平方和=1，行=源",
            "p_lam_source": "baseline-ec-v1 ec_refs.P_SHARED/LAM_REL 单源（父进程不复写）",
            "mnec": "不做（本任务口径）",
        },
        "data": {
            "root": str(FMT_DATA_ROOT), "n_125": int(metas[0]["n_125"]),
            "deviation_from_baseline_v4": (
                "fmt：EDRA release 1–40 Hz 带限 @250 Hz，无 ICA、无 notch，48 例，"
                "240.004 s/例，n_125=30001，split-half 120 段（60/半）；"
                "v4（10-13 任务）：bp1-40+notch50 @500 Hz，46 例，~170–340 s/例，"
                "n_125=21250，85 段（42/43/半）——同 λ/p 下数据源系统性差异，方向性对照只登记不判读"),
            "v4_caliber_ceilings_for_reference": {"gc": 0.9003, "gc_sd": 0.0296,
                                                  "gpdc": 0.7344, "gpdc_sd": 0.0387,
                                                  "n": 46,
                                                  "source": "outputs/baseline-ec-v1/main-01/eval/split_half.json"},
        },
        "split_half": {
            **ceilings,
            "seg_s": 2.0, "seg_samples": 250, "n_seg": n_seg,
            "caliber": "baseline-ec-v1 split_half.py 原样（偶段A/奇段B/尾巴丢弃/段内设计不跨段/offdiag 3782 Spearman）",
        },
        "refs_stats": refs_stats,
        "worker_isolation": {
            "interpreter": sys.executable, "script": str(WORKER),
            "cwd": str(BASELINE_DIR), "pythonpath": str(BASELINE_DIR),
            "asserts": "worker 内断言 config/gc_psd __file__ 归属；父进程零 import baseline 模块",
            "wall_s_per_subject": [round(w, 1) for w in worker_walls],
        },
        "compare_input_layout": {
            "refs_all.npz": {
                "gc": "(10,62,62) float32 行=源/列=靶，序=EXPECTED_TEST_IDS 升序",
                "gpdc": "(10,62,62) float32（=gpdc_bb 宽带）",
                "subject_ids": "(10,) str",
                "consumption": "refs={'gc':…f64,'gpdc':…f64} + sids → baseline-ec-v1 compare.run_comparison 输入形态（v4 侧 refs_all.npz 同构）",
            },
        },
        "wall_s_total": round(time.time() - t_start, 1),
    }
    (run_dir / "refs_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1))
    (run_dir / "config_snapshot.json").write_text(json.dumps({
        "run_id": args.run_id, "entry": str(_HERE / "refs_fmt.py"),
        "baseline_dir": str(BASELINE_DIR), "worker": str(WORKER),
        "fmt_data_root": str(FMT_DATA_ROOT), "fs_raw": FS_RAW, "fs": FS,
        "resample": [RESAMPLE_UP, RESAMPLE_DOWN],
        "expected_test_ids": list(sids), "outputs_root": _PC.OUTPUTS_ROOT,
        "blas_threads": {k: os.environ.get(k) for k in
                         ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                          "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
        "not_via_runpy": "B4 并行 CLI 冲突规避（任务派发口径，design §1 模块布局列 refs_fmt.py 为独立模块）",
    }, ensure_ascii=False, indent=1))

    for m, v in ceilings.items():
        log(f"[split-half] ceiling[{m}] = {v['mean']:.4f} ± {v['sd']:.4f}")
    log(f"[refs] 完成：refs_all.npz ({len(sids)},62,62)×{{gc,gpdc}} + refs_summary.json；"
        f"总耗时 {time.time() - t_start:.0f}s（compare 输入面就绪，B5 后半另派）")


if __name__ == "__main__":
    main()
