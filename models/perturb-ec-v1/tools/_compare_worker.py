#!/usr/bin/env python3
"""_compare_worker —— 比较器子进程 worker（compare_wrap.py 调用；design §1/§5 隔离纪律）。

运行形态：fresh python3 进程，由 compare_wrap.py 以 cwd=<baseline-ec-v1>、env
PYTHONPATH=<baseline-ec-v1> 启动——与 _refs_worker.py 同款形态；头部再防御性剔除
sys.path 中其他 /root/npi 仓库路径（含 '' 相对项——cwd 已钉在 baseline 目录）并断言
`config`/`compare` `__file__` 归属 baseline-ec-v1。**进程内调用不可行**：父进程
（perturb-ec-v1 目录）内 import compare 必被本变体 config.py 遮蔽击穿（compare.py
模块级 `import config as C` 并在 def 期求值 C.BOOTSTRAP_N——design §1 登记），故
一律子进程 + manifest/npz/json 文件面交换，父进程零 import baseline 模块。

职责（单 manifest.json → out_json + out_npz）：
  refs_all.npz（gc/gpdc (10,62,62) + subject_ids）× EC_pert 契约 npz（3 readout ×
  10 例，逐例 subject_ids 对齐断言后按 refs 序堆叠 (10,62,62)）× dist (62,62)
  → 逐 readout 调 baseline compare.run_comparison（逐被试距离偏相关 Spearman ρ +
  未偏 ρ + 单样本 t + 1000× bootstrap CI + 天花板比值；readout 内跨法 holm 值原样
  回传，Holm 族装配/上下文标注由父进程做——worker 不做判定）。

  复用的叶子函数（只读 import，零改写）：compare.load_ec_pert（契约校验：readout
  枚举/orientation/rstar 非负/slope·auc 负值存档取 |·|）、compare.run_comparison
  （offdiag_flat 3782 展平 + partial_spearman + ttest_1samp + bootstrap_ci_mean +
  ceiling_ratio + holm——全 baseline 单源，本 worker 不复写口径常量）。

用法（仅经 compare_wrap.py 调用，不直接手跑）：
  python3 _compare_worker.py <manifest.json>
"""

from __future__ import annotations

import os

# ---- BLAS 护栏（environment.md / _blas_guard 惯例：必须在 numpy 之前；----
# ---- 16 核 cgroup 限 8 线程；setdefault 幂等——环境继承下维持父进程值） ----
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "8")

import json     # noqa: E402
import sys      # noqa: E402
import time     # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

# ---- 路径消毒（isolation 纪律的防御层；正常启动下为空操作；_refs_worker 同款）----
_BASELINE = Path("/root/npi/models/baseline-ec-v1")
_REPO = Path("/root/npi")


def _sanitize_syspath():
    kept = []
    for p in list(sys.path):
        rp = Path(p).resolve() if p else Path.cwd()
        s = str(rp)
        if s.startswith(str(_REPO)) and not s.startswith(str(_BASELINE)):
            continue    # 其他仓库目录：剔除（防遮蔽）
        kept.append(p)
    sys.path[:] = kept
    if str(_BASELINE) not in sys.path:
        sys.path.insert(0, str(_BASELINE))


_sanitize_syspath()

import config as C  # noqa: E402
assert Path(C.__file__).resolve().parent == _BASELINE.resolve(), \
    f"config 遮蔽击穿：解析到 {C.__file__}（应归属 {_BASELINE}）——隔离纪律失败，中止"

import compare as CMP  # noqa: E402
assert Path(CMP.__file__).resolve().parent == _BASELINE.resolve(), \
    f"compare 解析异常：{CMP.__file__}（应归属 {_BASELINE}）"


READOUTS = ("slope", "auc", "rstar")
METHODS = ("gc", "gpdc")


def _jsonable(obj):
    """递归转 JSON 可序列化形态（tuple→list、numpy 标量→python 标量）。"""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def main():
    t0 = time.time()
    manifest = json.loads(Path(sys.argv[1]).read_text())
    refs_npz = manifest["refs_npz"]
    dist_npy = manifest["dist_npy"]
    ceilings = manifest["ceilings"]                      # {"gc": float, "gpdc": float}
    out_json, out_npz = Path(manifest["out_json"]), Path(manifest["out_npz"])

    # ---- 参考侧（observational directed-connectivity references）----
    d = np.load(refs_npz, allow_pickle=True)
    sids_ref = [str(s) for s in np.atleast_1d(d["subject_ids"])]
    S = len(sids_ref)
    refs = {m: np.asarray(d[m], dtype=np.float64) for m in METHODS}
    for m, mats in refs.items():
        assert mats.shape == (S, C.N_CHANNELS, C.N_CHANNELS), \
            f"refs[{m}] 形状 {mats.shape} != ({S},{C.N_CHANNELS},{C.N_CHANNELS})"
        assert np.isfinite(mats).all(), f"refs[{m}] 含非有限值"
    assert len(set(sids_ref)) == S, "refs subject_ids 有重复"

    # ---- 距离控制变量（通道对三维欧氏距离；父进程由 chan_coords_62.npy 算好）----
    dist = np.asarray(np.load(dist_npy), dtype=np.float64)
    assert dist.shape == (C.N_CHANNELS, C.N_CHANNELS), f"dist 形状异常: {dist.shape}"
    assert np.isfinite(dist).all() and float(np.abs(np.diag(dist)).max()) == 0.0, \
        "dist 含非有限值或对角非零"

    # ---- 逐 readout：契约 npz 逐例装载（load_ec_pert 校验）→ 对齐断言 → 堆叠 ----
    results, group_means, neg_counts = {}, {}, {}
    for readout in READOUTS:
        stacked, n_neg = [], 0
        for i, p in enumerate(manifest["ec_paths"][readout]):
            ep = CMP.load_ec_pert(p)                     # 契约校验（orientation 等）
            sid = [str(x) for x in np.atleast_1d(ep["subject_ids"])]
            assert len(sid) == 1 and sid[0] == sids_ref[i], (
                f"{p}: subject_ids {sid} 与 refs[{i}]={sids_ref[i]} 不对齐——逐被试"
                f"配对断言失败")
            ec = np.asarray(ep["ec"], dtype=np.float64)
            assert ec.shape == (C.N_CHANNELS, C.N_CHANNELS), f"{p}: ec 形状 {ec.shape}"
            assert np.isfinite(ec).all(), f"{p}: ec 含非有限值"
            if "ec_sign" in ep:
                n_neg += int((np.asarray(ep["ec_sign"]) < 0).sum())
            stacked.append(ec)
        stacked = np.stack(stacked)                      # (S,62,62) 行=源/列=靶
        group_means[readout] = stacked.mean(axis=0)
        neg_counts[readout] = n_neg
        res = CMP.run_comparison(
            refs, sids_ref,
            {"ec": stacked, "readout": readout, "subject_ids": sids_ref},
            dist, ceilings)
        assert res["ec_pert_granularity"] == "per_subject", \
            f"{readout}: run_comparison 粒度 {res['ec_pert_granularity']} != per_subject"
        results[readout] = res

    # ---- 数值面落盘（npz：逐被试 ρ + 组均矩阵；json：run_comparison 全量）----
    npz_kw = {"subject_ids": sids_ref, "dist": dist,
              **{f"refs_group_mean_{m}": refs[m].mean(axis=0) for m in METHODS},
              **{f"ec_pert_group_mean_{r}": group_means[r] for r in READOUTS}}
    for r in READOUTS:
        for m in METHODS:
            e = results[r]["methods"][m]
            npz_kw[f"rho_partial_{r}_{m}"] = np.asarray(e["rho_partial_per_subject"])
            npz_kw[f"rho_raw_mean_{r}_{m}"] = np.float64(e["rho_raw_mean"])
    np.savez_compressed(out_npz, **npz_kw)

    payload = {
        "ok": True,
        "n_subjects": S,
        "subject_ids": sids_ref,
        "readouts": results,                             # 全量（含 readout 内跨法 holm）
        "negative_value_counts_post_contract_abs": neg_counts,   # slope/auc 负值存档计数
        "config_sources": {
            "config_file": str(Path(C.__file__).resolve()),
            "compare_file": str(Path(CMP.__file__).resolve()),
            "bootstrap_n": int(C.BOOTSTRAP_N), "bootstrap_seed": int(C.BOOTSTRAP_SEED),
            "holm_methods": list(C.HOLM_METHODS),
        },
        "wall_s": round(time.time() - t0, 1),
    }
    out_json.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=1))

    # 单行 JSON 摘要（父进程解析进 run.log；数值事实以 out_json/npz 为准）
    head = {r: {m: round(results[r]["methods"][m]["rho_partial_mean"], 4)
                for m in METHODS} for r in READOUTS}
    print(json.dumps({"ok": True, "rho_partial_mean": head,
                      "wall_s": payload["wall_s"]}))


if __name__ == "__main__":
    main()
