"""data_io —— edra-fmt manifest + prepare_subject（复刻 refs_fmt.prepare_subject 逐式；design §2.1）。

口径（与 models/perturb-ec-v1/refs_fmt.py 逐式同构，公式复刻不 import）：
- 字段断言：filter=="1-40(release)"、pipeline=="EDRA_release_1-40_250Hz"、fs==250、
  truncated==False、shape (62,60001)——异常即 AssertionError（数据契约防漂移）；
- `resample_poly(x, 1, 2, axis=-1)` → (62,30001) @125 Hz（抗混叠 FIR 内置；
  release 已带限 1–40 Hz，**不重复滤波**）；
- 逐通道全记录 z-score（ddof=0，σ 下限 np.maximum(sd, 1e-12)）；
- float64 复原计算（主口径统计全 float64；落盘口径由 run.py 落 float32）。
"""

from __future__ import annotations

import os

# ---- BLAS 护栏（environment.md / baseline _blas_guard 惯例：必须在 numpy 之前；----
# ---- 16 核 cgroup 限 8 线程，setdefault 允许外部环境变量覆盖）                  ----
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "8")

import csv               # noqa: E402
import importlib.util    # noqa: E402
import sys               # noqa: E402
from pathlib import Path # noqa: E402

import numpy as np                   # noqa: E402
from scipy.signal import resample_poly  # noqa: E402

_HERE = Path(__file__).resolve().parent

# ---- 本变体 config 唯一名装载（design §1 四行配方：先注册 sys.modules 后 exec，----
# ---- 不占据 config 模块名；重复装载经 sys.modules 记忆化）                       ----
if "bgec_config" not in sys.modules:
    _SPEC = importlib.util.spec_from_file_location("bgec_config", str(_HERE / "config.py"))
    _C = importlib.util.module_from_spec(_SPEC)
    sys.modules["bgec_config"] = _C          # 先注册后 exec
    _SPEC.loader.exec_module(_C)
import bgec_config as C                      # noqa: E402


def load_manifest() -> dict:
    """fmt manifest.csv → {sid: {group, status, npz_path}}（数据集只读；refs_fmt 同式）。"""
    out = {}
    with open(C.DATA_ROOT / "manifest.csv", newline="") as f:
        for r in csv.DictReader(f):
            sid = int(r["subject_id"])
            out[sid] = {
                "group": r["group"], "status": r["status"],
                "npz_path": str(C.DATA_ROOT / r["group"] / f"{sid}_rest_v4.npz"),
            }
    return out


def prepare_subject(npz_path: str) -> dict:
    """fmt npz → 字段断言 → 125 Hz 抗混叠降采 → 逐通道全记录 z-score（ddof=0）。

    与 refs_fmt.prepare_subject 逐式同构（σ 下限 1e-12 逐字同式），仅任务口径
    独立成变体模块（路径隔离，无运行时复用点）。返回 {x (62,30001) f64 z-scored, meta}。
    """
    d = np.load(npz_path, allow_pickle=True)
    for k, want in C.EXPECTED_FIELDS.items():
        got = str(d[k])
        assert got == want, f"{npz_path}: 字段 {k}={got!r} != 契约 {want!r}"
    assert int(d["fs"]) == C.FS_RAW, f"{npz_path}: fs={int(d['fs'])} != {C.FS_RAW}"
    assert not bool(d["truncated"]), f"{npz_path}: truncated=True"
    x = np.asarray(d["data"], dtype=np.float64)
    assert x.shape == (C.N_CHANNELS, C.N_RAW), f"{npz_path}: 形状 {x.shape} 异常"
    x125 = resample_poly(x, C.RESAMPLE_UP, C.RESAMPLE_DOWN, axis=-1)
    mu = x125.mean(axis=-1, keepdims=True)
    sd = x125.std(axis=-1, keepdims=True)          # ddof=0（全记录统计）
    x125z = (x125 - mu) / np.maximum(sd, 1e-12)
    meta = {
        "subject_id": str(d["subject_id"]), "group": str(d["group"]),
        "filter": str(d["filter"]), "pipeline": str(d["pipeline"]),
        "suffix": str(d["suffix"]), "fs_raw": int(d["fs"]), "fs": C.FS,
        "n_raw": int(x.shape[-1]), "n_125": int(x125.shape[-1]),
        "resample": f"resample_poly(up={C.RESAMPLE_UP},down={C.RESAMPLE_DOWN})",
        "zscore": "per-channel whole-record ddof=0 sigma_floor=1e-12",
    }
    return {"x": x125z, "meta": meta}
