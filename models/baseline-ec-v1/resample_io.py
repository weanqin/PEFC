"""resample_io —— EDRA npz 读取/字段核对、500→125 Hz 抗混叠降采、逐通道 z、缓存与通道距离。

口径（design §2）：不重复滤波（断言 filter 字段）；z-score 逐被试逐通道全记录统计（ddof=0）；
缓存 float32；通道距离 = chan_coords_62.npy 三维欧氏。
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402


def load_manifest(data_root: Path = C.DATA_ROOT,
                  subject_filter: str = C.SUBJECT_FILTER) -> List[Dict]:
    """manifest.csv → 受试者条目（轻量本地实现，不引入 v1 data.py 的 torch 依赖）。"""
    rows = list(csv.DictReader(open(data_root / "manifest.csv")))
    if subject_filter == "status_ok":
        rows = [r for r in rows if r["status"] == "ok"]
    elif subject_filter is not None:
        rows = [r for r in rows if r["status"] == subject_filter]
    out = []
    for r in rows:
        out.append({
            "subject_id": r["subject_id"],
            "group": r["group"],
            "npz_path": str(data_root / r["group"] / f"{r['subject_id']}_rest_v4.npz"),
        })
    out.sort(key=lambda e: (e["group"], int(e["subject_id"])))
    return out


def load_and_prepare(npz_path: str, check_filter: bool = True) -> Dict:
    """读取 npz → 字段断言 → 降采 125 Hz → 逐通道 z。返回 {x125z (62,21250) f64, meta}。"""
    d = np.load(npz_path, allow_pickle=True)
    filt = str(d["filter"])
    if check_filter and filt != C.EXPECTED_FILTER:
        raise ValueError(
            f"{npz_path}: filter 字段 {filt!r} != 预期 {C.EXPECTED_FILTER!r}；"
            f"按契约不得静默跳过，须先核对预处理口径（edra.md §3）")
    x = np.asarray(d["data"], dtype=np.float64)          # (62, 85000)
    assert x.shape == (C.N_CHANNELS, 85000), f"数据形状异常: {x.shape}"
    x125 = resample_poly(x, C.RESAMPLE_UP, C.RESAMPLE_DOWN, axis=-1)  # 抗混叠 FIR 内置
    mu = x125.mean(axis=-1, keepdims=True)
    sd = x125.std(axis=-1, keepdims=True)
    x125z = (x125 - mu) / np.maximum(sd, 1e-12)
    meta = {
        "subject_id": str(d["subject_id"]), "group": str(d["group"]),
        "filter": filt, "ref_chain": str(d["ref_chain"]),
        "fs_raw": int(d["fs"]), "fs": C.FS,
        "n_raw": int(x.shape[-1]), "n_125": int(x125.shape[-1]),
    }
    return {"x": x125z, "meta": meta}


def cache_subject(prep: Dict, run_data_dir: Path) -> Path:
    run_data_dir.mkdir(parents=True, exist_ok=True)
    path = run_data_dir / f"{prep['meta']['subject_id']}.npz"
    np.savez_compressed(path, x=prep["x"].astype(np.float32), **prep["meta"])
    return path


def load_cached(path: Path) -> Dict:
    d = np.load(path, allow_pickle=True)
    meta = {k: str(d[k]) for k in ("subject_id", "group", "filter", "ref_chain")}
    meta.update({k: int(d[k]) for k in ("fs_raw", "fs", "n_raw", "n_125")})
    return {"x": np.asarray(d["x"], dtype=np.float64), "meta": meta}


def channel_distance(data_root: Path = C.DATA_ROOT) -> np.ndarray:
    """62×62 三维欧氏距离矩阵（米制；偏相关控制向量用）。"""
    coords = np.load(data_root / "chan_coords_62.npy")   # (62,3) float64
    assert coords.shape == (C.N_CHANNELS, 3)
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff ** 2).sum(-1))


def stage_resample(run_dir: Path, subjects: List[Dict], log=print) -> List[Path]:
    """resample 阶段：46 例 → data/*.npz。"""
    data_dir = run_dir / "data"
    paths = []
    for i, e in enumerate(subjects):
        prep = load_and_prepare(e["npz_path"])
        paths.append(cache_subject(prep, data_dir))
        log(f"[resample {i+1}/{len(subjects)}] {e['subject_id']} "
            f"({e['group']}) fs {prep['meta']['fs_raw']}→{prep['meta']['fs']} "
            f"n={prep['meta']['n_125']} ref={prep['meta']['ref_chain']}")
    summary = [load_cached(p)["meta"] for p in paths]
    (data_dir / "resample_meta.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    return paths
