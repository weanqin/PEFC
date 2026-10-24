#!/usr/bin/env python3
"""register_refs.py — mild GC 参考 v2（1–125 Hz AIC 选阶）注册（任务 10-24-exitgnn-ec-unbiased S0）.

来源：/root/autodl-tmp/EC_GC_1_125.zip（用户 2026-10-24 拍板提供，AskUserQuestion
「双GC对照，新加的GC对照在autodl-tmp/EC_GC_1_125」）→ staging
/root/autodl-tmp/ecgc_1_125_staging/（仅取 EC_GC_subject.npy + README.md）。

与 v1（mild40-trial500-zscore-fs250-fixed8-f1-45-favg，先例）的差异：频段 1–125 Hz
（v1 1–45）+ VAR 阶数 AIC 逐被试选阶 4–25（v1 固定 8；池化二元 VAR 完全模型 OLS-AIC，
统一 500−25 截断）。

注册内容：40 × refs_s{sid}.npz（ec float32 (128,128) mat[a,b]=GC(a→b) 行=源/列=靶
对角 0 + subject_id + group + params 串）+ SHA256SUMS + meta.json + README-source.md。

subject 映射（本脚本内**重做**四锚点 bitwise 复验，不沿用 v1 规划期验证——design §1.2）：
zip 按 EEG_ALL.mat 顺序编号 index 0–39（前 20 MDD 后 20 HC）；EEG_ALL 第 idx 片
[idx*30:(idx+1)*30] 与 manifest 数据文件 concat(x_p,x_f,axis=2) 四锚点
idx∈{0,19,20,39} ↔ {m_1, m_20, h_1, h_20}（h_20 即 sid 40）逐位相等。

校验：shape/dtype/对角 0/无负值/无 NaN + sid↔group 与 manifest 一致 + 量级复核 vs
源 README QC（mean=0.0475, max=0.568）。可重放（幂等覆盖本目录 npz，不动旧档）。

用法：/root/miniconda3/bin/python register_refs.py 2>&1 | tee register.log
"""
from __future__ import annotations

import os

# --- BLAS 守卫：numpy 导入前（npi 环境契约）---
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "8")

import hashlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import scipy.io as sio  # noqa: E402

ZIP_PATH = Path("/root/autodl-tmp/EC_GC_1_125.zip")
STAGING = Path("/root/autodl-tmp/ecgc_1_125_staging")
SRC_NPY = STAGING / "EC_GC_subject.npy"
SRC_README = STAGING / "README.md"
EEG_ALL = Path("/root/autodl-tmp/Mild Depression/EEG_ALL.mat")
MANIFEST = Path("/root/npi/models/latent-flow-v1/splits/mild_5fold_seed42.json")
OUT_DIR = Path(__file__).resolve().parent

PARAMS = (
    '{"method": "granger_bivariate_spectral (mne-connectivity spectral gc, '
    'state-space, Barnett & Seth 2015)", "order": "AIC-selected per subject '
    '(4-25); pooled bivariate VAR full-model OLS-AIC, unified 500-25 truncation", '
    '"fmin_hz": 1.0, "fmax_hz": 125.0, "faverage": true, "fs": 250.0, '
    '"window": "first 500 samples (2 s) per trial, 30 trials pooled", '
    '"preproc": "per-trial per-channel z-score (raw data unnormalized)", '
    '"data": "mild depression EEG_ALL.mat [1200,128,1500], 40 subjects '
    '(first 20 MDD, last 20 HC), Cz reference"}'
)

# 四锚点：EEG_ALL 顺序 idx ↔ manifest 文件（sid = idx + 1）
ANCHORS = [(0, "m_1", 1), (19, "m_20", 20), (20, "h_1", 21), (39, "h_20", 40)]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_anchors(groups: dict) -> str:
    """四锚点 bitwise 复验（design §1.2：重做不沿用 v1 规划期验证）。"""
    eeg = sio.loadmat(EEG_ALL, variable_names=["EEG"])["EEG"]
    assert eeg.shape == (1200, 128, 1500) and eeg.dtype == np.float64, \
        f"EEG_ALL shape/dtype unexpected: {eeg.shape} {eeg.dtype}"
    report = []
    for idx, stem, sid in ANCHORS:
        f = EEG_ALL.parent / f"{stem}.mat"
        assert f.exists(), f"anchor file missing: {f}"
        d = sio.loadmat(f, variable_names=["x_p", "x_f"])
        recon = np.concatenate([d["x_p"], d["x_f"]], axis=2)
        assert recon.shape == (30, 128, 1500), f"{stem}: recon {recon.shape}"
        maxabs = float(np.max(np.abs(eeg[idx * 30:(idx + 1) * 30] - recon)))
        assert maxabs == 0.0, f"anchor idx{idx}/{stem}: maxabsdiff={maxabs} != 0"
        assert groups[sid] == ("MDD" if idx < 20 else "HC"), \
            f"sid {sid}: group mismatch vs manifest ({groups[sid]})"
        report.append(f"idx{idx}={stem}(sid{sid}) maxabsdiff=0.0")
    return "; ".join(report)


def main() -> int:
    t0 = time.time()
    ec_all = np.load(SRC_NPY)
    assert ec_all.shape == (40, 128, 128) and ec_all.dtype == np.float64, \
        f"source shape/dtype unexpected: {ec_all.shape} {ec_all.dtype}"
    assert not np.isnan(ec_all).any(), "source contains NaN"

    man = json.load(open(MANIFEST))
    groups = {int(k): v["group"] for k, v in man["subjects"].items()}

    anchor_report = verify_anchors(groups)

    # ---- 注册（sid = index + 1）----
    written = []
    for idx in range(40):
        sid = idx + 1
        ec = ec_all[idx].astype(np.float32)
        assert ec.shape == (128, 128)
        assert np.all(np.diag(ec) == 0), f"idx {idx}: diagonal nonzero"
        assert ec.min() >= 0.0, f"idx {idx}: negative GC value {ec.min()}"
        assert groups[sid] == ("MDD" if idx < 20 else "HC"), \
            f"sid {sid}: group mismatch vs manifest ({groups[sid]})"
        np.savez(
            OUT_DIR / f"refs_s{sid}.npz",
            ec=ec,
            subject_id=np.int64(sid),
            group=np.str_(groups[sid]),
            params=np.str_(PARAMS),
        )
        written.append(sid)

    shutil.copyfile(SRC_README, OUT_DIR / "README-source.md")

    # ---- SHA256SUMS（40 npz + README-source）----
    lines = []
    for f in sorted(OUT_DIR.glob("refs_s*.npz")):
        lines.append(f"{sha256_file(f)}  {f.name}")
    lines.append(f"{sha256_file(OUT_DIR / 'README-source.md')}  README-source.md")
    (OUT_DIR / "SHA256SUMS").write_text("\n".join(lines) + "\n")

    # ---- 复核回读 + sanity + meta ----
    off = ~np.eye(128, dtype=bool)
    stats = {"n": 40, "diag_max": 0.0, "min": float(ec_all.min()),
             "max": float(ec_all.max()), "mean": float(ec_all.mean()),
             "offdiag_mean": float(ec_all[:, off].mean())}
    for sid in written:
        z = np.load(OUT_DIR / f"refs_s{sid}.npz", allow_pickle=True)
        assert int(z["subject_id"]) == sid and z["ec"].shape == (128, 128)
        stats["diag_max"] = max(stats["diag_max"], float(np.abs(np.diag(z["ec"])).max()))
    assert stats["diag_max"] == 0.0 and stats["min"] >= 0.0
    # 量级复核 vs 源 README QC（mean=0.0475, max=0.568）——防错包/版本错位
    assert abs(stats["mean"] - 0.0475) < 2e-3, \
        f"mean {stats['mean']:.4f} vs README QC 0.0475 超容差"
    assert abs(stats["max"] - 0.568) < 5e-3, \
        f"max {stats['max']:.4f} vs README QC 0.568 超容差"

    meta = {
        "task": "10-24-exitgnn-ec-unbiased S0 (user-provided GC reference v2, "
                "registered 2026-10-24)",
        "source": {
            "zip": str(ZIP_PATH), "zip_sha256": sha256_file(ZIP_PATH),
            "staging_npy": str(SRC_NPY),
            "generator": "mne_connectivity spectral gc (state-space), per source "
                         "README; AIC per-subject order selection pipeline as "
                         "described in README-source.md",
            "source_readme": "README-source.md (zip 内原文归档)",
        },
        "method_params": json.loads(PARAMS),
        "caliber_note": (
            "数据有效带宽 ≈0.5–50 Hz（低通 ≈50 Hz，Cz 参考）；51–125 Hz 频段在数据中"
            "已衰减 −24 dB 以下 ⇒ 1–125 Hz 积分值主要由 50 Hz 以下贡献（源 README 原文"
            "登记）。与 v1（fixed8-f1-45 档）逐被试矩阵 Pearson 相关 median=0.148 "
            "(min=-0.024, max=0.672, n=40)——两参考结构差异大，双对照各自独立登记，"
            "不做优劣判断。"),
        "subject_mapping": {
            "rule": "sid = EEG_ALL index + 1 (zip 顺序编号)",
            "verification": f"四锚点 bitwise 复验（本脚本重做）: {anchor_report}",
            "groups": "index 0-19 MDD (sid 1-20), index 20-39 HC (sid 21-40)",
            "manifest": str(MANIFEST) + " (SHA256 47987909c04348e57d98e77e166867"
                                        "8a7885109336df90bd0a5a635f865d881b)",
        },
        "files": [f"refs_s{sid}.npz" for sid in written] + [
            "README-source.md", "SHA256SUMS", "register_refs.py", "register.log",
            "meta.json"],
        "sanity": stats,
        "registered": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n")

    # ---- 日志（自挂自记：本文件 stdout 由调用侧重定向到 register.log）----
    print(f"[register] anchors: {anchor_report}")
    print(f"[register] wrote {len(written)} refs npz + SHA256SUMS + meta.json")
    print(f"[register] sanity: diag_max={stats['diag_max']} min={stats['min']:.4f} "
          f"max={stats['max']:.4f} mean={stats['mean']:.4f} "
          f"offdiag_mean={stats['offdiag_mean']:.5f} "
          f"(source README QC: mean=0.0475 max=0.568)")
    print(f"[register] elapsed {time.time() - t0:.1f}s -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
