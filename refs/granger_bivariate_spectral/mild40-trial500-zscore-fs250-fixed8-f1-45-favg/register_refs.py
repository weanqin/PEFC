#!/usr/bin/env python3
"""register_refs.py — mild GC 参考注册（任务 10-24-mild-w500-ec-unbiased S0）.

来源：/root/autodl-tmp/mild-depression-gc.zip（用户 2026-10-24 提供，mne-connectivity
状态空间谱 GC 生成，详见 README-source.md）→ staging /root/autodl-tmp/mild_gc_staging/。

注册内容：40 × refs_s{sid}.npz（ec float32 (128,128) mat[a,b]=GC(a→b) 行=源/列=靶
对角 0 + subject_id + group + params 串）+ SHA256SUMS + meta.json + README-source.md。

subject 映射（规划期已验，meta 留档）：zip 按 EEG_ALL.mat 顺序编号 index 0–39（前 20
MDD 后 20 HC）；EEG_ALL 第 i 片与 manifest 文件四锚点（m_1/m_20/h_1/h_20 ↔ sid
1/20/21/40）bitwise 相等 ⇒ sid = index + 1。

校验：shape/dtype/对角 0/无负值 + sid↔group 与 manifest 一致 + 与源 finalize 日志
（min=0 max=0.482）量级复核。可重放（幂等覆盖本目录 npz，不动旧档）。

用法：python3 register_refs.py
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io as sio

ZIP_PATH = Path("/root/autodl-tmp/mild-depression-gc.zip")
STAGING = Path("/root/autodl-tmp/mild_gc_staging/EC_GC")
SRC_NPY = STAGING / "EC_GC_subject.npy"
SRC_README = STAGING / "README.md"
MANIFEST = Path("/root/npi/models/latent-flow-v1/splits/mild_5fold_seed42.json")
OUT_DIR = Path(__file__).resolve().parent

PARAMS = (
    '{"method": "granger_bivariate_spectral (mne-connectivity spectral gc, '
    'state-space, Barnett & Seth 2015)", "order": "gc_n_lags=8 (VAR order fixed)", '
    '"fmin_hz": 1.0, "fmax_hz": 45.0, "faverage": true, "fs": 250.0, '
    '"window": "first 500 samples (2 s) per trial, 30 trials pooled", '
    '"preproc": "per-trial per-channel z-score (raw data unnormalized)", '
    '"data": "mild depression EEG_ALL.mat [1200,128,1500], 40 subjects '
    '(first 20 MDD, last 20 HC), Cz reference"}'
)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    t0 = time.time()
    ec_all = np.load(SRC_NPY)
    assert ec_all.shape == (40, 128, 128) and ec_all.dtype == np.float64, \
        f"source shape/dtype unexpected: {ec_all.shape} {ec_all.dtype}"

    man = json.load(open(MANIFEST))
    groups = {int(k): v["group"] for k, v in man["subjects"].items()}

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

    # ---- 复核回读 + meta ----
    off = ~np.eye(128, dtype=bool)
    stats = {"n": 40, "diag_max": 0.0, "min": 0.0,
             "max": float(ec_all.max()), "offdiag_mean": float(ec_all[:, off].mean())}
    for sid in written:
        z = np.load(OUT_DIR / f"refs_s{sid}.npz", allow_pickle=True)
        assert int(z["subject_id"]) == sid and z["ec"].shape == (128, 128)
        stats["diag_max"] = max(stats["diag_max"], float(np.abs(np.diag(z["ec"])).max()))
    assert stats["diag_max"] == 0.0 and stats["min"] >= 0.0

    meta = {
        "task": "10-24-mild-w500-ec-unbiased S0 (user-provided GC reference, "
                "registered 2026-10-24)",
        "source": {
            "zip": str(ZIP_PATH), "zip_sha256": sha256_file(ZIP_PATH),
            "staging_npy": str(SRC_NPY),
            "generator": "mne_connectivity.spectral_connectivity_epochs(method='gc'), "
                         "mne 1.11.0 / mne-connectivity 0.9.0, Python 3.13 (conda npi)",
            "source_readme": "README-source.md (zip 内原文归档)",
        },
        "method_params": json.loads(PARAMS),
        "subject_mapping": {
            "rule": "sid = EEG_ALL index + 1 (zip 顺序编号)",
            "verification": "EEG_ALL 第 i 片 vs manifest 文件四锚点 bitwise 相等: "
                            "idx0=m_1(sid1) idx19=m_20(sid20) idx20=h_1(sid21) "
                            "idx39=h_20(sid40), maxabsdiff=0.0",
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
    print(f"[register] wrote {len(written)} refs npz + SHA256SUMS + meta.json")
    print(f"[register] sanity: diag_max={stats['diag_max']} min={stats['min']:.4f} "
          f"max={stats['max']:.4f} offdiag_mean={stats['offdiag_mean']:.5f} "
          f"(source finalize log: min=0 max=0.482)")
    print(f"[register] elapsed {time.time() - t0:.1f}s -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
