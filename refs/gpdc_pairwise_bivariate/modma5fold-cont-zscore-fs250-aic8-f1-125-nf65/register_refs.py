#!/usr/bin/env python3
"""register_refs.py — MODMA 5 折 gpdc baseline 注册（可重放）.

任务 10-29-modma-baseline-refs。源 = 旧库原版矩阵（用户 2026-10-30 下载的
modma-gc-ec.zip 解包树）。本脚本从源树拷贝 fold0-4 的 subject_*.json（保留
原文件名），生成 SHA256SUMS 与 meta.json。重放：`python3 register_refs.py
[--src <解包根>]`。同型脚本另两份在 granger_bivariate_spectral 与
symbolic_transfer_entropy 档。
"""
from __future__ import annotations
import argparse, hashlib, json, shutil, sys, time
from pathlib import Path

METHOD_DIR = "gpd"          # 源树内方法子目录名
REFS_NAME = "gpdc_pairwise_bivariate"
N_FOLDS = 5
THIS = Path(__file__).resolve().parent


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/root/autodl-tmp/modma-gc-ec",
                    help="解包根（其下应有 baseline_connectivity_fold{0..4}/%s/）" % METHOD_DIR)
    a = ap.parse_args()
    src_root = Path(a.src)
    assert src_root.exists(), f"src not found: {src_root}"

    entries, per_fold = [], {}
    for f in range(N_FOLDS):
        sdir = src_root / f"baseline_connectivity_fold{f}" / METHOD_DIR
        files = sorted(sdir.glob("subject_*.json"))
        assert files, f"empty source dir: {sdir}"
        dst = THIS / f"fold{f}"
        dst.mkdir(parents=True, exist_ok=True)
        mapping = {}
        for fp in files:
            shutil.copy2(fp, dst / fp.name)
            entries.append((f"fold{f}/{fp.name}", sha256(dst / fp.name)))
            mapping[fp.name] = json.load(open(fp))["subject_file"]
        per_fold[f"fold{f}"] = mapping

    with open(THIS / "SHA256SUMS", "w") as fh:
        for rel, h in entries:
            fh.write(f"{h}  {rel}\n")

    p0 = json.load(open(src_root / "baseline_connectivity_fold0" / METHOD_DIR / "subject_0.json"))
    meta = {
        "registered_by": "10-29-modma-baseline-refs（用户 2026-10-30 指令）",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "method": p0["method"],
        "method_params": p0["method_params"],
        "source": {
            "zip": "/root/autodl-tmp/modma-gc-ec.zip（用户下载的旧库原版矩阵）",
            "extracted_root": str(src_root),
            "original_generation": "旧库 outputs/baseline_connectivity_fold{0..4}（2026-07-18/07-23）",
            "identity_evidence": ("与 gc 档同包同源；gc fold0/subject_0 已三方逐位验证 "
                                  "（zip = 旧库残留 = 10-27 现算），本方法沿用包级溯源"),
        },
        "direction_convention": "connectivity_matrix[a,b] = a→b（行=源/列=靶），128×128",
        "folds": {k: {"n": len(v), "files": v} for k, v in per_fold.items()},
        "n_files": len(entries),
        "checksum_file": "SHA256SUMS",
        "notes": ("MODMA resting 53 例 cohort 的 5 折 test 子集；gpd 档 fold2-4 各 11 例、"
                  "fold0-1 各 10 例（以 folds 映射为准）。fs250 全长连续、逐通道 z-score。"),
    }
    (THIS / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n")
    print(f"[register] {REFS_NAME}/{THIS.name}: {len(entries)} files, "
          f"{ {k: v['n'] for k, v in meta['folds'].items()} } -> SHA256SUMS + meta.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
