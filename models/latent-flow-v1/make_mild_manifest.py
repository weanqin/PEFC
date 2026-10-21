#!/usr/bin/env python3
"""make_mild_manifest.py — Mild Depression 数据集 5-fold 分层划分 manifest 生成（变体层）.

任务 10-21-latent-flow-mild-train（design §2.2，S1）。生成一次即钉死：
**已存在即拒跑**（防重复生成改口径）；训练入口 train_mild_depress.py 只读
复用并全量断言重放，不重新划分。

划分算法（RNG 调用序列钉死：每群恰一次 permutation 调用、零追加随机调用）：
  - 例集发现：sorted(glob('h_*.mat')) / sorted(glob('m_*.mat'))
    （必须排序后洗牌，glob 目录序不可作确定性来源）；
  - MDD、HC 文件各自 np.random.RandomState(42).permutation(20) 独立洗牌；
  - 连续 5 段 bounds [0,4,8,12,16,20]；
  - fold i 的 test = 两群第 i 段各 4 例（4+4=8）；
  - val = 两群第 (i+1)%5 段的前 2 例（各折轮转，2+2=4）；
  - train = 余 28 例。

生成时全量断言（失败即拒写）：40 例 HC20/MDD20、subject_id 互异、每例
30 trial、x_p (30,128,1125) / x_f (30,128,375)、fs=250、label 语义
（m_i=subject_id i / label_code 1；h_i=subject_id 20+i / label_code 0）、
逐折 28/4/8、三集无交且并集=全例、5 折 test 互斥覆盖全 40 例。

产物：models/latent-flow-v1/splits/mild_5fold_seed42.json（file 字段绝对路径）。

Run:  python3 /root/npi/models/latent-flow-v1/make_mild_manifest.py
"""
from __future__ import annotations

import os

# --- BLAS 守卫：必须在任何 numpy/scipy/torch 导入之前（npi 环境约束） ---
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

VARIANT_DIR = Path(__file__).resolve().parent
DATA_ROOT_DEFAULT = Path("/root/autodl-tmp/Mild Depression")
OUT_DEFAULT = VARIANT_DIR / "splits" / "mild_5fold_seed42.json"

# --- 契约常量（.trellis/spec/data/mild-depression.md） ---
N_CHANNELS = 128
TRIALS = 30
X_P_LEN = 1125
X_F_LEN = 375
FS = 250
N_FOLDS = 5
PER_GROUP = 20
SEG_BOUNDS = [0, 4, 8, 12, 16, 20]   # 连续 5 段，每段 4 例/群


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-root", type=str, default=str(DATA_ROOT_DEFAULT))
    p.add_argument("--out", type=str, default=str(OUT_DEFAULT))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def _check_subject_file(path: Path, group: str, sid: int):
    """逐例字段断言（生成与训练入口重放共用同口径）。"""
    m = sio.loadmat(path)
    xp, xf = m["x_p"], m["x_f"]
    assert xp.shape == (TRIALS, N_CHANNELS, X_P_LEN), \
        f"{path}: x_p {xp.shape} != ({TRIALS},{N_CHANNELS},{X_P_LEN})"
    assert xf.shape == (TRIALS, N_CHANNELS, X_F_LEN), \
        f"{path}: x_f {xf.shape} != ({TRIALS},{N_CHANNELS},{X_F_LEN})"
    assert m["fs"].item() == FS, f"{path}: fs {m['fs'].item()} != {FS}"
    label = str(np.asarray(m["label"]).ravel()[0])
    assert label == group, f"{path}: label {label!r} != {group!r}"
    code = 1 if group == "MDD" else 0
    assert m["label_code"].item() == code, \
        f"{path}: label_code {m['label_code'].item()} != {code}"
    assert m["subject_id"].item() == sid, \
        f"{path}: subject_id {m['subject_id'].item()} != {sid}"
    assert m["x_p_start_sample"].item() == 0 and \
        m["x_f_start_sample"].item() == X_P_LEN, \
        f"{path}: x_p/x_f start sample mismatch (trial 内 3:1 连续相接被破坏)"
    return {"trials": int(xp.shape[0]), "label_code": code}


def _discover(data_root: Path):
    """sorted(glob) 发现 + 文件名→(group, sid) 解析。"""
    hc = sorted(glob.glob(str(data_root / "h_*.mat")))
    mdd = sorted(glob.glob(str(data_root / "m_*.mat")))
    assert len(hc) == PER_GROUP, f"HC files != {PER_GROUP}: {len(hc)}"
    assert len(mdd) == PER_GROUP, f"MDD files != {PER_GROUP}: {len(mdd)}"
    out = {}
    for group, files in (("MDD", mdd), ("HC", hc)):
        for f in files:
            p = Path(f)
            prefix, num = p.stem[0], int(p.stem.split("_")[1])
            expected_group = "MDD" if prefix == "m" else "HC"
            assert expected_group == group, f"unexpected file name: {p}"
            sid = num if group == "MDD" else PER_GROUP + num
            out[sid] = (group, p)
    return out, mdd, hc


def make_splits(subjects: dict, mdd_files: list, hc_files: list, seed: int):
    """划分算法（design §2.2；RNG 调用序列钉死：每群恰一次 permutation）。"""
    mdd_ids = [1 + i for i in range(PER_GROUP)]                    # sid 1..20
    hc_ids = [PER_GROUP + 1 + i for i in range(PER_GROUP)]         # sid 21..40
    id_by_file = {str(v[1]): k for k, v in subjects.items()}

    def seg_ids(files):
        perm = np.random.RandomState(seed).permutation(PER_GROUP)  # 恰一次调用
        return [[id_by_file[str(Path(files[j]))] for j in perm[b0:b1]]
                for b0, b1 in zip(SEG_BOUNDS[:-1], SEG_BOUNDS[1:])]

    mdd_segs = seg_ids(mdd_files)
    hc_segs = seg_ids(hc_files)

    folds = []
    for i in range(N_FOLDS):
        test = sorted(mdd_segs[i] + hc_segs[i])                     # 4+4
        val_seg = (i + 1) % N_FOLDS
        val = sorted(mdd_segs[val_seg][:2] + hc_segs[val_seg][:2])  # 2+2
        train = sorted(set(mdd_ids + hc_ids) - set(test) - set(val))
        folds.append({"fold": i, "train": train, "val": val, "test": test})
    return folds


def assert_folds(folds: list, all_ids: list):
    """跨折全量断言（生成与训练入口重放共用同口径）。"""
    all_set = set(all_ids)
    assert len(all_set) == 2 * PER_GROUP, f"total subjects != 40: {len(all_set)}"
    test_union = []
    for f in folds:
        tr, va, te = set(f["train"]), set(f["val"]), set(f["test"])
        assert len(f["train"]) == 28, f"fold{f['fold']} train != 28: {len(f['train'])}"
        assert len(f["val"]) == 4, f"fold{f['fold']} val != 4: {len(f['val'])}"
        assert len(f["test"]) == 8, f"fold{f['fold']} test != 8: {len(f['test'])}"
        assert not (tr & va or tr & te or va & te), f"fold{f['fold']} overlap"
        assert tr | va | te == all_set, f"fold{f['fold']} union != all 40"
        n_mdd_te = sum(1 for s in f["test"] if s <= PER_GROUP)
        assert n_mdd_te == 4 and len(f["test"]) - n_mdd_te == 4, \
            f"fold{f['fold']} test not 4+4 stratified"
        test_union.extend(f["test"])
    assert len(test_union) == len(set(test_union)) == 2 * PER_GROUP, \
        "5-fold test sets not mutually exclusive / not covering all 40"


def main() -> int:
    args = parse_args()
    out_path = Path(args.out)
    if out_path.exists():
        raise SystemExit(f"[refuse] manifest already exists: {out_path}\n"
                         f"  划分一次生成即钉死；如确需重生成，先人工删除并重跑本脚本。")

    data_root = Path(args.data_root)
    subjects, mdd_files, hc_files = _discover(data_root)

    # --- 逐例字段断言（40 例全量） ---
    subj_record = {}
    for sid in sorted(subjects):
        group, path = subjects[sid]
        subj_record[str(sid)] = _check_subject_file(path, group, sid)
    n_mdd = sum(1 for v in subjects.values() if v[0] == "MDD")
    n_hc = len(subjects) - n_mdd
    assert n_mdd == PER_GROUP and n_hc == PER_GROUP, f"HC/MDD != 20/20: {n_hc}/{n_mdd}"
    assert len({s for s in subjects}) == 2 * PER_GROUP, "subject_id not unique"

    # --- 划分 + 跨折断言 ---
    all_ids = sorted(subjects)
    folds = make_splits(subjects, mdd_files, hc_files, args.seed)
    assert_folds(folds, all_ids)

    manifest = {
        "dataset": "mild_depress",
        "seed": args.seed,
        "n_folds": N_FOLDS,
        "split_algorithm": (
            "per-group sorted(glob) then RandomState(seed).permutation(20) "
            "(exactly one call per group); segments bounds [0,4,8,12,16,20]; "
            "fold i test = both groups' segment i (4+4); val = first 2 of both "
            "groups' segment (i+1)%5; train = remaining 28"),
        "subjects": {str(sid): {"group": subjects[sid][0],
                                "file": str(subjects[sid][1].resolve()),
                                "label_code": subj_record[str(sid)]["label_code"]}
                     for sid in all_ids},
        "folds": folds,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2) + "\n")

    sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
    print(f"[manifest] wrote {out_path}")
    print(f"[manifest] sha256={sha}")
    print(f"[manifest] 40 subjects (HC {n_hc} / MDD {n_mdd}); all per-file "
          f"asserts passed ({TRIALS} trials, x_p {X_P_LEN}/x_f {X_F_LEN}, fs {FS})")
    for f in folds:
        print(f"[fold {f['fold']}] train={len(f['train'])} val={f['val']} "
              f"test={f['test']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
