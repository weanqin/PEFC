"""test_data.py — edra fold0 dataloader 单测（脚本式，零 pytest 依赖）。

断言（implement.md 1.3）：
① 受试者不跨折且分层平衡；同一 seed 划分可复现
② z-score 统计仅来自训练折（与仅训练折受试者的独立重算逐元素一致）
③ 输出形状 (B,62,384)/(B,62,128)，float32
④ qc 过滤与窗数口径
运行：python3 models/gpfm-mamba-v1/tests/test_data.py（退出码 0 = 通过）
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as C  # noqa: E402
import data as D  # noqa: E402


def main() -> None:
    cfg = C.Config()
    entries = D.load_manifest(cfg.data_root, subject_filter="status_ok")
    n_hc = sum(1 for e in entries if e["group"] == "HC")
    n_mdd = sum(1 for e in entries if e["group"] == "MDD")
    print(f"[status_ok 过滤（用户 2026-10-12 拍板 b）] 总受试者 {len(entries)}（HC {n_hc} / MDD {n_mdd}）")
    assert len(entries) == 46 and n_hc == 24 and n_mdd == 22, "status_ok 过滤数量与 manifest 实测不符"

    # ① 受试者级分层划分：不跨折、平衡、可复现
    folds = D.stratified_subject_folds(entries, n_folds=5, seed=cfg.split_seed)
    all_ids = [e["subject_id"] for f in folds for e in f]
    assert sorted(all_ids) == sorted(e["subject_id"] for e in entries), "划分覆盖不全/重复"
    for fi, f in enumerate(folds):
        gs = [e["group"] for e in f]
        print(f"  fold{fi}: {len(f)} 受试者 (HC {gs.count('HC')} / MDD {gs.count('MDD')})")
    hc_counts = sorted(sum(1 for e in f if e["group"] == "HC") for f in folds)
    mdd_counts = sorted(sum(1 for e in f if e["group"] == "MDD") for f in folds)
    assert hc_counts == [4, 5, 5, 5, 5] and mdd_counts == [4, 4, 4, 5, 5], "分层不平衡"
    folds2 = D.stratified_subject_folds(entries, n_folds=5, seed=cfg.split_seed)
    assert [[e["subject_id"] for e in f] for f in folds] == [
        [e["subject_id"] for e in f] for f in folds2
    ], "同一 seed 划分不可复现"
    folds3 = D.stratified_subject_folds(entries, n_folds=5, seed=cfg.split_seed + 1)
    assert [[e["subject_id"] for e in f] for f in folds] != [
        [e["subject_id"] for e in f] for f in folds3
    ], "不同 seed 应给出不同划分（置换生效检查）"

    train, val, test = D.split_fold(
        entries, n_folds=cfg.n_folds, fold=0, split_seed=cfg.split_seed,
        n_val_subjects=cfg.n_val_subjects, val_subject_seed=cfg.val_subject_seed,
    )
    ids_train = {e["subject_id"] for e in train}
    ids_val = {e["subject_id"] for e in val}
    ids_test = {e["subject_id"] for e in test}
    assert ids_train.isdisjoint(ids_val) and ids_train.isdisjoint(ids_test) and ids_val.isdisjoint(ids_test), \
        "受试者跨折泄漏"
    assert len(val) == 2, f"内部验证应 2 例受试者，实际 {len(val)}"
    print(f"  train {len(train)} 受试者 {sorted(ids_train)}")
    print(f"  val   {len(val)} 受试者 {sorted(ids_val)}")
    print(f"  test  {len(test)} 受试者 {sorted(ids_test)}（fold0 held-out）")

    # ② z-score 统计仅来自训练折：独立重算对照
    mean, std = D.compute_channel_stats(train)
    mean2, std2 = D.compute_channel_stats(train)  # 确定性
    assert np.array_equal(mean, mean2) and np.array_equal(std, std2)
    # 手工独立重算（仅 train 受试者）
    s = np.zeros(62)
    ss = np.zeros(62)
    n = 0
    for e in train:
        d = np.load(e["npz_path"])["data"].astype(np.float64)
        s += d.sum(axis=1)
        ss += (d ** 2).sum(axis=1)
        n += d.shape[1]
    ref_mean = s / n
    ref_std = np.sqrt(np.maximum(ss / n - ref_mean ** 2, 1e-12))
    assert np.allclose(mean, ref_mean, rtol=1e-10, atol=1e-10), "均值统计与独立重算不一致"
    assert np.allclose(std, ref_std, rtol=1e-10, atol=1e-10), "std 统计与独立重算不一致"
    # 统计中不含 val/test 受试者：加入任一 val 受试者后数值应改变
    mean_v, _ = D.compute_channel_stats(train + val[:1])
    assert not np.allclose(mean, mean_v, rtol=1e-6), "统计疑似包含非训练折受试者"
    print(f"  z-score 统计：mean∈[{mean.min():.4f},{mean.max():.4f}] std∈[{std.min():.4f},{std.max():.4f}]（仅训练折 {len(train)} 例）")

    # ③ 数据集与形状（含划分固化校验）
    ds = D.build_datasets(cfg)
    mpath = ds["split_manifest"]
    assert mpath and os.path.exists(mpath), "划分固化 manifest 未落盘"
    import json as _json
    with open(mpath, encoding="utf-8") as f:
        man = _json.load(f)
    assert man["train"] == sorted(ids_train) and man["val"] == sorted(ids_val) and man["test"] == sorted(ids_test), \
        "固化 manifest 与实际划分不一致"
    ds2 = D.build_datasets(cfg)  # 二次构建：与固化 manifest 校验一致（不报错）
    assert [e["subject_id"] for e in ds2["train_entries"]] == [e["subject_id"] for e in ds["train_entries"]]
    print(f"  划分已固化: {mpath}")
    tr, va, te = ds["train"], ds["val"], ds["test"]
    win = cfg.past_len + cfg.future_len
    n_exp = (85000 - win) // cfg.stride + 1
    assert all(n == n_exp for n in tr.n_windows_per_subject()), "每受试者窗数与公式不符"
    print(f"[窗数] 每受试者 {n_exp}；train {len(tr)} / val {len(va)} / test {len(te)} 窗")
    assert len(tr) == len(train) * n_exp and len(te) == len(test) * n_exp

    loader = D.make_loader(tr, batch_size=8, shuffle=False, seed=cfg.seed, epoch=0, num_workers=0)
    xp, xf = next(iter(loader))
    assert xp.shape == (8, 62, 384) and xf.shape == (8, 62, 128), f"形状错误 {xp.shape} {xf.shape}"
    assert xp.dtype == torch.float32 and xf.dtype == torch.float32
    # 窗内容与原数组逐元素一致（预处理在 dataloader 内正确执行）
    si, s0 = tr.windows[0]
    arr = tr.arrays[si]
    assert np.array_equal(xp[0].numpy(), arr[:, s0 : s0 + 384])
    assert np.array_equal(xf[0].numpy(), arr[:, s0 + 384 : s0 + 512])
    # 训练折（z-score 口径来源）数据均值≈0 std≈1
    flat = np.concatenate([a.ravel() for a in tr.arrays])
    assert abs(flat.mean()) < 0.02 and abs(flat.std() - 1.0) < 0.02, "训练折 z-score 后不是 (0,1) 尺度"
    # ④ return_group 供分层核对
    ds_g = D.EdraWindowsDataset(
        test, ds["channel_mean"], ds["channel_std"],
        past_len=cfg.past_len, future_len=cfg.future_len, stride=cfg.stride, return_group=True,
    )
    _, _, g = ds_g[0]
    assert g in ("HC", "MDD")
    print("[PASS] test_data 全部断言通过")


if __name__ == "__main__":
    main()
