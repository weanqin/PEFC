"""edra fold0 dataloader（data-pipeline 契约：全部预处理在 dataloader 内完成）。

- 受试者级 5-fold 分层划分（MDD/HC，seed=42），study1 前置验证固定 fold0
- 受试者过滤口径 subject_filter='status_ok'（2026-10-12 用户门禁拍板 b，46 例 HC24/MDD22）
- 划分固化（用户要求）：首次划分落 manifest 到 splits/fold{N}_{filter}_seed{S}.json，
  之后每次构建校验一致，漂移（seed/口径/清单变化）立即报错
- z-score 逐通道统计，仅用训练折（已剔除内部验证受试者）估计
- 窗切分 [past=384, future=128] stride=64；输出 X_past (B,62,384) / X_future (B,62,128) float32
- 模型侧零变换（spec 纪律）
"""

from __future__ import annotations

import csv
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def load_manifest(data_root: str, subject_filter: str = "status_ok") -> List[Dict]:
    """读 manifest.csv；分组唯一依据 = group 字段；过滤口径由调用方登记。

    subject_filter: 'status_ok'（manifest status=='ok'，46 例）| 'qc_pass'（qc_pass=='True'，13 例）
    """
    path = os.path.join(data_root, "manifest.csv")
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    entries = []
    for r in rows:
        if subject_filter == "status_ok":
            keep = str(r["status"]).strip() == "ok"
        elif subject_filter == "qc_pass":
            keep = str(r["qc_pass"]).strip() == "True"
        else:
            raise ValueError(f"未知 subject_filter: {subject_filter}")
        if not keep:
            continue
        sid = int(r["subject_id"])
        group = r["group"].strip()
        p = os.path.join(data_root, group, f"{sid}_rest_v4.npz")
        if not os.path.exists(p):
            raise FileNotFoundError(f"manifest 声明的 npz 不存在: {p}")
        entries.append(
            {
                "subject_id": sid,
                "group": group,
                "npz_path": p,
                "n_bad_channels": int(r["n_bad_channels"]) if r["n_bad_channels"] else 0,
                "ref_chain": r["ref_chain"],
            }
        )
    entries.sort(key=lambda e: e["subject_id"])
    return entries


def stratified_subject_folds(entries: Sequence[Dict], n_folds: int, seed: int) -> List[List[Dict]]:
    """受试者级分层划分：每类内随机置换后，逐个分给该类计数最少的折（平局取最低折号）。

    同一受试者的全部窗必然落在同一折（划分单位 = 受试者）。
    """
    rng = np.random.default_rng(seed)
    groups = sorted({e["group"] for e in entries})
    counts = {g: [0] * n_folds for g in groups}
    folds: List[List[Dict]] = [[] for _ in range(n_folds)]
    for g in groups:
        idx = [i for i, e in enumerate(entries) if e["group"] == g]
        idx = [int(i) for i in rng.permutation(idx)]
        for i in idx:
            f = int(np.argmin(counts[g]))
            folds[f].append(entries[i])
            counts[g][f] += 1
    return folds


def split_fold(
    entries: Sequence[Dict],
    n_folds: int = 5,
    fold: int = 0,
    split_seed: int = 42,
    n_val_subjects: int = 2,
    val_subject_seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """返回 (train, internal_val, test) 受试者列表。

    test = 第 fold 折；train = 其余折；internal_val = train 折内每类切 1 例
    （合计 n_val_subjects，类不足时取该类全部）。
    """
    folds = stratified_subject_folds(entries, n_folds, split_seed)
    test = folds[fold]
    train_pool = [e for i, f in enumerate(folds) if i != fold for e in f]
    rng = np.random.default_rng(val_subject_seed)
    val: List[Dict] = []
    per_group = max(1, n_val_subjects // 2)
    groups = sorted({e["group"] for e in train_pool})
    for g in groups:
        cand = [e for e in train_pool if e["group"] == g]
        pick = [int(i) for i in rng.permutation(len(cand))[:per_group]]
        val.extend(cand[i] for i in pick)
    val_ids = {e["subject_id"] for e in val}
    train = [e for e in train_pool if e["subject_id"] not in val_ids]
    assert not ({e["subject_id"] for e in train} & {e["subject_id"] for e in test})
    assert not ({e["subject_id"] for e in val} & {e["subject_id"] for e in test})
    return train, val, test


def compute_channel_stats(entries: Sequence[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """逐通道 mean/std，float64 累加；估计口径 = 传入的受试者集合（调用方保证仅训练折）。"""
    n_ch = None
    s = None
    ss = None
    n = 0
    for e in entries:
        d = np.load(e["npz_path"])["data"].astype(np.float64)  # (C, T)
        if s is None:
            n_ch = d.shape[0]
            s = np.zeros(n_ch)
            ss = np.zeros(n_ch)
        s += d.sum(axis=1)
        ss += (d ** 2).sum(axis=1)
        n += d.shape[1]
    mean = s / n
    var = ss / n - mean ** 2
    std = np.sqrt(np.maximum(var, 1e-12))
    return mean, std


class EdraWindowsDataset(Dataset):
    """受试者列表 -> z-score 后内存数组 -> 滑窗 (X_past, X_future)。

    全部预处理（z-score、窗切分）在本类内完成；输出 float32。
    """

    def __init__(
        self,
        entries: Sequence[Dict],
        channel_mean: np.ndarray,
        channel_std: np.ndarray,
        past_len: int = 384,
        future_len: int = 128,
        stride: int = 64,
        return_group: bool = False,
    ):
        self.past_len = past_len
        self.future_len = future_len
        self.stride = stride
        self.return_group = return_group
        self.subjects: List[Dict] = []
        self.groups: List[str] = []
        self.arrays: List[np.ndarray] = []
        win = past_len + future_len
        self.windows: List[Tuple[int, int]] = []
        mean = channel_mean.astype(np.float32)
        std = channel_std.astype(np.float32)
        for si, e in enumerate(entries):
            d = np.load(e["npz_path"])["data"].astype(np.float32)
            z = (d - mean[:, None]) / std[:, None]
            self.arrays.append(np.ascontiguousarray(z))
            self.subjects.append(e)
            self.groups.append(e["group"])
            n_start = (z.shape[1] - win) // stride + 1
            self.windows.extend((si, k * stride) for k in range(n_start))

    def __len__(self) -> int:
        return len(self.windows)

    def n_windows_per_subject(self) -> List[int]:
        out = [0] * len(self.arrays)
        for si, _ in self.windows:
            out[si] += 1
        return out

    def __getitem__(self, idx: int):
        si, s = self.windows[idx]
        arr = self.arrays[si]
        past = torch.from_numpy(arr[:, s : s + self.past_len])
        fut = torch.from_numpy(arr[:, s + self.past_len : s + self.past_len + self.future_len])
        if self.return_group:
            return past, fut, self.groups[si]
        return past, fut


def split_manifest_path(cfg) -> str:
    """划分固化 manifest 路径：splits/fold{N}_{filter}_seed{S}.json（变体目录内，随仓库提交）。"""
    override = getattr(cfg, "split_manifest_override", "")
    if override:
        return override  # 变体自备固化 manifest（v4-B val4 口径，2026-10-16 用户 7/1/2 拍板；v1/v2/v3 无此字段=路径不变）
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "splits")
    return os.path.join(d, f"fold{cfg.fold}_{cfg.subject_filter}_seed{cfg.split_seed}.json")


def fixate_split(cfg, train: Sequence[Dict], val: Sequence[Dict], test: Sequence[Dict]) -> str:
    """固化划分：manifest 不存在则写入；存在则校验本次划分与之一致，漂移即报错。

    覆盖 roles = train（已剔除内部验证）/ val / test 的受试者清单与分组、
    过滤口径 / seeds / fold 参数——任一变化都会使 dict 不等而触发报错。
    """
    path = split_manifest_path(cfg)
    cur = {
        "filter": cfg.subject_filter,
        "n_folds": cfg.n_folds,
        "fold": cfg.fold,
        "split_seed": cfg.split_seed,
        "val_subject_seed": cfg.val_subject_seed,
        "train": sorted(e["subject_id"] for e in train),
        "val": sorted(e["subject_id"] for e in val),
        "test": sorted(e["subject_id"] for e in test),
        "groups": {str(e["subject_id"]): e["group"] for e in list(train) + list(val) + list(test)},
    }
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            saved = json.load(f)
        if saved != cur:
            raise RuntimeError(
                f"划分与固化 manifest 不一致: {path}\n"
                f"  固化: train{len(saved['train'])}/val{len(saved['val'])}/test{len(saved['test'])} "
                f"filter={saved['filter']} seed={saved['split_seed']}\n"
                f"  当前: train{len(cur['train'])}/val{len(cur['val'])}/test{len(cur['test'])} "
                f"filter={cur['filter']} seed={cur['split_seed']}\n"
                "划分漂移被拒绝；若确属有意口径变更，须显式更新/删除 manifest 并在 experiment.md 登记后重跑"
            )
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cur, f, indent=2, ensure_ascii=False)
    return path


def build_datasets(cfg, limit_train_per_group: Optional[int] = None,
                   include_test: bool = True) -> Dict[str, EdraWindowsDataset]:
    """fold0 完整管道：manifest -> 受试者过滤 -> 分层划分 ->（固化校验）-> 内部验证切分 -> z-score(仅训练折)。

    limit_train_per_group：冒烟用——训练折每类仅取前 N 例（受试者升序，确定性）；
    z-score 统计随子集重算（冒烟口径，正式 Run 不传该参数）。
    """
    entries = load_manifest(cfg.data_root, subject_filter=cfg.subject_filter)
    train, val, test = split_fold(
        entries,
        n_folds=cfg.n_folds,
        fold=cfg.fold,
        split_seed=cfg.split_seed,
        n_val_subjects=cfg.n_val_subjects,
        val_subject_seed=cfg.val_subject_seed,
    )
    manifest = fixate_split(cfg, train, val, test) if cfg.fixate_split else None
    if limit_train_per_group is not None:
        sub = []
        for g in sorted({e["group"] for e in train}):
            sub.extend([e for e in train if e["group"] == g][:limit_train_per_group])
        sub.sort(key=lambda e: e["subject_id"])
        train = sub
    mean, std = compute_channel_stats(train)  # 仅训练折（已剔除内部验证）
    kw = dict(past_len=cfg.past_len, future_len=cfg.future_len, stride=cfg.stride)
    out: Dict[str, EdraWindowsDataset] = {
        "train": EdraWindowsDataset(train, mean, std, **kw),
        "val": EdraWindowsDataset(val, mean, std, **kw),
        "channel_mean": mean,
        "channel_std": std,
        "train_entries": train,
        "val_entries": val,
        "split_manifest": manifest,
    }
    if include_test:
        out["test"] = EdraWindowsDataset(test, mean, std, **kw)
        out["test_entries"] = test
    return out


def make_loader(
    ds: EdraWindowsDataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    epoch: int,
    num_workers: int = 4,
) -> DataLoader:
    """epoch 种子化 DataLoader（resume 后数据顺序可复现）。"""
    g = torch.Generator()
    g.manual_seed(seed * 100003 + epoch)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=g,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=shuffle,  # 训练丢尾批保形状稳定；评估保留
        worker_init_fn=lambda wid: np.random.seed((seed + epoch * 1000 + wid) % (2 ** 31)),
    )
