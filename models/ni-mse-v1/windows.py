"""windows.py — fmt fold0 test 确定性窗网格 + 窗子集选择（design.md §2 + v1.5 §1.5 窗子集规范）。

- 窗网格：v1 `iterate_test_batches`（loader shuffle=False 确定性序；z-score 已在 dataloader
  内完成）→ 每例 930 窗（60001 样本 @384/128/stride64），被试窗全局索引 0..929 按受试者
  连续（EdraWindowsDataset 构造契约，此处再断言）。
- 窗子集（v1.3 钉定，确定性无放回抽取后按原序排列）：
    W_null(200/例)：`SeedSequence(entropy=2026, spawn_key=(20, sid))` 派生 rng 无放回均匀抽
    200 窗后按原序（升序）排列；
    W_pert(50/例)：`spawn_key=(21, sid)` 从该例 W_null 集合内再抽 50、按原序排列——
    **W_pert ⊂ W_null**（τ 的噪声地板校准在测响应的同一批窗上）。
- 两子集与全部被试 id 落 manifest.json（run.py）。
"""

from __future__ import annotations

import _blas_guard  # noqa: F401  必须最先

import numpy as np

import noise as NZ

W_NULL_DEFAULT = 200
W_PERT_DEFAULT = 50


def draw_subset(n_total: int, size: int, branch: int, sid: int) -> np.ndarray:
    """确定性无放回均匀抽取 size 窗后按原序（升序）返回（局部索引 0..n_total-1）。

    窗数不足时取全部（冒烟小例口径；正式 Run 200/50 ≪ 930 不触发）。
    """
    rng = np.random.default_rng(
        np.random.SeedSequence(entropy=NZ.MASTER_ENTROPY,
                               spawn_key=NZ.spawn_key_for(branch, sid))
    )
    take = min(int(size), int(n_total))
    idx = np.sort(rng.choice(int(n_total), size=take, replace=False))
    return idx


def build_window_plan(test_ds, w_null: int = W_NULL_DEFAULT,
                      w_pert: int = W_PERT_DEFAULT) -> dict:
    """逐被试窗子集计划 {sid: {"wnull": 全局索引数组, "wpert": 全局索引数组}}。

    wnull ⊂ [0, n_total)，wpert ⊂ wnull（断言），均升序（原序保持）；w = 全局窗索引
    （种子元组取值口径，Run 000 登记）。
    """
    per = test_ds.n_windows_per_subject()
    plan: dict = {}
    base = 0
    for si, e in enumerate(test_ds.subjects):
        sid = int(e["subject_id"])
        n_total = per[si]
        assert test_ds.windows[base][0] == si and test_ds.windows[base + n_total - 1][0] == si, \
            "窗全局索引非按受试者连续（EdraWindowsDataset 构造契约变化，窗子集需重写）"
        wnull = draw_subset(n_total, w_null, NZ.BRANCH["win_null"], sid) + base
        pos = draw_subset(len(wnull), w_pert, NZ.BRANCH["win_pert"], sid)
        wpert = np.sort(wnull[pos])               # wnull 升序 ⇒ 已升序；np.sort 保持口径显式
        assert set(wpert.tolist()) <= set(wnull.tolist()), "W_pert ⊄ W_null（窗子集契约破坏）"
        plan[sid] = {"wnull": wnull, "wpert": wpert}
        base += n_total
    assert base == len(test_ds), "各被试窗数累加与数据集长度不符"
    return plan
