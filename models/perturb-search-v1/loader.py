"""loader.py — θ* 冻结装载（v1-fmt）+ fold0 test 窗装载 + 窗集固定枚举（design §2；B0）。

只读依赖（path-isolation 契约，一律不写）：
- `models/gpfm-mamba-v1/`：sample/data/gpr_prior/model，经 sys.path；**v1 目录插 sys.path
  最前**——sample.py 顶层 `import config as C` 须解析到 v1 config（本任务不占用 `config` 名）。
- `models/gpfm-mamba-v1-fmt/`：config 经 importlib 唯一名 `v1fmt_config` 装载
  （sys.modules 先注册后 exec）；splits manifest 只读。
- `outputs/gpfm-mamba-fmt-f0/v1fmt-main-01/`：ckpt + channel_{mean,std}.npy，只读。
- 装载配方沿 `models/perturb-ec-v1/loader.py`（变体 config 换 v1-fmt；本任务自实现，不复用
  perturb-ec-v1 的 loader——后者读旧任务 config 网格与路径）。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_V1 = _HERE.parent / "gpfm-mamba-v1"

# 路径装配：先保证本目录可 import，再把 v1 插到最前（`config` 名留给 v1——消遮蔽纪律）
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_V1) not in sys.path:
    sys.path.insert(0, str(_V1))

import candidates as CD  # noqa: E402  本目录（含 search_config 唯一装载配方）

import sample as S       # noqa: E402  v1 sample.py（load_eval_model/build_eval_prior/EulerSampler/
                         #           iterate_test_batches，只读复用）
import numpy as np       # noqa: E402
import torch             # noqa: E402


def _load_unique(name: str, path: Path):
    """importlib 唯一名装载（design §2 配方）：sys.modules **先注册后 exec**
    （@dataclass 坑：dataclasses 按 cls.__module__ 查 sys.modules）；幂等复用同实例。"""
    if name in sys.modules:
        return sys.modules[name]
    assert path.exists(), f"装载目标不存在: {path}"
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod               # 必须先注册
    spec.loader.exec_module(mod)
    return mod


def load_v1fmt_config():
    """v1-fmt config 模块（唯一名 `v1fmt_config`；绝不占用 `config` 模块名）。"""
    SC = CD.load_config()
    return _load_unique("v1fmt_config", Path(SC.V1FMT_CONFIG_PATH))


def load_frozen_model(device: str = "cuda"):
    """θ* 冻结装载（只读）：load_eval_model（config_hash 严格校验 + EMA）+ build_eval_prior
    + EulerSampler。返回 (cfg, model, ema, prior, sampler, ck)。

    - cfg 一律来自 `v1fmt_config` 实例（iterate_test_batches / load_eval_model /
      build_eval_prior 全 cfg 驱动）。
    - gen_lowpass_hz 显式 `getattr(cfg, ..., 40.0)` 透传：v1-fmt cfg → **None**（fmt 产物
      口径）。构造器默认 40.0 + brickwall_lowpass 默认 fs=500 对 250 Hz 数据有效截止
      ~20 Hz 静默削波——已登记陷阱，B0 单测断言 `sampler.gen_lowpass_hz is None`。
    """
    SC = CD.load_config()
    cfg = load_v1fmt_config().build_config()
    model, ema, ck = S.load_eval_model(SC.CKPT_PATH, cfg, device)
    assert ck["config_hash"] == SC.EXPECTED_CONFIG_HASH, \
        f"ckpt hash {ck['config_hash']!r} != 登记 pin {SC.EXPECTED_CONFIG_HASH!r}（错拿 checkpoint）"
    prior = S.build_eval_prior(cfg, device)
    sampler = S.EulerSampler(
        model, prior, euler_steps=cfg.euler_steps, sigma_min=cfg.sigma_min,
        gen_lowpass_hz=getattr(cfg, "gen_lowpass_hz", 40.0),
    )
    return cfg, model, ema, prior, sampler, ck


def load_test_loader(batch_size: int = 4, num_workers: int = 2):
    """fold0 test 确定性 loader（复用 v1 `iterate_test_batches`；cfg 必须 = v1-fmt 实例）。

    **装载前置断言**：split_manifest_override 存在且等于登记路径——防 fixate_split
    「不存在则写入」分支把新 manifest 写进只读变体目录。返回 (loader, test_ds)。
    """
    SC = CD.load_config()
    manifest = Path(SC.SPLIT_MANIFEST)
    assert manifest.exists(), f"split manifest 不存在: {manifest}（装载前置断言）"
    cfg = load_v1fmt_config().build_config()
    assert Path(cfg.split_manifest_override) == manifest, \
        f"cfg.split_manifest_override {cfg.split_manifest_override} 与登记路径 {manifest} 不符"
    return S.iterate_test_batches(cfg, Path(SC.RUN_DIR), batch_size, num_workers)


def select_windows(test_ds):
    """窗集固定枚举（design §2 钉死）：fold0 test **splits manifest 顺序前 3 例**（10, 11, 12）
    × 每例数据集顺序前 8 窗。

    注：数据集内部受试者顺序 = split_fold 分层枚举序（实测 [44,24,17,…]，非 manifest 排序）——
    受试者定位按 manifest 顺序逐例查找，win_idx 亦按 manifest 顺序指派。

    返回 list[(win_idx, ds_idx, subject_id, window_in_subject)]，长度 24；win_idx =
    受试者序 × 8 + 例内窗序（10 → 0–7、11 → 8–15、12 → 16–23）。通道顺序 = 数据集
    npz 列序（与 chan_coords_62.npy / channel_{mean,std}.npy 对齐，splits manifest 同源）。
    """
    SC = CD.load_config()
    ids = [int(e["subject_id"]) for e in test_ds.subjects]
    missing = [s for s in SC.TEST_SUBJECTS_HEAD if s not in ids]
    assert not missing, f"manifest 前 3 例 {missing} 不在 test_ds.subjects（manifest 漂移）"
    per = test_ds.n_windows_per_subject()
    base_of = {}
    run = 0
    for si, sid in enumerate(ids):
        base_of[sid] = run
        run += per[si]
    assert run == len(test_ds), "各被试窗数累加与数据集长度不符"
    out = []
    for rank, sid in enumerate(SC.TEST_SUBJECTS_HEAD):        # rank = manifest 顺序（win_idx 依据）
        si = ids.index(sid)
        assert test_ds.windows[base_of[sid]][0] == si and \
            test_ds.windows[base_of[sid] + per[si] - 1][0] == si, \
            "窗全局索引非按受试者连续（EdraWindowsDataset 构造契约变化，窗枚举需重写）"
        assert per[si] >= SC.N_WINDOWS_PER_SUBJECT, \
            f"被试 {sid} 窗数 {per[si]} < {SC.N_WINDOWS_PER_SUBJECT}"
        for w in range(SC.N_WINDOWS_PER_SUBJECT):
            out.append((rank * SC.N_WINDOWS_PER_SUBJECT + w, base_of[sid] + w, sid, w))
    assert len(out) == SC.N_WINDOWS
    return out
