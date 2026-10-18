"""loader.py — θ* 冻结装载（v1-fmt）+ fold0 test 窗装载 + 窗集固定枚举（全 10 例）。

从 perturb-search-v1/loader.py 复制适配（design §2）：窗枚举扩全 10 例；
config 引用换本任务唯一名 `ec_config`。只读依赖同源（path-isolation，一律不写）。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_V1 = _HERE.parent / "gpfm-mamba-v1"

if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_V1) not in sys.path:
    sys.path.insert(0, str(_V1))

import candidates as CD  # noqa: E402  本目录（含 ec_config 唯一装载配方）

import sample as S       # noqa: E402  v1 sample.py（只读复用）
import numpy as np       # noqa: E402
import torch             # noqa: E402


def _load_unique(name: str, path: Path):
    """importlib 唯一名装载（sys.modules 先注册后 exec；幂等复用同实例）。"""
    if name in sys.modules:
        return sys.modules[name]
    assert path.exists(), f"装载目标不存在: {path}"
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_v1fmt_config():
    """v1-fmt config 模块（唯一名 `v1fmt_config`；不占 `config` 名）。"""
    SC = CD.load_config()
    return _load_unique("v1fmt_config", Path(SC.V1FMT_CONFIG_PATH))


def load_frozen_model(device: str = "cuda"):
    """θ* 冻结装载（只读）：load_eval_model（hash 严格校验 + EMA）+ prior + EulerSampler。"""
    SC = CD.load_config()
    cfg = load_v1fmt_config().build_config()
    model, ema, ck = S.load_eval_model(SC.CKPT_PATH, cfg, device)
    assert ck["config_hash"] == SC.EXPECTED_CONFIG_HASH, \
        f"ckpt hash {ck['config_hash']!r} != pin {SC.EXPECTED_CONFIG_HASH!r}（错拿 checkpoint）"
    prior = S.build_eval_prior(cfg, device)
    sampler = S.EulerSampler(
        model, prior, euler_steps=cfg.euler_steps, sigma_min=cfg.sigma_min,
        gen_lowpass_hz=getattr(cfg, "gen_lowpass_hz", 40.0),
    )
    return cfg, model, ema, prior, sampler, ck


def load_test_loader(batch_size: int = 4, num_workers: int = 2):
    """fold0 test 确定性 loader（复用 v1 iterate_test_batches；前置断言防 fixate_split 写入）。"""
    SC = CD.load_config()
    manifest = Path(SC.SPLIT_MANIFEST)
    assert manifest.exists(), f"split manifest 不存在: {manifest}"
    cfg = load_v1fmt_config().build_config()
    assert Path(cfg.split_manifest_override) == manifest, \
        f"cfg.split_manifest_override {cfg.split_manifest_override} 与登记 {manifest} 不符"
    return S.iterate_test_batches(cfg, Path(SC.RUN_DIR), batch_size, num_workers)


def select_windows(test_ds):
    """窗集固定枚举（design §2 钉死）：fold0 test **manifest 顺序全 10 例** × 每例前 8 窗。

    返回 list[(win_idx, ds_idx, subject_id, window_in_subject)]，长度 80；win_idx =
    受试者序×8 + 例内窗序（10 → 0–7 … 69 → 72–79）。通道顺序 = npz 列序（同源对齐）。"""
    SC = CD.load_config()
    ids = [int(e["subject_id"]) for e in test_ds.subjects]
    missing = [s for s in SC.TEST_SUBJECTS if s not in ids]
    assert not missing, f"manifest 例 {missing} 不在 test_ds.subjects（manifest 漂移）"
    per = test_ds.n_windows_per_subject()
    base_of = {}
    run = 0
    for si, sid in enumerate(ids):
        base_of[sid] = run
        run += per[si]
    assert run == len(test_ds), "各被试窗数累加与数据集长度不符"
    out = []
    for rank, sid in enumerate(SC.TEST_SUBJECTS):
        si = ids.index(sid)
        assert test_ds.windows[base_of[sid]][0] == si and \
            test_ds.windows[base_of[sid] + per[si] - 1][0] == si, \
            "窗全局索引非按受试者连续（构造契约变化，窗枚举需重写）"
        assert per[si] >= SC.N_WINDOWS_PER_SUBJECT, \
            f"被试 {sid} 窗数 {per[si]} < {SC.N_WINDOWS_PER_SUBJECT}"
        for w in range(SC.N_WINDOWS_PER_SUBJECT):
            out.append((rank * SC.N_WINDOWS_PER_SUBJECT + w, base_of[sid] + w, sid, w))
    assert len(out) == SC.N_WINDOWS
    return out
