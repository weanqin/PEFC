"""loader.py — θ* 冻结装载（v1-fmt）+ fold0 test 窗装载 + 24 窗固定枚举。

从 ec-nmtime-v1/loader.py 复制适配：窗集换回前 3 例 × 8 窗；config 唯一名 `pd_config`。
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

import candidates as CD  # noqa: E402

import sample as S       # noqa: E402
import numpy as np       # noqa: E402
import torch             # noqa: E402


def _load_unique(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    assert path.exists(), f"装载目标不存在: {path}"
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_v1fmt_config():
    SC = CD.load_config()
    return _load_unique("v1fmt_config", Path(SC.V1FMT_CONFIG_PATH))


def load_frozen_model(device: str = "cuda"):
    SC = CD.load_config()
    cfg = load_v1fmt_config().build_config()
    model, ema, ck = S.load_eval_model(SC.CKPT_PATH, cfg, device)
    assert ck["config_hash"] == SC.EXPECTED_CONFIG_HASH
    prior = S.build_eval_prior(cfg, device)
    sampler = S.EulerSampler(
        model, prior, euler_steps=cfg.euler_steps, sigma_min=cfg.sigma_min,
        gen_lowpass_hz=getattr(cfg, "gen_lowpass_hz", 40.0),
    )
    return cfg, model, ema, prior, sampler, ck


def load_test_loader(batch_size: int = 4, num_workers: int = 2):
    SC = CD.load_config()
    manifest = Path(SC.SPLIT_MANIFEST)
    assert manifest.exists()
    cfg = load_v1fmt_config().build_config()
    assert Path(cfg.split_manifest_override) == manifest
    return S.iterate_test_batches(cfg, Path(SC.RUN_DIR), batch_size, num_workers)


def select_windows(test_ds):
    """窗集固定枚举：fold0 test manifest 顺序前 3 例 × 每例前 8 窗（win_idx 0..23）。"""
    SC = CD.load_config()
    ids = [int(e["subject_id"]) for e in test_ds.subjects]
    missing = [s for s in SC.TEST_SUBJECTS if s not in ids]
    assert not missing, f"manifest 例 {missing} 不在 test_ds.subjects"
    per = test_ds.n_windows_per_subject()
    base_of, run = {}, 0
    for si, sid in enumerate(ids):
        base_of[sid] = run
        run += per[si]
    assert run == len(test_ds)
    out = []
    for rank, sid in enumerate(SC.TEST_SUBJECTS):
        si = ids.index(sid)
        assert per[si] >= SC.N_WINDOWS_PER_SUBJECT
        for w in range(SC.N_WINDOWS_PER_SUBJECT):
            out.append((rank * SC.N_WINDOWS_PER_SUBJECT + w, base_of[sid] + w, sid, w))
    assert len(out) == SC.N_WINDOWS
    return out
