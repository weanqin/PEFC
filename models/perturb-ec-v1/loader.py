"""loader.py — θ* 冻结装载 + fold0 test 窗装载 + 逐被试窗子采样（design §1/§3；B0）。

只读依赖（path-isolation 薄变体契约，一律不写）：
- `models/gpfm-mamba-v1/`：sample/data/gpr_prior/model，经 sys.path；**v1 目录插 sys.path
  最前**——sample.py 顶层 `import config as C` 须解析到 v1 config（本任务不占用 `config` 名，
  该 import 仅注解使用，无害）。
- `models/gpfm-mamba-v4a-fmt/`：config 经 importlib 唯一名 `v4afmt_config` 装载
  （sys.modules 先注册后 exec，design §1 四行配方）；splits manifest 只读。
- `outputs/gpfm-mamba-fmt-f0/v4afmt-main-01/`：ckpt + channel_{mean,std}.npy，只读。
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

import perturb as PB  # noqa: E402  本目录（含 config 唯一装载配方）

import sample as S    # noqa: E402  v1 sample.py（load_eval_model/build_eval_prior/EulerSampler/
                      #           iterate_test_batches，只读复用）
import numpy as np    # noqa: E402
import torch          # noqa: E402


def _load_unique(name: str, path: Path):
    """importlib 唯一名装载（design §1 配方）：sys.modules **先注册后 exec**
    （@dataclass 坑：dataclasses 按 cls.__module__ 查 sys.modules）；幂等复用同实例。"""
    if name in sys.modules:
        return sys.modules[name]
    assert path.exists(), f"装载目标不存在: {path}"
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod               # 必须先注册
    spec.loader.exec_module(mod)
    return mod


def load_v4afmt_config():
    """v4a-fmt config 模块（唯一名 `v4afmt_config`；绝不占用 `config` 模块名）。"""
    return _load_unique("v4afmt_config", Path(PB.load_config().V4AFMT_CONFIG_PATH))


def load_frozen_model(device: str = "cuda"):
    """θ* 冻结装载（只读）：load_eval_model（config_hash 严格校验 + EMA）+ build_eval_prior
    + EulerSampler。返回 (cfg, model, ema, prior, sampler, ck)。

    - cfg 一律来自 `v4afmt_config` 实例（iterate_test_batches / load_eval_model /
      build_eval_prior 全 cfg 驱动，已核不受遮蔽影响）。
    - gen_lowpass_hz 显式 `getattr(cfg, ..., 40.0)` 透传：v4a-fmt cfg → **None**（fmt 产物
      口径）。构造器默认 40.0 + brickwall_lowpass 默认 fs=500 对 250 Hz 数据有效截止
      ~20 Hz 静默削波 D——已登记陷阱（design §2.2 B-Major1），B0 单测断言
      `sampler.gen_lowpass_hz is None`。
    """
    PC = PB.load_config()
    cfg = load_v4afmt_config().build_config()
    model, ema, ck = S.load_eval_model(PC.CKPT_PATH, cfg, device)
    assert ck["config_hash"] == PC.EXPECTED_CONFIG_HASH, \
        f"ckpt hash {ck['config_hash']!r} != 登记 pin {PC.EXPECTED_CONFIG_HASH!r}（错拿 checkpoint）"
    prior = S.build_eval_prior(cfg, device)
    sampler = S.EulerSampler(
        model, prior, euler_steps=cfg.euler_steps, sigma_min=cfg.sigma_min,
        gen_lowpass_hz=getattr(cfg, "gen_lowpass_hz", 40.0),
    )
    return cfg, model, ema, prior, sampler, ck


def load_test_loader(batch_size: int = 4, num_workers: int = 2):
    """fold0 test 确定性 loader（复用 v1 `iterate_test_batches`；cfg 必须 = v4a-fmt 实例）。

    **装载前置断言**（design §1）：split_manifest_override 存在且等于登记路径——防
    fixate_split「不存在则写入」分支把新 manifest 写进只读变体目录。返回 (loader, test_ds)。
    """
    PC = PB.load_config()
    manifest = Path(PC.SPLIT_MANIFEST)
    assert manifest.exists(), f"split manifest 不存在: {manifest}（装载前置断言，design §1）"
    cfg = load_v4afmt_config().build_config()
    assert Path(cfg.split_manifest_override) == manifest, \
        f"cfg.split_manifest_override {cfg.split_manifest_override} 与登记路径 {manifest} 不符"
    return S.iterate_test_batches(cfg, Path(PC.RUN_DIR), batch_size, num_workers)


def subsample_windows_per_subject(test_ds, n_w: int = None, seed: int = None) -> dict:
    """逐被试确定性窗子采样（design §3：默认 N_w=32/例、seed 2026、逐例独立）。

    抽样式（**实现口径，B1 首个 Run 块登记 experiment.md；档位调整须重登记**）：
    逐例独立 `np.random.default_rng(seed*1_000_003 + subject_id)` 均匀**无放回**抽
    n_w 个窗，按数据集全局索引升序返回（保窗时间顺序）；窗数不足时取全部。
    返回 {subject_id: np.ndarray(全局窗索引, 升序)}。
    """
    PC = PB.load_config()
    n_w = PC.N_W if n_w is None else n_w
    seed = PC.SEED if seed is None else seed
    per = test_ds.n_windows_per_subject()
    out: dict = {}
    base = 0
    for si, e in enumerate(test_ds.subjects):
        n_total = per[si]
        assert test_ds.windows[base][0] == si and test_ds.windows[base + n_total - 1][0] == si, \
            "窗全局索引非按受试者连续（EdraWindowsDataset 构造契约变化，子采样需重写）"
        rng = np.random.default_rng(seed * 1_000_003 + int(e["subject_id"]))
        take = min(n_w, n_total)
        idx = np.sort(rng.choice(n_total, size=take, replace=False)) + base
        out[int(e["subject_id"])] = idx
        base += n_total
    assert base == len(test_ds), "各被试窗数累加与数据集长度不符"
    return out
