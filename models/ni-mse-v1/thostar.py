"""thostar.py — θ*（v5-fmt-01）冻结装载（design.md §2 装载契约 + §1b-8 顺序钉死）。

只读依赖（path-isolation 契约，一律不写）：
- `models/gpfm-mamba-v5-fmt/`：config.py / model.py / graph.py——**先 exec 预占
  `sys.modules["config"/"graph"/"model"]`（v5-fmt 文件、先注册后 exec——@dataclass 按
  cls.__module__ 查 sys.modules，不注册即崩）再 `import sample`（v1）**。顺序敏感
  （Run 007 实测教训）：v1 sample.py 导入时会把 v1 目录插回 sys.path[0]，若 config/model
  未预占缓存，其 `import config` / `from model import …` 会解析到 **v1** 文件。
- `models/gpfm-mamba-v1/`：sample.py / data.py / gpr_prior.py / mamba.py（v5-fmt 目录
  均无同名文件，经 sys.path 命中 v1）。
- `outputs/gpfm-mamba-f0/v5-fmt-01/`：ckpt（EMA shadow）+ channel_mean/std.npy + config.yaml。

七项 `__file__` 断言（design §2）：config/model/graph → v5-fmt 目录；
sample/data/gpr_prior/mamba → v1 目录。

EulerSampler 构造 `gen_lowpass_hz=None`（经 `getattr(cfg, "gen_lowpass_hz", 40.0)` 透传；
fmt 无低通口径——构造器默认 40.0 + brickwall fs=500 对 250 Hz 数据有效截止 ~20 Hz 静默
削波，已登记陷阱，装载后断言 `sampler.gen_lowpass_hz is None`）。

模型 / prior 构建参数**全部来自装载的 cfg**（kernel_t=ou、period_p=25、σ_s/jitter、
euler_steps=16、sigma_min=1e-4、geometry_path=fmt 原生——禁字面量重抄）；v1
`iterate_test_batches` / `load_eval_model` / `build_eval_prior` 均显式吃 cfg 实例，
v1 data.py 无模块级 config 依赖 ⇒ 天然拿到 v5-fmt config 实例（evaluate.py 包装先例）。
"""

from __future__ import annotations

import _blas_guard  # noqa: F401  必须最先：BLAS 五变量先于 numpy 装载

import importlib.util
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent          # models/ni-mse-v1
_MODELS = _HERE.parent
_V5F = _MODELS / "gpfm-mamba-v5-fmt"              # config/model/graph 来源（只读）
_V1 = _MODELS / "gpfm-mamba-v1"                   # sample/data/gpr_prior/mamba 来源（只读）

# ---- θ* 事实指纹（Run 000 登记；错拿即硬失败）----
RUN_DIR = "/root/npi/outputs/gpfm-mamba-f0/v5-fmt-01"
CKPT_PATH = f"{RUN_DIR}/checkpoints/gpfm-mamba-v5-fmt-best.pt"
EXPECTED_CONFIG_HASH = "a64c0d416756"
EXPECTED_EPOCH = 28                               # ckpt 字段为准（experiment「ep29」为 1 基显示差异）
EXPECTED_BEST_VAL = 0.7024236700547634
CHANNEL_MEAN_PATH = f"{RUN_DIR}/channel_mean.npy"
CHANNEL_STD_PATH = f"{RUN_DIR}/channel_std.npy"

# ---- fold0 manifest（只读复用 v1-fmt 固化划分；装载前置断言防 fixate_split 写入）----
SPLIT_MANIFEST = "/root/npi/models/gpfm-mamba-v1-fmt/splits/fold0_status_ok_seed42_val4.json"
EXPECTED_TEST_IDS = [10, 11, 12, 17, 24, 28, 30, 44, 61, 69]  # fmt manifest test 折（勘误后名单）

_SHADOWED = False


def _pre_occupy(name: str, path: Path):
    """v5-fmt 文件以唯一模块名预占 sys.modules（**先注册后 exec**，幂等）。"""
    if name in sys.modules:
        return sys.modules[name]
    assert path.exists(), f"预占目标不存在: {path}"
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod                       # 先注册（@dataclass 坑）
    spec.loader.exec_module(mod)
    return mod


def _assert_shadow() -> None:
    """七项 `__file__` 断言：config/model/graph→v5-fmt；sample/data/gpr_prior/mamba→v1。"""
    import config, data, gpr_prior, graph, mamba, model, sample  # noqa: F401
    expect = {"config": _V5F, "model": _V5F, "graph": _V5F,
              "sample": _V1, "data": _V1, "gpr_prior": _V1, "mamba": _V1}
    for name, parent in expect.items():
        mod = sys.modules[name]
        got = Path(mod.__file__).resolve().parent
        if got != parent:
            raise RuntimeError(f"装载遮蔽失败：{name} 解析到 {mod.__file__}（应为 {parent / (name + '.py')}）")


def shadow_modules() -> None:
    """装载遮蔽（幂等）：先插 [v5-fmt, v1] 双路径 → 预占 config/graph/model → import sample（v1）。"""
    global _SHADOWED
    if _SHADOWED:
        return
    for p in (str(_V1), str(_V5F)):               # 先插 v1 再插 v5-fmt ⇒ sys.path = [v5-fmt, v1, ...]
        if p not in sys.path:
            sys.path.insert(0, p)
    _pre_occupy("config", _V5F / "config.py")     # v5-fmt config（@dataclass Config）
    _pre_occupy("graph", _V5F / "graph.py")       # v5-fmt graph（model 的 A_eff 依赖）
    _pre_occupy("model", _V5F / "model.py")       # 其 `from graph import …` 命中缓存；
                                                  # `from mamba import …` 经 sys.path 命中 v1
    import sample  # noqa: F401                   # v1 sample.py（其顶层 sys.path.insert(v1) 此时无害：
                                                  # config/model 已在 sys.modules 缓存优先于路径解析）
    _assert_shadow()
    _SHADOWED = True


def load_cfg():
    """装载 v5-fmt config 实例并校验 config_hash pin（无 GPU / 无 ckpt 装载，report 期可用）。"""
    shadow_modules()
    import config as C
    cfg = C.Config()
    got = cfg.config_hash()
    if got != EXPECTED_CONFIG_HASH:
        raise RuntimeError(f"config_hash 不一致：cfg {got} != pin {EXPECTED_CONFIG_HASH}"
                           "（错拿 config / 口径漂移）")
    return cfg


def load_test_data(batch_size: int = 4, num_workers: int = 2):
    """fold0 test 确定性 loader（shuffle=False；复用 v1 `iterate_test_batches`，cfg=v5-fmt 实例）。

    装载前置断言（防 fixate_split「不存在则写入」分支写只读目录）：split_manifest_override
    存在且等于登记路径。返回 (loader, test_ds)；test_ds 逐窗索引访问同序（shuffle=False）。
    """
    cfg = load_cfg()
    manifest = Path(SPLIT_MANIFEST)
    assert manifest.exists(), f"split manifest 不存在: {manifest}（装载前置断言）"
    assert Path(cfg.split_manifest_override) == manifest, \
        f"cfg.split_manifest_override {cfg.split_manifest_override} != 登记 {manifest}"
    import sample as S
    loader, test_ds = S.iterate_test_batches(cfg, Path(RUN_DIR), batch_size, num_workers)
    got_ids = [int(e["subject_id"]) for e in test_ds.subjects]
    if sorted(got_ids) != EXPECTED_TEST_IDS:
        raise RuntimeError(f"test 折名单漂移：{sorted(got_ids)} != 登记名单 {EXPECTED_TEST_IDS}"
                           "（ds 内受试者序 = 分层划分置换序，与 manifest 排序清单按集合比对）")
    return loader, test_ds


def load_thostar(device: str = "cuda"):
    """θ* 冻结装载：EMA shadow 回灌（`with ema.swap(model)` 消费）+ 先验 + EulerSampler。

    返回 SimpleNamespace(cfg, model, ema, prior, sampler, ck, channel_mean, channel_std)。
    EulerSampler 构造 gen_lowpass_hz=None（断言锚点：默认 40.0 brickwall = 已登记陷阱）。
    """
    from types import SimpleNamespace

    import numpy as np
    import torch

    cfg = load_cfg()
    torch.backends.cudnn.deterministic = cfg.cudnn_deterministic
    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    import sample as S
    model, ema, ck = S.load_eval_model(CKPT_PATH, cfg, device)   # 内含 hash 校验 + EMA 装载
    if ck["config_hash"] != EXPECTED_CONFIG_HASH:                # 双保险（load_eval_model 已校验）
        raise RuntimeError(f"ckpt config_hash {ck['config_hash']} != pin {EXPECTED_CONFIG_HASH}")
    if int(ck["epoch"]) != EXPECTED_EPOCH or float(ck["best_val"]) != EXPECTED_BEST_VAL:
        raise RuntimeError(f"ckpt epoch/best_val 漂移：({ck['epoch']}, {ck['best_val']}) != "
                           f"pin ({EXPECTED_EPOCH}, {EXPECTED_BEST_VAL})")
    prior = S.build_eval_prior(cfg, device)
    sampler = S.EulerSampler(
        model, prior, euler_steps=cfg.euler_steps, sigma_min=cfg.sigma_min,
        gen_lowpass_hz=getattr(cfg, "gen_lowpass_hz", 40.0),     # v5-fmt cfg → None
    )
    assert sampler.gen_lowpass_hz is None, "fmt 口径必须 gen_lowpass_hz=None（默认 40.0 陷阱）"
    return SimpleNamespace(
        cfg=cfg, model=model, ema=ema, prior=prior, sampler=sampler, ck=ck,
        channel_mean=np.load(CHANNEL_MEAN_PATH),                 # 物理单位（σ_p^phys 汇报导出用）
        channel_std=np.load(CHANNEL_STD_PATH),
        device=device,
    )
