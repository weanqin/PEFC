#!/usr/bin/env python3
"""test_v1fmt.py — gpfm-mamba-v1-fmt 变体单测（implement.md B4；v4a tests 风格）。

① config 字段断言：design §1.2 表全部差异字段（data_root/outputs_root/variant_name/
   period_p=25/gen_lowpass_hz=None/fs=250/n_val_subjects=4/split_manifest_override=本目录 splits/
   early_stop_patience=10/lr_schedule="constant"/norm_type="layernorm"）+ 与 v1 逐字段一致的
   其余超参 + config_hash pin（默认 0dc0d5249c68 / epochs=1 bench 覆写 6fd4c101f263）
② 单遮蔽 + 前缀守卫：config 解析到 v1-fmt、model/mamba 解析到 v1（单遮蔽契约）；
   `v1fmt-` 前缀守卫正反例（train 含 --resume 派生名与缺省自动派生；evaluate/fc_metric
   必填 + 错前缀拦截）；fc_metric 包装注入 --outputs-root 新根
③ fold0 划分固化（R2）：fmt manifest 48 例（HC24/MDD24）→ 分层 5-fold seed42 + n_val=4 →
   **34/4/10**（7/1/2 长期口径语义的实际整数分配）+ val 分层 2HC+2MDD + 三折互斥 + 首建写入
   本目录 splits/ + 二建校验一致（fixate 幂等）
④ dataloader 契约预检（B5 冒烟前置）：2 例受试者 → 每例 930 窗（512 窗 stride 64）、
   data (62,60001) float32、X_past (B,62,384)/X_future (B,62,128) float32、z-score 输出有限

运行：python3 models/gpfm-mamba-v1-fmt/tests/test_v1fmt.py（③④ 读 fmt 数据集 manifest；
④ 载 2 例 npz，约 10s）
"""

from __future__ import annotations

import os

# BLAS 线程上限（environment.md 惯例：有效 16 核 cgroup，须在 numpy 导入前设置；可覆盖）
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "8")

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

_TESTS = Path(__file__).resolve().parent
_V1FMT = _TESTS.parent                               # models/gpfm-mamba-v1-fmt
_MODELS = _V1FMT.parent
_V1 = _MODELS / "gpfm-mamba-v1"

sys.path.insert(0, str(_V1))                         # data 等 v1 模块（只读复用）
sys.path.insert(0, str(_V1FMT))                      # config → v1-fmt（遮蔽）

import config as C  # noqa: E402
import data as D  # noqa: E402  # v1 data.py（split_manifest_override 钩子所在）
import model as M  # noqa: E402  # v1 model.py（norm_type 钩子——v1-fmt 无 model.py，单遮蔽）

V1FMT_HASH_DEFAULT = "0dc0d5249c68"   # v1-fmt 默认 config hash（2026-10-16 首建 pin）
V1FMT_HASH_EPOCHS1 = "6fd4c101f263"   # epochs=1（smoke/bench 覆写口径）pin
FMT_OUTPUTS_ROOT = "/root/npi/outputs/gpfm-mamba-fmt-f0"
FMT_DATA_ROOT = "/root/autodl-tmp/edra_processed_fmt"


def test_config_fields() -> None:
    """① design §1.2 差异字段逐项 + 与 v1 一致字段抽样 + hash pin + 敏感性。"""
    cfg = C.Config()
    # 差异字段（design §1.2 v1-fmt 表）
    assert cfg.data_root == FMT_DATA_ROOT
    assert cfg.outputs_root == FMT_OUTPUTS_ROOT and cfg.variant_name == "gpfm-mamba-v1-fmt"
    assert cfg.period_p == 25, f"period_p 应 25（D3 保物理时长），got {cfg.period_p}"
    assert cfg.gen_lowpass_hz is None, "gen_lowpass_hz 应 None（D2 不滤）"
    assert cfg.fs == 250.0, f"fs 应 250.0，got {cfg.fs}"
    assert cfg.n_val_subjects == 4
    assert cfg.split_manifest_override == str(_V1FMT / "splits" / "fold0_status_ok_seed42_val4.json")
    assert Path(cfg.split_manifest_override).parent == _V1FMT / "splits", "override 须落本目录 splits/"
    assert cfg.early_stop_patience == 10 and cfg.lr_schedule == "constant"
    assert cfg.norm_type == "layernorm", f"norm_type 应 layernorm（D4），got {cfg.norm_type}"
    # 与 v1 逐字段一致（抽样关键项；d_model=16 = v1 贯穿 trunk）
    assert (cfg.n_channels, cfg.past_len, cfg.future_len, cfg.stride) == (62, 384, 128, 64)
    assert (cfg.d_model, cfg.n_layers, cfg.d_state, cfg.d_conv, cfg.expand) == (16, 4, 16, 4, 2)
    assert (cfg.batch_size, cfg.epochs, cfg.lr, cfg.seed, cfg.ema_decay) == (256, 100, 1e-3, 1234, 0.999)
    assert (cfg.kernel_t, cfg.gamma, cfg.iso, cfg.sigma_s, cfg.jitter_s) == ("ou", 1.0, 1e-2, 0.080, 1e-3)
    assert (cfg.w_mae, cfg.w_cons, cfg.w_tv, cfg.w_corr) == (1.0, 1.0, 0.1, 0.1)
    assert cfg.subject_filter == "status_ok" and cfg.split_seed == 42 and cfg.fold == 0
    assert not hasattr(cfg, "bottleneck_rank"), "v1-fmt 不得含 bottleneck_rank（v4a 专属）"
    # hash pin + 敏感性
    h = cfg.config_hash()
    assert h == C.build_config().config_hash() == V1FMT_HASH_DEFAULT, \
        f"v1-fmt config_hash {h} != pin {V1FMT_HASH_DEFAULT}（口径变更后须显式重算登记）"
    assert C.build_config(epochs=1).config_hash() == V1FMT_HASH_EPOCHS1, "epochs=1 覆写 hash != pin"
    import dataclasses
    assert dataclasses.replace(cfg, period_p=50).config_hash() != h, "hash 对 period_p 不敏感"
    assert dataclasses.replace(cfg, norm_type="rmsnorm").config_hash() != h, "hash 对 norm_type 不敏感"
    assert dataclasses.replace(cfg, fs=500.0).config_hash() != h, "hash 对 fs 不敏感"
    # 模型结构面：norm_type="layernorm" 下 5 norm 位全 LayerNorm（详细断言见 test_fmt_hooks ②）
    m = M.VelocityNetwork(d_model=cfg.d_model, norm_type=cfg.norm_type)
    import torch.nn as nn
    norm_mods = [b.norm for b in m.blocks] + [m.head_norm]
    assert len(norm_mods) == 5 and all(isinstance(nm, nn.LayerNorm) for nm in norm_mods)
    assert m.num_params() == 15_856 + 80, f"v1-fmt 参数量 {m.num_params()} != 15936（+80 = 5×16 bias）"
    print(f"[config] §1.2 差异字段 11 项全命中；与 v1 一致字段抽样通过；hash {V1FMT_HASH_DEFAULT}"
          f"（epochs=1 {V1FMT_HASH_EPOCHS1}）pin 且对 p/norm/fs 敏感；结构 = d16 + LayerNorm 15,936 参数")


def test_masking_and_guards() -> None:
    """② 单遮蔽命中（config→v1-fmt、model/mamba→v1）+ v1fmt- 前缀守卫 + outputs-root 注入。"""
    def load_wrapper(mod_name: str, filename: str):
        spec = importlib.util.spec_from_file_location(mod_name, str(_V1FMT / filename))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        return mod

    tr = load_wrapper("v1fmt_train_wrapper_under_test", "train.py")
    tr._shadow_fmt_config()
    import config as _C
    import model as _M
    import mamba as _mb
    assert Path(_C.__file__).resolve().parent == _V1FMT, f"config 遮蔽未命中 v1-fmt：{_C.__file__}"
    assert Path(_M.__file__).resolve().parent == _V1, f"model 须解析到 v1（单遮蔽契约）：{_M.__file__}"
    assert _M.VelocityNetwork is M.VelocityNetwork, "遮蔽后的 model 与直接导入不一致"
    assert Path(_mb.__file__).resolve().parent == _V1, f"mamba 应解析到 v1：{_mb.__file__}"
    # train 前缀守卫：正例（显式 / resume 派生 / 缺省自动派生）+ 反例
    tr._guard_run_id(["prog", "--run-id", "v1fmt-smoke-01"])
    tr._guard_run_id(["prog", "--resume", "/x/v1fmt-main-01/checkpoints/gpfm-mamba-v1-fmt-last.pt"])
    argv_auto = ["prog"]
    tr._guard_run_id(argv_auto)
    assert argv_auto[-2] == "--run-id" and argv_auto[-1].startswith("v1fmt-"), argv_auto
    for bad in (["prog", "--run-id", "main-01"],
                ["prog", "--run-id", "v4afmt-smoke-01"],
                ["prog", "--resume", "/x/v4a-r16-main-01/checkpoints/x.pt"]):
        try:
            tr._guard_run_id(bad)
            raise AssertionError(f"train 前缀守卫未拦截：{bad}")
        except SystemExit:
            pass
    # evaluate / fc_metric 守卫：必填 + 前缀
    ev = load_wrapper("v1fmt_evaluate_wrapper_under_test", "evaluate.py")
    fc = load_wrapper("v1fmt_fc_metric_wrapper_under_test", "fc_metric.py")
    ev._guard_run_id(["prog", "--run-id", "v1fmt-main-01"])
    fc._guard_run_id(["prog", "--run-id", "v1fmt-main-01"])
    argv_fc = ["prog", "--run-id", "v1fmt-main-01"]
    fc._inject_outputs_root(argv_fc)
    assert argv_fc[-2:] == ["--outputs-root", FMT_OUTPUTS_ROOT], "fc_metric 包装未注入新根"
    assert fc.FMT_OUTPUTS_ROOT == FMT_OUTPUTS_ROOT
    argv_fc2 = ["prog", "--run-id", "v1fmt-main-01", "--outputs-root", "/tmp/x"]
    fc._inject_outputs_root(argv_fc2)
    assert argv_fc2[-1] == "/tmp/x", "显式 --outputs-root 被覆盖（应显式值优先）"
    for wrap, bad in ((ev, ["prog"]),
                      (ev, ["prog", "--run-id", "v4afmt-main-01"]),
                      (fc, ["prog"]),
                      (fc, ["prog", "--run-id", "main-01"])):
        try:
            wrap._guard_run_id(bad)
            raise AssertionError(f"守卫未拦截：{bad}")
        except SystemExit:
            pass
    print("[遮蔽/守卫] config 单遮蔽命中 v1-fmt（model/mamba→v1）；v1fmt- 前缀守卫 train 3 反例"
          " + evaluate/fc_metric 4 反例；fc_metric 注入 --outputs-root 新根（显式值优先）")


def test_split_fixation_48() -> None:
    """③ fold0 划分固化：48 例 → 34/4/10 + val 分层 2HC+2MDD + manifest 首建/二建一致。"""
    cfg = C.Config()

    class _NoOverride:  # 模拟 v1/v2/v3 config（无 split_manifest_override 字段 → 原路径，零漂移）
        fold, subject_filter, split_seed = cfg.fold, cfg.subject_filter, cfg.split_seed

    assert D.split_manifest_path(_NoOverride()) == str(_V1 / "splits" / "fold0_status_ok_seed42.json"), \
        "default-off 钩子破坏：无字段时未回到 v1 原 manifest 路径"
    assert D.split_manifest_path(cfg) == cfg.split_manifest_override

    entries = D.load_manifest(cfg.data_root, subject_filter=cfg.subject_filter)
    assert len(entries) == 48, f"status_ok 过滤后应 48 例，got {len(entries)}"
    groups_count = {}
    for e in entries:
        groups_count[e["group"]] = groups_count.get(e["group"], 0) + 1
    assert groups_count == {"HC": 24, "MDD": 24}, f"分组计数异常：{groups_count}"
    train, val, test = D.split_fold(
        entries, n_folds=cfg.n_folds, fold=cfg.fold, split_seed=cfg.split_seed,
        n_val_subjects=cfg.n_val_subjects, val_subject_seed=cfg.val_subject_seed,
    )
    ids = lambda xs: sorted(e["subject_id"] for e in xs)  # noqa: E731
    assert (len(train), len(val), len(test)) == (34, 4, 10), \
        f"划分整数分配漂移：train {len(train)} / val {len(val)} / test {len(test)}（应 34/4/10）"
    vg = sorted(e["group"] for e in val)
    assert vg == ["HC", "HC", "MDD", "MDD"], f"val 分层破坏（应 2 HC + 2 MDD）：{vg}"
    assert not (set(ids(train)) & set(ids(val))) and not (set(ids(train)) & set(ids(test))) \
        and not (set(ids(val)) & set(ids(test))), "三折受试者重叠（泄漏）"
    assert len(set(ids(train)) | set(ids(val)) | set(ids(test))) == 48
    p1 = D.fixate_split(cfg, train, val, test)   # 首建写入本目录 splits/（或已存在则校验一致）
    p2 = D.fixate_split(cfg, train, val, test)   # 二建校验一致（幂等）
    assert Path(p1) == Path(cfg.split_manifest_override) and p1 == p2
    saved = json.loads(Path(p1).read_text(encoding="utf-8"))
    assert saved["train"] == ids(train) and saved["val"] == ids(val) and saved["test"] == ids(test)
    assert len(saved["groups"]) == 48
    print(f"[划分固化] 48 例（HC24/MDD24）→ train {len(train)} / val {len(val)} / test {len(test)}"
          f"（34/4/10，7/1/2 长期口径实际分配）；val={ids(val)}；manifest 首建+二建一致：{p1}")


def test_dataloader_contract() -> None:
    """④ dataloader 契约预检：每例 930 窗、data (62,60001) float32、窗形状/finite（B5 冒烟前置）。"""
    cfg = C.Config()
    entries = D.load_manifest(cfg.data_root, subject_filter=cfg.subject_filter)[:2]
    d0 = np.load(entries[0]["npz_path"])
    assert d0["data"].shape == (62, 60001) and d0["data"].dtype == np.float32, \
        f"fmt 数据形状/类型漂移：{d0['data'].shape} {d0['data'].dtype}"
    mean = np.zeros(62)
    std = np.ones(62)
    ds = D.EdraWindowsDataset(entries, mean, std, past_len=cfg.past_len,
                              future_len=cfg.future_len, stride=cfg.stride)
    wpn = ds.n_windows_per_subject()
    assert wpn == [930, 930], f"每例窗数应 930（512 窗 stride 64），got {wpn}"
    xp, x1 = ds[0]
    assert xp.dtype == torch.float32 and x1.dtype == torch.float32
    assert tuple(xp.shape) == (62, 384) and tuple(x1.shape) == (62, 128)
    assert torch.isfinite(xp).all() and torch.isfinite(x1).all(), "原始数据非有限值"
    z = D.EdraWindowsDataset(entries, np.random.default_rng(0).standard_normal(62) * 0 + 0.1,
                             np.full(62, 2.0), past_len=cfg.past_len,
                             future_len=cfg.future_len, stride=cfg.stride)
    zp, zf = z[123]
    assert torch.isfinite(zp).all() and torch.isfinite(zf).all(), "z-score 后非有限值"
    print(f"[dataloader] data (62,60001) float32；每例 930 窗；X_past (62,384)/X_future (62,128)"
          f" float32；原始与 z-score 输出全有限（均值 0.1/std 2.0 模拟统计）")


def main() -> None:
    test_config_fields()      # ①
    test_masking_and_guards()  # ②
    test_split_fixation_48()  # ③（首跑固化 manifest）
    test_dataloader_contract()  # ④
    print("[PASS] test_v1fmt 四项全部断言通过")


if __name__ == "__main__":
    main()
