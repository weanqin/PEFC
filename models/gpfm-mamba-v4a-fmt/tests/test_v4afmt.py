#!/usr/bin/env python3
"""test_v4afmt.py — gpfm-mamba-v4a-fmt 变体单测（implement.md C3；v4a tests 风格）。

① config 字段断言：v4a 结构保持（bottleneck_rank=16、d_model=62）+ fmt 差异字段
   （data_root/outputs_root/variant_name/period_p=25/gen_lowpass_hz=None/fs=250/
   n_val_subjects=4/split_manifest_override=本目录 splits//early_stop_patience=10/
   lr_schedule="constant"/norm_type="layernorm"）+ config_hash pin（默认 05d9bb875a2c /
   epochs=1 bdab1f15c58b / r8 32bfd1deaa32 / r32 acba3715dc9b）
② 单遮蔽 + `v4afmt-` 前缀守卫 + --rank {8,16,32} 透传（argv 剥离两传法/非法值拦截/
   build_config 实例改写/rXX 标识校验/resume 派生名同检；接口对齐 v4a）
③ fold0 划分：48 例 → 34/4/10（同 v1-fmt 参数）+ 本目录 manifest 首建/二建一致 +
   **与 v1-fmt 的 manifest 逐字节一致**（同参数决定性划分，R2 关键断言）
④ 结构面：r16 瓶颈 + LayerNorm（5 norm 位全 nn.LayerNorm、含 bias 键）；
   参数量 130,366 = v4a 130,056 + 310（5×62 bias，容量登记）

运行：python3 models/gpfm-mamba-v4a-fmt/tests/test_v4afmt.py（④ 需 CUDA；全程约 1 分钟）
"""

from __future__ import annotations

import os

# BLAS 线程上限（environment.md 惯例：有效 16 核 cgroup，须在 numpy 导入前设置；可覆盖）
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "8")

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn

_TESTS = Path(__file__).resolve().parent
_V4AFMT = _TESTS.parent                               # models/gpfm-mamba-v4a-fmt
_MODELS = _V4AFMT.parent
_V1 = _MODELS / "gpfm-mamba-v1"
_V1FMT = _MODELS / "gpfm-mamba-v1-fmt"

sys.path.insert(0, str(_V1))                           # data/model 等 v1 模块（只读复用）
sys.path.insert(0, str(_V4AFMT))                       # config → v4a-fmt（遮蔽）

import config as C  # noqa: E402
import data as D  # noqa: E402  # v1 data.py（split_manifest_override 钩子）
import model as M  # noqa: E402  # v1 model.py（bottleneck_rank + norm_type 钩子）

V4AFMT_HASH_DEFAULT = "05d9bb875a2c"   # v4a-fmt 默认（r=16）config hash（2026-10-16 首建 pin）
V4AFMT_HASH_EPOCHS1 = "bdab1f15c58b"   # epochs=1（smoke/bench 覆写口径）pin
V4AFMT_HASH_RANK8 = "32bfd1deaa32"     # --rank 8 覆写口径 pin
V4AFMT_HASH_RANK32 = "acba3715dc9b"    # --rank 32 覆写口径 pin
FMT_OUTPUTS_ROOT = "/root/npi/outputs/gpfm-mamba-fmt-f0"
FMT_DATA_ROOT = "/root/autodl-tmp/edra_processed_fmt"
V4A_R16_PARAMS = 130_056               # v4a（RMSNorm）登记基线
V4AFMT_LN_PARAMS = 130_366             # +310 = 5×62 bias（容量登记）


def test_config_fields() -> None:
    """① v4a 结构保持 + fmt 差异字段 + hash 四 pin + 敏感性。"""
    cfg = C.Config()
    # v4a 结构保持（design §1.2：bottleneck_rank=16、d_model=62 保持）
    assert cfg.bottleneck_rank == 16 and cfg.d_model == 62
    assert not hasattr(cfg, "d_z") and not hasattr(cfg, "head_act"), "不得含 v4-B 专属字段"
    # fmt 差异字段
    assert cfg.data_root == FMT_DATA_ROOT
    assert cfg.outputs_root == FMT_OUTPUTS_ROOT and cfg.variant_name == "gpfm-mamba-v4a-fmt"
    assert cfg.period_p == 25 and cfg.gen_lowpass_hz is None and cfg.fs == 250.0
    assert cfg.n_val_subjects == 4
    assert cfg.split_manifest_override == str(_V4AFMT / "splits" / "fold0_status_ok_seed42_val4.json")
    assert cfg.early_stop_patience == 10 and cfg.lr_schedule == "constant"
    assert cfg.norm_type == "layernorm"
    # 与 v4a 一致（训练超参抽样）
    assert (cfg.batch_size, cfg.epochs, cfg.lr, cfg.seed, cfg.ema_decay) == (256, 100, 1e-3, 1234, 0.999)
    assert (cfg.n_layers, cfg.d_state, cfg.d_conv, cfg.expand, cfg.time_emb_dim) == (4, 16, 4, 2, 16)
    # hash pin + 敏感性
    h = cfg.config_hash()
    assert h == C.build_config().config_hash() == V4AFMT_HASH_DEFAULT, \
        f"v4a-fmt config_hash {h} != pin {V4AFMT_HASH_DEFAULT}（口径变更后须显式重算登记）"
    assert C.build_config(epochs=1).config_hash() == V4AFMT_HASH_EPOCHS1
    import dataclasses
    assert dataclasses.replace(cfg, bottleneck_rank=8).config_hash() == V4AFMT_HASH_RANK8
    assert dataclasses.replace(cfg, bottleneck_rank=32).config_hash() == V4AFMT_HASH_RANK32
    pins = {V4AFMT_HASH_DEFAULT, V4AFMT_HASH_EPOCHS1, V4AFMT_HASH_RANK8, V4AFMT_HASH_RANK32}
    assert len(pins) == 4, "四口径 hash 碰撞（pin 失效）"
    assert dataclasses.replace(cfg, norm_type="rmsnorm").config_hash() != h, "hash 对 norm_type 不敏感"
    assert dataclasses.replace(cfg, period_p=50).config_hash() != h, "hash 对 period_p 不敏感"
    print(f"[config] v4a 结构保持（r16/d62）+ fmt 差异字段全命中；hash 四 pin 就位"
          f"（默认 {V4AFMT_HASH_DEFAULT}/epochs1 {V4AFMT_HASH_EPOCHS1}/r8 {V4AFMT_HASH_RANK8}/"
          f"r32 {V4AFMT_HASH_RANK32}）且对 rank/norm/p 敏感")


def test_masking_guard_rank() -> None:
    """② 单遮蔽（config→v4a-fmt、model→v1）+ v4afmt- 前缀守卫 + --rank 正反例。"""
    def load_wrapper(mod_name: str, filename: str):
        spec = importlib.util.spec_from_file_location(mod_name, str(_V4AFMT / filename))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        return mod

    tr = load_wrapper("v4afmt_train_wrapper_under_test", "train.py")
    tr._shadow_fmt_config()
    import config as _C
    import model as _M
    import mamba as _mb
    assert Path(_C.__file__).resolve().parent == _V4AFMT, f"config 遮蔽未命中 v4a-fmt：{_C.__file__}"
    assert Path(_M.__file__).resolve().parent == _V1, f"model 须解析到 v1（单遮蔽契约）：{_M.__file__}"
    assert _M.VelocityNetwork is M.VelocityNetwork
    assert Path(_mb.__file__).resolve().parent == _V1
    # --rank 解析/剥离
    argv = ["prog", "--rank", "8", "--run-id", "v4afmt-r8-x", "--epochs", "1"]
    assert tr._pop_rank_arg(argv) == 8 and argv == ["prog", "--run-id", "v4afmt-r8-x", "--epochs", "1"]
    argv2 = ["prog", "--rank=16", "--run-id", "v4afmt-r16-x"]
    assert tr._pop_rank_arg(argv2) == 16 and argv2 == ["prog", "--run-id", "v4afmt-r16-x"]
    assert tr._pop_rank_arg(["prog", "--run-id", "v4afmt-smoke-01"]) is None
    try:
        tr._pop_rank_arg(["prog", "--rank", "5", "--run-id", "v4afmt-r5-x"])
        raise AssertionError("--rank 5（非法网格值）未被拦截")
    except SystemExit:
        pass
    # train 前缀守卫 + rXX 标识
    tr._guard_run_id(["prog", "--run-id", "v4afmt-smoke-01"], None)
    tr._guard_run_id(["prog", "--run-id", "v4afmt-r16-bench-1ep"], 16)
    tr._guard_run_id(["prog", "--resume", "/x/v4afmt-r8-main-01/checkpoints/x.pt"], 8)
    argv_auto = ["prog"]
    tr._guard_run_id(argv_auto, None)
    assert argv_auto[-1].startswith("v4afmt-")
    for bad, rank in ((["prog", "--run-id", "v1fmt-xxx"], None),
                      (["prog", "--run-id", "main-01"], None),
                      (["prog", "--resume", "/x/v4a-r16-main-01/checkpoints/last.pt"], None),
                      (["prog", "--run-id", "v4afmt-r16-x"], 8),
                      (["prog", "--run-id", "v4afmt-smoke-01"], 8),
                      (["prog", "--resume", "/x/v4afmt-r16-main-01/checkpoints/x.pt"], 8)):
        try:
            tr._guard_run_id(bad, rank)
            raise AssertionError(f"前缀/rXX 守卫未拦截：{bad}（rank={rank}）")
        except SystemExit:
            pass
    # --rank 覆写：build_config 实例改写 + hash = pin（事后恢复）
    orig_build = _C.build_config
    try:
        tr._apply_rank_override(8)
        cfg8 = _C.build_config()
        assert cfg8.bottleneck_rank == 8 and cfg8.config_hash() == V4AFMT_HASH_RANK8
        cfg8e = _C.build_config(epochs=1)
        assert cfg8e.bottleneck_rank == 8 and cfg8e.epochs == 1
        try:
            _C.build_config(bottleneck_rank=16)
            raise AssertionError("bottleneck_rank 直传覆写未被拦截")
        except KeyError:
            pass
    finally:
        _C.build_config = orig_build
    assert C.build_config().config_hash() == V4AFMT_HASH_DEFAULT, "恢复后默认构建漂移"
    # evaluate / fc_metric 守卫
    ev = load_wrapper("v4afmt_evaluate_wrapper_under_test", "evaluate.py")
    fc = load_wrapper("v4afmt_fc_metric_wrapper_under_test", "fc_metric.py")
    ev._guard_run_id(["prog", "--run-id", "v4afmt-r16-main-01"], None)
    ev._guard_run_id(["prog", "--run-id", "v4afmt-r8-main-01"], 8)
    ev_argv = ["prog", "--rank", "32", "--run-id", "v4afmt-r32-main-01"]
    assert ev._pop_rank_arg(ev_argv) == 32 and ev_argv == ["prog", "--run-id", "v4afmt-r32-main-01"]
    fc._guard_run_id(["prog", "--run-id", "v4afmt-r16-main-01"])
    argv_fc = ["prog", "--run-id", "v4afmt-r16-main-01"]
    fc._inject_outputs_root(argv_fc)
    assert argv_fc[-2:] == ["--outputs-root", FMT_OUTPUTS_ROOT]
    for wrap, bad, rank in ((ev, ["prog"], None),
                            (ev, ["prog", "--run-id", "v1fmt-main-01"], None),
                            (ev, ["prog", "--run-id", "v4afmt-r16-main-01"], 32),
                            (fc, ["prog"], None),
                            (fc, ["prog", "--run-id", "v4a-r16-main-01"], None)):
        try:
            if wrap is fc:
                wrap._guard_run_id(bad)
            else:
                wrap._guard_run_id(bad, rank)
            raise AssertionError(f"守卫未拦截：{bad}")
        except SystemExit:
            pass
    print("[遮蔽/守卫/--rank] config 单遮蔽命中 v4a-fmt（model/mamba→v1）；v4afmt- 前缀守卫 train 6 反例"
          " + evaluate/fc_metric 5 反例；--rank 剥离/非法值/实例改写（hash=pin）/rXX 标识全过；"
          "fc_metric 注入 --outputs-root 新根")


def test_split_matches_v1fmt() -> None:
    """③ fold0 划分：48 例 → 34/4/10 + 本目录 manifest 首建/二建 + 与 v1-fmt 逐字节一致。"""
    cfg = C.Config()
    entries = D.load_manifest(cfg.data_root, subject_filter=cfg.subject_filter)
    assert len(entries) == 48
    train, val, test = D.split_fold(
        entries, n_folds=cfg.n_folds, fold=cfg.fold, split_seed=cfg.split_seed,
        n_val_subjects=cfg.n_val_subjects, val_subject_seed=cfg.val_subject_seed,
    )
    ids = lambda xs: sorted(e["subject_id"] for e in xs)  # noqa: E731
    assert (len(train), len(val), len(test)) == (34, 4, 10), \
        f"划分整数分配漂移：{len(train)}/{len(val)}/{len(test)}（应 34/4/10）"
    vg = sorted(e["group"] for e in val)
    assert vg == ["HC", "HC", "MDD", "MDD"]
    p1 = D.fixate_split(cfg, train, val, test)
    p2 = D.fixate_split(cfg, train, val, test)
    assert Path(p1) == Path(cfg.split_manifest_override) and p1 == p2
    v1fmt_manifest = _V1FMT / "splits" / "fold0_status_ok_seed42_val4.json"
    assert v1fmt_manifest.exists(), "v1-fmt manifest 缺失（先跑 v1-fmt 测试/首跑固化）"
    assert Path(p1).read_bytes() == v1fmt_manifest.read_bytes(), \
        "v4a-fmt manifest 与 v1-fmt 非逐字节一致（同参数决定性划分破坏——两变体 test 可比性）"
    import json
    saved = json.loads(Path(p1).read_text(encoding="utf-8"))
    assert saved["test"] == [10, 11, 12, 17, 24, 28, 30, 44, 61, 69], \
        f"test 折 ID 漂移：{saved['test']}"
    print(f"[划分] 48 例 → 34/4/10；val={ids(val)}；manifest 首建+二建一致且与 v1-fmt **逐字节一致**；"
          f"test 10 例 {[10, 11, 12, 17, 24, 28, 30, 44, 61, 69]}")


def test_structure_r16_layernorm() -> None:
    """④ 结构面：r16 低秩瓶颈保持 + LayerNorm 5 norm 位 + 参数 130,366（+310）。"""
    assert torch.cuda.is_available()
    cfg = C.Config()
    m = M.VelocityNetwork(
        n_channels=cfg.n_channels, past_len=cfg.past_len, future_len=cfg.future_len,
        d_model=cfg.d_model, n_layers=cfg.n_layers, d_state=cfg.d_state,
        d_conv=cfg.d_conv, expand=cfg.expand, time_emb_dim=cfg.time_emb_dim,
        use_fast_path=cfg.use_fast_path, bottleneck_rank=cfg.bottleneck_rank,
        norm_type=cfg.norm_type,
    ).cuda()
    # 低秩瓶颈保持（v4a 结构）
    ip = m.in_proj
    assert isinstance(ip, nn.Sequential) and len(ip) == 2 and all(isinstance(x, nn.Linear) for x in ip)
    assert tuple(m.state_dict()["in_proj.0.weight"].shape) == (16, 79)
    # LayerNorm 5 norm 位 + bias 键
    norm_mods = [b.norm for b in m.blocks] + [m.head_norm]
    assert len(norm_mods) == 5 and all(isinstance(nm, nn.LayerNorm) for nm in norm_mods), \
        "5 个 norm 位未全部 LayerNorm"
    sd = m.state_dict()
    for i in range(4):
        assert f"blocks.{i}.norm.bias" in sd
    assert "head_norm.bias" in sd
    assert m.num_params() == V4AFMT_LN_PARAMS == V4A_R16_PARAMS + 310, \
        f"参数量 {m.num_params()} != {V4AFMT_LN_PARAMS}（应 = v4a 130,056 + 310 bias）"
    m.eval()
    with torch.no_grad():
        xp = torch.randn(2, 62, 384, device="cuda")
        xf = torch.randn(2, 62, 128, device="cuda")
        out = m(xp, xf, torch.tensor([0.1, 0.9], device="cuda"))
    assert out.shape == (2, 62, 128) and torch.isfinite(out).all()
    print(f"[结构] in_proj Sequential(79→16→62) 低秩保持；5 norm 位全 LayerNorm（含 bias 键）；"
          f"参数 {V4A_R16_PARAMS}→{V4AFMT_LN_PARAMS}（+310 = 5×62 bias）；前向 (2,62,128) 有限")


def main() -> None:
    test_config_fields()        # ①
    test_masking_guard_rank()   # ②
    test_split_matches_v1fmt()  # ③（首跑固化 + 与 v1-fmt 逐字节一致）
    test_structure_r16_layernorm()  # ④（CUDA）
    print("[PASS] test_v4afmt 四项全部断言通过")


if __name__ == "__main__":
    main()
