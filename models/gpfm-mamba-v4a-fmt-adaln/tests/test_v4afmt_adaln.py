#!/usr/bin/env python3
"""test_v4afmt_adaln.py — cond_adaln default-off 钩子 + gpfm-mamba-v4a-fmt-adaln 变体单测
（任务 10-17-gpfm-mamba-adaln-f0 implement.md A4 + B3；test_fmt_hooks/test_v4afmt 风格，独立脚本直跑）。

守护面（六段）：
① 零漂移：六旧变体（v1/v2/v3lp/v4a/v1-fmt/v4a-fmt）config_hash 逐位 = 冻结 pin，且均无
   cond_adaln 字段（getattr 透传默认 "off" 不变）；cond_adaln="off" 模型（v1 默认形 + v4a-fmt
   r16/LayerNorm 形）state_dict 键集/逐值/前向与**改动前代码**（git HEAD 的 v1 mamba.py+model.py，
   干净子进程 temp 目录用前解析）逐位一致（先例 test_fmt_hooks.py ②；本对比必须在钩子 commit
   之前有效——HEAD 尚为改前代码时运行）
② adaLN 分支：5 调制位键（blocks.{i}.mod.1.{weight,bias} ×4 + head_mod.1.{weight,bias}）齐全
   且形状 (124,16)/(124,)；参数 pin 140,906 = 基线 130,366 + 10,540；**fork_rng 契约**下同 seed
   共享权重与 off 分支逐位相同 + 恒等初始化前向与 LayerNorm 基线逐位一致（torch.equal 值相等
   语义——+0.0 把 LN 的 −0.0 映射 +0.0，值等字节异，review Suggestion）；非零调制权重活体断言；
   mod=None 分支与旧（git HEAD 改前）MambaBlock 逐位一致
③ 构造链 round-trip：经 sample.load_eval_model 同款 kwargs 路径构建（覆盖 sample.py 透传行
   真实执行面）+ 全部 mod 键装载；tiny EulerSampler 数步 CUDA 冒烟（adaLN 前向分支首触达前移）
④ 异常面：非法 cond_adaln 值构建期 ValueError；"zero"（B 案预留）NotImplementedError
⑤ config（B3）：唯 4 差异字段（cond_adaln/variant_name/outputs_root/split_manifest_override）
   + 其余与 v4a-fmt 逐字段一致 + hash pin（默认 9f0af705696c / epochs=1 8bd4ab0aae50）+
   对 cond_adaln 敏感；wrapper 三入口（train/evaluate/fc_metric）单遮蔽 + `v4afmt-adaln-`
   前缀守卫正反例（含 --resume 派生 run 名反例，path-isolation 协议）
⑥ fold0 划分（B3）：48 例 → 34/4/10；本目录 manifest 首建/二建一致 + 与 v4a-fmt 的 manifest
   **逐字节一致**（同参数决定性划分，R2 关键断言）

运行：python3 models/gpfm-mamba-v4a-fmt-adaln/tests/test_v4afmt_adaln.py（②③ 需 CUDA；全程约 2 分钟）
"""

from __future__ import annotations

import os

# BLAS 线程上限（environment.md 惯例：有效 16 核 cgroup，须在 numpy 导入前设置；可覆盖）
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "8")

import dataclasses
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

_TESTS = Path(__file__).resolve().parent          # models/gpfm-mamba-v4a-fmt-adaln/tests
_ADALN = _TESTS.parent                             # models/gpfm-mamba-v4a-fmt-adaln
_MODELS = _ADALN.parent
_REPO = _MODELS.parent                             # /root/npi
_V1 = _MODELS / "gpfm-mamba-v1"
_V4AFMT = _MODELS / "gpfm-mamba-v4a-fmt"

sys.path.insert(0, str(_V1))                       # data/model/sample 等 v1 模块（只读复用）
sys.path.insert(0, str(_ADALN))                    # config → 本变体（遮蔽）

import config as C  # noqa: E402
import data as D  # noqa: E402  # v1 data.py（split_manifest_override 钩子）
import mamba as MB  # noqa: E402  # v1 mamba.py（cond_adaln 钩子）
import model as M  # noqa: E402  # v1 model.py（cond_adaln 钩子）
import sample as S  # noqa: E402  # v1 sample.py（load_eval_model 透传行）

# ---- 登记值（冻结基准；任何口径变更须显式重算登记）----
V1_HASH = "c6ff80ac1c43"        # v1 main-01 落盘 resolved config hash
V2_HASH = "fc2a9ce14948"        # v2d62-main-01
V3_HASH = "0c2de30a4868"        # v3lp
V4A_HASH = "2a456201dc90"       # v4a r16
V1FMT_HASH = "0dc0d5249c68"     # v1-fmt 默认（test_v1fmt.py pin）
V4AFMT_HASH = "05d9bb875a2c"    # v4a-fmt 默认（test_v4afmt.py pin）
ADALN_HASH_DEFAULT = "9f0af705696c"  # 本变体默认（r=16+adaln）config hash（2026-10-17 首建 pin）
ADALN_HASH_EPOCHS1 = "8bd4ab0aae50"  # epochs=1（smoke/bench 覆写口径）pin
ADALN_OUTPUTS_ROOT = "/root/npi/outputs/gpfm-mamba-adaln-f0"
V4AFMT_LN_PARAMS = 130_366      # v4a-fmt（LayerNorm）基线参数量 pin
ADALN_PARAMS = 140_906          # +10,540 = 5×(16·124+124)（容量登记）


def test_old_variant_hash_zero_drift() -> None:
    """① 六旧变体 config_hash 逐位 = 冻结 pin；无 cond_adaln 字段（getattr 默认 off 不漂移）。"""
    variants = (
        (str(_V1), V1_HASH, "v1"),
        (str(_MODELS / "gpfm-mamba-v2"), V2_HASH, "v2d62"),
        (str(_MODELS / "gpfm-mamba-v3"), V3_HASH, "v3lp"),
        (str(_MODELS / "gpfm-mamba-v4a"), V4A_HASH, "v4a"),
        (str(_MODELS / "gpfm-mamba-v1-fmt"), V1FMT_HASH, "v1fmt"),
        (str(_V4AFMT), V4AFMT_HASH, "v4afmt"),
    )
    for path, pin, name in variants:
        code = (f"import sys; sys.path.insert(0, {path!r}); import config as C; c = C.Config(); "
                f"print(c.config_hash(), getattr(c, 'cond_adaln', 'off'))")
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert r.returncode == 0, f"{name} 子进程失败：{r.stderr[-500:]}"
        got_hash, got_adaln = r.stdout.split()
        assert got_hash == pin, f"{name} config_hash 漂移：{got_hash} != 基准 {pin}（钩子改坏旧 config）"
        assert got_adaln == "off", f"{name} cond_adaln 透传默认漂移：{got_adaln}"
    print(f"[hash 零漂移] v1 {V1_HASH} / v2 {V2_HASH} / v3lp {V3_HASH} / v4a {V4A_HASH} / "
          f"v1-fmt {V1FMT_HASH} / v4a-fmt {V4AFMT_HASH} 逐位一致；六变体无 cond_adaln 字段"
          f"（getattr → off）")


def _digest(m, seed: int, kwargs: dict) -> tuple:
    """进程内构建（当前代码）digest：返回 (keys_sha, values_sha, fwd_sha, n_keys, params)。"""
    torch.manual_seed(seed)
    mdl = M.VelocityNetwork(**kwargs).cuda()
    mdl.eval()
    g = torch.Generator(device="cuda").manual_seed(7)
    xp = torch.randn(2, 62, 384, device="cuda", generator=g)
    xf = torch.randn(2, 62, 128, device="cuda", generator=g)
    t = torch.tensor([0.1, 0.9], device="cuda")
    with torch.no_grad():
        out = mdl(xp, xf, t)
    keys = sorted(mdl.state_dict())
    blob = b"".join(mdl.state_dict()[k].detach().cpu().numpy().tobytes() for k in keys)
    return (hashlib.sha256(",".join(keys).encode()).hexdigest(),
            hashlib.sha256(blob).hexdigest(),
            hashlib.sha256(out.detach().cpu().numpy().tobytes()).hexdigest(),
            len(keys), mdl.num_params())


def _preref_digest(seed: int, kwargs_init: str) -> tuple:
    """git HEAD（改动前）v1 mamba.py+model.py 干净子进程同口径构建 digest（先例 test_fmt_hooks）。

    子进程把改动前两文件写到 temp 目录并置于 sys.path[0]——model.py 的
    `from mamba import build_mamba_backbone` 命中**改动前** mamba.py。kwargs_init 为
    VelocityNetwork(...) 的 kwargs 源码串（改前代码不识别 cond_adaln——off 语义即不传）。
    """
    tmp = tempfile.mkdtemp(prefix="v1_preref_adaln_")
    for fn in ("mamba.py", "model.py"):
        src = subprocess.check_output(
            ["git", "-C", str(_REPO), "show", f"HEAD:models/gpfm-mamba-v1/{fn}"], text=True)
        Path(tmp, fn).write_text(src)
    code = f"""
import sys, hashlib
sys.path.insert(0, {tmp!r})
import torch
torch.manual_seed({seed})
import model
m = model.VelocityNetwork({kwargs_init}).cuda()
m.eval()
g = torch.Generator(device='cuda').manual_seed(7)
xp = torch.randn(2, 62, 384, device='cuda', generator=g)
xf = torch.randn(2, 62, 128, device='cuda', generator=g)
t = torch.tensor([0.1, 0.9], device='cuda')
with torch.no_grad():
    out = m(xp, xf, t)
keys = sorted(m.state_dict())
blob = b''.join(m.state_dict()[k].detach().cpu().numpy().tobytes() for k in keys)
print(hashlib.sha256(','.join(keys).encode()).hexdigest(),
      hashlib.sha256(blob).hexdigest(),
      hashlib.sha256(out.detach().cpu().numpy().tobytes()).hexdigest(),
      len(keys), m.num_params())
"""
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, f"改动前模型子进程失败：{r.stderr[-800:]}"
    parts = r.stdout.split()
    return parts[0], parts[1], parts[2], int(parts[3]), int(parts[4])


def test_adaln_hook_zero_drift_and_branch() -> None:
    """①② cond_adaln="off" vs 改动前代码逐位一致 + adaLN 分支结构/参数/恒等初始化/活体/mod=None。"""
    assert torch.cuda.is_available(), "钩子等价性检查需 CUDA（mamba_ssm）"

    # ---- off 分支：与 git HEAD 改动前代码逐位一致（v1 默认形 + v4a-fmt r16/LN 形两种口径）----
    for seed, kw_off, kw_pre, n_par in (
        (1234, {}, "", 15_856),                                        # v1 全默认（d16/RMSNorm）
        (1234, dict(bottleneck_rank=16, d_model=62, norm_type="layernorm",
                    cond_adaln="off"),
         "bottleneck_rank=16, d_model=62, norm_type='layernorm'", V4AFMT_LN_PARAMS),  # v4a-fmt 形
    ):
        k_now, v_now, f_now, n_now, p_now = _digest(M, seed, kw_off)
        k_pre, v_pre, f_pre, n_pre, p_pre = _preref_digest(seed, kw_pre)
        assert k_now == k_pre and n_now == n_pre, f"off 分支 state_dict 键集与改动前不一致（{kw_off}）"
        assert v_now == v_pre, f"off 分支 state_dict 逐值与改动前不一致（{kw_off}）"
        assert f_now == f_pre, f"off 分支前向输出与改动前非逐位一致（{kw_off}）"
        assert p_now == p_pre == n_par, f"off 分支参数量 {p_now} != 登记基线 {n_par}"

    # ---- adaLN 分支：键/形状/参数 pin + fork_rng 共享权重契约 + 恒等初始化前向逐位一致 ----
    kw_ln = dict(bottleneck_rank=16, d_model=62, norm_type="layernorm")
    torch.manual_seed(1234); m_off = M.VelocityNetwork(**kw_ln).cuda().eval()
    torch.manual_seed(1234); m_on = M.VelocityNetwork(**kw_ln, cond_adaln="scale_shift").cuda().eval()
    sd_off, sd_on = m_off.state_dict(), m_on.state_dict()
    mod_keys = sorted(k for k in sd_on if k not in sd_off)
    want = [f"blocks.{i}.mod.1.{w}" for i in range(4) for w in ("bias", "weight")] + \
           [f"head_mod.1.{w}" for w in ("bias", "weight")]
    assert mod_keys == sorted(want), f"调制位键集不符：{mod_keys}"
    for k in mod_keys:
        assert tuple(sd_on[k].shape) == ((124,) if k.endswith("bias") else (124, 16)), \
            f"{k} 形状 {tuple(sd_on[k].shape)} 不符（应 (124,16)/(124,)）"
    assert m_on.num_params() == ADALN_PARAMS == m_off.num_params() + 10_540, \
        f"adaLN 参数量 {m_on.num_params()} != pin {ADALN_PARAMS}（应 = 基线 {V4AFMT_LN_PARAMS} + 10,540）"
    # fork_rng 契约：同 seed 共享权重逐位相同（含全部基线键）+ 调制末层全零
    shared = sorted(set(sd_off) & set(sd_on))
    assert len(shared) == len(sd_off) and all(torch.equal(sd_off[k], sd_on[k]) for k in shared), \
        "fork_rng 契约破坏：同 seed 共享权重与 off 分支不一致（RNG 中性构造失效）"
    assert all(bool((sd_on[k] == 0).all()) for k in mod_keys), "调制 MLP 末层未零初始化"
    # 恒等初始化前向：torch.equal 值相等语义（+0.0 把 LN 的 −0.0 映为 +0.0，值等字节异）
    g = torch.Generator(device="cuda").manual_seed(7)
    xp = torch.randn(2, 62, 384, device="cuda", generator=g)
    xf = torch.randn(2, 62, 128, device="cuda", generator=g)
    with torch.no_grad():
        o_off = m_off(xp, xf, torch.tensor([0.1, 0.9], device="cuda"))
        o_on0 = m_on(xp, xf, torch.tensor([0.1, 0.9], device="cuda"))
    assert torch.isfinite(o_on0).all() and o_on0.shape == (2, 62, 128)
    assert torch.equal(o_off, o_on0), "恒等初始化 adaLN 前向与 LayerNorm 基线非逐位一致（A 案破坏）"
    # 活体断言：非零调制权重 ⇒ 输出确实变化（调制路径真实生效）
    with torch.no_grad():
        for i in range(4):
            m_on.blocks[i].mod[-1].weight.normal_(0, 0.1)
            m_on.blocks[i].mod[-1].bias.normal_(0, 0.1)
        m_on.head_mod[-1].weight.normal_(0, 0.1)
        m_on.head_mod[-1].bias.normal_(0, 0.1)
        o_on1 = m_on(xp, xf, torch.tensor([0.1, 0.9], device="cuda"))
    assert not torch.equal(o_off, o_on1), "非零调制权重未改变输出（调制路径死路）"

    # ---- mod=None 分支：与旧（git HEAD 改前）MambaBlock 逐位一致（进程内双模块加载对比）----
    tmp = tempfile.mkdtemp(prefix="v1_preref_blk_")
    src = subprocess.check_output(
        ["git", "-C", str(_REPO), "show", "HEAD:models/gpfm-mamba-v1/mamba.py"], text=True)
    Path(tmp, "mamba.py").write_text(src)
    spec = importlib.util.spec_from_file_location("preref_mamba_under_test", str(Path(tmp) / "mamba.py"))
    pre_mb = importlib.util.module_from_spec(spec)
    sys.modules["preref_mamba_under_test"] = pre_mb
    spec.loader.exec_module(pre_mb)
    torch.manual_seed(99); blk_old = pre_mb.MambaBlock(62, 16, 4, 2, True, "layernorm").cuda().eval()
    torch.manual_seed(99); blk_new = MB.MambaBlock(62, 16, 4, 2, True, "layernorm",
                                                   cond_adaln="scale_shift", mod_dim=16).cuda().eval()
    sd_old, sd_new = blk_old.state_dict(), blk_new.state_dict()
    assert sorted(set(sd_new) & set(sd_old)) == sorted(sd_old) and \
        all(torch.equal(sd_old[k], sd_new[k]) for k in sd_old), "块级 fork_rng 契约破坏（共享权重不一致）"
    gb = torch.Generator(device="cuda").manual_seed(3)
    x = torch.randn(2, 512, 62, device="cuda", generator=gb)
    temb = torch.randn(2, 16, device="cuda", generator=gb)
    with torch.no_grad():
        out_old = blk_old(x)
        out_none = blk_new(x, None)      # mod=None：原式路径
        out_mod0 = blk_new(x, temb)      # 恒等初始化调制 = 纯 LN（值相等语义）
    assert torch.equal(out_old, out_none), "mod=None 分支与旧 MambaBlock 非逐位一致"
    assert torch.equal(out_old, out_mod0), "恒等初始化调制前向与旧 MambaBlock 非逐位一致"
    print(f"[钩子] off：v1 形 + v4a-fmt 形均与 git HEAD 改前逐位一致（键/值/前向）；"
          f"adaln：5 调制位键 (124,16)/(124,)、参数 {V4AFMT_LN_PARAMS}→{ADALN_PARAMS}（+10,540）、"
          f"fork_rng 共享权重契约成立、恒等初始化前向 torch.equal 基线、非零调制活体、"
          f"mod=None 与旧块逐位一致")


def test_load_eval_model_roundtrip_and_sampler() -> None:
    """③ sample.load_eval_model 同款 kwargs round-trip（透传行真实执行面）+ tiny EulerSampler CUDA 冒烟。"""
    assert torch.cuda.is_available()
    cfg = C.build_config()
    assert cfg.cond_adaln == "scale_shift"
    # 与 train.py 构造点同款 kwargs 建模型 + EMA → 存 ckpt → load_eval_model 装载
    torch.manual_seed(cfg.seed)
    model0 = M.VelocityNetwork(
        n_channels=cfg.n_channels, past_len=cfg.past_len, future_len=cfg.future_len,
        d_model=cfg.d_model, n_layers=cfg.n_layers, d_state=cfg.d_state,
        d_conv=cfg.d_conv, expand=cfg.expand, time_emb_dim=cfg.time_emb_dim,
        use_fast_path=cfg.use_fast_path,
        bottleneck_rank=getattr(cfg, "bottleneck_rank", None),
        norm_type=str(getattr(cfg, "norm_type", "rmsnorm")),
        cond_adaln=str(getattr(cfg, "cond_adaln", "off")),
    ).cuda()
    ema0 = M.EMA(model0, decay=cfg.ema_decay)
    with tempfile.TemporaryDirectory(prefix="adaln_rt_") as td:
        ck_path = str(Path(td) / "adaln-rt-best.pt")
        torch.save({"config_hash": cfg.config_hash(), "variant": cfg.variant_name,
                    "epoch": 0, "model": model0.state_dict(), "ema": ema0.state_dict()}, ck_path)
        # 真实执行面：load_eval_model 构造点（sample.py cond_adaln 透传行）+ config_hash 严格校验
        model1, ema1, ck = S.load_eval_model(ck_path, cfg, "cuda")
        sd1 = model1.state_dict()
        for k in (f"blocks.{i}.mod.1.{w}" for i in range(4) for w in ("weight", "bias")):
            assert k in sd1, f"load_eval_model 装载模型缺调制键 {k}"
        assert "head_mod.1.weight" in sd1 and "head_mod.1.bias" in sd1
        assert sorted(sd1) == sorted(model0.state_dict()), "round-trip 键集漂移"
        assert all(torch.equal(sd1[k], v.cuda()) for k, v in model0.state_dict().items()), \
            "round-trip 权重漂移"
        assert set(ema1.shadow) == set(sd1), "EMA shadow 键集缺调制键（泛型遍历应自动覆盖）"
        assert ck["variant"] == cfg.variant_name
    # tiny EulerSampler 数步 CUDA 冒烟（adaLN 前向分支经采样链路首触达）
    prior = S.build_eval_prior(cfg, "cuda")
    sampler = S.EulerSampler(model1, prior, euler_steps=4, sigma_min=cfg.sigma_min,
                             gen_lowpass_hz=getattr(cfg, "gen_lowpass_hz", 40.0))
    g = torch.Generator(device="cuda").manual_seed(cfg.seed)
    xp = torch.randn(2, 62, 384, device="cuda", generator=g)
    with ema1.swap(model1):
        smp = sampler.sample(xp, 2, generator=g)
    assert smp.shape == (2, 2, 62, 128) and torch.isfinite(smp).all(), \
        f"EulerSampler adaLN 冒烟失败：shape {tuple(smp.shape)}"
    print("[构造链] load_eval_model 同款 kwargs round-trip：全部 10 调制键装载 + EMA shadow 泛型覆盖；"
          "EulerSampler 4 步 CUDA 冒烟 (2,2,62,128) 有限")


def test_cond_adaln_validation() -> None:
    """④ 非法 cond_adaln 构建期 ValueError；"zero"（B 案预留）NotImplementedError。"""
    for ctor, name in ((lambda **kw: M.VelocityNetwork(**kw), "VelocityNetwork"),
                       (lambda **kw: MB.MambaBlock(16, 16, 4, 2, True, "rmsnorm", **kw), "MambaBlock")):
        try:
            ctor(cond_adaln="bogus")
            raise AssertionError(f"{name} 非法 cond_adaln 未被拦截（ValueError）")
        except ValueError:
            pass
        try:
            ctor(cond_adaln="zero")
            raise AssertionError(f"{name} cond_adaln='zero' 未抛 NotImplementedError")
        except NotImplementedError:
            pass
    print("[异常面] 非法值 ValueError / \"zero\" NotImplementedError（VelocityNetwork + MambaBlock 双入口）")


def test_config_fields_and_guards() -> None:
    """⑤ 唯 4 差异字段 + 其余与 v4a-fmt 逐字段一致 + hash pin/敏感性 + wrapper 遮蔽与前缀守卫正反例。"""
    # 唯 4 差异字段断言：与 v4a-fmt config 逐字段 diff
    spec = importlib.util.spec_from_file_location("v4afmt_config_ref", str(_V4AFMT / "config.py"))
    ref = importlib.util.module_from_spec(spec)
    sys.modules["v4afmt_config_ref"] = ref
    spec.loader.exec_module(ref)
    cfg, ref_cfg = C.Config(), ref.Config()
    diff = {f.name for f in dataclasses.fields(cfg)
            if getattr(cfg, f.name) != getattr(ref_cfg, f.name, "__MISSING__")}
    ref_names = {f.name for f in dataclasses.fields(ref.Config)}
    assert diff == {"cond_adaln", "variant_name", "outputs_root", "split_manifest_override"}, \
        f"与 v4a-fmt 差异字段超出唯 4 集：{diff}"
    assert ref_names <= {f.name for f in dataclasses.fields(type(cfg))}, "缺 v4a-fmt 既有字段"
    # 4 差异字段取值 + 基线保持
    assert cfg.cond_adaln == "scale_shift"
    assert cfg.variant_name == "gpfm-mamba-v4a-fmt-adaln"
    assert cfg.outputs_root == ADALN_OUTPUTS_ROOT
    assert cfg.split_manifest_override == str(_ADALN / "splits" / "fold0_status_ok_seed42_val4.json")
    assert (cfg.norm_type, cfg.bottleneck_rank, cfg.d_model, cfg.fs, cfg.period_p) == \
        ("layernorm", 16, 62, 250.0, 25)
    assert cfg.gen_lowpass_hz is None
    assert (cfg.seed, cfg.batch_size, cfg.lr, cfg.lr_schedule, cfg.ema_decay,
            cfg.early_stop_patience) == (1234, 256, 1e-3, "constant", 0.999, 10)
    # hash pin + 敏感性 + 唯一性
    h = cfg.config_hash()
    assert h == C.build_config().config_hash() == ADALN_HASH_DEFAULT, \
        f"config_hash {h} != pin {ADALN_HASH_DEFAULT}（口径变更后须显式重算登记）"
    assert C.build_config(epochs=1).config_hash() == ADALN_HASH_EPOCHS1
    assert dataclasses.replace(cfg, cond_adaln="off").config_hash() != h, "hash 对 cond_adaln 不敏感"
    assert len({ADALN_HASH_DEFAULT, ADALN_HASH_EPOCHS1}) == 2, "两口径 hash 碰撞（pin 失效）"

    # wrapper 三入口：单遮蔽 + 前缀守卫正反例（含 --resume 派生名）
    def load_wrapper(mod_name: str, filename: str, shadow: str | None = None):
        spec = importlib.util.spec_from_file_location(mod_name, str(_ADALN / filename))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        if shadow is not None:
            getattr(mod, shadow)()
        return mod

    tr = load_wrapper("adaln_train_wrapper_under_test", "train.py", "_shadow_adaln_config")
    import config as _C
    import model as _M
    import mamba as _mb
    assert Path(_C.__file__).resolve().parent == _ADALN, f"config 遮蔽未命中本变体：{_C.__file__}"
    assert Path(_M.__file__).resolve().parent == _V1, f"model 须解析到 v1（单遮蔽契约）：{_M.__file__}"
    assert _M.VelocityNetwork is M.VelocityNetwork
    assert Path(_mb.__file__).resolve().parent == _V1
    assert tr.RUN_PREFIX == "v4afmt-adaln-"
    # train 守卫正例（含 --resume 派生名）+ 自动 run_id 注入
    tr._guard_run_id(["prog", "--run-id", "v4afmt-adaln-smoke-01"])
    tr._guard_run_id(["prog", "--resume", "/x/v4afmt-adaln-main-01/checkpoints/last.pt"])
    argv_auto = ["prog"]
    tr._guard_run_id(argv_auto)
    assert argv_auto[-1].startswith("v4afmt-adaln-")
    # train 守卫反例（v4afmt-/裸名/他变体 resume 派生名）
    for bad in (["prog", "--run-id", "v4afmt-main-01"],          # v4a-fmt run 名（嵌套局限反例：须走本 wrapper）
                ["prog", "--run-id", "main-01"],
                ["prog", "--run-id", "v1fmt-main-01"],
                ["prog", "--resume", "/x/v4afmt-r16-main-01/checkpoints/last.pt"],   # --resume 派生名反例
                ["prog", "--resume", "/x/adaln-main-01/checkpoints/last.pt"]):
        try:
            tr._guard_run_id(bad)
            raise AssertionError(f"train 前缀守卫未拦截：{bad}")
        except SystemExit:
            pass
    # evaluate / fc_metric 守卫
    ev = load_wrapper("adaln_evaluate_wrapper_under_test", "evaluate.py", "_shadow_adaln_config")
    fc = load_wrapper("adaln_fc_metric_wrapper_under_test", "fc_metric.py")
    ev._guard_run_id(["prog", "--run-id", "v4afmt-adaln-main-01"])
    fc._guard_run_id(["prog", "--run-id", "v4afmt-adaln-main-01"])
    argv_fc = ["prog", "--run-id", "v4afmt-adaln-main-01"]
    fc._inject_outputs_root(argv_fc)
    assert argv_fc[-2:] == ["--outputs-root", ADALN_OUTPUTS_ROOT], "fc_metric 未注入本任务新根"
    for _, guard, bad in (
            (ev, lambda a: ev._guard_run_id(a), ["prog"]),
            (ev, lambda a: ev._guard_run_id(a), ["prog", "--run-id", "v4afmt-main-01"]),
            (ev, lambda a: ev._guard_run_id(a), ["prog", "--run-id", "main-01"]),
            (fc, lambda a: fc._guard_run_id(a), ["prog"]),
            (fc, lambda a: fc._guard_run_id(a), ["prog", "--run-id", "v4afmt-r16-main-01"])):
        try:
            guard(bad)
            raise AssertionError(f"守卫未拦截：{bad}")
        except SystemExit:
            pass
    print(f"[config/守卫] 与 v4a-fmt 唯 4 差异字段（cond_adaln/variant_name/outputs_root/"
          f"split_manifest_override）；hash pin {ADALN_HASH_DEFAULT}/{ADALN_HASH_EPOCHS1} 且对 "
          f"cond_adaln 敏感；train/evaluate/fc_metric 遮蔽命中本变体（model/mamba→v1），"
          f"v4afmt-adaln- 前缀守卫 8 反例全拦截（含 --resume 派生名），fc_metric 注入新根")


def test_split_matches_v4afmt() -> None:
    """⑥ fold0 划分：48 例 → 34/4/10 + 本目录 manifest 首建/二建一致 + 与 v4a-fmt 逐字节一致。"""
    cfg = C.Config()
    entries = D.load_manifest(cfg.data_root, subject_filter=cfg.subject_filter)
    assert len(entries) == 48
    train, val, test = D.split_fold(
        entries, n_folds=cfg.n_folds, fold=cfg.fold, split_seed=cfg.split_seed,
        n_val_subjects=cfg.n_val_subjects, val_subject_seed=cfg.val_subject_seed,
    )
    assert (len(train), len(val), len(test)) == (34, 4, 10), \
        f"划分整数分配漂移：{len(train)}/{len(val)}/{len(test)}（应 34/4/10）"
    p1 = D.fixate_split(cfg, train, val, test)   # 首跑固化 → 本目录 splits/
    p2 = D.fixate_split(cfg, train, val, test)   # 二建校验一致
    assert Path(p1) == Path(cfg.split_manifest_override) and p1 == p2
    v4afmt_manifest = _V4AFMT / "splits" / "fold0_status_ok_seed42_val4.json"
    assert v4afmt_manifest.exists(), "v4a-fmt manifest 缺失"
    assert Path(p1).read_bytes() == v4afmt_manifest.read_bytes(), \
        "本变体 manifest 与 v4a-fmt 非逐字节一致（同参数决定性划分破坏——两变体 test 可比性）"
    saved = json.loads(Path(p1).read_text(encoding="utf-8"))
    assert saved["test"] == [10, 11, 12, 17, 24, 28, 30, 44, 61, 69], \
        f"test 折 ID 漂移：{saved['test']}"
    print(f"[划分] 48 例 → 34/4/10；manifest 首建+二建一致且与 v4a-fmt **逐字节一致**；"
          f"test 10 例 {saved['test']}")


def main() -> None:
    test_old_variant_hash_zero_drift()          # ① 六旧变体 hash 零漂移
    test_adaln_hook_zero_drift_and_branch()     # ①② off vs 改前逐位一致 + adaLN 分支（CUDA）
    test_load_eval_model_roundtrip_and_sampler()  # ③ 构造链 round-trip + EulerSampler 冒烟（CUDA）
    test_cond_adaln_validation()                # ④ 异常面
    test_config_fields_and_guards()             # ⑤ config + wrapper 守卫（B3）
    test_split_matches_v4afmt()                 # ⑥ manifest 逐字节一致（B3；首跑固化）
    print("[PASS] test_v4afmt_adaln 六项全部断言通过")


if __name__ == "__main__":
    main()
