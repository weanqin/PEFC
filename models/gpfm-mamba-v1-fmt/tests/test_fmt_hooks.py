#!/usr/bin/env python3
"""test_fmt_hooks.py — v1 共享模块 default-off 钩子单测（任务 10-16-gpfm-mamba-fmt-f0 implement.md A5）。

守护面（五处钩子的默认分支零漂移 + fmt 分支生效）：
① 旧变体 config_hash 零漂移：v1/v2/v3/v4a 子进程口径逐位等于冻结基准（c6ff…/fc2a…/0c2d…/
   2a45…），且四变体均无 norm_type/fs 字段（getattr 透传默认 rmsnorm/500.0/40.0 不变）
② norm 钩子：默认 "rmsnorm" 分支与**改动前代码**（git HEAD 的 v1 mamba.py+model.py，干净
   子进程 temp 目录用前解析）同 seed 构建后 state_dict 键集/逐值/前向输出逐位一致；
   "layernorm" 分支 5 个 norm 位（4×MambaBlock pre-norm + head_norm）全部 nn.LayerNorm
   且 state_dict 含 *.norm.bias / head_norm.bias 键（容量 +5·d_model bias 参数）；
   非法 norm_type 构建期 ValueError；build_norm 两分支类型断言
③ run_all(skip_gc=True) 合成小输入冒烟：PSD/空间归因产物照落、**无任何 GC 产物**
   （gc_real/gc_gen/gc_F_p_* npy + gc 矩阵图）、结果 json 记 "gc": "skipped(blocked)" 且
   arrays/figures 无 gc 引用；默认分支（不传 skip_gc）同输入正对照 = GC 产物齐全（原行为）；
   签名 skip_gc 默认 False
④ fc_metric --outputs-root：默认值 = 原硬编码 /root/npi/outputs/gpfm-mamba-f0（--help 行为
   面断言）；显式传 tmp 根时全产物落 tmp（行为面正对照）
⑤ evaluate.py fs 透传三点：--skip-gc/--skip-gc-only argparse mutually exclusive group
   （同传构建期报错、单传可解析）；_traj_fig_data 的 "fs" 与 run_all 调用点 fs 均为
   float(getattr(cfg, "fs", 500.0))（AST 精确断言——旧 config 无 fs 字段 → 500.0 不漂移）

运行：python3 models/gpfm-mamba-v1-fmt/tests/test_fmt_hooks.py（② 需 CUDA——mamba_ssm；
①/④/⑤ 含子进程；全程约 2-3 分钟）
"""

from __future__ import annotations

import os

# BLAS 线程上限（environment.md 惯例：有效 16 核 cgroup，须在 numpy 导入前设置；可覆盖）
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "8")

import ast
import hashlib
import importlib.util
import inspect
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_TESTS = Path(__file__).resolve().parent          # models/gpfm-mamba-v1-fmt/tests
_V1FMT = _TESTS.parent                             # models/gpfm-mamba-v1-fmt
_MODELS = _V1FMT.parent
_REPO = _MODELS.parent                             # /root/npi
_V1 = _MODELS / "gpfm-mamba-v1"

sys.path.insert(0, str(_V1))                       # model/mamba/gc_psd 解析到 v1（只读复用）

import mamba as MB  # noqa: E402
import model as MD  # noqa: E402
import gc_psd as G  # noqa: E402

# ---- 登记值（冻结基准；任何口径变更须显式重算登记）----
V1_HASH = "c6ff80ac1c43"     # v1 main-01 落盘 resolved config hash
V2_HASH = "fc2a9ce14948"     # v2d62-main-01
V3_HASH = "0c2de30a4868"     # v3lp
V4A_HASH = "2a456201dc90"    # v4a r16（2026-10-16 首建 pin）
OLD_ROOT = "/root/npi/outputs/gpfm-mamba-f0"  # fc_metric 原硬编码根（默认值零漂移 pin）
V1_D16_RMS_PARAMS = 15_856   # v1 结构（d16/RMSNorm）参数量基线
V1_D16_LN_PARAMS = 15_936    # fmt layernorm（+80 = 5×16 bias）


def test_old_variant_hash_zero_drift() -> None:
    """① 旧四变体 config_hash 逐位 = 冻结基准；无 norm_type/fs 字段（getattr 默认不漂移）。"""
    variants = (
        (str(_V1), V1_HASH, "v1"),
        (str(_MODELS / "gpfm-mamba-v2"), V2_HASH, "v2d62"),
        (str(_MODELS / "gpfm-mamba-v3"), V3_HASH, "v3lp"),
        (str(_MODELS / "gpfm-mamba-v4a"), V4A_HASH, "v4a"),
    )
    for path, pin, name in variants:
        code = (f"import sys; sys.path.insert(0, {path!r}); import config as C; c = C.Config(); "
                f"print(c.config_hash(), getattr(c, 'norm_type', 'rmsnorm'), "
                f"getattr(c, 'fs', 500.0), getattr(c, 'gen_lowpass_hz', 40.0))")
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert r.returncode == 0, f"{name} 子进程失败：{r.stderr[-500:]}"
        got_hash, got_norm, got_fs, got_lp = r.stdout.split()
        assert got_hash == pin, f"{name} config_hash 漂移：{got_hash} != 基准 {pin}（钩子改坏旧 config）"
        assert got_norm == "rmsnorm", f"{name} norm_type 透传默认漂移：{got_norm}"
        assert float(got_fs) == 500.0, f"{name} fs 透传默认漂移：{got_fs}"
        assert float(got_lp) == 40.0, f"{name} gen_lowpass_hz 透传默认漂移：{got_lp}"
    print(f"[hash 零漂移] v1 {V1_HASH} / v2 {V2_HASH} / v3lp {V3_HASH} / v4a {V4A_HASH} 逐位一致；"
          f"四变体无 norm_type/fs 字段（getattr → rmsnorm/500.0/40.0）")


def _build_default_model_digest() -> tuple:
    """当前 v1 mamba.py+model.py 默认构建：返回 (keys_sha, values_sha, fwd_sha)。"""
    torch.manual_seed(1234)
    m = MD.VelocityNetwork().cuda()  # 全默认 kwargs = v1 结构（norm_type 默认 rmsnorm）
    m.eval()
    g = torch.Generator(device="cuda").manual_seed(7)
    xp = torch.randn(2, 62, 384, device="cuda", generator=g)
    xf = torch.randn(2, 62, 128, device="cuda", generator=g)
    t = torch.tensor([0.1, 0.9], device="cuda")
    with torch.no_grad():
        out = m(xp, xf, t)
    keys = sorted(m.state_dict())
    blob = b"".join(m.state_dict()[k].detach().cpu().numpy().tobytes() for k in keys)
    return (hashlib.sha256(",".join(keys).encode()).hexdigest(),
            hashlib.sha256(blob).hexdigest(),
            hashlib.sha256(out.detach().cpu().numpy().tobytes()).hexdigest(), keys)


def _preref_model_digest() -> tuple:
    """git HEAD（改动前）v1 mamba.py+model.py 干净子进程同口径构建：返回同结构 digest。

    子进程把改动前两文件写到 temp 目录并置于 sys.path[0]——model.py 的
    `from mamba import build_mamba_backbone` 命中**改动前** mamba.py（与本进程的改后模块隔离）。
    """
    tmp = tempfile.mkdtemp(prefix="v1_preref_")
    for fn in ("mamba.py", "model.py"):
        src = subprocess.check_output(
            ["git", "-C", str(_REPO), "show", f"HEAD:models/gpfm-mamba-v1/{fn}"], text=True)
        Path(tmp, fn).write_text(src)
    code = f"""
import sys, hashlib
sys.path.insert(0, {tmp!r})
import torch
torch.manual_seed(1234)
import model
m = model.VelocityNetwork().cuda()
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


def test_norm_hook_default_and_layernorm() -> None:
    """② norm 钩子：默认分支 vs 改动前代码逐位一致；layernorm 分支 5 norm 位全换 + bias 键。"""
    assert torch.cuda.is_available(), "norm 钩子等价性检查需 CUDA（mamba_ssm）"
    # 默认分支：与 git HEAD 改动前代码逐位一致（键集 / 逐值 / 前向输出）
    k_now, v_now, f_now, keys_now = _build_default_model_digest()
    k_pre, v_pre, f_pre, n_pre, p_pre = _preref_model_digest()
    assert k_now == k_pre, "默认分支 state_dict 键集与改动前不一致（零漂移破坏）"
    assert v_now == v_pre, "默认分支 state_dict 逐值与改动前不一致（初始化路径漂移）"
    assert f_now == f_pre, "默认分支前向输出与改动前非逐位一致（default-off 破坏）"
    assert p_pre == V1_D16_RMS_PARAMS, f"改动前参数量 {p_pre} != 登记基线 {V1_D16_RMS_PARAMS}"
    assert not any(k.endswith(".norm.bias") or k == "head_norm.bias" for k in keys_now), \
        "默认（RMSNorm）分支不应含任何 norm bias 键"
    assert isinstance(MB.build_norm(16, "rmsnorm"), nn.RMSNorm)
    assert isinstance(MB.build_norm(16, "layernorm"), nn.LayerNorm)
    try:
        MB.build_norm(16, "batchnorm")
        raise AssertionError("非法 norm_type 未被拦截")
    except ValueError:
        pass
    # 显式 rmsnorm == 不传（默认）同构
    torch.manual_seed(1); a = MD.VelocityNetwork(norm_type="rmsnorm").cuda()
    torch.manual_seed(1); b = MD.VelocityNetwork().cuda()
    sd_a, sd_b = a.state_dict(), b.state_dict()
    assert sorted(sd_a) == sorted(sd_b) and all(torch.equal(sd_a[k], sd_b[k]) for k in sd_a)
    # layernorm 分支：5 个 norm 位全部 nn.LayerNorm + bias 键 + 容量 +80（5×d16）
    m = MD.VelocityNetwork(norm_type="layernorm").cuda()
    norm_mods = [blk.norm for blk in m.blocks] + [m.head_norm]
    assert len(norm_mods) == 5 and all(isinstance(nm, nn.LayerNorm) for nm in norm_mods), \
        "layernorm 分支 5 个 norm 位未全部换为 nn.LayerNorm"
    sd = m.state_dict()
    for i in range(4):
        assert f"blocks.{i}.norm.weight" in sd and f"blocks.{i}.norm.bias" in sd, f"缺 blocks.{i}.norm bias 键"
    assert "head_norm.weight" in sd and "head_norm.bias" in sd, "缺 head_norm.bias 键"
    assert m.num_params() == V1_D16_LN_PARAMS == V1_D16_RMS_PARAMS + 80, \
        f"layernorm 参数量 {m.num_params()} != {V1_D16_LN_PARAMS}（应 = 基线 + 5×16 bias）"
    m.eval()
    with torch.no_grad():
        xp = torch.randn(2, 62, 384, device="cuda")
        xf = torch.randn(2, 62, 128, device="cuda")
        out = m(xp, xf, torch.tensor([0.1, 0.9], device="cuda"))
    assert out.shape == (2, 62, 128) and torch.isfinite(out).all()
    print(f"[norm 钩子] 默认 rmsnorm：键集/逐值/前向 vs 改动前（git HEAD）逐位一致；"
          f"layernorm：5 norm 位全 nn.LayerNorm、含 bias 键、参数 {V1_D16_RMS_PARAMS}→{V1_D16_LN_PARAMS}"
          f"（+80）；非法值 ValueError 拦截")


def _make_synthetic_gc_input(run_dir: Path, seed: int = 2026) -> None:
    """合成 gc_input.npz：2 受试者 × 8 窗 × (3, 128)（real 含已知 AR 结构，gen 独立噪声）。"""
    rng = np.random.default_rng(seed)
    S, NpS, C, F = 2, 16, 3, 128  # NpS=16：PSD 拼接（keep 半数 × 128 − drop 128 = 896 ≥ 512 点）
    gen = rng.standard_normal((S * NpS, C, F)).astype(np.float32)
    real = np.empty_like(gen)
    for s in range(S):
        x = np.zeros((C, NpS * F + 20))
        for tt in range(1, x.shape[1]):
            x[0, tt] = 0.5 * x[0, tt - 1] + 0.4 * x[1, tt - 1] + rng.standard_normal()
            x[1, tt] = -0.3 * x[1, tt - 1] + rng.standard_normal()
            x[2, tt] = rng.standard_normal()
        real[s * NpS:(s + 1) * NpS] = x[:, 20:].reshape(C, NpS, F).transpose(1, 0, 2)
    (run_dir / "eval").mkdir(parents=True, exist_ok=True)
    (run_dir / "figs").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        run_dir / "eval" / "gc_input.npz", gen=gen, real=real,
        win_subj=np.repeat([11, 22], NpS),
        win_start=np.tile(np.arange(NpS) * 64, S).astype(np.int64), gc_seed=1235)


def _make_synthetic_data_root(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    np.save(root / "chan_coords_62.npy", rng.standard_normal((62, 3)) * 0.05)


class _Jsonl:
    def __init__(self, path: Path):
        self.path = path
        self.recs = []

    def write(self, rec: dict) -> None:
        self.recs.append(rec)
        with open(self.path, "a") as f:
            f.write(json.dumps(rec, default=float) + "\n")


def test_run_all_skip_gc() -> None:
    """③ run_all skip_gc：True = PSD/空间照落、无任何 GC 产物、json 记 skipped；默认 = 原行为。"""
    assert inspect.signature(G.run_all).parameters["skip_gc"].default is False, \
        "skip_gc 签名默认值须为 False（default-off）"
    logs = []
    with tempfile.TemporaryDirectory(prefix="skipgc_") as td:
        tdp = Path(td)
        run_dir = tdp / "run"
        data_root = tdp / "data"
        _make_synthetic_gc_input(run_dir)
        _make_synthetic_data_root(data_root)
        jsonl = _Jsonl(run_dir / "eval" / "metrics.jsonl")
        res = G.run_all(run_dir=str(run_dir), data_root=str(data_root), sigma_s=0.08,
                        jitter_s=1e-3, p_max=2, bootstrap_n=2, fs=250.0, nperseg=512,
                        stride=64, past_len=384, log=logs.append, jsonl=jsonl, skip_gc=True)
        # PSD / 空间产物照落
        psd = np.load(run_dir / "eval" / "psd.npz")
        assert set(psd.files) >= {"freqs", "psd_real", "psd_gen", "n_segments"}
        assert abs(float(psd["freqs"][1]) - 250.0 / 512) < 1e-9, "PSD 频轴未按 fs=250 生成"
        assert (run_dir / "eval" / "spatial_corr.npz").exists()
        figs = list((run_dir / "figs").glob("*.png"))
        assert any("eval_psd_" in f.name for f in figs) and any("eval_spatial_corr_" in f.name for f in figs)
        # 无任何 GC 产物
        for bad in ("gc_real.npy", "gc_gen.npy", "gc_F_p_real.npz", "gc_F_p_gen.npz"):
            assert not (run_dir / "eval" / bad).exists(), f"skip_gc=True 仍落 GC 产物 {bad}"
        assert not any("eval_gc_matrix_" in f.name for f in figs), "skip_gc=True 仍落 GC 矩阵图"
        # 结果 json：记 skipped 且无 gc 数组/图引用
        saved = json.loads((run_dir / "eval" / "gc_psd_results.json").read_text())
        assert saved["gc"] == "skipped(blocked)", f"gc 字段应记 skipped(blocked)，got {saved['gc']!r}"
        assert set(saved["arrays"]) == {"psd", "spatial"}, f"arrays 含 GC 引用：{saved['arrays']}"
        assert not any("gc_matrix" in f for f in saved["figures"]), "figures 含 GC 图引用"
        assert saved["psd"]["resolution_hz"] == round(250.0 / 512, 4)
        assert res["gc"] == "skipped(blocked)", "run_all 返回值 gc 字段未记 skipped(blocked)"
        assert any(r.get("stage") == "gc" and r.get("gc") == "skipped(blocked)" for r in jsonl.recs), \
        "jsonl 未记录 gc skipped"
        assert any("skip_gc=True" in m for m in logs), "日志未显式记 skip_gc"
    # 默认分支正对照：同输入不传 skip_gc → GC 产物齐全（原行为不变）
    with tempfile.TemporaryDirectory(prefix="defgc_") as td:
        tdp = Path(td)
        run_dir = tdp / "run"
        data_root = tdp / "data"
        _make_synthetic_gc_input(run_dir)
        _make_synthetic_data_root(data_root)
        res = G.run_all(run_dir=str(run_dir), data_root=str(data_root), sigma_s=0.08,
                        jitter_s=1e-3, p_max=2, bootstrap_n=2, fs=500.0, nperseg=512,
                        stride=64, past_len=384, log=lambda m: None)
        for good in ("gc_real.npy", "gc_gen.npy", "gc_F_p_real.npz", "gc_F_p_gen.npz"):
            assert (run_dir / "eval" / good).exists(), f"默认分支缺 GC 产物 {good}（原行为破坏）"
        assert isinstance(res["gc"], dict) and "p_star" in res["gc"] and "bootstrap" in res["gc"]
        assert any("eval_gc_matrix_" in f.name for f in
                   (run_dir / "figs").glob("*.png")), "默认分支缺 GC 矩阵图（原行为破坏）"
    print("[run_all skip_gc] True：psd.npz/spatial_corr.npz/两图落盘（fs=250 频轴正确）、"
          "无 gc_real/gc_gen/gc_F_p_*/gc 矩阵图、json+jsonl+日志记 skipped(blocked)、"
          "arrays/figures 无 GC 引用；默认分支正对照：GC 产物齐全原行为不变")


def test_fc_metric_outputs_root() -> None:
    """④ fc_metric --outputs-root：默认 = 原硬编码根（--help 行为面）；显式 tmp 根产物落 tmp。"""
    r = subprocess.run([sys.executable, str(_V1 / "fc_metric.py"), "--help"],
                       capture_output=True, text=True)
    assert r.returncode == 0 and "--outputs-root" in r.stdout, "--outputs-root 参数缺失"
    assert OLD_ROOT in r.stdout, f"--help 未展示默认根 {OLD_ROOT}（默认值漂移）"
    spec = importlib.util.spec_from_file_location("fc_metric_under_test", str(_V1 / "fc_metric.py"))
    sys.path.insert(0, str(_V1))
    fm = importlib.util.module_from_spec(spec)
    sys.modules["fc_metric_under_test"] = fm
    spec.loader.exec_module(fm)
    assert fm.DEFAULT_OUTPUTS_ROOT == OLD_ROOT, "模块级默认根漂移"
    # 行为面正对照：显式 tmp 根 + 合成 gc_input（≥2 受试者）→ 产物落 tmp
    with tempfile.TemporaryDirectory(prefix="fcm_") as td:
        tdp = Path(td)
        _make_synthetic_gc_input(tdp / "r1")
        r = subprocess.run([sys.executable, str(_V1 / "fc_metric.py"), "--run-id", "r1",
                            "--outputs-root", str(tdp)], capture_output=True, text=True)
        assert r.returncode == 0, f"fc_metric 显式根运行失败：{r.stderr[-500:]}"
        assert (tdp / "r1" / "eval" / "fc_metric.json").exists(), "fc_metric.json 未落显式根"
        out = json.loads((tdp / "r1" / "eval" / "fc_metric.json").read_text())
        assert out["run_id"] == "r1" and out["n_subjects"] == 2
        assert (tdp / "r1" / "eval" / "fc_matrices.npz").exists()
    print(f"[fc_metric] --outputs-root 默认 = {OLD_ROOT}（--help + 模块常量双 pin）；"
          f"显式 tmp 根全产物（fc_metric.json/fc_matrices.npz）落 tmp")


def test_evaluate_fs_and_mutex() -> None:
    """⑤ evaluate.py：fs 三点 AST 精确断言（traj "fs"/run_all fs = getattr(cfg, fs, 500.0)）
    + --skip-gc/--skip-gc-only mutually exclusive（同传报错、单传可解析）。"""
    src = (_V1 / "evaluate.py").read_text()
    tree = ast.parse(src)
    want = ast.dump(ast.parse("float(getattr(cfg, 'fs', 500.0))", mode="eval").body)
    # (a) _traj_fig_data 内 dict 的 "fs" 值
    traj_fs = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_traj_fig_data":
            for sub in ast.walk(node):
                if isinstance(sub, ast.Dict):
                    for k, v in zip(sub.keys, sub.values):
                        if isinstance(k, ast.Constant) and k.value == "fs":
                            traj_fs = ast.dump(v) == want
    assert traj_fs, "_traj_fig_data 的 \"fs\" 非 float(getattr(cfg, 'fs', 500.0))（轨迹图时间轴漂移）"
    # (b) run_all 调用点 fs kwarg + skip_gc kwarg
    call_fs = call_skip = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fname = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
            if fname == "run_all":
                for kw in node.keywords:
                    if kw.arg == "fs":
                        call_fs = ast.dump(kw.value) == want
                    if kw.arg == "skip_gc":
                        call_skip = ast.dump(kw.value) == ast.dump(
                            ast.parse("bool(args.skip_gc_only)", mode="eval").body)
    assert call_fs, "run_all 调用点 fs 非 float(getattr(cfg, 'fs', 500.0))"
    assert call_skip, "run_all 调用点未接 skip_gc=bool(args.skip_gc_only)"
    # (c) mutually exclusive group：进程内 parse_args
    spec = importlib.util.spec_from_file_location("evaluate_under_test", str(_V1 / "evaluate.py"))
    ev = importlib.util.module_from_spec(spec)
    sys.modules["evaluate_under_test"] = ev
    spec.loader.exec_module(ev)  # 其 import config 命中 sys.path[0]=v1（v1 入口同口径）
    old_argv = sys.argv
    try:
        sys.argv = ["evaluate.py", "--run-id", "t", "--skip-gc-only"]
        ns = ev.parse_args()
        assert ns.skip_gc_only is True and ns.skip_gc is False, "--skip-gc-only 单传解析失败"
        sys.argv = ["evaluate.py", "--run-id", "t", "--skip-gc"]
        ns = ev.parse_args()
        assert ns.skip_gc is True and ns.skip_gc_only is False, "--skip-gc 单传解析失败"
        sys.argv = ["evaluate.py", "--run-id", "t", "--skip-gc", "--skip-gc-only"]
        try:
            ev.parse_args()
            raise AssertionError("--skip-gc 与 --skip-gc-only 同传未被互斥组拦截")
        except SystemExit:
            pass
    finally:
        sys.argv = old_argv
    print("[evaluate fs/mutex] _traj_fig_data \"fs\" 与 run_all fs 均 = float(getattr(cfg, 'fs', 500.0))；"
          "skip_gc=bool(--skip-gc-only)；两旗标互斥（同传 SystemExit、单传可解析）")


def main() -> None:
    test_old_variant_hash_zero_drift()   # ①
    test_norm_hook_default_and_layernorm()  # ②（CUDA）
    test_run_all_skip_gc()               # ③
    test_fc_metric_outputs_root()        # ④
    test_evaluate_fs_and_mutex()         # ⑤
    print("[PASS] test_fmt_hooks 五项全部断言通过")


if __name__ == "__main__":
    main()
