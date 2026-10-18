#!/usr/bin/env python3
"""run.py — ec-omtime25-f0 入口（O×M-time@25 EC 推断；输出变化仅计前 0.1 s/25 点）。

用法：
  cd models/ec-omtime25-v1
  python3 run.py smoke                       # B1：1 窗 497 行（机械正确性）
  nohup python3 run.py main > ../../outputs/ec-omtime25-f0/logs/main.log 2>&1 &   # B2：80 窗

流程：BLAS 护栏 → θ* 冻结装载（hash 校验 + EMA + gen_lowpass=None 断言）→ 逐窗固定枚举
（win_idx 0..79）→ O 批（497 行，一次 sample()/窗，显式 eps/eta）→ M-time@25 → dbar
（(62,9,62) 索引 [A,r,B]，0 槽恒等构造）→ 逐 subject（8 窗 median）+ global（80 窗
median）→ slope/AUC/r*(flip) 三 readout → 33 契约 npz + dbar_curves.npz + summary.md。
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse  # noqa: E402
import hashlib   # noqa: E402
import json      # noqa: E402
import subprocess  # noqa: E402
import sys       # noqa: E402
import time      # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch        # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import candidates as CD   # noqa: E402
import loader as LD       # noqa: E402
import metrics as MT      # noqa: E402
import readout as RO      # noqa: E402


def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=str(_REPO),
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _sha256_t(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()


def _fmt(v, spec: str = ".4g") -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return "nan" if not np.isfinite(v) else format(v, spec)
    return str(v)


def _run(stage: str) -> None:
    SC = CD.load_config()
    assert torch.cuda.is_available(), "mamba_ssm 官方 kernel 需要 CUDA"
    t_wall0 = time.perf_counter()
    is_smoke = stage == "smoke"
    out_dir = Path(SC.OUTPUTS_ROOT) / (SC.SMOKE_DIR if is_smoke else SC.RUN_DIR_OUT)
    out_dir.mkdir(parents=True, exist_ok=True)
    (Path(SC.OUTPUTS_ROOT) / SC.LOG_DIR).mkdir(parents=True, exist_ok=True)

    print(f"[{stage}] 装载 θ*（EMA + config_hash 校验）…", flush=True)
    cfg, model, ema, prior, sampler, ck = LD.load_frozen_model("cuda")
    assert sampler.gen_lowpass_hz is None, "gen_lowpass_hz 须 None（fmt 口径；fs=500 brickwall 坑）"
    coords = np.load(f"{cfg.data_root}/chan_coords_62.npy")
    _lo, _lds = LD.load_test_loader()
    wins = LD.select_windows(_lds)
    if is_smoke:
        wins = wins[:1]
    W = len(wins)
    g = len(SC.R_GRID)
    n_r = len(SC.R_SAMPLED)
    K, C, F = SC.K, SC.N_CHANNELS, cfg.future_len
    print(f"[{stage}] 窗集 {W} × 批 {SC.BATCH_ROWS} 行（62 源 × {n_r} 采样格点）× K={K}；"
          f"R_GRID={SC.R_GRID}", flush=True)

    dbar_all = np.zeros((W, SC.N_SRC, g, SC.N_CHANNELS), np.float32)
    eps_hash = np.zeros(W, dtype="<U64")
    eta_hash = np.zeros(W, dtype="<U64")
    sample_seconds = 0.0

    for w, (win_idx, ds_idx, sid, wsub) in enumerate(wins):
        xp, _xf = _lds[ds_idx]
        xp = xp.to("cuda", non_blocking=True)
        eps, eta = CD.crn_noise(win_idx, K, C, F, "cuda")
        eps_hash[w], eta_hash[w] = _sha256_t(eps), _sha256_t(eta)
        t0 = time.perf_counter()
        batch = CD.build_variant_batch(xp)
        assert batch.shape == (SC.BATCH_ROWS, C, cfg.past_len)
        with ema.swap(model):
            out = sampler.sample(batch, K, eps=eps, eta=eta)
        t_sample = time.perf_counter() - t0
        sample_seconds += t_sample
        assert out.shape == (SC.BATCH_ROWS, K, C, F)
        assert torch.isfinite(out).all(), "采样输出含非有限值"
        res = MT.compute(out.cpu().numpy().astype(np.float64))
        dbar_all[w][:, 1:, :] = res["D"].reshape(SC.N_SRC, n_r, SC.N_CHANNELS)  # [A,j,B]；0 槽恒等 0
        el = time.perf_counter() - t_wall0
        print(f"[{stage}] 窗 {w + 1}/{W}（sid={sid} win_idx={win_idx} 例内 {wsub}）"
              f" | 累计 {el / 60:.1f} min | ETA {el / (w + 1) * (W - w - 1) / 60:.1f} min"
              f" | sample {t_sample:.1f}s", flush=True)

    # ---- 落盘 dbar ----
    subj_ids = np.array([sid for _w, _ds, sid, _ws in wins])
    np.savez(out_dir / "dbar_curves.npz",
             dbar=dbar_all,
             eps_sha256=eps_hash, eta_sha256=eta_hash,
             subject_ids=subj_ids,
             win_idx=np.array([wi for wi, _d, _s, _w in wins]),
             window_in_subject=np.array([ws for _w, _d, _s, ws in wins]),
             r_grid=np.array(SC.R_GRID), K=np.int64(K), seed=np.int64(SC.SEED),
             batch_rows=np.int64(SC.BATCH_ROWS))
    wall = time.perf_counter() - t_wall0

    if is_smoke:
        d = dbar_all[0]
        assert (d[:, 0, :] == 0).all(), "dbar 0 槽须逐位 0（恒等构造）"
        assert (d[:, 1:, :] > 0).all() and np.isfinite(d).all(), "r>0 槽须正有限"
        print(f"[smoke] dbar 0 槽=0 ✓；r>0 槽正有限 ✓；wall {wall / 60:.1f} min → {out_dir}",
              flush=True)
        return

    # ---- 聚合（design §4：subject = 8 窗 median；global = 80 窗 median）----
    dbar_subj = np.stack([np.median(dbar_all[i * 8:(i + 1) * 8], axis=0)
                          for i in range(len(SC.TEST_SUBJECTS))])      # (10,62,9,62)
    dbar_glob = np.median(dbar_all, axis=0)                            # (62,9,62)

    # ---- readout 三件套 × (10 subject + global) = 33 契约 npz ----
    units = [(f"s{sid}", np.int64(sid), dbar_subj[i]) for i, sid in enumerate(SC.TEST_SUBJECTS)]
    units.append(("global", "global", dbar_glob))
    for tag, sid_val, dbar9 in units:
        dbr = np.transpose(dbar9, (0, 2, 1))          # [A, r, B] → [A, B, r]（readout 契约 r 末维）
        RO.save_ec_pert_npz(out_dir / f"ec_slope_{tag}.npz", RO.slope_readout(dbr),
                            "slope", sid_val)
        RO.save_ec_pert_npz(out_dir / f"ec_auc_{tag}.npz", RO.auc_readout(dbr),
                            "auc", sid_val)
        rstar = RO.rstar_readout(RO.dtil_curves(dbar9), SC.TAU_TILDE)   # dtil 输入原生 [A, r, B]
        RO.save_ec_pert_npz(out_dir / f"ec_rstar_{tag}.npz", RO.flip_rstar(rstar),
                            "rstar", sid_val)

    # ---- 描述统计六项（design §5；global 为主，subject 范围随附）----
    off = ~np.eye(SC.N_SRC, dtype=bool)
    slope_g = RO.slope_readout(np.transpose(dbar_glob, (0, 2, 1)))
    ratio_g = float(np.median(np.diag(slope_g)) / np.median(slope_g[off]))
    ratio_s = [float(np.median(np.diag(RO.slope_readout(np.transpose(d, (0, 2, 1))))) /
                      np.median(RO.slope_readout(np.transpose(d, (0, 2, 1)))[off]))
               for d in dbar_subj]
    dtil_g = RO.dtil_curves(dbar_glob)
    cens = {t: float((RO.rstar_readout(dtil_g, t)[off] == SC.R_NEXT).mean())
            for t in (SC.TAU_TILDE, SC.TAU_TILDE_SENS)}
    n_paths = len(RO.effective_paths(RO.rstar_readout(dtil_g, SC.TAU_TILDE)))
    dist = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))[off]
    rho_dist = float(spearmanr(slope_g[off], dist).statistic)
    slp_half1 = RO.slope_readout(np.transpose(np.median(dbar_subj[:5], axis=0), (0, 2, 1)))
    slp_half2 = RO.slope_readout(np.transpose(np.median(dbar_subj[5:], axis=0), (0, 2, 1)))
    rho_sh = float(spearmanr(slp_half1[off], slp_half2[off]).statistic)
    mono = np.ones((SC.N_SRC, SC.N_CHANNELS), dtype=bool)
    for j in range(g - 1):
        mono &= dbar_glob[:, j + 1, :] >= dbar_glob[:, j, :] * (1 - 1e-9)
    mono_frac = float(mono[off].mean())

    lines = [f"# O×M-time@25 EC 推断 — ec-omtime25-v1（v1-fmt θ* {SC.EXPECTED_CONFIG_HASH}）", "",
             f"- 窗集：{W} 窗（fold0 test 全 10 例 × 前 8 窗，win_idx 0..79）；K={K}；seed={SC.SEED}；"
             f"批 {SC.BATCH_ROWS} 行（N 单范式 × 8 采样格点）一次 sample()/窗。",
             f"- R_GRID={SC.R_GRID}（索引 0 ≡ r=0 ≡ D̄=0 恒等构造）；slope 段 {SC.SLOPE_SEG}；度量 = M-time@25（前 25 样本 = 0.1 s，用户指定）；"
             f"τ̃={SC.TAU_TILDE}（敏感 {SC.TAU_TILDE_SENS}）；R_NEXT={SC.R_NEXT}。",
             f"- 耗时：wall {wall / 60:.1f} min（sample 累计 {sample_seconds:.1f}s）。",
             f"- ckpt：epoch {ck['epoch']}，config_hash {ck['config_hash']}；"
             f"gen_lowpass_hz=None 断言过。", "",
             "## 描述统计（六项；global 为主）", "",
             f"1. **slope EC 自跨比**：global {ratio_g:.3f}"
             f"（subject 范围 {min(ratio_s):.3f}–{max(ratio_s):.3f}）——搜索任务 O×M-time @r0 自跨比 1.20（A 2.06 过线 / B 1.20 未过）。",
             f"2. **r\\* 删失率**（offdiag）：τ̃={SC.TAU_TILDE} → {cens[SC.TAU_TILDE]:.2%}；"
             f"τ̃={SC.TAU_TILDE_SENS} → {cens[SC.TAU_TILDE_SENS]:.2%}。",
             f"3. **有效路径数**（τ̃={SC.TAU_TILDE}，r\\* ≤ R_MAX={SC.R_MAX}）：{n_paths} / 3782。",
             f"4. **Spearman(slope EC, 通道距离)**（offdiag）：{rho_dist:.4f}。",
             f"5. **跨受试者 split-half**（例 #1–#5 vs #6–#10，slope EC offdiag 秩相关）："
             f"{rho_sh:.4f}。",
             f"6. **dbar 单调占比**（offdiag 逐边非减，Gate D 同构 sanity）：{mono_frac:.4f}。",
             "",
             "## 已知边界（照登）",
             "",
             "- O×M-time 口径：Gate A 过线（2.06）而 B 未过（1.20）⇒ 预期自跨比 ~1.2（强覆盖、弱归因），判读随附。",
             "- r ≥ 1.0（2×r0）高格点外推；r=0 槽为恒等构造（非采样）。",
             "- 本产物 = v1-fmt θ\\* 上的 EC\\_pert；无 GC 对照（用户拍板只出矩阵），"
             "契约 npz 已为后续对照任务预留接口。"]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    meta = {"command": f"python3 run.py {stage}", "git_head": _git_head(),
            "timestamp": datetime.now().isoformat(),
            "config": {k: v for k, v in vars(SC).items() if k.isupper()},
            "wall_seconds": wall, "sample_seconds": sample_seconds,
            "ckpt_epoch": int(ck["epoch"]), "ckpt_config_hash": ck["config_hash"],
            "euler_steps": cfg.euler_steps, "sigma_min": cfg.sigma_min,
            "gen_lowpass_hz": getattr(cfg, "gen_lowpass_hz", 40.0)}
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=str)
    print(f"[{stage}] 完成：wall {wall / 60:.1f} min → {out_dir}", flush=True)
    print(f"[{stage}] 描述统计：slope 自跨比 {ratio_g:.3f}；删失率 τ̃0.25 "
          f"{cens[SC.TAU_TILDE]:.1%}；有效路径 {n_paths}；split-half ρ {rho_sh:.3f}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["smoke", "main"])
    args = p.parse_args()
    _run(args.stage)


if __name__ == "__main__":
    main()
