#!/usr/bin/env python3
"""run.py — pastdep-audit-f0 入口（E1–E4；design §2）。

用法：
  cd models/pastdep-audit-v1
  python3 run.py smoke                       # 1 窗机械正确性
  nohup python3 run.py main > ../../outputs/pastdep-audit-f0/logs/main.log 2>&1 &

流程：BLAS 护栏 → θ\* 装载 → E1a 解析（prior.A 包络 T50_prior + 子窗质量）→
逐窗：E1b/E2/E4 批（435 行，一次 sample()）存 |δ̄| 曲线 + E3 四臂独立 seed 采样
在线算能量距离 → summary.md 判据表（三假说裁决）。
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
from scipy.spatial.distance import cdist  # noqa: E402

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import candidates as CD   # noqa: E402
import loader as LD       # noqa: E402


def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=str(_REPO), stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _sha256_t(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()


def _t50(c: np.ndarray) -> float:
    """峰后半衰（样本）：首个 f≥argmax 使 c[f] ≤ peak/2；无则 F-1。"""
    pk = int(np.argmax(c))
    below = np.where(c[pk:] <= c[pk] / 2.0)[0]
    return float(pk + below[0]) if len(below) else float(len(c) - 1)


def _energy_dist(x: np.ndarray, y: np.ndarray) -> float:
    """128 维向量样本的能量距离（K×128 各两组）。"""
    dxy = cdist(x, y); dxx = cdist(x, x); dyy = cdist(y, y)
    n, m = len(x), len(y)
    return float(2 * dxy.mean() - dxx.sum() / (n * (n - 1)) - dyy.sum() / (m * (m - 1)))


def _e1a_prior(prior) -> dict:
    SC = CD.load_config()
    A = prior.A.detach().cpu().numpy().astype(np.float64)      # (F, P)
    assert A.shape == (SC.FUTURE_LEN, SC.PAST_LEN), f"prior.A 形状 {A.shape}"
    env = np.abs(A).sum(axis=1)                                 # ‖A[f,:]‖₁ 包络
    pk = int(np.argmax(env))
    below = np.where(env[pk:] <= env[pk] / 2.0)[0]
    t50_prior = float(pk + below[0]) if len(below) else float(SC.FUTURE_LEN - 1)
    colmass = np.abs(A).sum(axis=0)                             # 逐 past 位置权重
    submass = np.array([colmass[lo:hi].sum() for lo, hi in SC.SUBWINDOWS])
    submass = submass / submass.sum()
    return {"env": env.astype(np.float32), "t50_prior": t50_prior, "submass": submass,
            "gamma_pred_t50": np.log(2.0) / (np.pi / 25.0)}       # OU 解析预测（对拍用）


def _run(stage: str) -> None:
    SC = CD.load_config()
    assert torch.cuda.is_available()
    t0 = time.perf_counter()
    is_smoke = stage == "smoke"
    out_dir = Path(SC.OUTPUTS_ROOT) / (SC.SMOKE_DIR if is_smoke else SC.RUN_DIR_OUT)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{stage}] 装载 θ*…", flush=True)
    cfg, model, ema, prior, sampler, ck = LD.load_frozen_model("cuda")
    assert sampler.gen_lowpass_hz is None
    e1a = _e1a_prior(prior)
    print(f"[{stage}] E1a 解析：env 峰@f={int(np.argmax(e1a['env']))}，"
          f"T50_prior={e1a['t50_prior']:.1f} 样本，OU 预测 {e1a['gamma_pred_t50']:.2f}；"
          f"子窗质量 {np.round(e1a['submass'], 4)}", flush=True)
    np.savez(out_dir / "A_prior.npz", env=e1a["env"], submass=e1a["submass"],
             t50_prior=e1a["t50_prior"], gamma_pred_t50=e1a["gamma_pred_t50"])

    _lo, lds = LD.load_test_loader()
    wins_full = LD.select_windows(lds)
    # E3 置换映射（design §2：同人 w+1 例内循环；跨人 = 下一受试者同例内窗）——基于全量窗表
    ds_of = {w[0]: w[1] for w in wins_full}
    same_map = {wi: ds_of[(wi // 8) * 8 + (wi % 8 + 1) % 8] for wi in ds_of}
    cross_map = {wi: ds_of[((wi // 8 + 1) % 3) * 8 + wi % 8] for wi in ds_of}
    wins = wins_full[:1] if is_smoke else wins_full
    W = len(wins)
    K, C, F = SC.K, SC.N_CHANNELS, cfg.future_len
    FUT = cfg.future_len
    nd, nw, ns = len(SC.DOSES), len(SC.SUBWINDOWS), SC.N_SRC
    print(f"[{stage}] 窗 {W} × 批 {SC.BATCH_ROWS} 行（{SC.ARMS}）；E3 四臂独立 seed", flush=True)

    # 结果容器（|δ̄| 曲线 + AUC；口径见 design §2/§3）
    e1_self = np.zeros((W, nd, ns, FUT), np.float32)       # 剂量臂 自响应
    e1_crossmed = np.zeros((W, nd, FUT), np.float32)       # 剂量臂 offdiag 靶中位
    e2_self = np.zeros((W, nw, ns, FUT), np.float32)
    e2_targetmed = np.zeros((W, nw, FUT), np.float32)      # 子窗臂 全靶中位
    e1_auc = np.zeros((W, nd, ns, C), np.float32)          # 逐 (源,靶) 全段 AUC
    e2_auc = np.zeros((W, nw, ns, C), np.float32)
    e3_dist = np.zeros((W, 4, C), np.float32)              # noise/same/cross/ablate vs ref
    eps_hash = np.zeros(W, dtype="<U64")

    for w, (win_idx, ds_idx, sid, wsub) in enumerate(wins):
        xp, _xf = lds[ds_idx]
        xp = xp.to("cuda", non_blocking=True) if xp.dim() == 3 else xp.unsqueeze(0).to("cuda")
        # ---- E1b/E2/E4 批（一次 sample）----
        eps, eta = CD.crn_noise(SC.SEED + win_idx, K, C, F, "cuda")
        eps_hash[w] = _sha256_t(eps)
        batch = CD.build_batch(xp)
        with ema.swap(model):
            out = sampler.sample(batch, K, eps=eps, eta=eta).float().cpu().numpy()
        assert out.shape == (SC.BATCH_ROWS, K, C, FUT) and np.isfinite(out).all()
        base = out[0]
        for a in range(ns):
            for k, arm in enumerate(SC.ARMS):
                dbar = np.abs((out[CD.variant_index(a, k)] - base).mean(axis=0))  # (C, FUT)
                if arm[0] == "dose":
                    di = SC.DOSES.index(arm[1])
                    e1_self[w, di, a] = dbar[a]
                    e1_crossmed[w, di] += np.median(dbar[np.arange(C) != a], axis=0) / ns
                    e1_auc[w, di, a] = dbar.mean(axis=1)
                else:
                    wi_ = arm[1]
                    e2_self[w, wi_, a] = dbar[a]
                    e2_targetmed[w, wi_] += np.median(dbar, axis=0) / ns
                    e2_auc[w, wi_, a] = dbar.mean(axis=1)
        # ---- E3 四臂（独立 seed；在线能量距离）----
        eps_r, eta_r = CD.crn_noise(SC.SEED + win_idx + SC.ARM_SEED_OFF["ref"], K, C, F, "cuda")
        ref = sampler.sample(xp, K, eps=eps_r, eta=eta_r)[0].float().cpu().numpy()
        for ai, arm in enumerate(("noise", "same", "cross", "ablate")):
            if arm == "noise":
                xp_a = xp
            elif arm == "same":
                xp_a = lds[same_map[win_idx]][0]
            elif arm == "cross":
                xp_a = lds[cross_map[win_idx]][0]
            else:
                xp_a = torch.zeros_like(xp)
            xp_a = xp_a.to("cuda") if xp_a.dim() == 3 else xp_a.unsqueeze(0).to("cuda")
            eps_a, eta_a = CD.crn_noise(SC.SEED + win_idx + SC.ARM_SEED_OFF[arm], K, C, F, "cuda")
            with ema.swap(model):
                y = sampler.sample(xp_a, K, eps=eps_a, eta=eta_a)[0].float().cpu().numpy()
            for ch in range(C):
                e3_dist[w, ai, ch] = _energy_dist(ref[:, ch], y[:, ch])
        el = time.perf_counter() - t0
        print(f"[{stage}] 窗 {w + 1}/{W}（sid={sid} win_idx={win_idx}）| {el / 60:.1f} min",
              flush=True)

    np.savez_compressed(out_dir / "curves.npz", e1_self=e1_self, e1_crossmed=e1_crossmed,
                        e2_self=e2_self, e2_targetmed=e2_targetmed, e1_auc=e1_auc,
                        e2_auc=e2_auc, eps_sha256=eps_hash,
                        subject_ids=np.array([s for _w, _d, s, _x in wins]),
                        win_idx=np.array([wi for wi, _d, _s, _x in wins]))
    np.savez(out_dir / "e3.npz", dist=e3_dist,
             arms=np.array(["noise", "same", "cross", "ablate"]))

    if is_smoke:
        assert (e1_self[0, 0] > 0).all() and np.isfinite(e1_self).all()
        assert np.isfinite(e3_dist).all() and np.median(e3_dist[0, 0]) > 0, \
            "E3 噪声臂距离中位须为正（有限样本估计可微负）"
        print(f"[smoke] 曲线正有限 ✓；E3 距离正 ✓ → {out_dir}", flush=True)
        return

    # ---- 判据表（design §2 预注册）----
    t50_prior = e1a["t50_prior"]
    t50_s = {d: float(np.median([_t50(e1_self[w, di, a]) for w in range(W)
                                 for a in range(ns)]))
             for di, d in enumerate(SC.DOSES)}
    t50_c = {d: float(np.median([_t50(e1_crossmed[w, di]) for w in range(W)]))
             for di, d in enumerate(SC.DOSES)}
    phi = {d: float(np.median([e1_self[w, di, a][:SC.MTIME_EARLY_SAMPLES].sum()
                               / max(e1_self[w, di, a].sum(), 1e-12)
                               for w in range(W) for a in range(ns)]))
           for di, d in enumerate(SC.DOSES)}
    rel = min(abs(t50_s[d] / t50_prior - 1.0) for d in SC.DOSES)
    e2_resp = e2_targetmed.mean(axis=(0, 2))                # (4,) 各子窗全段响应（靶中位×时间均值）
    e2_self_auc = e2_self.mean(axis=(0, 2))                 # (4,) 子窗自响应 AUC 均值
    d0 = np.median(e3_dist[:, 0, :], axis=1)                # 噪声地板（窗中位）
    rho = {arm: float(np.median(np.median(e3_dist[:, ai, :], axis=1) / d0))
           for ai, arm in enumerate(("noise", "same", "cross", "ablate"))}
    rho_iqr = {arm: float(np.percentile(np.median(e3_dist[:, ai, :], axis=1) / d0, 75)
                          - np.percentile(np.median(e3_dist[:, ai, :], axis=1) / d0, 25))
               for ai, arm in enumerate(("noise", "same", "cross", "ablate"))}
    h1 = rel < SC.T50_REL_TOL
    h3 = rho["same"] < SC.RHO_BANDS[0]
    h2 = rho["same"] >= SC.RHO_BANDS[0]

    lines = [f"# past 依赖审计（E1–E4）— pastdep-audit-v1（θ\\* {SC.EXPECTED_CONFIG_HASH}）", "",
             f"- 窗 {W} × 批 {SC.BATCH_ROWS} 行 + E3 四臂；wall {(time.perf_counter() - t0) / 60:.1f} min；"
             f"ckpt epoch {ck['epoch']}。",
             "",
             "## E1 双通路对拍",
             "",
             f"- 解析：env 峰@f={int(np.argmax(e1a['env']))}，**T50_prior = {t50_prior:.1f} 样本"
             f"（{t50_prior / SC.FS * 1000:.0f} ms）**；OU 零参数预测 {e1a['gamma_pred_t50']:.2f} 样本。",
             f"- 采样自响应 T50（中位）：" + "；".join(f"r={d:g} → {t50_s[d]:.1f}" for d in SC.DOSES),
             f"- 采样跨响应 T50（offdiag 靶中位）：" + "；".join(f"r={d:g} → {t50_c[d]:.1f}"
                                                              for d in SC.DOSES),
             f"- 对拍偏差（最优剂量）= {rel:.1%}（判据 ±{SC.T50_REL_TOL:.0%}）→ "
             f"**{'H1 支持（先验主导短记忆）' if h1 else 'H1 不支持：采样衰减偏离解析，速度网络携带额外记忆'}**",
             f"- φ（前 32 样本响应质量分数，自响应中位）：" +
             "；".join(f"r={d:g} → {phi[d]:.2f}" for d in SC.DOSES),
             "",
             "## E2 注入位置扫描（r=1.0，靶中位响应 × 自响应 AUC）",
             "",
             f"- 解析子窗质量（A 列权重归一）：{np.round(e1a['submass'], 4)}",
             f"- 采样靶中位响应：{np.round(e2_resp, 4)}；自响应 AUC：{np.round(e2_self_auc, 4)}",
             f"- 判读：{'仅晚子窗显著（短记忆）' if e2_resp[3] > 5 * max(e2_resp[:3]) else '早/中子窗亦有响应（存在长程利用）'}",
             "",
             "## E3 past 置换/消融（能量距离 ρ = d 臂/d 噪声地板；窗中位 [IQR])",
             "",
             f"- 噪声地板 d₀ 中位 = {np.median(d0):.4f}",
             f"- ρ(same) = {rho['same']:.2f} [{rho_iqr['same']:.2f}]；"
             f"ρ(cross) = {rho['cross']:.2f} [{rho_iqr['cross']:.2f}]；"
             f"ρ(ablate) = {rho['ablate']:.2f} [{rho_iqr['ablate']:.2f}]",
             f"- 判读（带 {SC.RHO_BANDS}）：same {'<1.2 ⇒ past 身份不影响 future 分布（支持 H3 面）'
                if h3 else '≥1.2 ⇒ 存在扰动测不到的信息依赖（支持 H2）'}；"
             f"cross {'> same ⇒ 个体级统计被使用' if rho['cross'] > rho['same'] + 0.1 else '≈ same ⇒ 个体差异未显式使用'}",
             "",
             "## 三假说裁决",
             "",
             f"- H1 结构性短记忆：**{'支持' if h1 else '不支持'}**（T50 对拍 {rel:.1%}）",
             f"- H2 信息依赖>敏感性：**{'支持' if h2 else '不支持'}**（ρ_same={rho['same']:.2f}）",
             f"- H3 几乎不依赖：**{'支持' if h3 else '不支持'}**（ρ_same={rho['same']:.2f}）"]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    meta = {"command": f"python3 run.py {stage}", "git_head": _git_head(),
            "timestamp": datetime.now().isoformat(),
            "config": {k: v for k, v in vars(SC).items() if k.isupper()},
            "ckpt_epoch": int(ck["epoch"]), "ckpt_config_hash": ck["config_hash"]}
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=str)
    print(f"[{stage}] 完成 → {out_dir}", flush=True)
    print("\n".join(lines[9:]), flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["smoke", "main"])
    args = p.parse_args()
    _run(args.stage)


if __name__ == "__main__":
    main()
