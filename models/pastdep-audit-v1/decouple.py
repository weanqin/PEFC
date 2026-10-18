#!/usr/bin/env python3
"""decouple.py — μ/前缀解耦交叉注入（Run 002；design §5）。

六臂（每窗 K=16）：R=(A,μ_A)、X_same=(B_same,μ_A)、X_cross=(B_cross,μ_A)、
M_B=(B_same,μ_B)、M_rand=(A, μ_A 通道置换)、Z=(0, μ_A)。双噪声集（配对 |δ̄| 谱 +
独立能量距离）；R 臂与原 sampler.sample 逐位相等断言（解耦实现的有效性证明）。
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "8")

import json          # noqa: E402
import sys           # noqa: E402
import time          # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np   # noqa: E402
import torch         # noqa: E402
from scipy.spatial.distance import cdist      # noqa: E402
from scipy.stats import wilcoxon              # noqa: E402

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import candidates as CD   # noqa: E402
import loader as LD       # noqa: E402
from run import _energy_dist, _t50   # noqa: E402

SEED_OFF = {"r1": 5000, "r2": 5050, "xs": 5200, "xc": 5300, "mb": 5400,
            "mr": 5500, "z": 5600, "pair": 5700}
MU_PERM_SEED = 99
ARMS = ("xs", "xc", "mb", "mr", "z")      # 能量距离臂（地板 = r2 vs r1）


def sample_decoupled(sampler, prefix: torch.Tensor, mu: torch.Tensor,
                     eps: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
    """逐位复刻 EulerSampler.sample，唯二差异：μ 由外部注入（不从前缀现算）。

    prefix (B,C,P)；mu (B,C,F)；eps/eta (B,K,C,F)。与原实现同式：
    x0 = bandlimit(μ + L_s ε L_tᵀ) + σ_min·η；Euler 步 v_θ(x, t, X_past=prefix)。"""
    pr = sampler.prior
    B, Cn, P = prefix.shape
    F_ = pr.future_len
    x0 = pr.bandlimit(mu.unsqueeze(1) + torch.einsum("ij,bqjk,lk->bqil", pr.L_s, eps, pr.L_t))
    if sampler.sigma_min > 0:
        x0 = x0 + sampler.sigma_min * eta
    dt = 1.0 / sampler.steps
    K = eps.shape[1]
    outs = []
    for c0 in range(0, B * K, sampler.chunk_bk):
        c1 = min(c0 + sampler.chunk_bk, B * K)
        x = x0.reshape(B * K, Cn, F_)[c0:c1].contiguous()
        xp = prefix.unsqueeze(1).expand(B, K, Cn, P).reshape(B * K, Cn, P)[c0:c1].contiguous()
        for i in range(sampler.steps):
            t = torch.full((x.shape[0],), i * dt, device=x.device, dtype=x.dtype)
            x = x + dt * sampler.model(xp, x, t)
        outs.append(x)
    out = torch.cat(outs, dim=0).reshape(B, K, Cn, F_)
    assert sampler.gen_lowpass_hz is None
    return out


def _run() -> None:
    SC = CD.load_config()
    assert torch.cuda.is_available()
    t0 = time.perf_counter()
    out_dir = Path(SC.OUTPUTS_ROOT) / SC.RUN_DIR_OUT
    cfg, model, ema, prior, sampler, ck = LD.load_frozen_model("cuda")
    K, C = SC.K, SC.N_CHANNELS
    F_ = cfg.future_len
    _lo, lds = LD.load_test_loader()
    wins = LD.select_windows(lds)
    ds_of = {w[0]: w[1] for w in wins}
    same_map = {wi: ds_of[(wi // 8) * 8 + (wi % 8 + 1) % 8] for wi in ds_of}
    cross_map = {wi: ds_of[((wi // 8 + 1) % 3) * 8 + wi % 8] for wi in ds_of}
    perm = np.random.default_rng(MU_PERM_SEED).permutation(SC.N_CHANNELS)
    perm_t = torch.as_tensor(perm, device="cuda")

    W = len(wins)
    dist = np.zeros((W, len(ARMS) + 1, C), np.float32)      # [floor, xs, xc, mb, mr, z]
    pdiff = np.zeros((W, len(ARMS), C, F_), np.float32)     # 配对 |δ̄(t)|（对 R）
    print(f"[decouple] 窗 {W} × 六臂 × 双噪声集；μ 置换 seed={MU_PERM_SEED}", flush=True)

    for w, (win_idx, ds_idx, sid, _wsub) in enumerate(wins):
        def _xp(dsi):
            x = lds[dsi][0]
            x = x if x.dim() == 3 else x.unsqueeze(0)
            return x.to("cuda", non_blocking=True)

        xpA = _xp(ds_idx)
        xpB = _xp(same_map[win_idx])
        xpC = _xp(cross_map[win_idx])
        with torch.no_grad():
            muA = prior.posterior_mean(xpA)
            muB = prior.posterior_mean(xpB)
            muR = muA[:, perm_t]

            def _sd(prefix, mu, seed):
                eps, eta = CD.crn_noise(SC.SEED + win_idx + seed, K, C, F_, "cuda")
                with ema.swap(model):
                    return sample_decoupled(sampler, prefix, mu, eps, eta)[0] \
                        .float().cpu().numpy()

            # 独立集：R1/R2（地板）+ 五臂
            r1 = _sd(xpA, muA, SEED_OFF["r1"])
            r2 = _sd(xpA, muA, SEED_OFF["r2"])
            arm_x = {"xs": _sd(xpB, muA, SEED_OFF["xs"]),
                     "xc": _sd(xpC, muA, SEED_OFF["xc"]),
                     "mb": _sd(xpB, muB, SEED_OFF["mb"]),
                     "mr": _sd(xpA, muR, SEED_OFF["mr"]),
                     "z": _sd(torch.zeros_like(xpA), muA, SEED_OFF["z"])}
            # 配对集：R 与五臂共享噪声
            eps_p, eta_p = CD.crn_noise(SC.SEED + win_idx + SEED_OFF["pair"], K, C, F_, "cuda")
            with ema.swap(model):
                rp = sample_decoupled(sampler, xpA, muA, eps_p, eta_p)
                # 有效性断言：解耦版(前缀=μ源) == 原 sampler.sample 逐位
                assert torch.equal(rp, sampler.sample(xpA, K, eps=eps_p, eta=eta_p)), \
                    "解耦实现与原 sampler 非逐位相等（有效性证明失败）"
                rp = rp[0].float().cpu().numpy()
                pd = {"xs": sample_decoupled(sampler, xpB, muA, eps_p, eta_p),
                      "xc": sample_decoupled(sampler, xpC, muA, eps_p, eta_p),
                      "mb": sample_decoupled(sampler, xpB, muB, eps_p, eta_p),
                      "mr": sample_decoupled(sampler, xpA, muR, eps_p, eta_p),
                      "z": sample_decoupled(sampler, torch.zeros_like(xpA), muA, eps_p, eta_p)}
            dist[w, 0] = [ _energy_dist(r1[:, ch], r2[:, ch]) for ch in range(C) ]
            for ai, a in enumerate(ARMS):
                y = arm_x[a]
                dist[w, ai + 1] = [_energy_dist(r1[:, ch], y[:, ch]) for ch in range(C)]
                pdiff[w, ai] = np.abs((pd[a][0].float().cpu().numpy() - rp).mean(axis=0))
        print(f"[decouple] 窗 {w + 1}/{W} | {(time.perf_counter() - t0) / 60:.1f} min", flush=True)

    np.savez_compressed(out_dir / "decouple.npz", dist=dist, pdiff=pdiff,
                        arms=np.array(["floor"] + list(ARMS)),
                        mu_perm=perm, seed_off=np.array([SEED_OFF[k] for k in SEED_OFF]),
                        subject_ids=np.array([s for _w, _d, s, _x in wins]))

    # ---- 判据（design §5 预注册）----
    d0 = np.median(dist[:, 0, :], axis=1)                    # 地板（窗中位）
    names = {"xs": "X_same=(B,μ_A)", "xc": "X_cross=(B_cross,μ_A)", "mb": "M_B=(B,μ_B)",
             "mr": "M_rand=(A,μ_rand)", "z": "Z=(0,μ_A)"}
    res = {}
    for ai, a in enumerate(ARMS):
        dmed = np.median(dist[:, ai + 1, :], axis=1)
        ratios = dmed / d0
        rho = float(np.median(ratios))
        iqr = float(np.percentile(ratios, 75) - np.percentile(ratios, 25))
        boot = np.array([np.median(ratios[np.random.default_rng(1000 + i).integers(0, W, W)])
                         for i in range(1000)])
        ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
        stat, p = wilcoxon(dmed, d0)
        res[a] = {"rho": rho, "iqr": iqr, "ci": ci, "p": float(p),
                  "t50": float(_t50(np.median(pdiff[:, ai], axis=(0, 1)))),
                  "pdiff_med": float(np.median(pdiff[:, ai]))}
    # Holm（5 臂）
    order = sorted(ARMS, key=lambda a: res[a]["p"])
    m = len(ARMS)
    holm = {}
    for i, a in enumerate(order):
        holm[a] = min(1.0, (m - i) * res[a]["p"])

    lines = [f"# Run 002 — μ/前缀解耦交叉注入（θ\\* {SC.EXPECTED_CONFIG_HASH}）", "",
             f"- 窗 {W}；wall {(time.perf_counter() - t0) / 60:.1f} min；解耦实现与原 sampler "
             f"逐位相等断言 24/24 过；μ 通道置换 seed={MU_PERM_SEED}。",
             f"- 噪声地板 d₀ 中位 = {np.median(d0):.4f}", "",
             "| 臂 | ρ 中位 [IQR] | bootstrap 95% CI | Wilcoxon p (Holm) | 配对 \\|δ̄\\| 中位 | t50 |",
             "|---|---|---|---|---|---|"]
    for a in ARMS:
        r = res[a]
        lines.append(f"| {names[a]} | {r['rho']:.2f} [{r['iqr']:.2f}] | "
                     f"[{r['ci'][0]:.2f}, {r['ci'][1]:.2f}] | {r['p']:.1e} ({holm[a]:.1e}) | "
                     f"{r['pdiff_med']:.4f} | {r['t50']:.0f} |")
    xs, mb, z = res["xs"], res["mb"], res["z"]
    verdict_h = xs["rho"] < 1.2
    lines += ["",
              f"- **裁决（预注册带 1.2/2.0）**：ρ(X_same)={xs['rho']:.2f} ⇒ "
              f"{'<1.2：(B,μ_A) 与 (A,μ_A) 分布不可区分 ⇒ H\\* 行为推论成立' if verdict_h else ('≥2：前缀直接影响显著 ⇒ 否定 H\\*' if xs['rho'] >= 2 else '1.2–2 带：部分依赖')}",
              f"- 分解：d(Z)/d(M_B) = {z['rho'] / max(mb['rho'], 1e-9):.2f}（前缀消融位移 / 完整换 past 位移）——"
              f"前缀通路占完整 past 信息的 {z['rho'] / max(mb['rho'], 1e-9):.0%} 量级（中位比）。",
              f"- μ 必要性：ρ(M_rand)={res['mr']['rho']:.2f}（打乱 μ 的位移）。"]
    (out_dir / "decouple_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    meta = {"command": "python3 decouple.py", "timestamp": datetime.now().isoformat(),
            "seed_off": SEED_OFF, "mu_perm_seed": MU_PERM_SEED,
            "ckpt_config_hash": ck["config_hash"]}
    with open(out_dir / "decouple_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    _run()
