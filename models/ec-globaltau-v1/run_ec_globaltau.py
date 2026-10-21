"""EC_pert runner — 07-04-global-tau 方案 × gpfm-mamba-v5-main 代理脑（任务 10-21-ec-globaltau-v5main）。

镜像 `latent_flow/scripts/run_oc_connectivity_global_tau.py` 两阶段主流程：
  Stage1：σ_ref=0.3 每源一遍前向 → (S,C) raw response → τ_global = Quantile_q（q=0.5）；
  Stage2：逐源 log_grid(0.01,0.3)×24 扫描 → walk_monotone_curve（标量全局 τ，clamp）→ σ*；
  强度映射 conn = −(log σ* + ½log 2πe)；sign = sign(mean-shift) ±1/0 → EC_pert = sign ⊙ conn。
σ 搜索/阈值/映射原语**直接 import** `latent_flow.attribution.bisect_sigma`（不自实现）。

用户钉定（PRD 硬约束）：扰动施加点 = z-score 后、入模前、x^p 段；入模 x_0 恒为 x^p_pert；
x^f_pert 仅由 x^p_pert 采样生成；一切 EC 设置复用 07-04。
θ\* 口径（briefing 2026-10-21）：EMA swap 全前向 + bf16 autocast（对 gpfm 正典 fp32 为登记偏离）。

用法（BLAS 守卫 env 前置 + MODMA_DATA_ROOT 不需要——本路径数据为 edra v4）：
    OMP_NUM_THREADS=8 ... python3 models/ec-globaltau-v1/run_ec_globaltau.py \
        --subject-idx 0 --subjects 1 --n-trials 32 \
        --output-dir /root/npi/outputs/10-21-ec-globaltau-v5main/run_01
    --selfcheck：只跑 S1 断言（加载/EMA swap 生效/ξ·(eps,eta) 逐位一致/扰动隔离/同输入跨调用逐位一致）。
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "/root/npi/models/latent-flow-v1")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from latent_flow.attribution.bisect_sigma import (
    _extract_mse,
    compute_global_tau,
    log_grid,
    walk_monotone_curve,
)
from latent_flow.scripts._connectivity_driver import (
    json_default,
    sanitize_for_json,
    subsample_trials,
)

from adapter import DEFAULT_CKPT, DEFAULT_RUN_DIR, FOLD0_TEST_IDS, load_surrogate, subject_global_windows
from probe_site import GpfmProbeSite

# conn 公式（用户拍板 2026-10-21，Run 002 起；07-04 原式为 -(log σ* + ½log 2πe)）
_LOG_2PIE = math.log(2.0 * math.pi * math.e)

DEFAULT_Q = 0.75
DEFAULT_SIGMA_REF = 0.3
DEFAULT_N_GRID = 24
SIGMA_MIN, SIGMA_MAX = 0.01, 0.3
DEFAULT_N_TRIALS = 32
AMP_DTYPE = torch.bfloat16   # briefing 拍板 2026-10-21（EMA + bf16；对 gpfm 正典 fp32 为登记偏离）


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default=DEFAULT_CKPT)
    p.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    p.add_argument("--fold", type=int, default=0, help="仅 fold0（v1 固化 manifest）")
    p.add_argument("--subject-idx", type=int, default=0)
    p.add_argument("--subjects", type=int, default=1)
    p.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS,
                   help="每例窗数上限（07-04 同参数；<=0 全量）")
    p.add_argument("--q", type=float, default=DEFAULT_Q)
    p.add_argument("--sigma-ref", type=float, default=DEFAULT_SIGMA_REF)
    p.add_argument("--sigma-min", type=float, default=SIGMA_MIN)
    p.add_argument("--sigma-max", type=float, default=SIGMA_MAX)
    p.add_argument("--n-grid", type=int, default=DEFAULT_N_GRID)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", default="")
    p.add_argument("--selfcheck", action="store_true",
                   help="S1 断言（不跑流水线），全过退出码 0")
    return p.parse_args()


# ---------------------------------------------------------------------------
# S1 断言（--selfcheck）
# ---------------------------------------------------------------------------

def _tensor_digest(t: torch.Tensor) -> str:
    return str(hash(t.detach().float().cpu().numpy().tobytes()))


def run_selfcheck(args, device):
    print("[S1] load surrogate (EMA shadow 装载) ...", flush=True)
    cfg, model, ema, sampler, test_ds = load_surrogate(args.checkpoint, args.run_dir, device)
    sid = FOLD0_TEST_IDS[args.subject_idx]
    widx = subject_global_windows(test_ds, sid)
    x_p = torch.stack([test_ds[gi][0] for gi in widx]).to(device)
    # n_trials 子采样（07-04 同机制）
    x_p, n_used = subsample_trials(x_p, args.n_trials, args.seed, args.subject_idx)
    N, C, P = x_p.shape
    F = sampler.prior.future_len
    print(f"[S1] subject s{sid}: windows {N} (n_used={n_used}) shape {tuple(x_p.shape)}", flush=True)

    gen = torch.Generator(device=device).manual_seed(args.seed * 10000 + sid)
    eps = torch.randn(N, 1, C, F, device=device, generator=gen)
    eta = torch.randn(N, 1, C, F, device=device, generator=gen)

    # 断言②：扰动隔离 —— 扰动仅落 x^p 通道 a（clone + 单通道构造后逐位核对）
    a = 0
    xi_probe = torch.randn(N, P, device=device,
                           generator=torch.Generator(device=device).manual_seed(args.seed + a))
    x_pert = x_p.clone()
    x_pert[:, a, :] = x_pert[:, a, :] + 0.1 * xi_probe
    mask = torch.ones(C, dtype=torch.bool, device=device); mask[a] = False
    assert torch.equal(x_pert[:, mask, :], x_p[:, mask, :]), "扰动泄漏到非源通道"
    print("[S1] assert perturbation-isolation: PASS", flush=True)

    site = GpfmProbeSite(sampler, x_p, eps, eta, seed=args.seed)
    site_raw = GpfmProbeSite(sampler, x_p, eps, eta, seed=args.seed)
    base_raw = site_raw.compute_baseline()   # swap 域外 = raw weights（对照）
    with ema.swap(model):
        base_in = site.compute_baseline()
        # 断言：同输入同批形跨调用逐位一致（替代 perturb-ec-pipeline D(r=0)=0；评审建议项）
        base_again = site.compute_baseline()
        assert torch.equal(base_in, base_again), "同输入跨调用输出不一致（基线）"
        print("[S1] assert same-input-cross-call bitwise (baseline): PASS", flush=True)
        # 断言：EMA swap 生效（swap 域内输出 ≠ raw 权重输出）
        diff = (base_in - base_raw).abs().max().item()
        assert diff > 0, "EMA swap 未生效（swap 内外输出相同）"
        print(f"[S1] assert EMA-swap-effective: PASS (max|Δ|={diff:.3e})", flush=True)
        # 断言①：ξ 跨 σ 逐位一致（set_source 后 ξ 摘要不变）
        site.set_source(a)
        d_before = _tensor_digest(site._xi)
        m1 = site.perturb_and_measure(0.05, a)
        d_mid = _tensor_digest(site._xi)
        m2 = site.perturb_and_measure(0.20, a)
        d_after = _tensor_digest(site._xi)
        assert d_before == d_mid == d_after, "ξ 随 σ 变动（违反 CRN）"
        print("[S1] assert xi-bitwise-across-sigma: PASS", flush=True)
        # 断言②补：(eps,eta) 跨调用逐位一致
        assert (_tensor_digest(site._eps), _tensor_digest(site._eta)) == (
            _tensor_digest(eps), _tensor_digest(eta)), "(eps,eta) 被改动"
        print("[S1] assert eps-eta-bitwise: PASS", flush=True)
        # 断言：响应随 σ 单调性抽查（MSE(σ 大) ≥ MSE(σ 小) 的靶数占优——软检查，只打印）
        frac = float((m2[0] >= m1[0]).float().mean().item())
        print(f"[S1] info: frac targets with MSE(0.20)≥MSE(0.05) = {frac:.3f}", flush=True)
    print("[S1] ALL PASS", flush=True)


# ---------------------------------------------------------------------------
# 两阶段主流程（镜像 run_oc_connectivity_global_tau.py）
# ---------------------------------------------------------------------------

def _stage1_response_scan(site, sources, sigma_ref, device, amp_dtype):
    response = np.full((sources, site.C), float("nan"), dtype=np.float32)
    mean_diff = np.full((sources, site.C), float("nan"), dtype=np.float32)
    for src in range(sources):
        site.set_source(src)
        result = site.perturb_and_measure(float(sigma_ref), src)
        mse_vec, md_vec = result
        response[src, :] = mse_vec.detach().cpu().numpy()
        mean_diff[src, :] = md_vec.detach().cpu().numpy()
    return torch.from_numpy(response), torch.from_numpy(mean_diff)


def _stage2_sigma_search(site, sources, grid, tau_global, sigma_min, sigma_max):
    sigma_matrix = np.full((sources, site.C), float("nan"), dtype=np.float32)
    for src in range(sources):
        site.set_source(src)
        mse_curve = torch.stack(
            [_extract_mse(site.perturb_and_measure(float(s), src)) for s in grid], dim=0
        )  # (n_grid, C)
        sigma_star = walk_monotone_curve(
            mse_curve, grid,
            torch.full((site.C,), float(tau_global)),
            sigma_min, sigma_max)
        sigma_matrix[src, :] = sigma_star.numpy()
    return sigma_matrix


def main():
    args = parse_args()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}", flush=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.selfcheck:
        run_selfcheck(args, device)
        return

    cfg, model, ema, sampler, test_ds = load_surrogate(args.checkpoint, args.run_dir, str(device))
    sid = FOLD0_TEST_IDS[args.subject_idx]
    widx = subject_global_windows(test_ds, sid)
    x_p_full = torch.stack([test_ds[gi][0] for gi in widx]).to(device)
    x_p, n_used = subsample_trials(x_p_full, args.n_trials, args.seed, args.subject_idx)
    if n_used != x_p_full.shape[0]:
        print(f"  subsampled {n_used}/{x_p_full.shape[0]} windows", flush=True)
    N, C, P = x_p.shape
    F = sampler.prior.future_len
    sources = C

    grid = log_grid(args.sigma_min, args.sigma_max, args.n_grid)
    out_dir = args.output_dir or f"/root/npi/outputs/10-21-ec-globaltau-v5main/run_01"
    os.makedirs(out_dir, exist_ok=True)
    print(f"Paradigm: global-quantile-tau (sigma* preserved, 07-04 verbatim) | "
          f"q={args.q} sigma_ref={args.sigma_ref} grid=log[{args.sigma_min},{args.sigma_max}] n={args.n_grid}",
          flush=True)
    print(f"Subject: s{sid} | windows {n_used} | sources {sources} | out {out_dir}", flush=True)

    # 生成噪声：被试级固定（跨源/跨 σ 逐位一致）
    gen = torch.Generator(device=device).manual_seed(args.seed * 10000 + sid)
    eps = torch.randn(N, 1, C, F, device=device, generator=gen)
    eta = torch.randn(N, 1, C, F, device=device, generator=gen)

    site = GpfmProbeSite(sampler, x_p, eps, eta, seed=args.seed)
    subj_t0 = time.time()
    with ema.swap(model):   # θ* 正典口径：全部前向（baseline+扰动）在 EMA swap 域内
        site.compute_baseline()
        print(f"  stage 1: response scan ({sources} sources @ sigma_ref={args.sigma_ref})", flush=True)
        response_matrix, mean_diff_matrix_t = _stage1_response_scan(
            site, sources, args.sigma_ref, device, AMP_DTYPE)
        mean_diff_matrix = mean_diff_matrix_t.numpy()
        tau_global = compute_global_tau(response_matrix, args.q)
        print(f"  stage 1: global tau (q={args.q}) = {tau_global:.6f}", flush=True)

        print(f"  stage 2: sigma* search ({sources} sources x {args.n_grid} grid)", flush=True)
        sigma_matrix = _stage2_sigma_search(
            site, sources, grid, tau_global, args.sigma_min, args.sigma_max)

    # 强度映射 + 方向符号（07-04 同式）
    sigma_t = torch.from_numpy(sigma_matrix.copy())
    connectivity = -(sigma_t.float().log() + _LOG_2PIE).numpy() + 1.0
    sign_matrix = np.sign(mean_diff_matrix)
    signed_connectivity = sign_matrix * connectivity

    elapsed_min = (time.time() - subj_t0) / 60.0
    n_at_min = int(np.sum(np.isfinite(sigma_matrix) & (sigma_matrix <= args.sigma_min + 1e-6)))
    n_at_max = int(np.sum(np.isfinite(sigma_matrix) & (sigma_matrix >= args.sigma_max - 1e-6)))
    n_total = sigma_matrix.size
    payload = {
        "subject_idx": int(args.subject_idx),
        "subject_file": f"s{sid}",
        "n_trials": int(n_used),
        "n_sources_requested": int(sources),
        "n_sources_succeeded": int(sources),
        "n_source_failures": 0,
        "method": "global_tau_quantile_signed (gpfm-v5-main surrogate, signal-space, sigma* preserved, directional sign)",
        "tau_global": float(tau_global),
        "q": float(args.q),
        "sigma_ref": float(args.sigma_ref),
        "sigma_matrix": sigma_matrix.tolist(),
        "connectivity_matrix": connectivity.tolist(),
        "connectivity_formula": "-(log sigma* + log(2*pi*e)) + 1",
        "raw_response_matrix": response_matrix.numpy().tolist(),
        "sign_matrix": sign_matrix.tolist(),
        "signed_connectivity_matrix": signed_connectivity.tolist(),
        "mean_diff_matrix": mean_diff_matrix.tolist(),
        "mu_bias_per_channel": (site._mu_bias.detach().cpu().tolist()
                                if site._mu_bias is not None else None),
        "noise_model": "biased_gaussian_mu05std (additive on x^p after z-score, before model)",
        "grid": "log_spaced",
        "hparams": {
            "q": float(args.q), "sigma_ref": float(args.sigma_ref),
            "sigma_min": float(args.sigma_min), "sigma_max": float(args.sigma_max),
            "n_grid": int(args.n_grid), "euler_steps": int(cfg.euler_steps),
            "batch_size": int(N), "seed": int(args.seed),
        },
        "checkpoint": os.path.basename(args.checkpoint),
        "surrogate": {
            "run_dir": args.run_dir, "variant": cfg.variant_name,
            "euler_steps": int(cfg.euler_steps), "sigma_min_sampler": float(cfg.sigma_min),
            "ema_weights": True, "amp": "bf16-autocast",
            "eps_eta_seed": int(args.seed * 10000 + sid),
            "past_len": int(cfg.past_len), "future_len": int(cfg.future_len),
            "stride": int(cfg.stride), "data_root": cfg.data_root,
        },
        "diagnostics": {
            "n_at_sigma_min": n_at_min, "n_at_sigma_max": n_at_max,
            "pct_at_sigma_min": n_at_min / n_total * 100 if n_total else 0.0,
            "pct_at_sigma_max": n_at_max / n_total * 100 if n_total else 0.0,
        },
        "elapsed_min": float(elapsed_min),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    payload = sanitize_for_json(payload)
    out_json = os.path.join(out_dir, f"subject_{args.subject_idx}.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=json_default, allow_nan=False)
    print(f"  subject {args.subject_idx} (s{sid}) done in {elapsed_min:.2f} min | "
          f"tau={tau_global:.6f} @min={n_at_min/n_total*100:.1f}% @max={n_at_max/n_total*100:.1f}% | "
          f"wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
