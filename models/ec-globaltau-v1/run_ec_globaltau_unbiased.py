"""EC_pert runner — 无偏噪声口径（μ=0，σ·ξ）× gpfm-mamba-v5-main（2026-10-21 用户拍板）.

与 07-04 正典 `run_ec_globaltau.py` 的唯一差异（登记偏离）：
  GpfmProbeSite(use_biased_noise=False) —— 关闭固定 μ_bias=0.5·std 方向偏置，扰动 = σ·ξ
  （ξ~N(0,1)，seed+source 播种，同一 ξ 跨 σ 复用），逐点加在 z-score 后、入模前的 x^p 源通道。
其余零差异：同 seeds / 同子采样 / EMA swap + bf16 / 两阶段流程（σ_ref 响应扫描 → 全局 q 分位 τ
→ log 网格 σ* 走查，σ* 经 import 的 walk_monotone_curve）/ conn = −(log σ* + ln 2πe) + 1。

本跑参数（用户拍板）：q=0.25 @ σ_ref=0.3，10 例 × 32 窗，log[0.01,0.48]×24。
注意：μ=0 下 mean-shift 无 07-23 方向性设计依据，sign 字段仅随档保存，不用于解读。

用法：
    OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 \
      /root/miniconda3/bin/python models/ec-globaltau-v1/run_ec_globaltau_unbiased.py \
        --subject-idx 0 --subjects 10 --n-trials 32 \
        --output-dir /root/npi/outputs/10-21-ec-globaltau-v5main/unbiased_mu0_ec
"""

import argparse
import json
import math
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

_LOG_2PIE = math.log(2.0 * math.pi * math.e)
DEFAULT_Q = 0.25          # 用户拍板：无偏口径用 q=0.25
SIGMA_MIN, SIGMA_MAX, DEFAULT_N_GRID = 0.01, 0.48, 24
DEFAULT_N_TRIALS, SEED = 32, 42


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default=DEFAULT_CKPT)
    p.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    p.add_argument("--subject-idx", type=int, default=0)
    p.add_argument("--subjects", type=int, default=10)
    p.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS)
    p.add_argument("--q", type=float, default=DEFAULT_Q)
    p.add_argument("--sigma-ref", type=float, default=0.3)
    p.add_argument("--sigma-min", type=float, default=SIGMA_MIN)
    p.add_argument("--sigma-max", type=float, default=SIGMA_MAX)
    p.add_argument("--n-grid", type=int, default=DEFAULT_N_GRID)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir",
                   default="/root/npi/outputs/10-21-ec-globaltau-v5main/unbiased_mu0_ec")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    grid = log_grid(args.sigma_min, args.sigma_max, args.n_grid)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Paradigm: global-quantile-tau UNBIASED (mu=0, sigma*xi) | q={args.q} "
          f"sigma_ref={args.sigma_ref} grid=log[{args.sigma_min},{args.sigma_max}]x{args.n_grid}",
          flush=True)

    cfg, model, ema, sampler, test_ds = load_surrogate(args.checkpoint, args.run_dir,
                                                       str(device))
    F = sampler.prior.future_len

    for i in range(args.subject_idx, args.subject_idx + args.subjects):
        sid = FOLD0_TEST_IDS[i]
        t0 = time.time()
        widx = subject_global_windows(test_ds, sid)
        x_p_full = torch.stack([test_ds[gi][0] for gi in widx]).to(device)
        x_p, n_used = subsample_trials(x_p_full, args.n_trials, args.seed, i)
        N, C, P = x_p.shape

        gen = torch.Generator(device=device).manual_seed(args.seed * 10000 + sid)
        eps = torch.randn(N, 1, C, F, device=device, generator=gen)
        eta = torch.randn(N, 1, C, F, device=device, generator=gen)

        site = GpfmProbeSite(sampler, x_p, eps, eta, seed=args.seed, use_biased_noise=False)
        assert site._mu_bias is None, "μ_bias 未关闭"

        with ema.swap(model):
            site.compute_baseline()
            # stage 1：σ_ref 响应扫描
            response = np.full((C, C), np.nan, dtype=np.float32)
            mean_diff = np.full((C, C), np.nan, dtype=np.float32)
            for src in range(C):
                site.set_source(src)
                mse_vec, md_vec = site.perturb_and_measure(float(args.sigma_ref), src)
                response[src, :] = mse_vec.detach().cpu().numpy()
                mean_diff[src, :] = md_vec.detach().cpu().numpy()
            tau_global = compute_global_tau(torch.from_numpy(response), args.q)
            # stage 2：σ* 走查（import 原语）
            sigma_matrix = np.full((C, C), np.nan, dtype=np.float32)
            tau_vec = torch.full((C,), float(tau_global))
            for src in range(C):
                site.set_source(src)
                mse_curve = torch.stack(
                    [_extract_mse(site.perturb_and_measure(float(s), src)) for s in grid],
                    dim=0)
                sigma_matrix[src, :] = walk_monotone_curve(
                    mse_curve, grid, tau_vec,
                    args.sigma_min, args.sigma_max).numpy()

        sigma_t = torch.from_numpy(sigma_matrix.copy())
        connectivity = -(sigma_t.float().log() + _LOG_2PIE).numpy() + 1.0
        sign_matrix = np.sign(mean_diff)

        elapsed = (time.time() - t0) / 60.0
        v = sigma_matrix.flatten()
        n_min = int(np.sum(v <= args.sigma_min + 1e-6))
        n_max = int(np.sum(v >= args.sigma_max - 1e-6))
        payload = {
            "subject_idx": int(i), "subject_file": f"s{sid}", "n_trials": int(n_used),
            "n_sources_requested": int(C), "n_sources_succeeded": int(C), "n_source_failures": 0,
            "method": ("global_tau_quantile UNBIASED (mu=0, sigma*xi, gpfm-v5-main surrogate, "
                       "signal-space, sigma* preserved; sign field saved but not directional "
                       "by design)"),
            "tau_global": float(tau_global), "q": float(args.q),
            "sigma_ref": float(args.sigma_ref),
            "sigma_matrix": sigma_matrix.tolist(),
            "connectivity_matrix": connectivity.tolist(),
            "connectivity_formula": "-(log sigma* + log(2*pi*e)) + 1",
            "raw_response_matrix": response.tolist(),
            "sign_matrix": sign_matrix.tolist(),
            "signed_connectivity_matrix": (sign_matrix * connectivity).tolist(),
            "mean_diff_matrix": mean_diff.tolist(),
            "noise_model": "unbiased_gaussian_mu0 (sigma*xi additive on x^p after z-score, before model)",
            "grid": "log_spaced",
            "hparams": {"q": float(args.q), "sigma_ref": float(args.sigma_ref),
                        "sigma_min": float(args.sigma_min), "sigma_max": float(args.sigma_max),
                        "n_grid": int(args.n_grid), "use_biased_noise": False,
                        "euler_steps": int(cfg.euler_steps), "batch_size": int(N),
                        "seed": int(args.seed)},
            "checkpoint": Path(args.checkpoint).name,
            "surrogate": {"run_dir": args.run_dir, "variant": cfg.variant_name,
                          "euler_steps": int(cfg.euler_steps), "ema_weights": True,
                          "amp": "bf16-autocast", "eps_eta_seed": int(args.seed * 10000 + sid),
                          "past_len": int(cfg.past_len), "future_len": int(cfg.future_len),
                          "stride": int(cfg.stride), "data_root": cfg.data_root},
            "diagnostics": {"n_at_sigma_min": n_min, "n_at_sigma_max": n_max,
                            "pct_at_sigma_min": n_min / v.size * 100,
                            "pct_at_sigma_max": n_max / v.size * 100,
                            "pct_interior": 100 - (n_min + n_max) / v.size * 100,
                            "sigma_median": float(np.median(v)),
                            "sigma_q25": float(np.quantile(v, 0.25)),
                            "sigma_q75": float(np.quantile(v, 0.75))},
            "elapsed_min": float(elapsed), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        payload = sanitize_for_json(payload)
        with open(out_dir / f"subject_{i}.json", "w") as fh:
            json.dump(payload, fh, indent=2, default=json_default, allow_nan=False)
        dg = payload["diagnostics"]
        print(f"  subject {i} (s{sid}) {elapsed:.2f} min | tau={tau_global:.3e} "
              f"interior={dg['pct_interior']:.1f}% @min={dg['pct_at_sigma_min']:.1f}% "
              f"@max={dg['pct_at_sigma_max']:.1f}% | wrote {out_dir}/subject_{i}.json",
              flush=True)


if __name__ == "__main__":
    main()
