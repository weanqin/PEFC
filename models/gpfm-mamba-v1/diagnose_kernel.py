"""训练折核诊断复跑（implement.md 2.3，design §2.1 附件）。

fold0 训练折正式 z-score 口径下：
- OU 时间核：R²(μ) / post_std / Σ_pp 条件数 / GT vs 先验样本粗度
- Ricker3D 空间核 σ_s ∈ {0.045, 0.080, 0.137}：min_eig / 邻位相关（解析+经验）/ 负瓣 / 有效秩
产物：markdown 写 .trellis/tasks/10-12-gpfm-mamba-edra-f0/research/kernel-re-diagnosis.md
运行：python3 models/gpfm-mamba-v1/diagnose_kernel.py（~1 min，须先在 experiment.md 登记 Run）
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as C  # noqa: E402
import data as D  # noqa: E402
from gpr_prior import GPRPrior, ricker3d_kernel  # noqa: E402

OUT_MD = "/root/npi/.trellis/tasks/10-12-gpfm-mamba-edra-f0/research/kernel-re-diagnosis.md"
N_WIN_PER_SUBJ = 40
N_PRIOR_SAMPLES = 400
DIAG_SEED = 1234


def main() -> None:
    torch.manual_seed(DIAG_SEED)
    np.random.seed(DIAG_SEED)
    cfg = C.Config()
    ds = D.build_datasets(cfg)
    train = ds["train"]
    prior = GPRPrior(
        np.load(f"{cfg.data_root}/chan_coords_62.npy"),
        past_len=cfg.past_len, future_len=cfg.future_len,
        kernel_t=cfg.kernel_t, gamma=cfg.gamma, period_p=cfg.period_p,
        iso=cfg.iso, pp_jitter=cfg.pp_jitter,
        sigma_s=cfg.sigma_s, jitter_s=cfg.jitter_s,
    )

    # ---- 确定性抽样诊断窗：每受试者等距 40 窗 ----
    n_subj = len(train.arrays)
    idx_per = [list(range(i, len(train), max(1, len(train) // N_WIN_PER_SUBJ)))[:N_WIN_PER_SUBJ] for i in range(n_subj)]
    idxs = [i for lst in idx_per for i in lst]
    xp = torch.stack([train[i][0] for i in idxs])  # (N,62,384)
    x1 = torch.stack([train[i][1] for i in idxs])  # (N,62,128)
    mu = prior.posterior_mean(xp)
    n = len(idxs)
    print(f"诊断窗 {n}（受试者 {n_subj} 例 × {N_WIN_PER_SUBJ}）")

    # ---- 时间侧（OU）----
    ss_res = float(((x1 - mu) ** 2).sum())
    ss_tot = float(((x1 - x1.mean(dim=(0, 2), keepdim=True)) ** 2).sum())
    r2_mu = 1.0 - ss_res / ss_tot
    post_std = np.sqrt(np.diag(prior.K_post_t))
    gt_std = float(x1.std(dim=-1, unbiased=True).mean())
    gt_rough = float((x1[..., 1:] - x1[..., :-1]).abs().mean())
    # 先验样本粗度在下方按 σ_s 分别采样统计
    Kpp = prior.K_t_full[: cfg.past_len, : cfg.past_len] + cfg.pp_jitter * np.eye(cfg.past_len)
    cond_pp = float(np.linalg.cond(Kpp))

    lines = [
        "# 核诊断复跑（fold0 训练折正式 z-score 口径）",
        "",
        "- 日期：2026-10-12；脚本 `models/gpfm-mamba-v1/diagnose_kernel.py`；Run 001（experiment.md）",
        f"- 数据：subject_filter=status_ok 过滤后 fold0 训练折 {n_subj} 例受试者（z-score 仅训练折），诊断窗 {n}（每例等距 {N_WIN_PER_SUBJ}）",
        f"- 时间核：OU γ={cfg.gamma} τ=i·π/p p={cfg.period_p} iso={cfg.iso}（折叠进 K'_t）pp_jitter={cfg.pp_jitter}",
        "",
        "## 时间侧（OU，与 σ_s 无关）",
        "",
        "| 指标 | 数值 | 适配研究对照（2 被试指示性） |",
        "|---|---|---|",
        f"| R²(μ) 后验均值 | {r2_mu:+.4f} | 0.01–0.03 |",
        f"| post_std（解析 diag(K'_{{t,f|p}})）| mean {post_std.mean():.3f} / min {post_std.min():.3f} / max {post_std.max():.3f} | 0.97–1.02 |",
        f"| GT future std（z-score 口径）| {gt_std:.3f} | ~1.0 |",
        f"| Σ_pp 条件数（iso+pp_jitter）| {cond_pp:.2e} | ~2.4e2 |",
        f"| GT future 粗度 \\|Δx\\| | {gt_rough:.4f} | 0.12–0.18 |",
    ]

    # ---- 空间侧 σ_s 扫描 ----
    coords = np.load(f"{cfg.data_root}/chan_coords_62.npy")
    d = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    off = d + np.eye(62) * 1e9
    nn_d = off.min(axis=1)  # 每电极最近邻距离
    nn_pairs = [(i, int(np.argmin(off[i]))) for i in range(62)]
    max_d = float(d[np.triu_indices(62, 1)].max())

    lines += [
        "",
        f"电极几何：最近邻距离 mean {nn_d.mean():.3f} m / min {nn_d.min():.3f} / max {nn_d.max():.3f}；阵列最大电极间距 {max_d:.3f} m",
        "",
        "## 空间侧（Ricker3D，σ_s 扫描；jitter_s=1e-3）",
        "",
        "| σ_s (m) | min_eig(未加 jitter) | +jitter 后条件数 | 邻位相关（解析 K_s@nn）| 邻位相关（先验样本经验）| min 非对角 | 负对占比 | 过零距离 √3σ vs 阵列 | 有效秩 exp(H) | 有效秩 Σλ/λmax | 先验样本粗度 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for sig in (0.045, 0.080, 0.137):
        Ks = ricker3d_kernel(coords, sig)
        w = np.linalg.eigvalsh(Ks)
        Ks_j = Ks + cfg.jitter_s * np.eye(62)
        p = w / w.sum()
        erank_h = float(np.exp(-(p @ np.log(p))))
        erank_ratio = float(w.sum() / w.max())
        ks_nn = float(np.mean([Ks[i, j] for i, j in nn_pairs]))
        iu = np.triu_indices(62, 1)
        min_off = float(Ks[iu].min())
        frac_neg = float((Ks[iu] < 0).mean())
        zc = np.sqrt(3.0) * sig
        # 先验样本经验邻位相关 + 粗度（40 窗 × 10 条 = 400 条先验样本，与 Run 001 登记一致）
        pr = GPRPrior(
            coords, past_len=cfg.past_len, future_len=cfg.future_len,
            kernel_t=cfg.kernel_t, gamma=cfg.gamma, period_p=cfg.period_p,
            iso=cfg.iso, pp_jitter=cfg.pp_jitter, sigma_s=sig, jitter_s=cfg.jitter_s,
        )
        draws = [pr.sample(xp[:N_WIN_PER_SUBJ]) for _ in range(N_PRIOR_SAMPLES // N_WIN_PER_SUBJ)]
        s = torch.cat(draws, dim=0)  # (400,62,128)
        emp_nn = []
        for i, j in nn_pairs:
            a = s[:, i, :].ravel().numpy()
            b = s[:, j, :].ravel().numpy()
            emp_nn.append(np.corrcoef(a, b)[0, 1])
        rough = float((s[..., 1:] - s[..., :-1]).abs().mean())
        cond_s = float(np.linalg.cond(Ks_j))
        lines.append(
            f"| {sig:.3f} | {w.min():+.2e} | {cond_s:.1e} | {ks_nn:+.3f} | {np.mean(emp_nn):+.3f} | "
            f"{min_off:+.3f} | {frac_neg:.1%} | {zc:.3f} m {'<' if zc < max_d else '>'} {max_d:.3f} m | "
            f"{erank_h:.1f} | {erank_ratio:.1f} | {rough:.4f} |"
        )
        print(f"σ_s={sig}: min_eig={w.min():+.2e} erank_h={erank_h:.1f} erank_ratio={erank_ratio:.1f} "
              f"ks@nn={ks_nn:+.3f} emp_nn={np.mean(emp_nn):+.3f} neg={frac_neg:.1%} rough={rough:.4f}")

    lines += [
        "",
        "## 结论口径",
        "",
        "- OU 时间侧与适配研究结论对照（R²(μ)≈0、post_std≈1 匹配 z-score 尺度、Σ_pp 条件数 ~1e2–1e3 数值稳定）是否维持；",
        "- Ricker3D 三档 σ_s 全部 PSD（min_eig>0）+ jitter 后 Cholesky 成功；",
        "- σ=0.080 默认档：邻位强正相关 + 阵列内负瓣（过零 √3σ<最大间距）+ 中等有效秩——语义为「平衡」档；",
        "  （0.045 强中心-周边、0.137 全局相干：过零出阵列、有效秩低）。",
        "- 具体判定见 experiment.md Run 001 结论。",
        "",
    ]
    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[written] {OUT_MD}")


if __name__ == "__main__":
    main()
