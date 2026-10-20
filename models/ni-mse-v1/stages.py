"""stages.py — null / perturb 两期核心 + R1/R2/R3/R4 机器规则（design.md §2 + §1b-3/4/6/7）。

null 期（每例）：
- Ŷ_ref：B_ref=64 **固定 ε 网格**（spawn_key=(10, sid)，逐被试一份、全窗复用——Ŷ_ref 为
  x_past 的确定性泛函）逐点均值；与 K_null=64 **独立 ε**（(11, sid, w) 逐窗物化）**并入
  同一 sample() 调用**（B·K = 64+64 = 128 ≤ 8192；单 chunk）；
- N[k,b,w] = 前 n₀ 样本 MSE(ŷ_k vs Ŷ_ref)；τ_b = 池化 (W_null×K_null) 逐通道 Q90
  （np.quantile 线性插值）+ Q95 旗标带 + split-half（池化样本 w-major 展平后奇偶二分，
  触发统计量 median_b(|Δτ_b|/τ_b)，max_b 同步登记）；
- **Ŷ_ref 的 W_pert 50 窗随 null/{sid}.npz 落盘**（perturb 期只读复用——消除跨期重算
  批形微漂移，§1b-5）；
- R1：任何 NaN/Inf 硬失败中止；R4：退化窗（参考轨迹零方差 / 生成幅度 >10×训练 std——
  z 空间训练折 std ≡ 1（dataloader 口径），阈值 = 10×1.0；Run 000 钉定）整窗排除，
  N 对应窗列置 NaN 不进池，日志登记。

perturb 期（每例，per-(a,c) 组织）：
- 行组织 = **(w,k) 展平进 B、K=1**（W_pert×K = 50×32 = 1,600 ≤ 8192）：x_past 每 (w,k)
  行独立注入 ξ_{sid,w,a,k}（past 条件化不变量——扰动后 past 是 GPR 条件先验与 ODE 积分的
  唯一条件源）；ξ/η/ε spawn_key 均含 k 且**不含 c** ⇒ 跨 c 臂 CRN 配对（剂量-响应无
  蒙特卡洛抖动）；eps/eta/x 在 chunk 分块前按 (B,K) 全量物化（§1b-3）；
- R[a,b,w,k] = 前 n₀ 样本 MSE(ŷ vs Ŷ_ref 落盘值) → R̄ = median_k median_w（稳健链只作用
  于响应层，不进 τ_b）；
- **R2 同批断言（§1b-4）**：每例首窗、a=0、每 c 臂——P(X,a,0) 扰动路径与干净基线并入
  同一 sample() 调用（行 0/1，同 chunk_bk=1024 分块内、同 eps/eta）逐位 torch.equal；
  失败 ⇒ 写 `.r2_failed` 标记文件 + 非零退出（后续 stage 拒跑），不吞异常；
- R1 逐期生效；R4 幅度判据逐 (w,a,c) 置 NaN 排除并留日志。

续跑粒度（Run 002 前补丁，2026-10-20）：null 按例跳过（既有）；**perturb 亦按例跳过**——
该例该臂 `perturb/{sid}_c{c}.npz` 已存在且 R_bar 有限、k_resp/窗子集一致、且产物内
r2_checked == |c 臂|（R2 每例硬门随产物落标记——R2 在例内 perturb 循环 a=0 处先于全部
产物落盘执行，产物存在即证明该例 R2 已通过；已含标记的例整例跳过，取舍登记于
experiment.md Run 002）。任何校验不满足 ⇒ 整例重算（CRN 确定性下覆写逐位一致）。
"""

from __future__ import annotations

import _blas_guard  # noqa: F401  必须最先

import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

import noise as NZ

SAMPLE_ROW_LIMIT = 8192          # §1b-3：每次 sample() 调用行上限 B·K


@dataclass
class Params:
    """扰动侧执行参数（本目录独立常量，不进 gpfm Config；v1.5/design §1 钉定值）。"""

    w_null: int = 200
    w_pert: int = 50
    k_resp: int = 32
    k_null: int = 64
    b_ref: int = 64
    c_arms: tuple = (0.05, 0.1, 0.3, 0.5)     # 0.5 主读出 / 0.05 阴性对照 / {0.1,0.3} 敏感性
    r3_splithalf_threshold: float = 0.10
    r4_amp_limit_z: float = 10.0              # z 空间 × 训练 std（≡1，Run 000 钉定）

    def as_dict(self) -> dict:
        return {"w_null": self.w_null, "w_pert": self.w_pert, "k_resp": self.k_resp,
                "k_null": self.k_null, "b_ref": self.b_ref, "c_arms": list(self.c_arms),
                "r3_splithalf_threshold": self.r3_splithalf_threshold,
                "r4_amp_limit_z": self.r4_amp_limit_z}


def n0_from_cfg(cfg) -> int:
    """future 前 0.1s 样本数 n₀ = ⌈0.1·fs⌉（fmt@250 = 25；参数取自装载 cfg，禁字面量）。"""
    return int(math.ceil(0.1 * float(getattr(cfg, "fs", 500.0))))


def _to_t(arr: np.ndarray, dev: str) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)).to(dev)


def _finite_t(t: torch.Tensor, ctx: str) -> None:
    """R1：NaN/Inf 硬失败（确定性调试语境，不静默丢弃）。"""
    if not torch.isfinite(t).all():
        raise RuntimeError(f"R1 NaN/Inf 硬失败：{ctx}（"
                           f"nan={int(torch.isnan(t).sum())} inf={int(torch.isinf(t).sum())}）")


def _check_rows(n_rows: int, ctx: str) -> None:
    if n_rows > SAMPLE_ROW_LIMIT:
        raise RuntimeError(f"{ctx}：B·K = {n_rows} > 上限 {SAMPLE_ROW_LIMIT}（§1b-3）")


def tau_stats(N: np.ndarray) -> dict:
    """τ_b 池化口径统计（design §1b/§3：池化 (W×K) 逐通道分位数，非中位链）。

    N 形状 (K, C, W)；NaN 窗列 = R4 排除（不进池）。split-half = 池化样本 w-major 展平
    （index = w·K + k）后奇偶二分各算 Q90，逐通道 |Δτ_b|/τ_b；触发统计量 = median_b
    （>10% ⇒ R3），max_b 同步登记（Run 000 钉定）。
    """
    C = N.shape[1]
    valid = np.isfinite(N).all(axis=(0, 1))
    if not valid.any():
        raise RuntimeError("tau_stats：全部窗被 R4 排除，τ_b 无估计样本")
    flat = N.transpose(1, 2, 0)[:, valid, :].reshape(C, -1)     # (C, W_valid*K) w-major
    tau_b = np.quantile(flat, 0.90, axis=1)
    tau_q95 = np.quantile(flat, 0.95, axis=1)
    h1 = np.quantile(flat[:, ::2], 0.90, axis=1)
    h2 = np.quantile(flat[:, 1::2], 0.90, axis=1)
    rel = np.abs(h1 - h2) / tau_b
    return {"tau_b": tau_b, "tau_q95": tau_q95, "sh_rel": rel,
            "sh_median": float(np.median(rel)), "sh_max": float(np.max(rel)),
            "n_pooled_per_channel": int(flat.shape[1])}


def null_stage(th, test_ds, plan: dict, p: Params, null_dir: Path, subjects) -> dict:
    """null 期：每例 τ_b + split-half + Q95 + Ŷ_ref(W_pert 窗) 落盘 null/{sid}.npz。"""
    null_dir.mkdir(parents=True, exist_ok=True)
    dev, cfg = th.device, th.cfg
    C, F = cfg.n_channels, cfg.future_len
    n0 = n0_from_cfg(cfg)
    summaries: dict = {}
    for sid in subjects:
        out_path = null_dir / f"{sid}.npz"
        if out_path.exists():                                   # 续跑：已完成例跳过
            z = np.load(out_path)
            summaries[sid] = {"skipped": True, "sh_median": float(z["sh_median"]),
                              "sh_max": float(z["sh_max"]), "r3_triggered": bool(z["r3_triggered"])}
            print(f"[null] s{sid} 已存在，跳过", flush=True)
            continue
        t0 = time.time()
        wnull = plan[sid]["wnull"]
        wpert = plan[sid]["wpert"]
        wpert_pos = {int(g): i for i, g in enumerate(wpert)}
        eps_ref_np = NZ.rng_for(NZ.BRANCH["eps_ref"], sid).standard_normal((p.b_ref, C, F))
        eta_ref_np = NZ.rng_for(NZ.BRANCH["eta"], sid).standard_normal((p.b_ref, C, F))
        N = np.full((p.k_null, C, len(wnull)), np.nan, dtype=np.float64)
        yref_wpert = np.full((len(wpert), C, F), np.nan, dtype=np.float32)
        r4_log: list = []
        n_rows = p.b_ref + p.k_null
        _check_rows(n_rows, "null 期单窗调用")
        with th.ema.swap(th.model):
            for wi, g in enumerate(wnull):
                g = int(g)
                xp, _ = test_ds[g]
                x = xp.to(dev).unsqueeze(0)                     # (1,C,P)
                eps_null_np = NZ.rng_for(NZ.BRANCH["eps_null"], sid, g).standard_normal((p.k_null, C, F))
                eta_null_np = NZ.rng_for(NZ.BRANCH["eta"], sid, g).standard_normal((p.k_null, C, F))
                eps = _to_t(np.concatenate([eps_ref_np, eps_null_np]), dev).unsqueeze(0)
                eta = _to_t(np.concatenate([eta_ref_np, eta_null_np]), dev).unsqueeze(0)
                out = th.sampler.sample(x, n_rows, eps=eps, eta=eta)[0]   # (b_ref+k_null, C, F)
                _finite_t(out, f"null s{sid} w{g} 采样输出")
                ref = out[: p.b_ref].mean(dim=0)                # Ŷ_ref (C,F)：固定网格逐点均值
                reason = None
                if bool((ref.var(dim=1) == 0).any()):
                    reason = "ref_zero_var"
                elif float(out.abs().max()) > p.r4_amp_limit_z:
                    reason = f"amp_gt_{p.r4_amp_limit_z:g}x_train_std_z"
                if reason is not None:                          # R4：整窗排除留日志（NaN 不进池）
                    r4_log.append({"window": g, "reason": reason})
                    continue
                diff = (out[p.b_ref:, :, :n0] - ref[None, :, :n0]) ** 2   # (k,C,n0)
                N[:, :, wi] = diff.mean(dim=2).cpu().numpy()
                if g in wpert_pos:
                    yref_wpert[wpert_pos[g]] = ref.cpu().numpy()
        ts = tau_stats(N)
        r3 = bool(ts["sh_median"] > p.r3_splithalf_threshold)
        np.savez(
            out_path,
            subject_id=sid, n0=n0, b_ref=p.b_ref, k_null=p.k_null,
            wnull_global=wnull.astype(np.int64), wpert_global=wpert.astype(np.int64),
            N=N.astype(np.float32),
            tau_b=ts["tau_b"], tau_q95=ts["tau_q95"],
            sh_rel=ts["sh_rel"], sh_median=ts["sh_median"], sh_max=ts["sh_max"],
            n_pooled_per_channel=ts["n_pooled_per_channel"], r3_triggered=r3,
            y_ref_wpert=yref_wpert, r4_log_json=json_dumps_r4(r4_log),
        )
        el = time.time() - t0
        summaries[sid] = {"skipped": False, "n_windows": int(len(wnull)),
                          "n_r4_excluded": len(r4_log), "r4": r4_log,
                          "sh_median": ts["sh_median"], "sh_max": ts["sh_max"],
                          "n_pooled_per_channel": ts["n_pooled_per_channel"],
                          "r3_triggered": r3, "elapsed_s": round(el, 1)}
        print(f"[null] s{sid} 完成 | 窗 {len(wnull)} | R4 排除 {len(r4_log)} | "
              f"池/通道 {ts['n_pooled_per_channel']} | split-half median "
              f"{ts['sh_median']:.3%} max {ts['sh_max']:.3%} | R3 {'触发' if r3 else '未触发'} | "
              f"{el:.0f}s", flush=True)
    return summaries


def json_dumps_r4(entries: list) -> str:
    import json
    return json.dumps(entries, ensure_ascii=False)


def r2_marker_path(out_root: Path) -> Path:
    return out_root / ".r2_failed"


def perturb_subject_cached(pert_dir: Path, sid, p: Params, K: int, wpert: np.ndarray):
    """perturb per-例续跑判定（Run 002 前补丁，2026-10-20）。

    该例全部 c 臂 `perturb/{sid}_c{c}.npz` 存在且：R_bar 有限 / k_resp 一致 / 窗子集与
    null 落盘一致 / r2_checked == |c 臂| ⇒ 返回跳过摘要 dict；任何不满足 ⇒ None
    （整例重算，CRN 确定性下覆写逐位一致）。R2 与跳过的取舍：R2 在例内 a=0 处先于
    产物落盘执行并随产物落 r2_checked 标记——产物含完整标记即该例硬门已过，整例可跳过。
    """
    r4_total = 0
    for c in p.c_arms:
        path = pert_dir / f"{sid}_c{c:g}.npz"
        if not path.exists():
            return None
        try:
            z = np.load(path)
            if not np.isfinite(z["R_bar"]).all():
                return None
            if int(z["k_resp"]) != int(K):
                return None
            if not np.array_equal(z["wpert_global"].astype(int), wpert):
                return None
            if int(z["r2_checked"]) != len(p.c_arms):
                return None
            r4_total += int(z["r4_count"])
        except Exception:
            return None
    return {"skipped": True, "elapsed_s": 0.0, "r2_checked": len(p.c_arms),
            "r4_count": r4_total}


def write_r2_marker(out_root: Path, message: str) -> None:
    import datetime
    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    r2_marker_path(out_root).write_text(f"{stamp} {message}\n", encoding="utf-8")


def perturb_stage(th, test_ds, plan: dict, p: Params, out_root: Path, subjects) -> dict:
    """perturb 期：per-(a,c) 窗循环 → R → R̄；R2 同批断言内嵌（失败 = 标记文件 + 非零退出）。

    续跑（Run 002 前补丁）：per-例跳过——全部 c 臂产物存在且有限、k_resp/窗子集一致、
    r2_checked == |c 臂|（R2 随产物落标记，产物在即证明该例 R2 已通过）⇒ 整例跳过；
    校验不满足 ⇒ 整例重算（确定性覆写）。
    """
    null_dir = out_root / "null"
    pert_dir = out_root / "perturb"
    pert_dir.mkdir(parents=True, exist_ok=True)
    dev, cfg = th.device, th.cfg
    C, F, P = cfg.n_channels, cfg.future_len, cfg.past_len
    n0 = n0_from_cfg(cfg)
    summaries: dict = {}
    for sid in subjects:
        null_path = null_dir / f"{sid}.npz"
        assert null_path.exists(), f"perturb 前置断言：{null_path} 不存在（先跑 --stage null）"
        nz = np.load(null_path)
        yref = nz["y_ref_wpert"]                                # (W_pert, C, F)
        wpert = nz["wpert_global"].astype(int)
        # 窗子集一致性：perturb 必须消费 null 落盘的同一 W_pert（参数漂移即拒跑）
        assert np.array_equal(wpert, plan[sid]["wpert"]), \
            f"s{sid} perturb 窗子集与 null 落盘不一致（参数漂移？重跑须同 w_null/w_pert）"
        K = p.k_resp
        cached = perturb_subject_cached(pert_dir, sid, p, K, wpert)
        if cached is not None:                      # 续跑：per-例跳过（Run 002 前补丁）
            summaries[sid] = cached
            print(f"[perturb] s{sid} 已存在（全部 c 臂产物有限且 r2_checked 完整），跳过",
                  flush=True)
            continue
        valid_w = np.isfinite(yref).all(axis=(1, 2))            # null 期 R4 排除传播
        if not valid_w.any():
            raise RuntimeError(f"s{sid}：W_pert 全部窗在 null 期被 R4 排除，无法测响应")
        W = len(wpert)
        n_rows = W * K + 2                                      # +2 = R2 行 0/1
        _check_rows(n_rows, "perturb 期 (a,c) 调用")
        Yref_t = _to_t(yref, dev)                               # (W,C,F)
        X = torch.stack([test_ds[int(g)][0] for g in wpert]).to(dev)      # (W,C,P)
        valid_w_t = torch.from_numpy(valid_w).to(dev)
        r2_count, r4_total = 0, 0
        t0 = time.time()
        R_bar = {c: np.full((C, C), np.nan) for c in p.c_arms}   # R_bar[c][a,b]
        R_medk = {c: np.full((C, W, C), np.nan, dtype=np.float32) for c in p.c_arms}
        degenerate_log = {c: [] for c in p.c_arms}
        Yref_rows = Yref_t.unsqueeze(1).expand(W, K, C, F).reshape(W * K, C, F)[:, :, :n0]
        with th.ema.swap(th.model):
            for a in range(C):
                # per-(w,k) 噪声一次物化，c 无关 ⇒ 跨 c 臂共享（CRN 配对）
                xi = np.stack([NZ.draw_xi(sid, int(g), a, k, P)
                               for g in wpert for k in range(K)])           # (W*K, P)
                eps = np.stack([NZ.draw_eps_pert(sid, int(g), a, k, (C, F))
                                for g in wpert for k in range(K)])          # (W*K, C, F)
                eta = np.stack([NZ.draw_eta(sid, int(g), a, k, (C, F))
                                for g in wpert for k in range(K)])
                xi_t, eps_t, eta_t = _to_t(xi, dev), _to_t(eps, dev), _to_t(eta, dev)
                X_rows = X.unsqueeze(1).expand(W, K, C, P).reshape(W * K, C, P)
                for c in p.c_arms:
                    rows = NZ.apply_noise(X_rows, a, float(c), xi_t)        # P(X,a,c)·扰动路径
                    if a == 0:                                              # R2：首窗首源每 c 臂
                        base_x = X[0]                                       # 干净基线 (C,P)
                        r2_p = NZ.apply_noise(base_x, a, 0.0, xi_t[0])      # P(X,a,0) 扰动路径
                        batch = torch.cat([r2_p.unsqueeze(0), base_x.unsqueeze(0), rows], dim=0)
                        eps_b = torch.cat([eps_t[:1], eps_t[:1], eps_t], dim=0)
                        eta_b = torch.cat([eta_t[:1], eta_t[:1], eta_t], dim=0)
                    else:
                        batch, eps_b, eta_b = rows, eps_t, eta_t
                    out = th.sampler.sample(batch, 1, eps=eps_b.unsqueeze(1),
                                            eta=eta_b.unsqueeze(1))[:, 0]   # (B,C,F)
                    if a == 0:
                        if not torch.equal(out[0], out[1]):                 # R2 同批逐位比对
                            dmax = float((out[0] - out[1]).abs().max())
                            n_bad = int((out[0] != out[1]).sum())
                            msg = (f"R2 失败：s{sid} a={a} c={c} 首窗 c=0 扰动路径 vs 干净基线"
                                   f" 非逐位恒等（max|Δ|={dmax:.3e}，不等元素 {n_bad}/"
                                   f"{out[0].numel()}）——种子树/装载 bug，中止排查")
                            write_r2_marker(out_root, msg)
                            print(f"[perturb] {msg}", file=sys.stderr, flush=True)
                            sys.exit(1)
                        r2_count += 1
                    _finite_t(out, f"perturb s{sid} a{a} c{c} 采样输出")
                    y = out[2:] if a == 0 else out                          # (W*K,C,F)
                    amp_w = y.abs().amax(dim=(1, 2)).view(W, K).max(dim=1).values  # 逐窗幅度
                    deg_w = (~valid_w_t) | (amp_w > p.r4_amp_limit_z)
                    if bool(deg_w.any()):
                        bad = [int(wpert[i]) for i in torch.nonzero(deg_w).flatten().tolist()]
                        degenerate_log[c].append({"a": a, "windows": bad})
                        r4_total += len(bad)
                    R_wkc = ((y[:, :, :n0] - Yref_rows) ** 2).mean(dim=-1).view(W, K, C)
                    medk = np.median(R_wkc.cpu().numpy(), axis=1)           # (W,C) median_k
                    medk[~valid_w | deg_w.cpu().numpy()] = np.nan           # R4 窗排除
                    R_medk[c][a] = medk.astype(np.float32)
                    R_bar[c][a] = np.nanmedian(medk, axis=0)                # median_w → R̄[a,b]
        for c in p.c_arms:
            out_path = pert_dir / f"{sid}_c{c:g}.npz"
            if not np.isfinite(R_bar[c]).all():
                raise RuntimeError(f"R1：s{sid} c={c} R̄ 含 NaN/Inf（有效窗耗尽或采样异常）")
            np.savez(out_path, subject_id=sid, c=float(c), n0=n0, k_resp=K,
                     wpert_global=wpert.astype(np.int64),
                     R_bar=R_bar[c].astype(np.float32), R_medk=R_medk[c],
                     r2_checked=r2_count, r4_count=int(r4_total),
                     r4_log_json=json_dumps_r4(degenerate_log[c]))
        el = time.time() - t0
        summaries[sid] = {"elapsed_s": round(el, 1), "r2_checked": r2_count,
                          "r4_count": int(r4_total)}
        print(f"[perturb] s{sid} 完成 | R2 断言 {r2_count}/{len(p.c_arms)} 通过 | R4 排除 "
              f"{r4_total} (a,窗) 项 | {el:.0f}s", flush=True)
    return summaries
