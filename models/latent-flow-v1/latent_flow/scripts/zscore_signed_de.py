"""对 de-signed 方法（有偏置噪声）得到的 de 矩阵做 z-score.

Task: 对 ``signed_connectivity_matrix``（de-signed 方法主输出，``noise_model=
biased_gaussian_mu05std``）做 off-diagonal z-score，产出独立的分析输出文件，
**不**回写原 JSON。

Why off-diagonal only: 对角线是自连接（source==target 的 sigma* 被 clamp 到
sigma_max，connectivity 全部等于 ``-(log sigma_max + 0.5*log(2*pi*e))``，例如
3.186），不是真实边，参与均值/方差会污染统计。

Convention reused: 与 ``_compare_signed_de_ablation.py::_zscore_offdiag`` 一致 —
取全部非对角元素，NaN-aware 计算 mean/std，整个矩阵（除对角线）统一标准化，
对角线置 NaN。本文件直接 import 该 helper 作为单一事实来源，避免约定漂移。

Inputs:
  - flat layout:   results/oc_connectivity_global_tau_signed*<tag>/subject_{i}.json
  - ablation tree: results/oc_connectivity_global_tau_signed_ablation/fold{k}/{variant}/subject_{i}.json

Output (mirror layout under a new root):
  results/oc_connectivity_global_tau_signed_zscore*/subject_{i}.json
  results/oc_connectivity_global_tau_signed_ablation_zscore/fold{k}/{variant}/subject_{i}.json

每个输出 JSON 只携带 z-score 后的矩阵和溯源元信息（轻量），并 sanity-gate：若
源 JSON 的 ``noise_model`` 不是 ``biased_gaussian_mu05std`` 则跳过并告警（防止
误处理 LatentProbeSite 等无偏置噪声路径的输出）。

Usage:
    # 默认：处理两套结果（flat signal_q0.5 + ablation 树）
    python -u -m latent_flow.scripts.zscore_signed_de

    # 只处理 flat 目录（glob 模式）
    python -u -m latent_flow.scripts.zscore_signed_de --only flat --flat-glob "oc_connectivity_global_tau_signed_signal_q0.5"

    # 只处理 ablation 树
    python -u -m latent_flow.scripts.zscore_signed_de --only ablation
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import numpy as np

# Reuse the project's canonical off-diag z-score helper as the single source of
# truth for the convention. Importing (rather than re-implementing) prevents
# normalization drift between analysis scripts.
from latent_flow.scripts._compare_signed_de_ablation import _zscore_offdiag

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

EXPECTED_NOISE_MODEL = "biased_gaussian_mu05std"


def _zscore_offdiag_zerodiag(src_mat):
    """Off-diag z-score (shared convention) with the diagonal set to NaN.

    The shared ``_zscore_offdiag`` writes z-scored values into the off-diagonal
    mask but leaves the diagonal holding the *copied* input values. For a
    signed-DE matrix the diagonal is the self-connection sigma_max clamp
    artifact (e.g. 3.186), not a real edge — keeping it would be misleading.
    We null it out so downstream consumers never mistake it for a normalized
    edge.
    """
    mat = np.asarray(src_mat, dtype=np.float64)
    mat[~np.isfinite(mat)] = np.nan
    out = _zscore_offdiag(mat)
    np.fill_diagonal(out, np.nan)
    return out


def _process_subject(in_path, out_path):
    """Read one signed-DE JSON, z-score signed_connectivity_matrix, write output.

    Returns (status, detail): status in {"ok", "skip", "error"}.
    """
    with open(in_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)

    noise_model = payload.get("noise_model")
    if noise_model != EXPECTED_NOISE_MODEL:
        return ("skip", f"noise_model={noise_model!r} != {EXPECTED_NOISE_MODEL!r}")

    src_mat = np.asarray(payload["signed_connectivity_matrix"], dtype=np.float64)
    z_mat = _zscore_offdiag_zerodiag(src_mat)
    # _zscore_offdiag returns all-NaN if <3 finite off-diag entries or std<1e-12.
    off = ~np.eye(src_mat.shape[0], dtype=bool)
    n_finite_off = int(np.isfinite(src_mat[off]).sum())
    n_finite_z = int(np.isfinite(z_mat).sum())
    if n_finite_z == 0:
        return ("skip", "z-score produced no finite values (degenerate matrix)")

    out_payload = {
        "subject_idx": payload.get("subject_idx"),
        "subject_file": payload.get("subject_file"),
        "source_json": os.path.relpath(in_path, ROOT).replace("\\", "/"),
        "method": payload.get("method"),
        "noise_model": noise_model,
        "normalization": {
            "target": "signed_connectivity_matrix",
            "mode": "zscore_offdiag",
            "formula": "(x - mean(offdiag)) / std(offdiag), NaN-aware",
            "diagonal": "NaN (self-connection excluded)",
            "n_finite_offdiag_input": n_finite_off,
            "n_finite_offdiag_output": n_finite_z,
        },
        "signed_connectivity_zscore": np.where(
            np.isfinite(z_mat), z_mat, None
        ).tolist(),
        "stats_input": {
            "offdiag_min": float(np.nanmin(src_mat[off])),
            "offdiag_max": float(np.nanmax(src_mat[off])),
            "offdiag_mean": float(np.nanmean(src_mat[off])),
            "offdiag_std": float(np.nanstd(src_mat[off])),
        },
        "stats_output": {
            "offdiag_min": float(np.nanmin(z_mat)),
            "offdiag_max": float(np.nanmax(z_mat)),
            "offdiag_mean": float(np.nanmean(z_mat)),
            "offdiag_std": float(np.nanstd(z_mat)),
        },
        "checkpoint": payload.get("checkpoint"),
        "timestamp": payload.get("timestamp"),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out_payload, fh, indent=2, allow_nan=False)
    return ("ok", str(out_path.relative_to(ROOT)))


def _iter_flat(flat_glob):
    """Yield (in_path, out_path) for a flat layout dir.

    in:  results/<flat_glob>/subject_{i}.json
    out: results/<flat_glob>_zscore/subject_{i}.json
    """
    src_dir = RESULTS_DIR / flat_glob
    if not src_dir.is_dir():
        return
    out_dir = RESULTS_DIR / f"{flat_glob}_zscore"
    for p in sorted(src_dir.glob("subject_*.json")):
        yield p, out_dir / p.name


def _iter_ablation():
    """Yield (in_path, out_path) for the ablation tree.

    in:  results/oc_connectivity_global_tau_signed_ablation/fold{k}/{variant}/subject_{i}.json
    out: results/oc_connectivity_global_tau_signed_ablation_zscore/fold{k}/{variant}/subject_{i}.json
    """
    src_root = RESULTS_DIR / "oc_connectivity_global_tau_signed_ablation"
    out_root = RESULTS_DIR / "oc_connectivity_global_tau_signed_ablation_zscore"
    if not src_root.is_dir():
        return
    for fold_dir in sorted(src_root.glob("fold*")):
        for variant_dir in sorted(p for p in fold_dir.iterdir() if p.is_dir()):
            for p in sorted(variant_dir.glob("subject_*.json")):
                rel = p.relative_to(src_root)
                yield p, out_root / rel


def _run_layout(name, pairs):
    n_ok = n_skip = n_err = 0
    for in_path, out_path in pairs:
        try:
            status, detail = _process_subject(in_path, out_path)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"  [err] {in_path.name}: {type(exc).__name__}: {exc}")
            n_err += 1
            continue
        tag = {"ok": "+", "skip": "-", "error": "!"}[status]
        print(f"  [{tag}] {in_path.relative_to(ROOT)} -> {detail}")
        if status == "ok":
            n_ok += 1
        elif status == "skip":
            n_skip += 1
        else:
            n_err += 1
    print(f"  {name}: ok={n_ok} skip={n_skip} err={n_err}")
    return n_ok, n_skip, n_err


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--only", choices=["flat", "ablation"], default="",
        help="process only one layout (default: both)",
    )
    ap.add_argument(
        "--flat-glob", default="oc_connectivity_global_tau_signed_signal_q0.5",
        help="flat layout dir name under results/ (default signal_q0.5)",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    total_ok = total_skip = total_err = 0

    if args.only in ("", "flat"):
        print(f"\n=== flat layout: {args.flat_glob} ===")
        ok, skip, err = _run_layout("flat", _iter_flat(args.flat_glob))
        total_ok += ok; total_skip += skip; total_err += err

    if args.only in ("", "ablation"):
        print(f"\n=== ablation tree ===")
        ok, skip, err = _run_layout(
            "ablation", _iter_ablation())
        total_ok += ok; total_skip += skip; total_err += err

    print(f"\n=== total: ok={total_ok} skip={total_skip} err={total_err} ===")
    if total_err:
        sys.exit(1)


if __name__ == "__main__":
    main()
