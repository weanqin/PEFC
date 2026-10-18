"""Compare baseline connectivity (GC, TE) vs OC-LatentFlow attribution.

Reads per-subject connectivity matrices from:
    results/baseline_connectivity/{gc,te}/subject_{idx}.json
    results/oc_connectivity/subject_{idx}.json
and reports structural contrasts (directedness, sparsity, rank, Frobenius
norm) plus the matrix-matrix correlation between each pair of methods.
Useful to sanity-check that the model-independent baselines and the
model-attribution matrices disagree in informative ways.

Usage (run from repo root D:/102-npi/ude-vae-npi):

    python -u -m latent_flow.baseline.compare_baseline_vs_oc \\
        --subjects 10
"""

import argparse
import json
import os

import numpy as np


def _load_matrix(path):
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    cm = np.asarray(payload["connectivity_matrix"], dtype=np.float64)
    # Replace NaN/Inf sentinels (written as "NaN" strings) with nan.
    cm[~np.isfinite(cm)] = np.nan
    return cm


def _offdiag(a):
    n = a.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return a[mask]


def _describe(cm):
    vals = _offdiag(cm)
    vals = vals[np.isfinite(vals)]
    n = cm.shape[0]
    directedness = (np.abs(cm - cm.T).mean()
                    / max(np.abs(cm).mean(), 1e-12))
    # Effective rank via singular-value entropy (normalised).
    sv = np.linalg.svd(cm, compute_uv=False)
    sv = sv[sv > 1e-12]
    p = sv / sv.sum()
    eff_rank = float(np.exp(-(p * np.log(p)).sum())) if sv.size else 0.0
    return {
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "max": float(vals.max()),
        "directedness(|M-M^T|/|M|)": float(directedness),
        "eff_rank": eff_rank,
        "frobenius": float(np.sqrt((vals ** 2).sum())),
    }


def _upper_triangle_corr(a, b):
    """Pearson r between flattened upper-triangles of two matrices."""
    mask = ~np.eye(a.shape[0], dtype=bool)
    fa = a[mask]; fb = b[mask]
    keep = np.isfinite(fa) & np.isfinite(fb)
    fa = fa[keep]; fb = fb[keep]
    if fa.size < 2:
        return float("nan")
    fa = fa - fa.mean(); fb = fb - fb.mean()
    denom = (np.sqrt((fa ** 2).sum()) * np.sqrt((fb ** 2).sum()))
    return float((fa * fb).sum() / denom) if denom > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, default=10)
    ap.add_argument("--baseline-dir", type=str,
                    default="results/baseline_connectivity")
    ap.add_argument("--oc-dir", type=str, default="results/oc_connectivity")
    args = ap.parse_args()

    methods = ["gc", "te", "gpd", "oc"]
    paths = {
        "gc": lambda i: os.path.join(args.baseline_dir, "gc", f"subject_{i}.json"),
        "te": lambda i: os.path.join(args.baseline_dir, "te", f"subject_{i}.json"),
        "gpd": lambda i: os.path.join(args.baseline_dir, "gpd", f"subject_{i}.json"),
        "oc": lambda i: os.path.join(args.oc_dir, f"subject_{i}.json"),
    }

    print(f"{'subj':>4} | " + " | ".join(f"{m:>40}" for m in methods))
    print("-" * (8 + 43 * len(methods)))
    for i in range(args.subjects):
        row = []
        mats = {}
        for m in methods:
            p = paths[m](i)
            if not os.path.exists(p):
                row.append(f"{m}: (missing)")
                continue
            cm = _load_matrix(p)
            mats[m] = cm
            d = _describe(cm)
            row.append(
                f"{m}:mean={d['mean']:.4g} dir={d['directedness(|M-M^T|/|M|)']:.3f} "
                f"rk={d['eff_rank']:.1f} fro={d['frobenius']:.3g}"
            )
        print(f"{i:>4} | " + " | ".join(f"{x:>40}" for x in row))

    # Cross-method correlation (upper triangle), averaged over subjects.
    print("\ncross-method upper-triangle Pearson r (mean over subjects):")
    pairs = [("gc", "te"), ("gc", "gpd"), ("te", "gpd"),
             ("gc", "oc"), ("te", "oc"), ("gpd", "oc")]
    for a, b in pairs:
        rs = []
        for i in range(args.subjects):
            pa, pb = paths[a](i), paths[b](i)
            if os.path.exists(pa) and os.path.exists(pb):
                rs.append(_upper_triangle_corr(_load_matrix(pa),
                                               _load_matrix(pb)))
        rs = [r for r in rs if np.isfinite(r)]
        mean_r = float(np.mean(rs)) if rs else float("nan")
        print(f"  {a} vs {b}: {mean_r:+.4f}  (n={len(rs)})")


if __name__ == "__main__":
    main()
