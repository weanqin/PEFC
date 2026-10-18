"""Compare parameter-level A_eff (mexhat) vs GC/TE baselines.

A_eff (results/graph_connectivity/graph_connectivity_fold{N}.json) is the
learned mexhat mixer matrix — fold-level (one matrix shared across subjects),
SIGNED, ~symmetric. GC/TE (results/baseline_connectivity/{gc,te}/subject_*.json)
are directed, UNSIGNED, per-subject matrices computed on raw EEG.

Three correlation views over off-diagonal entries (z-scored before Pearson so
the comparison is about STRUCTURE not scale):
  - r_abs   : |A_eff| vs GC/TE        (topology, ignores A's sign)
  - r_exc   : A_eff.clip(0) vs GC/TE  (excitatory weights only)
  - r_signed: A_eff vs GC/TE          (signed — tests if excite/inhibit maps to +/-)

A_eff is subject-independent, so it is correlated against each subject's
GC/TE matrix separately and the mean is reported.

Usage (from repo root):

    D:/program/anaconda/envs/npi/python.exe -u -m \
        latent_flow.scripts.compare_params_vs_baseline --fold 0
"""

import argparse
import json
import os
import sys

import numpy as np


DEFAULT_A_DIR = "results/graph_connectivity"
DEFAULT_BL_DIR = "results/baseline_connectivity"
DEFAULT_OUT = "results/compare_params_vs_baseline"


def _load_matrix(path):
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    cm = np.asarray(payload["connectivity_matrix"], dtype=np.float64)
    cm[~np.isfinite(cm)] = np.nan
    return cm


def _offdiag_mask(n):
    return ~np.eye(n, dtype=bool)


def _zscore_offdiag(m):
    out = m.astype(np.float64).copy()
    mask = _offdiag_mask(m.shape[0])
    v = out[mask]
    v = v[np.isfinite(v)]
    if v.size < 2:
        return np.zeros_like(out)
    mu, sd = v.mean(), v.std()
    out[mask] = (out[mask] - mu) / (sd + 1e-12)
    return out


def _pearson(a, b):
    keep = np.isfinite(a) & np.isfinite(b)
    if keep.sum() < 2:
        return float("nan")
    a = a[keep] - a[keep].mean()
    b = b[keep] - b[keep].mean()
    denom = np.sqrt((a ** 2).sum()) * np.sqrt((b ** 2).sum())
    return float((a * b).sum() / denom) if denom > 0 else float("nan")


def _describe(m):
    n = m.shape[0]
    off = m[_offdiag_mask(n)]
    off = off[np.isfinite(off)]
    if off.size == 0:
        return {}
    return {
        "mean": float(off.mean()),
        "std": float(off.std()),
        "min": float(off.min()),
        "max": float(off.max()),
        "neg_fraction": float((off < 0).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--a-dir", type=str, default=DEFAULT_A_DIR)
    ap.add_argument("--bl-dir", type=str, default=DEFAULT_BL_DIR)
    ap.add_argument("--out-dir", type=str, default=DEFAULT_OUT)
    args = ap.parse_args()

    a_path = os.path.join(args.a_dir, f"graph_connectivity_fold{args.fold}.json")
    if not os.path.exists(a_path):
        raise FileNotFoundError(f"A_eff not found: {a_path}")
    a_raw = _load_matrix(a_path)
    n = a_raw.shape[0]

    # Precompute A views (subject-independent).
    a_abs = _zscore_offdiag(np.abs(a_raw))
    a_exc = _zscore_offdiag(np.clip(a_raw, 0, None))
    a_signed = _zscore_offdiag(a_raw)

    print(f"=== A_eff vs GC/TE  (fold {args.fold}, mexhat) ===")
    print(f"A_eff: shape {a_raw.shape}  signed={(a_raw < 0).any()}  "
          f"{_describe(a_raw)}\n")

    results = {}
    for method in ("gc", "te", "gpd"):
        mdir = os.path.join(args.bl_dir, method)
        if not os.path.isdir(mdir):
            print(f"  [{method}] missing dir {mdir}, skip")
            continue
        per_subj = []
        print(f"--- {method.upper()} ---")
        for i in range(10):
            spath = os.path.join(mdir, f"subject_{i}.json")
            if not os.path.exists(spath):
                continue
            bl = _load_matrix(spath)
            if bl.shape[0] != n:
                print(f"  subj {i}: shape {bl.shape} mismatch, skip")
                continue
            bl_z = _zscore_offdiag(bl)
            mask = _offdiag_mask(n)
            r_abs = _pearson(a_abs[mask], bl_z[mask])
            r_exc = _pearson(a_exc[mask], bl_z[mask])
            r_signed = _pearson(a_signed[mask], bl_z[mask])
            per_subj.append({"subject": i, "r_abs": r_abs, "r_exc": r_exc,
                             "r_signed": r_signed})
            print(f"  subj {i}: r_abs={r_abs:+.3f}  r_exc={r_exc:+.3f}  "
                  f"r_signed={r_signed:+.3f}")

        def _mean(key):
            rs = [e[key] for e in per_subj if np.isfinite(e[key])]
            return float(np.mean(rs)) if rs else float("nan")
        agg = {
            "n_subjects": len(per_subj),
            "mean_r_abs": _mean("r_abs"),
            "mean_r_exc": _mean("r_exc"),
            "mean_r_signed": _mean("r_signed"),
            "per_subject": per_subj,
        }
        results[method] = agg
        print(f"  MEAN (n={len(per_subj)}): r_abs={agg['mean_r_abs']:+.3f}  "
              f"r_exc={agg['mean_r_exc']:+.3f}  "
              f"r_signed={agg['mean_r_signed']:+.3f}\n")

    payload = {
        "fold": args.fold,
        "A_method": "effective_adjacency (mexhat + gamma*W)",
        "interpretation": (
            "r_abs>0: mexhat learned topology agrees with the model-free "
            "GC/TE estimate of channel coupling. r_signed>0: even the sign "
            "(excite/inhibit) maps onto the (unsigned) baseline strength. "
            "Compare with the 06-29 finding (normalized OC<->GC r=+0.497, "
            "OC<->TE r=+0.378) to see if the direct A_eff read matches the "
            "earlier bisection-based result."),
        "results": results,
    }

    os.makedirs(args.out_dir, exist_ok=True)
    out_json = os.path.join(args.out_dir, f"compare_fold{args.fold}.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, allow_nan=False)
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
