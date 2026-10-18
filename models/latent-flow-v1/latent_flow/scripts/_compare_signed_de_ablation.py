"""跨变体跨 fold 的 signed DE vs GC/TE/gPDC/A_eff 相关性汇总（task 07-25）。

泛化自 ``_compare_signed_de_baselines.py``（fold 0 / nosc 单点），覆盖 4 个模型
变体 × 5 fold 的完整消融矩阵，产出 per_fold + cross_fold + sign_consensus JSON。

Inputs (per fold k, per variant v):
  - signed DE:  results/oc_connectivity_global_tau_signed_ablation/fold{k}/{v}/subject_{i}.json
  - A_eff:      results/graph_connectivity_ablation/fold{k}/{v}/graph_connectivity_fold{k}.json
  - baselines:  <root>/results/baseline_connectivity_fold{k}/{gc,te,gpd}/subject_{i}.json

Output:
  results/compare_signed_de_ablation/summary.json  (per_fold + cross_fold + sign_consensus)

Usage:
    python -u -m latent_flow.scripts._compare_signed_de_ablation \
        [--folds 0,1,2,3,4] [--variants baseline_screg_scale,nosc,nomexhat,now]
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

ROOT = Path(__file__).resolve().parents[1]                       # D:/102-npi/ude-vae-npi
RESULTS_DIR = ROOT / "results"     # project results
BL_ROOT = ROOT / "results"                                       # shared baseline root

VARIANTS = ["nosc", "now", "nomexhat"]
METHODS = ["gc", "te", "gpd"]
N_SUBJECTS = 10


# ---------------------------------------------------------------------------
# Matrix helpers (mirrors _compare_signed_de_baselines.py conventions)
# ---------------------------------------------------------------------------

def _load_matrix(json_path, key="connectivity_matrix"):
    with open(json_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    mat = np.asarray(payload[key], dtype=np.float64)
    mat[~np.isfinite(mat)] = np.nan
    return mat


def _zscore_offdiag(mat):
    """Z-score off-diagonal elements (mean=0, std=1), NaN-aware."""
    mask = ~np.eye(mat.shape[0], dtype=bool)
    vals = mat[mask]
    finite = np.isfinite(vals)
    if finite.sum() < 3:
        return mat * np.nan
    mu = vals[finite].mean()
    sigma = vals[finite].std()
    if sigma < 1e-12:
        return mat * np.nan
    out = mat.copy()
    out[mask] = (vals - mu) / sigma
    return out


def _pearson_r(a, b):
    """Pearson r on off-diagonal elements (both triangles)."""
    mask = ~np.eye(a.shape[0], dtype=bool)
    x = a[mask]
    y = b[mask]
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 3:
        return float("nan")
    return float(np.corrcoef(x[finite], y[finite])[0, 1])


def _r_abs(target, ref):
    """Pearson r on absolute values (topology only, ignore sign).

    Correct order (matches compare_params_vs_baseline.py): take |x| FIRST, then
    z-score. Taking |.| after z-score would mix weak (negative z) and strong
    (positive z) connections, destroying topology. Both args are raw matrices.
    """
    tz = _zscore_offdiag(np.abs(target))
    rz = _zscore_offdiag(np.abs(ref))
    return _pearson_r(tz, rz)


def _r_signed(target, ref):
    """Pearson r on signed values (tests sign agreement). Both args raw."""
    tz = _zscore_offdiag(target)
    rz = _zscore_offdiag(ref)
    return _pearson_r(tz, rz)


# ---------------------------------------------------------------------------
# Loaders (per fold)
# ---------------------------------------------------------------------------

def _load_signed_de(fold, variant):
    """Return (signed_mats, unsigned_mats, sign_mats); lists len == N_SUBJECTS (None if missing)."""
    d = RESULTS_DIR / "oc_connectivity_global_tau_signed_ablation" / f"fold{fold}" / variant
    signed, unsigned, signs = [], [], []
    for i in range(N_SUBJECTS):
        p = d / f"subject_{i}.json"
        if not p.exists():
            signed.append(None); unsigned.append(None); signs.append(None)
            continue
        signed.append(_load_matrix(p, "signed_connectivity_matrix"))
        unsigned.append(_load_matrix(p, "connectivity_matrix"))
        signs.append(_load_matrix(p, "sign_matrix"))
    return signed, unsigned, signs


def _load_aeff(fold, variant):
    """A_eff is subject-independent (one matrix per fold/variant). Return None if missing."""
    p = (RESULTS_DIR / "graph_connectivity_ablation" / f"fold{fold}" / variant
         / f"graph_connectivity_fold{fold}.json")
    if not p.exists():
        return None
    return _load_matrix(p, "connectivity_matrix")


def _load_baselines(fold):
    """Return {method: [subject mats or None]} for gc/te/gpd."""
    out = {}
    for m in METHODS:
        d = BL_ROOT / f"baseline_connectivity_fold{fold}" / m
        mats = []
        for i in range(N_SUBJECTS):
            p = d / f"subject_{i}.json"
            mats.append(_load_matrix(p) if p.exists() else None)
        out[m] = mats
    return out


# ---------------------------------------------------------------------------
# Correlation computations
# ---------------------------------------------------------------------------

def _agg_subject_independent(target_mats, ref_mat):
    """target_mats: list of per-subject mats; ref_mat: single subject-independent mat.
    Returns dict with r_abs and r_signed means (NaN-aware)."""
    rs_abs, rs_signed = [], []
    for t in target_mats:
        if t is None:
            continue
        rs_abs.append(_r_abs(t, ref_mat))
        rs_signed.append(_r_signed(t, ref_mat))
    return _summarize(rs_abs, rs_signed)


def _agg_per_subject(target_mats, ref_mats):
    """Both lists are per-subject; pair up non-None entries."""
    rs_abs, rs_signed = [], []
    for t, r in zip(target_mats, ref_mats):
        if t is None or r is None:
            continue
        rs_abs.append(_r_abs(t, r))
        rs_signed.append(_r_signed(t, r))
    return _summarize(rs_abs, rs_signed)


def _summarize(rs_abs, rs_signed):
    def _ms(xs):
        v = [x for x in xs if np.isfinite(x)]
        if not v:
            return {"mean": float("nan"), "std": float("nan"), "n": 0}
        return {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)}
    return {"r_abs": _ms(rs_abs), "r_signed": _ms(rs_signed)}


# ---------------------------------------------------------------------------
# Sign consensus across variants
# ---------------------------------------------------------------------------

def _sign_consensus(fold, variant_signs):
    """variant_signs: {variant: [per-subject sign matrices or None]}.
    Fraction of off-diagonal edges where all available variants agree on sign
    (all +1, or all -1). 0/absent entries excluded."""
    n = None
    per_subject = []
    for i in range(N_SUBJECTS):
        mats = []
        for v in VARIANTS:
            sm = variant_signs[v].get(fold, {}).get(i)
            if sm is not None:
                mats.append(np.asarray(sm, dtype=np.float64))
        if len(mats) < 2:
            continue
        if n is None:
            n = mats[0].shape[0]
        mask = ~np.eye(n, dtype=bool)
        agreed = 0
        total = 0
        for a in range(n):
            for b in range(n):
                if a == b:
                    continue
                signs = []
                for sm in mats:
                    val = sm[a, b]
                    if np.isfinite(val) and val != 0:
                        signs.append(1.0 if val > 0 else -1.0)
                if len(signs) < 2:
                    continue
                total += 1
                if all(s == signs[0] for s in signs):
                    agreed += 1
        if total > 0:
            per_subject.append({"subject": i, "agreed": agreed, "total": total,
                                "fraction": agreed / total})
    valid = [e["fraction"] for e in per_subject]
    return {
        "per_subject": per_subject,
        "mean_fraction": float(np.mean(valid)) if valid else float("nan"),
        "n_subjects": len(per_subject),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", default="0,1,2,3,4",
                    help="comma-separated fold indices")
    ap.add_argument("--variants", default=",".join(VARIANTS),
                    help="comma-separated variant names")
    ap.add_argument("--out-dir", default=str(RESULTS_DIR / "compare_signed_de_ablation"))
    args = ap.parse_args()

    folds = [int(x) for x in args.folds.split(",") if x.strip() != ""]
    variants = [v.strip() for v in args.variants.split(",") if v.strip() != ""]

    # Pre-load sign matrices for consensus (only what's needed)
    variant_signs = {v: {} for v in variants}

    per_fold = {}
    for fold in folds:
        print(f"\n{'=' * 80}\n=== Fold {fold} ===\n{'=' * 80}")
        baselines = _load_baselines(fold)
        fold_result = {}
        for variant in variants:
            print(f"\n--- {variant} ---")
            signed, unsigned, signs = _load_signed_de(fold, variant)
            variant_signs[variant][fold] = {i: signs[i] for i in range(len(signs))}
            aeff = _load_aeff(fold, variant)

            entry = {"signed_de": {}, "aeff": {}, "n_de_subjects": 0, "aeff_present": aeff is not None}

            # signed DE vs baselines (per-subject) and vs A_eff (subject-independent)
            for m in METHODS:
                entry["signed_de"][m] = _agg_per_subject(signed, baselines[m])
            if aeff is not None:
                entry["signed_de"]["aeff"] = _agg_subject_independent(signed, aeff)

            # A_eff vs baselines (subject-independent A_eff vs per-subject baseline)
            if aeff is not None:
                for m in METHODS:
                    entry["aeff"][m] = _agg_subject_independent(baselines[m], aeff)

            n_de = sum(1 for s in signed if s is not None)
            entry["n_de_subjects"] = n_de
            fold_result[variant] = entry

            # Print compact row
            def _fmt(d, key="r_abs"):
                if not d or d[key]["n"] == 0:
                    return "    n/a"
                return f"{d[key]['mean']:+.3f}±{d[key]['std']:.3f}"
            print(f"  signed DE vs GC : r_abs={_fmt(entry['signed_de'].get('gc'))}  "
                  f"r_signed={_fmt(entry['signed_de'].get('gc'), 'r_signed')}")
            print(f"  signed DE vs TE : r_abs={_fmt(entry['signed_de'].get('te'))}  "
                  f"r_signed={_fmt(entry['signed_de'].get('te'), 'r_signed')}")
            print(f"  signed DE vs gPD: r_abs={_fmt(entry['signed_de'].get('gpd'))}  "
                  f"r_signed={_fmt(entry['signed_de'].get('gpd'), 'r_signed')}")
            if "aeff" in entry["signed_de"]:
                print(f"  signed DE vs A_e: r_abs={_fmt(entry['signed_de'].get('aeff'))}  "
                      f"r_signed={_fmt(entry['signed_de'].get('aeff'), 'r_signed')}")

        per_fold[str(fold)] = fold_result

    # Cross-fold aggregation (mean ± std of per-fold r_abs/r_signed means)
    cross_fold = {}
    for variant in variants:
        cf = {"signed_de": {}, "aeff": {}}
        for section in ("signed_de", "aeff"):
            methods_present = set()
            for fold in folds:
                e = per_fold[str(fold)].get(variant, {}).get(section, {})
                methods_present.update(e.keys())
            for m in sorted(methods_present):
                abs_means = []
                signed_means = []
                for fold in folds:
                    e = per_fold[str(fold)].get(variant, {}).get(section, {}).get(m)
                    if e and e["r_abs"]["n"] > 0:
                        abs_means.append(e["r_abs"]["mean"])
                        signed_means.append(e["r_signed"]["mean"])
                cf[section][m] = {
                    "r_abs_mean_of_fold_means": float(np.mean(abs_means)) if abs_means else float("nan"),
                    "r_abs_std_of_fold_means": float(np.std(abs_means)) if abs_means else float("nan"),
                    "r_signed_mean_of_fold_means": float(np.mean(signed_means)) if signed_means else float("nan"),
                    "r_signed_std_of_fold_means": float(np.std(signed_means)) if signed_means else float("nan"),
                    "n_folds": len(abs_means),
                }
        cross_fold[variant] = cf

    # Sign consensus per fold
    consensus = {}
    for fold in folds:
        consensus[str(fold)] = _sign_consensus(fold, variant_signs)

    out = {
        "folds": folds,
        "variants": variants,
        "methods": METHODS,
        "n_subjects": N_SUBJECTS,
        "per_fold": per_fold,
        "cross_fold": cross_fold,
        "sign_consensus": consensus,
        "notes": {
            "r_abs": "Pearson r on |signed_DE| vs baseline (topology, ignores DE sign)",
            "r_signed": "Pearson r on signed_DE vs baseline (tests if excite/inhibit maps to +/-)",
            "aeff_subject_independent": "A_eff is one matrix per fold/variant; correlated vs each subject's baseline then averaged",
            "sign_consensus": "fraction of off-diagonal edges where >=2 available variants agree on +/-",
        },
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)

    # Print cross-fold summary table
    print(f"\n{'=' * 80}\n=== Cross-fold summary (mean of per-fold r_abs means) ===\n{'=' * 80}")
    header = f"{'variant':>24s} | {'DE-GC':>9s} {'DE-TE':>9s} {'DE-gPD':>9s} {'DE-Aeff':>9s} | {'Aeff-GC':>9s} {'Aeff-TE':>9s} {'Aeff-gPD':>9s}"
    print(header)
    print("-" * len(header))
    for variant in variants:
        cf = cross_fold[variant]
        def _cell(d, key="r_abs_mean_of_fold_means"):
            v = d.get(key)
            return f"{v:+.3f}" if np.isfinite(v) else "  n/a"
        de = cf["signed_de"]
        ae = cf["aeff"]
        print(f"{variant:>24s} | "
              f"{_cell(de.get('gc',{})):>9s} {_cell(de.get('te',{})):>9s} "
              f"{_cell(de.get('gpd',{})):>9s} {_cell(de.get('aeff',{})):>9s} | "
              f"{_cell(ae.get('gc',{})):>9s} {_cell(ae.get('te',{})):>9s} {_cell(ae.get('gpd',{})):>9s}")

    print(f"\nSaved: {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
