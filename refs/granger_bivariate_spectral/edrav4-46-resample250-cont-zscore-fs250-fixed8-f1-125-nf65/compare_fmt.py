"""Compare v4-resampled GC refs with the edrafmt48 archive (task 10-21).

Descriptive registration only (PRD R3): per overlapping subject, Spearman
rho + Pearson r over the 3782 offdiag edges (v4 refs vs fmt refs) and the
offdiag median ratio (v4/fmt). Group-level mean+/-sd (ddof=1). No threshold,
no cross-caliber comparison deliverable — this file registers that the
resampled-v4 caliber lands in a sane place relative to the fmt archive.

s56 (in v4, absent from fmt 48) is skipped and registered.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import glob
import json

import numpy as np
from scipy.stats import pearsonr, spearmanr

V4_DIR = "/root/npi/outputs/10-21-oldlib-gc-edrav4/refs-01"
FMT_DIR = ("/root/npi/refs/granger_bivariate_spectral/"
           "edrafmt48-cont-zscore-fs250-fixed8-f1-125-nf65")
OUT_JSON = ("/root/npi/outputs/10-21-oldlib-gc-edrav4/"
            "compare_fmt_summary.json")


def load_refs(d):
    out = {}
    for f in sorted(glob.glob(f"{d}/refs_s*.npz")):
        sid = int(os.path.basename(f).split("refs_s")[1].split(".")[0])
        z = np.load(f, allow_pickle=True)
        ec = z["ec"]
        assert ec.shape == (62, 62) and ec.dtype == np.float32, (d, f)
        out[sid] = {"ec": ec, "group": str(np.asarray(z["group"]).item())}
    return out


def main():
    v4, fmt = load_refs(V4_DIR), load_refs(FMT_DIR)
    assert len(v4) == 46, len(v4)
    assert len(fmt) == 48, len(fmt)
    overlap = sorted(set(v4) & set(fmt))
    v4_only = sorted(set(v4) - set(fmt))
    fmt_only = sorted(set(fmt) - set(v4))
    print(f"v4={len(v4)} fmt={len(fmt)} overlap={len(overlap)} "
          f"v4_only={v4_only} fmt_only={fmt_only}")

    eye = np.eye(62, dtype=bool)
    rows = []
    for sid in overlap:
        a = v4[sid]["ec"][~eye].astype(np.float64)
        b = fmt[sid]["ec"][~eye].astype(np.float64)
        assert (a >= 0).all() and (b >= 0).all(), sid
        rho = float(spearmanr(a, b).statistic)
        r = float(pearsonr(a, b).statistic)
        ratio = float(np.median(a) / np.median(b))
        rows.append({"subject_id": sid, "group": v4[sid]["group"],
                     "spearman_rho": rho, "pearson_r": r,
                     "median_ratio_v4_over_fmt": ratio,
                     "median_v4": float(np.median(a)),
                     "median_fmt": float(np.median(b))})
        print(f"s{sid} ({rows[-1]['group']}): rho={rho:.4f} r={r:.4f} "
              f"median v4/fmt={ratio:.3f}", flush=True)

    rhos = np.array([x["spearman_rho"] for x in rows])
    rs = np.array([x["pearson_r"] for x in rows])
    ratios = np.array([x["median_ratio_v4_over_fmt"] for x in rows])
    summary = {
        "purpose": "descriptive registration only (PRD R3), no threshold",
        "n_v4": len(v4), "n_fmt": len(fmt),
        "overlap_ids": overlap, "n_overlap": len(overlap),
        "v4_only_ids": v4_only, "fmt_only_ids": fmt_only,
        "edge_set": "offdiag 3782, ec[a,b]=a->b row=source, both sides",
        "per_subject": rows,
        "group": {
            "spearman_rho_mean": float(rhos.mean()),
            "spearman_rho_sd": float(rhos.std(ddof=1)),
            "spearman_rho_min": float(rhos.min()),
            "spearman_rho_max": float(rhos.max()),
            "pearson_r_mean": float(rs.mean()),
            "pearson_r_sd": float(rs.std(ddof=1)),
            "median_ratio_mean": float(ratios.mean()),
            "median_ratio_sd": float(ratios.std(ddof=1)),
        },
    }
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=1)
    g = summary["group"]
    print(f"\nGROUP: spearman rho={g['spearman_rho_mean']:.4f}"
          f"+/-{g['spearman_rho_sd']:.4f} "
          f"[{g['spearman_rho_min']:.4f}, {g['spearman_rho_max']:.4f}] | "
          f"pearson r={g['pearson_r_mean']:.4f}+/-{g['pearson_r_sd']:.4f} | "
          f"median ratio={g['median_ratio_mean']:.3f}"
          f"+/-{g['median_ratio_sd']:.3f}")
    print(f"saved {OUT_JSON}")


if __name__ == "__main__":
    main()
