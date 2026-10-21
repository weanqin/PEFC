"""Per-subject GC-EC on edra_process_v4 with the ORIGINAL latent_flow method.

Task 10-21-oldlib-gc-edrav4. Blueprint: refs/granger_bivariate_spectral/
edrafmt48-cont-zscore-fs250-fixed8-f1-125-nf65/gen_refs.py (task 10-19
Run 005); only the data source changed plus one added preprocessing step.

Caliber (user-approved 2026-10-21): edra_process_v4 native 500 Hz ->
scipy.signal.resample_poly(x, 1, 2) -> 250 Hz, then the EXACT edrafmt48
caliber: pairwise bivariate spectral GC (Geweke 1982 / Dhamala 2008),
FIXED AR order 8, n_freq=65 bins, integrated over 1-125 Hz, full continuous
per-subject timeseries (no windowing), per-channel z-score AFTER resampling
(ddof=0, std==0 -> 1, same convention as latent_flow baseline data_utils),
all 46 subjects (HC 24 / MDD 22), no qc_pass filtering (registered fact:
manifest qc_pass=True only 13/46; dataset released all as status=ok).

Resumable: skips subjects whose npz already exists.
Writes refs_s{id}.npz (ec float32 (62,62) row=source col=target, diagonal 0,
subject_id, group, params) into OUT.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import glob
import json
import time

import numpy as np
from scipy.signal import resample_poly

import sys
sys.path.insert(0, "/root/npi/models/latent-flow-v1")

from latent_flow.baseline.granger import granger_connectivity

OUT = "/root/npi/outputs/10-21-oldlib-gc-edrav4/refs-01"
V4_ROOT = "/root/autodl-tmp/edra_process_v4"
FMT_CHANNEL_REF = "/root/autodl-tmp/edra_processed_fmt/HC/13_rest_v4.npz"
N_JOBS = 12
PARAMS = {"method": "granger_bivariate_spectral (old-library defaults)",
          "order": "fixed 8", "n_freq": 65, "fmin_hz": 1.0, "fmax_hz": 125.0,
          "fs": 250.0, "integral": "trapezoidal",
          "reference": "Geweke 1982; Dhamala 2008",
          "data": "edra_process_v4 full continuous, resample_poly(1,2) "
                  "500->250 Hz (85000->42500 samples), per-channel z-score "
                  "after resampling"}


def load_v4_resampled(path, ch_ref):
    """Read one v4 npz, assert contract, resample to 250 Hz, z-score."""
    z = np.load(path, allow_pickle=True)
    data = z["data"]
    assert data.shape == (62, 85000), (path, data.shape)
    assert data.dtype == np.float32, (path, data.dtype)
    assert int(np.asarray(z["fs"]).item()) == 500, (path, "fs != 500")
    sid_file = int(os.path.basename(path).split("_")[0])
    sid_npz = int(np.asarray(z["subject_id"]).item())
    assert sid_npz == sid_file, (path, sid_npz, sid_file)
    group_dir = os.path.basename(os.path.dirname(path))
    group_npz = str(np.asarray(z["group"]).item())
    assert group_npz == group_dir, (path, group_npz, group_dir)
    ch = [str(c) for c in z["channel_names"]]
    assert ch == ch_ref, (path, "channel order mismatch vs fmt reference")

    res = resample_poly(data.astype(np.float64), 1, 2, axis=-1)
    assert res.shape == (62, 42500), (path, res.shape)
    res = res.astype(np.float32)

    mean = res.mean(axis=1, keepdims=True)
    std = res.std(axis=1, keepdims=True)
    std[std == 0] = 1.0  # avoid div by zero (mirrors MODMADataset / loader)
    res = (res - mean) / std
    assert np.all(np.isfinite(res)), (path, "non-finite after z-score")
    return res, sid_file, group_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=None,
                    help="run a single subject id only (smoke run)")
    args = ap.parse_args()

    ch_ref = [str(c) for c in
              np.load(FMT_CHANNEL_REF, allow_pickle=True)["channel_names"]]

    files = sorted(glob.glob(f"{V4_ROOT}/HC/*.npz")) + \
            sorted(glob.glob(f"{V4_ROOT}/MDD/*.npz"))
    assert len(files) == 46, len(files)
    os.makedirs(OUT, exist_ok=True)

    if args.smoke is not None:
        files = [f for f in files
                 if int(os.path.basename(f).split("_")[0]) == args.smoke]
        assert len(files) == 1, args.smoke

    t_all = time.time()
    for n, f in enumerate(files):
        sid = int(os.path.basename(f).split("_")[0])
        group = os.path.basename(os.path.dirname(f))
        out = f"{OUT}/refs_s{sid}.npz"
        if os.path.exists(out):
            print(f"[{n+1}/{len(files)}] s{sid} ({group}): exists, skip",
                  flush=True)
            continue
        data, sid, group = load_v4_resampled(f, ch_ref)
        t0 = time.time()
        ec = granger_connectivity(
            data.astype(np.float64), fs=250.0, max_order=8, n_freq=65,
            fmin=1.0, fmax=125.0, n_jobs=N_JOBS, fixed_order=8)
        dt = time.time() - t0
        np.savez_compressed(out, ec=ec.astype(np.float32),
                            subject_id=sid, group=group,
                            params=json.dumps(PARAMS))
        od = ec[~np.eye(62, dtype=bool)]
        assert (od >= 0).all(), (sid, "negative offdiag GC")
        zmean = float(np.abs(data).mean())
        print(f"[{n+1}/{len(files)}] s{sid} ({group}): {dt:.1f}s | "
              f"offdiag mean={od.mean():.4f} median={np.median(od):.4f} "
              f"max={od.max():.3f} | |x|~{zmean:.4f}", flush=True)
    print(f"ALL DONE in {(time.time()-t_all)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
