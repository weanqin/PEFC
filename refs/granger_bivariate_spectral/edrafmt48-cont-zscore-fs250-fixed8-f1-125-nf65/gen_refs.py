"""Recompute per-subject GC-EC on edra_fmt with the ORIGINAL latent_flow method.

Caliber (baseline/granger.py defaults, old-library): pairwise bivariate
spectral GC (Geweke 1982 / Dhamala 2008), FIXED AR order 8, n_freq=65 bins,
integrated over 1-125 Hz (Nyquist at fs=250), full continuous per-subject
timeseries (no windowing), per-channel z-score, all 48 edra_fmt subjects.

Resumable: skips subjects whose npz already exists. Writes refs_s{id}.npz
(ec float32 (62,62) row=source, params) into this directory.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import time
import glob
import json

import numpy as np

sys.path.insert(0, "/root/npi/models/latent-flow-v1")

from latent_flow.baseline.data_utils import load_subject_timeseries
from latent_flow.baseline.granger import granger_connectivity

OUT = "/root/npi/outputs/10-19-latent-flow-edra-fmt-train/run_01/gc_oldlib_refs"
N_JOBS = 12
PARAMS = {"method": "granger_bivariate_spectral (old-library defaults)",
          "order": "fixed 8", "n_freq": 65, "fmin_hz": 1.0, "fmax_hz": 125.0,
          "fs": 250.0, "integral": "trapezoidal",
          "reference": "Geweke 1982; Dhamala 2008",
          "data": "edra_fmt full continuous (60001 samples), per-channel z-score"}


def main():
    files = sorted(glob.glob("/root/autodl-tmp/edra_processed_fmt/HC/*.npz")) + \
            sorted(glob.glob("/root/autodl-tmp/edra_processed_fmt/MDD/*.npz"))
    assert len(files) == 48, len(files)
    t_all = time.time()
    for n, f in enumerate(files):
        sid = int(os.path.basename(f).split("_")[0])
        group = os.path.basename(os.path.dirname(f))
        out = f"{OUT}/refs_s{sid}.npz"
        if os.path.exists(out):
            print(f"[{n+1}/48] s{sid} ({group}): exists, skip", flush=True)
            continue
        data = load_subject_timeseries(f, normalize=True, dataset="edra_fmt")
        t0 = time.time()
        ec = granger_connectivity(
            data.astype(np.float64), fs=250.0, max_order=8, n_freq=65,
            fmin=1.0, fmax=125.0, n_jobs=N_JOBS, fixed_order=8)
        dt = time.time() - t0
        np.savez_compressed(out, ec=ec.astype(np.float32),
                            subject_id=sid, group=group,
                            params=json.dumps(PARAMS))
        od = ec[~np.eye(62, dtype=bool)]
        print(f"[{n+1}/48] s{sid} ({group}): {dt:.1f}s | "
              f"offdiag mean={od.mean():.3f} max={od.max():.3f}", flush=True)
    print(f"ALL DONE in {(time.time()-t_all)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
