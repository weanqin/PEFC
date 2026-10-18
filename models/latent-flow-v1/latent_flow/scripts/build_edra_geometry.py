"""Build the EDRA 62-channel 10-20 geometry artifact (``geometry_edra.pt``).

One-shot builder. Resolves the 62 EDRA electrode names to 3D positions
(metres, MNI head frame), asserts the name set matches a real EDRA ``.vhdr``
channel list, sanity-checks the coordinate scale against the MODMA geometry,
and writes ``geometry_edra.pt`` next to the EDRA data.

Coordinate source:
    - Prefers MNE-Python ``make_standard_m1020`` if importable (authoritative).
    - Otherwise falls back to the vendored table in
      ``data/electrode_positions.py`` (identical values).
    - ``Poz`` (absent from MNE's standard_1020) is the midline midpoint of
      Pz and Oz in both paths.

Run:  python -m latent_flow.scripts.build_edra_geometry
"""

import numpy as np
import torch

from latent_flow.config import EDRA_DATA_ROOT, EDRA_GEOMETRY_PATH, GEOMETRY_PATH
from latent_flow.data.brainvision import read_brainvision_header
from latent_flow.data.datasets import list_hc


def _coords_from_mne(names):
    """Resolve *names* via MNE standard_1020; POz = midpoint(Pz, Oz)."""
    import mne
    montage = mne.channels.make_standard_montage("standard_1020")
    ch_pos = montage.get_positions()["ch_pos"]
    out = []
    for n in names:
        if n == "Poz":  # absent from MNE standard_1020
            v = (np.array(ch_pos["Pz"]) + np.array(ch_pos["Oz"])) / 2.0
        else:
            v = np.array(ch_pos[n])
        out.append(v)
    return np.array(out, dtype=np.float32)


def _coords_from_vendored(names):
    """Resolve *names* via the vendored table (no MNE dependency)."""
    from latent_flow.data.electrode_positions import STANDARD_1020_METRES
    return np.array([STANDARD_1020_METRES[n] for n in names], dtype=np.float32)


def main():
    # 1. Read the canonical EDRA channel order from a real .vhdr.
    hc_files = list_hc(__import__(
        "latent_flow.data.datasets", fromlist=["get_dataset"]
    ).get_dataset("edra"))
    sample_vhdr = hc_files[0]
    nch, npt, fs, names, _ = read_brainvision_header(sample_vhdr)
    print(f"EDRA sample header {sample_vhdr}: {nch} ch, {fs} Hz, {npt} samples")
    print(f"  channel order: {names}")

    # 2. Resolve coords (MNE preferred, vendored fallback).
    try:
        import mne  # noqa: F401
        source = "MNE standard_1020"
        coords = _coords_from_mne(names)
    except ImportError:
        source = "vendored STANDARD_1020_METRES"
        coords = _coords_from_vendored(names)
    print(f"Coordinate source: {source}")

    # 3. Assert name set + order match the vendored table (consistency guard).
    from latent_flow.data.electrode_positions import EDRA_CHANNEL_ORDER
    assert names == EDRA_CHANNEL_ORDER, (
        "EDRA .vhdr channel order != vendored EDRA_CHANNEL_ORDER.\n"
        f"  vhdr:  {names}\n  table: {EDRA_CHANNEL_ORDER}"
    )

    # 4. Sanity-check the coordinate scale against MODMA geometry (both metres).
    if GEOMETRY_PATH.exists():
        mod = torch.load(str(GEOMETRY_PATH), weights_only=False)
        mod_coords = mod["coords"].float().numpy()
        from scipy.spatial.distance import pdist
        edra_med = float(np.median(pdist(coords)))
        mod_med = float(np.median(pdist(mod_coords)))
        ratio = edra_med / mod_med if mod_med > 0 else float("nan")
        print(f"Median pairwise distance: EDRA={edra_med:.4f} m, "
              f"MODMA={mod_med:.4f} m, ratio={ratio:.3f}")
        # Both are ~0.1-0.13 m head scales; allow a generous 3x tolerance.
        assert 0.3 < ratio < 3.0, (
            f"EDRA/MODMA coord scale mismatch (ratio {ratio:.3f}); check units."
        )
    else:
        print(f"(skipped MODMA parity check: {GEOMETRY_PATH} not found)")

    # 5. Save.
    payload = {
        "coords": torch.tensor(coords, dtype=torch.float32),
        "channel_names": list(names),
    }
    EDRA_GEOMETRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(EDRA_GEOMETRY_PATH))
    print(f"Wrote {EDRA_GEOMETRY_PATH}  coords={tuple(payload['coords'].shape)} "
          f"names={len(payload['channel_names'])}")


if __name__ == "__main__":
    main()
