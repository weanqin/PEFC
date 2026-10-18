"""BrainVision (.vhdr + .dat + .vmrk) reader for the EDRA dataset.

EDRA files are BrainVision Core Data Format:
    - ``.vhdr``  INI-style header (Common / Binary / Channel Infos sections)
    - ``.dat``   raw binary, IEEE_FLOAT_32, DataOrientation=VECTORIZED
                 (channel-major: ch1 all samples, ch2 all samples, ...)
    - ``.vmrk``  marker file (empty / 0-byte for EDRA resting-state)

The reader returns the same ``(n_channels, n_samples)`` float32 contract as
``data.modma.load_mat_file`` so the downstream ``MODMADataset`` windowing is
format-agnostic. Optional per-channel z-score matches the MODMA convention.

Only the VECTORIZED + IEEE_FLOAT_32 case is supported (that is what EDRA uses);
MULTIPLEXED and INT_16 raise NotImplementedError with a clear message rather
than silently misreading the binary.
"""

from pathlib import Path

import numpy as np


def _parse_vhdr(vhdr_path):
    """Parse the .vhdr header file into a dict of section -> {key: value}.

    Channel names are returned as an ordered list under ``["Channel Infos"][
    "names"]`` (index i <-> Ch<i+1>).
    """
    p = Path(vhdr_path)
    text = p.read_text(encoding="latin-1")  # BrainVision headers are latin-1

    sections = {}
    current = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(";"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1].strip()
            sections[current] = {}
            continue
        if current is None or "=" not in line:
            continue
        key, _, val = line.partition("=")
        sections[current][key.strip()] = val.strip()

    # Ordered channel-name list from [Channel Infos]: Ch<i>=<Name>,<ref>,...
    names = []
    chan_infos = sections.get("Channel Infos", {})
    # Keys are "Ch1".."ChN"; sort by integer suffix, not lexicographically.
    def _chidx(k):
        return int(k[2:]) if k.lower().startswith("ch") else 1 << 30

    for key in sorted(chan_infos, key=_chidx):
        if not key.lower().startswith("ch"):
            continue
        name = chan_infos[key].split(",")[0].strip()
        names.append(name)
    if names:
        sections["Channel Infos"]["names"] = names
    return sections


def read_brainvision_header(vhdr_path):
    """Return ``(n_channels, n_samples, fs, channel_names, dat_path)``.

    Resolves the referenced ``.dat`` binary path relative to the ``.vhdr``.
    """
    sections = _parse_vhdr(vhdr_path)
    common = sections.get("Common Infos", {})
    binary = sections.get("Binary Infos", {})

    if common.get("DataOrientation") != "VECTORIZED":
        raise NotImplementedError(
            f"Only DataOrientation=VECTORIZED is supported "
            f"(got {common.get('DataOrientation')!r}); "
            f"MULTIPLEXED BrainVision files are not implemented."
        )
    fmt = binary.get("BinaryFormat", "")
    if fmt != "IEEE_FLOAT_32":
        raise NotImplementedError(
            f"Only BinaryFormat=IEEE_FLOAT_32 is supported (got {fmt!r})."
        )

    n_channels = int(common["NumberOfChannels"])
    n_samples = int(common["DataPoints"])
    sampling_interval_us = float(common["SamplingInterval"])
    fs = 1_000_000.0 / sampling_interval_us

    names = sections.get("Channel Infos", {}).get("names")
    if names is not None and len(names) != n_channels:
        raise ValueError(
            f"Channel-name count ({len(names)}) != NumberOfChannels "
            f"({n_channels}) in {vhdr_path}"
        )

    dat_file = common["DataFile"]
    dat_path = Path(vhdr_path).parent / dat_file
    return n_channels, n_samples, fs, names, dat_path


def read_brainvision(vhdr_path, normalize=True):
    """Load a BrainVision recording as ``(n_channels, n_samples)`` float32.

    Args:
        vhdr_path: path to the ``.vhdr`` header (the ``.dat`` is resolved
            relative to it).
        normalize: if True, per-channel z-score (mean 0, std 1 per channel),
            matching ``MODMADataset``'s convention. std==0 is guarded.

    Returns:
        ndarray ``(n_channels, n_samples)`` float32.
    """
    n_channels, n_samples, _fs, _names, dat_path = read_brainvision_header(
        vhdr_path
    )
    # VECTORIZED = channel-major: ch1 samples, then ch2 samples, ...
    raw = np.fromfile(str(dat_path), dtype=np.float32)
    expected = n_channels * n_samples
    if raw.size != expected:
        raise ValueError(
            f"{dat_path}: expected {expected} float32 values "
            f"({n_channels} ch x {n_samples} samples), got {raw.size}."
        )
    data = raw.reshape(n_channels, n_samples)

    if normalize:
        mean = data.mean(axis=1, keepdims=True)
        std = data.std(axis=1, keepdims=True)
        std[std == 0] = 1.0  # avoid div by zero (mirrors MODMADataset)
        data = (data - mean) / std

    return data.astype(np.float32)
