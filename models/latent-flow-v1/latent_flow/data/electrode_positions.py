"""Standard 10-20 electrode 3D positions for the EDRA 62-channel montage.

Coordinates are in **metres** in the MNI head frame, sourced from MNE-Python's
``make_standard_montage('standard_1020')`` (task 07-28-edra-dataset-adaptation).
This vendored table makes the EDRA geometry reproducible WITHOUT an MNE install;
``scripts/build_edra_geometry.py`` prefers MNE when importable and falls back to
this table otherwise (identical values).

The one non-standard entry is ``Poz`` (POz), which is absent from MNE's
``standard_1020`` montage. It is defined by the standard convention as the
midline midpoint between Pz and Oz (equivalently, midway between PO3 and PO4),
both of which agree to within ~1 mm. The midpoint-of-(Pz,Oz) value is used.

The table is keyed by the EDRA native channel order (Ch1..Ch62), matching the
``.vhdr`` ``[Channel Infos]`` ordering so that ``coords[i]`` corresponds to
channel ``Ch<i+1>`` of the BrainVision file.
"""

import numpy as np


# EDRA native channel order (Ch1..Ch62), matching the .vhdr Channel Infos.
EDRA_CHANNEL_ORDER = [
    "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8", "T7",
    "T8", "P7", "P8", "Fz", "Cz", "Pz", "FC1", "FC2", "CP1", "CP2", "FC5",
    "FC6", "CP5", "CP6", "FT10", "TP9", "TP10", "F1", "F2", "C1", "C2", "P1",
    "P2", "AF3", "AF4", "FC3", "FC4", "CP3", "CP4", "PO3", "PO4", "F5", "F6",
    "C5", "C6", "P5", "P6", "AF7", "AF8", "FT7", "FT8", "TP7", "TP8", "PO7",
    "PO8", "Fpz", "CPz", "Poz", "Oz", "FCz",
]

# 3D positions in metres, MNI head frame, from MNE standard_1020.
# Poz is the midline midpoint of Pz and Oz (standard convention).
STANDARD_1020_METRES = {
    "Fp2": [0.029872, 0.084896, -0.00708],
    "F3": [-0.050244, 0.053111, 0.042192],
    "F4": [0.051836, 0.054305, 0.040814],
    "C3": [-0.065358, -0.011632, 0.064358],
    "C4": [0.067118, -0.0109, 0.06358],
    "P3": [-0.053007, -0.078788, 0.05594],
    "P4": [0.055667, -0.07856, 0.056561],
    "O1": [-0.029413, -0.112449, 0.008839],
    "O2": [0.029843, -0.112156, 0.0088],
    "F7": [-0.070263, 0.042474, -0.01142],
    "F8": [0.073043, 0.044422, -0.012],
    "T7": [-0.084161, -0.016019, -0.009346],
    "T8": [0.08508, -0.01502, -0.00949],
    "P7": [-0.072434, -0.073453, -0.002487],
    "P8": [0.073056, -0.073068, -0.00254],
    "Fz": [0.000312, 0.058512, 0.066462],
    "Cz": [0.000401, -0.009167, 0.100244],
    "Pz": [0.000325, -0.081115, 0.082615],
    "FC1": [-0.034062, 0.026011, 0.079987],
    "FC2": [0.034784, 0.026438, 0.078808],
    "CP1": [-0.035513, -0.047292, 0.091315],
    "CP2": [0.038384, -0.047073, 0.090695],
    "FC5": [-0.077215, 0.018643, 0.02446],
    "FC6": [0.079534, 0.019936, 0.024438],
    "CP5": [-0.079592, -0.046551, 0.030949],
    "CP6": [0.083322, -0.046101, 0.031206],
    "FT10": [0.084113, 0.014365, -0.050538],
    "TP9": [-0.085619, -0.046515, -0.045707],
    "TP10": [0.086162, -0.047035, -0.045869],
    "F1": [-0.027496, 0.056931, 0.060342],
    "F2": [0.029514, 0.057602, 0.05954],
    "C1": [-0.036158, -0.009984, 0.089752],
    "C2": [0.037672, -0.009624, 0.088412],
    "P1": [-0.02862, -0.080525, 0.075436],
    "P2": [0.03192, -0.080487, 0.076716],
    "AF3": [-0.033701, 0.076837, 0.021227],
    "AF4": [0.035712, 0.077726, 0.021956],
    "FC3": [-0.060182, 0.022716, 0.055544],
    "FC4": [0.062293, 0.023723, 0.05563],
    "CP3": [-0.063556, -0.047009, 0.065624],
    "CP4": [0.066612, -0.046637, 0.06558],
    "PO3": [-0.036511, -0.100853, 0.037167],
    "PO4": [0.036782, -0.100849, 0.036397],
    "F5": [-0.064466, 0.048035, 0.016921],
    "F6": [0.067914, 0.04983, 0.016367],
    "C5": [-0.08028, -0.01376, 0.02916],
    "C6": [0.083456, -0.012776, 0.029208],
    "P5": [-0.067272, -0.076291, 0.028382],
    "P6": [0.067888, -0.075904, 0.028091],
    "AF7": [-0.05484, 0.068572, -0.01059],
    "AF8": [0.055743, 0.069657, -0.010755],
    "FT7": [-0.080775, 0.01412, -0.011135],
    "FT8": [0.081815, 0.015417, -0.01133],
    "TP7": [-0.08483, -0.046022, -0.007056],
    "TP8": [0.085549, -0.045545, -0.00713],
    "PO7": [-0.05484, -0.097528, 0.002792],
    "PO8": [0.055667, -0.097625, 0.00273],
    "Fpz": [0.000112, 0.088247, -0.001713],
    "CPz": [0.000386, -0.047318, 0.099432],
    "Poz": [0.000216, -0.098003, 0.048636],
    "Oz": [0.000108, -0.114892, 0.014657],
    "FCz": [0.000376, 0.02739, 0.088668],
}


def get_edra_coords(channel_names=None):
    """Return ``(coords, names)`` for the EDRA 10-20 montage.

    Args:
        channel_names: optional ordered list; if given, the returned coords are
            re-ordered to match it (asserts every name is known). If None, the
            EDRA native order (Ch1..Ch62) is used.

    Returns:
        coords : float32 ndarray (62, 3) in metres, MNI head frame.
        names  : list[str] matching the coords row order.
    """
    if channel_names is None:
        channel_names = EDRA_CHANNEL_ORDER
    missing = [n for n in channel_names if n not in STANDARD_1020_METRES]
    if missing:
        raise ValueError(f"Unknown electrode names: {missing}")
    coords = np.array(
        [STANDARD_1020_METRES[n] for n in channel_names], dtype=np.float32
    )
    return coords, list(channel_names)
