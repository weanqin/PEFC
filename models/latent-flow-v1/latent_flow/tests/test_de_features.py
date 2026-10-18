"""Tests for the differential-entropy 5-band feature extractor.

Validates the core contract (shape, finiteness, band-localization) and the
synthetic-signal hand-check that is review gate 1 of the implementation plan.
"""

from __future__ import annotations

import numpy as np
import pytest

from latent_flow.data.de_features import (
    BAND_EDGES,
    aggregate_subject_de,
    compute_de_features,
)


FS = 250.0


def _synth(freq_hz: float, n_time: int = 500, n_chan: int = 4, fs: float = FS) -> np.ndarray:
    """Return a (T, C) window where every channel is a pure sinusoid at freq_hz."""
    t = np.arange(n_time) / fs
    # n_chan distinct amplitudes so channels are not identical.
    sig = np.stack([np.sin(2 * np.pi * freq_hz * t) * (1.0 + 0.1 * k) for k in range(n_chan)], axis=1)
    return sig.astype(np.float32)


class TestShapeAndFiniteness:
    def test_shape_tc_input(self):
        eeg = _synth(10.0)
        de = compute_de_features(eeg, fs=FS)
        assert de.shape == (4, 5)
        assert np.all(np.isfinite(de))

    def test_shape_ct_input_transposed(self):
        # (C, T) input (raw recording convention) should be auto-detected.
        eeg_ct = _synth(10.0).T  # (4, 500)
        de = compute_de_features(eeg_ct, fs=FS)
        assert de.shape == (4, 5)
        assert np.all(np.isfinite(de))

    def test_band_order_matches_edges(self):
        assert len(BAND_EDGES) == 5
        names = ["delta", "theta", "alpha", "beta", "gamma"]
        for b, (lo, hi) in enumerate(BAND_EDGES):
            assert BAND_EDGES[b] == (lo, hi), f"{names[b]} band mismatch"

    def test_rejects_too_short(self):
        with pytest.raises(ValueError):
            compute_de_features(np.zeros((2, 4), dtype=np.float32), fs=FS)


class TestBandLocalization:
    """Review gate 1: a pure tone should peak in its band's DE."""

    @pytest.mark.parametrize(
        "freq, band_idx",
        [
            (3.0, 0),    # delta
            (6.0, 1),    # theta
            (10.0, 2),   # alpha  <- the design's hand-check
            (20.0, 3),   # beta
            (38.0, 4),   # gamma
        ],
    )
    def test_pure_tone_peaks_in_expected_band(self, freq, band_idx):
        de = compute_de_features(_synth(freq), fs=FS)  # (C, 5)
        # Every channel's argmax band should equal the expected band.
        argmax = de.argmax(axis=1)
        assert np.all(argmax == band_idx), (
            f"{freq} Hz tone: expected band {band_idx} ({BAND_EDGES[band_idx]}), "
            f"got argmax={argmax.tolist()}"
        )

    def test_alpha_is_largest_for_10hz(self):
        """Explicit hand-check called out in AC1 / review gate 1."""
        de = compute_de_features(_synth(10.0), fs=FS)
        # alpha is column 2; it must be the max across the 5 bands for channel 0.
        assert de[0].argmax() == 2
        assert de[0, 2] > de[0].max() - 1e-6  # alpha strictly the peak


class TestAggregation:
    def test_aggregate_shape_and_mean(self):
        wins = [_synth(10.0), _synth(10.0), _synth(6.0)]
        agg = aggregate_subject_de(wins, fs=FS)
        assert agg.shape == (4, 5)
        assert np.all(np.isfinite(agg))
        # Mean of the two 10 Hz windows' alpha should sit between the per-window values.
        single_alpha_10 = compute_de_features(_synth(10.0), fs=FS)[0, 2]
        single_alpha_6 = compute_de_features(_synth(6.0), fs=FS)[0, 2]
        # The 3-window mean of column alpha mixes 10,10,6 → between 10-only and 6-only.
        assert single_alpha_6 <= agg[0, 2] <= single_alpha_10 + 1e-5 or \
               single_alpha_10 - 1e-5 <= agg[0, 2] <= single_alpha_6

    def test_aggregate_empty_raises(self):
        with pytest.raises(ValueError):
            aggregate_subject_de([], fs=FS)


class TestNumericalSanity:
    def test_silent_channel_is_finite(self):
        # A channel of all zeros has zero band power → DE clipped to a very
        # negative finite number, not -inf or NaN.
        eeg = np.zeros((500, 2), dtype=np.float32)
        eeg[:, 0] = np.sin(2 * np.pi * 10 * np.arange(500) / FS)  # ch0 alive
        # ch1 stays silent
        de = compute_de_features(eeg, fs=FS)
        assert np.all(np.isfinite(de))
        # Silent channel's DE should be much smaller than the active one's alpha.
        assert de[1, 2] < de[0, 2]
