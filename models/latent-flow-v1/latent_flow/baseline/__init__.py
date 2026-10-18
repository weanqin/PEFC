"""Traditional directed-connectivity baselines for MODMA EEG.

Three classical methods, each producing a per-subject directed 128x128
matrix (rows = source channel a, columns = target channel b) directly
comparable with ``results/oc_connectivity/`` (model attribution):

- ``granger``     -- pairwise bivariate spectral Granger Causality
                     (Geweke 1982 frequency-domain decomposition, full-band
                     integral). Kernel in ``granger.py``.
- ``symbolic_te`` -- symbolic Transfer Entropy via Bandt-Pompe permutation
                     patterns (m=3, tau=1). Kernel in ``transfer_entropy.py``.
- ``gpdc``        -- generalized Partial Directed Coherence (Baccala et al.
                     2007), MVAR-based frequency-domain partialised
                     connectivity. Kernel in ``gpdc.py``.

These are independent of the trained forecast model -- they estimate
connectivity straight from the (z-scored) EEG timeseries -- and serve as
the missing classical baselines against the OC-LatentFlow attribution
matrices. Data loading reuses ``data/modma.py`` and ``data/splitter.py``
without modification (see ``data_utils.py``).

A spectral DCM baseline was attempted (task 07-19) but deprecated: the
pairwise 2-node inversion collapses to the prior mean on realistic EEG.
The ``baseline/dcm.py`` kernel + ``tests/test_dcm_*.py`` are kept for
reference. See ``.trellis/tasks/07-19-dcm-ec-baseline/`` for the failure
analysis.
"""
