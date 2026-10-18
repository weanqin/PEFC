"""Subject-level stratified fold splitter for cross-validation.

Splits MDD and HC subjects independently then combines them per fold,
preserving the diagnostic-group proportion in each fold.
"""

import numpy as np


class SubjectFoldSplitter:
    """Split MDD and HC subjects into stratified folds.

    Each group is shuffled separately with a fixed seed, then divided into
    folds.  ``get_fold(i)`` returns the i-th segment of each group combined
    as the test set; the remaining subjects form the training set.

    Parameters
    ----------
    mdd_files : list[str]
        File paths for the MDD group.
    hc_files : list[str]
        File paths for the HC group.
    n_folds : int, default 5
        Number of folds.
    seed : int, default 42
        Random seed for reproducible shuffling.
    """

    # Legacy fold boundaries that match the original hard-coded indices in
    # ``train_ncde.py``.  They are kept so that existing experiments using
    # ``--fold`` with the standard MODMA dataset (24 MDD + 29 HC, 5 folds)
    # produce the exact same subject groupings.
    _LEGACY_BOUNDS = {
        (24, 5): [0, 4, 9, 14, 19, 24],
        (29, 5): [0, 6, 11, 17, 23, 29],
    }

    def __init__(self, mdd_files, hc_files, n_folds=5, seed=42):
        self.mdd_files = sorted(mdd_files)
        self.hc_files = sorted(hc_files)
        self.n_folds = n_folds
        self.seed = seed

        rng = np.random.RandomState(seed)
        self._mdd = list(self.mdd_files)
        self._hc = list(self.hc_files)
        rng.shuffle(self._mdd)
        rng.shuffle(self._hc)

        self._mdd_bounds = self._compute_bounds(len(self._mdd), n_folds)
        self._hc_bounds = self._compute_bounds(len(self._hc), n_folds)

    def _compute_bounds(self, n, n_folds):
        """Return fold boundary indices (length n_folds + 1)."""
        key = (n, n_folds)
        if key in self._LEGACY_BOUNDS:
            return self._LEGACY_BOUNDS[key]

        # Generic fallback: ``np.array_split`` then reverse so fold 0 gets
        # the smallest group (consistent with the legacy MDD pattern).
        splits = np.array_split(np.arange(n), n_folds)
        bounds = [0]
        for split in reversed(splits):
            bounds.append(bounds[-1] + len(split))
        return bounds

    def get_fold(self, fold_idx):
        """Return ``(train_files, test_files)`` for *fold_idx*.

        Raises
        ------
        ValueError
            If *fold_idx* is out of range.
        """
        if not 0 <= fold_idx < self.n_folds:
            raise ValueError(
                f"fold_idx must be in [0, {self.n_folds}), got {fold_idx}"
            )

        test_mdd = self._mdd[
            self._mdd_bounds[fold_idx] : self._mdd_bounds[fold_idx + 1]
        ]
        test_hc = self._hc[
            self._hc_bounds[fold_idx] : self._hc_bounds[fold_idx + 1]
        ]
        test = test_mdd + test_hc

        train_mdd = (
            self._mdd[: self._mdd_bounds[fold_idx]]
            + self._mdd[self._mdd_bounds[fold_idx + 1] :]
        )
        train_hc = (
            self._hc[: self._hc_bounds[fold_idx]]
            + self._hc[self._hc_bounds[fold_idx + 1] :]
        )
        train = train_mdd + train_hc

        return train, test

    def get_all_folds(self):
        """Return a list of ``(train_files, test_files)`` for all folds."""
        return [self.get_fold(i) for i in range(self.n_folds)]
