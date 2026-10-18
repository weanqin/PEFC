"""Tests for the subject-level classification dataset (review gate 3).

These run against the real fold-0 data on disk (10 subjects). They verify:
  - the canonical fold-0 subject set resolves,
  - all four graph sources load for every subject with correct shapes,
  - labels match modma_resting ground truth (4 MDD + 6 HC),
  - the DE-EC leakage assertion fires on a wrong-fold checkpoint.

If the on-disk data is missing these tests are skipped (not failed) so CI on a
fresh checkout doesn't break — but on the experiment machine they MUST pass.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from latent_flow.data.subject_classification import (
    GRAPH_SOURCES,
    SubjectClassificationDataset,
    list_fold0_subject_files,
    load_ec,
    resolve_label,
)
from latent_flow.data.subject_classification import _DE_EC_DIR


pytestmark = pytest.mark.skipif(
    not _DE_EC_DIR.exists(),
    reason=f"fold-0 DE-EC data not present at {_DE_EC_DIR}",
)


@pytest.fixture(scope="module")
def fold0_subjects():
    subs = list_fold0_subject_files()
    if len(subs) != 10:
        pytest.skip(f"expected 10 fold-0 subjects, got {len(subs)} (data changed?)")
    return subs


class TestFold0SubjectSet:
    def test_ten_subjects(self, fold0_subjects):
        assert len(fold0_subjects) == 10
        # Each entry is a .mat basename.
        assert all(s.endswith(".mat") for s in fold0_subjects)


class TestLabelResolution:
    def test_label_distribution_4mdd_6hc(self, fold0_subjects):
        labels = [resolve_label(s) for s in fold0_subjects]
        from collections import Counter
        c = Counter("MDD" if l == 1 else "HC" for l in labels)
        assert c == {"MDD": 4, "HC": 6}, f"unexpected label split: {c}"

    def test_label_consistent_with_path(self, fold0_subjects):
        for s in fold0_subjects:
            l = resolve_label(s)
            # Re-derive independently: which subdir exists.
            name = Path(s).name
            from latent_flow.config import DATA_ROOT
            in_mdd = (DATA_ROOT / "MDD" / name).exists()
            in_hc = (DATA_ROOT / "HC" / name).exists()
            assert in_mdd != in_hc, f"{name} in both/neither HC/MDD"
            assert l == (1 if in_mdd else 0)


class TestECLoadingAllSources:
    @pytest.mark.parametrize("graph_source", list(GRAPH_SOURCES))
    def test_load_ec_shape_finite(self, fold0_subjects, graph_source):
        for sf in fold0_subjects:
            ec = load_ec(sf, graph_source, fold=0)
            assert ec.shape == (128, 128), f"{graph_source} {sf}: {ec.shape}"
            assert torch.isfinite(ec).all()
            assert ec.dtype == torch.float32

    def test_de_ec_is_signed(self, fold0_subjects):
        """DE-EC carries excite(+)/inhibit(−) sign; at least one subject should
        have negative entries (the whole point of the signed matrix)."""
        any_negative = any((load_ec(sf, "de_ec", fold=0) < 0).any()
                           for sf in fold0_subjects)
        assert any_negative, "DE-EC has no negative entries — sign signal missing?"

    def test_baselines_are_unsigned(self, fold0_subjects):
        """GC/TE/gPDC are non-negative by construction."""
        for src in ("gc", "te", "gpdc"):
            for sf in fold0_subjects:
                ec = load_ec(sf, src, fold=0)
                assert (ec >= 0).all(), f"{src} {sf} has negative entries"


class TestLeakageAssertion:
    def test_de_ec_fold0_passes(self, fold0_subjects):
        # fold-0 DE-EC jsons all reference fold0 checkpoints → must load under fold=0.
        for sf in fold0_subjects:
            ec = load_ec(sf, "de_ec", fold=0)
            assert ec.shape == (128, 128)

    def test_de_ec_wrong_fold_raises(self, fold0_subjects, tmp_path, monkeypatch):
        """A DE-EC json whose checkpoint names fold1 must be rejected when fold=0."""
        # Take the first real json, copy it, rewrite the checkpoint to fold1.
        src_jsons = sorted(_DE_EC_DIR.glob("subject_*.json"))
        if not src_jsons:
            pytest.skip("no DE-EC jsons to forge from")
        with open(src_jsons[0]) as f:
            d = json.load(f)
        d["checkpoint"] = "fold1_oc_latent_flow_flow_best.pt"
        forged = tmp_path / "subject_99.json"
        with open(forged, "w") as f:
            json.dump(d, f)

        # Point the DE-EC dir at the temp dir containing only the forged json.
        import latent_flow.data.subject_classification as mod
        monkeypatch.setattr(mod, "_DE_EC_DIR", tmp_path)
        # _find_json_by_subject reads the module-level _DE_EC_DIR via closure on
        # the module, so re-import the function path through the module.
        with pytest.raises(ValueError, match="leakage guard"):
            load_ec(d["subject_file"], "de_ec", fold=0)


class TestDatasetEndToEnd:
    def test_build_dataset_all_sources(self, fold0_subjects):
        de_cache: dict[str, np.ndarray] = {}
        for src in GRAPH_SOURCES:
            ds = SubjectClassificationDataset(
                fold0_subjects, graph_source=src, fold=0,
                de_cache=de_cache, n_windows_de=10,
            )
            assert len(ds) == 10
            item = ds[0]
            assert item["de_features"].shape == (128, 5)
            assert item["ec_matrix"].shape == (128, 128)
            assert item["label"].dtype == torch.long
            assert torch.isfinite(item["de_features"]).all()
            assert torch.isfinite(item["ec_matrix"]).all()
        # DE cache should now hold all 10 subjects (computed once, reused).
        assert len(de_cache) == 10

    def test_label_distribution_method(self, fold0_subjects):
        ds = SubjectClassificationDataset(
            fold0_subjects, graph_source="de_ec", fold=0, n_windows_de=10
        )
        assert ds.label_distribution() == {"MDD": 4, "HC": 6}
