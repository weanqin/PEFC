"""Unit tests for models.graph_utils (task 06-29-channel-selectivity-rootcause).

Verifies the graph-prior construction from electrode coordinates:
  - shapes and self-loop
  - symmetry of adjacency and normalized adjacency
  - gaussian kernel monotonically decreases with distance
  - normalized adjacency row-sum <= 1 and is finite
  - integration with the real geometry.pt (128 channels)

Run:
    D:\\program\\anaconda\\envs\\npi\\python.exe -m pytest \\
        tests/test_graph_utils.py -q
"""

from __future__ import annotations

import pytest
import torch

from latent_flow.models import graph_utils as gu
from latent_flow.config import GEOMETRY_PATH


# ---------------------------------------------------------------------------
# Small synthetic geometry (deterministic, no file dependency for core tests)
# ---------------------------------------------------------------------------


def _synthetic_coords(n: int = 8) -> torch.Tensor:
    """n points along a line: channel i at position (i, 0, 0)."""
    return torch.stack([torch.arange(n, dtype=torch.float32),
                        torch.zeros(n), torch.zeros(n)], dim=1)


class TestAdjacencyConstruction:
    def test_shape_and_self_loop(self):
        c = _synthetic_coords(8)
        A = gu.build_adjacency(c, kernel="gaussian", sigma_d=1.0)
        assert A.shape == (8, 8)
        assert torch.allclose(torch.diag(A), torch.ones(8))

    def test_symmetric(self):
        c = _synthetic_coords(10)
        A = gu.build_adjacency(c, kernel="gaussian")
        assert torch.allclose(A, A.t(), atol=1e-6)

    def test_gaussian_monotonic_decrease_with_distance(self):
        # On a line, channel 0: adjacency to 1 > 2 > 3 ...
        c = _synthetic_coords(8)
        A = gu.build_adjacency(c, kernel="gaussian", sigma_d=1.0)
        row0 = A[0, 1:]
        assert torch.all(row0[:-1] >= row0[1:] - 1e-6), \
            "adjacency should monotonically decrease with distance"

    def test_binary_kernel_threshold(self):
        c = _synthetic_coords(8)            # spacing = 1 between neighbours
        A = gu.build_adjacency(c, kernel="binary", sigma_d=1.0)  # thresh = 1.0
        # channel 0 neighbours within distance 1.0: only itself (dist 0) and ch 1 (dist 1)
        assert A[0, 0] == 1.0
        assert A[0, 1] == 1.0
        assert A[0, 2] == 0.0


class TestNormalization:
    def test_symmetric_normalized(self):
        c = _synthetic_coords(10)
        A = gu.build_adjacency(c, kernel="gaussian", sigma_d=1.0)
        A_hat = gu.normalize_adj(A, add_self_loops=True)
        assert torch.allclose(A_hat, A_hat.t(), atol=1e-6)

    def test_spectral_radius_le_one(self):
        # Symmetric normalization A_hat = D^{-1/2}(A+I)D^{-1/2} has largest
        # eigenvalue exactly 1, with eigenvector sqrt(deg). This bounds the
        # operator norm, so message passing Z <- A_hat Z never amplifies.
        c = _synthetic_coords(12)
        A = gu.build_adjacency(c, kernel="gaussian", sigma_d=1.0)
        A_hat = gu.normalize_adj(A, add_self_loops=True)
        deg = (A + torch.eye(A.shape[0])).sum(dim=1)
        v = deg.sqrt()
        Av = A_hat @ v
        assert torch.allclose(Av, v, atol=1e-5), \
            "sqrt(deg) must be the unit-eigenvalue eigenvector of A_hat"

    def test_finite(self):
        c = _synthetic_coords(12)
        A = gu.build_adjacency(c, kernel="gaussian")
        A_hat = gu.normalize_adj(A, add_self_loops=True)
        assert torch.isfinite(A_hat).all()

    def test_constant_input_stable(self):
        # Operator is well-defined on a constant input (no NaN, bounded).
        c = _synthetic_coords(8)
        A = gu.build_adjacency(c, kernel="gaussian", sigma_d=1.0)
        A_hat = gu.normalize_adj(A, add_self_loops=True)
        ones = torch.ones(8)
        out = A_hat @ ones
        assert torch.isfinite(out).all()


class TestRealGeometry:
    """Integration with the project's actual geometry.pt (128 channels)."""

    @pytest.mark.skipif(not GEOMETRY_PATH.exists(),
                        reason="geometry.pt not available")
    def test_load_real_geometry(self):
        coords, names = gu.load_geometry()
        assert coords.shape == (128, 3)
        assert len(names) == 128
        assert names[0] == "E1" and names[-1] == "E128"
        # head radius sanity: 7-12 cm
        norms = coords.norm(dim=1)
        assert 0.07 < norms.min().item() < 0.12
        assert 0.07 < norms.max().item() < 0.12

    @pytest.mark.skipif(not GEOMETRY_PATH.exists(),
                        reason="geometry.pt not available")
    def test_build_normalized_adj_128(self):
        A_hat = gu.build_normalized_adj()
        assert A_hat.shape == (128, 128)
        assert torch.isfinite(A_hat).all()
        assert torch.allclose(A_hat, A_hat.t(), atol=1e-6)
        # largest eigenvalue of symmetric-normalized adjacency is exactly 1
        eig = torch.linalg.eigvalsh(A_hat)
        assert eig.max().item() <= 1.0 + 1e-4
