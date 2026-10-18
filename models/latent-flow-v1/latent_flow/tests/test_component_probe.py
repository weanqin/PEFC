"""Unit tests for ComponentProbeSite (task 06-29-channel-selectivity-rootcause).

Verifies the forward-surgery root-cause probe:
  - surgery="none" is byte-identical to the parent LatentProbeSite (regression)
  - input_proj_blockdiag / out_proj_blockdiag actually mutate the clone's weights
    into a block-diagonal-over-channels structure (verified on the weight matrix)
  - hidden_truncate installs a hook that zeroes hidden units [k:]
  - the caller's trained model is NEVER mutated (deepcopy isolation)
  - block-diagonal surgery makes each hidden unit depend on exactly one channel

Run:
    D:\\program\\anaconda\\envs\\npi\\python.exe -m pytest \\
        tests/test_component_probe.py -q
"""

from __future__ import annotations

import copy

import pytest
import torch

from latent_flow.attribution.bisect_sigma import (
    ComponentProbeSite,
    LatentProbeSite,
)
from latent_flow.models.oc_latent_flow import (
    FlowVelocityNet,
    OCLatentFlow,
)


# Small dims for speed (real = C128/D64)
C, D, T_LAT, HIDDEN = 8, 4, 4, 16
N_STEPS = 3


def _make_flow() -> FlowVelocityNet:
    torch.manual_seed(0)
    flow = FlowVelocityNet(n_channels=C, latent_dim=D, t_lat=T_LAT,
                           hidden_ch=HIDDEN, dilations=(1, 2))
    # __init__ zeros out_proj (bootstrap). A *trained* flow has non-zero
    # out_proj, so inject random weights to represent the trained state — the
    # surgery operates on trained weights, not the zero-init bootstrap.
    with torch.no_grad():
        flow.out_proj.weight.copy_(torch.randn_like(flow.out_proj.weight) * 0.1)
        flow.out_proj.bias.copy_(torch.randn_like(flow.out_proj.bias) * 0.01)
    return flow


def _make_subject(n: int = 3) -> torch.Tensor:
    """A fake z_obs-shaped tensor is NOT what LatentProbeSite wants — it
    needs x_obs so it can run the encoder. But to test the surgery in isolation
    we use a tiny wrapper model exposing .encoder and .flow."""
    g = torch.Generator().manual_seed(0)
    return torch.randn(n, 375, C, generator=g)


class _Wrapper(torch.nn.Module):
    """Minimal model exposing .encoder and .flow so LatentProbeSite works.

    The encoder here is a trivial per-channel linear map (B,T,C)->(B,C,T_lat,D);
    it does NOT need to be the real ConvEncoder — the probe only calls
    model.encoder(x_obs, return_skips=True) and reads mu."""
    def __init__(self, flow):
        super().__init__()
        self.flow = flow
        self.encoder = _TinyEncoder(C, D, T_LAT)
        # LatentProbeSite also reads model.decoder? No — it bypasses decoder.
        # It calls model.flow(z,t,z_cond). Good.

    def forward(self, *a, **k):  # pragma: no cover
        raise RuntimeError("not used")


class _TinyEncoder(torch.nn.Module):
    """Linear (B,T,C) -> (mu, logvar, skips) with mu shape (B,C,T_lat,D).

    The probe only needs mu of shape (B, C, T_lat, D); we mean-pool over T,
    broadcast to T_lat, and linearly lift the scalar to D features per channel.
    """
    def __init__(self, C, D, T_lat):
        super().__init__()
        self.lift = torch.nn.Linear(1, D)
        self.C, self.D, self.T_lat = C, D, T_lat

    def forward(self, x, return_skips=False):
        x = x.float()                              # (B, T, C)
        pooled = x.mean(dim=1)                     # (B, C)
        # (B, C) -> (B, C, T_lat, 1) -> lift -> (B, C, T_lat, D)
        mu = pooled.unsqueeze(2).expand(-1, -1, self.T_lat).unsqueeze(-1)
        mu = self.lift(mu)
        logvar = torch.zeros_like(mu)
        return mu, logvar, ([],)


class TestSurgeryMutation:
    """The surgery must actually alter the clone's weights into the claimed form."""

    def test_input_proj_blockdiag_each_unit_one_channel(self):
        flow = _make_flow()
        ComponentProbeSite._blockdiag_input_proj(flow, C, D)
        w = flow.input_proj.weight.data            # (HIDDEN, C*D*3)
        block_w = D * 3
        # Each hidden unit's non-zero entries must all lie in ONE channel block.
        for h in range(HIDDEN):
            nonzero_blocks = 0
            for c in range(C):
                blk = w[h, c * block_w:(c + 1) * block_w]
                if blk.abs().sum().item() > 0:
                    nonzero_blocks += 1
            assert nonzero_blocks == 1, \
                f"hidden unit {h} depends on {nonzero_blocks} channels, expected 1"

    def test_out_proj_blockdiag_each_unit_one_channel(self):
        flow = _make_flow()
        ComponentProbeSite._blockdiag_out_proj(flow, C, D)
        w = flow.out_proj.weight.data             # (C*D, HIDDEN)
        block_w = D
        for h in range(HIDDEN):
            nonzero_blocks = 0
            for c in range(C):
                blk = w[c * block_w:(c + 1) * block_w, h]
                if blk.abs().sum().item() > 0:
                    nonzero_blocks += 1
            assert nonzero_blocks == 1, \
                f"hidden unit {h} feeds {nonzero_blocks} output channels, expected 1"

    def test_hidden_truncate_hook_zeros_tail(self):
        flow = _make_flow()
        k = 4
        ComponentProbeSite._truncate_hidden(flow, k)
        assert hasattr(flow, "_surgery_hook")
        # Run a forward through input_proj and check the hook fires.
        x = torch.randn(2, T_LAT, C * D * 3)
        out = flow.input_proj(x)                  # hook mutates output in place
        assert out[..., k:].abs().sum().item() == 0.0
        assert out[..., :k].abs().sum().item() > 0.0


class TestDeepcopyIsolation:
    """The caller's trained model must NEVER be mutated.

    The probe deep-copies model.flow and reassigns model.flow to the clone, so
    to check the ORIGINAL trained flow we must capture its weights BEFORE
    constructing the probe and compare AFTER (the clone replaces model.flow).
    """

    def test_trained_weights_untouched_after_probe(self):
        flow = _make_flow()
        original_input_w = flow.input_proj.weight.data.clone()
        model = _Wrapper(flow)
        x = _make_subject()
        ComponentProbeSite(model, x, n_steps=N_STEPS, seed=0,
                           surgery="input_proj_blockdiag",
                           n_channels=C, latent_dim=D)
        # `flow` is the original trained object (captured above); its weights
        # must be unchanged even though model.flow now points at the clone.
        assert torch.equal(flow.input_proj.weight.data, original_input_w), \
            "trained input_proj was mutated by the probe (deepcopy leak)"


class TestNoneSurgeryRegression:
    """surgery='none' must produce identical results to LatentProbeSite.

    Both probes must use the SAME encoder + SAME flow weights, so the test
    shares one encoder (deterministic, seeded) and deep-copies only the flow.
    """

    def test_none_matches_parent_baseline(self):
        torch.manual_seed(123)
        enc = _TinyEncoder(C, D, T_LAT)            # seeded, fixed weights
        flow_a = _make_flow()
        x = _make_subject()

        # Two wrappers sharing the SAME encoder; flow_a vs its deep copy.
        ma = _Wrapper(copy.deepcopy(flow_a)); ma.encoder = enc
        mb = _Wrapper(copy.deepcopy(flow_a)); mb.encoder = enc

        site_parent = LatentProbeSite(ma, x, n_steps=N_STEPS, seed=0,
                                      condition_mode="coupled")
        site_none = ComponentProbeSite(mb, x, n_steps=N_STEPS, seed=0,
                                       surgery="none", n_channels=C,
                                       latent_dim=D, condition_mode="coupled")
        assert torch.allclose(site_parent._z_fc_base, site_none._z_fc_base,
                              atol=1e-6)
        site_parent.set_source(1)
        site_none.set_source(1)
        r_parent = site_parent.perturb_and_measure(0.2, 1)
        r_none = site_none.perturb_and_measure(0.2, 1)
        assert torch.allclose(r_parent, r_none, atol=1e-6)

    def test_invalid_surgery_raises(self):
        flow = _make_flow()
        model = _Wrapper(flow)
        x = _make_subject()
        with pytest.raises(ValueError):
            ComponentProbeSite(model, x, n_steps=N_STEPS, seed=0,
                               surgery="bogus", n_channels=C, latent_dim=D)

    def test_surgery_desc(self):
        flow = _make_flow()
        model = _Wrapper(flow)
        x = _make_subject()
        s = ComponentProbeSite(model, x, n_steps=N_STEPS, seed=0,
                               surgery="hidden_truncate", surgery_param=8,
                               n_channels=C, latent_dim=D)
        assert s.surgery_desc == "hidden_truncate(k=8)"
