"""Unit tests for OCLatentFlow (task 06-23-new-latent-flow-arch).

Verifies the shapes and forward/backward contracts of the two-stage
obs-conditioned latent flow:

1. ConvEncoder maps (B, T, C) -> (mu, logvar) of shape (B, C, T_lat, D), and
   accepts both obs (375) and fc (125) lengths.
2. FlowVelocityNet maps (z_t, t, z_obs) -> v of the same shape as z_obs.
3. ConvDecoder maps (z_obs, z_fc) -> (B, T_full, C).
4. forward_stage1 returns a scalar loss + a dict, and the loss is
   differentiable (backward populates encoder/proj/decoder grads).
5. forward_stage2 returns a scalar loss + dict; with encoder/proj/decoder
   frozen, grads reach the flow only.
6. sample(x_obs) returns (B, T_full, C) without NaN and the fc slice is
   non-degenerate.

Run:

    D:\\program\\anaconda\\envs\\npi\\python.exe -m pytest \\
        tests/test_oc_latent_flow.py -q

Per logging-guidelines.md, all output is ASCII (Windows GBK safe).
"""

from __future__ import annotations

import torch

from latent_flow.models.oc_latent_flow import (
    ConvDecoder,
    ConvEncoder,
    FlowVelocityNet,
    OCLatentFlow,
    ProjHead,
)


# ---------------------------------------------------------------------------
# Shared dims (small for fast CPU tests)
# ---------------------------------------------------------------------------

B, C, T_LAT, D = 2, 8, 16, 16          # small C/D for speed; real = 128/64
OBS_LEN, FC_LEN, T_FULL = 375, 125, 500


def _make_model():
    return OCLatentFlow(
        n_channels=C, latent_dim=D, t_lat=T_LAT,
        obs_len=OBS_LEN, forecast_len=FC_LEN,
        enc_base_ch=16, enc_levels=4,
        dec_base_ch=16, dec_levels=4,
        flow_hidden_ch=32,
    )


# ---------------------------------------------------------------------------
# Component shape tests
# ---------------------------------------------------------------------------

def test_encoder_shapes_obs_and_fc():
    enc = ConvEncoder(n_channels=C, latent_dim=D, t_lat=T_LAT,
                      base_ch=16, n_levels=4)
    for T_in in (OBS_LEN, FC_LEN):
        x = torch.randn(B, T_in, C)
        mu, logvar = enc(x)
        assert mu.shape == (B, C, T_LAT, D), f"obs T={T_in}: {mu.shape}"
        assert logvar.shape == (B, C, T_LAT, D)
        assert not torch.isnan(mu).any() and not torch.isnan(logvar).any()


def test_proj_head_shape_and_init_identity():
    proj = ProjHead(latent_dim=D)
    z_obs = torch.randn(B, C, T_LAT, D)
    z_fc = proj(z_obs)
    assert z_fc.shape == z_obs.shape
    # Zero-init weight+bias -> proj outputs zeros at init (sane decoder start)
    assert torch.allclose(z_fc, torch.zeros_like(z_fc), atol=1e-6)


def test_flow_velocity_net_shape():
    flow = FlowVelocityNet(n_channels=C, latent_dim=D, t_lat=T_LAT, hidden_ch=32)
    z_obs = torch.randn(B, C, T_LAT, D)
    z_t = torch.randn(B, C, T_LAT, D)
    t = torch.rand(B)
    v = flow(z_t, t, z_obs)
    assert v.shape == z_obs.shape
    assert not torch.isnan(v).any()


def test_flow_velocity_net_graph_mode_shape_and_grad():
    """R2 graph-mixer flow (task 06-29). C preserved through the mixer, dilated
    convs run with C folded into batch, per-channel output projection.
    """
    # Small synthetic adjacency (C x C) so the test does not depend on geometry.pt.
    a_hat = torch.eye(C) + 0.1 * torch.ones(C, C)
    a_hat = a_hat + a_hat.t()                      # symmetric
    flow = FlowVelocityNet(n_channels=C, latent_dim=D, t_lat=T_LAT,
                           hidden_ch=32, graph_mode=True, graph_k=2,
                           a_hat=a_hat)
    z_obs = torch.randn(B, C, T_LAT, D)
    z_t = torch.randn(B, C, T_LAT, D)
    t = torch.rand(B)
    v = flow(z_t, t, z_obs)
    assert v.shape == z_obs.shape
    assert not torch.isnan(v).any()

    # graph_out is zero-init -> v ~ 0 at bootstrap (same contract as the dense
    # path). Confirm; then check grads flow once out is non-zero.
    assert v.abs().max().item() < 1e-6

    with torch.no_grad():
        flow.graph_out.proj.weight.normal_(0, 0.1)
        flow.graph_out.proj.bias.normal_(0, 0.01)
    v = flow(z_t, t, z_obs)
    v.sum().backward()
    assert flow.graph_mixer.in_proj.weight.grad is not None
    assert flow.graph_mixer.msg_layers[0].weight.grad is not None
    assert flow.graph_mixer.in_proj.weight.grad.abs().mean() > 0
    assert flow.graph_mixer.msg_layers[0].weight.grad.abs().mean() > 0


def test_decoder_skip_is_used():
    """Skip connections must actually influence the output (not silently dropped)."""
    dec = ConvDecoder(n_channels=C, latent_dim=D, t_lat=T_LAT,
                      obs_len=OBS_LEN, forecast_len=FC_LEN,
                      base_ch=16, n_levels=4)
    enc = ConvEncoder(n_channels=C, latent_dim=D, t_lat=T_LAT,
                      base_ch=16, n_levels=4)
    x_obs = torch.randn(B, OBS_LEN, C)
    mu_obs, logvar_obs, skips = enc(x_obs, return_skips=True)
    z_obs = mu_obs
    z_fc = torch.randn(B, C, T_LAT, D)

    with torch.no_grad():
        out_with_skip = dec(z_obs, z_fc, skips=skips)
        out_no_skip = dec(z_obs, z_fc, skips=None)

    assert out_with_skip.shape == out_no_skip.shape == (B, T_FULL, C)
    # Outputs MUST differ — if equal, skips are being ignored/dropped
    assert not torch.allclose(out_with_skip, out_no_skip, atol=1e-5), \
        "skip connections are not influencing decoder output"


def test_encoder_returns_skips():
    """Encoder with return_skips=True returns skips matching n_levels."""
    enc = ConvEncoder(n_channels=C, latent_dim=D, t_lat=T_LAT,
                      base_ch=16, n_levels=4)
    x = torch.randn(B, OBS_LEN, C)
    mu, logvar, skips = enc(x, return_skips=True)
    assert len(skips) == 4  # n_levels
    for s in skips:
        assert s.dim() == 3  # (B*C, ch_i, T_down_i)
        assert s.shape[0] == B * C


def test_decoder_shape():
    dec = ConvDecoder(n_channels=C, latent_dim=D, t_lat=T_LAT,
                      obs_len=OBS_LEN, forecast_len=FC_LEN,
                      base_ch=16, n_levels=4)
    z_obs = torch.randn(B, C, T_LAT, D)
    z_fc = torch.randn(B, C, T_LAT, D)
    x_hat = dec(z_obs, z_fc)
    assert x_hat.shape == (B, T_FULL, C), f"{x_hat.shape}"
    assert not torch.isnan(x_hat).any()


# ---------------------------------------------------------------------------
# Stage-1 forward + backward
# ---------------------------------------------------------------------------

def test_stage1_forward_backward():
    model = _make_model()
    model.set_kl_beta(1e-4)
    x_obs = torch.randn(B, OBS_LEN, C)
    x_target = torch.randn(B, T_FULL, C)

    loss, losses_dict = model.forward_stage1(x_obs, x_target)
    assert loss.dim() == 0, "loss must be scalar"
    assert torch.isfinite(loss), "loss must be finite"
    for key in ("recon", "td", "bd", "corr", "cons", "kl", "beta"):
        assert key in losses_dict, f"missing loss key: {key}"

    loss.backward()
    # encoder/proj/decoder must receive gradients
    assert model.encoder.temporal_blocks[0][0].weight.grad is not None
    assert model.proj.proj.weight.grad is not None
    assert model.decoder.out_conv.weight.grad is not None
    # flow is NOT trained in stage 1 — grads should be None
    assert model.flow.input_proj.weight.grad is None


# ---------------------------------------------------------------------------
# Stage-2 forward + backward (frozen VAE, flow trains)
# ---------------------------------------------------------------------------

def test_stage2_forward_backward_flow_only():
    model = _make_model()
    # Freeze encoder/proj/decoder as stage 2 would
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    for p in model.proj.parameters():
        p.requires_grad_(False)
    for p in model.decoder.parameters():
        p.requires_grad_(False)

    x_obs = torch.randn(B, OBS_LEN, C)
    x_target = torch.randn(B, T_FULL, C)

    loss, losses_dict = model.forward_stage2(x_obs, x_target)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    for key in ("recon2", "corr2", "cons2"):
        assert key in losses_dict
    # No flow/proj distribution loss in stage 2 (v_mse was removed); the flow
    # is optimized only by gradients backpropagated through the frozen decoder.
    assert "v_mse" not in losses_dict

    loss.backward()
    # flow MUST receive gradients (it's the only trainable in stage 2)
    assert model.flow.input_proj.weight.grad is not None
    # frozen VAE params must NOT be updated (requires_grad=False)
    assert not model.encoder.temporal_blocks[0][0].weight.requires_grad
    assert not model.decoder.out_conv.weight.requires_grad


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def test_sample_shape_and_no_nan():
    model = _make_model()
    model.eval()
    x_obs = torch.randn(B, OBS_LEN, C)
    x_hat = model.sample(x_obs, n_steps=10)
    assert x_hat.shape == (B, T_FULL, C)
    assert not torch.isnan(x_hat).any()
    # fc slice should not be all-zero (sanity: decoder produced something)
    x_fc = x_hat[:, OBS_LEN:, :]
    assert x_fc.abs().mean() > 0


if __name__ == "__main__":
    # Allow running as a script
    import pytest, sys
    sys.exit(pytest.main([__file__, "-v"]))
