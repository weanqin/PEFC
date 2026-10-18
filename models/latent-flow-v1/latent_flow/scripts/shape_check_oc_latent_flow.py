"""Forward-shape audit for OC-LatentFlow (default config).

Hooks every conv / conv-transpose / linear inside the encoder, proj, flow,
and decoder and prints the real tensor shape after each op, for one batch
with the production config (B=2, C=128, obs_len=375, fc_len=125).

Run from repo root:

    D:/program/anaconda/envs/npi/python.exe -u -m \
        latent_flow.scripts.shape_check_oc_latent_flow

Stage-1 (encoder+proj+decoder) and stage-2 (flow) are both exercised; the
inference path (sample, flow Euler n_steps=10) is also shown end-to-end.
"""

import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import torch
import torch.nn as nn

from latent_flow.models.oc_latent_flow import OCLatentFlow


SEP = "-" * 78
LABEL_PAD = 46


def _shape(x):
    """tuple shape, or '<non-tensor>' if not a tensor."""
    if torch.is_tensor(x):
        return tuple(x.shape)
    return "<non-tensor>"


def hook_submodule(name, mod):
    """Register forward hooks on every Conv1d/ConvTranspose1d/Linear under mod.

    Prints the output shape of each, prefixed by the submodule path. Hook
    output for the *innermost* named module is used (avoids double-printing
    Sequentials that only forward).
    """
    targets = (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)
    seen = {}

    def make_hook(path):
        def hook(_m, _inp, out):
            print(f"  {path:<40} {_shape(out)}")
        return hook

    for subpath, sub in mod.named_modules():
        # Only hook leaf-ish modules that are themselves a target op.
        if isinstance(sub, targets):
            full = f"{name}.{subpath}" if subpath else name
            h = sub.register_forward_hook(make_hook(full))
            seen[full] = h
    return seen


def banner(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    torch.manual_seed(0)
    model = OCLatentFlow().to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"OCLatentFlow default config | {n_params/1e6:.2f}M params")

    # Register hooks on the four submodules we care about.
    handles = {}
    handles.update(hook_submodule("encoder", model.encoder))
    handles.update(hook_submodule("proj", model.proj))
    handles.update(hook_submodule("flow", model.flow))
    handles.update(hook_submodule("decoder", model.decoder))

    B = 2
    C = model.n_channels          # 128
    obs_len = model.obs_len       # 375
    fc_len = model.forecast_len   # 125
    t_full = model.t_full         # 500

    x_obs = torch.randn(B, obs_len, C, device=device)
    x_full = torch.randn(B, t_full, C, device=device)

    # ------------------------------------------------------------------
    # Stage 1: encoder -> proj -> decoder (skips captured)
    # ------------------------------------------------------------------
    banner("STAGE 1 — encoder -> proj -> decoder (forward_stage1)")
    with torch.no_grad():
        loss1, ldict1 = model.forward_stage1(x_obs, x_full)
    print(f"\n  stage1 loss = {float(loss1):.4f}")
    print("  loss components:", {k: float(v) for k, v in ldict1.items()})

    # ------------------------------------------------------------------
    # Stage 2: flow forward (encoder/proj/decoder frozen in real training)
    # ------------------------------------------------------------------
    banner("STAGE 2 — flow velocity net (forward_stage2)")
    with torch.no_grad():
        loss2, ldict2 = model.forward_stage2(x_obs, x_full)
    print(f"\n  stage2 loss = {float(loss2):.4f}")
    print("  loss components:", {k: float(v) for k, v in ldict2.items()})

    # ------------------------------------------------------------------
    # Inference: sample (flow Euler, n_steps=10)
    # ------------------------------------------------------------------
    banner("INFERENCE — sample(n_steps=10, use_proj=False)")
    with torch.no_grad():
        x_hat = model.sample(x_obs, n_steps=10, use_proj=False)
    print(f"\n  sample out  {_shape(x_hat)}  (obs+fc full window)")
    print(f"  fc slice    {_shape(x_hat[:, obs_len:, :])}")

    # ------------------------------------------------------------------
    # Inference via proj (stage-1 bootstrap, before flow training)
    # ------------------------------------------------------------------
    banner("INFERENCE — sample(use_proj=True)  [stage-1 bootstrap]")
    with torch.no_grad():
        x_hat_proj = model.sample(x_obs, n_steps=10, use_proj=True)
    print(f"\n  sample out  {_shape(x_hat_proj)}")

    for h in handles.values():
        h.remove()

    print("\nDone.")


if __name__ == "__main__":
    main()
