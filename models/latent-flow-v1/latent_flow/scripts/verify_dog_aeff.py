"""Verify a trained DoG EDRA checkpoint: negative-edge recovery + DoG params.

Run after Step 6b training completes:
    python scripts/verify_dog_aeff.py checkpoints/edra/fold0_oc_latent_flow_flow_best_graph_dog.pt
Compares against the Ricker (mexhat) EDRA baseline on disk.
"""
import os
import sys
import torch
from latent_flow.models.oc_latent_flow import LearnableMexHatMixer


def load_mixer_aeff(path, kernel, C):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    s = sd.get("state_dict", sd)
    p = "flow.graph_mixer."
    lat = s[p + "in_proj.weight"].shape[1] // 2
    hid = s[p + "in_proj.weight"].shape[0]
    m = LearnableMexHatMixer(C, lat, hid, kernel=kernel)
    m.load_state_dict({k.replace(p, ""): v for k, v in s.items() if k.startswith(p)},
                      strict=False)
    return m, m.effective_adjacency()


def stats(name, A, dp=None):
    C = A.shape[0]
    off = ~torch.eye(C, dtype=torch.bool)
    Aoff = A[off]
    neg = Aoff[Aoff < 0]
    print(f"\n[{name}]")
    print(f"  A_eff range = [{A.min():.4f}, {A.max():.4f}]")
    print(f"  neg edges  = {100*(Aoff < 0).float().mean():.1f}%  "
          f"(count {int((Aoff<0).sum())}/{Aoff.numel()})")
    if neg.numel():
        print(f"  neg mean|val| = {neg.abs().mean():.4f}   neg |sum| = {-neg.sum():.2f}")
    if dp:
        print(f"  dog params: sigma_e={dp['sigma_e']:.4f} sigma_i={dp['sigma_i']:.4f} "
              f"Ae={dp['Ae']:.4f} Ai={dp['Ai']:.4f}")
        ok = dp["sigma_i"] > dp["sigma_e"] and dp["Ae"] > dp["Ai"]
        print(f"  constraints hold (sigma_i>sigma_e, Ae>Ai): {ok}")


if __name__ == "__main__":
    # Resolve checkpoint dir relative to this file (the package root) so
    # the script works whether run from the repo root or the package dir.
    _PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _CKPT = os.path.join(_PKG, "checkpoints", "edra")
    dog_path = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(_CKPT, "fold0_oc_latent_flow_flow_best_graph_dog.pt")
    C = 62  # EDRA
    print(f"Verifying DoG checkpoint: {dog_path}")
    m_dog, A_dog = load_mixer_aeff(dog_path, "dog", C)
    stats("DoG EDRA (trained)", A_dog, m_dog.dog_params())

    # Compare against Ricker EDRA baseline if present.
    ricker_path = os.path.join(_CKPT, "fold0_oc_latent_flow_flow_best_graph_mexhat.pt")
    try:
        m_r, A_r = load_mixer_aeff(ricker_path, "ricker", C)
        stats("Ricker EDRA (baseline)", A_r)
    except FileNotFoundError:
        print(f"\n(Ricker baseline {ricker_path} not found, skipping comparison)")

    # Acceptance: DoG must have a NON-ZERO negative-edge fraction.
    off = ~torch.eye(C, dtype=torch.bool)
    neg_frac = float((A_dog[off] < 0).float().mean())
    print(f"\n>>> ACCEPTANCE: DoG neg-edge fraction = {neg_frac*100:.1f}% "
          f"({'PASS' if neg_frac > 0.0 else 'FAIL'}: target >> 0%)")
