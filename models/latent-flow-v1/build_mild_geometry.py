#!/usr/bin/env python3
"""build_mild_geometry.py — 生成 Mild Depression 的 128 导联 geometry 工件（变体层）.

任务 10-21-latent-flow-mild-train（S2，D1 = 数据根单文件豁免案）。

坐标基座：``mne.channels.make_standard_montage('GSN-HydroCel-128')``
（128 点，E1..E128 规范序）。数据源无通道名字段（上游处理报告 §3：
通道顺序与源数据完全一致、未重排），故本构建**强制假设**数据通道 i ↔
E(i+1)（D4 用户签认 2026-10-21）；若实际通道序非规范序，mexhat 图先验
错排（dense 流不受影响，--dense-fallback 兜底）。

模板一致性校验（防代码 bug，**非数据侧证据**）：与
``/root/autodl-tmp/modma/geometry.pt``（GSN-HydroCel-129 模板前 128 点
重建）逐位坐标比对——GSN-128 ≡ GSN-129 前 128，恒等属预期；该校验对
「数据通道 i ↔ E(i+1)」零证明力，仅能捕获本脚本/MNE 侧的构建错误。

保存格式与 ``latent_flow.models.graph_utils.load_geometry`` 消费契约一致
（复刻 build_edra_fmt_geometry.py 的保存契约）：
``{"coords": float32 Tensor(128,3), "channel_names": list[str]}``，默认落
数据根 ``geometry_mild.pt``（D1 豁免：单文件新增、零覆盖、已存在默认拒写
除非 --force）；``--out`` 可改落点。

Run:  python3 /root/npi/models/latent-flow-v1/build_mild_geometry.py
"""
from __future__ import annotations

import os
import sys

# --- BLAS 守卫：必须在任何 numpy/scipy/torch 导入之前（npi 环境约束） ---
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse
import warnings
from pathlib import Path

import numpy as np
import torch

VARIANT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(VARIANT_DIR))  # 使 latent_flow 可导入

DATA_ROOT_DEFAULT = Path("/root/autodl-tmp/Mild Depression")
OUT_DEFAULT = DATA_ROOT_DEFAULT / "geometry_mild.pt"
MODMA_GEOMETRY = Path("/root/autodl-tmp/modma/geometry.pt")
N_CHANNELS = 128


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=str, default=str(OUT_DEFAULT),
                   help=f"落点（默认 {OUT_DEFAULT}，D1 数据根豁免案）")
    p.add_argument("--force", action="store_true",
                   help="已存在时允许覆盖（默认拒覆盖）")
    p.add_argument("--compare-to", type=str, default=str(MODMA_GEOMETRY),
                   help="恒等比对参照（GSN-129 模板前 128 重建的 MODMA geometry）")
    return p.parse_args(argv)


def main() -> int:
    args = parse_args()
    out_path = Path(args.out)
    if out_path.exists() and not args.force:
        raise SystemExit(f"[refuse] {out_path} already exists (zero-overwrite "
                         f"discipline). Rerun with --force to overwrite.")

    # --- 基座：MNE GSN-HydroCel-128 标准模板 ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # mne 读取噪音
        import mne
        montage = mne.channels.make_standard_montage("GSN-HydroCel-128")
    names = list(montage.ch_names)
    assert len(names) == N_CHANNELS, f"montage n != {N_CHANNELS}: {len(names)}"
    expected = [f"E{i}" for i in range(1, N_CHANNELS + 1)]
    assert names == expected, "GSN-HydroCel-128 names not canonical E1..E128 order"
    ch_pos = montage.get_positions()["ch_pos"]
    coords = np.array([ch_pos[n] for n in names], dtype=np.float32)
    assert coords.shape == (N_CHANNELS, 3), f"coords shape {coords.shape}"
    assert np.isfinite(coords).all(), "coords contain non-finite values"
    radii = np.linalg.norm(coords.astype(np.float64), axis=1)
    print(f"[basis] MNE GSN-HydroCel-128: n={len(names)} order E1..E128; "
          f"radius median={np.median(radii):.4f} m "
          f"(uniform r=0.095: {np.allclose(radii, 0.095, atol=1e-3)})")

    # --- 恒等校验（防代码 bug，非数据侧证据） ---
    ref_path = Path(args.compare_to)
    if ref_path.exists():
        ref = torch.load(str(ref_path), weights_only=False)
        ref_coords = ref["coords"].float().numpy()
        ref_names = list(ref["channel_names"])
        assert ref_coords.shape == (N_CHANNELS, 3), \
            f"reference geometry shape {ref_coords.shape} != (128, 3)"
        names_ok = (ref_names == names)
        max_diff = float(np.abs(coords - ref_coords).max())
        exact = bool(np.array_equal(coords, ref_coords))
        print(f"[cross-check] vs {ref_path} (GSN-129 模板前 128 重建): "
              f"names match={names_ok}; max |diff| = {max_diff:.2e} m; "
              f"bitwise identical={exact}")
        print("[cross-check] NOTE: identity check guards against build-code "
              "bugs ONLY (GSN-128 ≡ GSN-129[:128] by construction); it is "
              "NOT evidence about the data channel order — data files carry "
              "no channel names (E-order assumption signed off as D4).")
        assert names_ok, "reference geometry channel names != E1..E128"
        assert exact, f"reference geometry coords differ (max |diff| = {max_diff})"
    else:
        print(f"[warn] reference geometry not found, cross-check skipped: {ref_path}")

    # --- 保存（契约与 build_edra_fmt_geometry.py 一致） ---
    payload = {
        "coords": torch.tensor(coords, dtype=torch.float32),
        "channel_names": list(names),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(out_path))
    sha = __import__("hashlib").sha256(out_path.read_bytes()).hexdigest()
    print(f"\nWrote {out_path}  coords={tuple(payload['coords'].shape)} "
          f"names={len(payload['channel_names'])} sha256={sha}")

    # --- 落盘后立即经 load_geometry 消费路径回读验证 ---
    from latent_flow.models.graph_utils import load_geometry
    c2, n2 = load_geometry(path=str(out_path))
    assert c2.shape == (N_CHANNELS, 3), f"round-trip shape {c2.shape}"
    assert n2 == expected and len(n2) == N_CHANNELS, "round-trip names mismatch"
    assert torch.equal(c2, torch.tensor(coords, dtype=torch.float32)), \
        "round-trip coords mismatch"
    print("load_geometry round-trip OK "
          f"(shape {tuple(c2.shape)}, dtype {c2.dtype})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
