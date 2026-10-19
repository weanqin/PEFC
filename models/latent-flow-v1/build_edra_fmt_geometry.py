#!/usr/bin/env python3
"""build_edra_fmt_geometry.py — 生成 edra_fmt 的 62 导联 geometry 工件（包装层）.

任务 10-19-latent-flow-edra-fmt-train（S2，D1=mexhat-graph 前置）。

坐标基座（用户拍板 2026-10-19，S2.2 判定门升级问询）：**数据根
``chan_coords_62.npy``**（forward 模型球面理想化 montage，全部 r=0.095 m），
而非 MNE standard_1020 realistic 模板。升级背景：MNE 模板与 npy 的逐通道方向
余弦未全部 ≥0.99（16/62 低于，min 0.9766 @T7，角差最大 12.4°）——坐标系同构、
同为米制、通道序一致，差异属模板族不同（idealized sphere vs realistic head）；
功能指标（两两距离矩阵相关 0.9929 / gaussian・mexhat 邻接相关 0.9965・0.9967）。

行序对齐证明（硬断言）：``chan_forward_62.fif`` 的通道名序列 == npz
``channel_names``（全部 48 例一致），且其通道位置 == npy 数值。

保存格式与 ``latent_flow/models/graph_utils.load_geometry`` 消费契约一致：
``{"coords": float32 Tensor(62,3), "channel_names": list[str]}``，落
``geometry_edra_fmt.pt`` 到数据根（D2 豁免案：单文件新增、零覆盖）。

Run:  python3 /root/npi/models/latent-flow-v1/build_edra_fmt_geometry.py
"""
from __future__ import annotations

import os
import sys

# --- BLAS 守卫：必须在任何 numpy/scipy/torch 导入之前（npi 环境约束） ---
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "8")

from pathlib import Path

import numpy as np
import torch

VARIANT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(VARIANT_DIR))  # 使 latent_flow 可导入

from latent_flow.config import EDRA_FMT_DATA_ROOT, EDRA_FMT_GEOMETRY_PATH  # noqa: E402
from latent_flow.data.datasets import get_dataset, list_hc, list_mdd  # noqa: E402
from latent_flow.data.electrode_positions import EDRA_CHANNEL_ORDER  # noqa: E402

COS_GATE = 0.99
NPY_PATH = EDRA_FMT_DATA_ROOT / "chan_coords_62.npy"
FIF_PATH = EDRA_FMT_DATA_ROOT / "chan_forward_62.fif"


def _load_channel_names() -> list:
    """从全部 48 例 npz 读取 channel_names 并断言完全一致。"""
    desc = get_dataset("edra_fmt")
    files = list_hc(desc) + list_mdd(desc)
    names_ref = None
    for f in files:
        with np.load(f, allow_pickle=True) as z:
            names = [str(x) for x in z["channel_names"]]
        if names_ref is None:
            names_ref = names
        elif names != names_ref:
            raise SystemExit(f"channel_names mismatch across subjects: {f}")
    return names_ref


def _cosines(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    return (a * b).sum(axis=1) / (na * nb + 1e-12)


def main() -> int:
    names = _load_channel_names()
    n_files = len(list_hc(get_dataset("edra_fmt"))) + len(list_mdd(get_dataset("edra_fmt")))
    print(f"channel_names from {n_files} npz (identical): n={len(names)}")
    print(f"  order: {names}")

    # 与包内 vendored EDRA_CHANNEL_ORDER 一致性守卫（Poz/POz 拼写差异豁免：
    # fmt npz 用 'POz'，旧 .vhdr 用 'Poz'，均指同一导联）。
    norm_a = [n.replace("POz", "Poz") for n in names]
    assert norm_a == list(EDRA_CHANNEL_ORDER), (
        "fmt channel order != vendored EDRA_CHANNEL_ORDER (modulo Poz/POz)\n"
        f"  fmt: {norm_a}\n  vendored: {list(EDRA_CHANNEL_ORDER)}"
    )

    # --- 基座：chan_coords_62.npy ---
    ref = np.load(NPY_PATH).astype(np.float64)
    assert ref.shape == (62, 3), f"npy shape {ref.shape} != (62, 3)"
    assert np.isfinite(ref).all(), "npy contains non-finite values"
    radii = np.linalg.norm(ref, axis=1)
    med_d = float(np.median(__import__("scipy.spatial.distance", fromlist=["pdist"])
                            .pdist(ref)))
    print(f"[basis] {NPY_PATH}: shape={ref.shape} radius median={np.median(radii):.4f} m "
          f"(uniform r=0.095: {np.allclose(radii, 0.095, atol=1e-3)})")
    print(f"[basis] median pairwise distance: {med_d:.4f} m")
    assert 0.05 < med_d < 0.2, f"coord scale out of head range: {med_d}"

    # --- 行序对齐证明（硬断言）：fif 通道名 == npz 名，fif 位置 == npy ---
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # mne forward 读取噪音
        import mne
        fwd = mne.read_forward_solution(str(FIF_PATH))
    fif_names = [ch["ch_name"] for ch in fwd["info"]["chs"]]
    fif_locs = np.array([ch["loc"][:3] for ch in fwd["info"]["chs"]], dtype=np.float64)
    assert fif_names == names, "chan_forward_62.fif channel order != npz channel_names"
    loc_diff = float(np.abs(fif_locs - ref).max())
    # fif locs are float32, npy float64 → pure storage rounding; 1e-6 m = 1 µm
    assert loc_diff < 1e-6, f"fif locs != npy coords (max |diff| = {loc_diff})"
    print(f"[alignment] fif names == npz channel_names ({len(names)} ch); "
          f"fif locs == npy (max |diff| = {loc_diff:.2e})")

    coords = ref  # (62, 3) metres — basis is the npy itself

    # --- 登记（非门禁）：MNE standard_1020 模板差（模板族差异，用户已知悉拍板） ---
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            from latent_flow.scripts.build_edra_geometry import _coords_from_mne
            mne_coords = _coords_from_mne(names)
        cos_mne = _cosines(mne_coords, ref)
        ang = np.degrees(np.arccos(np.clip(cos_mne, -1, 1)))
        dist = np.linalg.norm(mne_coords - ref, axis=1)
        print(f"[info-only] MNE template vs npy basis: cos min={cos_mne.min():.6f} "
              f"median={np.median(cos_mne):.6f}; below {COS_GATE}: "
              f"{int((cos_mne < COS_GATE).sum())}/62; ang max={ang.max():.1f}deg; "
              f"euclid max={dist.max():.4f} m — registered template-family "
              f"difference (user decision 2026-10-19: npy basis)")
    except Exception as exc:  # informational only
        print(f"[info-only] MNE comparison skipped: {exc}")

    # --- S2.2 判定门（本基座下生成物 == npy，门为恒等校验） ---
    cos = _cosines(coords, ref)
    eu = np.linalg.norm(coords - ref, axis=1)
    print(f"[S2.2 gate] generated vs {NPY_PATH.name}: cos min={cos.min():.6f} "
          f"(gate {COS_GATE}); euclid max={eu.max():.2e} m")
    assert (cos >= COS_GATE).all(), "identity gate failed (impossible for npy basis)"

    payload = {
        "coords": torch.tensor(coords, dtype=torch.float32),
        "channel_names": list(names),
    }
    EDRA_FMT_GEOMETRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(EDRA_FMT_GEOMETRY_PATH))
    print(f"\nWrote {EDRA_FMT_GEOMETRY_PATH}  coords={tuple(payload['coords'].shape)} "
          f"names={len(payload['channel_names'])} (basis: chan_coords_62.npy)")
    # 落盘后立即经 load_geometry 消费路径回读验证
    from latent_flow.models.graph_utils import load_geometry
    c2, n2 = load_geometry(path=str(EDRA_FMT_GEOMETRY_PATH))
    assert c2.shape == (62, 3) and len(n2) == 62 and n2 == list(names)
    assert torch.allclose(c2, torch.tensor(coords, dtype=torch.float32))
    print("load_geometry round-trip OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
