#!/usr/bin/env python3
"""compare_gc.py — 三方 r*-strength EC × 双变量 GC 参考的受试者级 Spearman 相关
（10-18-ec-omtime25-f0 Run 002；用户 2026-10-18 拍板：主口径 = strength EC 非 slope）。

消费（只读）：三方 ec_rstar_s{sid}.npz（τ̃=0.25，已 flip，大=强）+ GC refs_s{sid}.npz。
产物：outputs/ec-omtime25-f0/run_01/gc_compare_strength.{md,npz}
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import candidates as CD

SC = CD.load_config()
SUBJ = SC.TEST_SUBJECTS
VARIANTS = {
    "N×M-time@32": Path("/root/npi/outputs/ec-nmtime-f0/run_01"),
    "O×M-time@25": Path(SC.OUTPUTS_ROOT) / "run_01",
    "O×M-PSD": Path("/root/npi/outputs/ec-ompsd-f0/run_01"),
}
GC_DIR = Path("/root/npi/outputs/bivar-gc-ec-fmt/refs-01/refs")
off = ~np.eye(62, dtype=bool)

rows, rhos = {}, {}
for name, d in VARIANTS.items():
    rs = []
    for sid in SUBJ:
        st = np.load(d / f"ec_rstar_s{sid}.npz")["ec"]
        gc = np.load(GC_DIR / f"refs_s{sid}.npz")["ec"]
        assert st.shape == gc.shape == (62, 62)
        rs.append(float(spearmanr(st[off], gc[off]).statistic))
    rhos[name] = np.array(rs)
    rows[name] = " | ".join(f"{v:+.3f}" for v in rs)

lines = [f"# 三方 r\\*-strength EC × GC（受试者级 Spearman，offdiag 3782 边）— Run 002",
         "",
         f"- 主口径 = **r\\*-strength**（τ̃=0.25、flip 后大=强、删失=0；用户 2026-10-18 拍板，"
         f"替换 slope 口径）；GC = 双变量 ln(SSR) EC（10-17-bivar 任务 refs-01）；受试者 = fold0 test 10 例。",
         "",
         "| 变体 | " + " | ".join(str(s) for s in SUBJ) + " | ρ̄ ± sd | 范围 |",
         "|---|" + "---|" * (len(SUBJ) + 2)]
for name in VARIANTS:
    r = rhos[name]
    lines.append(f"| {name} | " + " | ".join(f"{v:+.3f}" for v in r) +
                 f" | **{r.mean():+.3f} ± {r.std():.3f}** | [{r.min():+.3f}, {r.max():+.3f}] |")
lines += ["",
          f"- 参照系：GC split-half 天花板 0.9243（方法间相关可达上界的量级参照）；"
          f"10-17 任务 G×M-RMS slope×GC ρ̄ = −0.007（p=0.581，前次阴性基线）。",
          "- 判读纪律：本 Run 为相关登记，非收敛判定（无 Holm/多对照面）——收敛结论留待正式对照任务。"]
out = Path(SC.OUTPUTS_ROOT) / "run_01"
out.mkdir(parents=True, exist_ok=True)
(out / "gc_compare_strength.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
np.savez(out / "gc_compare_strength.npz",
         variants=np.array(list(VARIANTS)), subjects=np.array(SUBJ),
         rho=np.stack([rhos[n] for n in VARIANTS]))
print("\n".join(lines))
