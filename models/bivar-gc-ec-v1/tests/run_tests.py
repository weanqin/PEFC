#!/usr/bin/env python3
"""tests/run_tests —— bivar-gc-ec-v1 单测（独立脚本、逐项断言打印、rng seed 2026）。

T1 方向：合成二通道 VAR(2) 仅 A→B 耦合 → GC(A→B) > GC(B→A) 且 GC(B→A) < 0.01。
T2 独立零基线：两通道独立 AR(1) → 双向 |GC| < 0.01。
T3 Gram-slice 等价：8ch/T=3000 随机数据（连续 + 分段），主路径 vs naive lstsq 的
   SSR/BIC 逐项对比 rtol=1e-10；GC 对比 rtol=1e-10 + atol=1e-12 下限（≈0 值为噪声
   支配，见 t3 内注释）。
T4 逐位复现：同输入两次调用 array_equal。
T5 装载断言：字段不匹配/截断/形状异常的假 npz（tmp 目录）→ AssertionError。
T6 非负性：T1/T2 数据上 offdiag≥0 / 对角=0 / 全有限（真实数据非负断言在 run.py 主流程）。

单测用小规模合成数据；1 例真实数据冒烟由 B0 smoke 单独做（run.py --stage smoke），不进单测。
用法：python3 models/bivar-gc-ec-v1/tests/run_tests.py（全绿退出码 0）。
"""

from __future__ import annotations

import os

# ---- BLAS 护栏（environment.md 惯例：必须在 numpy 之前；setdefault 可外部覆盖）----
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "8")

import sys              # noqa: E402
import tempfile         # noqa: E402
from pathlib import Path # noqa: E402

import numpy as np      # noqa: E402

_HERE = Path(__file__).resolve().parent
_VARIANT = _HERE.parent
if str(_VARIANT) not in sys.path:
    sys.path.insert(0, str(_VARIANT))

import bivar_gc as bg   # noqa: E402
import data_io          # noqa: E402

SEED = 2026
_N_PASS = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    """逐项断言打印（全绿退出 0；失败即刻 AssertionError）。"""
    global _N_PASS
    assert cond, f"[FAIL] {name} {detail}"
    _N_PASS += 1
    print(f"[PASS] {name}" + (f" — {detail}" if detail else ""))


# ---- 合成数据生成器（T1/T2/T6 复用；rng seed 2026 每测试独立重置） ----

def sim_coupled_var2(rng: np.random.Generator, T: int = 20000, burn: int = 200) -> np.ndarray:
    """二通道 VAR(2)：B 方程含 A 滞后（0.35/−0.15），A 方程不含 B —— 仅 A→B 耦合。

    独立白噪新息；系数平稳（A: 0.55/−0.3，B 自身: 0.5/−0.2）；去 burn-in。
    """
    e = rng.standard_normal((2, T + burn))
    x = np.zeros((2, T + burn))
    for t in range(2, T + burn):
        x[0, t] = 0.55 * x[0, t - 1] - 0.30 * x[0, t - 2] + e[0, t]        # A：不含 B
        x[1, t] = 0.50 * x[1, t - 1] - 0.20 * x[1, t - 2] \
            + 0.35 * x[0, t - 1] - 0.15 * x[0, t - 2] + e[1, t]            # B：含 A 滞后
    return x[:, burn:]


def sim_independent_ar1(rng: np.random.Generator, T: int = 20000, burn: int = 200) -> np.ndarray:
    """两通道独立 AR(1)（φ=0.6，独立新息）——独立零基线。"""
    e = rng.standard_normal((2, T + burn))
    x = np.zeros((2, T + burn))
    for t in range(1, T + burn):
        x[0, t] = 0.6 * x[0, t - 1] + e[0, t]
        x[1, t] = 0.6 * x[1, t - 1] + e[1, t]
    return x[:, burn:]


# ---- T1 方向 ----

def t1_direction() -> None:
    x = sim_coupled_var2(np.random.default_rng(SEED))
    r = bg.bivar_gc_matrix(x, p_max=24)
    gc_ab = float(r["ec"][0, 1])   # 行=源(A=ch0) 列=靶(B=ch1) → A→B
    gc_ba = float(r["ec"][1, 0])
    check("T1 GC(A→B) > GC(B→A)", gc_ab > gc_ba,
          f"gc_ab={gc_ab:.5f} gc_ba={gc_ba:.5f} p*={r['orders'][0, 1]}")
    check("T1 GC(B→A) < 0.01（独立白噪新息，反向近零）", gc_ba < 0.01,
          f"gc_ba={gc_ba:.6f}（阈值 0.01）")


# ---- T2 独立零基线 ----

def t2_independence() -> None:
    x = sim_independent_ar1(np.random.default_rng(SEED))
    r = bg.bivar_gc_matrix(x, p_max=24)
    check("T2 双向 |GC| < 0.01（独立 AR(1) 零基线）",
          abs(float(r["ec"][0, 1])) < 0.01 and abs(float(r["ec"][1, 0])) < 0.01,
          f"gc_ab={r['ec'][0, 1]:.6f} gc_ba={r['ec'][1, 0]:.6f} "
          f"p*={r['orders'][0, 1]}")


# ---- T3 Gram-slice ↔ naive lstsq 等价 ----

def t3_equivalence() -> None:
    rng = np.random.default_rng(SEED)
    Cn, T, p_max = 8, 3000, 24
    rtol = 1e-10
    atol_gc = 1e-12   # GC 对比加 atol 下限：纯随机数据个别方向 GC≈0（如 1e-10 量级，
                      # 真值本身为噪声支配），两算术路径无法对齐 10 位有效数字；atol=1e-12
                      # 仍强制绝对一致到 12 位小数（远低于任何有意义的 GC 量级）。
                      # SSR/BIC 为大量级量，保持纯 rtol=1e-10。
    p_grid = tuple(range(1, p_max + 1))
    # (a) 逐对逐阶 SSR/BIC/p*/GC 全量对比（连续输入）
    x = rng.standard_normal((Cn, T))
    gram = bg.build_gram(x, p_max)
    for a in range(Cn - 1):
        for b in range(a + 1, Cn):
            fast = bg.pair_fit(gram, a, b, p_grid, p_max)
            ref = bg.naive_pair_fit(x, a, b, p_grid)
            np.testing.assert_allclose(fast["ssr_u"], ref["ssr_u"], rtol=rtol)
            np.testing.assert_allclose(fast["ssr_r"], ref["ssr_r"], rtol=rtol)
            np.testing.assert_allclose(fast["bic"], ref["bic"], rtol=rtol)
            assert fast["p_star"] == ref["p_star"], \
                f"pair ({a},{b}): p* 不一致 fast={fast['p_star']} naive={ref['p_star']}"
            np.testing.assert_allclose([fast["gc_ab"], fast["gc_ba"]],
                                       [ref["gc_ab"], ref["gc_ba"]],
                                       rtol=rtol, atol=atol_gc)
    check("T3a 连续输入：28 对 × 24 阶 SSR/BIC/GC/p* 全量等价 (rtol=1e-10)", True)
    # (b) 矩阵级：主路径 vs naive 全矩阵
    r_fast = bg.bivar_gc_matrix(x, p_max)
    r_ref = bg.naive_bivar_gc_matrix(x, p_max)
    np.testing.assert_allclose(r_fast["ec"], r_ref["ec"], rtol=rtol, atol=atol_gc)
    assert np.array_equal(r_fast["orders"], r_ref["orders"])
    check("T3b 连续输入：EC 矩阵 + 阶数矩阵等价", True)
    # (c) 分段输入（split-half 段式路径）：等价 + 行数核对（防段式静默退化为连续）
    xs = rng.standard_normal((4, Cn, 800))
    r_seg = bg.bivar_gc_matrix(xs, p_max)
    r_seg_ref = bg.naive_bivar_gc_matrix(xs, p_max)
    np.testing.assert_allclose(r_seg["ec"], r_seg_ref["ec"], rtol=rtol, atol=atol_gc)
    assert np.array_equal(r_seg["orders"], r_seg_ref["orders"])
    check(f"T3c 分段输入 (4,C,800)：EC/阶数与 naive 等价；n_eff={int(r_seg['n'])}"
          f"（=4×(800−24) 段内行集）", int(r_seg["n"]) == 4 * (800 - 24))


# ---- T4 逐位复现 ----

def t4_bitwise_reproducibility() -> None:
    rng = np.random.default_rng(SEED)
    Cn, T = 6, 1500
    # 小规模耦合 VAR（确定性数据，非纯噪声）：对角 AR(1) 0.5 + 少量固定跨通道耦合
    e = rng.standard_normal((Cn, T))
    x = np.zeros((Cn, T))
    for t in range(1, T):
        x[:, t] = 0.5 * x[:, t - 1] + e[:, t]
        x[1, t] += 0.2 * x[0, t - 1]     # 0→1
        x[3, t] += -0.15 * x[2, t - 1]   # 2→3
        x[5, t] += 0.1 * x[4, t - 1]     # 4→5
    r1 = bg.bivar_gc_matrix(x, p_max=12)
    r2 = bg.bivar_gc_matrix(x, p_max=12)
    check("T4 同输入两次调用：EC 逐位相同 (array_equal)",
          np.array_equal(r1["ec"], r2["ec"]))
    check("T4 阶数矩阵逐位相同 (array_equal)",
          np.array_equal(r1["orders"], r2["orders"]))
    check("T4 行数一致", r1["n"] == r2["n"] == float(T - 12))


# ---- T5 装载断言 ----

def _write_fake_npz(path, *, data=None, fs=250, truncated=False,
                    filt="1-40(release)", pipeline="EDRA_release_1-40_250Hz"):
    d = np.zeros((62, 60001), dtype=np.float32) if data is None else data
    np.savez(path, data=d, channel_names=np.array(["ch"] * 62),
             fs=fs, subject_id="t01", group="HC", suffix="rest_v4",
             pipeline=pipeline, filter=filt, n_samples=int(d.shape[-1]),
             truncated=truncated)


def t5_loader_asserts() -> None:
    with tempfile.TemporaryDirectory() as td:
        cases = {
            "filter 字段不匹配": dict(filt="bp1-40+notch50(保真冗余)"),
            "pipeline 字段不匹配": dict(pipeline="EDRA_v4"),
            "fs 不匹配 (500)": dict(fs=500),
            "truncated=True": dict(truncated=True),
            "形状异常 (62,60000)": dict(data=np.zeros((62, 60000), dtype=np.float32)),
        }
        for name, kw in cases.items():
            p = str(Path(td) / "fake.npz")
            _write_fake_npz(p, **kw)
            try:
                data_io.prepare_subject(p)
                raised = False
            except AssertionError:
                raised = True
            check(f"T5 装载断言触发：{name} → AssertionError", raised)


# ---- T6 非负性（T1/T2 数据） ----

def t6_nonnegativity() -> None:
    for name, x in (("T1 耦合 VAR(2)", sim_coupled_var2(np.random.default_rng(SEED))),
                    ("T2 独立 AR(1)", sim_independent_ar1(np.random.default_rng(SEED)))):
        r = bg.bivar_gc_matrix(x, p_max=24)
        od = r["ec"][~np.eye(2, dtype=bool)]
        check(f"T6 {name}：offdiag 全部 ≥ 0", (od >= 0).all(),
              f"min={od.min():.3e}")
        check(f"T6 {name}：对角 = 0", float(np.abs(np.diag(r["ec"])).max()) == 0.0)
        check(f"T6 {name}：全有限", bool(np.isfinite(r["ec"]).all()))


def main() -> int:
    t0 = __import__("time").time()
    print(f"=== bivar-gc-ec-v1 单测（rng seed {SEED}；numpy {np.__version__}）===")
    t1_direction()
    t2_independence()
    t3_equivalence()
    t4_bitwise_reproducibility()
    t5_loader_asserts()
    t6_nonnegativity()
    print(f"=== 全部通过：{_N_PASS} 项断言（{__import__('time').time() - t0:.1f}s）===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
