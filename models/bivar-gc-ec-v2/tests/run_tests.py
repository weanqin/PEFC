#!/usr/bin/env python3
"""tests/run_tests —— bivar-gc-ec-v2 单测（独立脚本、逐项断言打印、rng seed 2026；design §4 八项）。

T1 方向：合成二通道 VAR(2) 仅 A→B 耦合 → GC(A→B) > GC(B→A) 且 GC(B→A)≈0（经 v2
   import 的 v1 bivar_gc 路径，P_MAX=48 新档位）。
T2 独立零基线：两通道独立白噪 → 双向 |GC| < 0.01。
T3 样本内非负断言（真实数据）：装载 1 例真实受试者全 62 通道 1891 对读出 offdiag≥0
   （同时全流程冒烟；真实数据全量非负断言在 run.py 主流程逐例复核）。
T4 Gram-slice ↔ naive lstsq 等价（rtol=1e-10；锁定 p_max=48 新档位索引 + v2 的 v1
   import 链路无遮蔽漂移；连续 + 分段输入）。
T5 逐位复现：同输入两次调用 array_equal（确定性契约）。
T6 装载与归属断言：fs≠250 / 长度≠60001 / truncated → AssertionError；bivar_gc /
   split_half 归属 v1、data_io 归属 v2；v1 bgec_config 与 v2 bgec2_config 实例独立。
T7 split 段长锁定：合成 T=1500、seg=500 → n_seg==3（锁定 v2 编排路径，防 v1 缺省
   250 漂移面——v1 bgec_config.SEG_SAMPLES=250 事实同锁）。
T8 时间反转改变断言：强单向合成 VAR（仅 A→B）翻转后 GC_rev(B→A) > GC_rev(A→B)
   且 |GC_rev(A→B) − GC_fwd(A→B)| 超显式阈值；GC_rev 有限非负；T 与 p_max=48 入断言。

用法：python3 models/bivar-gc-ec-v2/tests/run_tests.py（全绿退出码 0）。
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
_VARIANT = _HERE.parent                                  # models/bivar-gc-ec-v2
_V1 = _VARIANT.parent / "bivar-gc-ec-v1"
if str(_VARIANT) not in sys.path:
    sys.path.insert(0, str(_VARIANT))                    # v2 居首
if str(_V1) not in sys.path:
    sys.path.append(str(_V1))                            # v1 仅追加

import bivar_gc as bg        # noqa: E402（须解析到 v1）
import split_half as sh      # noqa: E402（须解析到 v1）
import data_io               # noqa: E402（须解析到 v2）
import bgec2_config as C2    # noqa: E402（v2 config，唯一名）

SEED = 2026
_N_PASS = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    """逐项断言打印（全绿退出 0；失败即刻 AssertionError）。"""
    global _N_PASS
    assert cond, f"[FAIL] {name} {detail}"
    _N_PASS += 1
    print(f"[PASS] {name}" + (f" — {detail}" if detail else ""))


# ---- 合成数据生成器（rng seed 2026 每测试独立重置） ----

def sim_coupled_var2(rng: np.random.Generator, T: int = 20000, burn: int = 200,
                     ab1: float = 0.35, ab2: float = -0.15) -> np.ndarray:
    """二通道 VAR(2)：B 方程含 A 滞后、A 方程不含 B —— 仅 A→B 单向耦合。

    独立白噪新息；系数平稳（A 自身: 0.55/−0.3，B 自身: 0.5/−0.2）；去 burn-in。
    """
    e = rng.standard_normal((2, T + burn))
    x = np.zeros((2, T + burn))
    for t in range(2, T + burn):
        x[0, t] = 0.55 * x[0, t - 1] - 0.30 * x[0, t - 2] + e[0, t]        # A：不含 B
        x[1, t] = 0.50 * x[1, t - 1] - 0.20 * x[1, t - 2] \
            + ab1 * x[0, t - 1] + ab2 * x[0, t - 2] + e[1, t]              # B：含 A 滞后
    return x[:, burn:]


# ---- T1 方向（P_MAX=48 新档位；经 v2 import 的 v1 bivar_gc 路径） ----

def t1_direction() -> None:
    assert bg.__file__ == str(_V1 / "bivar_gc.py")     # 归属前置（详 T6）
    x = sim_coupled_var2(np.random.default_rng(SEED))
    r = bg.bivar_gc_matrix(x, p_max=C2.P_MAX)
    gc_ab = float(r["ec"][0, 1])   # 行=源(A=ch0) 列=靶(B=ch1) → A→B
    gc_ba = float(r["ec"][1, 0])
    check("T1 GC(A→B) > GC(B→A)", gc_ab > gc_ba,
          f"gc_ab={gc_ab:.5f} gc_ba={gc_ba:.5f} p*={r['orders'][0, 1]}（p_max={C2.P_MAX}）")
    check("T1 GC(B→A) < 0.01（独立白噪新息，反向近零）", gc_ba < 0.01,
          f"gc_ba={gc_ba:.6f}（阈值 0.01）")


# ---- T2 独立零基线（独立白噪） ----

def t2_independence() -> None:
    rng = np.random.default_rng(SEED)
    x = rng.standard_normal((2, 20000))
    r = bg.bivar_gc_matrix(x, p_max=C2.P_MAX)
    check("T2 双向 |GC| < 0.01（独立白噪零基线）",
          abs(float(r["ec"][0, 1])) < 0.01 and abs(float(r["ec"][1, 0])) < 0.01,
          f"gc_ab={r['ec'][0, 1]:.6f} gc_ba={r['ec'][1, 0]:.6f} "
          f"p*={r['orders'][0, 1]}")


# ---- T3 样本内非负断言（真实数据 1 例全 62 通道） ----

def t3_realdata_nonneg() -> None:
    manifest = data_io.load_manifest()
    sid = C2.EXPECTED_IDS[0]
    e = manifest[sid]
    assert e["status"] == "ok"
    prep = data_io.prepare_subject(e["npz_path"])
    x, m = prep["x"], prep["meta"]
    assert x.shape == (C2.N_CHANNELS, C2.N_RAW)
    r = bg.bivar_gc_matrix(x, p_max=C2.P_MAX)          # 内部含逐对 offdiag≥0 断言
    od = r["ec"][~np.eye(C2.N_CHANNELS, dtype=bool)]
    check(f"T3 真实数据 s{sid}（{m['group']}）：全 1891 对 offdiag ≥ 0", (od >= 0).all(),
          f"min={od.min():.3e} median={np.median(od):.4f} max={od.max():.4f} "
          f"n_eff={int(r['n'])}")
    check(f"T3 真实数据 s{sid}：对角 = 0 且全有限",
          float(np.abs(np.diag(r["ec"])).max()) == 0.0 and bool(np.isfinite(r["ec"]).all()))
    check(f"T3 真实数据 s{sid}：阶数矩阵对称且 p* ∈ [1,{C2.P_MAX}]",
          bool((r["orders"] == r["orders"].T).all()
               and r["orders"][0, 1] >= 1 and r["orders"][0, 1] <= C2.P_MAX),
          f"p*(0,1)={r['orders'][0, 1]}")


# ---- T4 Gram-slice ↔ naive lstsq 等价（rtol=1e-10） ----

def t4_equivalence() -> None:
    rng = np.random.default_rng(SEED)
    Cn, T, p_max = 8, 3000, C2.P_MAX
    rtol = 1e-10
    atol_gc = 1e-12   # GC 对比加 atol 下限：纯随机数据个别方向 GC≈0（噪声支配真值），
                      # 两算术路径无法对齐 10 位有效数字；SSR/BIC 为大量级量保持纯 rtol。
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
    check(f"T4a 连续输入：28 对 × {p_max} 阶 SSR/BIC/GC/p* 全量等价 (rtol=1e-10；"
          f"p_max={p_max} 新档位索引锁定)", True)
    # (b) 矩阵级：主路径 vs naive 全矩阵
    r_fast = bg.bivar_gc_matrix(x, p_max)
    r_ref = bg.naive_bivar_gc_matrix(x, p_max)
    np.testing.assert_allclose(r_fast["ec"], r_ref["ec"], rtol=rtol, atol=atol_gc)
    assert np.array_equal(r_fast["orders"], r_ref["orders"])
    check("T4b 连续输入：EC 矩阵 + 阶数矩阵等价", True)
    # (c) 分段输入（split-half 段式路径）：等价 + 行数核对（防段式静默退化为连续）
    xs = rng.standard_normal((4, Cn, 800))
    r_seg = bg.bivar_gc_matrix(xs, p_max)
    r_seg_ref = bg.naive_bivar_gc_matrix(xs, p_max)
    np.testing.assert_allclose(r_seg["ec"], r_seg_ref["ec"], rtol=rtol, atol=atol_gc)
    assert np.array_equal(r_seg["orders"], r_seg_ref["orders"])
    check(f"T4c 分段输入 (4,C,800)：EC/阶数与 naive 等价；n_eff={int(r_seg['n'])}"
          f"（=4×(800−{p_max}) 段内行集）", int(r_seg["n"]) == 4 * (800 - p_max))


# ---- T5 逐位复现（确定性契约） ----

def t5_bitwise_reproducibility() -> None:
    rng = np.random.default_rng(SEED)
    Cn, T = 6, 1500
    e = rng.standard_normal((Cn, T))
    x = np.zeros((Cn, T))
    for t in range(1, T):
        x[:, t] = 0.5 * x[:, t - 1] + e[:, t]
        x[1, t] += 0.2 * x[0, t - 1]     # 0→1
        x[3, t] += -0.15 * x[2, t - 1]   # 2→3
        x[5, t] += 0.1 * x[4, t - 1]     # 4→5
    r1 = bg.bivar_gc_matrix(x, p_max=12)
    r2 = bg.bivar_gc_matrix(x, p_max=12)
    check("T5 同输入两次调用：EC 逐位相同 (array_equal)",
          np.array_equal(r1["ec"], r2["ec"]))
    check("T5 阶数矩阵逐位相同 (array_equal)",
          np.array_equal(r1["orders"], r2["orders"]))
    check("T5 行数一致", r1["n"] == r2["n"] == float(T - 12))


# ---- T6 装载与归属断言 ----

def _write_fake_npz(path, *, data=None, fs=250, truncated=False,
                    filt="1-40(release)", pipeline="EDRA_release_1-40_250Hz"):
    d = np.zeros((62, 60001), dtype=np.float32) if data is None else data
    np.savez(path, data=d, channel_names=np.array(["ch"] * 62),
             fs=fs, subject_id="t01", group="HC", suffix="rest_v4",
             pipeline=pipeline, filter=filt, n_samples=int(d.shape[-1]),
             truncated=truncated)


def t6_loader_and_ownership() -> None:
    # (a) data_io 契约断言（v2 250 Hz 版：fs≠250 / 长度≠60001 / truncated）
    with tempfile.TemporaryDirectory() as td:
        cases = {
            "filter 字段不匹配": dict(filt="bp1-40+notch50(保真冗余)"),
            "pipeline 字段不匹配": dict(pipeline="EDRA_v4"),
            "fs 不匹配 (500)": dict(fs=500),
            "truncated=True": dict(truncated=True),
            "长度≠60001 (62,60000)": dict(data=np.zeros((62, 60000), dtype=np.float32)),
        }
        for name, kw in cases.items():
            p = str(Path(td) / "fake.npz")
            _write_fake_npz(p, **kw)
            try:
                data_io.prepare_subject(p)
                raised = False
            except AssertionError:
                raised = True
            check(f"T6a 装载断言触发：{name} → AssertionError", raised)
    # (b) import 归属：bivar_gc/split_half → v1（只读复用）、data_io → v2（独立实现）
    check("T6b bivar_gc 归属 v1 目录", bg.__file__ == str(_V1 / "bivar_gc.py"),
          bg.__file__)
    check("T6b split_half 归属 v1 目录", sh.__file__ == str(_V1 / "split_half.py"),
          sh.__file__)
    check("T6b data_io 归属 v2 目录",
          data_io.__file__ == str(_VARIANT / "data_io.py"), data_io.__file__)
    # (c) 唯一 config 名实例独立：v1 bgec_config（125 Hz 档）vs v2 bgec2_config（250 Hz 档）
    import bgec_config as C1
    check("T6c bgec_config 与 bgec2_config 为不同模块实例",
          C1 is not C2 and C1.__file__ == str(_V1 / "config.py")
          and C2.__file__ == str(_VARIANT / "config.py"))
    check("T6c 口径常量互不串扰：P_MAX 24(v1) vs 48(v2)、SEG_SAMPLES 250(v1) vs 500(v2)、"
          "FS 125(v1) vs 250(v2)",
          (C1.P_MAX, C2.P_MAX) == (24, 48) and (C1.SEG_SAMPLES, C2.SEG_SAMPLES) == (250, 500)
          and (C1.FS, C2.FS) == (125, 250))
    check("T6c v2 config 独立性：改 v1 实例属性不影响 v2 实例",
          (setattr(C1, "P_MAX", 999) or C2.P_MAX == 48) and (setattr(C1, "P_MAX", 24) or True))


# ---- T7 split 段长锁定（v2 编排路径；防 v1 缺省 250 漂移面） ----

def t7_split_seg_lock() -> None:
    rng = np.random.default_rng(SEED)
    x = rng.standard_normal((3, 1500))
    halves = sh.build_halves(x, seg=500)               # v2 编排：显式 seg=500
    check("T7 build_halves(x, seg=500) → n_seg==3", halves["n_seg"] == 3,
          f"n_seg={halves['n_seg']}")
    check("T7 偶段A(2段)/奇段B(1段) 形状 (n,C,500)",
          halves["A"].shape == (2, 3, 500) and halves["B"].shape == (1, 3, 500),
          f"A={halves['A'].shape} B={halves['B'].shape}")
    ra = sh.half_ec(halves["A"], p_max=C2.P_MAX)
    rb = sh.half_ec(halves["B"], p_max=C2.P_MAX)
    check(f"T7 每半行数 = n_seg_half×(500−{C2.P_MAX})：A=904 / B=452（段内设计不跨段）",
          float(ra["n"]) == 2 * (500 - C2.P_MAX) and float(rb["n"]) == 1 * (500 - C2.P_MAX),
          f"n_a={int(ra['n'])} n_b={int(rb['n'])}")
    # 漂移面事实锁：v1 build_halves 缺省落 v1 config SEG_SAMPLES=250（这就是必须显式
    # 传 seg=500 的原因；v2 split worker 禁用缺省路径并断言 n_seg==120）
    import bgec_config as C1
    halves_default = sh.build_halves(x)                # 不传 seg → v1 缺省 250
    check("T7 漂移面登记：build_halves(x) 缺省 seg=v1 SEG_SAMPLES=250 → n_seg==6（≠3），"
          "v2 编排必须显式传 seg=500",
          C1.SEG_SAMPLES == 250 and halves_default["n_seg"] == 6,
          f"v1 SEG_SAMPLES={C1.SEG_SAMPLES}，缺省 n_seg={halves_default['n_seg']}")


# ---- T8 时间反转改变断言（design §4-8） ----

def t8_time_reversal_changes_gc() -> None:
    T, p_max = 20000, C2.P_MAX
    assert (T, p_max) == (20000, 48)                   # T 与 p_max 入断言锁定
    x = sim_coupled_var2(np.random.default_rng(SEED), T=T, ab1=0.50, ab2=-0.25)  # 强单向
    fwd = bg.bivar_gc_matrix(x, p_max=p_max)
    x_rev = np.ascontiguousarray(x[:, ::-1])           # 与 run.py reverse 同式翻转
    rev = bg.bivar_gc_matrix(x_rev, p_max=p_max)
    assert np.isfinite(rev["ec"]).all() and (rev["ec"][~np.eye(2, dtype=bool)] >= 0).all()
    fwd_ab, fwd_ba = float(fwd["ec"][0, 1]), float(fwd["ec"][1, 0])
    rev_ab, rev_ba = float(rev["ec"][0, 1]), float(rev["ec"][1, 0])
    check("T8 前向强单向成立：GC_fwd(A→B) > GC_fwd(B→A)", fwd_ab > fwd_ba,
          f"fwd_ab={fwd_ab:.4f} fwd_ba={fwd_ba:.4f}")
    # 翻转后方向互换：原 B 方程含 A 滞后 ⇒ 反转序列中 B 的"未来"驱动 A 的"过去"读数
    check("T8 GC_rev(B→A) > GC_rev(A→B)（翻转改变方向读数，防实现无视翻转）",
          rev_ba > rev_ab, f"rev_ab={rev_ab:.4f} rev_ba={rev_ba:.4f}")
    delta = abs(rev_ab - fwd_ab)
    check("T8 |GC_rev(A→B) − GC_fwd(A→B)| > 0.05（显式阈值）",
          delta > 0.05, f"delta={delta:.4f}（rev_ab={rev_ab:.4f} vs fwd_ab={fwd_ab:.4f}）")
    check("T8 GC_rev 有限且 offdiag 非负（同口径断言在翻转数据上成立）",
          bool(np.isfinite(rev["ec"]).all()) and float(np.abs(np.diag(rev["ec"])).max()) == 0.0)


def main() -> int:
    t0 = __import__("time").time()
    print(f"=== bivar-gc-ec-v2 单测（rng seed {SEED}；numpy {np.__version__}；"
          f"p_max={C2.P_MAX}）===")
    t1_direction()
    t2_independence()
    t3_realdata_nonneg()
    t4_equivalence()
    t5_bitwise_reproducibility()
    t6_loader_and_ownership()
    t7_split_seg_lock()
    t8_time_reversal_changes_gc()
    print(f"=== 全部通过：{_N_PASS} 项断言（{__import__('time').time() - t0:.1f}s）===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
