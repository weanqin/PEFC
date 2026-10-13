"""test_compare —— ①偏相关解析例 ②offdiag 展平序/维度 ③Holm ④EC_pert 契约
⑤bootstrap 确定性 ⑥run_comparison 合成冒烟。运行：退出码 0 = 通过。
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _blas_guard  # noqa: F401,E402  # 须在 numpy 之前：BLAS 线程护栏


import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import compare  # noqa: E402


def main():
    rng = np.random.default_rng(2026)

    # ① 解析例：a,b 独立，c=a+b → partial(a,b|c) = −1（秩版本，有限样本容差）
    a = rng.standard_normal(20000)
    b = rng.standard_normal(20000)
    c = a + b
    pv = compare.partial_spearman(a, b, c)
    assert pv < -0.9, f"偏相关解析例失败: {pv:.3f}（应≈−1）"
    # 对照：不偏相关时 a,b 相关≈0
    assert abs(compare._pearson(a, b)) < 0.05
    print(f"① 偏相关解析例 OK（partial={pv:.3f}）")

    # ② offdiag 展平：3782 维、行优先、M[0,1] 为首元素
    M = np.arange(62 * 62, dtype=float).reshape(62, 62)
    v = compare.offdiag_flat(M)
    assert v.size == 62 * 61 == 3782
    assert v[0] == M[0, 1], "行优先序错误（应 M[0,1] 打头）"
    print("② offdiag 展平 OK")

    # ③ Holm：{a:0.01,b:0.04} → {a:0.02,b:0.04}（m=2）
    adj = compare.holm({"a": 0.01, "b": 0.04})
    assert abs(adj["a"] - 0.02) < 1e-12 and abs(adj["b"] - 0.04) < 1e-12
    adj2 = compare.holm({"a": 0.03, "b": 0.04})
    assert abs(adj2["a"] - 0.06) < 1e-12 and abs(adj2["b"] - 0.06) < 1e-12  # 单调化抬升
    print("③ Holm OK")

    # ④ EC_pert 契约：rstar 负值必须报错；slope 负值取 |·| 并存符号
    with tempfile.TemporaryDirectory() as td:
        p1 = Path(td) / "p1.npz"
        np.savez(p1, ec=-np.ones((62, 62)), readout="rstar", subject_ids="global")
        try:
            compare.load_ec_pert(p1)
            raise AssertionError("rstar 负值未报错")
        except ValueError:
            pass
        p2 = Path(td) / "p2.npz"
        m = np.abs(rng.standard_normal((62, 62)))
        m[3, 5] = -0.5
        np.savez(p2, ec=m, readout="slope", subject_ids="global",
                 orientation="larger_is_stronger")
        d = compare.load_ec_pert(p2)
        assert d["ec"][3, 5] == 0.5 and d["ec_sign"][3, 5] == -1
        # orientation 契约断言
        p3 = Path(td) / "p3.npz"
        np.savez(p3, ec=m, readout="slope", subject_ids="global",
                 orientation="smaller_is_stronger")
        try:
            compare.load_ec_pert(p3)
            raise AssertionError("orientation 契约未拦截")
        except ValueError:
            pass
    print("④ EC_pert 契约 OK")

    # ⑤ bootstrap 确定性（同 seed 同 CI）
    vals = rng.standard_normal(46) * 0.2 + 0.3
    ci1 = compare.bootstrap_ci_mean(vals)
    ci2 = compare.bootstrap_ci_mean(vals)
    assert ci1 == ci2, "bootstrap 不确定"
    print("⑤ bootstrap 确定性 OK")

    # ⑥ run_comparison 合成冒烟：pert 与 gc 强相关、与 gpdc 弱相关
    S, Cc = 6, 62
    base = rng.random((Cc, Cc)); np.fill_diagonal(base, 0)
    refs = {"gc": np.stack([base + 0.1 * rng.standard_normal((Cc, Cc)) for _ in range(S)]),
            "gpdc": np.stack([rng.random((Cc, Cc)) for _ in range(S)])}
    np.fill_diagonal(refs["gc"][0], 0)  # 对角不影响（展平去对角）
    ec_pert = {"ec": compare.synthetic_ec_pert_demo(base), "readout": "slope",
               "subject_ids": "global"}
    dist = np.abs(rng.standard_normal((Cc, Cc))) + np.eye(Cc)
    res = compare.run_comparison(refs, [f"s{i}" for i in range(S)], ec_pert, dist,
                                 ceilings={"gc": 0.8, "gpdc": 0.8})
    assert res["methods"]["gc"]["rho_partial_mean"] > res["methods"]["gpdc"]["rho_partial_mean"]
    assert set(res["holm_p"]) == {"gc", "gpdc"}
    assert "ceiling_ratio" in res["methods"]["gc"]
    print(f"⑥ run_comparison 冒烟 OK（gc ρ={res['methods']['gc']['rho_partial_mean']:.3f} > "
          f"gpdc ρ={res['methods']['gpdc']['rho_partial_mean']:.3f}）")

    print("[PASS] test_compare")


if __name__ == "__main__":
    main()
