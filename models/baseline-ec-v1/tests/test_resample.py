"""test_resample —— ①带内幅值保持 ②抗混叠抑制 ③长度折半×4 ④真实 npz 全链（可选）。

运行：python3 models/baseline-ec-v1/tests/test_resample.py（退出码 0 = 通过）
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _blas_guard  # noqa: F401,E402  # 须在 numpy 之前：BLAS 线程护栏

import numpy as np
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C  # noqa: E402


def main():
    rng = np.random.default_rng(2026)
    fs0 = 500.0

    # ① 带内（10 Hz）幅值保持（含弱噪声）
    t = np.arange(85000) / fs0
    x = np.sin(2 * np.pi * 10 * t) + 0.05 * rng.standard_normal(t.size)
    x125 = resample_poly(x, 1, 4)
    r_in = float(np.sqrt(np.mean(x ** 2)))
    r_out = float(np.sqrt(np.mean(x125 ** 2)))
    assert 0.9 < r_out / r_in < 1.1, f"带内幅值漂移: {r_out/r_in:.3f}"
    f125 = np.fft.rfftfreq(x125.size, 1 / 125.0)
    spec = np.abs(np.fft.rfft(x125))
    peak = f125[np.argmax(spec)]
    assert abs(peak - 10.0) <= 0.5, f"峰频漂移 {peak}"
    print("① 带内幅值/峰频保持 OK")

    # ② 抗混叠：90 Hz（> 目标 Nyquist 62.5）应被强抑制
    x90 = np.sin(2 * np.pi * 90 * t)
    y90 = resample_poly(x90, 1, 4)
    # 折叠频率 125-90=35 Hz 处不应出现大分量（< -20 dB 即幅值比 <0.1）
    spec90 = np.abs(np.fft.rfft(y90))
    band = (f125 > 30) & (f125 < 40)
    ref_band = (f125 > 80) | (f125 < 5)
    amp_fold = spec90[band].max()
    assert amp_fold < 0.05 * len(y90) / 2 * 1e-1 or amp_fold / (spec90.max() + 1e-12) < 0.1, \
        f"混叠残余过大: fold={amp_fold:.1f} vs max={spec90.max():.1f}"
    print("② 抗混叠抑制 OK")

    # ③ 长度 85000 → 21250
    assert x125.shape[-1] == 21250, x125.shape
    print("③ 长度 OK")

    # ④ 真实 npz 全链（数据在位时；filter 断言 + z 均值~0/方差~1 + 形状）
    real = Path(C.DATA_ROOT) / "HC" / "13_rest_v4.npz"
    if real.exists():
        import resample_io
        prep = resample_io.load_and_prepare(str(real))
        x = prep["x"]
        assert x.shape == (62, 21250), x.shape
        assert np.allclose(x.mean(-1), 0, atol=1e-6)
        assert np.allclose(x.std(-1), 1, atol=1e-4)
        assert prep["meta"]["fs"] == 125 and prep["meta"]["fs_raw"] == 500
        print("④ 真实 npz 全链 OK（HC/13）")
    else:
        print("④ 真实 npz 不在位，跳过")

    print("[PASS] test_resample")


if __name__ == "__main__":
    main()
