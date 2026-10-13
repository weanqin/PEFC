"""baseline-ec-v1 config —— 全部口径常量（design.md 的代码化；改动=口径变更须登记）。"""

from pathlib import Path

REPO_ROOT = Path("/root/npi")
DATA_ROOT = Path("/root/autodl-tmp/edra_process_v4")
OUTPUTS_ROOT = REPO_ROOT / "outputs" / "baseline-ec-v1"
V1_DIR = REPO_ROOT / "models" / "gpfm-mamba-v1"   # gc_psd.py 只读复用来源

# ---- 数据口径 ----
FS_RAW = 500                       # 原始采样率（契约）
FS = 125                           # 目标采样率（带限 1-40 Hz，Nyquist 62.5 带内无损）
RESAMPLE_UP, RESAMPLE_DOWN = 1, 4  # resample_poly 因子（自带抗混叠 FIR）
N_CHANNELS = 62
EXPECTED_FILTER = "bp1-40+notch50(保真冗余)"  # 逐例断言；异常即中止（design §2）
SUBJECT_FILTER = "status_ok"       # 46 例，与 Study I 相同口径

# ---- VAR / EC 口径 ----
P_MAX = 24                         # BIC 搜索上限（125 Hz；ASCT p/2 谱峰锚点 7-16）
BANDS = {                          # 上限 40 Hz 由数据滤波硬界（edra.md 实测）
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "lowgamma": (30.0, 40.0),
}
BROADBAND = (1.0, 40.0)
N_FFT = 250                        # 0.5 Hz 分辨率 @125 Hz（gPDC 频率网格）

# ---- split-half ----
SPLIT_SEG_S = 2.0                  # 奇偶交错分段长度（秒）
SPLIT_SEG_SAMPLES = int(SPLIT_SEG_S * FS)   # 250

# ---- 比较器 ----
BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 2026              # 项目惯例（gc_psd BOOTSTRAP_SEED 同源）
HOLM_METHODS = ("gc", "gpdc")      # 主轨两参考（TE 已取消，2026-10-13 用户拍板）
MNEC_LAGS_FALLBACK = 10            # mne-c 交叉核对在 p* 不可用时的滞后上限

# ---- mne-connectivity 交叉核对 ----
MNEC_FDECIM = 64
