"""pastdep-audit-v1 配置（design §2–§3 字面转录；10-18-pastdep-audit-f0 唯一配置源）。

纯常量模块；importlib 唯一名 `pd_config`（不占 `config` 名）。
"""

# ---- θ* 冻结定位（同前两任务，只读）----
CKPT_PATH = ("/root/npi/outputs/gpfm-mamba-fmt-f0/v1fmt-main-01/checkpoints/"
             "gpfm-mamba-v1-fmt-best.pt")
RUN_DIR = "/root/npi/outputs/gpfm-mamba-fmt-f0/v1fmt-main-01"
V1FMT_CONFIG_PATH = "/root/npi/models/gpfm-mamba-v1-fmt/config.py"
EXPECTED_CONFIG_HASH = "0dc0d5249c68"
SPLIT_MANIFEST = ("/root/npi/models/gpfm-mamba-v1-fmt/splits/"
                  "fold0_status_ok_seed42_val4.json")

FS = 250.0
N_CHANNELS = 62
PAST_LEN = 384
FUTURE_LEN = 128

# ---- 窗集（同 perturb-search-f0）：fold0 test 前 3 例 × 前 8 窗 = 24 窗 ----
TEST_SUBJECTS = (10, 11, 12)
N_WINDOWS_PER_SUBJECT = 8
N_WINDOWS = 24            # win_idx = 受试者序×8 + 例内窗序（0..23）

# ---- CRN ----
K = 16
SEED = 2026               # E1b/E2 批 = SEED+win_idx；E3 各臂 = SEED+win_idx+ARM_SEED_OFF[arm]
ARM_SEED_OFF = {"ref": 0, "noise": 1000, "same": 2000, "cross": 3000, "ablate": 4000}

# ---- N 算子常量（与前两任务逐位一致）----
NOISE_WAVEFORM_SEED = 2026
NOISE_BAND = (1.0, 40.0)
NOISE_LEN = PAST_LEN

# ---- E1b/E4 剂量臂 + E2 子窗臂（批构造）----
DOSES = (0.5, 1.0, 2.0)                     # 全 past N 注入（E1b 衰减谱 + E4 边界条件）
E2_DOSE = 1.0                               # 子窗扫描剂量（钉死）
SUBWINDOWS = ((0, 96), (96, 192), (192, 288), (288, 384))   # boxcar 4×96；RMS 子窗内重归一；
                                                            # σ_A = 全 past std（design §2 钉死）
ARMS = tuple([("dose", d) for d in DOSES] + [("win", i) for i in range(4)])
N_ARMS = len(ARMS)                          # 7
N_SRC = N_CHANNELS
BATCH_ROWS = 1 + N_SRC * N_ARMS             # 435 = 1 + 62×7；行 1 + a·7 + k

# ---- 判据（预注册）----
T50_REL_TOL = 0.20           # |T50_sample/T50_prior − 1| < 20% ⇒ H1（先验主导）
RHO_BANDS = (1.2, 2.0)       # ρ<1.2 无信息依赖；1.2–2 边缘；≥2 强依赖
MTIME_EARLY_SAMPLES = 32     # φ = 前 32 样本响应质量分数（对齐 M-time 口径）

# ---- 产物根 ----
OUTPUTS_ROOT = "/root/npi/outputs/pastdep-audit-f0"
RUN_DIR_OUT = "run_01"
SMOKE_DIR = "smoke/run_01"
LOG_DIR = "logs"
