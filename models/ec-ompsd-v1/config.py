"""ec-ompsd-v1 配置（O×M-PSD；10-18-ec-ompsd-f0 唯一配置源；镜像 ec-nmtime 结构）。"""

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

TEST_SUBJECTS = (10, 11, 12, 17, 24, 28, 30, 44, 61, 69)
N_WINDOWS_PER_SUBJECT = 8
N_WINDOWS = 80

K = 16
SEED = 2026         # 逐窗 manual_seed(SEED+win_idx) 先 eps 后 eta（CRN 契约）
PARADIGM = "O"

R_GRID = (0.0, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0)
R_SAMPLED = R_GRID[1:]
R_MAX = R_GRID[-1]
R_NEXT = R_GRID[-1] + (R_GRID[-1] - R_GRID[-2])
SLOPE_SEG = R_GRID[:5]
SLOPE_SEG_SENS = R_GRID[:4]
N_SRC = N_CHANNELS
BATCH_ROWS = 1 + N_SRC * len(R_SAMPLED)    # 497

WELCH_NPERSEG = 128
WELCH_NOVERLAP = 64
PSD_EPS = 1e-12
PSD_BAND = (1.0, 40.0)

D_AGG = "median"
TAU_TILDE = 0.25
TAU_TILDE_SENS = 0.50
ORIENTATION = "larger_is_stronger"
SPLIT_HALF = ("s1-s5", "s6-s10")

OUTPUTS_ROOT = "/root/npi/outputs/ec-ompsd-f0"
RUN_DIR_OUT = "run_01"
SMOKE_DIR = "smoke/run_01"
LOG_DIR = "logs"
