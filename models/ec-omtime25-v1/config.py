"""ec-omtime25-v1 配置（O×M-time@25；10-18-ec-omtime25-f0 唯一配置源；骨架复制自 ec-ompsd-v1 适配）。"""

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

MTIME_EARLY_SAMPLES = 25      # 主标量 = |δ̄| 前 25 样本（0.1 s）均值——用户 2026-10-18 指定
                             #（替换 ec-nmtime 的 32 点/128 ms 口径；full/peak 仍随附）

D_AGG = "median"
TAU_TILDE = 0.25
TAU_TILDE_SENS = 0.50
ORIENTATION = "larger_is_stronger"
SPLIT_HALF = ("s1-s5", "s6-s10")

OUTPUTS_ROOT = "/root/npi/outputs/ec-omtime25-f0"
RUN_DIR_OUT = "run_01"
SMOKE_DIR = "smoke/run_01"
LOG_DIR = "logs"
