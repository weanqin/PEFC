"""ec-nmtime-v1 配置（design §2–§5 预注册字面转录；10-18-ec-nmtime-f0 唯一配置源）。

纪律：R_GRID / slope 段 / τ̃ / 窗集 / 聚合一经本文件 + experiment.md Run 001 登记即钉死。
搜索侧参数一律放本模块，绝不进 gpfm Config（v1-fmt config_hash 逐位不可变）。
装载纪律：经 importlib 唯一名 `ec_config` 装载（sys.modules 先注册后 exec），
不占据 `config` 模块名（留给 gpfm-mamba-v1）。纯常量模块。

方向约定（全仓库统一）：行 = 源 source（A）、列 = 靶 target（B）。
"""

# ---- θ* 冻结定位（同 perturb-search-f0，只读）----
CKPT_PATH = ("/root/npi/outputs/gpfm-mamba-fmt-f0/v1fmt-main-01/checkpoints/"
             "gpfm-mamba-v1-fmt-best.pt")   # EMA 权重，best（epoch=81 0-based）
RUN_DIR = "/root/npi/outputs/gpfm-mamba-fmt-f0/v1fmt-main-01"
V1FMT_CONFIG_PATH = "/root/npi/models/gpfm-mamba-v1-fmt/config.py"
EXPECTED_CONFIG_HASH = "0dc0d5249c68"
SPLIT_MANIFEST = ("/root/npi/models/gpfm-mamba-v1-fmt/splits/"
                  "fold0_status_ok_seed42_val4.json")

# ---- fmt 范式事实（断言用）----
FS = 250.0
N_CHANNELS = 62
PAST_LEN = 384
FUTURE_LEN = 128

# ---- 窗集（design §2 钉死）：fold0 test manifest 顺序全 10 例 × 每例前 8 窗 ----
TEST_SUBJECTS = (10, 11, 12, 17, 24, 28, 30, 44, 61, 69)   # fold0 test 全体（manifest 顺序）
N_WINDOWS_PER_SUBJECT = 8
N_WINDOWS = len(TEST_SUBJECTS) * N_WINDOWS_PER_SUBJECT      # 80
# win_idx 规则（钉死）= 受试者 manifest 序 × 8 + 例内窗序（例 #1 → 0–7 … 例 #10 → 72–79）。

# ---- CRN 采样（design §2）----
K = 16
SEED = 2026         # 逐窗 torch.Generator.manual_seed(SEED + win_idx)，先 eps 后 eta

# ---- 输入端：N 单范式（常量与 perturb-search-v1 登记逐位一致）----
PARADIGM = "N"
NOISE_WAVEFORM_SEED = 2026     # g(t)：固定 seed FFT 域 1–40 Hz 掩码；RMS 重归一 1；全局共享单波形
NOISE_BAND = (1.0, 40.0)
NOISE_LEN = PAST_LEN

# ---- R_GRID（design §3 预注册钉死）：9 点含原点，索引 0 ≡ r=0 ≡ D̄=0 恒等构造不采样 ----
R_GRID = (0.0, 0.03125, 0.0625, 0.125, 0.25, 0.5, 0.75, 1.0, 1.5)
R_SAMPLED = R_GRID[1:]                     # 8 个采样格点（批构造消费）
R_MAX = R_GRID[-1]                         # 1.5
R_NEXT = R_GRID[-1] + (R_GRID[-1] - R_GRID[-2])   # 2.0，线性外推推导（删失编码档，禁字面量）
SLOPE_SEG = R_GRID[:5]                     # {0, 0.03125, 0.0625, 0.125, 0.25}（含原点锚）
SLOPE_SEG_SENS = R_GRID[:4]                # 敏感性段照登
N_SRC = N_CHANNELS
BATCH_ROWS = 1 + N_SRC * len(R_SAMPLED)    # 497 = 1 baseline + 62 源 × 8 格点

# ---- 输出端：M-time（主标量口径同 perturb-search-f0 钉死值）----
MTIME_EARLY_SAMPLES = 32      # D = |δ̄(t)| 前 32 样本（128 ms）均值；full/peak 随附登记

# ---- 聚合（design §4）----
D_AGG = "median"              # 逐 subject = 8 窗逐元素 median；global = 80 窗逐元素 median

# ---- τ̃ 与 readout（design §5；evaluation-calibers τ v2）----
TAU_TILDE = 0.25              # r* 主口径：D̃_{A→B}(r) = D̄_{A→B}(r)/D̄_{B→B}(r) ≥ 0.25
TAU_TILDE_SENS = 0.50         # 敏感性口径照登
ORIENTATION = "larger_is_stronger"   # 契约 npz 字段（rstar 存盘前 flip）
SPLIT_HALF = ("s1-s5", "s6-s10")     # 描述统计⑤预注册指派：例 #1–#5 vs #6–#10

# ---- 产物根（path-isolation）----
OUTPUTS_ROOT = "/root/npi/outputs/ec-nmtime-f0"
RUN_DIR_OUT = "run_01"
SMOKE_DIR = "smoke/run_01"
LOG_DIR = "logs"
