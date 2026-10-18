"""perturb-search-v1 搜索侧口径常量（design §2–§5 预注册字面转录；10-18-perturb-search-f0 唯一配置源）。

纪律（design 预注册声明 + prd 口径纪律）：
- 候选集 / r0 / gates / 探针 / 窗集 / win_idx 枚举一经本文件 + experiment.md 首个 Run 块登记即钉死；
  主 Run 后最多一次「档位级」调整（K、窗数）并登记，候选定义不得中途改。
- 搜索侧参数一律放本模块，**绝不进 gpfm Config**（v1-fmt config_hash 逐位不可变，
  ckpt 校验需原 hash 0dc0d5249c68）。
- 装载纪律：本模块经 importlib 唯一名 `search_config` 装载（sys.modules 先注册后 exec），
  不占据 `config` 模块名——该名留给 gpfm-mamba-v1（sample.py 顶层 import 消费）。
- 纯常量模块（非 dataclass）。

方向约定（全仓库统一）：行 = 源 source（A）、列 = 靶 target（B）；offdiag 有向边 3782。
"""

# ---- θ* 冻结定位（prd R1，只读；v1-fmt 前置验证级，结论表述「v1-fmt θ* 上的筛选证据」）----
CKPT_PATH = ("/root/npi/outputs/gpfm-mamba-fmt-f0/v1fmt-main-01/checkpoints/"
             "gpfm-mamba-v1-fmt-best.pt")   # EMA 权重，best（epoch=81 0-based）
RUN_DIR = "/root/npi/outputs/gpfm-mamba-fmt-f0/v1fmt-main-01"   # channel_{mean,std}.npy（评估侧 z-score）
V1FMT_CONFIG_PATH = "/root/npi/models/gpfm-mamba-v1-fmt/config.py"   # importlib 唯一名 v1fmt_config
EXPECTED_CONFIG_HASH = "0dc0d5249c68"   # ckpt 内 config_hash pin（load_eval_model 严格校验）
SPLIT_MANIFEST = ("/root/npi/models/gpfm-mamba-v1-fmt/splits/"
                  "fold0_status_ok_seed42_val4.json")   # 装载前置断言用（防 fixate_split 写入分支）
EC_PERTURB_LEAF = "/root/npi/models/perturb-ec-v1/perturb.py"   # apply_gain 只读复用（纯函数，无 config 访问）

# ---- fmt 范式事实（v1-fmt config.py 已核，断言用）----
FS = 250.0          # 采样率（Nyquist 125 Hz；数据 1–40 Hz 带限预处理）
N_CHANNELS = 62     # 通道数
PAST_LEN = 384      # past 段样本数
FUTURE_LEN = 128    # future 段样本数

# ---- 窗集（design §2 钉死）：fold0 test 受试者 manifest 顺序前 3 例 × 每例数据集顺序前 8 窗 ----
TEST_SUBJECTS_HEAD = (10, 11, 12)   # fold0 test = (10,11,12,17,24,28,30,44,61,69) 的前 3 例
N_WINDOWS_PER_SUBJECT = 8           # 每例取数据集顺序前 8 窗（确定性，不随机）
N_WINDOWS = 3 * N_WINDOWS_PER_SUBJECT   # 24 窗
# win_idx 规则（钉死）= 受试者 manifest 顺序 × 每例前 8 窗固定枚举序：
#   例 #1（sid 10）窗 0–7 → win_idx 0–7；例 #2（sid 11）→ 8–15；例 #3（sid 12）→ 16–23。
# 两阶段窗枚举一致 ⇒ 同窗 eps/eta 跨阶段逐位相同（噪声级配对，硬断言）。

# ---- CRN 采样（design §2/§5 规模预算）----
K = 16              # CRN 样本数/窗
SEED = 2026         # eps/eta 生成种子：逐窗 torch.Generator.manual_seed(SEED + win_idx)，先 eps 后 eta

# ---- 输入端候选（design §3 表，6 类范式 × 每类一个代表；r0 预注册钉死）----
PARADIGMS = ("G", "N", "O", "P", "L", "B")
R0 = {"G": 0.25, "N": 0.5, "O": 1.0, "P": 3.0, "L": 0.5, "B": 1.0}
PARADIGM_NAMES = {
    "G": "乘性增益 x_A ← (1+r)·x_A（past 全段）",
    "N": "加性带限噪声 x_A ← x_A + r·σ_A·g(t)（g = 1–40 Hz 带限固定 seed 波形，RMS=1）",
    "O": "恒定偏置 x_A ← x_A + r·σ_A（全段常数，DC 平移）",
    "P": "瞬态脉冲 x_A ← x_A + r·σ_A·h(t)（h = 20 样本汉宁脉冲，中心 past 末端 −40 样本）",
    "L": "钳位损毁 x_A ← μ_A + (1−r)·(x_A − μ_A)，实现取代数形式 x_A − r·(x_A − μ_A)",
    "B": "带限增益 x_A ← x_A + r·BP_{8–13Hz}(x_A)（α 带分量增益 1+r，零相位）",
}

# ---- 算子实现参数（design §3 预注册钉死）----
NOISE_WAVEFORM_SEED = 2026    # N 的 g(t)：固定 seed FFT 域 1–40 Hz 掩码波形；带限后 RMS 重归一到 1；
                              # **全局共享单波形**（跨 (A, 窗) 不变）
NOISE_BAND = (1.0, 40.0)      # g(t) 的带限（Hz，rfft 掩码 |spec| 频点归属）
NOISE_LEN = PAST_LEN          # 波形长度 384 样本
PULSE_LEN = 20                # P 的 h(t)：20 样本（80 ms）汉宁脉冲（对称窗，端点为零）
PULSE_CENTER_OFFSET = 40      # 脉冲中心 = past 末端 −40 样本（起止样本 [384−40−10, 384−40+10) = [334, 354)）
ALPHA_BAND = (8.0, 13.0)      # B 算子带通与 M-env 包络带（α 带，Hz）
BP_BUTTER_ORDER = 3           # Butterworth 阶（契约「≤6 阶」内取 3：bandpass 总阶 2N=6）；sos + filtfilt 零相位
                              # （默认 padlen，384 样本下安全；scipy 1.18.1 已在 environment.md 登记）
SIGMA_STD_NOTE = "σ_A = 该窗源通道 past 段 std（torch .std(dim=-1)，默认无偏 ddof=1；z-score 后数据 σ_A≈1，仍逐窗实算）"

# ---- 输出端候选（design §4 表，6 类度量 × 每类一个代表）----
METRICS = ("M-RMS", "M-PSD", "M-band", "M-env", "M-FC", "M-time")

# ---- 度量实现参数（design §5 预注册钉死）----
WELCH_NPERSEG = 128           # M-PSD/M-band 共用 Welch：fs=250、nperseg=128、Hann 窗、50% overlap
WELCH_NOVERLAP = 64           # ⇒ bin 宽 1.95 Hz；detrend=scipy 默认 'constant'
PSD_EPS = 1e-12               # log10 前 ε=1e-12 地板（防 log(0)；PSD~1e-3 量级下不扭曲，仅护零）
PSD_BAND = (1.0, 40.0)        # M-PSD 谱距离频段（bin 中心落入带内为归属规则）
BANDS = (("delta", 1.0, 4.0), ("theta", 4.0, 8.0), ("alpha", 8.0, 13.0),
         ("beta", 13.0, 30.0), ("gamma", 30.0, 40.0))   # M-band 五带（bin 中心归属；δ 带 1–4 Hz 仅 ~1–2 bin、
                                                        # 方差大属已知照登）
MTIME_EARLY_SAMPLES = 32      # M-time 主标量 = |δ̄(t)| 前 32 样本（128 ms）均值；全段峰值与早期/全段比值照登

# ---- null 地板（design §5 预注册钉死）----
NULL_SPLIT = (8, 8)           # baseline 行 K=16 拆 8+8 两半：h1 = base[:8] 作 pert 侧、h2 = base[8:] 作 base 侧，
                              # 按各度量同构计算两半间距离（归一化统计 s_{w,B} / env 分母取 base 侧 = h2）
NULL_STD_DDOF = 1             # 逐边 24 个 null 值取 std（样本 std，ddof=1）

# ---- 聚合口径 ----
D_EDGE_AGG = "median"         # 逐边 24 窗 D 取 median（沿 perturb-ec-pipeline「受试者级聚合 = 窗 median」惯例）
GATE_EDGE_SET = "offdiag"     # Gate A/B 统计的边集 = offdiag 有向边 3782（evaluation-calibers EC 方向约定）

# ---- gates 阈值与 split-half 指派（design §5 预注册）----
GATE_A_Z = 2.0                # Gate A 敏感性：median_edges(z) ≥ 2（z≥2 边占比作登记项不作门）
GATE_B_RATIO = 2.0            # Gate B 特异性：median(D(A→A)) / median(D(A→B≠A)) ≥ 2（Spearman(D, 通道距离) 照登不作门）
GATE_C_RHO = 0.5              # Gate C 稳定性：跨受试者 split-half 边值秩相关 ρ ≥ 0.5
GATE_C_SPLIT = ("s1+s2", "s3")  # 预注册指派：例 #1+#2（16 窗）vs 例 #3（8 窗）；单划分 2v1 不均衡、无重采样属已知薄弱性

# ---- Gate D 单调性筛选（design §5 增补，用户 2026-10-18 拍板）----
# 机制：批构造扩 r 维（行 = 1 + 62 源 × Σ范式格点数；r0 本点并入网格不重复行；同批一次
# sample() ⇒ 跨 r 的 CRN 逐位配对 = 「仅改变 r」）。逐 {范式, 度量} 对计算 |Δ| 聚合曲线：
# 逐 r 先取 D_edge（24 窗 median，沿全 scorecard 惯例）→ offdiag 有向边池 median → 曲线；
# 曲线相邻步 100% 单调不减才过（GATE_D_STRICT=1.0，用户拍板「聚合曲线 100%」粒度）；
# 违反情形（步序、两端 r、曲线值、相对降幅）明细落盘。逐边单调占比为登记项不作门。
# Gate A/B/C 口径不变：一律取 r = r0 格点切片评（与原单强度预注册逐位一致）。
R_GRID_MULT = (0.25, 0.5, 1.0, 2.0, 4.0)   # 5 点倍率梯 ×各自 r0（用户拍板；1.0 = r0 本点）
R_CLAMP = {"G": 0.32, "L": 1.0}            # 逐算子上限钳制（用户拍板）：G = 外推警示界 0.32（前任务
                                           # design §2.1）、L = 1.0 语义界（r>1 变反相信号）；
                                           # N/O/P/B 无已登记语义界不钳（高倍率外推警示随注记照登）
R_GRID = {                                 # 上述规则展开的字面网格（钳制 + 升序去重；预注册钉死）
    "G": (0.0625, 0.125, 0.25, 0.32),      # {0.25,0.5,1,2,4}×0.25 → 0.5/1.0 钳 0.32 后去重
    "N": (0.125, 0.25, 0.5, 1.0, 2.0),
    "O": (0.25, 0.5, 1.0, 2.0, 4.0),
    "P": (0.75, 1.5, 3.0, 6.0, 12.0),
    "L": (0.125, 0.25, 0.5, 1.0),          # 2.0×0.5 钳 1.0 后与 1.0 重复 → 去重
    "B": (0.25, 0.5, 1.0, 2.0, 4.0),
}
GATE_D_STRICT = 1.0                        # 曲线相邻步单调占比 ≥ 1.0 才过（= 100% 无违反步）
GATE_D_RTOL = 1e-9                         # 「不减」相对容差：|Δ|_{k+1} < |Δ|_k·(1−rtol) 才记违反（浮点护栏）
GATE_D_AGG = "offdiag_median_curve"        # 聚合口径登记：D_edge(24 窗 median) → offdiag 边池 median
# r_2 恒等（用户 2026-10-18 口径确认）：全部范式 R_GRID[p].index(R0[p]) == 2（倍率梯中位
# mult=1.0；G/L 钳制去重后 r0 仍居格点序 2 = 去重格的上中位）——A/B/C 评估点 = 网格中位
# 格点 r_2 = r0 本点，无插值无额外行；单测 ⑧ 锁定。

# ---- 两阶段流程（design §6）----
PHASE1_PROBES = ("G", "N")          # 阶段一固定双探针（用户 2026-10-18 拍板；单探针备选已作废）
PHASE1_METRICS = METRICS            # 阶段一扫全部 6 个度量候选 → 选 M*
PHASE2_PARADIGMS = PARADIGMS        # 阶段二扫全部 6 个范式候选（G、N 再评入列）→ 选 P*
PHASE2_METRICS = ("M-RMS", "M-PSD", "M-time")
                                    # 阶段二度量集（**用户 2026-10-18 拍板**：阶段一如实阴性无单 M*，
                                    # 取 Gate A 最强 M-PSD / Gate B 唯一过线 M-time / 现行基准 M-RMS
                                    # 三度量继续——「单 M*」预注册规则的显式偏离，登记在案；
                                    # P* 逐度量独立判定，联合 {P×M} 终选以 3×6 矩阵呈报留用户拍板）
N_SRC = N_CHANNELS                  # 源通道数（fold0 62 通道全体；行 = 源）
PHASE1_GRID_POINTS = len(R_GRID["G"]) + len(R_GRID["N"])             # 9 = G 4 + N 5
PHASE2_GRID_POINTS = sum(len(R_GRID[p]) for p in PHASE2_PARADIGMS)   # 28 = 4+5+5+5+4+5
PHASE1_BATCH_ROWS = 1 + N_SRC * PHASE1_GRID_POINTS    # 559 = 1 baseline + 62 源 × 9 格点（G4+N5）
PHASE2_BATCH_ROWS = 1 + N_SRC * PHASE2_GRID_POINTS    # 1737 = 1 + 62 × 28
# 变体批行序（沿 perturb-ec-v1 build_variant_batch 同构扩 r 维）：行 0 = baseline；其后**源 A 主序、
# 范式次序、r 末序（R_GRID 升序）**，idx = 1 + a·N_RT + j（j = 该阶段 (范式, r) 平铺格点序）。
# Gate A/B/C 取 r0 格点切片；Gate D 用全网格（design §5/§6）。

# ---- 阶段间对照（design §2/§6）----
CROSS_PHASE_RTOL = 2e-6       # baseline 输出跨阶段紧容差 allclose（rtol；跨批形 cuBLAS/mamba 核无逐位保证，
                              # 逐位相等如实测得则照登，不作逐位门）

# ---- 可解释性同分决胜序（design §3/§4「一句话理由」派生，预注册；仅 Gate B 与 Gate C 同分时生效）----
METRIC_INTERP_ORDER = ("M-band", "M-env", "M-time", "M-RMS", "M-PSD", "M-FC")
#   M-band（EEG 频带可解释性强）> M-env（振幅耦合载体、连接文献常用）> M-time（判响应起点与时程）
#   > M-RMS（现行 D 同构基准）> M-PSD（谱域完整但抽象）> M-FC（网络态度量非纯靶响应，纯度 caveat）
PARADIGM_INTERP_ORDER = ("G", "N", "B", "L", "P", "O")
#   G（现行 A 案参照、神经科学标准干预）> N（perturb-and-record 标准形态）> B（频率选择性）
#   > L（损毁式、数值安全不外推）> P（瞬态可辨但剂量高）> O（DC 平移属外推输入、随附警示）

# ---- 产物根（artifacts-management.md；run_01 由 experiment.md Run 块登记）----
OUTPUTS_ROOT = "/root/npi/outputs/perturb-search-f0"
PHASE1_DIR = "phase1-metric-search/run_01"
PHASE2_DIR = "phase2-paradigm-search/run_01"
SMOKE_DIR = "smoke/run_01"
LOG_DIR = "logs"
