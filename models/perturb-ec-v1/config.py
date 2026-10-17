"""perturb-ec-v1 扰动侧口径常量（design §2 四要素预注册；10-17-perturb-ec-f0 任务唯一配置源）。

**方向约定（全仓库统一，evaluation-calibers.md）**：EC 矩阵**行 = 源 source（A）、列 = 靶
target（B）**；有向边展平非三角（62×61 = 3782）。本任务一切产物遵守该约定。

纪律（design §1）：
- 扰动侧参数一律放本模块，**绝不进 gpfm Config**（v4a-fmt config_hash 逐位不可变，
  ckpt 校验需原 hash 05d9bb875a2c）。
- 装载纪律：一律经 importlib 唯一名 `perturb_config`（perturb.load_config() 配方），
  **不占据 `config` 模块名**——该名留给 gpfm-mamba-v1（sample.py 顶层 import 消费）。
- 纯常量模块（非 dataclass，无需 build_config）；R_NEXT / 翻转偏移 / 斜率段一律由
  R_GRID 推导（禁字面量），档位调整权（evaluation-calibers 变更纪律）启用时自动一致。
"""

# ---- θ* 冻结定位（prd R1，只读；前置验证级，结论标注「流程验证·单模型·单折」）----
CKPT_PATH = ("/root/npi/outputs/gpfm-mamba-fmt-f0/v4afmt-main-01/checkpoints/"
             "gpfm-mamba-v4a-fmt-best.pt")   # best ep45（0-based epoch=44），EMA 权重
RUN_DIR = "/root/npi/outputs/gpfm-mamba-fmt-f0/v4afmt-main-01"   # channel_{mean,std}.npz（评估侧 z-score）
V4AFMT_CONFIG_PATH = "/root/npi/models/gpfm-mamba-v4a-fmt/config.py"   # 唯一名 v4afmt_config 装载
EXPECTED_CONFIG_HASH = "05d9bb875a2c"   # ckpt 内 config_hash pin（load_eval_model 严格校验）
SPLIT_MANIFEST = ("/root/npi/models/gpfm-mamba-v4a-fmt/splits/"
                  "fold0_status_ok_seed42_val4.json")   # 装载前置断言用（防 fixate_split 写入分支）
EXPECTED_TEST_IDS = (10, 11, 12, 17, 24, 28, 30, 44, 61, 69)   # fold0 test 折（与 fmt 任务逐 ID 一致）

# ---- r 网格（design §2.4：低剂量加密，9 点含 0；0.01 起对数等比 ×2，上界全幅增益 ~128%）----
R_GRID = (0.0, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28)
R_MAX = R_GRID[-1]                       # 网格上界（有效路径判定 r* ≤ R_MAX）
R_RATIO = R_GRID[-1] / R_GRID[-2]        # 对数等比（=2.0），由网格推导
R_NEXT = R_MAX * R_RATIO                 # 右删失编码 r_max^+（=2.56）——由 R_GRID 推导，禁字面量；
                                         # 亦 = rstar 翻转偏移（strength = R_NEXT − r*）
PERT_R_GRID = R_GRID[1:]                 # 采样档（8 档）：r=0 由 CRN 恒等按构造取 D=0，不采样
N_R_PERT = len(PERT_R_GRID)
N_SRC = 62                               # 源通道数（fold0 62 通道全体；行=源）
N_VARIANTS = 1 + N_SRC * N_R_PERT        # 497 = baseline + 全 (源A, 档r) 并批（design §2.2）

# ---- 低剂量斜率段（design §2.4：主段 5 点含原点；敏感性段 4 点；索引于 R_GRID 前缀）----
SLOPE_SEG_IDX = 5                         # R_GRID[:5] = {0, 0.01, 0.02, 0.04, 0.08}
SLOPE_SENS_IDX = 4                        # R_GRID[:4] = {0, 0.01, 0.02, 0.04}
SLOPE_SEG = R_GRID[:SLOPE_SEG_IDX]
SLOPE_SEG_SENS = R_GRID[:SLOPE_SENS_IDX]

# ---- CRN 采样与窗子采样（design §2.2/§3；与训练 seed 1234 / val eps seed 2026 独立声明）----
K = 8                    # CRN 样本数/窗（null 对数 K(K−1)/2 = 28 对/窗/通道；注册/评估 K=16 差异登记）
N_W = 32                 # 窗/例（确定性子采样；930 窗全集 vs 子采样代表性诊断见 design §6/§7-7）
SEED = 2026              # 扰动侧统一 seed：eps/eta 生成（逐窗 SEED+窗序号）+ 窗子采样（逐例独立）

# ---- τ 响应阈值分位（design §2.3：null 逐被试分池；主口径 95%，90/99 敏感性登记）----
TAU_QUANTILES = (0.90, 0.95, 0.99)
TAU_MAIN = 0.95

# ---- τ v2 操作口径（design §2.3 v2，用户 2026-10-17 拍板；Run 004 readout-01 首次执行）----
# 操作性 τ = 相对自响应比 D̃_{A→B}(r) = D̄_{A→B}(r)/D̄_{B→B}(r) 上的阈值 τ̃（主 0.25）；
# 敏感性 0.50。绝对口径（null 分池分位）降为材质性观察照登（上方 TAU_QUANTILES/TAU_MAIN）。
TAU_TILDE_MAIN = 0.25
TAU_TILDE_SENS = 0.50

# ---- 比较器契约（design §5 / evaluation-calibers Study II 节）----
ORIENTATION = "larger_is_stronger"   # slope/auc 天然大=强免翻转；rstar 须调用方先翻转再存

# ---- 产物根（artifacts-management.md；run-id 由 experiment.md 各 Run 块登记）----
OUTPUTS_ROOT = "/root/npi/outputs/perturb-ec-f0"
