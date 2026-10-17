"""bivar-gc-ec-v1 config —— 全部口径常量（design.md §1/§2 代码化；改动=口径变更须登记）。

装载纪律：本文件**不得**被直接 `import config`（不占据 config 模块名）——一律经
importlib 唯一名 `bgec_config` 四行配方装载（先注册 sys.modules 后 exec，见
data_io.py / split_half.py / run.py 内联配方；reproducibility.md 惯例）。

定位：observational directed-connectivity reference 的**双变量口径**（pairwise
bivariate GC），与 baseline-ec-v1 全模型条件 GC 为不同口径，产物与文档显式区分、
不得混用（prd.md 任务配置第 3 条）。
"""

from pathlib import Path

REPO_ROOT = Path("/root/npi")
DATA_ROOT = Path("/root/autodl-tmp/edra_processed_fmt")   # edra-fmt 契约（spec/data/edra-fmt.md）
OUTPUTS_ROOT = REPO_ROOT / "outputs" / "bivar-gc-ec-fmt"

# ---- 数据口径（与 refs_fmt.prepare_subject 逐式同构；design §2.1）----
FS_RAW = 250                    # fmt 数据契约采样率（逐例断言）
FS = 125                        # 目标采样率（resample_poly 抗混叠降采）
RESAMPLE_UP, RESAMPLE_DOWN = 1, 2   # 250→125 Hz（release 已带限 1–40 Hz，不重复滤波）
N_CHANNELS = 62
N_RAW = 60001                   # fmt 契约长度（240.004 s）→ 降采后 30001
N_125 = 30001                   # resample_poly(x,1,2) 契约输出长度（smoke 断言用）
EXPECTED_FIELDS = {"filter": "1-40(release)", "pipeline": "EDRA_release_1-40_250Hz"}

# ---- 双变量 GC 口径（design §2.2；一经 refs-01 定版不得中途换）----
P_MAX = 24                      # BIC 候选网格 p∈{1..24}（对齐 baseline BIC 惯例）
                                # 行集对齐惯例：全候选阶共用 row_from=P_MAX（嵌套行集）

# ---- split-half 噪声天花板（design §2.3，复刻 baseline 段式）----
SEG_S = 2.0                     # 奇偶交错分段长度（秒）
SEG_SAMPLES = int(SEG_S * FS)   # 250

# ---- 运行（environment.md：cgroup 有效 16 核；BLAS 护栏五变量 setdefault）----
N_PROC = 16                     # main/split 并行 worker 数（spawn + worker 单线程 BLAS）
BLAS_THREADS_DEFAULT = "8"      # 入口护栏默认；可被外部环境变量覆盖
SEED = 2026                     # RNG 惯例（单测 rng seed）

# ---- 受试者清单（manifest 48 例全 status=ok，2026-10-17 核对；装载时逐例复核）----
EXPECTED_IDS = (1, 4, 5, 7, 8, 10, 11, 12, 13, 17, 18, 19, 20, 21, 22, 24, 25, 26,
                27, 28, 29, 30, 35, 36, 37, 38, 39, 40, 43, 44, 46, 47, 48, 49, 50,
                52, 53, 55, 58, 59, 60, 61, 62, 64, 65, 67, 68, 69)
