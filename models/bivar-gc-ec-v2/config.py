"""bivar-gc-ec-v2 config —— 全部口径常量（design.md §1/§2 代码化；改动=口径变更须登记）。

装载纪律：本文件**不得**被直接 `import config`（不占据 config 模块名）——一律经
importlib 唯一名 `bgec2_config` 四行配方装载（先注册 sys.modules 后 exec，见
data_io.py / run.py 内联配方；reproducibility.md 惯例）。

定位：observational directed-connectivity reference 双变量口径的**原始采样率档**
（250 Hz 原始全长：不降采、不切窗），与 125 Hz 首版 `models/bivar-gc-ec-v1/`
平行，产物与文档显式区分、绝不混用（prd.md R4；outputs 根亦严格隔离）。
"""

from pathlib import Path

REPO_ROOT = Path("/root/npi")
DATA_ROOT = Path("/root/autodl-tmp/edra_processed_fmt")   # edra-fmt 契约（spec/data/edra-fmt.md）
OUTPUTS_ROOT = REPO_ROOT / "outputs" / "bivar-gc-ec-fmt-250"

# ---- 数据口径（250 Hz 原始全长；design §2.1——与 v1 唯一数据差异 = 无降采）----
FS_RAW = 250                    # fmt 数据契约采样率（逐例断言）
FS = 250                        # 拟合采样率 = 原始（无 resample 步）
N_CHANNELS = 62
N_RAW = 60001                   # fmt 契约长度（240.004 s），原样进入拟合
EXPECTED_FIELDS = {"filter": "1-40(release)", "pipeline": "EDRA_release_1-40_250Hz"}

# ---- 双变量 GC 口径（design §2.2；一经 refs-01 定版不得中途换）----
P_MAX = 48                      # BIC 候选网格 p∈{1..48}：保物理滞后 192 ms（125 Hz@24 ≡ 250 Hz@48）
                                # 行集对齐惯例：全候选阶共用 row_from=P_MAX（嵌套行集）

# ---- split-half 噪声天花板（design §2.3，段式同 v1、段长随 fs 翻倍保 2 s）----
SEG_S = 2.0                     # 奇偶交错分段长度（秒）
SEG_SAMPLES = int(SEG_S * FS)   # 500（2 s @ 250 Hz；v2 split 本地编排显式传参，勿落 v1 缺省 250）

# ---- 运行（environment.md：cgroup 有效 16 核；BLAS 护栏五变量 setdefault）----
N_PROC = 16                     # main/reverse/split 并行 worker 数（spawn + worker 单线程 BLAS）
BLAS_THREADS_DEFAULT = "8"      # 入口护栏默认；可被外部环境变量覆盖
SEED = 2026                     # RNG 惯例（单测 rng seed）

# ---- 受试者清单（manifest 48 例全 status=ok，与 v1 同清单；装载时逐例复核）----
EXPECTED_IDS = (1, 4, 5, 7, 8, 10, 11, 12, 13, 17, 18, 19, 20, 21, 22, 24, 25, 26,
                27, 28, 29, 30, 35, 36, 37, 38, 39, 40, 43, 44, 46, 47, 48, 49, 50,
                52, 53, 55, 58, 59, 60, 61, 62, 64, 65, 67, 68, 69)
