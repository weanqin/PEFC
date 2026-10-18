# Research: latent_flow 移植进 /root/npi 的适配研究

- Query: 为「把 /root/autodl-tmp/latent_flow 移植进 /root/npi 作为新的目标实现路径」做适配研究（workflow 1.0c 新路径触发）
- Scope: mixed（latent_flow 内部考古 + npi 侧惯例/契约对照 + 本机环境实测）
- Date: 2026-10-18
- 源库：`/root/autodl-tmp/latent_flow`（7z 解压，**全程只读未改动**；本研究的测试验证经 /tmp 影子沙箱完成）

---

## 0. 一页结论（TL;DR）

| 问题 | 结论 |
|---|---|
| 落位形态 | **`models/latent-flow-v1/latent_flow/` 嵌套包**（零源码改动可导入；运行 `cd models/latent-flow-v1 && python -m latent_flow.scripts.<x>`） |
| 基线 | **内容取工作树（= HEAD `9efe1e3` + 6 个真实改动文件），git 锚点取 HEAD**；166 个"修改"里 160 个是纯行尾差异，未提交实验全部"完成"带预注册判决，非半成品 |
| 前置恢复 | `data/` 9 个加载器文件被 7z 删失，**必须 `git checkout HEAD -- data/`**（已验证：恢复前 302 测试收集全挂，恢复后 293/302 通过） |
| 环境适配 | **零源码改动可跑**：Python 3.12/torch 2.8/无 3.13-only API/无 torch 2.9+ 符号/Agg 已统一/device 自动；改动集中在包装层（BLAS 守卫、outputs 落点）与 data root env var |
| 数据接口 | EDRA：旧库吃 BrainVision .vhdr（本机 0 个），新契约是 npz —— 需写 shim（本任务只声明边界）；MODMA zip 在位但**当前读取异常**（见 §4.4） |
| 测试 | 本机沙箱实测 **293 passed / 4 failed（全是缺 MODMA 数据文件）/ 5 skipped（既有 TODO + 缺 geometry.pt）/ 30.5s** |
| 裁剪 | OC 活跃面与 de/dysib 阴性线物理可分（`__init__.py` 不 import de/dysib；OC 归因入口不依赖 de 系）；全量 170 py / 45,582 行（另 archive 8,127 行不参与运行） |

---

## 1. 落位形态

### 1.1 两侧惯例的实测事实

**npi 惯例**（证据：`/root/npi/models/` 18 个变体目录 + `.trellis/spec/research/path-isolation.md`）：

- `find /root/npi/models -maxdepth 2 -name "__init__.py"` → **0 个**：平铺模块、无包结构。
- 入口直跑 + `sys.path.insert` 本包目录后 `import config`：实例 `models/perturb-ec-v1/loader.py:22-25`（同时插 v1 目录与自身目录，遮蔽式复用基线）。
- 契约要求（path-isolation.md）：一条路径 = 一个目录（代码 `models/<方法名>-<变体名>/`、配置随目录、输出 `outputs/<变体名>/`）；新变体新目录；禁止跨变体读中间产物。

**latent_flow 事实**（证据：README.md + grep 实测）：

- 自包含包：根 `__init__.py` re-export OCLatentFlow/MODMADataset/BaseTrainer/MetricAggregator/Visualizer + config 常量。
- 绝对导入 `from latent_flow.xxx`：**130 个 .py 文件**使用（`grep -rln "from latent_flow\.\|import latent_flow\." --include="*.py" | wc -l`）。
- 运行方式（README「Running」节原文）："Run modules from the **parent** of this directory … `python -m latent_flow.scripts.train_oc_latent_flow`"，并明确 "the package directory uses an underscore (`latent_flow`), not a hyphen — a hyphen would make it non-importable"。
- 路径耦合点收敛在 `config.py`：`_PROJECT_ROOT = Path(__file__).resolve().parent`（config.py:22），唯一派生 `CHECKPOINT_DIR`（config.py:26）；数据根默认包内 `data/` 但**全部可用 env var 覆盖**（MODMA_DATA_ROOT / EDRA_DATA_ROOT / SC_DATA_ROOT / MODMA_DEVICE，README「Configuration」节）。输出目录经 CLI 参数传递（`scripts/_connectivity_driver.py:406-410` 构造函数入参 `output_dir, outputs_dir`），不硬编码。
- `_PROJECT_ROOT` 的其他 8 个消费文件全在 dysib 线（`grep -rln _PROJECT_ROOT`：config.py + 5 个 dysib 脚本 + train_dysib{,v22}.py）——OC 活跃面无直接依赖。

### 1.2 方案评估

| 方案 | 做法 | 源码改动量 | 评估 |
|---|---|---|---|
| A. `models/latent-flow-v1/` 直接作包根 | 目录名即包名 | **不可行**：连字符目录不可导入；要嘛改 130 个文件的导入前缀，要嘛目录改用下划线破坏 npi 命名惯例（`models/latent_flow-v1/` 与 `<方法名>-<变体名>` 惯例不符且历史上无先例） | 排除 |
| **B. `models/latent-flow-v1/latent_flow/` 嵌套（推荐）** | 变体目录内放完整 `latent_flow` 包 | **0 行**：包内所有 `latent_flow.xxx` 绝对导入原样成立；`python -m latent_flow.scripts.<x>` 从 `models/latent-flow-v1/` 目录运行（cwd 进 sys.path），或写 npi 风格 `run.py` 包装（`sys.path.insert(0, 变体目录)` 后 `import latent_flow`） | 变体目录承载 README/run 包装/与 npi 的 outputs 挂接；`_PROJECT_ROOT` 自动 = 包根，checkpoints 落包内（npi `.gitignore` 已含 `*.pt`，不入库） |
| C. 平铺拆散进 `models/latent-flow-v1/` 直属文件 | 按 npi 惯例把 oc_latent_flow.py 等摊平、改 `import config` | ~130 文件导入重写 + 子包（data/training/viz/attribution/baseline/analysis）扁平化 | 违背「源码零改动/最小改动」目标，且丢掉 `python -m` 可运行性；仅当后续要深度融入 npi 变体体系时才值得 |

**方案 B 与 npi 惯例的相容性说明**：path-isolation.md 只约束「目录隔离 + 配置随目录 + outputs 命名 + 不跨变体写」，不约束 import 风格；npi 侧已有先例是跨目录 sys.path 复用（perturb-ec-v1/loader.py 插 gpfm-mamba-v1）。嵌套包不与任何既有变体重名/交叠。唯一需要在 design.md 声明的偏离：该路径入口不是「直跑本目录 py」而是 `python -m`（或 run.py 包装转译）。

---

## 2. 基线选择

### 2.1 三个候选的实测状态

| 候选 | commit / 日期 | 内容 |
|---|---|---|
| main | `ff4b8b4` 2026-08-11 | 无 de/dysib 探针线（de 线、fm_gate 线均不存在） |
| HEAD（当前检出） | `9efe1e3` 2026-08-19（分支 08-15-substrate-cross-channel-probe） | vs main：**+204 文件 / +44,162 行**（de/dysib 探针线 + spec）；README/FOUNDATION 所述 OC 成果均在 |
| 工作树 | mtime 至 2026-08-25 | HEAD + 未提交改动（下述） |

### 2.2 未提交改动的真实性质（关键证据）

`git status --porcelain`：166 M + 48 ?? + 645 D。逐类核实：

**(a) 645 个删除 = 7z 打包时被排除的目录，不是代码删除**：按顶层目录统计——`.trellis` 337、`.zcode` 183、`.claude` 52、`.agents` 52、`.codex` 8（AI 工具链）、**`data/` 9 个加载器 .py**（`__init__.py, brainvision.py, datasets.py, de_features.py, electrode_positions.py, modma.py, norm_stats.py, splitter.py, subject_classification.py`，`git ls-tree HEAD data/` 全部可恢复）、`oc_latent_flow_arch/` 3、`karpathy-guidelines` 1。

**(b) 166 个修改中只有 6 个是真实内容改动**（`git diff -w --numstat HEAD --diff-filter=M` 过滤零行差）：

| 文件 | 真实 diff | 性质 |
|---|---|---|
| `models/dysib.py` | +606/-1 | 追加 DYSIB v2.2 paper-faithful 窗集合类（additive，文件头声明 "purely additive"） |
| `tests/test_dysib.py` | +441/-0 | v2.2 配套测试（本次沙箱实测通过） |
| `training/trainer.py` | +6/-6 | checkpoint 载入泛化：`format_version==1` 判断放宽为 `"state_dict" in payload`（v1/v2 checkpoint seam 兼容） |
| `analysis/fm_gate/probe_matrix.py` | +16/-4 | probe matrix 小改 |
| `experiments/DySIB-EEG.md` | +149 | v2.2 设计文档 |
| `experiments/.obsidian/workspace.json` | 编辑器状态 | 忽略 |

其余 160 个文件全是**行尾（空白）差异**（`git diff -w --stat` 对 `models/oc_latent_flow.py`、`attribution/bisect_sigma.py`、`config.py`、`__init__.py` 等输出为空）——即 OC 核心代码工作树与 HEAD **逐字相同**。

**(c) 48 个未跟踪**：`results/` 下 npz/png（可再生）、三份实验笔记（`experiments/08-19-dysib-01-v22.md`、`08-23-task-snr-scan.md`、`08-24-band-conditional-gain.md`）、dysib v22 脚本/训练/测试 5 个 py、一份论文 PDF。

**(d) 未提交实验全部完结，非半成品**（三份笔记 frontmatter `status` 字段）：
- 08-19 dysib v22：**完成（verdict ZERO_FACE，预注册路由 = DySIB 线终止归档）**
- 08-23 task-snr-scan：**完成（verdict BR-BAND）**
- 08-24 band-conditional-gain：**完成（verdict CB-REDUNDANCY-DEEP，band-split 路线冻结）**

### 2.3 推荐

**git 基线 = HEAD `9efe1e3`；内容基线 = 工作树**（两者对 OC 活跃面逐字一致；工作树多出的 dysib v22 线是自带测试的完结代码）。main 排除（缺 de/dysib 线，而该线的阴性判决文档有归档价值）。入库时注意：7z 工作树是 LF、git HEAD blob 是 CRLF——**入库前应统一行尾并加 `.gitattributes`（`*.py text eol=lf`）**，否则 160 文件的伪 diff 会原样复现污染 npi 仓库历史。

---

## 3. 环境适配点清单

对照 `/root/npi/.trellis/research/environment.md`（Python 3.12.3 / torch 2.8.0+cu128 / RTX 4080 SUPER / 有效 16 核 / outputs 根 `/root/npi/outputs/<任务名>/`）。

### 3.1 兼容性实测/排查结果

| 项 | 证据 | 结论 |
|---|---|---|
| Python 3.13→3.12 | grep `TypeIs\|ReadOnly\|Path.from_uri\|except\*` 于全部活跃 py → 0 命中；`from __future__ import annotations` 已普遍使用 | 无 3.13-only 依赖 |
| torch 2.12 nightly→2.8.0 | requirements.txt 自声明 floor `torch>=2.6`；grep 无 `torch.compile`/`torch.compiler`/flash/scaled_dot_product；`torch.load(..., weights_only=False)`（trainer.py:369,383 等 5 处）在 2.8 可用；`torch.amp.autocast`/`torch.utils.checkpoint`/cudnn 均旧 API | **无版本敏感 API** |
| GPU sm_120→sm_89 | 旧机 5070Ti 需 nightly；4080 SUPER 在 cu128 stable 支持列 | 降风险方向，无适配 |
| numpy 2.x | `baseline/granger.py:173` 已写 `_trapz = getattr(np, "trapezoid", np.trapz)` 兼容层 | 已处理 |
| 依赖版本对照 | npi 实测：torch 2.8.0 ✓(≥2.6)、numpy 2.3.2 ✓(≥2.0)、scipy 1.18.1 ✓(≥1.13)、sklearn 1.9.1 ✓(≥1.5)、mne 1.13.2 ✓(≥1.7)、matplotlib 3.10.5 ✓(≥3.9)、tqdm 4.66.2（requirements 注记开发机 4.69，floor 是 ≥4.66）✓、joblib 1.6.0 ✓、dateutil ✓、pytest 9.1.1 ✓(≥8) | **全部满足，无需装包** |
| numba | README「Required packages」提到 numba，但 `grep -rln numba --include="*.py"` → 0 命中（README 陈旧）；本机未装 | 无碍 |
| matplotlib 后端 | `viz/visualizer.py:8`、`viz/connectivity.py:26`、`scripts/diag_oc_latent_flow.py:31` 等统一 `matplotlib.use("Agg")` | headless 安全 |
| device | `config.py:46-50`：`MODMA_DEVICE` env 覆盖，否则 `cuda if available`；本机 cuda=True | 直接可用 |
| BLAS 16 核守卫 | latent_flow **无** `_blas_guard` 惯例；baseline 有 joblib 并行（`granger.py:282` loky backend、`gpdc.py:165` threading backend，`n_jobs` CLI 参数）；`analysis/fm_gate/probe_matrix.py:419` 与 dysib 评估脚本有 `torch.set_num_threads` | **包装层适配点**（见 §3.2） |

### 3.2 Windows 残留清单（全部为无害或低危）

| 残留 | 位置 | 处置建议 |
|---|---|---|
| `sys.stdout.reconfigure(encoding="utf-8")  # gbk safety on Windows` | ~10+ 个 scripts（eval_de_vae.py:29 等） | Linux 无害，可保留（零改动优先） |
| `D:\program\anaconda\...` / `D:\102-npi` 硬编码路径 | **仅 archive/**（docstring 示例）与部分测试 docstring | archive 不随移则自动消失 |
| `KMP_DUPLICATE_LIB_OK` | 仅 archive/ 5 个脚本 | 同上 |
| `open()` 不带 encoding | scripts/ ~10 处（JSON 读写） | Linux 默认 UTF-8 无害；仅在写含非 ASCII 的 JSON 时理论上有差异，保持现状 |
| `.obsidian` nutstore-sync 插件（坚果云同步） | `experiments/.obsidian/plugins/nutstore-sync/` | 不随移 |
| README「Windows Gotcha」节（conda run gbk 崩溃） | 文档 | 移植后加一段 Linux 运行说明即可 |

### 3.3 需要动的具体文件与改动性质

**零改动（env var / CLI 参数即可）**：
- 数据根：`MODMA_DATA_ROOT` / `EDRA_DATA_ROOT` / `SC_DATA_ROOT` → 指向 `/root/autodl-tmp/...`（config.py:24,33,40 均先查 env）。
- device：默认自动；如需指定 `MODMA_DEVICE=cuda:0`。
- 输出目录：scripts 均有 `--output-dir` 类参数（`_connectivity_driver.py` 实证）→ 指向 `/root/npi/outputs/<任务名>/...`。

**包装层新增（不改 latent_flow 源码）**：
- 入口 BLAS 守卫：按 npi 惯例（`_blas_guard.py`，OMP/OPENBLAS 等 setdefault，numpy 导入前）写变体目录级 `run.py` 包装或 sitecustomize 式前置；同时 baseline 的 `--n_jobs` 上限按 16 核口径给默认值。
- `python -m` 运行说明 / run.py 包装（sys.path.insert 变体目录）。

**可选源码微改（design 决定）**：
- `CHECKPOINT_DIR`（config.py:26）如要迁出包内 → 一行改动或保持包内（`*.pt` 已被 npi .gitignore 覆盖）。
- 删除 scripts 中 gbk reconfigure 行（纯卫生，非必需）。

---

## 4. 与 Study I/II 的接口

### 4.1 数据接口差距

**旧库数据契约**（git HEAD `data/datasets.py` `_REGISTRY` + `data/modma.py` 模块 docstring）：

| 维度 | MODMA | EDRA（旧库） |
|---|---|---|
| 格式 | `.mat`（键 `data`，(128, ~42501) float32） | **BrainVision `.vhdr+.dat`**（62ch，60000 样本） |
| 采样率 | 250 Hz（config `FS=250`） | 250 Hz |
| 分组 | `HC/` `MDD/` 子目录 | 平铺 `H_*.vhdr` / `M_*.vhdr` 前缀 |
| 归一化 | 默认逐被试 z-score；或 `norm_stats` 训练折统计（held-out 纪律已有实现，`modma.py::_load_subject`） | 同左 |
| 窗口 | 2s=500 样本 = obs 375 + fc 125，stride 250（config `WINDOW_SAMPLES/OBS_SAMPLES/STEP_SAMPLES`） | 同左 |
| geometry | `DATA_ROOT/geometry.pt`（GSN-HydroCel-129） | `EDRA_DATA_ROOT/geometry_edra.pt`（10-20，`scripts/build_edra_geometry.py` 生成，mne 优先 + vendored 坐标表回退） |

**npi 现行契约**（`.trellis/spec/data/edra.md`、`edra-fmt.md`、`research/data-pipeline.md`）：

- edra（v4 管线）：受试者级 npz `data (62, 85000)` @ **500 Hz**，HC24/MDD22，manifest+QC；edra-fmt（release 转换）：npz `data (62, 60001)` @ **250 Hz**，48 例，`chan_coords_62.npy` / `chan_forward_62.fif` 随数据集根。
- 窗口：`[past=384, future=128]`、stride 64、切分与归一化**全部在 dataloader 内**（模型侧零变换），5-fold 受试者级分层 + fold0 内层 32/4/10（v4 系口径）。

**差距量化（EDRA 为例）**：文件格式（.vhdr → npz）、通道蒙太奇一致（62ch 10-20，坐标文件在位）、采样率（500 或 250 vs 旧库 250——若走 edra-fmt 则一致）、窗长（375/125 vs 384/128，语义近但样本数不等）、分组发现（vhdr glob → npz 目录+manifest）、归一化口径（旧库两模式齐备，但默认逐被试）。**结论：跑 EDRA 需要一个 npz→`MODMADataset` 兼容 shim（新增 `file_format="npz"` 的 Descriptor + reader）+ 用 `chan_coords_62.npy` 生成 `geometry_edra.pt`**（`build_edra_geometry.py` 现实现会断言一个真实 .vhdr 通道表，无 .vhdr 时需小改或换输入）。本任务只移植不融合，design.md 只需声明该边界。

### 4.2 bisection σ 与 perturb-ec-v1 四要素的方法学对照

（旧库证据：`attribution/bisect_sigma.py:1-80` 模块 docstring + 常量节；npi 侧：`.trellis/spec/research/perturb-ec-pipeline.md`）

| 要素 | latent_flow bisect_sigma | npi perturb-ec-v1 | 差异性质 |
|---|---|---|---|
| 扰动算子 (P) | 源通道输入加性高斯 `N(0, σ²)` | 增益 `x_A ← (1+r)·x_A`（past 全段，r≥0 无量纲） | **算子类不同**（加性噪声 vs 乘性增益）；同为「剂量-响应」搜索 |
| 响应度量 (D) | 逐靶 `MSE_b(σ)`（对无扰动 forecast 的偏差） | CRN 配对差 `RMS_{k,t}(δ)/s_w` | 度量统计量不同（MSE vs 归一化 RMS） |
| 确定性机制 | 依赖 OC stage-2 路径确定性（`z_obs = μ_obs` 无重参数化，docstring "perturbed-vs-baseline deltas are clean"）——**无显式 CRN** | 显式 CRN：全部变体并入同一 `sample()` 调用同批、跨变体共享 eps/eta，D(r=0) 逐位恒等 | **目标相同、机制不同**；换噪声采样路径（如 OC flow Euler）时 npi 契约的「同批+共享噪声」仍适用，需在 design 声明等价实现 |
| 阈值 (τ) | `τ = α · median_b(MSE_b(σ_max))`，**单一共享阈**（中位数防超敏靶污染——docstring 记载了 per-target 阈值翻车史：2000x 响应展布导致强连接被判弱） | τ̃ 相对自响应 `D̄_{A→B}/D̄_{B→B}` 阈 0.25（敏感 0.50）+ 绝对口径 null 分位照登 | 归一化基准不同（对源内响应分布的中位数 vs 对靶自响应的比值） |
| readout | `σ*^a[b] = max σ s.t. MSE_b(σ) ≤ τ`（**临界扰动强度**；连接强度 = `-(log σ* + 0.5·log(2πe))` 微分熵式，**大=强**）；forward-only 网格 24 前向 + 插值，无梯度 | 主=低剂量斜率、次=AUC、降级=r*（右删失 + `R_NEXT−r*` 翻转，`larger_is_stronger` 契约 npz） | **σ* 即本研究 r* 概念的先前实现**；方向约定最终同向（旧库公式自带翻转使大=强，npi 契约 orientation 一致，可直接产 `ec_pert_*` 契约 npz 供 `baseline-ec-v1/compare.py` 消费——属后续任务） |
| 方向语义 | `connectivity[a,b]`：a=源、b=靶 | EC 矩阵行=源、列=靶 | 一致 |

### 4.3 旧库数据集在 npi 机器的在位检查（2026-10-18 实测）

| 项 | 状态 |
|---|---|
| `/root/autodl-tmp/EDRA_processed/`（635M，release .mat @250Hz） | 存在、可读 |
| `/root/autodl-tmp/edra_processed_fmt/`（636M，npz 48 例） | 存在、可读（npi 现行 fmt 契约） |
| `/root/autodl-tmp/edra_process_v4/`（1017M，npz @500Hz） | 存在、可读（npi 现行 edra 契约） |
| 全盘 `.vhdr` | **0 个**（`find /root/autodl-tmp -name "*.vhdr" | wc -l`）→ 旧库 BrainVision 加载器在本机无原生数据可吃 |
| `chan_coords_62.npy` / `chan_forward_62.fif` | 在 edra_process_v4 与 edra_processed_fmt 根目录（geometry 生成的可用输入） |
| `/root/autodl-tmp/modda_resting_process.zip`（1.07GB，53 个 .mat，HC/MDD 子目录） | **异常**：目录项可见（`find -ls` 显示 inode 与 1,065,675,704 字节），但 `open()`/`head` 一律 ENOENT，3 次重试 + 对照组（EDRA_processed.zip 可读 PK 头）确认为该文件自身的可见性/句柄异常（xfs 本地盘，`.autodl` 守护进程活动期）；**恢复可读前 MODMA 线不可跑**，EDRA 线不受影响 |

### 4.4 Study I/II 边界声明素材（供 design.md 引用）

- 旧库已有同框架要素的先前实现：两阶段代理模型（对应 Study I 的 X_past→X_future）、σ 二分（对应 Study II 四要素的 P/D/τ/readout 组合）、GC/TE 基线对比（`baseline/`：Granger 自实现 ridge 版、symbolic TE Bandt-Pompe numpy 自实现、gPDC——**无 idtxl/statsmodels 依赖**，与 npi「TE 全系取消、statsmodels 未装」不冲突）。
- FOUNDATION.md 记录的旧库定量基线（移植后可比对）：跨被试 Pearson 0.172(proj)/0.157(flow)；Mexican-hat mixer 0.449→0.747；DE vs GC r_abs=0.585(MODMA fold2)/0.413(EDRA)；offdiag CV 128%、距离衰减 r=0.842。
- npi 侧既有判据（surrogate-brain-qualification.md 指标 1/3 通过标准、perturb-ec-pipeline.md 的 CRN 契约与 readout 三件套）在旧库无对应物——**移植 ≠ 合格**，θ* 资格须按 npi 规范重新验收。

---

## 5. 测试面

**沙箱实测**（/tmp/lf_check：全树符号链接 + 仅 `data/` 用 `git show HEAD:` 物化，源树零改动；Python 3.12.3 / torch 2.8.0+cu128 / pytest 9.1.1 / BLAS 限 8 线程）：

| 步骤 | 结果 |
|---|---|
| 恢复 data/ 前收集 | **22 errors**——`import latent_flow` 即炸：`__init__.py:16 from latent_flow.data.modma import ...` → `ModuleNotFoundError: No module named 'latent_flow.data'`（全量收集 0 项） |
| 恢复 data/ 9 文件后收集 | **302 项**（tests/ 23 文件、284 个 `def test_` + 参数化；无 conftest.py/pytest.ini/pyproject） |
| 全量运行 | **293 passed / 4 failed / 5 skipped，30.5s** |
| 4 失败 | 全部 `test_subject_classification.py`（`subject_classification.py:336` FileNotFoundError：逐文件路径解析 HC/MDD 标签需 MODMA 数据集文件在 `DATA_ROOT/HC|MDD` 在位）——**数据可用性依赖，非环境兼容性** |
| 5 跳过 | 3 = `test_coord_audit.py`（"run_beta_sweep not yet implemented (Phase 4)"，上游既有 TODO）；2 = `test_graph_utils.py:103,115`（"geometry.pt not available"，数据集工件缺失） |

注：README/FOUNDATION 的「95 个单元测试」是 2026-08-01 时点口径；当前实为 302 收集项。数据集文件（MODMA mat / geometry.pt）补齐后，预期仅剩 3 个既有 TODO 跳过。

---

## 6. 风险

### 6.1 未提交状态的半成品性质 → **低风险，已排除**

见 §2.2：OC 核心工作树 ≡ HEAD（仅行尾）；真实改动集中在 dysib v2.2 完结线（自带 441 行测试且通过）；三份未提交实验笔记全部「完成 + 预注册判决」。唯一"坏"状态是 data/ 删失与行尾漂移，均有明确处置（git 恢复 + .gitattributes）。

### 6.2 de/dysib 阴性线随移与否（全量 vs 裁剪）

事实基础：
- README「Script Status」：active 脚本仅 OC 簇 7 个（train/eval/diag/plot/plot_stage2/shape_check/probe + `run_oc_connectivity_attribution`）+ baseline 组；ADR-0004 已归档 JET/legacy 六线。
- 但 `models/` 6 文件中 `de_vae.py / dysib.py / dcgnn_classifier.py`、`analysis/de`、`analysis/fm_gate`、scripts 中 ~30 个 de/dysib/fm_gate 脚本属阴性/终止线（de 线 layerwise/probe 判决、dysib 线 ZERO_FACE 终止、DCGNN 分类阴性）。
- **物理可分性已验证**：`__init__.py` 只 import oc_latent_flow/data/trainer/metrics/visualizer/config（不 import de/dysib）；OC 归因入口 `scripts/_connectivity_driver.py` 只依赖 bisect_sigma/config/data/splitter（不 import de 系）。裁剪不破包根。
- 交叉点：`tests/test_de_vae.py` 反向 import oc_latent_flow（裁剪时按测试归属走）；`models/graph_utils.py` 为 OC 与 de 线共享（保留）。
- 体量：全量非 archive 170 py / 45,582 行；archive/ 另 8,127 行（README 明言"do not participate in runs"，可直接排除）。

两案权衡：
- **全量随移**：保历史完整（`docs/OC_LatentFlow_DE_confirmed_decisions_summary.md` 1040 行等判决文档与代码互证）、review 面大但可借 302 单测兜底；代价是 npi 仓库一次性 +45.6k 行、其中过半是已终止线。
- **裁剪 OC 面**：入库体积约减半、npi 侧心智负担小；代价是丢阴性线代码（判决文档可单独保留为 docs），且裁剪边界要逐文件核对 import 闭包。
- 折中（推荐给 design 评估）：**代码裁剪 + 判决文档全量保留**（docs/ + experiments/ 笔记归档价值独立于代码）。

### 6.3 45.6k 行一次入库的 review 策略

- 不逐行 review 算法内部：信任三道既有防线——302 单测（本机已实测 293 过）、`docs/FOUNDATION.md`/decision summary 的预注册判决链、git 历史完整可溯（HEAD 9efe1e3）。
- review 焦点收窄到三类：(a) config/路径层（_PROJECT_ROOT 派生、env var 面）；(b) npi 包装层（BLAS 守卫、outputs 落点、run.py）；(c) 裁剪边界（若裁剪：import 闭包完整性 = 裁剪后全量 pytest 通过）。
- 入库卫生：行尾统一 LF + `.gitattributes`；排除 `archive/`、`experiments/.obsidian/`、`results/`（4.8G）、`outputs/`（147M）、`logs/`（2.6M）、`__pycache__`。

### 6.4 checkpoints 558MB 处置

- 内容：`checkpoints/` 558M，含 fold0 OC-LatentFlow 全系权重（`fold0_oc_latent_flow_flow_best_graph_mexhat*.pt` 12+ 变体、gru/var forecaster、de/dysib 子目录、edra 子目录）。
- npi `.gitignore` 已含 `*.pt` / `*.ckpt` / `*.pth` / `outputs/` → 无论放包内还是 outputs 都不入库。
- 建议方案（design 拍板）：保留在 autodl-tmp 侧（如 `/root/autodl-tmp/latent_flow/checkpoints/` 原位不动或迁 `/root/autodl-tmp/latent_flow_port/checkpoints/`），npi 侧 config 用 env var/一行改动指回——旧 θ 权重是「复现 FOUNDATION 数字」与后续对照的唯一凭据，不可弃；轻量产物（图/JSON）按 npi 惯例走 `/root/npi/outputs/<任务名>/`。
- `norms/fold0_modma_L128.npz`（DE 线归一化统计，小文件）随代码走。

### 6.5 其他登记

- `modda_resting_process.zip` 读取异常（§4.3）——移植验收若含 MODMA 线，先修复该文件（重新上传/同步完成后再验）；EDRA 线不受阻。
- 旧库 `.trellis/`（337 文件被 7z 排除但 git HEAD 里有）含其 spec/journal：**不随移进 npi 的 .trellis**（两套 Trellis 体系会冲突）；如需考古走 git HEAD。
- `experiments/` 是 Obsidian 库（19M，含 PDF/中文笔记）：有归档价值，建议以 docs 形式选择性收录或整目录放 autodl-tmp 侧。

---

## 7. 推荐决策包（供 design.md 直接引用）

1. **落位形态**：`models/latent-flow-v1/latent_flow/` 嵌套包（方案 B）。变体目录内：完整包 +（可选）`run.py` npi 风格包装 + 移植说明 README。运行入口：`cd models/latent-flow-v1 && python -m latent_flow.scripts.<name>`（或 run.py 转译）。源码导入零改动。
2. **基线**：git 锚点 HEAD `9efe1e3`（分支 08-15-substrate-cross-channel-probe），内容叠加工作树 6 个真实改动文件（dysib v22 线可随可裁）；`data/` 9 加载器 `git checkout HEAD -- data/` 恢复；入库统一 LF + `.gitattributes`；排除 archive/、.obsidian、results/outputs/logs/、旧 .trellis。
3. **裁剪策略**：默认「代码裁剪 OC 面 + 判决文档（docs/ + experiments 笔记）全量保留」；裁剪闭包 = `config.py, __init__.py, models/{oc_latent_flow,graph_utils}.py, attribution/, baseline/, viz/{metrics,visualizer,connectivity,nature_style}.py, training/{trainer,cli,seeding}.py, data/ 全部, scripts/ 的 7 个 active + _connectivity_driver.py, tests/ 对应子集`（裁剪后以全量 pytest 通过为闭包正确性判据）；若 design 倾向全量，review 策略按 §6.3 收窄。
4. **适配改动清单**：零改动类（数据根 env var、device 自动、Agg、依赖全满足）｜包装类（入口 BLAS 守卫 + baseline n_jobs ≤16、outputs 指向 `/root/npi/outputs/<任务名>/`、run.py）｜可选微改类（CHECKPOINT_DIR 指向 autodl-tmp 的 558M 旧权重、gbk 行清理、build_edra_geometry 的 .vhdr 断言改 npz 输入）。
5. **验证顺序**：①移植后全量 pytest（预期 293+/302，数据集无关）→ ②`python -m latent_flow.scripts.shape_check_oc_latent_flow` → ③`train_oc_latent_flow --smoke`（不污染正式 ckpt，GPU 4080S）→ ④EDRA npz shim + geometry_edra.pt 生成 + 窗数核对（fmt 契约 930 窗/例）→ ⑤`run_oc_connectivity_attribution` 小规模冒烟（1 被试 × 少量源）→ ⑥（MODMA zip 修复后）MODMA 线对照 FOUNDATION 数字。
6. **边界声明**（design.md 必写）：本任务只移植不融合——σ 二分与 perturb-ec-v1 四要素的算子/度量/τ 机制差异照 §4.2 声明；旧库 θ* 不自动继承 npi 代理脑资格，须按 surrogate-brain-qualification.md 重新验收；EDRA 数据接入 shim 属后续任务。

## Caveats / Not Found

- `modda_resting_process.zip` 在位但读取异常（目录项可见、open ENOENT），未探究修复（超出只读研究范围）；修复前 MODMA 相关结论（含 4 个数据依赖测试失败→通过的预期）未实证。
- 工作树与 HEAD 的行尾方向（7z 侧 LF、git blob CRLF）由 diff -w 空结论推断，未逐文件核 CRLF 字节（对结论无影响）。
- de 线脚本（train_de_vae/train_de_flow 等）未逐个运行验证（超 302 单测覆盖面之外的冒烟未做）；本机无其数据（.vhdr）故运行验证本就不可行。
- 旧库 `.trellis/spec/domain.md`（FOUNDATION 引用的 Key Findings 详录）在 7z 中被排除、仅存在于 git HEAD——本报告未展开考古，需要时 `git show HEAD:.trellis/spec/domain.md`。
