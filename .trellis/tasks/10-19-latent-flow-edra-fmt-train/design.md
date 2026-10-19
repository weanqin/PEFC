# design — 10-19-latent-flow-edra-fmt-train

> 技术设计。前置事实（2026-10-19 规划期实测核对）：`MODMADataset` 本就参数化 window_size/step/obs_size 且 `extract_windows` 边界规则（从 0 起、不补齐、丢尾）与 fmt 契约 930 窗公式逐位一致；`ConvEncoder` 以 `AdaptiveAvgPool1d(t_lat)` 钉时间维，obs_len/forecast_len 为构造参数（384/128 无架构障碍）；`norm_stats.compute_train_norm_stats` 提供训练折统计严格 held-out 路径。

## 1. 总体形态

三层改动面，包内既有逻辑**零修改**（仅 additive 追加），npi 适配主体在包装层：

| 层 | 改动 | 内容 |
|---|---|---|
| 包内 additive | `config.py` | 追加 `EDRA_FMT_DATA_ROOT`（env 同名，默认 `/root/autodl-tmp/edra_processed_fmt`）与 `EDRA_FMT_GEOMETRY_PATH = EDRA_FMT_DATA_ROOT / "geometry_edra_fmt.pt"` 两个常量；既有行不动 |
| 包内 additive | `data/datasets.py` | `_REGISTRY` 追加 `"edra_fmt"` DatasetDescriptor（channels=62、fs=250.0、file_format="npz"、geometry_path=EDRA_FMT_GEOMETRY_PATH、normalize_per_channel=True、group_layout="subdir"、`HC/*.npz` / `MDD/*.npz`） |
| 包内 additive | `data/modma.py` | `_load_subject` 追加 `"npz"` 分支：`np.load(path)["data"].astype(np.float32)`（归一化走既有统一路径，mat/brainvision 分支不动） |
| 包装层新增 | `models/latent-flow-v1/train_edra_fmt.py` | 本任务训练入口（见 §4），与 run.py 同级同惯例（BLAS 守卫等） |
| 包装层新增 | `models/latent-flow-v1/build_edra_fmt_geometry.py` | 仅当 stage2 选 mexhat-graph 时需要（见 §6 决策 D1/D2）：从 npz `channel_names` 解析 10-20 坐标生成 `geometry_edra_fmt.pt`（复用包内 `build_edra_geometry` 的 MNE standard_1020 解析 + POz 中点逻辑，import 复用不复制；MNE 1.13.2 已装；落点随 D2 拍板；消费侧 `load_geometry(path=…)` 显式传路径——其 dataset 分派不识别 edra_fmt） |
| 包装层微改 | `models/latent-flow-v1/.gitignore` | 追加 `latent_flow/logs/`（stage2 history JSON 内置落点，见 §5） |
| 零改动 | 其余全部 | 训练器/模型/既有数据线/pytest 全集/源库/旧产物 |

npz 文件名布局实测：`<root>/{HC,MDD}/<subject_id>_rest_v4.npz`；npz 键含 `data (62,60001) float32`、`channel_names (62,)`、`subject_id`、`group`、`fs=250`。

## 2. 数据适配（R2 落实）

### 2.1 划分复用（不重新划分）

- manifest：`/root/npi/models/gpfm-mamba-v1-fmt/splits/fold0_status_ok_seed42_val4.json`（**只读**；与 v4a-fmt 拷贝逐字节一致已实测）。
- 入口解析：`train/val/test` 整数 subject_id 列表 + `groups` 映射 → 映射为 `<root>/<group>/<id>_rest_v4.npz` 路径。
- 硬断言（失败即拒跑）：① train+val+test 并集 = manifest groups 全集 48 例且文件全部存在；② 计数 34/4/10；③ test 列表（排序后）= `[10,11,12,17,24,28,30,44,61,69]`（spec 登记值）；④ 每例 npz 内 `group`/`subject_id` 字段与路径一致。
- **不调用** `SubjectFoldSplitter`，不使用其 seed；本任务 seed=42 仅用于训练复现（`set_seed(42)`）。

### 2.2 时间窗（不重新生成）

`MODMADataset(subject_files=…, dataset="edra_fmt", window_size=512, step=64, obs_size=384, norm_stats=…)`：

- 窗口契约与上一轮逐位同规：总窗 512（obs 384 + fc 128）、stride 64、边界 `range(0, n−512+1, 64)` → 每例 (60001−512)//64+1 = **930 窗**（入口对每例打印并断言；任何被试窗数 ≠930 即失败）。
- `t_obs/t_target` 时间轴 dt=1/FS，FS=250 模块常量与 fmt 采样率一致，零适配。
- x_target (512) 为重建目标，x_obs (384) 为编码器输入——与旧库 375/125 同构、仅样本数不同（fc 段长度 128 由 `x_target[:, 384:, :]` 切出）。

### 2.3 归一化（防泄漏口径，与上一轮一致）

- `compute_train_norm_stats`（包内自带）在 **train 34 例** 上估计逐通道 mean/std（float64 累加），返回 `(mean, std)` 各 `(62,1)` + 训练文件 hash；
- train/val/test 三个 dataset 全部传同一 `norm_stats`（MODMADataset 既有语义：给定 norm_stats 时忽略逐被试归一化，三折同一统计）；
- 统计量与训练文件 hash 写入两阶段配置 JSON（可复现凭据）。

## 3. 模型与两阶段（R3 落实）

复用 `OCLatentFlow`，仅窗口长度参数化改变：

- Stage 1：`OCLatentFlow(n_channels=62, latent_dim=64, t_lat=16, obs_len=384, forecast_len=128)`（dense 默认流，stage1 不训练流），`forward_stage1`（VAE 损失：recon/corr/cons + td/bd 边界 + KL 退火 beta→`--beta-kl` 默认 1e-4）。
- Stage 2：同构模型（流按 D1 决策构建：dense 或 `--graph-mode --graph-kernel mexhat`），加载 stage1 ckpt（`load_state_dict(strict=False)`，**非 flow 键失配即抛错**——复用 `_run_stage2` 既有断言），冻结 encoder/proj/decoder（+decoder_fc 若存在），trainable = flow 参数（日志打印数量为证）；`forward_stage2` 训练流。

**Stage-2 CFM 口径（如实登记，向用户显式声明）**：`forward_stage2` 为条件流匹配构造——`t~U(0,1)`，`z_t = (1−t)·z_obs + t·z_fc_target`（z_fc_target = 同一冻结 encoder 编码真实 x_fc），速度场 `v_pred = flow(z_t, t, z_obs)`，单步预测 `z_fc_hat = z_t + (1−t)·v_pred` 经冻结解码器出 fc，由 **fc 信号损失（recon2/corr2/cons2）端到端优化**；**无显式速度 MSE 回归项**（源码注释明示 v_pred 无显式目标，其目标由解码器信号损失隐式定义）。推理 `sample()` 走 10 步 Euler。该口径即旧库「潜空间流匹配/CFM 变体」的原始实现，属本任务「潜空间 CFM」约束的满足方式；若用户要求教科书式 velocity-MSE CFM，属口径变更另议（不在本任务默认范围）。

**gpfm 隔离**：训练入口与新增代码零 `models/gpfm-mamba*` import；AC6 以 import 扫描核对。

## 4. 训练入口 `train_edra_fmt.py`（包装层）

职责与结构（单文件，~250 行）：

1. BLAS 守卫（numpy/torch 导入前 setdefault ≤8，同 run.py 惯例）+ `sys.path` 注入变体目录；
2. CLI：`--stage {1,2}`、`--run-dir`（默认 `/root/npi/outputs/10-19-latent-flow-edra-fmt-train/run_01`）、`--manifest`（默认上述固定路径）、`--epochs 50 --lr 1e-3 --batch-size 32 --patience 10 --seed 42`（旧库 CLI 默认）、`--beta-kl 1e-4`、stage2 专属 `--graph-mode/--graph-kernel/--mexhat-gamma/…`（按 D1）、`--vae-ckpt`（stage2 默认指向 run_01 内 stage1 产物）、`--limit-subjects N`（仅冒烟：每折截前 N 例，正式跑禁用）；
3. §2 全部断言 → loaders（train shuffle / val、test 顺序）→ norm_stats；
4. 模型构建：复刻 `train_oc_latent_flow._build_model` 但 obs_len=384/forecast_len=128（**不复用原函数**因其硬编码 375/125；geometry 按 D1 传 coords）；
5. 训练执行：**直接 import 复用** `Stage1Step/Stage2Step/_KLAnnealCallback/_run_stage1/_run_stage2`（训练循环零复制）；ckpt 经 `--save-path` 语义重定向到 run_01。**args 命名空间钉死**：`dataset="edra_fmt"`（决定 stage2 history 包内落点 logs/edra_fmt/）+ `unfreeze_vae=False`/`anti_uniform_lambda=0.0`/`sparse_lambda=0.0`/`lr_scheduler="none"`/`lr_warmup_epochs=0` 显式默认；测试集 smoke 评估用 wrapper 自带 384 版（`_smoke_eval` 硬编码 OBS_LEN=375，形状恰好兼容会静默错值，**不得 import**）；
6. 落盘：`configs/stage{1,2}_config.json`（全参数 + manifest 摘要 + norm hash + git HEAD + 窗数断言记录）；history——**stage1 由 wrapper 对 `_run_stage1` 返回的 History 对象自行 `to_json` 直写 run_01**（`_run_stage1` 无内置 history 落盘），stage2 训练后把包内 `latent_flow/logs/edra_fmt/*_history.json` 复制进 run_01（源位置为训练器内置副产物，已 gitignore）；
7. `--smoke`：数据面全断言仍生效（冒烟即验证 shim 正确性）；冒烟链路按 implement.md S3.3 两步执行（stage1 冒烟以 patience=1 + 临时 save-path 落 `smoke_tmp_vae.pt`，供 stage2 冒烟真实走到 ckpt 加载/冻结路径，冒烟后删）。

超参默认 = 旧库 CLI 默认（epochs 50 / lr 1e-3 / batch 32 / patience 10）；fold0 单 seed 前置验证（study1 训练顺序门禁：5-fold 须经用户另行同意）。

## 5. 产物布局（run_01）

```
outputs/10-19-latent-flow-edra-fmt-train/run_01/
├── configs/stage1_config.json, stage2_config.json
├── checkpoints/fold0_edra_fmt_vae_best.pt, fold0_edra_fmt_flow_best[_graph_mexhat].pt
├── logs/stage1.log, stage2.log（nohup）, stage1_history.json（wrapper 自写）/ stage2_history.json（复制自包内 logs/edra_fmt/）
└── (experiment.md Run 块登记全部路径)
```

隔离保障：入口所有写路径由 `--run-dir` 派生；不触碰包内默认 `CHECKPOINT_DIR`（唯一例外 = 训练器内置 history JSON 的包内 `logs/edra_fmt/` 副本，gitignore + 复制进 run_01 双保险）。

## 6. 决策点（1.4 评审拍板）

- **D1 stage2 流变体**（均为潜空间 CFM，差异为速度场内通道混合先验）：
  - **推荐 `mexhat-graph`**：旧库 EDRA 线 SOTA 配置（`EDRA_DEFAULT_CHECKPOINT` 即 mexhat；FOUNDATION 记录 mixer 使跨被试 EC 相关 0.172/0.157 → 0.747），中心-环绕侧抑制结构利于跨通道传导特异性（Study II 缺口的直接关切）；代价 = 需生成 `geometry_edra_fmt.pt`。**证据基座加注（评审 A）**：该 SOTA 证据来自旧库 **500 Hz edra 的 EC 归因线**，非 fmt 范式、非 Study-I 训练质量指标——mexhat 收益在本 run 属**结构先验迁移假设**，本任务无评估环节不检验该假设，不据此对本 run 质量作任何预期承诺。
  - 备选 `dense`：纯 CFM 无结构先验、零 geometry 依赖、最保守（且 D2/geometry 风险自动消解）。
  - 首跑不叠加 A_eff 正则项（旧 EDRA 默认 ckpt 无 reg 后缀）。
- **D2 geometry 落点**（仅 D1=mexhat 时相关）：
  - 案 a（推荐）：落数据根 `/root/autodl-tmp/edra_processed_fmt/geometry_edra_fmt.pt`——数据集级可复用产物（后续 fold/归因任务引用同一稳定路径），与根内 `chan_coords_62.npy`/`chan_forward_62.fif` 辅助文件先例一致；**须用户拍板豁免**（prd R1 禁写条款的唯一显式例外，单文件新增、零覆盖、可独立删除）。
  - 案 b：落 `run_01/` + `EDRA_FMT_GEOMETRY_PATH` env 覆盖——数据根零写入，代价是路径 run 级私有、后续任务需重指。
- **D3 超参**：默认沿用旧库 CLI 默认（见 §4）；如用户要求改动属配置变更，登记后执行。
- **D0（1.4 显式拍板）stage-2 CFM 口径确认**：见 §3——潜空间 CFM 构造 + 信号损失隐式优化速度场、无显式速度 MSE 项。此为「潜空间 CFM 变体」的旧库原始满足方式；用户若意图教科书式 velocity-MSE CFM，属口径变更另立任务。

## 7. 风险与对策

| 风险 | 对策 |
|---|---|
| AdaptiveAvgPool 使 384→16 池化比 24（旧 375→23.4），latent 时间分辨率语义微变 | 架构无障碍（设计如此）；design 如实登记，判读时不与旧库 MODMA/EDRA 数字直接对比 |
| train 34 例 ×930 窗全内存预载 ≈ 7 GB（float32 x_obs+x_target） | 503 GB 内存无压力；冒烟先验证加载时间 |
| 包内 additive 改动破坏冻结 pytest 口径 | 改动均为新分支/新条目，不触既有路径；AC7 全量回归 67P/0F/17S |
| BaseTrainer 无断点续训：中断重跑 = 从 epoch 0 重来 | 同 run_01 内仅覆盖自身 ckpt，无既有产物风险；nohup 日志可判断中断点，重跑代价 = 时间 |
| stage2 history JSON 落包内 | gitignore `latent_flow/logs/` + 复制进 run_01；README-port 补一行说明；args 钉死 dataset="edra_fmt" 防误写 logs/modma/ |
| 训练时长不确定（首次在 48 例 ×930 窗规模上跑） | stage1 先行 nohup，日志实时可查；epoch 耗时实测后登记；patience 10 早停兜底 |

## 8. 回滚

改动全部为新增文件 + 三处 additive 追加（config/datasets/_load_subject 各数行）+ 变体级 `models/latent-flow-v1/.gitignore` 一行 + README-port.md 补两行。回滚：`git checkout -- models/latent-flow-v1/`（还原全部 5 个改动文件）+ 删两新增 wrapper 脚本 + 删 `outputs/10-19-latent-flow-edra-fmt-train/` 整目录 + 删包内 `latent_flow/logs/edra_fmt/` + 删数据根豁免文件 `geometry_edra_fmt.pt`（D2 豁免单文件）；无任何既有产物被覆盖（不可逆操作为零）。
