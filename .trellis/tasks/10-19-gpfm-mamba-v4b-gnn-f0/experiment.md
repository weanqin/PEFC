# 实验日志 — 10-19-gpfm-mamba-v4b-gnn-f0（gpfm-mamba-v4b(ch1) × LearnableMexHatMixer GNN 先验融合）

> 每次「跑批/训练/评估/统计」追加一个 Run 块：跑前先写「目的+配置」，跑后同会话补「结果+结论」，不留空块。
> 格式契约见 `.trellis/spec/research/experiment-log.md`；产物只落本任务声明的变体目录（`.trellis/spec/research/path-isolation.md`）。
> 全部 Run 属 **前置验证 · fold0 · 单 seed**（study1 门禁，surrogate-brain-qualification.md §判定流程 2）。

## S1 结构落地登记（2026-10-19，非 Run 块——结构/单测/零改动核对的实测登记）

### 新变体目录 `models/gpfm-mamba-v4b-gnn/`（commit 22afbc5 工作树新增）

文件清单：`config.py` / `graph_head.py`（GraphHead + 几何单一 choke point + 正则组合器）/ `model.py`（VelocityNetwork v4bg + EMA）/ `train.py` / `evaluate.py`（双遮蔽包装，前缀守卫 `v4bg-`，train 侧含 W 范数轨迹双口径仪表）/ `tests/test_v4b_gnn.py`（十二项）。

### 实测登记值

- **参数量（精确）**：全模型 **12,864** = 基座 506（seg 2 + in_proj 19 + trunk 4×121 + RMSNorm(1) 1）+ GraphHead **12,358**（in_proj 128 + msg 2×4,160 + W 3,844 + σ 1 + out_proj 65）——design §2.4 预算（head 12,358 / 总量 ≈12.9k）**精确吻合**；对照 v4b-d1 总量 64,056（head 63,550）。
- **config_hash pin**：默认 `5920e339c28c`；epochs=1 bench 覆写 `30c4a8adae57`。零漂移复验：v1 `c6ff80ac1c43` / v2 `fc2a9ce14948` / v3lp `0c2de30a4868` / v4 `91dfd211dbc3` / v4a `2a456201dc90` 逐位不变。
- **AC2 等价（实测强于断言口径）**：ricker（σ=0.05 非默认）与 dog（4 参数全非默认）两核下 A_eff 与 latent-flow-v1 源 `LearnableMexHatMixer.effective_adjacency` **max|Δ| = 0.00e+00**（断言口径 allclose rtol=1e-6；pairwise_d 逐位一致）。
- **trunk 同构复验**：同 seed(1234) 下 v4bg 的 trunk 部分 state_dict 与 v4b-d1（v4 model.py, d_model=1）**逐位相同**（单变量归因口径基础，design §0/§2.1）。
- **通道对齐门禁实测（S1.3）**：GraphHead choke point 扫描数据根 `/root/autodl-tmp/edra_process_v4` 全部 npz = **46/46 例** `channel_names` 与 `geometry_edra_fmt.pt` 顺序全等（零 mismatch；扫描耗时 ~0.02 s，lru_cache 按路径缓存）；单测含 mismatch/例数不符正反例。
- **native 坐标交叉登记（S1.3，非门禁）**：数据根原生 `chan_coords_62.npy`（行序无名字佐证）与 `geometry_edra_fmt.pt` coords **逐元素相等**（allclose atol=1e-6 全等；最近邻匹配 = 恒等置换，max 最近距 3.8e-9；逐轴 Pearson = 1.0000）——行序身份一致，几何单源无矛盾。
- **单测**：新测 `python3 -m pytest models/gpfm-mamba-v4b-gnn/tests/test_v4b_gnn.py -q` = **12 passed**（连跑 5 次稳定）；回归 v1 七测（test_model/test_sample/test_gpr/test_losses/test_data/test_gc_psd/test_bandlimit 直接执行）全 PASS；v4 `test_v4b.py` PASS；v4a `test_v4a.py` PASS。
- **d1 边界符号量化观察（实测，判读注记）**：d_ch=1 下 head 前共享 `RMSNorm(1)` 将池化特征量化为符号（|边界值| ∈ [0.877, 1.0]、均值 0.99999，即 x/√(x²+ε) ≈ sign(x)）；梯度经 ε 软带流过（随机初始化下 |grad W|≈97、|grad σ|≈805，量级健康）。含义：图头输入为 62 通道符号模式（v4b-d1 的 cMLP 同边界，非本变体引入）；对 H4 判读的权重=阴性结果时「限 d1 基座」表述再添一条结构性注脚（与 H3 登记的 trunk 容量塌缩同源）。
- **S1.0 git 快照（零改动基线，2026-10-19 19:30 前）**：`git status --porcelain` =
  ```
   M .trellis/tasks/archive/2026-10/10-19-latent-flow-edra-fmt-train/experiment.md
   M models/latent-flow-v1/latent_flow/baseline/data_utils.py
   M models/latent-flow-v1/latent_flow/viz/connectivity.py
  ?? .trellis/tasks/10-19-gpfm-mamba-v4b-gnn-f0/
  ```
  （前 3 处 = design §10-4 拍板「悬置」的既有未提交改动，本任务不触碰。）**S1 完成后对比：新增仅 `?? models/gpfm-mamba-v4b-gnn/`，其余逐字节不变**——v1*/v2/v3/v4*/v4a*/latent-flow-v1/common/ 侧零变化。

### 与 design 的偏离登记（逐条，无静默变通）

1. **v4 config 的 `d_z`/`head_act` 字段未继承**（design §3「继承 v4 全部字段」的字面读法）：cMLP 头被替换后两字段无对应结构，保留即死字段且会静默进 config_hash；由 `graph_*` 字段取代（d_z/head_act 先例的延续），包装一致性断言改为 graph_* 全组。训练/数据/先验/采样口径字段全部逐字段继承。
2. **正则旋钮落点**：`uniformity/sparsity` 两方法在 GraphHead 源语义保留 + `graph_regularization()` 组合器（λ=0 返回精确 0）+ config 双 λ 字段；**未接入 v1 训练损失**（λ_u=λ_s=0 默认口径与 PRD R6「旋钮落地即可」一致——单测⑨在组件级断言「λ>0 含惩罚项/λ=0 逐位一致」；未来启用属口径变更另登记）。
3. **方向单测的「仅通道 i 响应」实现为响应集恰为 {i, j}**：源组件消息传递含恒等残差（in_proj(x_j) 直通），扰动源通道 j 经自身残差流必然自响应（平凡）；检验对象 = i 是**唯一跨通道**响应者 + 反向注入（W[j,i]=1）时 i 零响应——源语义 A[i,j]=j 驱动 i 的严格刻画。
4. **W 范数轨迹仪表经运行时 EMA 子类替换实现**（v1 train.py 命名空间 `mod.EMA`，仅本包装进程内生效）：v1 文件零改动、checkpoint "ema" 键结构不变、不消耗 RNG/不改训练数值；每 step 记录 raw/EMA shadow 双口径至 `logs/w_norm_traj.jsonl`。

## S1' v5 结构落地登记（2026-10-19 二次转向后；本轮 = 结构 + 单测 + 最小验证，不含任何训练 run）

> 跑前登记（S1'.0）：v4bg S1/S2 产物停用留档（Run 001/002 为其记录，v4bg-main-01 永不启动）；本轮按 PRD 顶部二次转向执行 **v5**（v2 基线 × 轴分解交替）。registry（`.trellis/research/context.md`）回填由主会话执行，本 Agent 不触碰该文件。

### S1'.0 git 快照（零改动基线，2026-10-19 v5 动工前）

`git status --porcelain` =
```
 M .trellis/tasks/archive/2026-10/10-19-latent-flow-edra-fmt-train/experiment.md
 M models/latent-flow-v1/latent_flow/baseline/data_utils.py
 M models/latent-flow-v1/latent_flow/viz/connectivity.py
?? .trellis/tasks/10-19-gpfm-mamba-v4b-gnn-f0/
?? models/gpfm-mamba-v4b-gnn/
```
（前 3 处 = 悬置未提交改动不触碰；`?? models/gpfm-mamba-v4b-gnn/` = v4bg S1/S2 停用留档产物，本轮起零改动。）完成后对比：新增仅 `models/gpfm-mamba-v5/` 与本任务目录内文件。

> **中断与基线注记（2026-10-19 本次派发如实登记）**：上一次 trellis-implement 派发中途中断，已留下 `models/gpfm-mamba-v5/` 部分成品（config.py 147 / graph.py 239 / model.py 206 / train.py 124 / evaluate.py 108 行 / tests/test_v5.py 627 行——上快照为该次派发动工前所记，故未含 v5 行）。本次派发模式 = **校验补完**：逐文件对照 design §2 与派发简报核对（缺则补、错则改、对则留），不盲信现有内容、不推倒重写。**本次派发的零改动基线 = 上快照 + `?? models/gpfm-mamba-v5/`（上述现状）**；完成后对比口径相应为"变化仅落在 `models/gpfm-mamba-v5/` 目录内与本任务目录文件"。

### S1'.1–S1'.2 结构与单测（校验补完登记，2026-10-19）

**逐文件校验结论（缺补/错改/对留）——六文件全部留用零修改**：

- `config.py`（147 行）：与 v2 副本逐字段比对一致，差异仅 variant_name=`"gpfm-mamba-v5"` + 七个图字段（graph_kernel=ricker / graph_gamma=0.1 / graph_gamma_m=1.0 / graph_w_init="" / graph_lambda_uniform=0.0 / graph_lambda_sparse=0.0 / geometry_path）+ d_model=62 注释；build_config/hash/dump_yaml 惯例与 v2 同实现。→ **留**。
- `model.py`（206 行）：输入构造（seq+seg+temb → (B,512,79)）/in_proj Linear(79→62)/head（RMSNorm(62)+Linear 62→62）/截取 future 128/time_embedding/EMA 与 v1 model.py **逐字一致**（逐行比对）；trunk = 4×[Mamba 时间块 → 空间块 → LayerNorm] **逐块迭代**（`blocks[i] → spatial_blocks[i](A_eff) → block_lns[i]`，非整段 Sequential）；A_eff 循环外单次求值、跨块共享；钩子三参（bottleneck_rank/norm_type/cond_adaln）接受并断言默认（None/"rmsnorm"/"off"——v1 train.py:273-281 无条件透传三 kwargs 可直跑）；d_model==n_channels 构造断言（要求③防线）。→ **留**。
- `graph.py`（239 行）：A_eff = γ_m·M(d;σ) + γ·W 与 latent-flow-v1 源 `LearnableMexHatMixer`（oc_latent_flow.py 约 313–520 行）**逐式一致**（ricker log_sigma / dog 4 参数重参数化、γ/γ_m/pairwise_d 三 buffer、W 零初始化不对称化 + w_init 注入（含 .pt/.npy/张量三态）、uniformity/sparsity/dog_params 源语义、einsum "ij,blj->bli" 源语义 A[i,j]=j 驱动 i）；几何单一 choke point `load_geometry` lru_cache + 数据根全 46 例 channel_names 顺序全等断言 + 例数断言；与 v4bg `graph_head.py` 无代码复用关系（公式源 = latent-flow 源码）。→ **留**。
- `train.py`（124 行）/`evaluate.py`（108 行）：config+model+graph **三模块双遮蔽**（各含解析路径自检：graph 命中 v5、mamba 命中 v1），run_id 前缀守卫 `v5-`（train 含 --resume 派生 run 名校验与缺省自动派生；evaluate 强制显式 --run-id），d_model/graph_* 六字段运行时一致性断言（v4 双遮蔽先例 + graph 扩展）。→ **留**。
- `tests/test_v5.py`（627 行）：design §4 十项逐项覆盖（明细见下），CUDA 要求与 v1 test_model/test_sample 及 v4bg test 惯例一致（mamba_ssm 官方 kernel）。→ **留**。

**实测登记值**：

- **参数量（精确）**：v5 全模型 **152,647** = v2 基线 **132,682**（seg 2 + in_proj 4,960 + blocks 4×30,938=123,752 + head_norm 62 + head 3,906，v1 model.py 独立装载 d_model=62 并列实测）+ 增量 **19,965**（W 3,844 + σ 1 + msg 4×3,906=15,624 + LN 4×124=496）——design §2.5 预算 **精确吻合**。
- **config_hash pin**：v5 默认 **`1ac433fdd3d0`**（对 graph_gamma/graph_kernel/d_model 敏感）；零漂移复验：v1 `c6ff80ac1c43` / v2 `fc2a9ce14948` 逐位不变（独立子进程隔离复算）。
- **AC2 等价（实测强于断言口径）**：ricker（σ=0.05 非默认）与 dog（σe 0.03/σi 0.09/Ae 1.3/Ai 0.45 全非默认）两核下 A_eff 与 latent-flow-v1 源 `effective_adjacency` **max|Δ| = 0.000e+00**（断言口径 allclose rtol=1e-6；pairwise_d 逐位一致；dog_params 四参数一致）。
- **交替结构（前向钩子序）**：`[mamba_i, spatial_i, ln_i] ×4` 严格交替（每时间块后恰一空间块一 LN）；msg/LN 逐块独立实例；全模型恰一个 MexHatAdjacency（A_eff 跨块共享单实例，spatial step 不注册 adjacency 子模块）。
- **方向单测（B-3）**：γ_m=0 + W[i,j]=1 单边注入下 A_eff=γ·W 精确（|A[i,j]−γ|<1e-7、A[j,i]=0）；正向注入响应集恰 {j(残差自响应), i(唯一跨通道)}、反向注入仅 {j}；读出转置口径 A_eff.T[j,i]=γ。
- **因果性**：return_all 口径下 future 扰动不改 past / future 后段扰动不改前段 / past 前缀条件生效；空间块位置逐点（扰动时间位 l 只影响 l 位——组件级）。
- **几何门禁**：真数据根全 **46/46** 例 channel_names 与 geometry_edra_fmt.pt 顺序全等（lru_cache 按 (geometry, data_root) 命中）；σ init = median_d = **0.120868**（与 v4bg Run 002 登记的 0.12087 同几何一致）；名序不一致/例数不符反例均报错。
- **GPU 冒烟（单测①）**：B=256 fwd+bwd 44 ms，peak **2.38 GiB**——交替结构保持 (B,512,62) 激活、无 v4bg CI 折叠的 B·C 膨胀（对照 v4bg bench 14.6 GiB），32 GB 卡余量充裕。
- **单测**：`python3 -m pytest models/gpfm-mamba-v5/tests/test_v5.py -q` = **10 passed**（197.8 s）；回归独立复跑（非仅经 ⑩ 子进程）：v1 七测（test_bandlimit/test_data/test_gc_psd/test_gpr/test_losses/test_model/test_sample）逐个直接执行全 PASS + v4 `test_v4b.py` PASS + v4a `test_v4a.py` PASS。

### S2' 最小验证（2026-10-19，无任何训练 run）

- **构建**：`VelocityNetwork()`（全默认参，经几何 choke point 扫描）成功；num_params = **152,647**（与单测⑤一致）。
- **前向**：随机 (4,62,384)/(4,62,128)/t → v_θ **(4,62,128)**（finite、float32）；return_all → **(4,62,512)**；future 切片与 return_all allclose（atol=1e-6）。
- **v5 名称实例化**：`build_config()` → variant_name **gpfm-mamba-v5** / config_hash **1ac433fdd3d0** / d_model==n_channels==62；checkpoint 主名经 v1 train.py 的 `f"{cfg.variant_name}-best/last.pt"` 自动派生为 `gpfm-mamba-v5-{best,last}.pt`（与 v1/v2 天然隔离）。
- **包装链可跑（不启动训练）**：`python3 models/gpfm-mamba-v5/train.py --help` exit 0（三模块遮蔽 + 运行时一致性断言 + v1 train.py importlib 装载全链路）；前缀守卫反例：train `--run-id v2d62-hijack`、evaluate 缺 --run-id、evaluate `--run-id main-01` 均 exit 1 拦截。
- **差异陈述**：design §5 五维表（通道处理/块结构/归一化/可解释连接矩阵/接口兼容）随交付汇报输出。

### S1'/S2' 完成时零改动核对

git 快照对比：`git status --porcelain` 与本次派发基线**逐条相同**（3 处悬置未提交文件 diffstat 不变：archive experiment.md +188 / data_utils.py +4 / connectivity.py +56；`?? models/gpfm-mamba-v4b-gnn/` 未触碰；`?? models/gpfm-mamba-v5/` 校验后零修改留用），新增仅本任务目录内 experiment.md 补记；无 outputs/ 新目录（未启动任何 run）。

### 偏离登记（S1'/S2' 范围，逐条）

1. **implement.md S1'.3（registry 回填 context.md）未由本 Agent 执行**：派发简报明确 `.trellis/research/context.md` 由主会话回填、本 Agent 不触碰——较 implement.md 的"随本轮落档"为口径收紧，非静默省略；回填责任在主会话。
2. **S1'.0 快照的记录时点**：上一次派发在创建 v5 文件**之前**记录了快照（故快照不含 v5 行）；本次以"中断注记 + 基线并入 v5 现状"的方式衔接（见上方注记），不重写历史快照。
3. 代码层面无偏离：六文件校验后全部留用零修改，design §2/§4 十项与简报六要求逐项吻合（含参数量预算精确吻合）。

## Run 001 — v4bg 冒烟（smoke：限批全链路）（2026-10-19）

### 目的与假设

- 验证 v4bg 变体（v4b d_ch=1 基座 + GraphHead）经 v1 训练循环的全链路机械正确性：双遮蔽装载、前向/反向、损失有限、val4 内部验证、best/last checkpoint 与产物目录落盘、W 范数仪表工作。不做任何质量判读。

### 配置

- 代码：`models/gpfm-mamba-v4b-gnn/`（git 22afbc5 工作树；train.py 双遮蔽包装 + W 范数仪表）
- 参数：全部默认（d_model=1、graph ricker/K=2/h=64/γ=0.1/γ_m=1.0/W 零初始化/λ_u=λ_s=0；batch 256、seed 1234、EMA 0.999、恒定 lr 1e-3、早停 10、gen_lowpass 40 Hz；划分 32/4/10，manifest 只读复用 v4 的 fold0 val4）；`--smoke` = 每类 2 例（合计 4 例）1 epoch
- 环境：base conda（Python 3.12.3 / torch 2.8.0+cu128 / mamba_ssm 2.3.1）@ 本地云服务器（RTX 4080 SUPER 32 GB，16 核 cgroup，BLAS 限 8 线程）
- 数据：`/root/autodl-tmp/edra_process_v4`（status_ok 46 例，fold0 train 折冒烟子集 4 例）；几何 `/root/autodl-tmp/edra_processed_fmt/geometry_edra_fmt.pt`（只读）
- 命令：`python3 models/gpfm-mamba-v4b-gnn/train.py --smoke --run-id v4bg-smoke-a`（显式输出重定向：`> outputs/gpfm-mamba-f0/v4bg-smoke-a/logs/train.log 2>&1` 由 v1 train.py Logger 双写 stdout/日志实现）

### 结果

- 指标：params **12,864**（[model] 行与单测一致）| train 4 例 5,284 窗 20 step，全损失有限（total 5.60→1.53）| val4 total **4.16142**（新 best）| it/s 3.43（首 step 预热后 ~3.6）| peak **14.633 GiB** | W 范数轨迹 20 点：raw 0.0516→0.8422，EMA 5.2e-5→9.0e-3（`logs/w_norm_traj.jsonl`）
- 产物：`outputs/gpfm-mamba-f0/v4bg-smoke-a/`（config.yaml + env_snapshot.txt + channel_mean/std.npy + logs/{train.log, metrics.jsonl, summary.json, w_norm_traj.jsonl} + checkpoints/{best,last}.pt 八类齐；ckpt 契约键全：model/ema/optimizer/scheduler/rng/best_val/config_hash=30c4a8adae57）
- 图表：无（冒烟不做图）

### 结论

- 支持（机械正确性）：v4bg 经 v1 训练循环全链路（双遮蔽装载/前向反向/损失有限/val4 确定性复算/best+last 落盘/W 仪表双口径）无异常；**W 零初始化下 raw 范数 20 step 内 0.05→0.84（16×），W 可学信号明确**（design §8 风险 2 的首证——最终以 Run 002 全量口径为准）。

## Run 002 — v4bg 1-epoch 基准（bench：32 例全量）（2026-10-19）

### 目的与假设

- R9 门禁材料：参数量精确值、峰值显存、it/s、val4 内部验证总 loss、**W 范数轨迹 raw 与 EMA shadow 双口径**（epoch 内逐 step，≥3 点）——评估「W 零初始化 + γ=0.1 下梯度尺度是否足以驱动 W 学习」（design §8 风险 2/3 的直接证据）与 100-epoch 外推时长。不做质量判读。

### 配置

- 代码：同 Run 001
- 参数：同 Run 001 唯 `--epochs 1 --bench`（resolved config hash 随 epochs 覆写为 `30c4a8adae57`）；训练折全量 32 例
- 环境：同 Run 001
- 数据：同 Run 001（train 32 例全量 + val 4 例内部验证）
- 命令：`python3 models/gpfm-mamba-v4b-gnn/train.py --epochs 1 --bench --run-id v4bg-bench-1ep`

### 结果

- 指标：params **12,864**（精确）| train 32 例 42,272 窗 165 step，40.0 s/epoch（train 段，不含 val）| **it/s 4.12** | **峰值显存 14.633 GiB**（32 GB 卡余量充足）| **val4 内部验证总 loss 3.27717**（新 best；mae 1.5407 / cons 1.6706 / tv 0.2306 / corr 0.4284；x̂1 std(EMA) 0.7375 vs x1 std 0.8533）| 100-epoch 外推 **≈1.11 h**（train 段外推；加逐 epoch val/checkpoint 写盘实际约 1.3–1.6 h 量级）| config_hash `30c4a8adae57`
- **W 范数轨迹（raw 与 EMA shadow 双口径，epoch 内 165 点全量落盘，抽样登记）**：

  | step | raw ‖W‖ | EMA ‖W‖ |
  |---|---|---|
  | 0 | 0.0612 | 0.000061 |
  | 40 | 1.1124 | 0.026648 |
  | 80 | 2.0971 | 0.086146 |
  | 120 | 2.9424 | 0.182015 |
  | 164 | 4.0396 | 0.320960 |

- 附带登记：σ（ricker log_sigma.exp）init=median_d=0.12087 → 1 epoch 后 raw **0.11444** / EMA shadow **0.12025**（核尺度微调中）；γ·W 对 A_eff 的贡献尺度 = 0.1×4.04 ≈ **0.40**（vs γ_m·M ≈ 1.0——1 epoch 即 W 已成 A_eff 的实质性成分）。
- 产物：`outputs/gpfm-mamba-f0/v4bg-bench-1ep/`（八类齐，同 Run 001 布局；best ckpt W raw ‖W‖=4.0396 / EMA shadow=0.3210 与轨迹末端一致）
- 图表：无（bench 不做图）

### 结论

- 支持（R9 门禁材料齐备）：①机械/资源可行——显存 14.6 GiB、4.1 it/s、主 Run 预计 ~1.1–1.6 h；②**W 学得动（design §8 风险 2 缓解）**——raw ‖W‖ 单 epoch 0.061→4.04（66×），EMA 口径 0.321 且斜率未衰减；③风险重估——W 增长强劲反而提示 §8 风险 5（先验 γ_m·M 被 W 稀释）与正则未启用（λ=0 口径）值得主 Run 读出期重点看（A_eff 行和分布/M vs W 分解）；④1-epoch val4 3.277 不作质量判读。
- 下一步：**G-R9 硬门禁——本摘要汇报用户、明确确认后方可启动主 Run `v4bg-main-01`**（100 epoch 上限 + 早停 10 + nohup 脱离式 + 显式输出重定向；S3.0 跑前登记另起 Run 003 块）。

## 用户拍板登记（2026-10-20，S3' 开跑前）

- **取消 v4bg 线（用户明示）**：「取消 gpfm-mamba-v4b-gnn，v4b 无法执行轴分解的任务」——与 2026-10-19 二次转向登记一致（v4bg S1/S2 产物停用留档、Run 001/002 为其记录、`v4bg-main-01` 永不启动）；本次为用户对取消的再确认 + 理由明示（v4b d_ch=1 基座无法承载轴分解——d_model=C=62 的通道维显式聚合需 v2 基座）。
- **直接进行 v5 训练（用户明示）**：「请你直接进行 gpfm-mamba-v5 的训练」——解释为 **G-R9 确认已由用户本次指令直接给出**，S3' 链路不再暂停等确认：smoke → bench（材料照常登记留档）→ 主 Run `v5-main-01` 连续执行。bench 材料仍完整落 experiment.md（W 范数 raw/EMA、显存、it/s、时长外推），供事后审阅。

## 用户拍板登记（2026-10-20，读出与参考归档阶段）

- **GC 计算主路径 = 原库 GC（用户明示）**：「计算GC使用原库GC，即latent_flow里的GC计算方式，并将原库GC登记为GC计算主路径」——实现 = `models/latent-flow-v1/latent_flow/baseline/granger.py`（成对双变量谱 GC，Geweke 1982/Dhamala 2008）；背景事实（归档任务 Run 005 实测）：npi 时域条件 GC 与原库谱 GC 逐被试 Pearson 仅 +0.0359±0.0756，两实现近乎不相关，实现选择即口径——后续一切 × GC 对照默认走原库实现。已回填 registry（context.md 2026-10-20 段）。
- **GC 参考数据 = fmt（用户明示）**：「数据使用fmt数据」——复用现成 refs（见下条），不现算；v5 的 A_eff 训练自 edra_process_v4（500 Hz 旧范式）而 GC 参考为 fmt（250 Hz），跨库对照为用户明示口径（通道序同源 geometry_edra_fmt.pt，对齐成立）。
- **baseline refs 归档 `npi/refs/`（用户明示，四项要求：方法/参数分目录、保留原名、记录来源与口径、不覆盖）**：原库 GC 在 edra_fmt 48 例的 EC 矩阵自 `outputs/10-19-latent-flow-edra-fmt-train/run_01/gc_oldlib_refs` 归档至 `refs/granger_bivariate_spectral/edrafmt48-cont-zscore-fs250-fixed8-f1-125-nf65/`（48 npz + gen_refs.py/log + SHA256SUMS 逐位校验 + meta.json；固定阶 8/n_freq 65/1–125 Hz/fs 250/全长连续 60001 样本/逐通道 z-score；ec[a,b]=a→b 行=源）。MANIFEST.md 建档（追加式，新方法/新参数另建子目录）。
- **NI-GC 方法设计交付（用户明示五要素，设计非执行；文档 v1.4 更名 `{TASK_DIR}/ni-mse-design.md`，下同）**：`{TASK_DIR}/ni-mse-design.md`——信号空间无偏噪声注入 + latent_flow 原库 GC 读出 + posterior null（future 前 0.1s MSE，K_null=256）Q90 逐靶通道门控 + ΔEC 双轨推断；含完整流程表（A–H 步骤）、异常规则 R1–R6、种子树（master 2026）、计算量分期与讨论（可交换 null/逐通道 τ vs bisect 共享阈值教训/0.1s 窗敏感性/多重比较 Q95+FDR 变体/拼接口径/带宽适配）。执行属后续另立任务。（注：本条为 v1.0 交付时点记录；v1.1–v1.4 修订见后续各条，GC 重算轨/拼接/带宽等段已在 v1.4 废除。）
- **NI-GC v1.1/v1.2 修订（用户四点质询 + 前提更正，2026-10-20）**：①阈值流程精简版**用户认可**为新默认（Q90 点估计 + split-half + Q95 旗标；bootstrap/种子重跑/c 无关性/合并 τ/FDR 降级可选臂，损失-缓解表随档）；②K_null 256→64（Run 007 pilot：精度只依赖池 W×K，全窗外推 CV 0.26%）；③慢传导敏感性臂（0.2s/全窗）用户拍板取消（漏检风险接受）；④**σ_p 前提更正（用户指出）**：注入发生在数据标准化之后、模型输入之前 ⇒ σ_p 重定义于**标准化空间**，σ_p ≡ c 无量纲、算子为 `X'[a,:]=X[a,:]+c·ξ`，ŝ_a 从算子中移除、仅作物理单位汇报换算（c=0.5 → σ_p=0.5 标准化空间，等效物理 2.02–3.52/中位 2.46 仅判读用）；默认 c=0.5 不变，§1.2/§2 伪代码/流程表同步。
- **NI-GC v1.3 参数钉定（用户认可全方案，2026-10-20）**：null 侧 **K_null=64 × W_null=200 窗/例**（池 12,800/通道，外推 CV ≈0.67%）；响应侧 **W_pert=50 窗/例**（K=32 不变）；设计补一致性规范 **W_pert ⊂ W_null**（响应窗从 null 窗子集内确定性再抽——τ 噪声地板校准在测响应的同一批窗上）+ 窗子集确定性选择规范（SeedSequence(master,"win-null/pert",subject) 无放回抽取、按原序排列、落 manifest）；计算量修订：MSE 轨 ~22 min + GC 轨分期 <30 min，全程预留 ~1.5 h。执行属后续另立任务。
- **NI-MSE-50-q0.90 v1.4 三项用户更正（2026-10-20）**：①**取消生成信号上重算 GC 整轨**（原要素①的 ΔEC 读法废弃）——EC 侧改为 **|EC_pert|（扰动-推断产物，S/G 矩阵）× real GC refs 直接 Pearson/Spearman 相关**，refs 只读 `outputs/10-19-latent-flow-edra-fmt-train/run_01/gc_oldlib_refs`（npi/refs 归档同源），不重算；口径 = offdiag 3,782 有向对 direct 主（同向无转置）+ flip 检查、距离偏 Spearman 判定轨随附（聚合层级见 v1.4a 勘误）。②**扰动 past 条件化不变量显式化**：ξ 先注入 past，扰动后 past 为 GPR 条件先验（posterior_mean）与 ODE 积分的唯一条件源；禁止输出侧加噪等替代实现；R2 恒等断言即其机器可检形式。③**更名与重定位**：NI-GC → **NI-MSE-50-q0.90**，定位 = **study2 中计算 EC_pert 的一条路径**，完整 study 路径 = **study1-gpfm-mamba-v5 × study2 NI-MSE-50-q0.90**；设计文档更名 `ni-gc-design.md` → `ni-mse-design.md`（Run 007 校准产物目录 ni-gc-calib-01 为历史名保留）；计算量降至全程 ~30 min（GC 轨移除）。
- **NI-MSE v1.4a 勘误（2026-10-20，用户质询后实测核对）**：v1.4 所记「跨库 ⇒ 被试集不同源 ⇒ 无逐被试配对」**错误**——实测 refs 48 例（gc_oldlib_refs）与 v4 status_ok 46 例同属 EDRA 队列（refs 覆盖 v4 全集 45/46，多 {11,21,48} 缺 {56}），**fold0 test 10 例 [12,17,18,24,30,38,44,55,58,61] 与 refs 重叠 10/10**（v1 固化 manifest 只读核对）；逐被试配对无实现阻断，**升为主口径**（EC_pert_s × ec_refs_s，n=10 mean±std，对齐归档 Run 003/005 协议），组级单次降副口径（同 10 例 refs 均值；48 例全均值为敏感性）。跨范式（v4/500 Hz 窗 × fmt/250 Hz 全长）保留为解释性 caveat。ni-mse-design.md §1.1/§5/修订链同步。
- **计划新增：v5-fmt-01 训练任务（用户 2026-10-20 拍板）**：用 **edra_fmt（250 Hz）** 训练 gpfm-mamba-v5，产出新 study1 变体，run-id **`v5-fmt-01`**（与 `v5-main-01`（edra_process_v4/500 Hz）并列，同 outputs 根 `gpfm-mamba-f0/`）；训练完成后以该变体为 θ\* 接入 **study2 NI-MSE-50-q0.90** EC_pert 计算路径（n₀ = ⌈0.1×250⌉ = 25 样本；fmt fold0 划分只读复用 v1-fmt manifest 34/4/10）。**收益（用户明示）**：代理脑与 real GC refs（fmt 48 例）同处 fmt 管线 ⇒ 消除「跨范式配对」caveat，逐被试配对口径更清晰（refs = fmt 全部 48 例 ⇒ fmt test 任意 10 例必然 10/10 在 refs 内）。登记于本任务计划（implement.md S7'），执行属后续任务立项。
- **任务暂停 + 新立项（用户 2026-10-20 拍板）**：本任务（10-19-gpfm-mamba-v4b-gnn-f0）**暂停**——S7' 计划即刻转出为独立任务 **`10-20-gpfm-mamba-v5-fmt-f0`**（v5 变体 × fmt 数据训练 v5-fmt-01，study1；PRD/implement/上下文清单已播种，活动指针已切换）；本任务保留 in_progress 状态待恢复，未竟项 = S4'/S5'/S6' 质检收尾 + Run 017 同口径评估（v1/v2/v5 对照）——恢复时点由用户定。

## Run 006 — v5 A_eff/W 读出 + |A_eff| × 原库 GC 相关（2026-10-20）

### 目的与假设

- v5-main-01 主 Run 后的 A_eff/W 可解释读出（design §3：EMA shadow、转置、距离偏 Spearman 判定轨沿用 v4bg 版处置；协议对齐归档任务 10-19-latent-flow-edra-fmt-train Run 002/003 的 gen_aeff_viz.py / gen_aeff_gc_corr.py）：σ/W 学习结果、A_eff 结构统计（offdiag CV/行和 CV/不对称度/中心-环绕/距离衰减）、M vs W 分解；|A_eff| 与原库 GC EC（fmt refs）逐被试 Pearson+Spearman（方向对齐主口径 + 转置检查）。不做代理脑合格性判定。

### 配置

- 代码：`outputs/gpfm-mamba-f0/v5-main-01/readout/gen_v5_aeff_readout.py`（脚本随档；只读 ckpt 与 refs，只写 readout/ 目录）
- A_eff 源：`outputs/gpfm-mamba-f0/v5-main-01/checkpoints/gpfm-mamba-v5-best.pt` 的 **EMA shadow**（design 口径；raw 对照值随记）
- 重建：`MexHatAdjacency(kernel=ricker, γ=0.1, γ_m=1.0)`（v5 graph.py，几何 choke point 46 例断言）+ shadow 的 graph.\* 子字典 strict 装载
- GC 参考：`refs/granger_bivariate_spectral/edrafmt48-cont-zscore-fs250-fixed8-f1-125-nf65/refs_s*.npz`（48 例；ec[a,b]=a→b 行=源）
- 对齐：A_eff[i,j]=j 驱动 i（行=靶）→ **主口径 |A_eff| vs ec.T**；不转置作方向性检查副口径；每被试 offdiag n=3,782 对；Pearson（np.corrcoef）+ Spearman（scipy）；mean±std(ddof=1)/median/range；组均值 GC 单次相关；minmax 不变性核验（协议沿用）
- 附带（描述性，超出归档协议部分明确标注）：M/γ·W 分量 vs 组均值 GC 的承载分解；读出图三张（A_eff/M/W 热图，对外呈现转置为行=源/列=靶）
- 环境：同 Run 003（CPU 读出）
- 命令：`python3 outputs/gpfm-mamba-f0/v5-main-01/readout/gen_v5_aeff_readout.py`

### 结果

**A_eff 结构统计（EMA shadow，ep64 best ckpt；对照 latent-flow run_01 同协议读出值）**：

| 指标 | v5-main-01 | latent-flow run_01（归档） |
|---|---|---|
| σ 学习值 / σ÷中位电极距 | **0.14190 m / 1.174** | 0.14555 / 1.204 |
| A_eff offdiag CV | 1.686 | 1.164 |
| A_eff 行和 CV | **0.307** | 0.010（近均匀） |
| A_eff 不对称度（Fro） | **0.0237**（近对称） | 0.258 |
| W 不对称度（Fro）/ W offdiag 均绝值 | 1.357 / 0.0609 | — / 1.892 |
| offdiag 正分量占比 | 0.652 | 0.839 |
| σ内均值 / σ外均值 | **+0.446 / −0.181（环绕净抑制显现）** | +0.268 / +0.011（近零） |
| \|A_eff\|×距离 Pearson / Spearman | **−0.841 / −0.717**（强距离衰减） | −0.884（Pearson） |
| 对角均值 | 0.994 | 0.802 |

- **M vs W 分解（design §8 风险 5 读出）**：γ_m·‖M‖_F = 28.41 vs γ·‖W‖_F = 0.495——**W 仅占 A_eff 的 1.7%**（Fro 与 offdiag 均值两口径一致 1.72–1.74%）；几何先验主导，W 为小残差修正。
- **|A_eff| × 原库 GC EC（48 例，offdiag 3,782 对/例）**：
  - 主口径（对齐，|A_eff| vs ec.T）：Pearson **+0.4386 ± 0.0873**（median +0.4465，range [+0.1187, +0.5967]，48 例全正）；Spearman **+0.2510 ± 0.0616**（range [+0.0635, +0.3749]）
  - 方向性检查（不转置）：+0.4387 / +0.2510——**与主口径无差异**（有向分量不携带与 GC 方向一致的信息）
  - 组均值 GC 单次相关：Pearson +0.5831 / Spearman +0.3139
  - **分量承载分解（组均值，描述性附加）**：γ_m·|M| Spearman **+0.3141** ≈ A_eff 全量 +0.3139；γ·|W| +0.0459——**相关几乎全部由对称距离核承载**
  - minmax 不变性核验通过（全 48×4 序列 max|Δr| < 1e-12）
  - 对照 latent-flow run_01 同读出（归档 Run 005）：Pearson +0.4414±0.0929 / Spearman +0.2012±0.0493、组均值 +0.5849/+0.2529——**v5 与其同水平**（同为几何驱动的结构先验）
- 产物：`outputs/gpfm-mamba-f0/v5-main-01/readout/`（aeff_stats.json + aeff_vs_gc_stats.json + aeff_{heatmap,kernel_M,residual_W}.png 三图（行=源/列=靶转置呈现）+ gen_v5_aeff_readout.py 脚本随档）
- 图表：三热图（见上）；GC 散点图未做（归档协议亦无）

### 结论

- 支持（A_eff 结构先验成立）：强距离衰减 + 局部兴奋/**环绕净抑制**（σ外 −0.181，v5 的 ricker 环绕抑制未被 W 补偿抹平——latent-flow run_01 该处近零）+ 行和 CV 0.307（路由异质，非各向同性平均化）；σ 学习 0.142 m 与 latent-flow 同量级。
- **风险 5（W 稀释几何先验）实测不成立**：W 仅占 1.7%，几何先验主导——但反面是 **W 的有向信息量小**（A_eff 不对称度仅 2.4%，×GC 相关的承载分解 γ·|W| +0.046）；「可学习有向残差」机制在 v5 训练动态下收敛为小修正项（γ=0.1 + 无正则口径）。
- |A_eff|×原库 GC ≈ +0.44/+0.25（两指标、两方向对齐一致）与 latent-flow run_01 等水平；判读注意沿用归档结论：Pearson≫Spearman 系重尾幅度驱动；aligned=flipped + 分量分解 ⇒ **相关载体是对称距离核，不能解读为方向性功能连接一致**；A_eff 编码空间结构先验而非被试特异功能连接。
- 边界：本读出 ≠ H4' 判定（GC 载体判据 = Run 017 同口径评估的 GC(gen) vs GC(real) 距离偏 Spearman 判定轨，未跑）；≠ 代理脑合格性判定。
- **勘误（Run 005 结论口径）**：Run 005 所记「γ·‖W‖_F ≈ 0.509 vs γ_m·M ≈ 1.0——W 已达几何先验项一半量级」系量纲误配（把 γ_m=1.0 的尺度因子当作 M 的范数量级；‖M‖_F 实为 28.4）——正确口径见本 Run「M vs W 分解」：**W 占 A_eff 约 1.7%**。「W 学得动」（范数离开零）结论不受影响，「一半量级」表述作废。

### 图表说明

三热图（A_eff/M/W，行=源/列=靶）：A_eff 呈现以对角自连接（~1）+ 近邻兴奋 + 远距抑制的带通几何结构；M 为标准 ricker 带通核；W 幅度小（±0.36 max）呈细粒度有向纹理。

## Run 007 — NI-GC 设计校准 pilot（null Q90 精度 vs K_null；轻量统计，非训练/评估）（2026-10-20）

### 目的与假设

- ni-gc-design.md 的四项用户质询取数：①σ_p 实际幅度（读 run 归一化统计即可）；②K_null 候选 {32/64/128/256} 的 Q90 估计精度与耗时代价——用 v5-main-01 θ\* 真实 posterior 采样实测（1 被试 64 窗，B_ref=64 + 256 null 独立采样，子采样重抽 200 次 + split-half），不拍脑袋；④简化阈值流程的精度依据（split-half 单项检查的覆盖力）。产出喂给设计文档 v1.1 修订。

### 配置

- 代码：`outputs/gpfm-mamba-f0/ni-gc-calib-01/gen_calib.py`（脚本随档；只读 ckpt，只写本目录）
- 模型：v5-main-01 best（EMA swap 装载，config_hash 校验 `1ac433fdd3d0`）；v1 sample.py 的 `EulerSampler`（eps/eta 显式，gen_lowpass 40 Hz 正式口径）
- 数据：edra_process_v4 fold0 test 10 例的**确定性 loader 前 64 窗**（batch 16；W=64 校准口径，非全量）
- 统计：N 定义 = 前 50 样本（0.1s@500Hz）逐通道 MSE vs B_ref=64 均值参考；分析 = 逐通道 Q90（ECDF 线性）对 K_null ∈ {32,64,128,256} 子采样（200 次）→ CV(Q90) 中位/p90 + split-half max|Δτ|/τ 中位；W 缩放臂（K=64 固定，W ∈ {8,16,32,64}）
- 种子：ref=20260 / null=20261 / 分析 rng=2026（master 2026 派生）
- 命令：`python3 outputs/gpfm-mamba-f0/ni-gc-calib-01/gen_calib.py`

### 结果

- 运行：ref（B_ref=64×64 窗）4.28 s；null（256×64 窗）16.98 s → **实测采样速率 ≈1.04 ms/(窗·条)**（v1.0 设计按训练环 44ms/批外推的 ~1.5 h 全量预算据此修订为 ~8 h）。
- **K 表（W=64 固定，CV(Q90) 中位 / split-half max|Δτ|/τ 中位）**：K=32（池 2,048）1.78% / 8.9%；K=64（4,096）**1.18% / 6.8%**；K=128（8,192）0.69% / 4.8%；K=256（16,384，基准）— / 3.6%。**W 表（K=64 固定）**：池 512/1k/2k/4k → CV 5.54/3.90/2.47/1.21%——**两轴同 √n 收缩律，精度只依赖池 W×K**；全窗 W≈1,308/例外推：K=64 池 ≈8.4 万/通道 → CV ≈0.26%、split-half ≈1.5%。
- 产物：`outputs/gpfm-mamba-f0/ni-gc-calib-01/`（calib_results.json + gen_calib.py；含 tau_full 中位与 null 池统计）。
- 实现注记（校准中发现，非偏离）：①v1 sample.py 会把自身目录插到 sys.path[0]——pilot 须按 v5 train.py 契约**先占 sys.modules[config/model/graph] 再 import sample**（首次尝试未预占即装载到 v1 model，结构断言拦截，修复后过）；②采样侧 reflect pad 有 CUDA 网格上限，(W,K) 一次前向须分块 ≤8×64（v1 评估单批 4×16 量级同因）。

### 结论

- 支持（设计参数修订依据）：K_null 256→**64**（全窗池下 CV 0.26%，split-half 1.5% ≪ 10% 触发线）；抽窗子集（200 窗/例）时 K=64 → CV 0.67% 仍充裕，缩窗与缩 K 在池量上等价可互换。
- ni-gc-design.md 已修 v1.1：σ_p 语义/实测幅度（c=0.5 → 物理噪声 std 2.02–3.52、中位 2.46；z 空间 0.5；注入方差占通道总方差 20%）；K_null 表入 §1.3；慢传导敏感性臂（0.2s/全窗）**用户拍板取消**（漏检风险接受，只留 0.1s 窗）；阈值流程精简版为新默认（Q90 点估计 + split-half + Q95 旗标；bootstrap CI/种子重跑/c 无关性/合并 τ 交叉/FDR 变体降级可选强化臂，损失与缓解表随档）。

## Run 003 — v5 冒烟（smoke：限批全链路）（2026-10-20）

### 目的与假设

- 验证 v5 变体（v2 基线 + 4×[Mamba→GNN 空间块→LayerNorm] 交替）经 v1 训练循环的全链路机械正确性：三模块双遮蔽装载、前向/反向、损失有限、val4 内部验证、best/last checkpoint 落盘。不做任何质量判读。

### 配置

- 代码：`models/gpfm-mamba-v5/`（git 22afbc5 工作树；train.py 三模块双遮蔽包装 + 运行时一致性断言）
- 参数：全部默认（d_model=62、4 交替块、graph ricker/γ=0.1/γ_m=1.0/W 零初始化/λ_u=λ_s=0、geometry_edra_fmt.pt；batch 256、seed 1234、EMA 0.999、恒定 lr 1e-3、早停 10、gen_lowpass 40 Hz；划分 32/4/10，manifest 只读复用 v1 固化 fold0）；`--smoke` = 每类 2 例（合计 4 例）1 epoch
- 环境：base conda（Python 3.12.3 / torch 2.8.0+cu128 / mamba_ssm 2.3.1）@ 本地云服务器（RTX 4080 SUPER 32 GB，16 核 cgroup，BLAS 限 8 线程）
- 数据：`/root/autodl-tmp/edra_process_v4`（status_ok 46 例，fold0 train 折冒烟子集 4 例）；几何 `/root/autodl-tmp/edra_processed_fmt/geometry_edra_fmt.pt`（只读）
- 命令：`python3 models/gpfm-mamba-v5/train.py --smoke --run-id v5-smoke-a`

### 结果

- 指标：params **152,647**（[model] 行与单测⑤一致）| train 4 例 5,284 窗 20 step，全损失有限（total 1.4691→1.2928）| val2 2,642 窗 total **1.22209**（新 best；mae 0.6280 / cons 0.5344 / tv 0.2172 / corr 0.3795；x̂1 std(EMA) 0.8152 vs x1 std 0.9717）| it/s 收尾 9.6（预热爬坡中）| peak **2.315 GiB**（vs v4bg smoke 14.633 GiB——无 CI 折叠膨胀，与单测 GPU 冒烟 2.38 GiB 一致）| config_hash `fbc0fbefd08a`（smoke 覆写口径）
- 产物：`outputs/gpfm-mamba-f0/v5-smoke-a/`（config.yaml + env_snapshot.txt + channel_mean/std.npy + logs/{train.log, metrics.jsonl, summary.json} + checkpoints/{gpfm-mamba-v5-best,last}.pt 八类齐）
- 图表：无（冒烟不做图）

### 结论

- 支持（机械正确性）：v5 交替结构经 v1 训练循环全链路（三模块双遮蔽装载/前向反向/损失有限/val 确定性复算/best+last 落盘）无异常；显存 2.3 GiB 余量充裕。

## Run 004 — v5 1-epoch 基准（bench：32 例全量）（2026-10-20）

### 目的与假设

- 主 Run 前材料登记（用户已直训授权，非门禁）：参数量、峰值显存、it/s、val4 内部验证总 loss、100-epoch 外推时长、W/σ 可学性点值证据（ckpt 事后提取 raw + EMA 双口径——v4bg 逐步仪表未移植 v5 包装，偏离登记见后）。不做质量判读。

### 配置

- 代码：同 Run 003
- 参数：同 Run 003 唯 `--epochs 1 --bench`；训练折全量 32 例
- 环境：同 Run 003
- 数据：同 Run 003（train 32 例全量 + val 4 例内部验证）
- 命令：`python3 models/gpfm-mamba-v5/train.py --epochs 1 --bench --run-id v5-bench-1ep`

### 结果

- 指标：params **152,647** | train **34 例 44,914 窗 175 step**（fold0 36 例训练折 − 2 例内部验证；v1/v2 同款 34/2 口径，manifest 校验 v1 固化文件只读）| train total 1.2549 | **val total 1.10462**（新 best；mae 0.5723 / cons 0.4721 / tv 0.2014 / corr 0.4011；x̂1 std(EMA) 0.7371 vs x1 std 0.8373）| **10.3 s/epoch（train 段）**| **it/s 16.98**（收尾 19.6）| **peak 2.315 GiB** | 100-epoch 外推 **≈0.286 h**（train 段；加逐 epoch val/checkpoint 实际约 0.4–0.7 h 量级）| config_hash `fbc0fbefd08a`（bench 覆写口径）
- **W/σ 可学性点值（best ckpt 事后提取，raw + EMA shadow 双口径）**：W (62,62) 零初始化 → 1 epoch（175 step）后 raw **‖W‖_F = 1.2356** / EMA shadow **0.1329**（decay 0.999 下滞后量级自洽：0.161×running avg）/ raw max|W| = 0.0821；σ init=median_d 0.120868 → raw **0.126186** / EMA **0.121400**（微调中）；γ·W 对 A_eff 贡献尺度 = 0.1×1.24 ≈ **0.12**（vs γ_m·M ≈ 1.0——1 epoch 时 W ≈ A_eff 的 12%，增长斜率缓于 v4bg head 版同期 0.40，跨块共享下步长分布更稀）
- 产物：`outputs/gpfm-mamba-f0/v5-bench-1ep/`（八类齐同 Run 003 布局；best ckpt epoch 0 / global_step 175 / best_val 1.10462）
- 图表：无（bench 不做图）

### 结论

- 支持（主 Run 材料齐备）：①机械/资源可行——显存 2.3 GiB、17 it/s、主 Run 预计 ~0.3–0.7 h；②**W 学得动（design §8 风险 2 缓解）**——零初始化 1 epoch raw ‖W‖ 0→1.24；③v4bg 版 §8 风险 5（W 稀释几何先验）留主 Run 读出期看 A_eff 行和分布/M vs W 分解；④1-epoch val 1.105 不作质量判读。
- **偏离登记（vs design §S3'/Run 002 v4bg 版口径）**：v5 包装未移植 v4bg 的逐步 W 范数轨迹仪表（运行时 EMA 子类注入为 v4bg train.py 特有实现），bench 改为 ckpt 事后提取点值双口径替代——证据强度弱于逐 step 轨迹（无斜率/单调性信息），主 Run 期间如需轨迹属代码增改另登记。

## Run 005 — v5 主 Run（main：100 epoch 上限 + 早停 10）（2026-10-20）

### 目的与假设

- H4' 主判定材料：v5（v2 基线 × 轴分解交替）fold0 单 seed 完整训练——收敛曲线、best val、早停轮次；训练后接 Run 017 同口径评估（对照 v1/v2/v5）与 A_eff/W 读出（判定轨距离偏 Spearman）。R9 确认 = 用户 2026-10-20 直训指令（登记于「用户拍板登记」节）。

### 配置

- 代码：同 Run 003（git 22afbc5 工作树）
- 参数：全部默认（= Run 003 口径；epochs 100、早停 10、batch 256、seed 1234、EMA 0.999、恒定 lr 1e-3、gen_lowpass 40 Hz 正式口径；无 --smoke/--bench 覆写，config_hash 应回默认 `1ac433fdd3d0`）
- 环境：同 Run 003
- 数据：同 Run 004（train 34 例全量 + val 2 例内部验证，test 10 例留评估）
- 命令：`mkdir -p outputs/gpfm-mamba-f0/v5-main-01/logs && nohup python3 models/gpfm-mamba-v5/train.py --run-id v5-main-01 > /root/npi/outputs/gpfm-mamba-f0/v5-main-01/logs/train.log 2>&1 &`（脱离式 + 显式重定向）

### 结果

- **跑满 100/100 epoch**（总墙钟 ~17 min，mean 9.76 s/epoch train 段，17.93 it/s，peak 2.315 GiB，commit 22afbc5）
- **best val total 0.73155 @ epoch 64**（best ckpt global_step 11,375）；末 epoch val 0.73426；best 后 35 epoch 无新 best、均在 best+0.003 内窄幅波动（0.733 量级平台）
- 收敛轨迹（val total）：ep0 1.10461 → ep10 0.88893 → ep25 0.75653 → ep50 0.73318 → **ep64 0.73155** → ep99 0.73426；train total 1.2549→0.7790；末 epoch val 分量 mae 0.4166 / cons 0.2763 / tv 0.1367 / corr 0.2775；x̂1 std(EMA) 0.7347 vs x1 std 0.8373
- config_hash **`1ac433fdd3d0`**（与 S1' 登记的默认 pin 逐位一致，无覆写漂移）
- **W/σ 终值（best ckpt，raw + EMA shadow 双口径）**：‖W‖_F raw **5.0901** / EMA **4.9451**（EMA 已追平 raw——充分训练后双口径收敛一致）；raw max|W| = 0.3594；σ raw **0.142705** / EMA **0.141899**（init = median_d 0.120868，+18%）；**γ·‖W‖_F ≈ 0.509 vs γ_m·M ≈ 1.0**——可学习有向残差已达几何先验项一半量级，A_eff 为实质性双成分（design §8 风险 5「W 稀释几何先验」的读出期分解对象）
- 产物：`outputs/gpfm-mamba-f0/v5-main-01/`（八类齐：config.yaml + env_snapshot.txt + channel_mean/std.npy + logs/{train.log, metrics.jsonl, summary.json} + checkpoints/{gpfm-mamba-v5-best,last}.pt）
- 图表：无（训练 run 不做图，评估阶段出）

### 结论

- 支持（训练完成、材料就绪）：全 100 epoch 稳定收敛、无发散/NaN/损失 spike；best ckpt（ep64）为后续评估的 θ\* 候选；W 与 σ 均离开初始化且 EMA 收敛一致（design §8 风险 2/3 缓解证据补强）。
- 边界：val 内部验证总 loss ≠ 质量/代理脑合格性判定；**Run 017 同口径评估（对照 v1/v2/v5）与 A_eff/W 读出（距离偏 Spearman 判定轨）未跑**，属本 Run 之后的步骤。
- **勘误登记（跑前文案 vs 实跑口径，三项）**：①**早停未启用**——v5 config 为 v2 副本、无 `early_stop_patience` 字段，v1 train.py getattr 默认 0=禁用（该钩子系 v4-B 口径，v4 家族字段），跑满 100 属口径内行为，best ckpt 语义等价于早停产物；②**lr 实为 cosine 退火**（T_max=epochs=100，1e-3→2.47e-7）——v2 副本无 lr_schedule 字段、v1 循环默认 cosine，与 v2d62-main-01 同口径（对照公平性不受影响）；③划分口径 = fold0 训练折 36 例 − 内部验证 2 例 = **34 例训练** + test 10（Run 003/005 跑前登记中「早停 10 / 恒定 lr 1e-3 / 32/4/10」为 v4bg→v5 转向时 config 家族切换的文案残留，以产物 config.yaml 与本勘误为准）。

## 归档登记（2026-10-21）

- 用户拍板归档本任务；**未竟随档：Run 017 同口径评估（对照 v1/v2/v5）与 A_eff/W 读出（距离偏 Spearman 判定轨）**，如需执行另立新任务（θ\* 候选 = `outputs/gpfm-mamba-f0/v5-main-01/` best ckpt ep64，config_hash `1ac433fdd3d0`）。
- 收尾提交范围：`models/gpfm-mamba-v4b-gnn/`（S1–S2 完成后二次转向停用留档）+ `models/gpfm-mamba-v5/`（S1' 交付 + v5-main-01 训练已跑）+ latent-flow `baseline/data_utils.py` npz 加载分支 + `viz/connectivity.py` `plot_matrix_heatmap`（A_eff/W 读出绘图）。
- 归档前单测复核：v4bg **12/12**、v5 **10/10** 全绿。**坑登记**：两变体目录均无包结构，同一条 pytest 命令混跑两目录会因顶层模块名（`model`/`config` 等）互相遮蔽造成批量假失败（实测 v4bg 7 项 red，报错显示的是 v5 的 152,647 参数模型）；须逐目录单独跑（本仓回归惯例即分目录执行）。
