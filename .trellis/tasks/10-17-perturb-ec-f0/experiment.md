# 实验日志 — 10-17-perturb-ec-f0（fold0 虚拟扰动→EC 有效路径推断流程，v4a-fmt 冻结 θ\*）

> 每次「跑批/训练/评估/统计」追加一个 Run 块：跑前先写「目的+配置」，跑后同会话补「结果+结论」，不留空块。
> 格式契约见 `.trellis/spec/research/experiment-log.md`；产物只落本任务声明的变体目录（`.trellis/spec/research/path-isolation.md`）。

## Run 001 — smoke-01：1 例（subject 10）× 2 窗全链扰动→D→τ→readout 骨架冒烟（2026-10-17）

### 目的与假设

- 验证 implement.md B1：冻结 θ\* + 真实 fold0 test 数据上「虚拟扰动（497 并批 CRN 采样）→ 窗级 D 曲线 → null 池 → τ 分位 → readout 骨架（slope/AUC/r\*）」全链跑通（1 例 × 子采样前 2 窗，分钟级）。
- 检查点（implement.md B1 原文口径）：① D 曲线无 NaN；② A=B 自响应显著大于 A≠B（**定性**——B0 冒烟曾观察 r=1.28 自响应 0.316 vs 跨通道 0.227 仅 ~1.4×，可能「不显著」，如实登记并按 design §7 Risk#1 语义解读：跨通道仅速度网络单通路 + rank-16 瓶颈，弱传播本身是合法结论方向）；③ CRN 恒等运行时复验（r=0 变体 vs baseline 同批逐位相等）；④ 额外登记跨通道边 r_max 达 τ 比例（删失率粗估，smoke 级）。
- 假设：剂量-响应曲线单调不减、低剂量段近线性；null 地板与响应分离可辨。异常（非单调/NaN）→ design §8 回滚点②（停，诊断后重设计网格或算子并登记）。

### 配置

- 代码：`models/perturb-ec-v1/`（B0 提交 + B1 增量 `run.py --stage smoke`）+ `common/visualizer.py` 纯增量扩展 dose_response 图类型；git HEAD 回填见结果节。
- **四要素口径（design §2 预注册；本块 = 首次 Run 级登记，主 Run（B3）定版前不再变更档位）**
  - **P 扰动算子 = A 案增益调制**（用户 2026-10-17 拍板）：`x_A(t) ← (1+r)·x_A(t)`，t ∈ past 全段 384 样本 @250 Hz；注入层 = 模型输入层 X_past 通道 A；主流程仅正方向 r ≥ 0；B 案（加性带限波形）文档级预留不实现。幅度域外推警示：r ≥ 0.32 属网络外推行为（slope 主段 r ≤ 0.08 受保护）。
  - **D 响应度量 = CRN 配对归一化 RMS**：`D_{A→B}(r;w) = RMS_{k,t}(x̂^{(A,r)}_{w,k,B} − x̂^{(0)}_{w,k,B}) / s_{w,B}`，`s_{w,B} = RMS_{k,t}(x̂^{(0)} − mean_k)`（baseline 生成分布自身波动尺度，纯模型侧）。批采样纪律：每窗**一次** `sample()` 调用，baseline + 62 源 × 8 档 = **497 并批**（chunk_bk=1024）；**逐窗一组 eps/eta 跨 (A, r, baseline) 全共享**，seed = 2026 + 窗序号（win_idx = 全局窗序号；smoke 迭代顺序 = **例序 EXPECTED_TEST_IDS 升序（首例 = subject 10）** + 例内子采样索引升序、从 0 连续——与主 Run 预定迭代顺序一致，smoke eps/eta = 主 Run 前缀；勘误：test_ds 自然序首例为 44，非 ID 升序，故例序显式取 ID 升序）。
  - **τ 阈值 = 逐被试分池 null 95% 分位**（q90/q99 敏感性）：null = 同窗 K=8 条 baseline 两两 **28 对**同构距离 `d(i,j) = RMS_t(x̂_i − x̂_j)/s_B`（与 D 同构、扰动为零）。smoke 池 = 2 窗 × 62 通道 × 28 对 = 3472 样本——**smoke 级小样本，τ 仅形态参考**（主 Run 池 ≈ 5.6 万/例）。
  - **临界强度搜索 + readout**：R_GRID 9 点 {0, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28}（对数等比 ×2；r=0 由 CRN 恒等按构造 D=0，不采样）；R_NEXT = R_MAX×R_RATIO = **2.56**（由 R_GRID 推导，右删失编码与 rstar 翻转偏移共用）；slope 主段 = R_GRID[:5] = {0, 0.01, 0.02, 0.04, 0.08}（敏感性段 [:4]）；AUC = 全网格梯形 / R_MAX。
- **固定口径（config.py / design §3）**：K = 8 条 CRN 样本/窗（null 28 对/窗/通道；与注册/评估 K=16 的差异属设计差异——本任务为扰动配对推断）；N_w = 32 窗/例（**smoke 仅取子采样前 2 窗**）；窗子采样实现口径 = 逐例 `default_rng(2026×1000003 + subject_id)` 均匀**无放回**抽 N_w 窗、升序（保窗时间顺序）。
- **实现口径声明**：`gen_lowpass_hz = None`（v4a-fmt cfg 显式透传；构造器默认 40.0 + brickwall_lowpass 默认 fs=500 对 250 Hz 数据有效截止 ~20 Hz 会静默削波 D——B-Major1 陷阱已规避，运行时再断言 `sampler.gen_lowpass_hz is None`）。**方向约定（全仓库统一）**：EC/D 张量**行 = 源 A、列 = 靶 B**；D 曲线索引 [A, r 档 j, B]。smoke 聚合口径：D̄ = 2 窗均值（主 Run 为逐被试 median_w，smoke 不做受试者级 median）。
- 环境：本地（AutoDL 云服务器）base conda Python 3.12.3 / torch 2.8.0+cu128 / RTX 4080 SUPER 32 GB（environment.md 2026-10-10）；BLAS 护栏 8 线程（run.py 入口 numpy 前 setdefault）。
- 数据：θ\* = `outputs/gpfm-mamba-fmt-f0/v4afmt-main-01/checkpoints/gpfm-mamba-v4a-fmt-best.pt`（best ep45，config_hash 05d9bb875a2c，EMA 权重，只读）；fold0 test 第一例 **subject 10**（EdraWindowsDataset 只读 past 384 / future 128 / stride 64 @250 Hz，评估侧 z-score = run 目录 channel_{mean,std}）。
- 命令：`python3 models/perturb-ec-v1/run.py --stage smoke --run-id smoke-01`（前台分钟级；产物 `outputs/perturb-ec-f0/smoke-01/`：d_curves.npy (2,62,8,62)、null_pool.npy (2,28,62)、config_snapshot.json、logs/run.log、figs/dose_response_smoke.png）

### 结果

- 代码版本：git HEAD **5e66c88**（B0 提交）+ B1 未提交增量（`run.py` smoke 主体、`common/visualizer.py` dose_response 图类型）；运行 19.6 s（含装载；每窗 497 变体采样 ~5 s），前台单卡。
- 执行事实：subject 10 子采样前 2 窗 = 全局索引 [8417, 8452]（win_idx 0/1）；D (2,62,8,62)、null (2,28,62)，窗级 D 索引 [A, r 档, B]。
- **τ（2 窗 null 池 3472 样本）**：q90=1.7449、q95=1.8200（主口径）、q99=1.9448（median 1.4940，分布窄——null 集中于 ~√2·s）。
- **检查点逐项判定**：
  - ① 曲线无 NaN：**通过**（D/null 全有限非负）。
  - ② A=B 自响应显著大于 A≠B（定性）：**不显著，如实登记**——r=1.28 处自响应 median **0.3275**（Q1 0.2518 / Q3 0.4397）vs 跨通道 median **0.2327**（Q95 0.5095、max 0.9401）；median 比仅 **1.41×**，**26.4%** 跨通道边 ≥ 自响应中位（分布重叠大）。与 B0 冒烟观察（0.316 vs 0.227）一致。按 design §7 Risk#1 语义解读：先验逐通道 + CRN 零随机贡献 ⇒ 跨通道仅速度网络单通路（rank-16 瓶颈），扰动传导**空间弥散**——佐证：低剂量段（r≤0.08）自/跨 slope median 0.1986 vs 0.1975 几乎相等。回滚点②不触发（形态无异常）。
  - ③ CRN 恒等复验：**通过**（窗 0 真实数据、a=0、r=0、同批 B=2 逐位相等，运行时再断言）。
  - ④ 删失率粗估：跨通道 3782 边在 r_max 达 τ 比例 = **0.0%**（q90/q95/q99 全部 **100% 右删失**）——D̄(r_max) 跨通道 max 0.9401 << τ q90 1.7449。
- 曲线形态定性（好于预期）：跨通道 Spearman(D̄, r) median=**1.000**（Q1/Q3 均 1.000），8 档完全非降比例 **100%**（3782 边全部完美单调）；低剂量段近线性。剂量-响应表（2 窗均值 D̄ median，自/跨）：

  | r | 0.01 | 0.02 | 0.04 | 0.08 | 0.16 | 0.32 | 0.64 | 1.28 |
  |---|---|---|---|---|---|---|---|---|
  | 自响应 | 0.0028 | 0.0055 | 0.0111 | 0.0222 | 0.0445 | 0.0900 | 0.1778 | 0.3275 |
  | 跨通道 | 0.0020 | 0.0040 | 0.0079 | 0.0158 | 0.0315 | 0.0624 | 0.1234 | 0.2327 |

- readout 骨架（2 窗均值口径）：slope 跨通道 median 0.1975 / max 0.7993；AUC median 0.1205；r\* 全删失（median = R_NEXT 2.56）。
- 产物：`outputs/perturb-ec-f0/smoke-01/`：d_curves.npy (2,62,8,62)、null_pool.npy (2,28,62)、readout_smoke.npz（slope/auc/rstar×3τ + τ 值）、config_snapshot.json（全常量+ckpt+env+窗索引+seed 公式）、smoke_metrics.json（机器可读汇总）、logs/run.log。
- 图表：`outputs/perturb-ec-f0/smoke-01/figs/dose_response_smoke.png`——4 条代表曲线（self A=11 红；跨通道强 12→46 / 中 23→20 / 弱 27→15）+ τ q95 主线（q90/q99 敏感细线）；y 轴全范围直接可见「全部响应曲线远低于 τ」的分离缺口（图内注记 smoke 级小样本口径）。

### 结论

- **部分支持**（流程可跑通 = 支持；「自响应显著大于跨通道」= 不支持）：
  1. **B1 放行**：全链（497 并批 CRN 采样 → D → null → τ → readout 骨架 → 图）机械正确性成立——无 NaN、r=0 逐位恒等、全边完美单调 + 低剂量近线性；design §8 回滚点②不触发，进入 B2 bench。
  2. **Risk#1 信号坐实（smoke 级）**：(a) 自/跨分离不显著（1.41×，26.4% 跨边超自中位）——θ\* 扰动传导空间弥散、通道特异性弱；(b) r\* readout 当前 τ 口径下全右删失。结构性注记：null = 不同 eps 的独立样本对距离（集中 ~√2·s、窄分布），CRN 配对 D 消除蒙特卡洛抖动后**确定性响应天然远小于该地板**——「D < τ」≠「无响应」，是「响应幅度 < 生成过程自身样本间波动」；slope/AUC readout 不受影响（主 readout 设计上受保护，design §2.4）。
  3. **不构成口径变更**（四要素口径维持预注册，主 Run 定版前不动档位）：删失率与 τ 形态以主 Run（N_w=32、median 聚合）为准复核；若维持全删失 → 按 AC6 缺口诊断路径登记，档位调整权（网格上界扩档 + §2.1 外推重登记）是否启用留 B2 后决策。
- 下一步：B2 bench（1 例 N_w=32 计时外推 + GPU 显存峰值 + N_w=32 vs 930 全集对照抽查）→ 档位定版决策点（同步 .trellis/research/progress.md 由主会话统一）。

## Run 002 — bench-01：1 例（subject 10）N_w=32 计时定档 + 子采样代表性抽查 + τ 决策数据预览（2026-10-17）

### 目的与假设

- implement.md B2：subject 10 全 32 窗扰动链计时（每窗一次 `sample()`、497 并批）→ **10 例（320 窗）外推总时长** + GPU 显存峰值（`torch.cuda.max_memory_allocated/_reserved`）→ 档位维持/降档判定（design §3 档位纪律：**原则维持**，仅外推超 4 h 或显存超限才建议降档；档位调整权只有一次且留给 τ/网格决策）。
- 代表性抽查（design §7-7）：同例 930 窗全集 **baseline 侧**量（逐窗逐通道 s_{w,B} 与 null 距离分布）vs N_w=32 子样本同分布（QQ 散点 + 分位数表 + KS）——扰动侧全集成本高，抽查限 baseline 侧可得量（design §3 原文口径）。
- **τ 决策数据预览（只算不判）**：(a) 预注册 τ（独立样本对池 q90/95/99）下删失率（跨通道 3782 边）；(b) 相对自响应口径预览 D̃_{A→B}(r) = D̄_{A→B}(r)/D̄_{B→B}(r)（同 r、逐边）：r∈{0.08,0.32,1.28} 分布（median/Q1/Q3/max）+ 若 τ̃=0.25/0.50 的 r\* 删失率；(c) 低剂量段 slope 跨通道分布（median/Q1/Q3/max/min）。全部标注「**预览、非 readout 承诺、τ 口径待用户拍板**」。
- 假设：外推时长 << 4 h（smoke ~5 s/窗 → 320 窗 ~27 min 采样）；N_w=32 子样本在 baseline 侧量上与 930 窗全集同分布（中央分位数相对偏差小）；预注册 τ 下删失率仍≈100%（Run 001 已见全删失）。

### 配置

- 代码：`models/perturb-ec-v1/`（B0 提交 5e66c88 + B1 未提交增量 + B2 增量：`run.py --stage bench`、`response.py` 批量 baseline 扫描、`loader.py` 受试者窗区段助手、`common/visualizer.py` 纯增量 qq_compare/tau_preview 图类型）；git HEAD 回填见结果节。
- **四要素口径 = Run 001 全部不变**（P=A 案增益、D=CRN 配对归一 RMS 497 并批、τ=null 逐被试分池 95% 分位（q90/q99 敏感性）、R_GRID 9 点 / K=8 / N_w=32 / seed 公式与方向约定逐字沿用）。本 Run 与 smoke 的唯一区别 = 窗数（前 2 窗 → 全 32 窗，同子采样种子，故 smoke eps/eta = 本 Run 窗 0–1 前缀）。
- 主链：subject 10 全 32 窗（`--subjects 1`），每窗一次 `sample()`（baseline + 62 源 × 8 档 = 497 并批，chunk_bk=1024），eps/eta seed = 2026 + 窗序号（0..31）；逐窗计时（采样段/后处理段分开，CUDA synchronize 包夹）落 logs/run.log + timing.json；GPU 峰值循环前 `reset_peak_memory_stats`。
- 代表性扫描（bench 专用附加，**不进主 Run 外推成本**）：subject 10 全部 930 窗 baseline-only 批采样（32 窗/批 × 30 批，每批一次 `sample()`、B×K=256 展平），逐窗独立 eps/eta seed = 2026 + 1_000_000 + 窗内偏移（0..929，与批划分无关）；产出逐窗 s_{w,B} (930,62) 与 null 距离 (930,28,62)；**子样本 vs 全集对比在同一扫描内取 32 窗 vs 全体**（同噪声 → 隔离窗选择效应），另与主链 null 池 τ 对照（不同 eps → 噪声稳健性）；批量实现与逐窗实现首窗逐元素同构断言。
- τ 预览数据源：per-edge 32 窗 median D̄（dbar_median.npy (62,9,62)，含 r=0 零列，索引 [A, r, B]）；τ 来自主链 null 池（32×28×62 = 55,552 样本）。
- 环境：同 Run 001（本地 AutoDL，base conda Python 3.12.3 / torch 2.8.0+cu128 / RTX 4080 SUPER 32 GB，BLAS 8 线程护栏）。
- 命令：`python3 models/perturb-ec-v1/run.py --stage bench --subjects 1 --run-id bench-01`（前台分钟级）；产物 `outputs/perturb-ec-f0/bench-01/`：d_curves.npy (32,62,8,62)、null_pool.npy (32,28,62)、dbar_median.npy (62,9,62)、scan_s.npy (930,62)、scan_null.npy (930,28,62)、timing.json、config_snapshot.json、bench_metrics.json、logs/run.log、figs/{dose_response_bench,qq_representativeness,tau_preview}.png。

### 结果

- 代码版本：git HEAD **5e66c88**（B0 提交）+ B1/B2 未提交增量（`run.py` smoke+bench、`response.py` 批量 baseline 扫描、`loader.py` 受试者窗区段助手、`common/visualizer.py` dose_response/qq_compare/tau_preview 三图类型——后两型纯增量新增）；运行 **179.9 s**（装载+主链+扫描+统计+图），前台单卡。
- 执行事实：subject 10 子采样 32 窗（win_idx 0..31，全局 8417…9238）；**确定性复现断言通过：bench 窗 0–1 ≡ smoke-01 逐位一致**（同 seed 同批形，跨 Run 复现纪律兑现）。
- **计时**（每窗一次 sample()、497 并批、CUDA synchronize 包夹）：sample 段 mean **5.15 s/窗**（std 0.04；min 5.13 / max 5.41，首窗 warmup 可忽略），D/null 后处理 <0.01 s/窗；主链 32 窗 165.2 s；**GPU 峰值 allocated 2.99 GiB / reserved 3.53 GiB**（卡总 31.5 GiB，占用 ~11%）。
- **外推主 Run（10 例 × 32 窗 = 320 窗）≈ 28 min（0.46 h；+10% 余量 ≈ 30 min）**——4 h 阈值的 ~12%。代表性扫描（930 窗 baseline-only、30 批 × 32 窗、每批一次 sample()）仅 6.7 s（7 ms/窗），bench 专用不进主 Run 成本。
- **代表性抽查（baseline 侧，design §7-7；同 930 窗扫描内 32 窗 vs 全集，同逐窗 eps → 隔离窗选择效应）**：
  - s_{w,B}（n=1,984 vs 57,660）：KS D=0.0302（**p=0.059，不拒绝同分布**）；分位 5–95% 最大相对偏差 **0.93%**（median 1.0308 vs 1.0262，+0.45%）。
  - null d(i,j)（n=55,552 vs 1,614,480）：KS D=0.0042（p=0.30）；分位 5–95% 最大相对偏差 **0.32%**（median 1.4949 vs 1.4938）。
  - τ 三口径对照：q95 主链池 1.8236 / 扫描子样本（同 32 窗、不同 eps）1.8247（噪声效应 0.1%）/ 扫描全集（930 窗）1.8305（窗选择效应 0.3%）；q90/q99 同量级（噪声 ≤0.2%、窗选择 ≤0.5%）。
- **τ 决策数据预览（只算不判——预览、非 readout 承诺、τ 口径待用户拍板）**：
  - (a) 预注册 τ 删失率（跨通道 3782 边，per-edge 32 窗 median D̄）：q90/q95/q99 全部 **100.0% 右删失**——D̄(r_max) 跨通道 median 0.2306 / max 1.1395 << τ q90 1.7485（Run 001 smoke 全删失在 N_w=32/median 口径下复现并坐实）。
  - (b) 相对自响应口径 D̃_{A→B}(r) = D̄_{A→B}(r)/D̄_{B→B}(r)（同 r、逐边）：

    | r | 0.08 | 0.32 | 1.28 |
    |---|---|---|---|
    | median | 0.793 | 0.792 | 0.781 |
    | Q1 | 0.486 | 0.480 | 0.496 |
    | Q3 | 1.304 | 1.304 | 1.237 |
    | max | 6.593 | 6.710 | 5.276 |

    若 τ̃=0.25 → r\* 删失率 **4.5%**；τ̃=0.50 → **21.3%**。跨通道中位响应 ≈ 靶通道自响应的 0.78–0.79×，且对 r 近似常数（与 Run 001 低剂量段自/跨 slope 几乎相等的观察一致——空间弥散）。
  - (c) slope（低剂量段 {0,0.01,0.02,0.04,0.08}，跨通道 3782 边）：median **0.1957**（Q1 0.1348 / Q3 0.2788 / max 0.7929 / min 0.0357）——不依赖 τ，主 readout 设计上受保护。
- 产物：`outputs/perturb-ec-f0/bench-01/`：d_curves.npy (32,62,8,62)、null_pool.npy (32,28,62)、dbar_median.npy (62,9,62，[A, r, B] 含 r=0 零列)、scan_s.npy (930,62)、scan_null.npy (930,28,62)、timing.json（逐窗计时+外推+显存）、config_snapshot.json、bench_metrics.json（机器可读全量，含 DISCLAIMER）、logs/run.log。
- 图表：figs/dose_response_bench.png（4 代表曲线 + 主链 τ q95 线，y 全范围直见地板分离缺口）、figs/qq_representativeness.png（s 与 null_d 双面板 QQ + y=x + KS 注记）、figs/tau_preview.png（面板 a 地板分离 / b D̃ 箱线 + τ̃ 线 + 删失率 / c slope 分布；注记 PREVIEW ONLY）。

### 结论

1. **档位定版：维持 N_w=32 / K=8 / R_GRID 9 点不动**（假设支持）——外推 0.46 h << 4 h、GPU 峰值 reserved 3.53/31.5 GiB（~11%），无降档必要；**档位调整权（仅一次）原封保留给 τ/网格决策**（Run 001 结论 3 预留用途）。B3 主 Run 可直接进（nohup ~30 min 量级）。
2. **N_w=32 子采样代表性成立**（baseline 侧证据，假设支持）：KS 均不拒绝同分布（p=0.059/0.30），中央分位最大相对偏差 ≤0.93%，τ 三口径互差 ≤0.5%——32 窗子采样对 null 池/τ/生成波动尺度统计相对 930 窗全集几乎无损失。抽查限 baseline 侧可得量（扰动侧全集 ≈ 497× 全窗外推 ~2.3 h/例，未做——design §3 原文口径）。
3. **τ 口径决策数据齐备，判定权在用户**（本 Run 只算不判）：预注册 τ（独立样本对池）下 r\* readout 在正式口径（N_w=32、median 聚合）**全右删失**，结构性原因同 Run 001（CRN 消蒙特卡洛抖动 ⇒ 确定性响应天然低于 ~√2·s 生成地板，「D < τ」≠「无响应」）；相对口径预览显示 τ̃=0.25/0.50 可得 4.5%/21.3% 删失率、D̃ 跨通道中位 ~0.78 且近 r 无关。拍板选项：维持预注册 τ（r\* 全删失如实登记，slope/AUC 主 readout 不受影响）或启用相对口径/网格扩档（须走口径变更纪律重登记 + §2.1 外推警示随附）。
4. 下一步：B3 主 Run（main-01，Run 块登记后 nohup，10 例 × 32 窗）→ B4 readout + 诊断七件套；τ 拍板与档位决策同步 `.trellis/research/progress.md` 由主会话统一。

## Run 003 — main-01：10 例扰动批采样主 Run（B3，2026-10-17）

### 目的与假设

- implement.md B3：fold0 test **10 例全量** × N_w=32 扰动批采样（每窗一次 `sample()`、497 并批 CRN）→ 逐例窗级 D 曲线 + null 池 + per-subject D̄ median 曲线 + τ 三分位（逐例分池）落盘，为 B4 readout / B5 比较器供数——**本 Run 不做 readout/比较**（B4/B5 事），不改四要素口径常量。
- 断点续跑支持（design §8 回滚点③）：逐例落盘 + `done_s{sid}.json` 标记 + 例内 `partial_s{sid}.npz` 逐窗落盘；重跑跳过已完成例、例内自断点窗续（win 数据与 eps/eta 仅依赖 win_idx，续跑结果不变）。
- 假设：(a) bench 外推兑现——实际总时长 ~30 min 量级（5.15 s/窗 × 320 窗 + 装载）；(b) **subject 10（例序首位）32 窗与 bench-01 逐位一致**（同子采样种子、同 eps/eta seed 公式 win_idx 0..31、同批形 497 并批，跑后断言）；(c) 10 例 D/null 全有限非负、曲线形态与 Run 001/002 定性一致。

### 配置

- 代码：`models/perturb-ec-v1/`（B0 提交 5e66c88 + B1/B2 未提交增量 + B3 增量 `run.py --stage main`）；git HEAD 回填见结果节。
- **四要素口径 = Run 001/002 全部不变**（P=A 案增益调制、D=CRN 配对归一 RMS（每窗一次 sample()、baseline+62 源×8 档=497 并批、chunk_bk=1024、逐窗一组 eps/eta 跨全变体共享）、τ=null 逐被试分池分位（本 Run 只落池与分位，不进判定）、R_GRID 9 点 / K=8 / N_w=32 / 方向约定行=源列=靶；**档位维持 Run 002 定版，不动**）。
- **例序与 win_idx（Run 001 勘误后登记口径逐字沿用）**：例序 = EXPECTED_TEST_IDS 升序 [10, 11, 12, 17, 24, 28, 30, 44, 61, 69]（首例 subject 10，非 test_ds 自然序）；win_idx = 主 Run 内**全局窗序号跨例连续**（例 i 窗 j → win_idx = 32·i + j，i∈0..9、j∈0..31，共 0..319）→ subject 10 的 win_idx 0..31 与 bench-01 完全相同 → eps/eta 逐位同（seed=2026+win_idx）。窗子采样 = 逐例 `default_rng(2026×1000003+subject_id)` 无放回升序（loader.subsample_windows_per_subject，与 smoke/bench 同实现同种子 → subject 10 窗集 = bench-01 窗集）。
- **τ 口径注记**：τ 口径决策（绝对 vs 相对自响应）由主会话与用户进行中——本 Run **只落盘 D 曲线与 null 池（τ 无关数据）**；`tau_s{sid}.json` 三分位（null 逐例分池 q90/q95/q99）照 bench 同式落盘，标「预注册绝对口径，供登记」，不构成 readout 承诺。
- 断点续跑机制：例完成 = 四文件（d_curves_s{sid} / null_s{sid} / dbar_median_s{sid} / tau_s{sid}）+ done 标记齐；例内每窗追加 partial_s{sid}.npz（n_done 计数 + 逐窗数组），完成后删除 partial。
- 环境：同 Run 001/002（本地 AutoDL base conda Python 3.12.3 / torch 2.8.0+cu128 / RTX 4080 SUPER 32 GB，BLAS 8 线程护栏）；env 快照 `env_snapshot.txt`（版本信息 + pip freeze）随产物落盘（复现三要素补齐）。
- 命令：`mkdir -p outputs/perturb-ec-f0/main-01/logs && nohup python3 models/perturb-ec-v1/run.py --stage main --run-id main-01 > outputs/perturb-ec-f0/main-01/logs/nohup.log 2>&1 &`（与 implement.md 速查命令唯一差异：stdout 镜像落 **nohup.log** 而非 run.log——run.log 由进程内 logger 追加写，若 nohup 再重定向同文件会逐行双写；nohup.log 兼收 stdout 镜像 + 崩溃 traceback）。
- 产物布局 `outputs/perturb-ec-f0/main-01/`：`d_curves_s{sid}.npy`（32,62,8,62 [w,A,r 档 j∈1..8,B]）×10、`null_s{sid}.npy`（32,28,62）×10、`dbar_median_s{sid}.npy`（62,9,62 [A,r∈9 点,B] 含 r=0 零列）×10、`tau_s{sid}.json` ×10、`config_snapshot.json`、`env_snapshot.txt`、`main_metrics.json`（逐例耗时表/断言/显存，机器可读）、`logs/{run.log,nohup.log}`（逐窗进度 + ETA + 逐例耗时）。完成后进程内汇总断言：10 例全在、shape/finite 齐全、subject 10 ≡ bench-01 d_curves/null 逐位。

### 结果

- 代码版本：git HEAD **5e66c88**（B0 提交）+ B1/B2/B3 未提交增量（`run.py` smoke+bench+main、`response.py` 批量扫描、`loader.py` 区段助手、`common/visualizer.py` 三图类型）；运行 **1659.5 s（27.7 min）** nohup 单卡（setup 5.4 s + 320 窗实采，零跳过、零断点启用——首轮完整跑完）。
- **逐例耗时表**（sample 段 mean 全例 5.14–5.15 s/窗；全 320 窗 sample mean 5.146 s、std 0.014、min 5.13 / max 5.35）：

  | subject | 10 | 11 | 12 | 17 | 24 | 28 | 30 | 44 | 61 | 69 |
  |---|---|---|---|---|---|---|---|---|---|---|
  | 例序（win_idx 区段） | 0–31 | 32–63 | 64–95 | 96–127 | 128–159 | 160–191 | 192–223 | 224–255 | 256–287 | 288–319 |
  | 例耗时 s | 165.5 | 165.1 | 165.2 | 165.2 | 165.2 | 165.2 | 165.2 | 165.3 | 165.2 | 165.2 |
  | τ q95（null 55,552 样本/例） | 1.8236 | 1.8338 | 1.8448 | 1.8305 | 1.8247 | 1.8309 | 1.8384 | 1.8313 | 1.8303 | 1.8411 |

  （τ q90 跨例 1.7485–1.7641、q99 1.9627–1.9957——逐例分池分位跨例高度一致，窄幅波动。）
- **汇总断言全通过**：10 例 × 4 文件齐全；shape (32,62,8,62)/(32,28,62)/(62,9,62) 全例正确；D/null/dbar 全有限非负；τ json 逐例 subject_id 对齐；**确定性复现断言通过：subject 10 d_curves + null ≡ bench-01 逐位（`np.array_equal` 精确相等 = True，强于 allclose(1e-6) 门）**——同子采样、同 eps/eta seed（win_idx 0..31）、同批形跨 Run 复现纪律兑现。
- **GPU 峰值**：allocated **2.99 GiB** / reserved **3.53 GiB** / 卡总 31.5 GiB（~11%，与 bench 实测一致）；外推假设兑现（27.7 min ≈ bench 外推 28 min，4 h 阈值 12%）。
- **逐例定性一致性**（dbar_median 侧快速核，全 10 例）：跨通道 8 档完全非降比例 **100.0%**（与 Run 001/002 完美单调一致）；自/跨 median 比 1.20–1.44×（空间弥散跨例复现）；D̄(r_max) 跨通道 median 0.19–0.27 / max 0.67–1.32，全部 << 该例 τ q95（~1.82–1.84）——Run 002 预览的「地板分离缺口」在 10 例全部成立（预注册 τ 下 r\* 全删失的跨例形态坐实，正式判定留 B4/τ 拍板）。
- 产物：`outputs/perturb-ec-f0/main-01/`：d_curves_s{10,11,12,17,24,28,30,44,61,69}.npy ×10、null_s*.npy ×10、dbar_median_s*.npy ×10、tau_s*.json ×10（预注册绝对口径标注）、done_s*.json ×10（断点标记）、config_snapshot.json（全常量 + 逐例窗索引 + win_idx 公式）、env_snapshot.txt（版本 + pip freeze，复现三要素补齐）、main_metrics.json（机器可读全量含 320 窗逐窗计时）、logs/{run.log,nohup.log}（逐窗进度 + ETA + 逐例耗时）。
- 偏差登记（1 项，已在配置节声明）：nohup stdout 镜像落 **logs/nohup.log** 而非 implement.md 速查命令的 `> logs/run.log`——run.log 由进程内 logger 追加写，避免双写重复行；nohup.log 兼收崩溃 traceback。无其他偏差。

### 结论

1. **假设 (a)(b)(c) 全部支持，B3 放行**：(a) 总时长 27.7 min 兑现 bench 外推（28 min）——档位维持决策数据再确认；(b) subject 10 ≡ bench-01 **逐位精确相等**（array_equal），例序勘误口径（EXPECTED_TEST_IDS 升序 + win_idx 跨例连续）实现正确；(c) 10 例 shape/finite 全过、形态定性跨例一致（100% 完美单调、弥散传导、地板分离缺口全例成立）。
2. **B4 readout 供数就绪（τ 口径 v2 已拍板）**：逐例窗级 D 曲线（τ 无关数据）+ null 池全量落盘；**跑批进行中主会话完成 τ 口径决策 v2（用户 2026-10-17 拍板：操作性 τ = 相对自响应 τ̃=0.25、敏感性 0.50；绝对口径降为材质性观察照登——design §2.3 v2 / prd R2·AC4 同步更新）**。main-01 数据对两口径全覆盖：D̃(r) = D̄_{A→B}(r)/D̄_{B→B}(r) 由 dbar_median_s{sid}.npy 直接可算（B4 事），null 池三分位（tau_s*.json，预注册绝对口径标注）即绝对口径照登件——本 Run 未做 readout 判定与 v2 无冲突。本 Run 跨例观察（绝对 τ 下全删失形态在 10 例全体成立）与 v2 变更依据（bench 实测 100% 右删失）一致。
3. 断点续跑机制（done 标记 + 例内 partial）实现并就位但首轮未启用（零中断）；后续复算/扩展 Run 可直接依赖。
4. 下一步：B4 readout（三 readout → 受试者级 EC 矩阵 + load_ec_pert 契约 npz）+ 诊断七件套（design §6）；B5 参考侧（refs/split-half）+ 比较器。τ 拍板与 progress.md 同步由主会话统一。
