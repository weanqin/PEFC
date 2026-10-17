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

## Run 004 — readout-01：B4 三 readout + 有效路径集 + 诊断七件套（τ 口径 v2 首个正式 Run）（2026-10-17）

### 目的与假设

- implement.md B4（design §2.3 v2 / §2.4 / §6）：从 main-01 产物（只读、**不重采样、不加载 θ\* 网络**）计算 ① 三 readout（slope 主 / AUC 次 / r\* 右删失）→ 受试者级 EC 矩阵 + load_ec_pert 契约 npz ×3 readout ×10 例；② 有效路径集（τ̃=0.25 主 + τ̃=0.50 敏感性 + slope top-100 全 10 例池/逐例）；③ 诊断七件套（design §6 图 ①–⑦）；④ 汇总指标（删失率两档、r\* 分布形态、slope/AUC 分布、绝对口径照登）。
- 假设（bench Run 002 预览外推 + design §2.3 已知性质登记）：(a) τ̃=0.25 删失率 ~5% 量级、τ̃=0.50 ~20% 量级（跨 10 例）；(b) D̃ 近 r 无关 ⇒ r\* 近二元（首档即达或删失）——「临界强度」剂量语义弱承载，如实登记不强行解读；(c) 绝对口径（D̄ ≥ null q95）全删失（Run 003 跨例形态坐实的正式登记件）；(d) 全边完美单调（Spearman=1.0）跨例复现。

### 配置

- 代码：`models/perturb-ec-v1/run.py` 新增 `--stage readout / --stage figs` 两 stage（B4；refs/split-half/compare 归 B5 另行处理）+ `readout.py` D̃ 助手 + `config.py` τ̃ 常量；`common/visualizer.py` **纯增量** 5 图类型（mono_r2 / floor_separation / spatial_decay / diag_vs_cross / ec_heatmap_grid；既有 dose_response 复用于 ①）。git HEAD 回填见结果节。
- **输入面（main-01 只读）**：`dbar_median_s{sid}.npy` (62,9,62) 索引 **[A, r 档（含 0 零列）, B]**（逐例 32 窗 median D̄）、`null_s{sid}.npy` (32,28,62)、`tau_s{sid}.json`（绝对口径三分位，逐例分池）。
- **τ v2 操作口径（design §2.3 v2，用户 2026-10-17 拍板；本 Run 首个正式执行）**：D̃_{A→B}(r) = D̄_{A→B}(r)/D̄_{B→B}(r)（同 r、逐边、r=0 槽置 0 与 Run 002 预览实现一致；分母 = 靶通道自响应，存在靶自响应偏小的比值放大——判读纪律随附，不将 D̃ 数值作跨靶比较）；**τ̃ 主 = 0.25、敏感性 = 0.50**（config.py 常量）；r\* = min{r∈R_GRID, r>0 : D̃ ≥ τ̃}，网格内未达 → 右删失编码 R_NEXT = 2.56（由 R_GRID 推导禁字面量）。
- **绝对口径照登（材质性观察，不进主判定）**：r\*_abs = r\*(D̄ ≥ 该例 null q95) + q90/q99 敏感性 + 删失率（预期 100%）；**逐通道 τ_B**（null 按 B 通道分池 q95，逐例）落盘作敏感性。
- **readout 三件套（design §2.4）**：slope = 低剂量段 R_GRID[:5]={0,0.01,0.02,0.04,0.08} OLS（敏感性段 [:4]）；AUC = 全网格梯形/R_MAX；rstar 进比较器前翻转 strength = R_NEXT − r\*（删失→0 最弱；orientation='larger_is_stronger'）。
- **契约 npz**：`ec_pert_{slope|auc|rstar}_s{sid}.npz` ×30（字段 ec (62,62) 行=源列=靶 / readout / subject_ids / orientation；对角保留自响应实值，比较器 offdiag 展平不用对角）+ `readout_products.npz`（逐例三 readout 矩阵 + r\* 三口径 + 组均矩阵 + subject_ids，figs/比较阶段复用）。
- **有效路径集**：τ̃=0.25 主 = {(A,B): A≠B, r\* ≤ R_MAX}（非删失边，readout.effective_paths）逐例清单 + union/intersection/边频次统计；τ̃=0.50 同式敏感性；slope top-100 = 全 10 例池（逐边跨例 median 排序；附跨例 min–max、τ̃0.25 非删失例数、通道对欧氏距离 mm）+ 逐例（附 auc/r\*）。
- **诊断七件套（design §6；全部 300 dpi PNG 落 readout-01/figs/，图注 ASCII——DejaVu 无 CJK）**：① 剂量-响应样例曲线 ×3 例（**抽样 s10/s28/s69**，各 20 组 A→B：自响应 2 + 跨通道强 6/中 6/弱 6，按 D̄(r_max) 分位选取）；② 单调性分布（跨例跨边 Spearman(D̄, r)，9 点含 r=0）+ 低剂量段 OLS R² 分布；③ 噪声地板分离（null 池 vs 各 r 档跨通道 D̄ 分布 + 绝对 τ 三线 + 删失率柱：绝对 q90/95/99 与相对 τ̃0.25/0.50 两族）；④ 空间衰减（D̄(r_max) vs 通道对欧氏距离散点 + 分箱 median，chan_coords_62.npy @ cfg.data_root，定性 sanity 不作判据）；⑤ 对角 vs 跨通道量级 + **先验自响应分量解析量化**（下述）；⑥ EC 矩阵热图 3 readout ×（组均 + s10 + s69）；⑦ top-100 有效路径清单表（markdown + csv）。
- **⑤先验自响应分量（解析、零网络前向——design §2.1「自响应的先验分量量化」）**：x0 级解析重建 = bandlimit(μ + L_s·ε·L_tᵀ) + σ_min·η，μ = posterior_mean(x_past)、**ε/η 用主 Run 同 CRN 种子**（seed=2026+win_idx，win_idx=32·si+wseq，同 device cuda 生成——精确重建）；D_prior(A,r;w) = r·RMS_t(μ_A)/s_x0(w,A)，s_x0 = RMS_{k,t}(x0 − mean_k)（与 response.baseline_scale 同估计式）；先验分量占比 share = D̄_prior_A(r)/D̄_{A→A}(r)（逐通道后取分布）。**尺度注记（登记偏差）**：观测 D 的归一尺度为终样本 s_final（主 Run 未逐窗落盘、不重采样不可得），s_x0 为 x0 级尺度，share 读数随附 scale 校正近似 s_final ≈ median(null)/√2（√2 近似，独立高斯对差）。附 x0 级恒等校验运行时复验：posterior_mean 扰动差 B≠A 通道严格为零（先验逐通道事实，design §2.1）。
- 环境：同 Run 001–003（本地 AutoDL base conda Python 3.12.3 / torch 2.8.0+cu128；readout stage 纯 numpy CPU，figs stage 仅先验重建用 cuda、零网络前向）；命令：`python3 models/perturb-ec-v1/run.py --stage readout --run-id readout-01` → `python3 models/perturb-ec-v1/run.py --stage figs --run-id readout-01`。

### 结果

- 代码版本：git HEAD **2950630**（B3 提交）+ B4 未提交增量（`run.py` readout+figs 两 stage、`readout.py` dtil_curves、`config.py` τ̃ 常量、`common/visualizer.py` 纯增量 5 图类型 mono_r2/floor_separation/spatial_decay/diag_vs_cross/ec_heatmap_grid）；readout stage **0.3 s**（纯 CPU）+ figs stage **10.0 s**（数据装载 + 先验 x0 级解析重建 320 窗，零网络前向、零重采样）。
- **删失率（跨通道 3782 边/例，r\* = R_NEXT 编码）**：

  | 口径 | τ̃0.25（主） | τ̃0.50（敏感性） | 绝对 q90 | 绝对 q95 | 绝对 q99 |
  |---|---|---|---|---|---|
  | 跨例池（37820 边·例） | **3.69%** | **21.35%** | 100.0% | 100.0% | 100.0% |
  | 逐例范围 | 2.1–5.3% | 19.1–25.8% | 100% | 100% | 100% |

  假设 (a) 支持且与 bench 预览定量一致（s10：4.5%/21.3% 逐位复现）；绝对口径全删失坐实（材质性观察照登：D̄(r_max) 跨通道 median 0.2206 << null 跨例池 q90 1.7485/q95 1.8333——CRN 确定性响应天然低于独立样本对地板，结构性注记同 Run 001/002）。逐通道 τ_B（null 按 B 分池 q95）落盘：跨例 median 1.798–1.863（窄幅）。
- **r\* 分布形态（τ̃0.25，37820 边·例）**：首档 0.01 = **94.4%**、中间档（0.02–1.28）= 1.9%、删失 2.56 = 3.7%——**近二元形态坐实**（D̃ 近 r 无关 ⇒ 首档即达或删失），「临界强度」剂量语义在本 θ\* 上弱承载，如实登记不强行解读（假设 (b) 支持）。τ̃0.50 首档 73.7%/删失 21.3% 同形态。
- **D̃ 分布（跨例池）**：r=0.08/0.32/1.28 median = **0.770/0.767/0.762**（Q1 0.48–0.49、Q3 1.17–1.22、max 7.9–11.1）——近 r 无关复现 bench 预览；max 比值放大（靶自响应偏小通道）按判读纪律随附，不将 D̃ 数值作跨靶比较。
- **slope / AUC 分布**：slope 主段跨通道池 median **0.1755**（Q1 0.1232 / Q3 0.2526 / min 0.0284 / max 1.2161）、自响应 median 0.2228（自/跨中位比 1.27×）；敏感性段 median 0.1755（与主段一致——近线性所致）；AUC 跨通道 median **0.1117**（Q1 0.0791 / Q3 0.1573 / max 0.7166）、自响应 median 0.1437。
- **有效路径集**：τ̃0.25 逐例 **3583–3703** 边（94.7–97.9%）、union **3782**（全边至少 1 例有效）、intersection **3201**（84.6%，3201 边 10/10 例全有效；边频次直方图 1–9 例共 581 边）；τ̃0.50 逐例 2808–3058、union 3677、intersection 1830。slope top-100：pooled 首位 **A8→B56 / A8→B60**（median 0.5140、10/10 例非删失、通道距 30 mm），其后续 A41→B0（0.490，188 mm）等；逐例 csv ×10 + pooled csv + markdown 表落盘。
- **诊断七件套（figs/，8 PNG 300 dpi）**：①剂量-响应样例 ×3 例（s10/s28/s69 各 20 组，曲线 id 落 figs_metrics.json）；②**Spearman(D̄, r) median = 1.0000、完全非降 100.0%（37820 边·例）、低剂量 OLS R² median = 1.0000**——假设 (d) 支持，完美单调跨例全量复现；③地板分离（null 池 vs 各 r 档 D 箱线 log-y + 绝对 τ 三线 + 删失率柱两族）；④**空间衰减：Spearman ρ(D̄(r_max), 距离) = −0.050（平，无衰减结构）**——定性 sanity，弥散传导的又一佐证（design §7 Risk#1 语义），不作判据；⑤对角 vs 跨通道 + 先验分量（下）；⑥EC 热图 3 readout ×（组均+s10+s69）；⑦top-100 清单表 markdown。
- **⑤先验自响应分量（解析量化）**：x0 级恒等复验**通过**（μ 扰动差 = r·μ_A 且 B≠A 严格为零，win_idx=0、r=1.28）；share = D̄_prior/D̄_self median **0.425–0.432，r∈[0.01,1.28] 全档近常数**（Q1 0.29–0.31 / Q3 0.60–0.61）；尺度对照：s_x0 中位 **0.926** vs 隐含 s_final ≈ median(null)/√2 = **1.056**（校正因子 ×0.877）⇒ **scale 校正后先验分量占比 ≈ 0.37–0.38**——即自响应约四成来自先验均值投影（逐通道时间投影的确定性平移）、约六成来自速度网络通路放大；跨通道响应按先验逐通道事实结构性无先验贡献。
- 产物（`outputs/perturb-ec-f0/readout-01/`）：契约 npz **ec_pert_{slope,auc,rstar}_s{sid}.npz ×30**（ec (62,62) 行=源列=靶 / readout / subject_ids / orientation=larger_is_stronger，rstar 已翻转 strength=R_NEXT−r\*，删失→0、对角保留实值 2.55）、`readout_products.npz`（逐例 slope/slope_sens/auc + r\* 五口径 + strength + 三组均矩阵 + subject_ids）、`effective_paths_ttil025.json` / `ttil050.json`（逐例清单 + union/intersection/边频次）、`slope_top100_pooled.csv` + `slope_top100_s{sid}.csv ×10`、`top100_paths_table.md`、`readout_metrics.json`（机器可读全量）、`prior_component.json`、`figs_metrics.json`、`config_snapshot.json`、`logs/run.log`。
- 偏差登记（2 项，均已在配置节预告）：(a) 先验分量 share 的归一尺度用 s_x0（x0 级）替代不可得的 s_final（终样本级，主 Run 未逐窗落盘、不重采样约束下无法重建）——读数随附 √2 近似 scale 校正（0.926/1.056=0.877），校正前后占比区间 0.37–0.43 如实登记；(b) figs stage 数据装载经 test_ds 逐窗取 x_past（只读数据、不触 θ\* 网络——readout 约束的「不重采样」指 497 变体 CRN 采样未重跑，先验解析重建属 design §2.1「零采样成本」既定路线）。无其他偏差。
- 复验：B0 单测套件跑后复跑**全绿**（①CRN ②口径 ③readout 数学 + smoke 段，无回归）；subject 10 删失率/slope 与 bench-01 预览逐位一致（4.5%/21.3%/0.1957）。

### 结论

1. **AC4 判定：流程可跑通 = 支持；响应真实性 = 成立（CRN 确定性响应，非噪声）**。(a) 全链 readout 机械正确：10 例 × 三 readout + 契约 npz + 路径集 + 七件套一次跑通（readout 0.3 s + figs 10 s，零重采样零网络前向）；(b) 响应真实性证据：37820 边·例 **100.0% 完美单调**、低剂量 R² = 1.0000（r=0 恒等 + 剂量依赖连续单调 = 确定性信号特征，蒙特卡洛抖动已被 CRN 契约消除）；(c) 噪声地板与 τ 标定完成：绝对口径三分位（q90 1.749–1.764 / q95 1.824–1.845 / q99 1.963–1.996 逐例）与 **100% 删失率照登**（材质性观察：任何可行剂量下响应均未达生成条件分布自身波动量级——结构性，非缺陷判读）+ v2 相对口径 τ̃0.25/0.50 删失率 3.7%/21.4% 与路径集敏感性齐备。
2. **r\* readout 近二元形态（已知性质坐实）**：94.4% 边首档即达 τ̃0.25 ⇒ r\* 的「临界强度」语义弱承载——有效路径集（τ̃0.25 主口径）可用但判读应侧重 slope（主 readout，τ 无关）；D̃ 中位 0.76–0.77 且近 r 无关 + 空间衰减平（ρ=−0.050）+ 自/跨中位比仅 1.27× ⇒ **θ\* 扰动传导空间弥散、通道特异性弱**（design §7 Risk#1 三重独立佐证，与 Run 001–003 一致）。
3. **先验/网络通路分解（新增证据）**：自响应 ≈ 37–43% 先验均值投影 + 57–63% 速度网络通路；跨通道响应 100% 经速度网络单通路（rank-16 瓶颈）——为 H3 弱项的机理注记提供量化基线（后续变体对比可用同式复算）。
4. **B5 放行**：比较器输入面就绪（30 契约 npz + refs-01 参考矩阵）；slope 主 readout 不依赖 τ、受保护；空间弥散预期下结构收敛性（AC5/AC6）以实际比较为准，阴性 = 证据不足非证伪（power 注记随附，design §5）。
5. 下一步：B5 参考侧 + 比较器（refs-01 已另行登记 Run 005）；B6 质检收尾（trellis-check 路径扫描：common/visualizer.py 增量为 design 声明已知写入点）。

## Run 005 — refs-01：B5 前半参考侧——fmt 10 例 observational refs（GC/gPDC）+ split-half 天花板（2026-10-17）

### 目的与假设

- implement.md B5 前半（design §4）：fold0 test 10 例 fmt 同源数据上重算 observational directed-connectivity references（ridge 条件 GC + |gPDC|² 宽带）与 split-half 噪声天花板，为 compare 阶段（B5 后半，主会话另行派发）备好输入面。**本 Run 不做 compare、不做 readout、不做 mne-c**。
- 假设：(a) 10 例 GC/gPDC 矩阵 shape (62,62)、全有限；gc 对角按构造为 0、gpdc 对角（自连接）非零照存（比较器 offdiag_flat 展平自然剔除）；(b) fmt 侧 split-half 天花板方向与 10-13 任务 v4 数据口径一致（gc > gpdc，量级 ~0.9/0.73 附近），幅度受数据源差异（无 ICA/notch、时长 240 s vs ~170–340 s）偏移——**方向性对照只登记不判读**。

### 配置

- 代码：`models/perturb-ec-v1/refs_fmt.py`（**独立 CLI，不经 run.py**——B4 readout 代理并行开发中，CLI 冲突规避；design §1 模块布局本就列 refs_fmt.py 为独立模块）+ `models/perturb-ec-v1/tools/_refs_worker.py`（子进程 worker）。**baseline-ec-v1 与 run.py 零改写**、数据集只读、无 GPU 依赖（CPU-only）。
- 数据：`/root/autodl-tmp/edra_processed_fmt/{HC,MDD}/{sid}_rest_v4.npz`（(62,60001) float32 @250 Hz）；fold0 test 10 例 = EXPECTED_TEST_IDS 升序 [10,11,12,17,24,28,30,44,61,69]（组别经 manifest.csv 解析，全部 status=ok）；逐例字段断言 fs=250、filter='1-40(release)'、pipeline='EDRA_release_1-40_250Hz'、truncated=False。
- **重采样+z-score（design §4：与 baseline-ec-v1 resample_io 同构，仅数据源与降采比不同）**：float64 载入 → `resample_poly(up=1, down=2, axis=-1)`（抗混叠 FIR 内置）→ (62,30001) @125 Hz → 逐通道全记录 z-score（ddof=0，σ 下限 1e-12，与 resample_io 逐字同式）→ 落盘 float32 `data/x_s{sid}.npy`、计算时复原 float64（与 resample_io cache_subject→load_cached 精度往返一致）。
- **EC 参考（沿用 10-13 任务 Run 001 定标）**：`refs_for_subject(x, p=8)`（p/λ 不在父进程复写常量——由 baseline-ec-v1 ec_refs.P_SHARED=8/LAM_REL=1e-2 单源输出，实际值从 worker 产物读回登记）；GC=ridge 条件 ln(SSR 比)（行=源/列=靶/对角=0）、gPDC=|gPDC|² 宽带 1–40 Hz（[0,1]、列平方和=1、行=源存储）。mne-c 交叉核对**不做**（本任务口径）。
- **子进程隔离（design §1 纪律，硬性）**：逐例 fresh `python3 tools/_refs_worker.py <x.npy> <out.npz>`（cwd=baseline 目录、env PYTHONPATH 仅 baseline 目录）；worker 头部再防御性剔除 sys.path 中其他 /root/npi 仓库路径并断言 `config.__file` 归属 baseline-ec-v1、`gc_psd.__file` 归属 gpfm-mamba-v1；npz 文件面交换，**父进程零 import baseline 模块**（ec_refs/compare/split_half 模块级 `import config as C` 进程内必被本变体目录 config.py 遮蔽击穿——design §1 已登记）。worker 头部复制 BLAS 五变量 setdefault 块（numpy 前；环境继承下幂等）。
- **split-half 天花板（design §4 必做；照 baseline-ec-v1 split_half.py 口径原样）**：worker 内直接 `split_half.ceiling_for_subject(x, p)`——偶段→半 A / 奇段→半 B、段长 250 样本（2 s @125 Hz）、尾巴丢弃、各半以 (n_seg,62,250) 多段窗喂 Gram（段内设计不跨段）、逐例 Spearman(offdiag 3782 边)。fmt 侧 30001//250=120 段 → 每半 60 段（v4 侧 21250//250=85 段 → 42/43 段，段数差异随数据登记）。汇总两法 mean±sd（ddof=1，与 ceilings_all 同式）。
- 汇总产物与 **compare 输入面（B5 后半 compare 代理读本 Run 产物，不重算）**：
  - `outputs/perturb-ec-f0/refs-01/refs_all.npz`：键 `gc` (10,62,62) float32、`gpdc` (10,62,62) float32（gpdc = gpdc_bb 宽带）、`subject_ids` (10,) 字符串数组，序 = EXPECTED_TEST_IDS 升序，行=源/列=靶；compare 侧 `refs={"gc":…, "gpdc":…}` + sids 即 baseline-ec-v1 compare.run_comparison 的输入形态（参考 v4 侧 refs_all.npz 同构）。
  - `refs/refs_s{sid}.npz` ×10：gc/gpdc_bb/gpdc_bands (5,62,62)/lb_frac/n/p/lam/sh_rho_gc/sh_rho_gpdc/n_samples/split_seg/n_seg。
  - `refs_summary.json`：口径（p/λ/fs 250→125/重采样式/z-score 式/存储精度）+ split-half 两法 mean±sd 与逐例 + **数据源偏差登记**（fmt 1–40 Hz 带限·无 ICA·无 notch·48 例·240 s vs v4 bp1-40+notch·46 例·~170–340 s；n_125 30001 vs 21250；split-half 段数 120 vs 85）+ refs 基本统计 + worker 隔离形态。
  - 另 `data/resample_meta.json`（10 例重采样元数据）、`config_snapshot.json`、`logs/run.log`。
- 环境：同 Run 001–003（本地 AutoDL base conda Python 3.12.3；BLAS 8 线程护栏——refs_fmt.py 与 worker 头部各 setdefault）；CPU-only 前台，预期分钟级。
- 命令：`python3 models/perturb-ec-v1/refs_fmt.py --run-id refs-01`（前台）。
- 与 10-13 任务 v4 口径天花板的方向性对照数值（**只登记不判读**）：baseline-ec-v1 main-01（46 例 v4 侧）split-half gc 0.9003±0.0296、gpdc 0.7344±0.0387。

### 结果

- 代码版本：git HEAD **2950630** + 本 Run 未提交增量（`models/perturb-ec-v1/refs_fmt.py`、`models/perturb-ec-v1/tools/_refs_worker.py` 两新文件；baseline-ec-v1 / run.py 零改写——`git status` 核实干净）；运行 **63 s** 前台 CPU-only（逐例 worker wall 5.2–6.7 s = refs 0.9–1.3 s + split-half 1.0–1.6 s + 解释器启动；阶段 A 重采样 10 例 ~3 s）。
- 执行事实：10 例字段断言全过（fs=250、filter='1-40(release)'、pipeline='EDRA_release_1-40_250Hz'、truncated=False）；重采样 (62,60001)@250 → (62,30001)@125、z-score 后落 float32；worker 隔离断言全过（`config.__file` → baseline-ec-v1/config.py、`gc_psd.__file` → gpfm-mamba-v1/gc_psd.py，逐例子进程 stdout JSON 回传）；实际口径 p=**8**、λ_rel=**1e-2**（λ_abs≈299.8–300，z-score 单位；从 worker 产物读回登记，父进程零复写）。
- **GC 基本统计**（offdiag 3782 边 ×10 例 = 37,820 样本）：min **−1.61e−3** / median **1.86e−4** / max **2.68e−2**；**对角 = 0**（ridge_conditional_gc 按构造只填非对角，逐例断言过）；≤0 占 **18.3%**（ridge 收缩口径微负，排序语义不受影响——ec_refs 已登记性质）；≤1e−3 占 **84.2%**、≤1e−2 占 99.8%（ln SSR 比口径下绝对值普遍小、靠秩排序比较）。
- **gPDC 基本统计**（offdiag 同上）：min **4.1e−5** / median **5.9e−3** / max **2.89e−1**（[0,1] 域内，逐例断言过）；≤0 占 0%（恒正）；≤1e−3 占 **8.7%**、≤1e−2 占 64.6%（稀疏度低于 GC 的阈值表现——两法量纲不同，各自秩比较不受影响）；**对角（自连接）非零照存**，median **0.173**（≈自连接主导各源列归一质量；比较器 offdiag_flat 展平自然剔除，已登记）。
- lb_frac = **1.000 ×10**（全通道 Ljung-Box p<0.01）——ec_refs 已登记「LB 对真实数据超严，lb_frac≈1 属常态，不作选阶判据」，照登。
- **split-half 天花板（fmt 10 例，120 段 → 60 段/半）**：
  - **gc：0.9598 ± 0.0142**（逐例 0.9290–0.9844）
  - **gpdc：0.9636 ± 0.0203**（逐例 0.9302–0.9951）
- **方向性对照（只登记不判读）**：v4 口径（46 例、85 段）gc 0.9003±0.0296、gpdc 0.7344±0.0387 → fmt 口径（10 例、120 段）gc 0.9598±0.0142、gpdc 0.9636±0.0203。**两法均更高、gPDC 增幅更大（+0.23 vs gc +0.06）、且 fmt 侧两法天花板近似持平（v4 侧 gc 明显高于 gpdc）**。候选来源已在配置节偏差登记项内（每半段数 60 vs 42/43、无 ICA/无 notch、时长 240 s 全例一致 vs 170–340 s 案例异质、例数 10 vs 46）——不展开判读。
- 产物：`outputs/perturb-ec-f0/refs-01/`：`refs_all.npz`（gc (10,62,62) + gpdc (10,62,62) float32 + subject_ids，序 = EXPECTED_TEST_IDS 升序）、`refs/refs_s{sid}.npz` ×10（gc/gpdc_bb/gpdc_bands (5,62,62)/lb_frac/n/p/lam/sh_rho_gc/sh_rho_gpdc/n_samples/split_seg/n_seg）、`data/x_s{sid}.npy` ×10 + `data/resample_meta.json`、`refs_summary.json`（口径+天花板+偏差登记+worker 隔离形态+compare 输入面，机器可读）、`config_snapshot.json`、`logs/run.log`。
- 汇总完整性断言全过：10 例 shape (62,62) ×两法、全有限、gc 对角全零、gpdc∈[0,1]、subject_ids 序正确。
- 偏差登记（2 项，均在配置节预声明）：① 入口 = 独立 `refs_fmt.py` 而非 `run.py --stage refs`（B4 并行 CLI 冲突规避，任务派发口径）；② refs 与 split-half 同 Run 一次完成（implement.md 分列两 stage，合并为单脚本单遍——两者共享重采样产物与 worker 调用，成本 refs+2×半段 Gram，无口径影响）。

### 结论

1. **B5 前半放行**：fmt 同源参考侧（GC/gPDC）+ split-half 天花板在 10 例上完整就绪，机械正确性成立（隔离断言、shape/finite/对角/值域断言全过）；refs+split-half 全程 63 s，分钟级预期兑现。
2. **compare 输入面就绪（B5 后半前置条件达成）**：`refs_all.npz` 键/形状/顺序契约 = `refs={'gc': (10,62,62), 'gpdc': (10,62,62)}` + subject_ids（行=源/列=靶），与 baseline-ec-v1 compare.run_comparison 的输入形态及 v4 侧 refs_all.npz 同构——compare 代理（主会话另派）直接读 `outputs/perturb-ec-f0/refs-01/refs_all.npz` + `refs_summary.json`（天花板比值用 split_half 节），**不需重算参考侧**。
3. **天花板方向性登记（不判读）**：fmt 侧两法天花板均高于 v4 口径且近似持平（0.960/0.964）——compare 阶段计算天花板比值时用本 Run fmt 侧数值（同数据源才可比），v4 侧数值仅作背景对照；两数据源差异项已固化为 refs_summary.json 的 deviation_from_baseline_v4 字段。
4. 下一步：B5 后半 compare（EC_pert ×3 readout vs {gc, gpdc}，主比较 = slope × 2 法 Holm）由主会话在 B4 readout 产物就绪后另派代理执行；本阶段不触 compare。

## Run 006 — compare-01：B5 后半比较器——EC_pert ×3 readout vs observational refs 结构收敛性主比较 + AC5/AC6 判定（2026-10-17）

### 目的与假设

- implement.md B5 后半（design §5 / evaluation-calibers「Study II 扰动 EC 对比口径」）：readout-01 契约 npz（30 件，只读）× refs-01 两参考（gc/gpdc (10,62,62) + fmt 侧 split-half 天花板）→ **主比较（slope × {gc, gpdc}）** + 上下文（auc/rstar 同流程数值、未偏 ρ、天花板比值）→ comparison.json / comparison_matrices.npz / 汇总 forest 图；AC5（比较器完整运行）+ AC6（判定语句）在本 Run 落定。
- 假设：机械面（契约/对齐/隔离断言）全过；主 readout slope 的 ρ̄ 量级预期偏低——Run 004 已三重佐证 θ\* 扰动传导空间弥散（空间衰减平 ρ=−0.050、自/跨中位比 1.27×、D̃ 近 r 无关），结构收敛性属**如实判定**事项；阴性 = 证据不足非证伪（power 注记随附）。

### 配置

- 代码：`models/perturb-ec-v1/compare_wrap.py`（独立 CLI，不经 run.py——与 refs_fmt.py 同款任务派发口径）+ `models/perturb-ec-v1/tools/_compare_worker.py`（子进程 worker）；`common/visualizer.py` **纯增量** compare_forest 图类型（design §1 已声明共享文件写入点）。**baseline-ec-v1 / run.py / readout-01 / refs-01 产物零改写**。
- **主比较（预注册唯一）**：主 readout slope × 两参考法 → 逐被试**距离偏相关 Spearman ρ**（v̂ = rank 三向量偏相关；控制变量 = 通道对三维欧氏距离，chan_coords_62.npy（fmt/v4 两处逐位相同）；3782 有向边展平非三角）→ 单样本 t（n=10，df=9）× **Holm 跨 2 法（族 = {slope×gc, slope×gpdc}）** + 1000×受试者 bootstrap 95% CI（seed 2026，baseline config.BOOTSTRAP_N/BOOTSTRAP_SEED 单源）+ ρ̄/ceiling 比值（ceiling = refs-01 fmt 侧 split-half 两法 mean：gc 0.9598 / gpdc 0.9636——同数据源才可比，v4 侧数值仅背景）。
- **上下文（不进 Holm 族，产物标注 context）**：auc/rstar 同流程数值（t、raw p、bootstrap CI、ceiling 比值）；未偏 Spearman ρ（run_comparison rho_raw）。rstar 进比较器前已翻转（strength = R_NEXT − r\*，readout-01 契约 orientation=larger_is_stronger）。
- **复用与隔离（_refs_worker 同款形态）**：baseline compare 叶子函数（load_ec_pert / offdiag_flat / partial_spearman / holm / bootstrap_ci_mean / run_comparison）一律经 fresh python 子进程 import（cwd=baseline 目录、env PYTHONPATH 仅 baseline 目录、worker 头部防御性剔除其他 /root/npi 仓库路径 + `config`/`compare` `__file__` 归属断言）；父进程零 import baseline 模块（compare.py 模块级 `import config as C` + def 期求值 C.BOOTSTRAP_N，进程内必被遮蔽击穿）；manifest/npz/json 文件面交换。
- **输入面（只读）**：`readout-01/ec_pert_{slope|auc|rstar}_s{sid}.npz` ×30（逐被试配对——逐例 subject_ids 与 refs_all.npz subject_ids 逐位对齐断言后按该序堆叠 (10,62,62) 进 run_comparison 逐被试配对路径）+ `refs-01/refs_all.npz` + `refs-01/refs_summary.json`（天花板来源）。
- 产物：`outputs/perturb-ec-f0/compare-01/`：`comparison.json`（全量数值 + 口径 + 术语纪律 + power 注记 + worker 隔离形态）、`comparison_matrices.npz`（组均 EC_pert ×3 readout + 组均 refs ×2 + dist (62,62) + subject_ids）、`figs/compare_forest.png`（三 readout × 两法 ρ̄+CI forest，ASCII 注记）、`dist_matrix.npy`、`worker_manifest.json`、`config_snapshot.json`、`logs/run.log`。
- **术语纪律（prd R8）**：GC/gPDC 一律 observational directed-connectivity references；结论表述 = 结构收敛性验证，产物与登记零 ground-truth / 恢复真实因果措辞。
- **power 注记（design §5）**：df=9、Holm 校正后 α≈0.025、逐被试 ρ 典型 sd 0.25–0.35 下可检效应量 ρ̄≳0.55–0.65；阴性结果 = 证据不足，走 AC6 缺口诊断路径，不表述为「证伪收敛」。
- 命令：`python3 models/perturb-ec-v1/compare_wrap.py --run-id compare-01`（前台，CPU 分钟级）。

### 结果

- 代码版本：git HEAD **2950630** + 本 Run 未提交增量（`models/perturb-ec-v1/compare_wrap.py` + `models/perturb-ec-v1/tools/_compare_worker.py` 两新文件、`common/visualizer.py` 纯增量 compare_forest 第 14 图类型）；baseline-ec-v1 / run.py / readout-01 / refs-01 零改写（git status 核实 baseline 目录干净；config/readout/run/tests 的未提交增量为 Run 004/005 既有，非本 Run 触碰）。运行 **2 s** 前台 CPU-only（worker wall 1.4 s = 3 readout × 10 例 × 2 法 × 3782 边偏相关 + 6 组 bootstrap 1000×）。
- 隔离与契约断言全过：worker `config`/`compare` `__file__` 归属 baseline-ec-v1（stdout JSON 回传登记进 comparison.json）；30 契约 npz 逐例 subject_ids 与 refs_all 逐位对齐（10/10 × 3 readout）；run_comparison 粒度 per_subject（逐被试配对路径）；契约负值存档计数 0（slope/auc 无负值、rstar 翻转后非负）。
- 距离控制变量：chan_coords_62.npy（fmt 侧；与 v4 侧逐位相同）→ (62,62) 欧氏距离 **mm**（coords 单位米 ×1000，run.py `_dist_mm_matrix` 同式）；offdiag min/median/max = **27.2 / 120.9 / 190.0 mm**（与 Run 004 top-100 距离读数同量纲）。偏相关为秩变换、尺度不变——mm 仅为产物口径一致。
- **全量数值表（三 readout × 两法；ceiling = refs-01 fmt 侧 split-half mean：gc 0.9598 / gpdc 0.9636；sd 为逐被试 ρ 的 ddof=1；「—」= 不进 Holm 族）**：

  | readout × 法 | ρ̄（偏相关主指标） | sd | t（df=9） | raw p | Holm p | bootstrap CI95 | 未偏 ρ̄ | ρ̄/ceiling | 正号例 |
  |---|---|---|---|---|---|---|---|---|---|
  | **slope × gc（主）** | **−0.0068** | 0.037 | −0.573 | 0.581 | 0.581 | [−0.0296, +0.0145] | +0.0031 | −0.007 | 4/10 |
  | **slope × gpdc（主）** | **+0.0222** | 0.032 | +2.191 | 0.056 | **0.112** | [+0.0031, +0.0407] | +0.0335 | +0.023 | 8/10 |
  | auc × gc（上下文） | −0.0063 | 0.039 | −0.514 | 0.620 | — | [−0.0304, +0.0162] | +0.0040 | −0.007 | 5/10 |
  | auc × gpdc（上下文） | +0.0216 | 0.033 | +2.062 | 0.069 | — | [+0.0021, +0.0402] | +0.0333 | +0.022 | 8/10 |
  | rstar × gc（上下文） | −0.0033 | 0.031 | −0.338 | 0.743 | — | [−0.0182, +0.0178] | −0.0007 | −0.003 | 4/10 |
  | rstar × gpdc（上下文） | +0.0223 | 0.040 | +1.750 | 0.114 | — | [+0.0020, +0.0483] | +0.0249 | +0.023 | 7/10 |

- **AC6 判定语句（初步证据门槛 = ρ̄>0 且 Holm 校正 p<0.05，两法均须过）**：**未达门槛**——slope × gc：ρ̄=−0.0068（<0，Holm p=0.581）；slope × gpdc：ρ̄=+0.0222>0 但 Holm p=0.112 ≥ 0.05；`gate_pass_overall=False`（comparison.json 机器可读 gate 字段）。
- 一致性注记（只登记不判读）：gPDC 侧三 readout ρ̄ 均 ≈ +0.022、bootstrap CI 均不含 0、8/10（slope）例正号——方向微弱一致但量级 ρ̄/ceiling ≈ 2.3%（可忽略）；GC 侧三 readout ρ̄ ≈ 0（CI 全含 0）。百分位 bootstrap CI 与 t 检验在边缘不同步（slope×gpdc CI 不含 0 而 raw p=0.056）——判定以预注册 t×Holm 门为准，CI 照登。
- **检验力实测校准**：逐被试 ρ sd 实测 **0.031–0.040**（远小于 design §5 预登记假设 0.25–0.35——该假设源自 v4 侧 EC_pert 来源多样性）⇒ df=9、Holm 首检 α≈0.025（临界 t≈2.69）下实际可检 ρ̄ ≈ 2.69×0.033/√10 ≈ **0.028**：本 Run 对 ≳0.03 量级效应有检出力，观测 gPDC +0.022 未过门**不是检验力不足所致**（且该量级相对天花板已可忽略）。预登记 power 注记（ρ̄≳0.55–0.65）与实测校准两条一并随附。
- 产物：`outputs/perturb-ec-f0/compare-01/`：`comparison.json`（全量数值 + 口径 + 术语纪律 + power 注记 + gate/worker_isolation 字段）、`comparison_matrices.npz`（组均 EC_pert ×3 + 组均 refs ×2 + dist mm + subject_ids + 逐被试 ρ 6 组）、`figs/compare_forest.png`（6 行 forest，主族蓝/上下文灰、ρ=0 参考线、行尾 ρ̄/CI/Holm p/ceiling 比值注记，300 dpi ASCII）、`dist_matrix.npy`、`worker_manifest.json`、`worker_results.json`（worker 原始数值面）、`config_snapshot.json`、`logs/run.log`。
- 偏差登记（1 项，首跑即修重跑）：首跑两点实现层小缺陷——① 父进程 JSON 装配引用未定义键（n_subjects，KeyError）；② dist 矩阵漏乘 1000（coords 单位米）——偏相关秩不变、ρ 数值无影响，但产物口径与 Run 004 不一致。均修复后**清目录重跑**，登记数值为修复后单次运行（2 s）；无口径层偏差。

### 结论

1. **AC5 判定：达成**——比较器完整运行：主指标（逐被试距离偏相关 Spearman ρ）+ 未偏 ρ 上下文 + 单样本 t × Holm（族 = {slope×gc, slope×gpdc}）+ 1000×受试者 bootstrap CI（seed 2026，baseline config 单源）+ fmt 侧 split-half 天花板（Run 005）与 ρ̄/ceiling 比值全部落盘；隔离纪律兑现（父进程零 import baseline 模块、worker `__file__` 归属断言过、npz/json 文件面交换）。
2. **AC6 判定：未达初步证据门槛（阴性 = 证据不足，非证伪）**。主 readout slope 对两参考法：gc ρ̄=−0.0068（Holm p=0.581）、gpdc ρ̄=+0.0222（Holm p=0.112）——均未过「ρ̄>0 且 Holm p<0.05」。表述纪律：**「该 θ\* 上未获结构收敛性的初步证据」**，不表述为「证伪收敛」；限定 = 前置验证级单模型单折 + observational directed-connectivity references + 检验力注记（实测可检 ρ̄≈0.028，阴性非 power 不足所致）。
3. **缺口诊断（哪个环节弱）**：(a) **非 readout 环节**——三 readout ρ̄ 按法几乎相同（gc ≈ −0.005 / gpdc ≈ +0.022），slope 秩结构与参考的弱关联是 readout 不变量；r\* 近二元（Run 004：94.4% 首档即达）亦未进一步降低 gPDC 关联（rstar×gpdc +0.0223 ≈ slope×gpdc +0.0219）⇒ 换 readout 不改变结论；(b) **非参考侧噪声**——refs split-half 天花板 0.960/0.964（内部高度可靠）；(c) **主缺口 = θ\* 跨通道传导的通道特异性不足（D 弥散）**——Run 004 三重佐证（空间衰减平 ρ=−0.050、自/跨中位比 1.27×、D̃ 近 r 无关）+ 本 Run 第四重独立证据（EC_pert 边级秩结构与 observational refs 几乎零重合，gPDC 侧 +0.022 ≈ ceiling 的 2.3%）；机理登记（Run 004 先验/网络分解）：跨通道响应 100% 经速度网络单通路（rank-16 输入瓶颈，H3 已知弱项）。
4. **下一步选项（不在本任务内执行）**：(i) 框架级——在跨通道通路容量更高的后续代理脑变体上复跑同流程（四要素 + 比较器代码全复用，换 ckpt 即可，流程验证目的已达成）；(ii) 口径级——B 案加性带限波形算子（design §2.1 文档级预留；启用须走口径变更纪律重登记）；(iii) readout 级——非线性 readout（曲率/饱和点）优先级低（本 Run 证据：readout 替换不改变结论）；(iv) 检验力级——多折/多例扩展可缩 CI，但效应量相对天花板已可忽略（2.3%），非主要缺口。
5. **术语纪律自检过**：产物与登记中 GC/gPDC 一律 observational directed-connectivity references、结论 = 结构收敛性验证、零 ground-truth / 恢复真实因果表述（comparison.json terminology 字段为显式否定式声明「NOT ground-truth recovery」；图注 ASCII 同款）。
6. 下一步：B6 质检收尾（trellis-check 全量：路径扫描 common/visualizer.py 增量为 design 声明写入点、experiment.md 完整性；3.3 spec 更新候选——Study II 四要素首版流程规范沉淀 + 阴性结果判读纪律）。
