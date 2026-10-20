# 实验日志 — 10-19-latent-flow-edra-fmt-train（latent-flow-v1 × edra_fmt 两阶段训练）

> 每次「跑批/训练/评估/统计」追加一个 Run 块：跑前先写「目的+配置」，跑后同会话补「结果+结论」，不留空块。
> 格式契约见 `.trellis/spec/research/experiment-log.md`；产物只落本任务声明的变体目录（`.trellis/spec/research/path-isolation.md`）。

## Run 001 — edra_fmt 首个 latent-flow 两阶段正式训练（stage1 VAE → stage2 mexhat 潜空间 CFM）（2026-10-19）

### 目的与假设

- 在 edra_fmt（48 例 @250 Hz）上以 latent-flow-v1 完成 study1 候选路径的首个正式训练 run：stage1 单独训 VAE，stage2 冻结 VAE 训潜空间 CFM 流匹配（mexhat-graph 变体，D1 拍板）。
- 假设：数据 shim（npz Descriptor + manifest 只读复用 + 384/128/64 窗）与两阶段训练循环在满量（34/4/10 例 × 930 窗）上可稳定收敛（val loss 下降、早停或达 50 epoch 上限）；产出 θ 候选供后续代理脑合格性检验。
- 本 run **不**回答代理脑有效性问题（预测质量 ≠ Surrogate Brain Validity，两问题独立评价）。

### 配置

- 代码：`models/latent-flow-v1/`（git 基线 3aec527 + 本任务 S1–S3 改动：config/datasets/modma 三处 additive + `train_edra_fmt.py` + `build_edra_fmt_geometry.py`）
- 数据：`/root/autodl-tmp/edra_processed_fmt/`（48 例 npz，HC 24 / MDD 24）；划分只读复用 `models/gpfm-mamba-v1-fmt/splits/fold0_status_ok_seed42_val4.json`（fold0 内层 34/4/10，test=[10,11,12,17,24,28,30,44,61,69]）
- 时间窗：window 512（obs 384 + fc 128）/ stride 64 / 边界丢尾，每例 930 窗（入口断言）
- 归一化：训练折 34 例逐通道 mean/std（float64），val/test 复用（`norm_stats` 严格 held-out 路径）
- 模型：`OCLatentFlow(n_channels=62, latent_dim=64, t_lat=16, obs_len=384, forecast_len=128)`；stage2 流 = mexhat-graph（`graph_k=2, out_mix_alpha=0.5, mexhat_gamma=0.1, gamma_m=1.0`，geometry = npy 基座 `geometry_edra_fmt.pt`，D4 拍板），无 A_eff 正则
- 超参（两阶段同）：epochs 50 / patience 10 / lr 1e-3 恒定 / batch 32 / seed 42 / beta_kl 1e-4（stage1 KL 退火 20% epochs）/ Adam / AMP bf16 / max_grad_norm 1.0
- 环境：base @ 本机（Python 3.12.3 / torch 2.8.0+cu128 / RTX 4080 SUPER 32GB），对照 `.trellis/research/environment.md`
- 命令（可复现）：
  - `nohup python3 /root/npi/models/latent-flow-v1/train_edra_fmt.py --stage 1 > …/run_01/logs/stage1.log 2>&1 &`
  - `nohup python3 /root/npi/models/latent-flow-v1/train_edra_fmt.py --stage 2 --vae-ckpt …/run_01/checkpoints/fold0_edra_fmt_vae_best.pt > …/run_01/logs/stage2.log 2>&1 &`
- 产物根：`/root/npi/outputs/10-19-latent-flow-edra-fmt-train/run_01/`（configs / checkpoints / logs）

### 结果

- **两阶段均完成**（各跑满 50 epoch 上限、未触发早停；train/val 曲线见 history JSON）：
  | 阶段 | wall | best val loss（best epoch） | 末 epoch val 分量 | test smoke（384/128 口径） |
  |---|---|---|---|---|
  | stage1 VAE | 3956.4 s（66 min） | **1.5526**（E50/50） | E50: recon 0.353 / corr 0.694 / cons 0.477 | MAE 1.0435, std_ratio 1.3377（proj bootstrap 路径） |
  | stage2 CFM 流（mexhat-graph） | 3616.3 s（60 min） | **0.5977**（E49/50） | E50: recon2 0.301 / corr2 0.109 / cons2 0.195 | **MAE 0.8708, std_ratio 1.0431**（flow 路径） |
- **阶段顺序证据**：stage2 config `checkpoints.stage1_vae` = run_01 内 stage1 产物路径；日志「flow keys re-initialized fresh: 15 missing, 4 unexpected（graph 流预期重初始化，非 flow 键零失配）」；「Trainable params (flow only): 476,357」= VAE 全冻结。
- **数据口径证据**（入口断言实测输出）：manifest 48/34/4/10 OK，test=[10,11,12,17,24,28,30,44,61,69]，HC 24/MDD 24，48 例 npz 字段+形状全过，每例窗数==930；windows 31620/3720/9300（train/val/test），norm=训练折统计（files_hash=e276f424b3df）。
- **产物**：`outputs/10-19-latent-flow-edra-fmt-train/run_01/`——checkpoints/{fold0_edra_fmt_vae_best.pt 12.8MB, fold0_edra_fmt_flow_best_graph_mexhat.pt 5.0MB}、configs/{stage1,stage2}_config.json（含 git HEAD 3aec527、manifest sha、窗口/归一化记录）、logs/{stage1,stage2}.log + history JSON（stage1 wrapper 自写 / stage2 复制自包内 logs/edra_fmt/）+ 冒烟 history 证据；geometry_edra_fmt.pt（npy 基座，D4）落数据根。
- **回归与隔离**：全量 pytest **67 passed / 0 failed / 17 skipped**（终态复跑 10.3s，冻结口径不变）；`git status` 仅 5 改（三包内 additive + .gitignore + README-port）+ 2 新（两 wrapper）+ 任务目录；包内无 checkpoints/；数据根仅新增豁免单文件 geometry_edra_fmt.pt；gpfm 文本命中 3 处均为只读引用（manifest 路径常量 + docstring + README-port 说明段），`import.*gpfm` 零命中、无代码依赖。
- KL 观察：两阶段 KL 原始值 ~93（beta=1e-4 贡献 ~0.009）——与旧库 fold1-4 已登记行为同型（08-01 任务曾以 beta=1e-3 消融），如实登记不改判。

### 结论

- **支持**：latent-flow-v1 × edra_fmt 两阶段管线在满量 34/4/10 例 × 930 窗上稳定可跑且收敛（两阶段 val loss 均单调下降至 50 epoch 上限，无发散/NaN/OOM）；数据划分与时间窗严格复用上一轮固化口径（断言全过），产物完全隔离可复现。θ 候选（stage2 flow ckpt + stage1 VAE）已产出。
- 流路径 test smoke（MAE 0.8708 / std_ratio 1.0431）优于 stage1 proj bootstrap（1.0435 / 1.3377），方向符合「流学了动力学而非复制 proj」（旧库 decision-B 修复的动机）；但**这只是 smoke 级 sanity，不是代理脑合格性判定**——按 surrogate-brain-qualification.md，固定 θ\* 前须另走三项检验（属后续任务）。
- 下一步 / 派生决策项：①评估与代理脑合格性检验（另立任务）；②σ 二分归因在本 θ\* 上的小规模冒烟（对照 perturb-ec 口径差异表）；③5-fold 扩展须用户另行拍板（study1 门禁）。

## Run 002 — A_eff 学习结构可视化与统计（2026-10-19，任务归档后追加，轻量无新任务）

### 目的与假设

- 展示并量化 run_01 stage2 mexhat 流学到的有效连接矩阵 A_eff = γ_m·M(d) + γ·W（读 ckpt，零训练）。

### 配置

- 代码：`outputs/10-19-latent-flow-edra-fmt-train/run_01/figures/gen_aeff_viz.py` + viz 扩展 `plot_matrix_heatmap`（latent_flow/viz/connectivity.py，additive）
- 输入：`fold0_edra_fmt_flow_best_graph_mexhat.pt`（只读）+ geometry_edra_fmt.pt（npy 基座）

### 结果

- 图：`outputs/10-19-latent-flow-edra-fmt-train/run_01/figures/aeff_{heatmap,kernel_M,residual_W}.{png,svg,pdf}`；统计：`aeff_stats.json`
- 关键数字：σ学习=0.1456 m（σ/中位电极距=1.20，零跨略超中位距）；A_eff offdiag CV=**116%**（强异质路由，旧库口径 128% 同量级）；**行和 CV=1.0%**（每通道总输入近均衡——谱平衡但路由结构化，与旧库 mexhat 63% 显著不同，首跑无 anti-uniformity 正则下如实登记）；|A_eff|×距离相关 **−0.884**（强距离衰减）；中心-环绕签名：σ内均值 +0.268 / σ外 +0.011（**环绕净抑制未显现**，近零而非负——弱抑制瓣被 γ·W 残差补偿，γ=0.1、W offdiag 均绝值 1.89）；有向性：A_eff 不对称度 25.8%（W 35.1%，非对称残差使矩阵可载方向）；对角均值 0.802。

### 结论

- 支持：mexhat 先验在 edra_fmt 上学到**强结构化局部连接**（局部兴奋 + 距离衰减 + 显著异质路由），且 W 残差引入实质方向性——结构先验迁移假设（D1）的必要面成立；但「中心-环绕侧抑制」的环绕净负未在 A_eff 上显现、行和近均匀，两点与旧库叙述有偏差，判读任何 EC 结论时须随附。
- 派生决策项：行和均衡 + 环绕近零是否影响跨通道传导特异性（Study II 关切）→ 留待代理脑合格性检验/归因任务判。

## Run 003 — |A_eff| × 真实数据双变量 GC EC 逐被试相关（2026-10-19，任务归档后追加，轻量无新任务）

### 目的与假设

- 检验 run_01 stage2 mexhat 流学到的 |A_eff| 与 edra_fmt 真实观测 EEG 的成对双变量 GC EC 是否存在逐被试结构相关（用户指定：Pearson + Spearman、逐被试、mean±std 汇报）。

### 配置

- 代码：`outputs/…/run_01/aeff_vs_gc/gen_aeff_gc_corr.py`（只读 ckpt 与 refs）
- A_eff 源：`fold0_edra_fmt_flow_best_graph_mexhat.pt`（同 Run 002 提取口径）
- GC 参考：`outputs/bivar-gc-ec-fmt-250/refs-01/refs/`（48 例，250 Hz 原始全长档，任务 10-18-bivar-gc-fmt-250；选档依据 = fs 与训练数据一致；125 Hz 档按两档隔离纪律不并列）
- 对齐：`ec[a,b]=a→b（行=源）` vs `A_eff[i,j]=j 驱动 i（行=靶）` → 主口径 = |A_eff| 对 ec.**T**；ec 不转置作方向性检查副口径
- 每被试 offdiag n=3,782 对；Spearman 经 scipy spearmanr

### 结果

- **主口径（方向对齐，|A_eff| vs ec.T）**：Pearson **+0.0130 ± 0.0560**（median +0.0025，range [−0.094, +0.160]）；Spearman **−0.0095 ± 0.0518**（median −0.0060，range [−0.126, +0.138]）；n=48
- 方向性检查（不转置）：Pearson +0.0112 ± 0.0535 / Spearman −0.0137 ± 0.0506——与主口径无差异（有向分量不携带与 GC 方向一致的信息）
- 组均值 GC（48 例平均后单次相关）：Pearson +0.0340 / Spearman −0.0480
- 统计判读：mean/SE ≈ 0.013/0.008（Pearson）——总体相关 ≈ 0；单被试 range 跨零对称
- 产物：`aeff_vs_gc/aeff_vs_gc_stats.json`（含 48 例逐被试原始值）+ 脚本随档

### 结论

- **不支持**：|A_eff| 与真实数据成对 GC EC 无逐被试相关（两指标、两种方向对齐均 ≈ 0）。与旧库登记不矛盾——FOUNDATION 的 DE×GC r_abs 0.585/0.413 指 **σ 二分归因产物**与 GC 的相关，非 mixer 矩阵 A_eff 本身；A_eff 是几何驱动的结构先验（距离衰减 −0.88、局部兴奋），本就不必然追踪逐被试功能耦合。与 Run 002「行和近均匀」观察一致指向：A_eff 编码的是**空间结构先验**而非被试特异功能连接。
- 派生决策项：A_eff 的价值检验应走旧库同款路径（θ\* 冻结后 σ 二分归因 → 归因 EC × GC 对照），属后续归因任务；本结果作为「mixer 矩阵 ≠ 功能连接直接读出」的基线事实登记。

## Run 004 — 同口径补充：offdiag minmax 归一化后重算（2026-10-19，任务归档后追加）

### 目的与假设

- 用户指定：|A_eff| 与 GC 的 offdiag 各做 minmax 归一化到 [0,1] 后再算 Run 003 的相关（验证归一化不改变结论）。

### 配置

- 同 Run 003（参考 250 Hz 档 48 例、方向对齐主口径）；归一化 = **每矩阵 offdiag 全局 minmax**（GC 逐被试、|A_eff| 一次）；脚本同文件已扩展（`minmax()` + invariance 断言）。

### 结果

- 全部相关系数与 Run 003 **逐位相同**（48 例 × 4 序列 max|Δr| = 1.1e-16，脚本内置 atol=1e-12 断言通过）：Pearson +0.0130±0.0560 / Spearman −0.0095±0.0518（对齐）等。
- 结论为数学必然：minmax 是逐变量正仿射变换，Pearson 对仿射不变、Spearman 对严格单调变换不变——归一化前后相关恒等。

### 结论

- **支持（Run 003 结论加固）**：≈0 相关不是量纲/量程伪影，归一化后逐位不变。注：若改用**逐行（逐源）minmax**则非全局仿射、相关会变——该口径未在本 Run 范围（用户如需可另算）。

## Run 005 — 原库方法在 edra_fmt 重算 GC 参考 + |A_eff| 再对照（2026-10-19，任务归档后追加）

### 目的与假设

- 用户指定：用原库（latent_flow baseline/granger.py）方法在 edra_fmt 48 例上重算 GC-EC 参考，再与 |A_eff| 对照（同 Run 003 协议），并可视化重算矩阵。动机 = Run 003 用 npi 版参考得 ≈0，检验是否参考实现口径依赖。

### 配置

- 代码：`baseline/data_utils.py` 补 npz 分支（additive 4 行，与 S1 shim 同款）；`gc_oldlib_refs/gen_refs.py`（重算）+ `gen_compare.py`（对照+图）
- 口径 = 原库默认：成对双变量谱 GC（Geweke 1982/Dhamala 2008），**固定阶 p=8**、65 频点、**1–125 Hz 梯形积分**、全连续时序（60001 样本，不窗化）、逐通道 z-score、fs=250；48 例全量（12 workers × OMP=1，24.2 min，~30 s/例）
- 对照协议同 Run 003（逐被试 Pearson+Spearman over 3782 offdiag 对、方向对齐 = |A_eff| vs ec.T、mean±std）；附加交叉核对：两套 GC 实现彼此相关

### 结果

- **|A_eff| × 原库 GC**：Pearson **+0.4414 ± 0.0929**（median +0.4535，range [+0.142, +0.609]，48 例全正）；Spearman **+0.2012 ± 0.0493**（range [+0.088, +0.314]）；组均值单次 Pearson +0.5849 / Spearman +0.2529；aligned≈flipped（差 <0.001 → 相关由 |A_eff| 对称距离核承载，非方向分量）
- **两套 GC 实现交叉核对**：原库谱 GC vs npi 时域 GC（BIC p\*=24, ln SSR）逐被试 Pearson **+0.0359 ± 0.0756**、组均值 +0.1537/Spearman +0.1577——**近乎不相关**
- 重算矩阵：offdiag mean 36–103、max ~794（ln 单位全带积分，量纲与 npi 版不可比）；图 = 组均值热图 + HC(s13)/MDD(s1) 各一张（magma，行=源/列=靶）
- 产物：`gc_oldlib_refs/refs_s{1..69}.npz ×48` + `aeff_vs_gc_oldlib_stats.json`（含逐被试原始值与交叉核对）+ 3 图 + `gen_refs.log`
- 回归：pytest 67P/0F/17S 不变（data_utils 改动后复跑）

### 结论

- **Run 003 的 ≈0 结论被判定为参考实现口径依赖**：|A_eff| 与原库口径 GC 呈稳定中度正相关（秩相关 0.20 保守口径也 48 例全正）。两套「成对双变量 GC」实现彼此仅 ~0.04 相关——阶选择（固定 8 vs BIC 顶格 24）、估计器（谱域全带积分 vs 时域 ln SSR）、频带加权差异足以让参考矩阵近乎正交，**GC 参考的口径敏感性本身是本 Run 最重要的方法学事实**。
- 判读注意：① Pearson(0.44) ≫ Spearman(0.20) → 线性相关部分由重尾幅度驱动（max ~794 vs mean ~50），秩口径更保守；② aligned=flipped → 相关载体是 |A_eff| 的**对称距离核**（距离衰减结构），与双变量 GC 的局部强耦合（含体积传导/共享源成分）同构——**不能解读为方向性功能连接一致**；③ 与旧库 FOUNDATION 的 DE×GC r_abs 0.413（EDRA）对照：mixer 先验单独×GC 已达同量级，说明旧库归因的相关有相当部分可由几何距离结构解释。
- 派生决策项：两套 GC 参考的口径分叉如何收敛（阶选择敏感性扫描 / 条件 GC / 一致性检验）→ 留后续任务拍板；本 Run 事实全部随档。

## Run 006 — σ 二分 EC 归因首跑：run_01 θ\* × fold0 测试 10 例（2026-10-19，任务归档后追加）

### 目的与假设

- 用原库 SignalProbeSite + global-tau σ 二分方法（Run 前一轮详解的四要素）在 run_01 stage2 mexhat θ\* 上推断 fold0 测试 10 例（manifest [10,11,12,17,24,28,30,44,61,69]）的有向 EC——latent-flow 线首个归因产物。

### 配置

- 代码：`ec_attribution/run_ec_attribution.py`（断点可续；schema 对齐原库 global-tau runner）
- 口径：SignalProbeSite（biased noise=0.5·std 定向偏置、batch 32、seed 42、n_steps 10）+ 两阶段 global-tau（stage1 @σ_ref=0.3 → τ=q0.5 全局分位；stage2 log 网格 24 点 [0.01,0.3] 单调游走+线性插值）；窗 512/64/384；归一化=训练折 34 例统计（hash e276f424…复算一致）
- **每被试 n_trials=32（用户指定）**：930 窗确定性无放回抽样（RandomState(seed·1000+sid)，旧库 subsample_trials 惯例）——快览口径，σ\* 噪声大于全量口径
- 耗时：10 例 14.9 min（全 930 trial 口径探针标定为 42 min/例、10 例≈7 h，备查）
- 产物：`ec_attribution/subject_{id}.json ×10`（σ/conn/signed/response/mean_diff/diagnostics）+ 每例 3 图 + 组均值 2 图 + `group_summary.npz` + config.json

### 结果

- 逐例稳定：τ∈[1.5e-3, 5.2e-3]，σ\* offdiag ≈0.19±0.12，conn CV 1.55–1.87
- **删失结构**：@max ≈50.3–50.7%（**结构性**——τ=q0.5 为全响应分布中位数，故约半数 (a,b) 对在 σ_max 处响应仍低于 τ → 右删失，q<0.5 可缓解，口径备注）；@min 18–31%
- **被试间一致性**：offdiag EC 两两 Spearman **0.623 ± 0.072**（45 对）——中度共享结构（旧库缺陷版曾 0.90 被判「过于不变」，本 θ\* 0.62）
- **EC × 观测 GC ≈ 0（两套参考均不相关）**：组均值 EC × 原库谱 GC Pearson +0.091 / Spearman +0.001（对齐 0.090/−0.002）；× npi 时域 GC Pearson −0.035 / Spearman +0.022（对齐 −0.079/−0.110）——与旧库 FOUNDATION 的 DE×GC 0.585/0.413 不同水平，如实登记
- **EC × |A_eff| = 0.45/0.44（Spearman，转置对齐 0.58）**——归因读出显著继承 mixer 几何先验结构

### 结论

- **流程兑现**：σ 二分四要素在 latent-flow θ\* 上端到端跑通、确定性契约沿袭（seed+source 噪声、单调游走）、产物齐全可复现。
- **结构性证据阴性（与 Study II 2026-10-17 gpfm 结论同型）**：该 θ\* 归因 EC 与两套观测 GC 参考（含 Run 005 中与 |A_eff| 相关 0.44 的原库口径）均零相关——EC 的跨被试一致性主要来自共享几何先验（EC×|A_eff| 0.45-0.58），而非被试特异的功能结构；**本 θ\* 未获「EC 对齐观测因果度量」的初步证据**。
- 口径注意：n_trials=32 快览口径 + q0.5 结构性半右删失，两者都会压缩判别力；全量口径/q 扫描属后续任务决策。
- 派生决策项：①是否按 surrogate-brain-qualification 走正式三项检验再定 θ\* 去留；②归因口径迭代（q、n_trials、τ 定义）vs 换 θ\*（损失侧跨通道监督）——待用户拍板。

## Run 007 — 逐被试 |EC_pert| × GC 相关（2026-10-19，任务归档后追加）

### 目的与假设

- 用户指定：逐被试对比 Run 006 的 |EC_pert|（σ 二分连接强度）与 GC 的 Pearson/Spearman——检验组均值 ≈0（Run 006）是否掩盖被试级异质性。

### 配置

- 10 例测试被试 × 两套 GC 参考（Run 005 原库谱 GC / npi 时域 GC，均逐被试匹配 subject_id）；offdiag 3782 对；**方向两边同约（行=源）无需转置**（EC sigma_matrix 行=源、GC ec 行=源），转置作 flip 检查；|EC_pert| 按字面取绝对值，符号版（conn 不取绝对）作语义对照。

### 结果

- **全部 ≈0，无被试例外**：|EC_pert| 版——原库 GC Pearson +0.0349±0.0490 / Spearman +0.0205±0.0406；npi GC +0.0113±0.0461 / +0.0118±0.0457；符号版同水平（+0.041/+0.019 与 +0.013/+0.012）；flip 检查同（+0.034/+0.017）。
- 逐被试 range：Pearson ∈ [−0.068, +0.102]，单被试最大 s30 +0.102——无任何被试达中度相关；无 HC/MDD 分离迹象。
- 产物：`ec_attribution/ec_vs_gc_persubj.json`（逐被试全量原始值）。

### 结论

- **支持（Run 006 结论加固）**：EC × GC 零相关不是组均值掩盖的异质性——被试级逐一为零。该 θ\* 的归因 EC 与观测 GC 在任何粒度、任何参考实现、任何方向对齐下均不对齐；跨被试一致性 0.62 的来源（几何先验，EC×|A_eff| 0.45-0.58）与 GC 无关。与 Study II 2026-10-17 阴性结论同型，且本线在 48 例全量 GC 参考下依然成立。

## Run 008 — 无偏噪声（纯 N(0,1)，μ_bias 关）同口径 EC 对照（2026-10-19，任务归档后追加）

### 目的与假设

- 用户指定：噪声构造切换为无偏（`use_biased_noise=False`，纯 σ·ξ），n_trials=32、同 θ\*/同 manifest 10 例/同 global-tau 口径重推 EC——检验定向偏置对归因读出的影响。

### 配置

- 脚本参数化（`--unbiased --out-root`，Run 006 产物零触碰）；产物根 `ec_attribution_unbiased/`；方向号（signed EC）在无偏噪声下**无效**（mean_diff≈噪声），不产出 signed 图；其余口径与 Run 006 逐项同（seed 42 同噪声模板、同 log 网格、同 τ 定义）。14.8 min / 10 例。

### 结果

- **@min 删失 = 0.0%（10/10 例）**，vs 有偏版 18–31%——去掉常值偏置后小 σ 纯方差效应不再瞬间越阈；@max ≈50% 维持（q0.5 结构性）；τ 1.9e-3~7.5e-3，σ\* offdiag ≈0.20±0.11（有偏版 0.19±0.12 同水平）。
- 被试间一致性 Spearman **0.646 ± 0.071**（有偏版 0.623）。
- **EC(无偏) vs EC(有偏) 逐被试 Spearman 0.919 ± 0.010**——两种噪声构造的 EC 读出高度一致（μ_bias 主要改变删失结构，不改变连接排序）。
- |EC| × GC 逐被试：原库 GC Pearson +0.0471±0.0493 / Spearman +0.0221±0.0451；npi GC +0.0051/+0.0064——**零相关维持**（Run 007 结论在无偏构造下复现）。
- EC × |A_eff|：Pearson 0.4369 / Spearman 0.4285（转置对齐 0.5714）——与有偏版（0.45/0.44/0.58）同水平，几何先验承载结论不变。
- 产物：`ec_attribution_unbiased/`（10 JSON + σ/conn 图 ×10 + 组均值图 + group_summary.npz + run_ec.log）；Run 006 脚本现承载两种口径（config.json 记录 use_biased_noise）。

### 结论

- **支持**：噪声构造（有偏/无偏）对该 θ\* 的 EC 排序读出几乎无影响（0.919 一致），结论体系（跨被试一致来自几何先验、与观测 GC 零相关）在两种构造下稳定；主要差异是删失结构（@min 0% vs 18-31%）与方向号可用性（有偏独有）。
- 口径备注：无偏版 @min=0 意味着全部强连接判定来自区间内 σ\*，判别力略好于有偏版（其 18-31% 挤在 σ_min 边界桶）；后续若做正式归因，无偏构造 + 另配方向号方案（如成对偏置 ±μ）是更干净的分解。

## Run 009 — 固定 σ=0.5 无偏响应 EC：单次前向 + MSE 直读（2026-10-19，任务归档后追加）

### 目的与假设

- 用户指定口径：无偏噪声 ξ~N(0,1)、σ=0.5、**每源一次前向不搜索 σ\***、**逐靶 MSE 直接作为 a→b 连接强度**，可视化并汇报与 GC 的相关（即 `compute_relative_response` 响应矩阵读出，σ 取 0.5 超出旧库 [0.01,0.3] 上限）。

### 配置

- 同 Run 006/008 口径（10 测试例、n_trials=32 确定性抽样、训练折归一化、同 θ\*、噪声模板 seed+source 跨被试共享）；读出 = 扰动前向与冻结基线的逐靶 MSE，行=源/列=靶，大=强（无熵式翻转）。0.7 min 全量。
- 产物：`ec_response_sigma05/`（npz + config + 每例 log10 MSE 热图 + 组均值热图）

### 结果

- 逐例 offdiag MSE：mean 2.6e-3（s24/s30）~ 1.05e-2（s61），max ≈ 2e-2~6e-2——被试间幅度差 ≈4×，与各被试归一化后通道 std（0.63~1.16，上一轮实测）单调对应。
- **EC_mse × GC 仍 ≈0**：逐被试——原库 GC Pearson +0.0509±0.0421（range [−0.022,+0.119]）/ Spearman +0.0197±0.0430；npi GC +0.0158/+0.0204；组均值单次 Pearson +0.103（oldlib）/ +0.031（npi），flip 同水平。
- 一致性：被试间 Spearman **0.777±0.050**（σ\* 版 0.646——直读响应更被试不变）；EC_mse vs σ\*-EC 逐被试 Spearman **0.909±0.003**（读出高度等价）；EC_mse × |A_eff| 0.448/0.440（转置 0.566）——几何先验承载同前。

### 结论

- **支持（Run 006–008 结论第四次复现）**：从 σ\* 搜索退化为固定 σ=0.5 单次前向 MSE 直读、或噪声构造切换，EC 读出排序稳定（0.91 一致）、与观测 GC 零相关稳定（两套参考、逐被试与组均值、flip 检查全零）。该 θ\* 的扰动响应结构由 mixer 几何先验主导（EC×|A_eff| 恒 ≈0.44-0.45），不含对齐观测 GC 的功能结构——阴性结论对读出口径鲁棒。
- 口径备注：MSE 直读的幅度含被试间归一化 std 差异（跨被试幅度不可直接比，排序相关不受影响）；σ=0.5 超旧库上限属用户指定口径，响应大概率进入饱和区（多数边 @max 删失的 σ\* 版佐证），对 Pearson 的重尾影响与 Run 005 同型注意。
