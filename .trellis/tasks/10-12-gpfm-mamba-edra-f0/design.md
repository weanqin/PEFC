# Design — GP先验流匹配 Mamba 代理脑（edra fold0）

> 方法适配研究（1.0c 产物）：`research/gpfm-mamba-adaptation.md`（含核对比表与风险分析）。本 design 与该文档冲突时，以用户拍板结果为准并同步两处。

## 1. 总体架构与数据流

```
npz (edra v4, 62ch×85000×受试者)
  └─ EdraWindowsDataset(fold=0, qc过滤, stride)        # dataloader 内：z-score(train折统计)+窗切分
       X_past ∈ R^{B×62×384},  X_future ∈ R^{B×62×128}
  ┌──────────────────────────────┬─────────────────────────────────┐
│ GPR 条件先验（预计算固定算子）  │ Mamba 速度网络 v_θ(x_t,t,X_past) │
│ μ=A·y^p (128×384 时间侧算子)    │ past 384 token = 前缀条件        │
│ x0^f = μ + L_s·ε·L_tᵀ (时空可分)│ future 128 token = x_t+时间嵌入  │
└──────────────┬───────────────┴──────────────┬──────────────────┘
               │  x_t = (1−t)x0 + t·x1 (+σ_min·η),  t~logit-normal(−0.8,0.8)
               ▼                              ▼
        速度头 v_θ ──► 派生预测 x̂1 = x_t + (1−t)·v_θ
               ▼
   loss = 1·mae + 1·cons + 0.1·tv + 0.1·corr   （无速度回归项）
               ▼
   采样：Euler ODE（默认16步）从 GPR 先验积分 → 生成未来段
               ▼
   前置验证（fold0 单 seed）：MAE 主指标 + CRPS/r 次指标；GC/PSD/视觉三检验
```

## 2. 数学定义

### 2.1 GPR 条件先验 — 时空可分核（用户 2026-10-12 指定：`K_s ⊗ K_t` 乘积形式）

联合先验引入跨通道相关：`Cov(x0[c,t], x0[c',t']) = K_s(c,c') · K_t(t,t')`（对官方逐通道独立口径的结构性扩展）。

- **闭式解（全观测可分 GPR，数学推导已核验）**：Σ_joint = K_s ⊗ K_t，past 全观测（62 通道×384 点全观测）⇒
  - 后验均值算子 **A 与逐通道独立情形完全相同**（K_s 在 Σ_fp Σ_pp⁻¹ 中消去）：`μ = A·y^p`，`A = K'_t,fp (K'_t,pp)⁻¹`（128×384，纯时间侧）；
  - 后验协方差**保持可分**：`Σ_{f|p} = K_s ⊗ K'_{t,f|p}`，`K'_{t,f|p} = K'_t,ff − K'_t,fp (K'_t,pp)⁻¹ K'_t,pf`（128×128）；
  - 采样：`x0^f = μ + L_s·ε·L_tᵀ`，`ε ~ N(0, I_{62×128})`，`L_s = chol(K_s)`（62×62）、`L_t = chol(K'_{t,f|p})`——**两个小矩阵 Cholesky，固定算子预计算性质完整保留**（比逐通道实现仅多一个 62×62 因子，开销可忽略）。
- **时间网格与时间核**：`τ_i = i·π/p`（p=50，α 节律周期）；γ=1；K_t 默认 **OU**（诊断：数值最稳、后验展宽≈1.0 匹配 z-score 尺度；候选对比见 adaptation Q1）。
- **白噪/jitter 折叠口径（保持 Kronecker 结构的关键）**：白噪**不**加联合恒等项（会破坏可分性且需 23808 维求解），而是折叠进因子：`K'_t = K_t + iso·I_T`（iso=1e-2；含义=时间白噪被 K_s[c,c'] 加权跨通道共享，登记为口径）；K_s 构造自带对角占优（几何/经验核均单位对角）；`K'_t,pp` 求逆前额外 `+1e-4·I`；Cholesky 失败即报错不静默。
- **实现要点**：A 用 `cholesky_solve`（非显式求逆）；全部 float64 构造 → float32 buffer；每通道边缘方差 = diag(K'_{t,f|p})（K_s 单位对角）⇒ z-score 尺度匹配性质与逐通道版一致。
- **空间核 K_s（已定，用户 2026-10-12：Mexican-hat 的 Ricker 形式）**——采用**三维正定形式** `K_s(c,c') = (1 − d²/(3σ²))·exp(−d²/(2σ²))`，d=头皮欧氏距离（chan_coords_62.npy）：
  - **为何不是严格 1D Ricker** `(1 − d²/σ²)e^{−d²/2σ²}`：该形式仅在一维正定（FT=ω²e^{−ω²}≥0），径向代入三维距离不保证 PSD——**本机实测**真实 62 电极点集上 σ∈{0.030,0.045,0.060,0.090} 全部非正定（min_eig −4.0 ~ −10.5）；硬用须特征值裁剪（静默改核形状），弃。三维形式 = −∇²高斯（三维 Mexican hat），FT = k²·Gaussian ≥ 0 对任意 ℝ³ 点集**保证 PSD**，同族小波、同中心-周边语义，实测四档 σ 全部 PSD ✓。
  - **形状与默认 σ=0.080 m（用户 2026-10-12 定）**：单位对角（K_s(0)=1，边缘方差匹配保持）；实测轮廓——邻位(0.030) +0.89、0.060 m +0.63、0.090 m +0.30、0.121 m +0.08、过零 √3σ≈0.139 m（**在阵列内**）、0.160 m −0.045、0.190 m −0.053，负电极对 702/1891（37%）——中心同步 + 周边抑制语义在阵列内真实表达；有效秩 7.5/62（top-1 模式 22%，空间多样性充足）。σ 扫描 {0.045, 0.080, 0.137} 入诊断步（2.3），三档语义：强中心-周边 / **平衡（默认 0.080）** / 全局相干（负瓣出阵列，过零 0.237 m > 最大距离 0.190 m，有效秩仅 ~3，非默认）。
  - **数值口径**：LoG 核在 ℝ³ 积分为零 ⇒ σ 增大时常数向量方向特征值趋零（结构性）；σ=0.080 时 min_eig +4.0e-7（真实余量），按惯例 `K_s ← K_s + jitter_s·I`，jitter_s=1e-3（加 jitter 后条件数 1.4e4，float64 预计算安全）。
  - 语义与归因：K_s 注入的是**零滞后**中心-周边反相关（有向滞后结构仍由网络时间混合学习）；指标 4（R̂ vs K_s vs 真实 R 三方对比，§6）为归因测量；identity-K_s 消融臂留作需要时的薄变体；经验相关核弃（用户已选 Ricker，不再作为 -v2 默认项）。

- **时间核 K_t 候选**（沿用 adaptation Q1 诊断，作为 §2.1 已定时间侧默认 OU 的依据）：OU（R²(μ)=0.01–0.03，post_std 0.97–1.02，Σ_pp 条件数 ~2.4e2 最稳，**默认**）；Matérn-3/2（粗度 0.5–0.7× 最贴近 GT，备选 -v2）；SE（iso=1e-4 数值崩溃，需 iso≥1e-2 且无优势）；纯 Periodic（后验塌缩 post_std=0.010，**排除**）；SE×PE（留作节律缺失时后续变体）。
- **对 Q2 的解释口径（必须登记）**：K_s 注入的是**零滞后**跨通道相关；GC 基于滞后预测改善，理论上不被零滞后相关直接伪造，但生成组的空间结构部分来自先验而非网络所学——指标 1（GC 一致性）结论须注明此归因局限，并以 identity-K_s 消融量化先验贡献。K_s 同时是 d_model=16 信息瓶颈（§9.7）的先验侧缓解措施（跨通道零滞后结构由先验承担，网络专注时序动力学）。
- **有意偏离登记**：① 时空可分核本身是对论文逐通道独立先验的结构性扩展（用户指定）；② loc_freq 周期中心化跳过（EEG 无稳定周期，z-score 已在 dataloader 完成）。
- 诊断步骤（implement 2.3）扩展：可分采样的边缘统计（每通道方差 ≈ diag(K'_{t,f|p})、通道间相关 ≈ K_s）、**A 不变性**（与逐通道实现逐元素一致，rtol 1e-5）、两因子 Cholesky 全候选通过。

### 2.2 流匹配路径、时间采样与一步生成（用户 2026-10-12 定稿）

- 路径（论文 Eq.4，OT/rectified）：`x_t = t·x1 + (1−t)·x0 + σ_min·η`，σ_min=1e-4，η~N(0,I)。
- **t 采样（有意偏离论文 U[0,1]，登记）**：`t = sigmoid(z)`，`z ~ N(−0.8, 0.8²)`（logit-normal）——中位数 t≈0.31、集中低中段，t>0.8 概率 <1%，从源头规避 (1−t)→0 的梯度消失区（§9.1）；实现上 clamp 到 [1e-4, 1−1e-4] 防数值边界。该分布使训练目标成为按 logit-normal 密度加权的 FM 期望（重要性加权口径，高低 t 两端仍有非零覆盖）。
- **速度头参数化**：网络输出 `v_θ(x_t, t, X_past)`（ODE 采样所需的标准 FM 速度场）。
- **一步生成（训练期规格，用户定稿）**：训练中的「生成」一律是**单步映射** `x̂1 = x_t + (1−t)·v_θ`（从 x_t 一步到 t=1 的估计），**训练全流程不做 ODE 积分**；四项损失全部作用于该一步预测。多步 Euler 积分**仅用于训练后的采样/评估**（§5，步数 16 可配）。
- 已知性质：t→1 时 (1−t)→0 的梯度衰减——logit-normal 采样已把该区训练权重压到 <1%，仍保留按 t 分桶监控作验证。

### 2.3 复合损失（初始权重 w = {1, 1, 0.1, 0.1}，全部可配置；定义域=future 128 列）

对一步生成 `x̂1, x1 ∈ R^{B×62×128}`（B 窗、62 通道、T=128 未来步；逐窗 t_b ~ logit-normal）：

- **mae** = `mean_{b,c,τ} |x̂1 − x1|` —— 逐元素绝对误差的全均值（分母 B·C·T）。
- **cons** = `mean_{b,c} [ |std_τ(x̂1) − std_τ(x1)| + |mean_τ(x̂1) − mean_τ(x1)| ]` —— 逐 (b,c) 沿时间轴计算统计量（std 用 torch 默认 ddof=1），两 L1 项先相加再对 B·C 求均值。
- **tv** = `mean_{b,c,τ} |x̂1[τ+1] − x̂1[τ]|` —— 生成侧时间一阶差分 L1（分母 B·C·(T−1)=127）；**纯平滑惩罚、不对照 GT 差分**（默认口径，登记）。
- **corr** = `mean_{b,c} [1 − r_bc]`，Pearson r 用和式定义（分母自洽、无 ddof 歧义）：`r = S_xy / (√(S_xx·S_yy) + eps)`，`S_xy = Σ_τ (x̂−μ̂)(x−μ)` 等；r 截断到 [−1,1]；近平坦通道（S_xx 或 S_yy < eps=1e-6）置 r=0 后纳入均值。
- **总损失** `L = 1·mae + 1·cons + 0.1·tv + 0.1·corr`（batch 级加权求和，四项分别记录）。
- **best checkpoint 判据（用户定稿）**：内部验证集上的**总 loss**——确定性复算：固定 t 网格 {0.1, 0.3, 0.5, 0.7, 0.9} + 固定 ε（seed）下复合损失对网格均值，跨 epoch 可比；每 epoch 末评估，创新优即覆写 best。
- corr 数值稳定补充：H0 下 E[1−r]≈1 的损失地板为无害常数（日志体现）；全部四项 NaN guard。

## 3. Mamba 速度网络（d_model=16）

- Token 序列长度 512：前 384 为条件前缀（past），后 128 为流变量位置（future）。
- 输入特征（逐时间步）：62 通道值 + 段类型嵌入（past/future 可学习标量）；future token 额外加全局 t 的正弦嵌入（dim 16，广播到所有 future 位置）。投影 Linear(→16)。
- 骨干：`n_layers=4` 个 Mamba 块（预 RMSNorm + 残差）：`d_model=16`（用户定）、`d_state=16`、`d_conv=4`、`expand=2`（d_inner=32）。因果（单向）——past 前缀天然作为条件被 future 位置通过递归看到。
- 输出头：RMSNorm + Linear(16→62)，仅取 future 128 位置 → `v_θ ∈ R^{B×62×128}`。
- EMA（**decay 0.999**，2026-10-12 口径变更：论文 0.9999 的 τ≈10⁴ 步 > 本程 17.5k 步的前半程；默认经 config 传入），推理/评估用 EMA 权重（可配置）。
- **实现路线（已定）**：官方 mamba-ssm——**已安装并验证（2026-10-12 用户通知）：mamba_ssm 2.3.1 + causal_conv1d 1.7.0，GPU 前向冒烟通过（d_model=16/L=512），triton 3.4.0 未被动**；安装由用户委派另一会话完成，本任务侧全程未安装。代码按官方 API（`mamba_ssm.Mamba`）编写。单测要求：前向形状、因果性（扰动 future 输入不改 past 位置输出）、梯度有限。

## 4. 数据模块

- `EdraWindowsDataset`：受试者级 5-fold 分层（MDD/HC，划分 seed=42 登记后固定），fold0；**受试者过滤（2026-10-12 用户门禁拍板 b）：manifest `status=='ok'`（46 例 HC24/MDD22；qc_pass=True 仅 13 例——33 例挂 s7_interp_carry 软门但管线标 ok）**；**划分固化**：fold0 划分 manifest 落 `models/gpfm-mamba-v1/splits/fold0_status_ok_seed42.json`，此后每次构建校验一致、漂移即报错；窗切分 stride=64（fold0 训练折 34 例 44914 窗）；训练折内切 2 例（25,40）作内部验证（模型选择用，登记）。
- z-score：逐通道均值/方差，**仅训练折（不含内部验证/测试）估计**，在 dataloader 内应用；统计量随 Run 登记。
- 输出张量：`X_past (B,62,384)`、`X_future (B,62,128)`，float32；`group` 标签随 batch（分层核对用）。模型侧零变换（spec 纪律）。

## 5. 训练与采样

- 优化：AdamW（lr=1e-3, wd=1e-4, cosine 退火），grad clip 0.5、**EMA 0.999**（2026-10-12 口径变更登记：论文 0.9999 的 τ≈10⁴ 步在本程 17.5k 步下前半程影子权重严重滞后，0.999 的 τ≈10³ 步≈5.7 epoch 匹配短程；评估/采样用 EMA 权重），batch=256（4080S 32GB 足够，参数量 ~10⁵ 级），epochs=100（固定训练，不做早停；**best 按内部验证总 loss（固定 t 网格确定性复算，§2.3）每 epoch 评估更新**，last 每 epoch 覆写）；46 例口径实测 8.78 s/epoch、100-epoch ≈ 14.6 min（nohup 后台）。
- **1-epoch 基准测试（用户要求，含确认门禁）**：全量训练折数据跑 1 epoch，记录 peak 显存（`torch.cuda.max_memory_allocated`）、epoch 墙钟时间、it/s、样本吞吐、四项损失走势与按 t 分桶 sanity，外推 100 epoch 总时长并登记 experiment.md 与 run 目录；**基准摘要汇报用户 → 用户确认后方可进入完整训练，或先据结果优化模型（未确认不启动 §5 完整训练）**。
- 每 epoch 记录：train 四项损失/总损失、内部验证总 loss（+四项）、按 t 十分桶的 mae、‖v_θ‖ 均值、生成幅度监控（EMA 权重下 x̂1 的 std_t 均值）。
- **监控 / checkpoint / 断点续传（R9，用户 2026-10-12 新增）**：
  - 监控：训练每 step 进度行（epoch/step/total、四项损失+总损失、it/s、ETA、lr）→ stdout（nohup 日志）；每 epoch 追加 `logs/metrics.jsonl`（上述全量指标 + 耗时）；采样/评估各阶段（逐批采样、GC 逐阶段、PSD）打印进度与 ETA，**且评估侧结构化结果与进度同样落盘**（`logs/eval.log` + `logs/eval_metrics.jsonl`，便于事后分析）。零新依赖：jsonl + visualizer 出曲线，不用 tensorboard。
  - Checkpoint：**每 epoch 结束**——内部验证**总 loss**（固定 t 网格 {0.1,0.3,0.5,0.7,0.9} + 固定 ε 的确定性复算，§2.3）创新优 → 覆写 `checkpoints/gpfm-mamba-f0-best.pt`；无条件覆写 `checkpoints/gpfm-mamba-f0-last.pt`；二者均含全量续传状态（model + EMA + optimizer + scheduler + epoch/step + python/numpy/torch/cuda RNG 状态 + best 指标 + config hash），参数量 ~10⁵ 级、单文件 <2 MB。
  - Resume：`train.py --resume <last.pt>` 恢复全部状态从下一 epoch 继续；dataloader 以 epoch 种子化 generator（resume 后数据顺序可复现）；续跑按 artifacts-management 口径记为**同一 Run 的延续**（experiment.md 补记断点位置与续跑差异）。
- 采样：Euler ODE，t: 0→1，步数默认 16；每窗采 K=16 条（评估）。
- **确定性与随机性（reproducibility 契约）**：python/numpy/torch/random seed 同源固定 + dataloader shuffle/worker seed；`torch.backends.cudnn.deterministic=True`、`benchmark=False` 显式声明入 config；关不掉的非确定性来源（如原子浮点归约）在结论中声明量级。
- **产物目录（reproducibility 契约）**：`outputs/gpfm-mamba-f0/run_<id>/` 下 `config.yaml`（resolved 全配置含 seed，跑前 dump）、`env_snapshot.txt`（pip freeze）、`logs/train.log`、`logs/metrics.jsonl`、`checkpoints/`、`figs/`、`eval/`；复现三要素（命令+快照+resolved config）齐备 Run 才完结。
- 跑前登记：experiment.md Run 块（目的/配置含 seed=1234 与指标口径/代码 commit）。

## 6. 评估（前置验证，fold0 单 seed）

**主指标唯一**：测试折（fold0 held-out 受试者）`MAE = mean|x̂(1) − x1|`，x̂(1) 为 **ODE 积分终点样本**（与训练期派生预测 x̂1 在 t=1 处重合，但评估一律用采样终点、不用训练期公式）；点预测口径：每窗 K 条样本取均值 x̄ 计 MAE（登记）。次指标：样本 CRPS（numpy 公式）、逐通道 Pearson r 均值、与「持久性基线（x̂=past 末值复制）」「GPR 后验均值基线（x̂=A·y^p）」对比。

**Q2 三检验**（surrogate-brain-qualification.md）——**计算口径默认 future-only**（评审 A 发现：整窗 512 点中 75% 是两组共享的真实 past，GC/Welch 估计会被真实段主导而「免费抬高」相似度，削弱对 Q2 的证别力；若用户拍板保留整窗，则作为附加对照同时报告，口径入 Run 块）：
1. GC 连接一致性（future 128 点，真实侧取同位置真实 future）：全模型 VAR(p) OLS（Gram 矩阵一次算出，**BIC 在真实组上选 p∈{1..10} 后两组共用**，口径登记）；成对条件 GC 用块 Schur 补/列删除小规模重解求 F 统计（避免 3782 次全量重拟合，分钟级）；有向 62×62 矩阵去对角展平，真实组 vs 生成组（生成侧每窗固定 seed 取 1 条样本）算 Pearson r + Spearman ρ + 95% CI（对窗/被试 bootstrap 或置换，次数登记）+ BH-FDR 附 raw p；GC 参数两组完全一致并登记。
2. 人工视觉检验：测试集若干窗的真实 vs 生成多轨迹叠加图（统一 visualizer），交用户判定。
3. PSD 相似性（future 段，两组完全相同的分段与拼接方式）：测试折按 512 点周期对齐抽取互不重叠的未来段、逐样本拼接成连续序列后做 Welch（Hann，nperseg=512，分辨率 0.98 Hz；不拼接的退路口径为逐 future 段 128 点、3.91 Hz，登记取舍）；逐通道真实 vs 生成谱 log-log 叠加 + 全通道平均谱对比图。
4. **零滞后空间结构归因诊断（K_s 解释口径的测量手段）**：测试折上计算生成样本 62×62 零滞后相关矩阵 R̂，与 K_s、真实 R 三方对比（Frobenius/上三角相关 + 热图，经 visualizer）——R̂≈K_s≠R ⇒ 先验主导；R̂≈R（几何 K_s≠R 时更有说服力）⇒ 网络习得真实空间结构；identity-K_s 消融臂（薄变体，需用户同意才跑）提供下界对照。依据：本任务四项损失均逐通道分解，无显式跨通道监督，生成侧零滞后空间结构不受损失直接约束（design §2.3），须实测归因；GC 所需滞后结构不受 K_s 直接供给（零滞后核），经此诊断与指标 1 分开陈述。

Q1（预测质量）与 Q2（代理脑有效性）结论在 experiment.md 分开陈述。

## 7. 路径清单（path-isolation 契约）

| 类型 | 路径 | 说明 |
|---|---|---|
| 新建代码（变体） | `models/gpfm-mamba-v1/` | config.py / data.py / gpr_prior.py / mamba.py / model.py / losses.py / train.py / sample.py / evaluate.py / gc_psd.py / tests/（脚本式单测，零 pytest 依赖） |
| 新建共享代码 | `common/visualizer.py` | 仓库首个代码任务，按 visualization.md 创建统一 visualizer（nature-figure figure contract）；作为共享只读区起点，本任务声明依赖，后续变体只读复用、扩展走共享区演进 |
| 只读数据 | `/root/autodl-tmp/edra_process_v4/` | 数据契约 `.trellis/spec/data/edra.md` |
| 新建输出 | `/root/npi/outputs/gpfm-mamba-f0/run_<id>/{config.yaml, env_snapshot.txt, logs/, checkpoints/, figs/, eval/}` | **outputs 根（用户 2026-10-12 已定）：`/root/npi/outputs/<任务名>/`**——按任务名独立目录、落仓库内、`.gitignore` 已排除 outputs/ 不入版本库（轻量产物；数据集等大文件仍在数据盘） |
| 登记 | `{TASK_DIR}/experiment.md` | Run 块四要素 + 产物互链 |

无既有变体冲突（`models/` 与 `common/` 均为本任务首建）；无跨变体中间产物读取。outputs 根在仓库外（数据盘），且 `.gitignore` 已含 `outputs/` 与 `*.pt`，版本库无需额外动作。

## 8. 决策点状态

| 决策 | 状态 | 值 |
|---|---|---|
| 骨干 / d_model / 损失形式 / 权重 / 数据 fold0 / GPR 条件先验 / 核=时空可分 K_s⊗K_t / GC-PSD future-only / outputs 根 / **K_s=Ricker(Mexican-hat 三维正定形式, σ=0.080 m)** / **t~logit-normal(−0.8,0.8) + 训练一步生成 + best=内部验证总 loss** | **已定（用户）** | Mamba, 16, mae+cons+tv+corr {1,1,0.1,0.1}；K_s σ=0.080 m + jitter_s=1e-3；`/root/npi/outputs/<任务名>/` |
| K_t 时间核 | AI 默认（诊断依据） | OU（γ=1, τ=i·π/p, p=50, iso=1e-2）；Matérn-3/2 备选 |
| Mamba 实现 | **已定并完成**：官方 mamba-ssm 2.3.1 | 已装已验证（2026-10-12），triton 3.4.0 未被动 |
| identity-K_s 消融臂 | AI 默认**不跑** | 需 K_s 归因时建薄变体 `gpfm-mamba-v1-ksid` 再跑 |
| 其余超参（层数/stride/lr/epochs/采样步数等） | AI 默认，全部入 config 并登记，评审可改 | 见 §3–§6 |

## 9. 风险与缓解

1. **(1−t) 梯度衰减（adaptation R1）**：**已从源头缓解（用户定稿）——t~logit-normal(−0.8,0.8) 使 t>0.8 的训练权重 <1%**；仍监控分桶 loss-vs-t + 采样终点误差作验证；若诊断显示高 t 区仍欠约束，预案：t 分布参数微调或（需用户同意）小权重速度项。
2. **tv 纯平滑惩罚 → 幅度/高频压制**：监控 PSD 高频段与幅度直方图；预案：权重降至 0.01–0.05 或改为对照 GT 差分（口径变更须登记）。
3. **GPR 先验均值主导**（μ=Ay^p 外推已接近 x1 时模型退化为小修正）：监控 ‖v‖ 与 x̂1−μ；本数据 GPR 均值 R²≈0–0.05，先验价值是结构化基分布而非预测增益，结论中如实陈述。
4. **corr 数值稳定**：eps + 截断 + 近平坦通道屏蔽 + nan_to_num 守护；单测覆盖常数序列边界。
5. **Cholesky/PSD**：float64 构造 + iso(1e-2) + Σ_pp 额外 1e-4 jitter；失败即报错不静默；SE 核 iso=1e-4 已知崩溃（选核规避）。
6. **Mamba 路线不确定**：见决策点；推荐纯 torch 自实现（官方 wheel 有 triton≥3.5 import 风险，事实见 adaptation Q3）。
7. **d_model=16 信息瓶颈（adaptation R6）**：62 通道压 16 维，跨通道时序结构可能不足——GC 一致性（指标 1）为首要观测；变体级预案（不动 d_model）：Mamba 块间插 62×62 通道混合线性层或加深 4→6–8 层。
8. **GC 计算量**：全模型 VAR + Schur 补重解，分钟级可控；若 BIC 选出大 p 导致设计阵过大，按受试者抽样（口径登记）。
9. **z-score/划分泄漏**：单测断言受试者不跨折、统计仅训练折。
10. **前置验证通过标准占位**：指标 1/3 通过标准由用户拍板（规范明示），本任务只落数值与图。

## 10. 回滚

- 全部新增位于 `models/gpfm-mamba-v1/` 与 `outputs_root/gpfm-mamba-f0/`，无既有文件修改；回滚 = 删两目录。
- 里程碑提交（数据模块/先验/模型/训练各自 commit），任一阶段失败可回退到上一 commit。
