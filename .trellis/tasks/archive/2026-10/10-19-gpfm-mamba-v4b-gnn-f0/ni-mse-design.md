# 设计 — NI-MSE-50-q0.90：study2 EC_pert 计算路径（信号空间无偏噪声注入 × MSE 响应 × 50 窗 × Q90 门控）

> 2026-10-20，任务 10-19-gpfm-mamba-v4b-gnn-f0 会话内用户指定五要素的设计交付（**设计文档，未执行**；执行属后续另立任务）。
> **定位（v1.4 用户更正）**：本方法**不是独立方法 NI-GC**，而是 **study2 中计算 EC_pert 的一条路径**；**完整 study 路径 = study1-gpfm-mamba-v5 × study2 NI-MSE-50-q0.90**。命名 NI-MSE-50-q0.90 = Noise Injection（信号空间无偏噪声）+ MSE 响应 + 50 窗/例 + Q90 门控。
> 用户五要素原文：① ec 计算约束 = 明确调用 latent_flow 库中 ec 计算方法；② 无偏高斯噪声注入 = 信号空间零均值高斯；③ null 响应 N = 模型 posterior 构造、future 段前 0.1s 的 MSE（维度/聚合/重复数/种子/异常值）；④ 阈值 = N 的 90% 分位数（估计方法/单双侧/稳定性/CI）；⑤ 完整流程（输入/扰动/ec/null/阈值/输出/异常规则/可复现）。
> 先例锚点：`models/perturb-ec-v1/`（A-案增益算子 + CRN 确定性契约 + τ 双口径，spec/research/perturb-ec-pipeline.md）；`models/latent-flow-v1/latent_flow/attribution/bisect_sigma.py`（σ 二分 + 共享中位阈值教训）；归档任务 10-19-latent-flow-edra-fmt-train Run 006–009（有偏/无偏噪声构造、n_trials=32、σ=0.5 直读 MSE 响应矩阵）；**原库 GC = GC 计算主路径**（用户 2026-10-20 拍板，registry 已登记）——本路径的 real GC refs 即其产物，只读复用、不重算（v1.4）。
> **v1.1（2026-10-20，用户四点质询后修订）**：①σ_p 语义/量纲/实测幅度补全（§1.2）；②K_null 256→**64**（Run 007 校准 pilot 实测：精度只依赖池 W×K，§1.3 候选表）；③慢传导敏感性臂（0.2s/全窗）**取消**（用户拍板，只保留 0.1s 最快传导窗，§5）；④阈值流程精简版为新默认（split-half + Q95 旗标；bootstrap/种子重跑/c 无关性/合并 τ 交叉/FDR 变体降级为可选强化臂，§1.4）。
> **v1.2（2026-10-20，用户前提更正 + 拍板）**：阈值流程精简版**用户认可**；**σ_p 重定义于标准化空间**（注入在数据标准化之后、模型输入之前——v1.1 的 c·ŝ_a 物理换算表述废弃为非算子成分）：算子改为 `P(X, a, c): X'[a,:] = X[a,:] + c·ξ`，σ_p ≡ c 无量纲，ŝ_a 仅用于物理单位汇报换算（§1.2 + §2 伪代码同步）；c=0.5 → σ_p = 0.5（标准化空间），等效物理 2.02–3.52（中位 2.46，仅判读用）。默认值 c=0.5 不变。
> **v1.3（2026-10-20，用户认可全方案并钉执行参数）**：null 侧 **K_null=64 × W_null=200 窗/例**（池 12,800/通道 → pilot 外推 CV≈0.67%）；响应侧 **W_pert=50 窗/例**（K=32 不变）；**W_pert ⊂ W_null**（响应窗取自 null 窗子集内——τ 的噪声地板校准在测响应的同一批窗上）；窗子集确定性选择规范与修订后的计算量预算见 §1.5。
> **v1.4（2026-10-20，用户三项更正）**：①**取消「生成信号上重算 GC」整轨**（原要素①的 GC 重算读法废弃）——EC 侧只做 **|EC_pert|（扰动-推断产物）× real GC refs 的 Pearson/Spearman 相关**，refs 只读取 `outputs/10-19-latent-flow-edra-fmt-train/run_01/gc_oldlib_refs`（= `refs/granger_bivariate_spectral/edrafmt48-...` 归档），**不对生成信号重算 GC**；②**扰动 past 条件化不变量**显式化：ξ 先注入 past，扰动后 past 是 GPR 条件先验与 ODE 积分的唯一条件源（§1.2）；③**更名与重定位**：NI-GC → **NI-MSE-50-q0.90**（study2 的 EC_pert 计算路径；完整路径 = study1-gpfm-mamba-v5 × study2 NI-MSE-50-q0.90），文件同步更名 `ni-mse-design.md`（原名 ni-gc-design.md；Run 007 校准产物目录 ni-gc-calib-01 为历史名保留）。
> **v1.4a（2026-10-20，勘误：被试清单实测核对）**：v1.4 的「跨库 ⇒ 被试集不同源 ⇒ 无逐被试配对」论断**错误**——refs 48 例与 v4 status_ok 46 例同属 EDRA 队列，**fold0 test 10 例与 refs 重叠 10/10**；逐被试配对（EC_pert_s × ec_refs_s，n=10）可行并升为**主口径**，组级单次降为副口径（§1.1/§5 已同步）。
> **v1.5（2026-10-20，用户新增计划）**：新增 **v5-fmt-01 训练任务**（gpfm-mamba-v5 × edra_fmt/250 Hz，与 v5-main-01 并列）并在完成后作为 θ\* 接入本路径——**主实例化切换为 v5-fmt-01**（n₀=25、fmt fold0 34/4/10、与 refs 同管线 ⇒ 跨范式 caveat 消除）；v5-main-01 降为过渡实例化（§0/§5 同步；训练任务计划见任务 implement.md S7'）。

## 0. 方法定位与命名

- **NI-MSE-50-q0.90 = study2 中计算 EC_pert 的一条路径**（非独立方法）：对冻结 θ\* 的代理脑，在**信号空间**（通道轴观测信号，非潜空间）向源通道 a 的 past 段注入零均值高斯噪声，以**扰动后 past 为唯一条件源**生成 future；用「前 0.1s 逐靶 MSE 响应矩阵 + posterior null 90% 分位门控」得到**有向 EC_pert**（响应强度 S + 门控 G）。**完整 study 路径 = study1-gpfm-mamba-v5 × study2 NI-MSE-50-q0.90**：study1 供 θ\*（v5-main-01），本路径供 study2 的 EC_pert 与其 × real GC 的相关判读。
- **主实例化（v1.5 切换：θ\* = `v5-fmt-01`，计划中——2026-10-20 用户新增 v5-fmt 训练任务，训练完成后接入）**：gpfm-mamba-v5 于 **edra_fmt（250 Hz）** 训练，run-id `v5-fmt-01`（与 `v5-main-01`（edra_process_v4/500 Hz）并列）；fs=250 Hz，**前 0.1s = 前 25 个 future 样本**（⌈0.1·fs⌉）；fold0 划分只读复用 v1-fmt manifest（34/4/10）。**过渡实例化（v5-fmt-01 落地前可用）**：θ\* = `v5-main-01` best ckpt（EMA shadow，Run 006 口径），fs=500 Hz、n₀=50、v4 fold0 manifest（34/2/10）——跨范式 caveat 见 §5。**收益（用户明示）**：v5-fmt-01 与 real GC refs 同处 fmt 管线 ⇒ 消除跨范式配对 caveat，逐被试配对口径更清晰（refs = fmt 全部 48 例 ⇒ fmt test 任意 10 例必然 10/10 在 refs 内）。
- 「模型 posterior」的操作性定义（按模型族）：gpfm-mamba 族 = 条件流匹配采样 `EulerSampler.sample(x_past, K, eps)`（GPR 条件先验 x0 + σ_min·ε 起步，Euler 16 步；ε 显式传入即完全确定——v1 sample.py 契约）；latent-flow 族 = encoder q_φ(z|x) 采样 + 潜空间流采样后解码。下文以 gpfm 族书写，latent-flow 族仅算子接口不同。

## 1. 五要素落实

### 1.1 EC_pert 与 real GC 对照（要素① v1.4 重定义：不做生成侧 GC 重算）

- **EC_pert = 本路径的扰动-推断产物**：S[a,b]（超限比强度，行=源 a/列=靶 b，§1.5 步骤 F）与门控 G——有向矩阵，方向语义与 GC 同构（a 扰动 → b 响应）。**不对模型生成信号拼接或重算 GC**（v1.0–v1.3 的 ΔEC 轨废弃）。
- **real GC 参考只读复用**：`outputs/10-19-latent-flow-edra-fmt-train/run_01/gc_oldlib_refs`（= `refs/granger_bivariate_spectral/edrafmt48-cont-zscore-fs250-fixed8-f1-125-nf65/` 归档；原库 GC 48 例，ec[a,b]=a→b 行=源/列=靶）。不重算、不改带、不重采样。
- **相关性口径**：offdiag 3,782 有向对展平——EC_pert 与 ec 同为行=源 ⇒ **直接配对（无转置）为主口径**；EC_pert vs ec.T 作方向性检查副口径（direct ≈ flipped ⇒ 相关由对称分量承载，方向信息弱）。指标 = Pearson + Spearman（scipy）。
- **聚合层级（v1.4a 勘误：被试清单已实测核对，v1.4 的「被试集不同源 ⇒ 无逐被试配对」论断错误）**：refs 48 例与 v4 status_ok 46 例为**同一 EDRA 队列**（refs 覆盖 v4 全集中的 45 例，多出 {11,21,48}、缺 {56}）；**fold0 test 10 例 [12,17,18,24,30,38,44,55,58,61] 与 refs 重叠 10/10** ⇒ **逐被试配对可行且为主口径**：每例 EC_pert_s vs 该被试自身 ec_refs_s（offdiag 3,782 对 Pearson+Spearman），n=10 汇报 mean±std/median/range（对齐归档 Run 003/005 协议形态）；副口径 = 组级单次（10 例 EC_pert 逐元素中位矩阵 vs **同 10 例** refs 均值；48 例全均值作敏感性登记）。跨范式（v4 窗/500 Hz × fmt 全长/250 Hz，同人不同处理管线）是配对内的**解释性 caveat 而非实现阻断**（§5）。
- **距离偏 Spearman 判定轨随附**（主口径）：控制变量 = 通道对欧氏距离（geometry_edra_fmt.pt pairwise_d offdiag），残差化后 Spearman——沿用 evaluation-calibers 主指标口径与 Run 006 教训（几何距离结构可独立承载相关，不控制距离则相关不可解读为功能连接一致）。

### 1.2 无偏高斯噪声注入（要素②）

- 算子：`P(X, a, c)`：`X'[a, :] = X[a, :] + c·ξ`，`ξ ~ N(0, I_L)`（**零均值加性**，无偏 = E[噪声]=0；对照归档 Run 006 有偏构造 σ·(1+ξ) 与 Run 008 无偏构造 σ·ξ 的区分——本设计取后者语义）。
- **σ_p 的定义（v1.2 用户前提更正后重设）**：σ_p := c，**定义在标准化空间**（= 噪声注入的实际位置：dataloader 逐通道 (x−mean)/std 之后、模型 forward 之前），无量纲，即「σ_p = 注入噪声的标准差，单位 = 该通道训练折标准差的倍数」。**算子不出现 ŝ_a**——标准化空间内每通道 std = 1（训练折口径），「相对各通道自身 std 等强度扰动」的性质由标准化本身继承，无需逐通道乘子；`channel_std.npy` 只在把结果**换算回物理单位做汇报**时使用（σ_p^phys = c·ŝ_a^phys，描述性换算非算子成分）。测试窗通道 std ≈1 但非严格等于 1（训练折统计的固有性质，标准 z-score 口径内接受）。
- **c=0.5 的实际注入幅度**：σ_p = **0.5（标准化空间）** = 该通道训练折信号 std 的 50%（SNR = 1/c = 2 即 6 dB；注入后方差占该通道总方差 c²/(1+c²) = 20%）。物理单位换算（仅供判读，v5-main-01 训练折统计实测：channel_std 4.03–7.04、中位 4.91）⇒ 等效噪声 std 2.02–3.52（中位 2.46）。
- 注入位置 = 模型输入层 X_past 通道 a 全段（Stage II 正典位置，perturb-ec-v1 A-案同位；增益算子换为加性噪声；**位于标准化之后**——v1.2 明确）。
- **扰动 past 条件化不变量（v1.4 显式化，用户要求②）**：ξ 先注入 past 段，**扰动后的 past 是 future 生成的唯一条件源**——(i) GPR 条件先验的 posterior_mean 由扰动 past 计算（x0 = bandlimit(μ′ + L_s ε L_tᵀ)，μ′ = prior.posterior_mean(x_past′)；L_s/L_t 为固定核不随数据变）；(ii) Euler 积分每步的条件输入 xp′ = 扰动 past（`sampler.sample(x_past′, …)` 内部两处均吃 x_past′）。**禁止**任何替代实现：干净 past 生成后输出侧加噪、或干净 past 条件 + 仅先验均值扰动等。机器可检断言 = R2（同 ε/η 下 c=0 输出与未扰动基线逐位恒等——恒等性本身证明唯一差异输入是 past 段）。
- 幅度默认 **c=0.5**；敏感性网格 c ∈ {0.1, 0.3, 0.5}（旧库 σ∈[0.01, 0.3]、归档 Run 009 取 0.5 的先例区间）。
- 噪声种子：ξ 逐 (被试, 窗, 源 a, 重复 k) 独立（`np.random.SeedSequence(master).spawn` 派生树，§3）；跨被试不共享（与归档 Run 008 模板跨被试共享的口径差异登记：本设计每窗独立注入，null 逐窗构造无需跨被试模板）。
- 无扰动对照 `P(X, a, 0) = X`（逐位恒等，作为种子/CRN 正确性的机器可检断言）。

### 1.3 null 响应 N 的构造（要素③）

- **定义**：null 响应 = 无扰动条件下的纯 posterior 采样变异性，与响应统计量同构造：
  `N_k[b, w] = (1/n₀)·Σ_{t<n₀} ( Ŷ_k^{(w)}[b, t] − Ŷ_ref^{(w)}[b, t] )²`，
  其中 `n₀ = ⌈0.1·fs⌉`（v4@500 = **50**；fmt@250 = 25）为 future 段前 0.1s 样本数；`Ŷ_k^{(w)}` = 窗 w 第 k 次**独立** posterior 采样（fresh ε_k，无 CRN）；`Ŷ_ref^{(w)}` = 窗 w 的 posterior 参考轨迹（B_ref = **64** 条固定 ε 网格采样的逐点均值）。
- **MSE 维度与聚合**：单值 N 的维度 = (通道 b, 窗 w, 重复 k)；MSE 沿时间轴（前 n₀ 样本）计算，**不跨通道聚合**（逐通道噪声地板不同）；推断用统计量为「k 上中位数 → 窗上中位数 →（被试级矩阵）→ 组级中位数 + 逐被试表」的稳健聚合链（中位数抗重尾，见异常值政策）。
- **重复采样次数（v1.1 依 Run 007 校准 pilot 修订）**：null 采样 K_null = **64**（原 256）；响应侧 K = **32**（归档 n_trials=32 先例）；参考轨迹 B_ref = 64。依据：**Q90 精度只依赖池大小 W×K（逐通道 null 样本总数），不单是 K**——pilot 实测（1 被试 64 窗）CV(Q90) 随池规模 √n 收缩，全窗 W≈1,308/例时 K=64 的池 ≈ 8.4 万/通道 → 外推 CV ≈ 0.26%。候选值表（耗时 = null 阶段 10 例全窗，实测速率 1.04 ms/(窗·条)；精度 = pilot @W=64 实测，全窗外推值括注）：

| K_null | pilot 池/通道 (W=64) | CV(Q90) 中位 @pilot | split-half max\|Δτ\|/τ 中位 @pilot | 全窗 W=1308 外推 CV | null 阶段耗时（10 例全窗） |
|---|---|---|---|---|---|
| 32 | 2,048 | 1.78% | 8.9% | ≈0.40% | ~7 min |
| **64（默认）** | 4,096 | 1.18% | 6.8% | **≈0.26%** | **~15 min** |
| 128 | 8,192 | 0.69% | 4.8% | ≈0.19% | ~30 min |
| 256 | 16,384 | （基准） | 3.6% | ≈0.13% | ~60 min |

  附：缩 W 不缩 K 等价（W 臂实测：池 512/1k/2k/4k → CV 5.54/3.90/2.47/1.21%，同一 √n 律）。**v1.3 用户钉定执行参数：null = K_null 64 × W_null 200 窗/例（池 12,800/通道，外推 CV ≈0.67%，split-half 预计 ~4% < 10% 触发线）；响应 = K 32 × W_pert 50 窗/例（R̄ 聚合样本 1,600/(a,b)）；W_pert ⊂ W_null**。
- **随机种子**：ε 网格（参考）、ε_k（null）、ξ（扰动）全部由主种子派生树显式生成并落 manifest（§3）。
- **异常值处理**：NaN/Inf 出现即**硬失败中止**（确定性调试语境，不静默丢弃）；重尾不剔除、不 Winsorize（默认口径），以中位数聚合 + 分位数阈值吸收；「敏感性臂」登记 trimmed-mean(5%) 对照；退化窗（参考轨迹方差为 0 / 生成幅度 > 10×训练 std）整窗排除并留日志（异常规则 R4/R6）。
- 统计性质（讨论见 §5）：H₀ =「源 a 对靶 b 无模型内传导」时，R[a,b]（与 N 同构造、唯一差别是 a 处 σ_p>0）与 N[b] 同分布——构造对称性给出可交换的 null。

### 1.4 阈值确定（要素④；v1.1 精简版为默认）

**精简版（新默认，三件事）**：
1. `τ_b = np.quantile(N[b], 0.90)`（逐靶通道 b，62 个；ECDF 线性插值 = np.quantile 默认法，单行调用非额外步骤；不做参数拟合）；
2. **split-half 一行输出**：池拆两半各算 Q90，逐通道 `max|Δτ_b|/τ_b` 报中位数（>10% 触发 R3 加密）——零额外采样成本的唯一稳定性证据（Run 007 pilot：默认配置全窗外推 ~1.5%）；
3. **[Q90, Q95) 边际带旗标**（同一 ECDF 多取一个分位点，零成本；`≥ Q95` 记稳健档）。
判定 = **单侧上尾** `G[a,b] = 1[R̄[a,b] > τ_b]`。

**删除项与损失（v1.1 拍板，降级为可选强化臂，默认不做）**：
| 删除 | 损失 | 残余缓解 |
|---|---|---|
| bootstrap 1000 CI/SE | τ_b 无不确定度量化 | split-half 数值即经验稳定性替代（pilot 中两者同量级）；Q95 旗标带承担边际解释 |
| 独立种子全重跑 | 无跨种子可复现证据 | R2 逐位恒等断言 + 种子树确定性单种子登记；采样稳定性已由 split-half 覆盖 |
| null 与 c 无关性检查 | 种子树泄漏无专用探针 | R2 恒等 + ξ/ε 生成器分离设计（泄漏概率低、后果中等，接受） |
| 合并全局 τ 交叉验证 | 逐通道 vs 合并齐性无证据 | 报告逐通道 τ_b 的 CV 一行统计（不进任何门控逻辑） |
| Q95/Q99/BH-FDR 变体矩阵 | 多重比较无校正读出 | 保留 Q95 旗标；正式校正属判读期后处理（材料已在，无需新实验） |

（原 v1.0 全流程文本见归档：本节 git 历史或任务 jsonl；如执行期判读争议大可整体恢复为强化臂。）

### 1.5 扰动—推断完整流程（要素⑤）

**输入**：
- 模型：`outputs/gpfm-mamba-f0/v5-main-01/checkpoints/gpfm-mamba-v5-best.pt`（EMA shadow 装载）+ 同目录 `channel_mean/std.npy` + config.yaml。
- 数据（v1.5 双实例化）：**主 = edra_processed_fmt（θ\* v5-fmt-01）fold0 test 10 例**（v1-fmt 固化 manifest 只读 34/4/10）；过渡 = edra_process_v4（θ\* v5-main-01）fold0 test 10 例（v1 manifest 34/2/10）。确定性 loader shuffle=False；窗 = past 384/future 128/stride 64；**v1.3 钉定抽窗子集：null 200 窗/例、响应 50 窗/例（⊂ null 子集），确定性选择规范见流程表后**（每例窗数以实例化数据为准，子集大小不变）。
- 参考（对照主轨，v1.4）：`outputs/10-19-latent-flow-edra-fmt-train/run_01/gc_oldlib_refs`（= `refs/granger_bivariate_spectral/edrafmt48-cont-zscore-fs250-fixed8-f1-125-nf65/`，48 例原库 GC EC，只读）。跨库对照（v4 模型 × fmt refs）为用户指定口径；无需另算 v4 real refs。

**步骤**（每步输入→输出/关键参数）：

| # | 步骤 | 输入 → 输出 | 关键参数 |
|---|---|---|---|
| A | 装载与校验 | ckpt/manifest/norm → 冻结 θ\*、窗网格、通道序 | EMA shadow；geometry 通道序断言 |
| B | 参考轨迹 | null 窗子集（200/例）各窗 X_past → Ŷ_ref | B_ref=64 固定 ε 网格 |
| C | null 构造 | 200 窗 × 独立采样 → {N_k[b,w]} (62, 200, 64) → τ_b（Q90 点估计 + split-half + Q95 旗标，v1.1 精简版） | K_null=64；W_null=200 |
| D | 扰动生成 | 响应窗子集（50/例，⊂ null 子集）(a, k) 网格：P(X,a,c) → 独立 ε 采样 → R[a,b,w,k] = 前 n₀ 样本 MSE vs Ŷ_ref | c=0.5（网格 {0.1,0.3,0.5}）；K=32；W_pert=50 |
| E | EC_pert × real GC 相关 | S/G 矩阵 + refs（只读 48 例）→ offdiag 3,782 对 Pearson+Spearman | 主口径组级单次（EC_pert 中位数 vs refs 均值）+ 逐被试副口径；direct 主/flip 检查；距离偏 Spearman 随附 |
| F | 阈值判定 | R̄（k、w 中位数）+ τ_b → G[a,b]（Q90 门）/ 边际带 / 超限比 S[a,b]=R̄/τ_b | 单侧上尾；Q95 旗标（v1.1 精简版） |
| G | 推断输出 | **EC_pert**（S/G 矩阵，逐被试+组级）+ × real GC 相关统计（含距离偏判定轨） | §1.1 口径 |
| H | 异常判定 | 全程 → 通过/旗标/中止 | 规则 R1–R6（下） |

**窗子集确定性选择规范（v1.3）**：
- W_null(200)：对每例的被试窗网格（确定性 loader 顺序编号 0..N_w−1），`SeedSequence(master, "win-null", subject_id)` 派生 RNG 无放回均匀抽取 200 窗后**按原序排列**（保持时间顺序，供 GC 轨拼接）。
- W_pert(50)：`SeedSequence(master, "win-pert", subject_id)` 从该例 W_null 集合内再抽 50 窗、按原序排列——**响应窗 ⊂ null 窗**，保证 τ_b 的噪声地板校准在测响应的同一批窗上（窗间 null 异质时不出子集漂移）。
- 两个子集与全部被试 id 落 manifest.json；复算逐位一致（同种子树）。

**异常判定规则**：
- R1（中止）：任何 NaN/Inf（生成、统计、阈值）→ 带上下文中止，不静默丢样本。
- R2（中止）：σ_p=0（或 c=0）重跑与基线**非逐位恒等** → 种子树/装载 bug（perturb-ec-v1 CRN 契约的 r=0 恒等检验移植）。
- R3（旗标→加密）：null split-half `max|Δτ_b|/τ_b > 10%` → K_null 加倍重跑 null。
- R4（排除+日志）：退化窗（参考轨迹零方差 / 生成幅度 >10× 训练 std）。
- R5（旗标不排除）：被试级 EC_pert offdiag 中位数偏离组级 >3 MAD。
- R6（旗标）：S[a,b] > 10（响应超阈 10 倍）→ 登记为超敏边，判读时防个别边主导秩相关。

**可复现实验设置**：
- 种子树（全落 manifest.json）：master=**2026**（读出侧惯例）→ spawn 三支：ε_ref（B_ref 网格）/ ε_null（K_null）/ ξ_pert（(被试,窗,源,重复) 四元组派生）；bootstrap seed=2026；loader 顺序固定（shuffle=False, seed=cfg.seed）；`cudnn.deterministic=True`；env_snapshot 随档。
- 命令形态：`nohup python3 <driver> --stage {null|perturb|report} --run-id ni-mse-v5-01 > .../logs/train.log 2>&1 &`（分期断点续跑；每期产物 npz/jsonl 命名到 (stage, subject, source)）。
- 产物根：`outputs/<任务>/ni-mse-v5-01/`（null/、perturb/、report/、manifest.json）——执行任务立项时定。

**计算量预算（v1.3 参数 + v1.4 砍 GC 轨，依 Run 007 实测速率 1.04 ms/(窗·条)）**：null+ref = 10 例 × 200 窗 × (64+64) = 25.6 万条 ≈ **4.5 min**；扰动响应 = 10 例 × 50 窗 × 62 源 × 32 重复 = 99.2 万条 ≈ **17 min**；refs 相关与距离偏统计为纯 CPU 矩阵运算（秒级）。**全程（含装载/落盘开销）预留 ~30 min 单卡量级**。

## 2. 伪代码（可执行骨架）

```python
# 阶段 C：null
for w in windows:                      # 确定性顺序
    ref = mean(sampler.sample(x_past[w], K=B_ref, eps=eps_grid))      # (62,128)
    for k in range(K_null):
        yk = sampler.sample(x_past[w], K=1, eps=eps_null[k, w])       # 独立 posterior
        N[k, :, w] = ((yk[:, :n0] - ref[:, :n0]) ** 2).mean(axis=1)   # 逐通道、前 0.1s
tau_b = np.quantile(N.reshape(62, -1), 0.90, axis=1)                  # 逐通道 Q90 + bootstrap CI

# 阶段 D：扰动—响应（与 null 同构造，唯一差别 = 源通道加性噪声）
for a in range(62):
    for k in range(K):
        xp = x_past[w].clone(); xp[a] += c * xi[a, k, w]              # 无偏高斯（标准化空间，σ_p ≡ c）
        yp = sampler.sample(xp, K=1, eps=eps_null_independent)        # 独立 ε（可交换 null）
        R[a, :, w, k] = ((yp[:, :n0] - ref[:, :n0]) ** 2).mean(axis=1)
R_bar = median_k(median_w(R))                                          # 稳健聚合
G = R_bar > tau_b;  S = R_bar / tau_b                                  # 门控 + 超限比

# 阶段 E：EC_pert × real GC 相关（refs 只读；无生成侧 GC 重算）
ec_refs_mean = mean over 48 例 refs_s*.npz["ec"]                    # (62,62) 行=源
off = ~np.eye(62, dtype=bool)
ec_pert_group = np.median(np.stack([S_s for s in test10]), axis=0)  # 组级 EC_pert
pearson  = corrcoef(ec_pert_group[off], ec_refs_mean[off])           # direct 主口径（同向无转置）
spearman = spearmanr(ec_pert_group[off], ec_refs_mean[off])
spearman_flip = spearmanr(ec_pert_group[off], ec_refs_mean.T[off])   # 方向性检查副口径
# 距离偏判定轨：residualize 双方于 pairwise_d[off] 后再 Spearman（组级单次；逐被试副口径同式）
```

## 3. 与既有路径的关系

| 维度 | perturb-ec-v1（spec 已沉淀） | latent_flow σ 二分（归档 Run 006–009） | **NI-MSE-50-q0.90（本路径）** |
|---|---|---|---|
| 扰动空间/算子 | 输入层 past 增益 (1+r)·x | 潜空间观测噪声 σ 搜索（bisection） | **信号空间源通道加性零均值高斯 c·ξ（标准化空间）** |
| null/阈值 | τ̃=0.25 相对自响应（双口径 v2） | τ = α·median_b(MSE@σ_max) 共享中位 | **posterior 采样 null 的 Q90，逐靶通道** |
| 响应读出 | 剂量-响应 slope 三件套 | σ\* → 微分熵连接强度；σ=0.5 直读 MSE（Run 009） | **前 0.1s 逐靶 MSE → 超限比 S + 门控 G = EC_pert** |
| EC 对照 | 比较器侧 GC/gPDC 相关 | 原库 GC refs 对照（Run 005） | **\|EC_pert\| × real GC refs 直接相关（Pearson/Spearman + 距离偏判定轨）；不做生成侧 GC 重算** |
| 确定性 | CRN 契约（r=0 恒等） | 确定性抽样模板 | **种子树 + R2 恒等断言；null 侧刻意独立 ε（可交换性）** |

## 4. 判读与验证计划（执行任务时的 AC 候选）

1. R2 恒等断言通过（c=0 逐位恒等）；split-half 稳定性 <10%（登记数值；v1.1 精简版仅此一项稳定性检查）。
2. 阴性对照：c→0.05 时 G 的通过率 → ~10%（Q90 名义水平）；S 分布 → 1 附近。
3. **EC_pert × real GC refs**（§1.1 口径）：组级单次 Pearson/Spearman + 距离偏 Spearman + flip 检查 + 逐被试副口径 mean±std（n=10）。
4. 结构预测（可检验）：v5 的 A_eff 几何主导（Run 006：M 占 98.3%、W 方向信息弱）⇒ 预期 EC_pert 呈强距离衰减、方向性检出弱、与 refs 的相关或大部分由对称距离结构承载（flip≈direct 即其信号）——若方向性显著超出 W 信息量，提示响应非经 A_eff 的其他通路（LN/头部投影），需归因排查。

## 5. 讨论（风险与边界）

- **null 的可交换性是近似**：H₀ 下 R 与 N 同分布依赖「a 处注入不改变 b 的条件分布」这一模型性质；v5 中跨通道通路不止 A_eff（LayerNorm 逐通道独立、head Linear(62→62) 也可载跨通道权重），故 R 度量的是**整模型传导**而非 A_eff 单机制——这正是代理脑 EC 的本意，但与 A_eff 读出（Run 006）解耦判读。
- **逐通道 τ vs 共享 τ**：bisect_sigma 的教训（逐靶阈值绑定自身响应 → 超敏靶反转信号）不适用于本构造——τ_b 绑定的是**噪声地板**（null），与响应幅度无关；反而吸收通道异方差。仍报告合并 τ 与齐性检查作交叉验证。
- **前 0.1s 窗（v1.1 用户拍板：只保留此窗）**：扰动经条件生成最快在 future 前段显现（模型一步动力学内），0.1s 聚焦最快传导；**慢传导（>0.1s）漏检风险经用户 2026-10-20 拍板接受**（原 0.2s/全窗敏感性臂取消）。窗短（50 样本）估计方差大，靠 K×W 中位数聚合。
- **跨库对照的判读边界（v1.4a 修订 / v1.5 计划收窄）**：EC_pert 来自 v4/500 Hz × v5 θ\*（test 10 例），refs 为 fmt/250 Hz 48 例原库 GC——**被试同队列且 test 10 例 10/10 全在 refs 内（逐被试配对可行，见 §1.1 勘误）**；跨范式差异（同人不同处理管线：fs/预处理/时长支撑——EC_pert 自 50 窗 × refs 自全长 60001 样本）是配对内的解释性 caveat；通道几何同源（geometry_edra_fmt.pt 同一 62ch 名序）保证矩阵索引对齐成立。**v1.5：θ\* 切换 v5-fmt-01（同 fmt 管线）后本 caveat 整条消除**——代理脑条件分布与 refs 同数据处理，逐被试配对唯一剩余限制 = 样本量 n=10。
- **相关的承载结构风险（Run 006 教训直接沿用）**：|A_eff| × 同款 refs 的相关为 Pearson +0.44/Spearman +0.25 且几乎全由对称距离核承载——EC_pert 若同样 flip≈direct + 距离偏后相关消失，则其与 real GC 的一致同样是几何结构使然，非方向功能连接；方向性证据须看 direct−flip 差与距离偏后残差相关。
- **Q90 的多重比较**：主读出按用户指定保持 Q90，判读以 Q95 旗标带为稳健参考（v1.1 精简版）。
- **边界**：本设计未执行；θ\* 为「候选」非「合格代理脑」（三项检验另立）；NI-MSE-50-q0.90 属 study2 EC_pert 计算路径，完整路径 = study1-gpfm-mamba-v5 × study2 NI-MSE-50-q0.90（v1.4 定位）。
