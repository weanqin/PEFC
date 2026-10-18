# 实验日志 — 10-18-bivar-gc-fmt-250（edra-fmt 250 Hz 原始全长双变量 GC 真实参照）

> 每次「跑批/训练/评估/统计」追加一个 Run 块：跑前先写「目的+配置」，跑后同会话补「结果+结论」，不留空块。
> 格式契约见 `.trellis/spec/research/experiment-log.md`；产物只落本任务声明的变体目录（`.trellis/spec/research/path-isolation.md`）。
> 口径总源：design.md。产物根：`/root/npi/outputs/bivar-gc-ec-fmt-250/`（新根，与 125 Hz 的 `outputs/bivar-gc-ec-fmt/` 严格隔离）。

## Run smoke-01 — 全链路冒烟 + 条件数实测（2026-10-18 完成；触发条件数暂停门禁，见结论）

### 目的与假设

- v2 全链路（装载→Gram→逐对 BIC→ln(SSR) readout→落盘）在 250 Hz 档可跑通；250 Hz 过采样下 1891 对 Gram 条件数实测（预期高于 v1 的 ~1e5–1e7 带，观测是否触发 halt 门限）。

### 配置

- 代码：`models/bivar-gc-ec-v2/`（首版）+ 只读复用 `models/bivar-gc-ec-v1/bivar_gc.py`
- 参数：受试者 = EXPECTED_IDS 前 2 例；fs=250 原始全长（60001 点，不降采、不切窗）；P_MAX=48；OLS
- 环境：本机（environment.md 契约内；CPU-only、BLAS 五变量护栏）
- 数据：`/root/autodl-tmp/edra_processed_fmt/`（只读）
- 命令：`nohup python3 models/bivar-gc-ec-v2/run.py --stage smoke --run-id smoke-01 &`

### 结果

- 2 例（s1 MDD / s4 HC）总耗时 49 s（2 进程并行，逐例拟合 34 s 含条件数登记）；前置：v2 单测 8 项 31 断言全绿（73.4 s，rng 2026）。
- **GC 量级人工核对通过**：offdiag min/median/max = **0.0017 / 0.0159 / 0.1808**（ln 单位；与 v1 refs-01 的 0.0007/0.0153/0.4356 同量级，median 几乎一致）；断言全过（全有限 / 对角=0 / offdiag≥0 / 阶数对称）。
- **条件数实测（1891 对/例，gram_pair_cond @p=48）**：s1 min/median/max = 2.1e13 / 7.4e13 / **5.0e15**；s4 = 2.0e13 / 8.3e13 / 3.1e15——高于 v1 实测带 ~1e5–1e7 达 **6–9 个量级，越过 >1e10 halt 门限 ⇒ 触发预注册暂停门禁（design §2.2/§5-4）**；最差对 slice 特征谱 min eig=2.5e-10 > 0（非奇异，solve 未报错）。
- **数值稳健性诊断（只读 console，不改变口径、不落产物）**：fast Gram-slice vs naive `lstsq`（SVD 稳定路径）同对同阶对比（s1 最差/中位/最优/近邻 4 对 + 8 通道子集 28 对）：GC 读数绝对差 ≤ 2.7e-4（= offdiag min 0.0017 的 ~16%、median 0.0159 的 ~1.7%）；SSR 相对差 ≤ 2.8e-4；BIC 相对差 ≤ 8.2e-4（p=47/48 间 BIC 差同量级——BIC 曲线顶部平坦，47/48 之分为数值噪声级；最差对 cond=5.0e15 处 fast p\*=47 vs naive p\*=48 即此机制）。
- **阶数分布（如实登记，非 100% 顶格）**：p\*=48 占 3607/3782（95.4%）、p\*=47 占 175/3782（s4 174 + s1 1）——结合诊断，47/48 边界属数值噪声级分界（顶格预期大体成立但非严格 100%）。
- 产物：`outputs/bivar-gc-ec-fmt-250/smoke-01/{refs_all.npz (2,62,62) f32, refs/refs_s{1,4}.npz, data/prep_meta.json, refs_summary.json（含 condition_numbers per_subject）, config_snapshot.json, logs/}`。
- 代码登记（run 后修正，仅报告层）：refs_summary 的 pooled 阶数分布原沿用 v1 跨例求和式（键 95/96，混合对语义失真），已改 `_order_hist_stacked` 堆叠计数（键=真实 p\*）并从已落盘逐例 npz 重算该字段——**数值管线零改动**，逐例 npz 与首跑逐位一致。

### 结论

- v2 链路在 250 Hz 档跑通，GC 量级与 v1 同量级，实现正确性断言全过；**但条件数 1e13–5e15 ≫ 1e10 门限、显著越出 v1 带 → 按预注册门禁暂停请示**，未启用任何正则化/对角加载（遵守「禁止自行切换」）。诊断证据（vs SVD 稳定路径 GC 差 ≤3e-4、全部硬断言过、Gram 非奇异）表明 OLS 读数在本条件数下实际稳健，唯 p\* 顶部 47/48 之分为数值噪声级（影响 175/3782 对的阶数登记精度，不影响 GC 读数量级）。是否按原口径继续 refs-01 由用户拍板。

### 门禁处置（2026-10-18，主会话代决策）

- **拍板：按原 OLS 口径继续**——不启用任何正则化/对角加载、不改 v1、不变 P_MAX=48；用户未实时应答，按自主处置原则执行**可逆选项**（产物按 run_id 落盘，弃用重跑零成本）。
- 依据：①fast↔naive（SVD 稳定路径）输出级一致——GC 绝对差 ≤2.7e-4（median GC 的 ~1.7%）；②实现正确性硬断言（非负/对角零/有限/SSR 单调）全过；③Gram 非奇异（min eig=2.5e-10 > 0）；④250 Hz 相邻滞后共线性升高系设计已预期方向（design §2.2），实测分布已登记。
- **升级措施（诊断层，不改数值管线）**：refs-01 / rev-01 逐例抽样 50 对、sh-01 每半 25 对（rng 2026 确定性）做 fast↔naive 交叉校验（fast p\* 单阶，naive 行集 row_from=P_MAX 显式对齐），GC 绝对差 max 与 SSR 相对差 max 落入各 Run summary 键 `fast_naive_check`——把 smoke 级证据升级为逐 Run 证据。
- p\* 顶部 47/48 之分为数值噪声级（BIC 顶部平坦 + 条件数 1e13–5e15），caliber 注记如实说明，不调参干预。

## Run refs-01 — 主 Run：48 例 250 Hz 全长参照（2026-10-18 完成）

### 目的与假设

- 交付 48 例受试者级 62×62 双变量 GC EC 矩阵（行=源/列=靶/对角=0）+ 逐对 BIC 阶数分布；预期 p\*=48 顶格（如实登记）。

### 配置

- 代码：`models/bivar-gc-ec-v2/` + 只读复用 v1 `bivar_gc.py`
- 参数：48 例全量；row_from=P_MAX=48（n_eff=59953/例）；逐对 BIC p∈{1..48}；ln(SSR_r/SSR_u) 双向；OLS 非负断言；16 进程并行
- 交叉校验（门禁处置升级措施）：逐例随机抽 50 对（rng 2026，确定性）fast↔naive（SVD lstsq）@p\* 读出比对，GC 绝对差 max / SSR 相对差 max 入 refs_summary.json 键 `fast_naive_check`
- 环境：本机（同上）；nohup 长跑
- 数据：同 smoke-01（48 例全量）
- 命令：`nohup python3 models/bivar-gc-ec-v2/run.py --stage main --run-id refs-01 &`

### 结果

- 48/48 例（HC 24 / MDD 24），wall **418 s**（16 进程 spawn、worker 单线程 BLAS；逐例 94–157 s 含 50 对 fast↔naive 交叉校验）。
- **refs_all.npz**：bivar_gc (48,62,62) f32（行=源/列=靶/对角=0 ✓）；offdiag **min=0.0010 / median=0.0189 / max=0.4103**（ln 单位；与 v1 125 Hz refs-01 的 0.0007/0.0153/0.4356 同量级带）。
- **阶数分布（90768 对·例 pooled，如实登记）**：p\*=48 占 87856（**96.8%**）、p\*=47 占 2898（3.2%）、更低阶 14（p\*=44:1、p\*=46:13，<0.1%）——非 100% 顶格；caliber 注记：BIC 顶部平坦 + Gram 条件数 1e13–5e15 下顶部两阶之分与零星散布均为数值噪声级，实效读出 ≈ 固定阶 VAR(48) 双变量 GC；不调参干预。
- **fast_naive_check（逐例 50 对，rng 2026）**：GC 绝对差 max = **8.4e-4**（per-subj 2.2e-7 – 8.4e-4；对照 smoke 级全曲线 ≤3e-4，读出级一致性成立）；SSR 相对差 max = 0.778 系近零残差非对角元分母伪影（定位 s20 pair(3,7)@p\*=48：|naive|=1.3e-5 vs diag ~0.03，两侧相关 -0.0005 vs -0.0001、相关尺度差 ≤4e-4、GC 差 8.6e-5）——解释注记入 summary 键。
- **确定性基础登记**：全部 worker 显式 1 线程 BLAS；1 线程同码径重算 s1 与 smoke-01 产物**逐位一致**（跨 Run 复现基础成立）；对照实验：8 线程 vs 1 线程同例 GC 差 ~2.4e-4 = 条件数放大下的固有数值散度带（登记为读出不确定度量级）。
- 断言全过：全有限 / 对角=0 / offdiag≥0（48 例 × 1891 对）/ 阶数对称（run.py 主流程逐例复核）。
- 图表（B5，经 common/visualizer 正典路由，纯复用零改动）：`figs/bivar_gc_group_means.png`（ec_matrix 三面板：HC n=24 / MDD n=24 / 全体 n=48 组均，Blues 非负、对角掩膜；组均量级 median=0.027 / max=0.064，目检通过）；`figs/bivar_gc_subjects_sample.png`（ec_heatmap_grid 等距 6 例，note 注明 250 Hz raw full-length · p\*=48 占 96.8% 顶部噪声级分布）。**阶数分布图裁量**：分布非单点但集中于顶部（96.8% p\*=48 + 3.2% p\*=47 + <0.1% 散布），沿 v1 先例以 refs_summary.json 登记呈现，不为分布图新增 visualizer 图型（共享文件零改动）。
- 产物：`outputs/bivar-gc-ec-fmt-250/refs-01/{refs_all.npz, refs/refs_s*.npz ×48, data/prep_meta.json, refs_summary.json（fast_naive_check + 条件数注记引 smoke-01）, config_snapshot.json, logs/}`。

### 结论

- **refs-01 交付成立**：48 例 250 Hz 原始全长双变量 GC 参照落盘，量级与 v1 同带，实现正确性断言全过；逐 Run fast↔naive 证据（GC ≤8.4e-4）支持门禁处置（按原 OLS 口径继续）的稳健性；无 smoke 未见的新异常（无 NaN / 无断言失败 / 无大规模低阶散布——44/46 仅 14 对·例）。p\* 分布如实登记（96.8% 顶格 + 数值噪声级尾部），grid 扩展仍属 v1 遗留未决决策。

## Run refs-01-reprod — 确定性抽查（AC6，2026-10-18 完成）

### 目的与假设

- 同输入重跑逐位一致（确定性契约）。

### 配置

- 与 refs-01 完全同配置（含逐例 50 对 fast↔naive 交叉校验）；scratch run_id，产物仅用于逐字节比对 per-subject npz。

### 结果

- 重跑完成：wall 403 s（vs refs-01 418 s，同 16 进程）。
- **逐字节比对全过（最强档）**：48 个 per-subject `refs_s*.npz` **raw bytes 全部逐字节相同**（数组级 array_equal 自然全过：ec/orders/n/subject_id 全键）；`refs_all.npz` raw bytes 逐字节相同；summary 除 wall 计时字段外全等，fast_naive_check 数值（pooled gc_absdiff_max=8.4e-4 / ssr_reldiff_max=0.7777）亦逐位复现。
- 基础事实链：worker 显式 1 线程 BLAS ⇒ 跨 Run 逐位复现（对照登记：BLAS 线程数 1→8 改变归约序，同例 GC 差 ~2.4e-4，为条件数放大下的固有散度带，不进产物口径）。

### 结论

- **AC6 判定通过（最强档：raw bytes 级）**——同输入重跑输出逐位一致；refs-01 产物为确定性可复现交付。

## Run rev-01 — 时间反转检验（2026-10-18 完成）

### 目的与假设

- GC 方向读数的时间序敏感性指纹：翻转时间轴后同口径重算，预登记 fwd/rev 关系（非硬门限，判读框架见 design §2.4）。
- 预期（增量复审校准后措辞）：fwd/rev 边级读数显著相关（共享对称二阶耦合成分）但方向内容不同（单侧 VAR 反转系数改变）；不对称度的**边级分布将重排**，其组级标量均值可能同量级保持——以实测如实登记；若与预期方向不符亦如实登记，不影响 refs-01 交付有效性。

### 配置

- 代码：`models/bivar-gc-ec-v2/run.py --stage reverse`（装载链复用主链 + `X_rev = X_z[:, ::-1]`）
- 参数：与 refs-01 逐字同口径（不降采、单段全长、row_from=48、逐对 BIC p∈{1..48}、ln(SSR) 双向、OLS）；fwd 基准 = refs-01
- 交叉校验（门禁处置升级措施）：同 refs-01（逐例 50 对 fast↔naive @p\*，rng 2026），入 rev-01 refs_summary.json 键 `fast_naive_check`
- 命令：`nohup python3 models/bivar-gc-ec-v2/run.py --stage reverse --run-id rev-01 &`

### 结果

- 48/48 例，wall **358 s**（16 进程；逐例同管线）；GC_rev 落盘 `refs_all.npz`（含 `reversed=True` 标记）+ 逐例 `refs_s*.npz`；非负断言全过（offdiag min=0.0009）。
- **四组预登记指标（组级 mean±sd，ddof=1）**：
  1. `edge_fwd_rev`（ρ(offdiag GC_fwd, GC_rev)）= **0.6608 ± 0.2014**（逐例 0.16–0.95）；
  2. `edge_fwd_mirrorrev`（ρ(GC_fwd, GC_revᵀ)）= **0.3236 ± 0.1323**；
  3. 幅值：median offdiag fwd = 0.0220±0.0103、rev = 0.0218±0.0099，**比值 rev/fwd = 0.9968 ± 0.0569**；
  4. 方向不对称度 mean|GC(A→B)−GC(B→A)|：fwd = **0.0191**±0.0120、rev = **0.0191**±0.0119（组级标量同量级保持，与预登记预期一致；边级分布重排体现于 ρ 而非该标量）。
- rev 侧 offdiag min/median/max = 0.0009 / 0.0190 / 0.4269（与 fwd 0.0010/0.0189/0.4103 同带）；阶数分布同顶部模式（p\*=48 占 87837/90768=96.8%、47 占 2921、更低阶 10）。
- fast_naive_check：GC 绝对差 max = **1.08e-3**（与 refs-01 的 8.4e-4 同带）；SSR 相对差 max = 1.39 同为近零非对角元分母伪影（注记同 refs-01）。

### 结论

- **按预登记框架判读（描述性指纹，无硬门限）**：GC 方向读数**含实质时间方向成分**——ρ(fwd,rev)=0.66 显著低于 1（若读数被对称滞后耦合主导则应 ≈1），且**不满足镜像对应假设**（ρ(fwd,revᵀ)=0.32 更低：反转过程是另一个 VAR，方向读数并非简单方向互换）；同时幅值尺度（比值 1.00）与不对称度组级标量（0.0191 双侧）在反转下保持——与「时间反演二阶结构不变、反向 VAR 系数不同」的理论预期吻合。判读：本口径 GC 读数对时间序敏感、方向成分非伪影主导，但对称成分仍贡献约 1/3–2/3 的边级方差（逐例异质 0.16–0.95），EC 推断线引用本参照时应知悉该指纹；refs-01 交付有效性不受影响。

### 对比交付（用户 2026-10-18 追加）

- 代码：`models/bivar-gc-ec-v2/compare_rev.py`（独立 CLI：读 refs-01 / rev-01 两端 `refs_all.npz` **纯计算**，不重跑拟合、无 RNG、summary 无时间戳/计时字段；复用 v2 config 唯一名装载配方与 data_io 通用断言——装载态复核形状/EXPECTED_IDS 对齐/reversed 标记/有限/对角零/offdiag≥0 + manifest groups 与两端 refs_summary 一致）。产物：`rev-01/compare_summary.json` + `rev-01/figs/compare_group_means_fwd_rev_revT.png`（ec_matrix 三联 GC̄_fwd | GC̄_rev | GC̄_revᵀ，既有图型复用）+ `rev-01/figs/compare_edges_fwd_vs_rev_revT.png`（**新图型 edge_scatter**：边级原始取值散点 + 恒等线；visualizer 纯增量扩展，白名单理由：qq_compare 为分位数级散点、gc_matrix 散点绑定条件 GC 热图面板，均不覆盖「边级原始读数 + 恒等线」图型）。refs-01 与 rev-01 既有产物只读未动。
- **组级（GC̄ 组均值矩阵，3782 有向边）**：ρ(GC̄_fwd, GC̄_rev) = **0.9029**（Pearson r = 0.9300）；ρ(GC̄_fwd, GC̄_revᵀ) = **0.1852**（r = 0.2330）——组均板上同位高度一致、镜像对应假设同样不成立；GC̄ 幅值同带（offdiag median fwd 0.0275 / rev 0.0274）。
- **Top-50 边重叠**：fwd ∩ rev = **34/50**（Jaccard 0.5152）；fwd ∩ revᵀ = **4/50**（Jaccard 0.0417）。
- **方向翻转刻画（fwd Top-50 边 A→B）**：rev **同位**均值 **0.0487** vs **镜像位**均值 **0.0357**（fwd 自身 0.0481）——强边反转后保持在原方向读数上、非方向互换。
- **通道级（组均值出/入强度 = offdiag 行和/列和，62 通道）**：出-出 ρ = **0.9335**、入-入 ρ = **0.9441**、出-入 ρ = **0.0755**——源的出强度/靶的入强度角色在反转下保持、不互换（逐例同构：out-out 0.6888±0.2289、out-in 0.3469±0.2474、in-in 0.6140±0.2886，mean±sd ddof=1）。
- **逐例交叉验证（断言级）**：本交付以 f32 落盘态重算 ρ(fwd_s, rev_s)，与 rev-01 refs_summary 已登记 edge_fwd_rev（f64 计算态）逐例比对 **max|Δ| = 1.69e-7**（allclose rtol=1e-6/atol=1e-6 全过，偏差属 f32 量化带），组级 mean±sd 复现 0.6608±0.2014。
- **确定性验证**：同输入重跑 compare_rev.py 两次，compare_summary.json + 两图 **raw bytes 逐位一致**（sha256 校验全过）。
- **结论**：组级与结构级指纹与 rev-01 逐例判读一致并进一步确认——fwd/rev 边级读数高度相关（组均板 ρ=0.90、Top-50 重叠 68%）但**方向内容真实存在**：镜像对应（revᵀ）在各层面均显著更弱（ρ=0.19、重叠 8%、出-入 ρ≈0.08），强边反转后保持原方向而非互换、通道出入角色不互换。按预登记框架（描述性、非硬门限、ρ 与幅值联合判读）：本参照的 GC 方向读数对时间序敏感、非对称伪影主导，EC 推断线引用时应知悉该指纹；refs-01 交付有效性不受影响。

## Run sh-01 — split-half 噪声天花板（2026-10-18 完成）

### 目的与假设

- 参照自身质量登记：奇偶分半两半独立同口径 GC 的 offdiag Spearman ρ → 天花板 mean±sd（48 例）。

### 配置

- 代码：`models/bivar-gc-ec-v2/run.py --stage split`（**v2 本地编排**：`build_halves(x, seg=500)` + `half_ec(·, p_max=48)`，不调 v1 `ceiling_for_subject`——其无 seg 参数不透传，审查发现 1）
- 参数：段 500 样本（2 s @250 Hz）；运行时口径锁 n_seg==120 / 每半 27120 行；每半独立逐对 BIC（P_MAX=48）
- 交叉校验（门禁处置升级措施）：每半各抽 25 对（同一 rng 2026 抽样）fast↔naive @p\*，入 split_half.json 键 `fast_naive_check`
- 命令：`nohup python3 models/bivar-gc-ec-v2/run.py --stage split --run-id sh-01 &`

### 结果

- 48/48 例，wall **276 s**（16 进程；逐例 87–100 s 两半 × 全口径含逐对 BIC）；**运行时口径锁全过**（逐例断言 n_seg==120、每半 27120 行；`seg_samples=500` 落 summary）。
- **split-half 天花板 = 0.8981 ± 0.1414**（mean±sd，ddof=1，48 例；逐例 ρ 0.379–0.995，中位 0.9561）。低尾由少数例下拉（s28=0.379、s8=0.422、s21=0.581）；与 v1 125 Hz 天花板 0.9243±0.0447 同量级带（250 Hz 档 sd 更大，逐例如实登记，不做跨口径判读——R4 不做对比交付）。
- fast_naive_check（每半 25 对）：GC 绝对差 max = **1.50e-3**（与 refs-01 的 8.4e-4 同带，行数减半 + 分段路径下略宽）；SSR 相对差 max = 0.188 同为近零非对角元分母伪影类（注记同 refs-01）。
- 产物：`outputs/bivar-gc-ec-fmt-250/sh-01/{split_half.json（ceiling per_subject + fast_naive_check + 口径锁字段）, config_snapshot.json, logs/}`。

### 结论

- **AC4 判定通过**：250 Hz 档参照自身噪声天花板 0.8981±0.1414 落盘；天花板量级与 125 Hz 档一致（同 0.9 量级、独立口径各自登记），支撑本参照作为 EC 推断线可用观测参考的质量登记；v2 本地编排（显式 seg=500 + 运行时口径锁）有效防住 v1 缺省 250 样本段静默漂移面。

## Run trgc-01 — TRGC 净时序方向分析（2026-10-18，用户追加）

### 目的与假设

- 从已落盘数组直接计算 `TRGC_s = GC_fwd_s − GC_rev_s`（逐例逐边），做两件事：①组级拓扑是否符合生理常识——闭眼静息态枕/顶枕 → 额叶沿前后轴的方向偏置，并对照「高信噪比通道主导」的 SNR 假象控制；②受试者间一致性生存分析——逐边单样本 t 检验 + BH-FDR + bootstrap CI，数存活的「净时序方向显著边」。
- 预期（登记，非硬门限）：若本口径方向读数含生理方向成分（rev-01 指纹的延伸），后→前 边群 TRGC 应系统高于 前→后；SNR 对照若 ρ 显著为正则提示高幅通道系统性当源、拓扑判读须打折。
- 判读框架（预登记）：TRGC = 净时序方向**读数**（fwd−rev 差分），显著边 ≠ 生理因果确证——双变量 GC 口径 + 头皮体积传导局限如实注明（沿 rev-01 / compare_rev 预登记框架延伸）。

### 配置

- 代码：`models/bivar-gc-ec-v2/trgc.py`（独立 CLI，读已落盘数组纯计算，不重跑拟合）；统计函数只读复用 `models/gpfm-mamba-v1/gc_psd.py` 的 `t_two_sided_p` / `bh_fdr`（路径登记 + 归属断言；v1 零改动）
- 输入（只读）：`refs-01/refs_all.npz`（GC_fwd）+ `rev-01/refs_all.npz`（GC_rev，reversed=True）；通道名取数据集 npz `channel_names`；前后轴 = `chan_coords_62.npy` y 轴（先看坐标结构后定：z 为竖直、x 左右，y 即前后轴，与面内主成分之一对照 max(|r_PC1|,|r_PC2|)≥0.95 断言——实测 y 与 PC2 高度一致 r=+0.9831、PC1=−0.181 属左右轴；+y=前 由 10-20 标签梯度 Fpz>Cz>Oz 实证；选择登记入 summary `axis_registration`）
- 参数：48 例；TRGC>0 = 净前向 / TRGC<0 = 净反向；t 检验 df=47、BH-FDR q<0.05；bootstrap 受试者重采样 2000 次 percentile CI 95%（rng 2026）；拓扑 = y 轴 rank 三等分 ROI（post 21 / mid 21 / ant 20），后→前 vs 前→后 边群均值差 + 符号翻转置换 5000 次（rng 2026，双侧 p=(1+#{|D_perm|≥|D_obs|})/(n_perm+1)）；边级 Spearman(TRGC̄, 轴向位移 disp=y_tgt−y_src)；Top-30 |TRGC̄| 边表；SNR 对照 = 逐通道原尺度 std（z-score 前，ddof=0，48 例平均）vs 组级 TRGC 出强度 Spearman
- 图：visualizer 纯增量三图型（`signed_matrix` / `topo_arrows` / `volcano_edges`，白名单理由见结果节）；产物落 `outputs/bivar-gc-ec-fmt-250/trgc-01/`
- 环境：本机（environment.md 契约内；CPU-only、BLAS 五变量护栏）
- 命令：`python3 models/bivar-gc-ec-v2/trgc.py --fwd-run refs-01 --rev-run rev-01 --run-id trgc-01`

### 结果

- 单进程纯计算跑通；装载断言全过（两端 refs_all 装载契约 / reversed 标记 / manifest groups 一致 / 轴登记断言 Fpz>Cz>Oz / 三等分 21/21/20）。产物 5 件：`trgc-01/trgc_summary.json` + `data/edges_table.npz`（逐边 3782 条全量表）+ `figs/` 三图。
- **确定性验证**：同输入重跑两遍，5 件产物（summary/npz/3 图）**sha256 逐位一致**；bootstrap 与置换各自独立 default_rng(2026)、einsum 不经 BLAS、np.savez 定长 zip 时间戳。
- **TRGC̄ 量级**：offdiag min/median/max = **−0.0102 / +7.1e-5 / +0.0108**（组均值；中位仅为 GC̄ offdiag median 0.027 的 ~0.3%）；逐边效应量 dz（mean/sd）median=0.005、|dz|max=0.54；组均值板上净前向 1940 边 / 净反向 1842 边（近对称）。
- **生存分析（两口径两方向）**：口径① t 检验（df=47）+ BH-FDR：**q<0.05 存活 0（净前向 0 / 净反向 0）**——max|t|=3.73 → p_min=5.1e-4 → **q_min=0.997**，3782 重测下无一存活；口径② bootstrap（受试者重采样 2000 次，percentile CI 95%）：**CI 不含 0 = 146（净前向 64 / 净反向 82）**——与全局零假设下纯机会水平（≈3782×5%≈189）同量级甚至更低；**两法交集 = 0**。
- **拓扑（前后轴）**：后→前 边群均值 2.3e-6 vs 前→后 3.4e-4，差 **−3.4e-4**，符号翻转置换（5000 次，seed 2026）**p=0.7437**（检验修正后值，见下方注记）——无前后方向偏置（数值上 ant→post 略高但与零假设完全一致）；边级 Spearman(TRGC̄, 轴向位移) **ρ=+0.0597（p=2.4e-4）**——方向上与「后→前」一致但效应量可忽略（ρ²≈0.4% 方差，且 3782 边共享端点非独立、p 不具独立推断效力）。
- **Top-30 |TRGC̄| 边（前 5）**：P5→P7 +0.0108 / Oz→POz −0.0102 / O2→Oz −0.0101 / P6→P8 +0.0096 / Oz→PO4 −0.0090——全为顶叶-枕叶内部短程边、方向混杂（完整 30 条边表含通道名/坐标/ROI 归属/方向，入 trgc_summary.json `topology.top_edges`）。
- **SNR 对照**：**ρ(逐通道原尺度 std（z-score 前，48 例平均）, TRGC̄ 出强度) = −0.0013（p=0.99）**——高幅通道不系统性当源；辅助：std 沿前后轴梯度 ρ=−0.414（p=8.2e-4，枕部幅值更高，与闭眼枕 α 一致）——幅值梯度存在但未转化为方向偏置，拓扑零结果并非被 SNR 假象掩盖。
- **图与 visualizer 白名单（三图型纯增量，2026-10-18）**：①`signed_matrix`（trgc_group_mean.png）——白名单理由：ec_matrix 展示契约为非负幅值（clip≥0 + Blues vmin=0，docstring 明示），无法承载带符号 TRGC̄（负值=净反向读数会被静默截断为 0），故派发指令原提「复用 ec_matrix」因契约冲突改为增量图型，特此登记；②`topo_arrows`（trgc_top30_topo.png）——既有图型无「电极坐标 + 有向边箭头」空间拓扑图型（ec_matrix/gc_matrix/spatial_corr 均热图、edge_scatter 为散点）；③`volcano_edges`（trgc_effect_q.png）——edge_scatter 轴域锁定 [0,lim] + 恒等线（非负读数对语义），不覆盖「带符号效应量 vs −log10(q)」。
- **检验修正（trellis-check 2026-10-18 发现并就地修复）**：置换统计量原实现漏除 n——置换侧 einsum 输出为逐例求和尺度、观测侧为组均值尺度，相差 48 倍，|D_perm|≥|D_obs| 判定阈值等效缩小 48 倍、p 被系统性推高（原登记 p=0.9948 即此伪高值）。已修正为与观测侧同式（`Gp/n_subj`，trgc.py 注释同步登记）并重跑：置换 p **0.9948 → 0.7437**（判读不变：p≫0.05 零假设一致；自洽核对 SE(D)=7.0e-3/√48≈1.0e-3，d_obs=−3.4e-4 即 z≈−0.34）。该值仅入 summary `topology.perm_p_two_sided`，`edges_table.npz` 与三图不含此值（重跑后 sha256 逐位不变）；修复后同输入重跑两遍，5 件产物 sha256 逐位一致（确定性复核随修复重做）。上文本小节原 p=0.9948 处已同步更正。

### 结论

- **生存分析两口径一致判读：本参照中不存在受试者间一致的「净时序方向显著边」**——t+BH-FDR 下 3782 边无一存活（q_min=0.997）；bootstrap CI 口径 146 条与纯机会水平同量级（≤189 期望）、两法交集 0。逐例逐边的净方向读数（fwd−rev）被受试者间噪声主导：组级 GC̄ 板上的时间序敏感指纹（rev-01 / compare_rev：ρ(fwd,rev)=0.66–0.90 + 镜像不对称）**不能上推为受试者间一致的有向边**。
- **拓扑判读：不符合「闭眼静息态枕/顶枕 → 额叶前后方向偏置」的生理常识预期**——ROI 块置换 p=0.744（检验修正后值，零假设一致；原实现含 48 倍尺度失配伪高值 p=0.995，见结果节修正注记）、边级轴向梯度 ρ=0.06 可忽略；SNR 对照通过（ρ≈−0.00），零结果非高幅通道假象。存在极弱的前向梯度迹象（ρ=+0.06）但不构成可用的方向读数。
- **判读沿预登记框架（如实注明）**：TRGC = 净时序方向读数（fwd−rev 差分），显著边 ≠ 生理因果确证——双变量 GC 口径 + 头皮体积传导局限；本 Run 为如实的阴性登记：EC 推断线引用本参照时方向性表述应以 GC 读数本身（及其 rev-01 指纹）为界，不应引用 TRGC 净差分作为受试者级有向边证据。refs-01 / rev-01 既有交付有效性不受影响。
- 下一步（派生决策项，登记非执行）：若需受试者级方向证据，可另立任务考察更长记录 / 分段聚合降低 TRGC 方差，或改用组水平板统计（compare_rev 已有）而非逐边生存口径。

## Run diag-01 — TRGC 分半信度 + 净出度-通道属性相关（两问合一诊断，2026-10-18，用户追加）

### 目的与假设

- **问 A**：单受试者内部 TRGC（= GC_fwd − GC_rev）的奇偶分半重测信度——trgc-01 已证 TRGC 逐边读数被受试者间噪声主导，本问补登记其**单受试者内部**噪声底：把 fwd 与 rev 各自在奇/偶两半上重拟合（逐例 4 次矩阵拟合），逐例 Spearman ρ(offdiag TRGC_A, offdiag TRGC_B) → 组级 mean±sd。预期（登记非门限）：TRGC 分半 ρ 低于 fwd 分半天花板（0.8981±0.1414 带）——差分放大相对噪声；实测如实登记。对比读数三者并列：fwd 分半 vs TRGC 分半 vs rev 分半。
- **问 B**：GC_fwd 净出度（出−入）被什么单通道属性预测——Spearman(净出度， lag-1 ACF / 滞后 1..48 窗口 ACF / 原尺度幅值)，组级 + 逐例分布；附 GC_fwd 出强度（不减入）vs 幅值作对照（预期复现 trgc-01 SNR 对照的 ≈0 带值）。诊断性登记（预登记框架沿 trgc-01），不做因果表述。
- **内置交叉验证（确定性互证）**：问 A 的 fwd 分半逐例 ρ 与 sh-01 已登记 ceiling.per_subject 断言 allclose（同码径同 worker 契约，应逐位一致）；问 B 的通道 std 与 TRGC̄ 出强度分别对照 trgc-01 edges_table.npz `chan_std` 与 `snr_control.spearman_out.rho`。

### 配置

- 代码：`models/bivar-gc-ec-v2/diag_posthoc.py`（独立 CLI 新文件；v1/v2 与全部既有产物只读零改动）
- 输入（只读）：数据集 `/root/autodl-tmp/edra_processed_fmt/` 48 例（问 A 重拟合 + 问 B 自相关/幅值）；`refs-01/refs_all.npz`（净出度/出强度）+ `rev-01/refs_all.npz`（TRGC̄ 出强度对照）+ `sh-01/split_half.json`（交叉验证靶）+ `trgc-01/{trgc_summary.json, data/edges_table.npz}`（对照靶）
- 问 A 口径：数据切分与 sh-01 逐字同式（`split_half.build_halves(x, seg=500)` 奇偶分半/尾段丢弃/段内设计不跨段；运行时口径锁 n_seg==120、每半 27120 行）；rev 半侧与 rev-01 同式（`X_rev = X_z[:, ::-1]` 后再 build_halves）；每半独立同口径（`half_ec(p_max=48)`，含逐对 BIC）；逐例 ρ_fwd / ρ_rev / ρ_TRGC（TRGC_A = ec_fwdA − ec_revA，TRGC_B 同构）→ 组级 mean±sd（ddof=1）
- 问 B 口径：净出度 = offdiag 行和−列和（对角=0，62 维；组均值矩阵 + 逐例）；自相关 = z-score 后全记录逐通道 lag-1 与滞后 1..48（=192 ms，P_MAX 物理窗）均值 ACF（有偏分母 T），逐例算后取组均值；幅值 = z-score 前原尺度逐通道 std（ddof=0，trgc-01 SNR 对照同式）组均值
- 环境：本机（environment.md 契约内；CPU-only、BLAS 五变量护栏、16 进程 spawn + worker 单线程）
- 命令：`nohup python3 models/bivar-gc-ec-v2/diag_posthoc.py --run-id diag-01 &`（确定性验证：重跑入 scratch `diag-01-reprod`，summary+npz sha256 逐位比对）
- 图：**仅落 json 不出图**——净出度 vs ACF 散点若出图须新增图型（edge_scatter 展示契约为非负轴域 [0,lim] + 恒等线，净出度为带符号读数，语义冲突），诊断性登记数字全量落 diag_summary.json + diag_table.npz，共享文件 common/visualizer.py 零改动（白名单纪律裁量，理由登记）

### 结果

- 三遍执行：pass1 → `diag-01`（wall 346 s）；pass2 → scratch `diag-01-reprod`（363 s，sha256 比对）；pass3 → **同一命令重跑入 `diag-01`**（358 s，最强档逐位比对）。16 进程 spawn、worker 单线程 BLAS；问 B 主进程纯计算。
- **内置交叉验证全过（逐位档）**：①ρ_fwd vs sh-01 `ceiling.per_subject` max|Δ| = **0.0**（allclose rtol=atol=1e-12；组级逐位复现 0.8980781277222123 ± 0.14143963105413276）；②通道 std vs trgc-01 `edges_table.npz::chan_std` max|Δ| = **0.0**；③TRGC̄ 出强度~幅值 ρ = **−0.0012842789** 与 trgc-01 `snr_control.spearman_out.rho` 逐位一致（|Δ|=0）。
- **问 A（三者并列，组级 mean±sd ddof=1；逐例范围）**：
  - fwd 分半 = **0.8981 ± 0.1414**（0.379–0.995；= sh-01 天花板逐位复现）；
  - rev 分半 = **0.8985 ± 0.1550**（0.215–0.995；低尾 s28=0.2145、s8=0.2888）；
  - **TRGC 分半 = 0.7079 ± 0.3074**（**−0.557–0.973**；负尾 s8=−0.5567、s25=−0.1109、s24=−0.0139）——比 fwd/rev 低约 0.19、sd 翻倍、出现负值例（净方向读数在两半间翻号）。
- **问 B（组级 Spearman，n=62 通道）**：净出度~lag-1 ACF **ρ=+0.1238**（p=0.34）；净出度~窗口 ACF（1..48 lag=192 ms）**ρ=−0.2136**（p=0.096）；净出度~幅值 **ρ=+0.1996**（p=0.12）——无一显著。逐例分布（mean±sd）：+0.0380±0.2567 / +0.0071±0.2539 / +0.1032±0.1968（均近零、离散宽）。组级向量：lag-1 ACF median 0.9346（0.903–0.948，250 Hz 短滞后强记忆）；窗口 ACF median 0.0876；净出度 −0.341–+0.626。
- **对照（出强度不减入 vs 幅值）**：GC_fwd 出强度~幅值组级 **ρ=+0.3634**（p=3.7e-3；逐例 +0.1353±0.1940）——**非 ≈0**（见结论措辞核注）；TRGC̄（净）出强度~幅值 = −0.0013 与 trgc-01 登记值逐位复现（断言过）。
- **确定性验证（最强档）**：pass1 vs pass3 同命令重跑，`diag_summary.json` sha256 `e90b1b4c…` 与 `data/diag_table.npz` sha256 `ebd9f46d…` **逐位一致**；pass2（scratch run_id）npz 同样逐位一致，summary 仅 `command` 字段回显的 run_id 不同（provenance 字段，非计算内容）。
- 产物：`outputs/bivar-gc-ec-fmt-250/diag-01/{diag_summary.json, data/diag_table.npz（48×ρ_fwd/ρ_rev/ρ_TRGC + 净出度/出强度/两口径 ACF/std 各 (48,62) + 组级 62 维向量）, logs/}`；scratch 证据 `diag-01-reprod/`；无图（json-only，理由见配置节）；`common/visualizer.py`、v1、refs-01/rev-01/sh-01/trgc-01 既有产物零改动。

### 结论

- **问 A：TRGC 单受试者内部信度显著低于两侧分半天花板**——fwd ≈ rev ≈ 0.90（分半天花板是管线/数据属性，与时间方向无关），而净差分读数 TRGC 分半仅 0.7079±0.3074 且 3/48 例为负（两半间净方向翻号）。与 trgc-01 互补并互证：净时序方向读数（fwd−rev）**在受试者间（无边存活）与单受试者内部（ρ 大幅降 + 负尾）两个层面都不可靠**，TRGC 不应用作受试者级有向边读数；方向性表述以 GC 读数本身（及其 rev-01 指纹）为界。与登记预期一致（差分放大相对噪声），如实登记。
- **问 B：净出度不被任何单通道属性显著预测**——lag-1/窗口 ACF、幅值三个组级 Spearman |ρ|≤0.21 且均 p>0.05，逐例均值近零：GC 出入不对称非单通道统计（记忆/幅值）伪影。**措辞核注（与派发预期不符处如实登记）**：派发预期「出强度~幅值应复现 trgc-01 的 ≈0」在**净（TRGC̄）出强度**上成立且逐位复现（−0.0013）；但 **GC_fwd 出强度（不减入）~幅值 = +0.36**——高幅通道确实系统性当 GC 源，该源倾向在 fwd/rev 两侧同构（compare_rev 出-出 ρ=0.93 佐证）、在净差分中相消（TRGC̄ 对照 ≈0）：幅值预测 GC 读数的总量/源质量，而非方向不对称。判读沿预登记框架（诊断性登记、无因果表述）。
- refs-01 / rev-01 / sh-01 / trgc-01 既有交付有效性不受影响；TRGC 侧使用建议（勿作受试者级读数）供 EC 推断线引用参照时知悉。
