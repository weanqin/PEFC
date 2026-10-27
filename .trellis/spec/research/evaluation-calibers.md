# 评估口径规范

**原则**：指标定义先于实验运行；口径一经确定，任务中途不换。

## 跑前

- **每个指标写清三件事**：定义（公式或精确调用）、数据切分（train/val/test 划分方式与固定划分标识，如 fold id）、聚合方式（按样本 / 按被试 / 按折平均——混用即口径错误）
- **对比基准明确**：与哪个基线比、基线的代码版本与口径是否一致；口径不同的对比必须显式声明「不可直接比较」
- **主指标唯一**：预先指定 1 个主指标 + 少量次指标，防止事后「挑好看的汇报」

## 跑后

- 汇报主指标必须附不确定性：多 seed（关键结论 ≥3）均值 ± 标准差，或按领域惯例给置信区间 / 显著性检验
- 单 seed 得出的结论必须标注「单次运行」；论文级关键结论须多 seed 或多折支撑，否则降级为「初步观察」

## 口径变更纪律

- 中途换指标 / 换切分 / 换聚合方式 = **新口径**：旧 Run 结论保持原口径不动，新 Run 显式声明新口径及与旧口径的不可比部分
- 事后分析的多个指标口径统一记录在 experiment.md 或长期实验笔记，防止「口径漂移」无声发生
- 与既往实验对比时（见 [experiment-log.md](./experiment-log.md)），第一步永远是核对两边口径是否一致

## Study II 扰动 EC 对比口径（2026-10-13 登记，任务 10-13-baseline-ec-v1）

- **术语纪律**：GC/gPDC/（若后补的）TE 一律称 **observational directed-connectivity references**；目标是**结构收敛性验证**，不得表述为 ground-truth / 恢复真实因果。
- **比较关系**：主问题 = 逐参考法独立配对比较 EC_pert↔EC^(m)（m∈{GC, gPDC}；TE 全系经用户 2026-10-13 拍板取消——16 核 cgroup 实测 S=50/S=100 置换 35–70 天不可行）；参考法互比不作科学问题（GC 与 gPDC 共享 VAR 拟合，互一致部分构造使然）。
- **主指标（唯一）**：逐被试**距离偏相关 Spearman ρ**（3782 有向边展平，通道对三维欧氏距离作偏出变量）；检验 = 单样本 t × Holm（跨参考法）+ 1000×受试者 bootstrap CI（seed 2026）+ ρ/split-half 天花板比值。未偏相关 ρ 仅作上下文。空间偏相关为主指标系用户三轮明确要求（Run 008/009 教训：K_s~R=0.891，光滑空间结构虚抬朴素矩阵相关）。次级指标轨全部取消（用户指定）。
- **参考值口径**：原始边值直入比较器（用户指定）；GC=ridge 条件 ln(SSR 比)（λ_rel=1e-2、p=8 固定，定标证据见该任务 experiment.md Run 001）、gPDC=|gPDC|²（宽带 1–40 Hz）。
- **EC_pert readout 设计约束（Study II 落地时执行）**：主 readout=**低剂量斜率**（剂量-响应 D_j(r) 线性段 OLS 斜率），AUC 次、r* 降级（右删失标注 + τ 阈值敏感性登记）；r 网格低密度段加密；斜率/AUC 天然大=强，免方向翻转。比较器接口契约：npz{ec (62,62), readout∈{slope,auc,rstar}, subject_ids|'global', orientation='larger_is_stronger'}（rstar 须调用方先翻转）。
- **方向约定全仓库统一**：EC 矩阵行=源、列=靶；有向边展平非三角（3782）。

### 首版回填（2026-10-17，任务 10-17-perturb-ec-f0）

- **τ 双口径 v2（用户拍板）**：操作性 = **相对自响应 τ̃=0.25**（D̃=D̄_{A→B}/D̄_{B→B}，敏感 0.50；D̃ 不作跨靶数值比较）；预注册绝对口径（独立样本对 q95）降为**材质性观察**照登（v4a-fmt θ\* 实测 100% 删失：q95≈1.82σ vs 全边最大响应 1.14σ）。变更依据：CRN 契约下 D 确定性，绝对口径实为极端材质性标尺。
- **CRN 确定性契约、增益算子、readout 行为、隔离装载矩阵、首版实测基线**：可执行契约沉淀于 [perturb-ec-pipeline.md](./perturb-ec-pipeline.md)（study2 任务必读）。首验 AC6 判定：v4a-fmt θ\* 上未获结构收敛初步证据，缺口=跨通道传导特异性（四重证据），非流程/readout/参考侧问题。

## 双变量 GC 观测参考口径（2026-10-17 登记，任务 10-17-bivar-gc-ec-fmt，用户四轮拍板）

- **口径内容**：edra-fmt **48 例全量**、250→125 Hz（`resample_poly(1,2)` 抗混叠、release 已带限不重复滤波）、逐通道全记录 z-score（ddof=0 σ≥1e-12）；**成对双变量 GC**——逐无序对 {A,B} 一次二通道 VAR(p) 拟合（全通道滞后 Gram 一次构建 + 逐对 slice 快路径，与 naive lstsq 等价性单测锁定 rtol=1e-10），GC(A→B)=ln(SSR_r^B/SSR_u^B) 同阶双向读出，**不条件化于其余 60 通道**；行=源/列=靶/对角=0；**OLS 无收缩**（二通道 2p+1≤49 回归元、pair-Gram 条件数实测 ~1e5–1e7，与全模型 62ch ~1e18 需 ridge 情形不同构；offdiag≥0 为样本内必然并作实现正确性断言，负值=实现错误）。
- **逐对 BIC**：p∈{1..24} 嵌套行集 row_from=24；BIC=ln|Σ̂|+K²p·ln(n)/n（K=2，截距不进惩罚）。**实测事实：n≈3 万下惩罚项极小，48 例 × 1891 对 100% 顶格 p\*=24**——阶数分布退化单点如实登记；网格扩展（如 {1..48}）属口径变更，未决。
- **产物与基线**：`outputs/bivar-gc-ec-fmt/refs-01/`——refs_all.npz (48,62,62) f32（offdiag min/median/max = 0.0007/0.0153/0.4356 ln 单位）+ refs_summary + **split-half 天花板 0.9243±0.0447**（v4 侧条件 GC 天花板 0.9003±0.0296 同量级，方向性对照只登记不判读）；代码 `models/bivar-gc-ec-v1/`（20 项单测；跨 Run 逐字复现）。
- **与既有条件 GC 口径区分**：baseline-ec-v1 主口径 = 全模型条件 GC（ridge λ_rel=1e-2、p=8 固定、收缩下允许微负）；本口径 = 双变量无收缩非负——两口径平行**不得混用**；fmt 条件 GC 任何形式（10 例复用 / 48 例重算 / 时点）滞后决策（用户 2026-10-17）。
- **术语与边界**：observational directed-connectivity reference，不得表述为 ground-truth；**不做 EC_pert 与 GC 的比较**（用户 2026-10-17 第四轮拍板），比较器参考列接入滞后（后续比较时本口径可直接作参考列 + 天花板分母）。

## 双变量 GC 观测参考口径 · 250 Hz 原始全长（2026-10-18 登记，任务 10-18-bivar-gc-fmt-250，用户拍板；与上方 125 Hz 条目并列区分、不得混用）

- **口径内容**：edra-fmt **48 例全量**、**250 Hz 原始全长原样进入拟合（无降采步；60001 点不切窗）**——与 125 Hz 条目的唯一数据差异 = 不降采；逐通道全记录 z-score（ddof=0 σ≥1e-12）同式；**成对双变量 GC** 与 125 Hz 条目逐式同构（全通道滞后 Gram 一次构建 + 逐对 slice 快路径 + ln(SSR_r/SSR_u) 双向读出、不条件化于其余 60 通道；行=源/列=靶/对角=0；OLS 无收缩，offdiag≥0 为实现正确性断言）；**P_MAX=48** 保物理滞后 192 ms（125 Hz@24 ≡ 250 Hz@48，滞后范围不变量）。
- **逐对 BIC**：p∈{1..48} 嵌套行集 row_from=48（n_eff=59953/例）；**实测事实（如实登记）：p\*=48 占 96.8% + p\*=47 占 3.2% + <0.1% 散布 44/46——顶部集中但非 100% 顶格**；顶部两阶之分与零星散布为数值噪声级（见下条条件数），实效读出 ≈ 固定阶 VAR(48) 双变量 GC（caliber 注记向下游言明估计量性质）；网格扩展属 v1 遗留未决决策不变。
- **条件数实测带与数值策略（250 Hz 特有登记）**：250 Hz 过采样下相邻滞后共线性升高，pair-Gram 条件数实测 **2.0e13–5.0e15**（v1 125 Hz 带 ~1e5–1e7 的 6–9 个量级之上；smoke-01 触发预注册 halt 门限，经主会话代决策**按原 OLS 口径继续**——不启用正则化/对角加载、不改 v1、不变 P_MAX）。**依据 = fast↔naive 交叉校验**（fast Gram-slice vs naive lstsq SVD 稳定路径，逐 Run 逐例抽样 50 对、sh-01 每半 25 对，rng 2026）：GC 读出绝对差 max ≤ **1.5e-3**（smoke 全曲线级 ≤3e-4），全部硬断言（非负/对角零/有限/SSR 单调）过、Gram 非奇异（min eig 2.5e-10）；确定性 = worker 1 线程 BLAS 下跨 Run **raw bytes 级逐位复现**（BLAS 线程数改变归约序 ⇒ 同例 GC 差 ~2.4e-4，为条件数放大下的固有数值散度带，读出不确定度量级登记）。
- **产物与天花板**：`outputs/bivar-gc-ec-fmt-250/refs-01/`——refs_all.npz (48,62,62) f32（offdiag min/median/max = **0.0010/0.0189/0.4103** ln 单位）+ refs_summary（fast_naive_check）+ **split-half 天花板 0.8981±0.1414**（`sh-01/`，段 500 样本/2 s、奇偶分半、每半 27120 行、逐例 ρ 中位 0.9561）；代码 `models/bivar-gc-ec-v2/`（8 项 31 断言单测；只读复用 v1 bivar_gc/split_half 可传参函数，**不调 v1 ceiling_for_subject**——无 seg 参数不透传，v2 本地编排 + 运行时口径锁 n_seg==120/每半 27120 行）。
- **时间反转指纹（rev-01，描述性登记）**：`X_rev = X_z[:, ::-1]` 同口径重算——ρ(offdiag fwd, rev)=**0.6608±0.2014**（≪1 ⇒ 读数含实质时间方向成分、非对称耦合主导）、ρ(fwd, revᵀ)=0.3236±0.1323（不满足镜像对应）、幅值比 rev/fwd=0.9968±0.0569、不对称度组级标量双侧 0.0191 保持；GC_rev 落盘 `rev-01/`（reversed=true 标记）。EC 推断线引用本参照时应知悉该指纹。
- **与 125 Hz 条目的关系（口径隔离）**：两条参照平行（不同采样率口径），**不做对比交付**（用户 2026-10-18 拍板）——不出对比表/图，产物根/变体目录严格分离（`outputs/bivar-gc-ec-fmt-250/` vs `outputs/bivar-gc-ec-fmt/`）；后续如需跨口径比较须另立任务并显式登记不可比部分。
- **术语与边界**：沿 125 Hz 条目（observational directed-connectivity reference，非 ground-truth；不做 EC_pert 比较）。


## 原库谱 GC 观测参考口径 · edra_v4 重采样 250 Hz（2026-10-21 登记，任务 10-21-oldlib-gc-edrav4，用户拍板）

- **主路径登记**：GC 计算主路径 = 原库 GC（2026-10-20 拍板，见 `.trellis/research/context.md` 技术路线表）——后续 |A_eff|/EC 类 × GC 对照一律用 `models/latent-flow-v1/latent_flow/baseline/granger.py::granger_connectivity`（成对双变量谱 GC，Geweke 1982 / Dhamala 2008；ec[a,b]=a→b 行=源/列=靶/对角=0；对齐口径 |A_eff| vs ec.T）。本条目为首条落入本文件的该主路径口径。
- **口径内容**：`/root/autodl-tmp/edra_process_v4/` **46 例全量**（HC 24 + MDD 22；**不做 qc_pass 筛选**——manifest qc_pass=True 仅 13/46，均以 status=ok 发布，46 例全算为用户口径，事实随档）、原生 500 Hz 85000 点/例 → `scipy.signal.resample_poly(x, 1, 2, axis=-1)` → 250 Hz 42500 点/例（唯一新增预处理步，确定性无随机源；数据 bp1-40 带限，新 Nyquist 125 Hz 无混叠内容）→ 重采样后逐通道 z-score（ddof=0，σ==0 置 1，同旧库 loader）→ 全长单段连续（不切窗）→ `granger_connectivity(fs=250, max_order=8, n_freq=65, fmin=1, fmax=125, fixed_order=8)`（无 AIC 扫描；float64 计算 / float32 落盘）。
- **产物**：`refs/granger_bivariate_spectral/edrav4-46-resample250-cont-zscore-fs250-fixed8-f1-125-nf65/`——46 × refs_s{id}.npz（ec float32 (62,62) 行=源）+ meta.json + gen_refs.py/log + compare_fmt_summary.json + SHA256SUMS；任务 outputs 留 `outputs/10-21-oldlib-gc-edrav4/`（工作副本 + 日志）。生成 15.4 min（12 进程 ~20.4 s/例）；确定性 = 同例删除重跑 npz 容器级 SHA256 逐位一致（AC4 实测）。
- **与 edrafmt48 档的关系（口径隔离，不做对比交付）**：edrafmt48 档 = 同实现同参数 × fmt 管线数据（240 s/例，60001 点）；本档 = 同实现同参数 × v4 原管线重采样（170 s/例，42500 点）。**重叠 45/46**（HC 24/24 全重叠；v4 MDD s56 为 fmt 所无，fmt 独有 MDD {11,21,48} 为 v4 弃用例）——下游两档对照一律取 45 例交集。实测对照（描述性登记，非对比交付）：offdiag 3782 边 Spearman ρ=0.2311±0.1026、Pearson r=0.4268±0.1266、median 比值 v4/fmt=1.651±0.664（低-中正相关 + v4 幅值系统偏高，与数据时段/管线差异相容；非 ≈1 属预期）。offdiag 量级两档同数量级（v4 mean ~55–95 ln 单位 vs fmt ~36–91）。
- **术语与边界**：observational directed-connectivity reference，非 ground-truth；不做 EC_pert 比较；引用本档时须言明重采样步骤与 qc_pass 事实。

## gpfm-mamba 路径生成产物口径：输出侧 40 Hz 低通（2026-10-13 登记，任务 10-12-gpfm-mamba-edra-f0，用户拍板）



- **口径内容**：gpfm-mamba 路径（`models/gpfm-mamba-v1/v2/v3` 及后续派生）的生成样本自本口径起，一律先经「reflect-pad(±64) + rFFT 砖墙 >40 Hz 置零」低通（fs=500 Hz），再进入一切下游产物与指标：K 样本**逐条**滤波且在聚合（点预测均值/medoid/分位带）**之前**；GC/PSD 输入（`gc_input.npz`）、FC 指标、空间 R̂、轨迹图同源。**真实侧不滤**（数据 v4 已 bp1-40+notch50，契约见 [../data/edra.md](../data/edra.md)）。
- **实现契约**：唯一正典实现 = `models/gpfm-mamba-v1/gpr_prior.py::brickwall_lowpass(x, cut_hz, fs=500.0, pad=64)`（torch，任意 ndim≥2；置于 gpr_prior 因其已被全部消费方 import，避免反向依赖/循环导入）。装配点 = `EulerSampler.sample()` 的 ODE 终点返回前（单一入口覆盖 Pass A / Pass B / 图全部消费方）；截止频率经 `getattr(cfg, "gen_lowpass_hz", 40.0)` 透传——**不得向已定版变体 config.py 加字段**（基线 config_hash 逐位不可变）；`evaluate.py` 提供 `--gen-lowpass-hz` 覆盖（0=关闭，用于旧口径复现，须在 Run 块登记）。
- **变更纪律**：旧 Run（007/010/016 等）数值保持原口径不动；新口径 Run 显式声明；跨口径对比标注不可比部分。依据链：Run 012 根因（OU 时间核 1/ω² 谱尾，先验侧不可修）→ Run 013 验证（MAE −5%、谱归零、reflect-pad 无振铃）→ Run 016 先验侧路线证伪（网络重新注入 40–100 Hz），详见该任务 experiment.md。

## R1 触发解读规则（2026-10-13 登记，任务 10-12 实测沉淀）

- R1 监控的「采样终点 MAE ≥ 训练末期 x̂1-MAE × 1.5」分支在**训练折拟合加深的模型上会机械触发**——分母（train 折一步 mae）变小而非分子（test 折终点）变差（实测 v2=1.532 / v3=1.555 触发，终点 MAE 均未劣化于对照）。**触发后定性流程**：先核对训练末期 mae 是否显著低于同折对照，再定性「高 t 欠约束风险」或「口径效应」；登记必须附分子分母两值与对照参考。

## edra-fmt（新预处理范式）产物口径（2026-10-17 登记，任务 10-16-gpfm-mamba-fmt-f0，用户拍板）

- **genGC vs real GC 一致性对比暂时阻塞**：`gc_psd.run_all(skip_gc=True)`（default-off 钩子；evaluate `--skip-gc-only`）——GC 一致性不出现在任何产物；PSD/空间归因照跑。解除时点未定，解除时须先拍板 250 Hz 下 GC 口径（VAR 阶数采样点 vs 墙钟、fs 透传）。
- **生成侧不加 40 Hz 低通**：`gen_lowpass_hz=None`（真实侧本就 1–40 Hz 带限，直接观察模型是否自行待在带内；实测 fold0 单 seed 生成侧 >40 Hz 占比 ≈ 真实 3×，大体在带内）。**旧范式（edra 500 Hz）40 Hz 低通口径不变**；跨范式谱类对比须登记该差异。
- **`period_p` 50→25**：fs 减半时 GPR 时间核 p 同步减半保物理时长（`gpr_prior.py` τ 网格采样点单位）。跨范式对比 = 数据+先验（+低通+norm+纪律）多变量同变，非单变量，判读只做方向性观察并随附差异清单。
- **fs=250 全链透传**：config `fs` 字段经 `getattr(cfg, "fs", 500.0)` 进 PSD Welch 与轨迹图（旧变体无字段 = 500 零漂移）。指标3 频段边界适配见 surrogate-brain-qualification.md（判读窗 1–30 Hz、照登段 40–125）。
- **norm RMSNorm→LayerNorm**（`norm_type` 钩子，默认 rmsnorm 零漂移）：fmt 两变体 5 个 norm 位全换；容量 +80（v1-fmt d16）/+310（v4a-fmt d62）。

## EC_pert · global-tau × gpfm-mamba-v5-main 口径（2026-10-21 登记，任务 10-21-ec-globaltau-v5main，用户拍板）

- **方法骨架**：07-04-global-tau（latent-flow-v1）逐字复用——加性高斯 σ·ξ + μ_bias=0.5·ch_std（方向符号）/ per-target MSE vs 无扰动确定性基线 / σ_ref=0.3 → 全局 q=0.5 分位 τ → log_grid[0.01,0.3]×24 单调走查 σ\*（clamp）→ conn=−(log σ\*+½log 2πe) → EC_pert=sign⊙conn；σ 搜索原语直接 import `bisect_sigma.py`。扰动施加点 = z-score 后、入模前、x^p 段（x_0 恒为 x^p_pert）。
- **代理脑**：v5-main-01 θ\*（`outputs/gpfm-mamba-f0/v5-main-01/checkpoints/gpfm-mamba-v5-best.pt`，best_val 0.73155@ep64）——**EMA 正典权重 + bf16 autocast**（用户拍板；对 gpfm 正典 fp32 为登记偏离；explore 先例 10-21-v5main-ec-explore 系 raw-weights 未 swap，其 m1/m2 数字与本口径**不可数值直比**）。生成噪声 (eps,eta) 被试级固定（seed*1e4+sid），跨源/跨 σ 逐位一致；ξ 按源 seed+source。
- **代理脑契约映射**：edra_process_v4 @500 Hz（窗 1321/例，past 384/future 128/stride 64）、euler_steps=16、fold0 test 10 例 [12,17,18,24,30,38,44,55,58,61]、生成侧 40 Hz 低通口径照挂（sampler 终点）。
- **实现**：`models/ec-globaltau-v1/`（import-only，两源包零改动）；确定性 = 同 seed 复跑 σ\*/τ/conn 逐位一致（s12 实测）+ S1 五断言（扰动隔离/同输入跨调用逐位/EMA swap 生效/ξ 与 (eps,eta) 逐位一致）。
- **对照链**：refs edrav4-46 档（行=源，aligned 主口径 + flipped 诊断，ρ 描述性）；冒烟 s12 实测 r_signed_aligned=−0.149 / r_abs=0.151（单例，不外推）。
- **边界**：EC_pert 结果不构成代理脑资格；n_trials=32 冒烟口径，主跑规模待二次拍板。
- **口径修订（2026-10-21，Run 002，用户拍板）**：q 0.5→**0.75**、σ 网格 [0.01,0.3]→**[0.01,0.48]×24**（σ_ref=0.3 不变）、conn 公式 07-04 原式 → **−(log σ\*+log 2πe)+1**（用户修订；公式串随 JSON 落盘）。run_01（原口径）产物保留，两口径不混比。10 例×32 窗主读出：r_abs +0.101±0.042（10/10 全正）/ r_signed −0.071±0.029（10/10 全负）。

## 代理脑生成质量数值抽查口径：逐点 coverage + medoid 幅度比（2026-10-21 登记，任务 10-21-latent-flow-mild-train Run 004–006）

- **用途**：条件生成代理脑（θ 候选）的快速数值判读，先于/伴随波形人工判读；对「smoke std_ratio≈1 但波形判读不通过」类表观矛盾给硬判据。
- **口径**：每 trial 采 K=32 条生成样本（seed 42 与 viz 同批）；①**逐时间点 Q05–Q95 覆盖率** = 真实 future 各时刻落在 32 样本分位带内的比例（90% 带名义覆盖应≈90%）；②**medoid/real std 比** = medoid 样本（k\*=argmin 平均两两 RMSE，实际样本禁统计重构）逐通道 std 对真实 future 逐通道 std 的比值（按 trial 均值报告 mean/min/max）。
- **双口径纪律（防末端发散伪影）**：全指标同时报「全程」与「去末端 25 点」——Euler 末步发散尖峰会抬高 medoid/real 表观值（实测全程≈1 而去末端≈0.6）；coverage 两口径均报（实测两口径可同为 0%）。
- **判读基准（Mild Depression × latent-flow-v1 实测）**：三窗口口径（3:1/5:1/9:1）coverage 一律 0.00%、medoid 主体 0.5–0.74 ⇒ 生成分布逐点方差不足 + 末端发散属**结构性缺陷**，训练预算/窗口划分修复无效——后续代理脑候选建议先过本抽查再谈合格性三项检验。
- **禁读**：不得把 smoke 的池化 std_ratio 当校准证据（重尾样本可使其≈1 而分位带仍窄）。

### 增记一：确定性采样口径下 coverage 的适用性限定（2026-10-24，任务 10-24-latent-flow-mild-w500）

- **口径事实**：latent-flow-v1 `sample()` 全确定性（z_obs=μ + 确定性 Euler + 确定性 decoder，
  无后验采样）——K 条"生成样本"由同一 x_obs 复制而来、**逐位相同** ⇒ 逐点 Q05–Q95 分位带
  宽度恒为零，coverage **构造性 = 0，不携带质量信息、无证伪力**（上轮三口径 coverage 0%
  的主成分即此结构性结果，medoid/real std 欠幅才是训练分布缺陷的直接证据）。
- **判读规则**：确定性采样口径下本抽查的判读以 **medoid/real std 比 + 波形人工判**为主；
  coverage 照登并注明「结构性 0」。要检验条件分布逐点校准（coverage 的本义）须另轮开放
  后验采样（推断侧口径变更，须拍板）。
- **去末端口径**：定点沿用绝对 25 点（10-21 登记）；forecast 段长度变化时占比随之变化
  （375 点 6.7% → 125 点 20%），跨口径判读须知占比差异。

### 增记二：前缀窗 500（375/125）口径首测（2026-10-24，任务 10-24-latent-flow-mild-w500）

- 配置：fold0、batch 64/lr 5e-4/stage2 decoder+flow 联合（仅冻结 encoder）、anti-uniform
  0.1、归一化前 500 点、确定性采样（用户 2026-10-24 拍板）；test 3 例（sid 1/7/21）×
  30 trial × K=32 seed 42，与上轮 Run 004–006 同 triad。
- 实测：**medoid/real std 全程 0.957（0.67–1.18）/ 去末 25 点 0.996（0.62–1.29）**，逐例
  均值 0.88–1.05——上轮 0.5–0.74 欠幅在该口径下基本消除；coverage 双口径 0.0000
  （结构性，见增记一）。单次运行（fold0 单 seed，bf16 AMP）。
- **出口 GNN 变体复测（10-24-latent-flow-exit-gnn，lr 1e-3 二因素）**：medoid/real std 回落
  全程 0.645/去末 0.581（逼近 10-21 旧线区间），val corr2 持平——出口邻居混合的"再平均"
  效应在幅度维度立判；判读该变体时幅度回归为负证据。

## 不变性断言的精度路径纪律（2026-10-24 登记，任务 10-24-exitgnn-ec-unbiased Run 003）

- **坑**：对「公式 B = 公式 A + 常数」类单调重映射做 Spearman/Pearson 不变性断言时，若 A
  取自 json 落档的 **float32 舍入值**而 B 由 σ\* 以 **float64 重算**，舍入可引入新并列
  （实测 s5：1−ρ=7.3e-13）——精度伪影，非口径漂移；`== 1.0` 硬断言会随机误伤。
- **契约**：不变性断言一律在**同一条精度路径**上做（由原始量以同一 dtype 重算 A 与 B），
  容差取 1e-9（房内先例口径）；json 落档值的舍入差另作诊断字段随档登记（本例
  max|Δ|=2.7e-7）。适用一切后续 conn/r\* 公式重映射与「逐位不变」类校验。

## 架构级先验不自动传递到 EC 读出——对称性诊断实证纪律（2026-10-25 登记，任务 10-25-symw-ec-unbiased）

- **坑**：sym-W 变体（SymWAdam 每步 W←(W+Wᵀ)/2，best ckpt W torch.equal 级对称） ⇒ 直觉
  预期「A_eff 无向 ⇒ EC_pert 坍缩为无向、方向性按构造受限」并据此预设与有向 GC 相关的
  上限。**实证相反**：EC v4 矩阵 M vs Mᵀ offdiag Pearson 组均 −0.002±0.135、相对不对称度
  ‖M−Mᵀ‖_F/‖M‖_F（v4）≈1.02，与 w500（+0.05/0.99）、exit-gnn（−0.015/1.08）同量级——
  EC 方向结构由非对称逐通道编码/出口投影 + 数据依赖 σ\* 承载，W 对称仅约束 GNN 消息传递
  通路，不约束端到端读出。
- **契约**：凡以「架构级约束 ⇒ EC 读出必然 X」为前提的 caveat/预期（对称化、掩码、低秩
  等），交付前必须做**读出侧实证诊断**，不得默认成立。对称性诊断最小集 = conn 矩阵 M vs
  Mᵀ 的 offdiag Pearson+Spearman + 相对不对称度；比值类指标须**双口径注记**（常数平移改
  ‖M‖ 不改 ‖M−Mᵀ‖ ⇒ 比值口径敏感、相关系数不受影响）。
- **判读分支**：corr(M,Mᵀ)→1 且 asym→小 ⇒ 坍缩成立（预期兑现）；取反向分支 ⇒ caveat 收窄
  为「约束仅作用于该通路」，相关弱的瓶颈归因须另找（本例 = 整体对齐弱而非对称性坍缩，
  aligned +0.134 > flipped +0.102 仍弱保持方向特异）。
- **Wrong vs Correct**：Wrong = 在 PRD/README 写「A_eff 无向 ⇒ EC 无向」并按无向口径直接
  预设 GC 相关上限；Correct = 同句 caveat 附「待对称性诊断实证」并落实诊断（本任务
  `models/latent-flow-v1/symmetry_diag_symw.py` 先例，读 ec_main 逐例 json 纯后处理）。

## 跨折 EC 复用的折键控校验契约（2026-10-25 登记，任务 10-25-symw-ec-f12）

- **坑**：EC 脚本的 norm SHA 断言若钉死单折训练登记值（fold0），跨折复用时要么静默换档
  （口径漂移）要么拒跑——且各折 norm 来自各折 train split，**不可跨折借用**。
- **契约**：跨折 EC 复用时 norm SHA 必须 **fold 键控映射**，值取**训练侧 configs 的
  norm_stats.mean_std_sha256 登记值**（stage1/stage2 须一致，四方核对：常量=训练登记=PRD=
  逐例 payload）；manifest 若单文件全折共用则 SHA 断言不变。ckpt 守卫同步折锁定：f12 类
  run_dir 断言 ckpt 文件名 `fold{k}_` 前缀与 `--fold` 一致（防折-权重错配），先例 run_dir
  收紧为 `--fold 0`（else 半支补齐拦截——"路径子串断言"对不含该子串的负例无约束力）。
- **跨折并置口径**：各折 θ 为不同代理脑实例 + test 被试互斥 ⇒ 并置为 DESCRIPTIVE（不作
  复现性判读）；跨折一致性结论须以"形态同型"（如 GC 相关符号结构/σ* 中位量级）而非数值
  重合来表述（先例：crossfold_summary_f12.py，fold0/1/2 三折对照）。

## md-latent-flow 线 EC 对照配置固定（2026-10-27 登记，任务 10-27-md-latent-flow-config-fix）

- **线默认 conn 公式 = v4**：−(ln σ*+0.5·ln2πe)+5/7——`ec_mild_unbiased.py` 主跑直出
  （公式串随档落盘含 "5/7"）；房内 v1 −(ln σ*+ln2πe)+1 退役为历史口径。v4 = v1 + 常数
  平移 ≈ +1.1332 ⇒ 秩相关读数不变；网格 [0.01,0.48] 上全正（min=+0.0293）⇒ abs≡raw。
- **GC 对照 refs = v1 mild 档单参考**（1–45 Hz fixed8）；v2 1-125hz（aic4-25 档）对比项
  经用户拍板注销（refs 物理文件已删，`refs/MANIFEST.md` 留注销注记）。在档 dual 时代产物
  （outputs 内 gccorr_dual_summary.json 的 v2 块）为历史数据，**保留不重生成**。
- **配置切换的兼容契约**：① 断言守卫按「时代两态」放宽——crossfold `asserts_passed
  in (40, 32)`（40 = 双参考时代在档 / 32 = v1 单参考复跑），其余值仍拒（守卫目的不变）；
  ② json 侧诊断按档 `connectivity_formula` 声明自适应取对照式（recompute rounding 诊断，
  键 `json_f32_rounding_maxabs`；公式错配表现为 ~1.13 平移量级而非舍入量级，可即时识别）；
  ③ 产物文件名 `gccorr_dual_summary.json` 为在档 schema 契约名，**不随配置改名**（下游
  crossfold/viz 读取路径指向在档文件）。
