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
