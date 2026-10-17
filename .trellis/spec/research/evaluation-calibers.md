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
