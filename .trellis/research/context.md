# 科研方向档案（长期记忆）

> **维护规则**：SessionStart hook 把本文件完整注入每个新对话；重大节点/决策后当场更新；保持 ≤80 行，细节写入「关键文档地图」指向的文档。
> **双重身份**：「技术路线（当前活跃）」表同时是任务路由（workflow 1.0b）与新路径/方法检测（1.0c）的对照 registry；新方法经 research 采纳后须在 3.3 回填此表。
> **初始化**：2026-10-10，依据用户提供的《基于 EEG 数据的虚拟扰动推断研究框架》定稿填写。

## 研究主题

**基于 EEG 代理脑虚拟扰动的节点连接推断**：先训练「过去 EEG → 未来 EEG」的条件生成代理模型（surrogate brain），再在代理脑内做虚拟扰动实验，以「临界扰动强度」定义节点间有向连接。权威框架文档：[research-direction.md](../../docs/research/research-direction.md)。

直接参考文献：Luo et al., *Mapping effective connectivity by virtually perturbing a surrogate brain*, arXiv:2301.00148v3（NPI）。**口径边界**：参考文献只给出「代理脑 + perturb-and-record」框架；「过去 EEG→未来 EEG」任务定义与「临界扰动强度」推断方法均为本研究自主设计，不是参考文献已给结论。

## 技术路线（当前活跃）

| 路线 | 一句话方法 | 代码位置 | 状态 |
|---|---|---|---|
| Study I：EEG Conditional Future Generation | 训练代理脑 X_past→X_future（时序预测/条件生成/时间修补三视角），含 EEG 表征、未来生成、代理脑有效性三层次 | `models/gpfm-mamba-v1/`（+v2/v4b/v4a/v1-fmt/v4a-fmt 变体目录） | gpfm-mamba v1/v2/v4 已实现，fold0 单 seed 前置验证（任务 10-12）与 H3 消融（任务 10-16）完成；**5-fold 候选集已拍板 {v1, v4a}**；新范式 edra-fmt fold0 前置验证完成（任务 10-16-gpfm-mamba-fmt-f0，2026-10-17，LayerNorm/p=25/无低通/GC 阻塞口径）；5-fold 主实验待启动 |
| Study II：Critical Perturbation Connectivity Inference | 固定 θ* 后扰动源节点 A→测目标节点 B 响应→搜临界强度 r*_A→B=inf{r:Δ(r)≥τ}→临界扰动矩阵→连接强度 | 依赖 Study I | 四要素均未定义 |

Study I 是 Study II 的前置（先建立可信代理脑，再做虚拟扰动）。注意：**预测质量 ≠ Surrogate Brain Validity**，须作为两个独立问题评价。

## 既有结论（已验证）

尚无实验结论。已定案的是框架层事实（见框架文档）：两阶段流程（Stage I 训练并固定 → Stage II 扰动推断）、14 步完整流程、核心四问 Q1–Q4、术语表。

2026-10-12 定案：数据集为 **edra**（`/root/autodl-tmp/edra_process_v4/`，HC 24 / MDD 22 受试者级 npz，62 通道、500 Hz，契约 `.trellis/spec/data/edra.md`）；时间窗统一 `[past=384, future=128]`；5-fold 分层划分与 dataloader 统一接口见 `.trellis/spec/research/data-pipeline.md`，可视化统一 visualizer 见 `.trellis/spec/research/visualization.md`。

2026-10-13 消融结论（fold0 单 seed，Run 010）：**「d_model=16 跨通道瓶颈是 GC 偏弱主因」不成立**——d_model 16→62（参数 ×8.4 单变量）后 GC 一致性反降（r 0.282→0.231）、生成 GC 幅值更低（真实 40%→25%）；Q1 改善边际（MAE 0.704，未超 GPR 基线 0.687）；FC 一致性 0.976、空间 R̂~R 0.969 略优。GC 偏弱指向损失/训练目标（无跨通道时序监督）或数据，为下一消融方向。

2026-10-13 高频问题三连裁决（Run 012–016）：生成侧 >40 Hz 功率偏高**主因 = OU 时间核 1/ω² 谱尾**（iso 仅 ~1% 次因）、网络 40–100 Hz 去噪不充分、log 刻度+真实侧数值地板放大。**先验侧三条路线全部证伪**：Run 014 带限核（条件方差塌缩不可用）、Run 015/016 先采样后带限（数值可行但网络把 40–100 Hz 重新注入至 ~6.7%、幅度过冲 +36%、GC 塌缩 0.231→0.022，阴性）。**正式裁决（用户拍板）＝输出侧 40 Hz 低通为 gpfm-mamba 路径产物口径**（evaluation-calibers.md，Run 017 落地）；正式口径下意外修复 GC 幅值校准（40%→92%）。v3 不作 5-fold 候选。

2026-10-16 H3 判定（任务 10-16 归档，用户拍板）：**「跨通道混合可局部移植为 GC 载体」不支持**——v4a rank-16 输入接口（GC r 0.092，五方最低、≈v2 CI 重叠）与 v4b cMLP 顶层读出（d1 0.133 / d8 0.163）两臂均不回升（v1 0.2755），缺口共同指向**损失侧**（无跨通道时序监督）或 v1 优势载体「深度持续混合」（16 维贯穿 trunk 全深，原 C 方案域不追）。v4a 逐通道质量五方最优（MAE 0.69540 / CRPS 0.52392 / 逐通道 r 0.18317 首超 GPR 均值基线），GC 缺口与质量缺口解耦。**5-fold 候选集 = {v1, v4a}**（v2/v4b-d1/d8 落选）。fold0 累计口径：32/4/10 划分、早停 10、恒定 lr 1e-3、gen_lowpass 40 Hz。

2026-10-17 edra-fmt 新范式 fold0 前置验证（任务 10-16-gpfm-mamba-fmt-f0，单 seed，LayerNorm/p=25/无低通/**genGC vs real GC 阻塞**口径）：v1-fmt/v4a-fmt 均过指标1（FC 0.970/0.981）与指标3（1–30 Hz |Δlog10| 0.317/0.301）；**v4a-fmt 质量全维度最优**（MAE 0.70565 / CRPS 0.53912 / 逐通道 r 0.12335 超 GPR-r 0.116），候选排序与旧范式一致（判读权重向 v4a 侧——v1 侧跨范式叠加纪律变量）；无低通下生成侧 >40 Hz 功率占比 ≈ 真实 3.0×（≈95% 功率在带内，大体自行待在带内）；v4a-fmt R1 触发定性为口径效应（登记三值）。跨范式 = 多变量设计（fs/预处理/48例/p/低通/norm/纪律），仅方向性判读。5-fold（何范式/口径）待用户拍板。

## 待用户拍板

- 代理脑模型选型（Study I 结构与训练方案）
- 扰动算子 P(X,A,r)：加性/其他形式、输入层或中间表示、单时刻或时间段、正负方向、强度 r 定义
- 输出变化度量 D（未来波形差异 → 响应强度）
- 响应阈值 τ：须区分模型自然波动与扰动真实响应，应作为独立算法模块设计
- 临界强度搜索策略（逐步增加 / 区间搜索 / 数值优化）
- r* → 连接强度的最终映射

## 关键文档地图

- `docs/research/research-direction.md` — 研究框架权威文档（主题/两任务/方法框架/两子研究/术语表/核心逻辑）
- `.trellis/spec/research/` — 科研工作流规范九份（实验日志、路径隔离、可复现性、产物管理、评估口径、实验模板、代理脑合格性检验、数据管道、可视化）
- `.trellis/spec/research/surrogate-brain-qualification.md` — Study I 代理脑合格性检验规范（任务路由到 `study1` 时经 1.0b/1.3 调用；固定 θ\* 前须完成三项检验）
- `.trellis/research/environment.md` — 环境契约（写/跑代码前必读）
- **实验笔记目录与命名惯例**：随 Trellis 任务走 —— 每任务 `{TASK_DIR}/experiment.md`（创建任务时自动播种，格式见 spec/research/experiment-template.md）；跨任务独立实验记录目录未设立，如需要由用户指定

## AI 不得擅自触碰（须先问用户）

- 参考文献口径边界：不得把「临界扰动强度推断」表述为参考文献已给结论（框架文档明文区分）
- Stage 顺序：代理脑训练完成并固定参数（θ=θ*）之后才可做扰动实验；不得在未固定模型上做连接推断
- 数据集与预处理口径未拍板前，不得自行下载数据；项目数据一律存放于 `/root/autodl-tmp/`（用户 2026-10-10 指定），不得散落到工作区
- 以上前两条为框架文档自带的守护项，其余守护项（定版口径、不可覆盖产物目录等）逐项确认与增删以用户意见为准
