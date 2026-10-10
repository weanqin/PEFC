# 科研方向档案（长期记忆）

> **维护规则**：SessionStart hook 把本文件完整注入每个新对话；重大节点/决策后当场更新；保持 ≤80 行，细节写入「关键文档地图」指向的文档。
> **初始化**：2026-10-10，依据用户提供的《基于 EEG 数据的虚拟扰动推断研究框架》定稿填写。

## 研究主题

**基于 EEG 代理脑虚拟扰动的节点连接推断**：先训练「过去 EEG → 未来 EEG」的条件生成代理模型（surrogate brain），再在代理脑内做虚拟扰动实验，以「临界扰动强度」定义节点间有向连接。权威框架文档：[research-direction.md](../../docs/research/research-direction.md)。

直接参考文献：Luo et al., *Mapping effective connectivity by virtually perturbing a surrogate brain*, arXiv:2301.00148v3（NPI）。**口径边界**：参考文献只给出「代理脑 + perturb-and-record」框架；「过去 EEG→未来 EEG」任务定义与「临界扰动强度」推断方法均为本研究自主设计，不是参考文献已给结论。

## 技术路线（当前活跃）

| 路线 | 一句话方法 | 代码位置 | 状态 |
|---|---|---|---|
| Study I：EEG Conditional Future Generation | 训练代理脑 X_past→X_future（时序预测/条件生成/时间修补三视角），含 EEG 表征、未来生成、代理脑有效性三层次 | 尚无（待建） | 框架已定，模型/数据选型未开始 |
| Study II：Critical Perturbation Connectivity Inference | 固定 θ* 后扰动源节点 A→测目标节点 B 响应→搜临界强度 r*_A→B=inf{r:Δ(r)≥τ}→临界扰动矩阵→连接强度 | 依赖 Study I | 四要素均未定义 |

Study I 是 Study II 的前置（先建立可信代理脑，再做虚拟扰动）。注意：**预测质量 ≠ Surrogate Brain Validity**，须作为两个独立问题评价。

## 既有结论（已验证）

尚无实验结论。已定案的是框架层事实（见框架文档）：两阶段流程（Stage I 训练并固定 → Stage II 扰动推断）、14 步完整流程、核心四问 Q1–Q4、术语表。

## 待用户拍板

- EEG 数据集选择与预处理口径（存放位置已定：`/root/autodl-tmp/`；具体数据集待定）
- 代理脑模型选型（Study I 结构与训练方案）
- 扰动算子 P(X,A,r)：加性/其他形式、输入层或中间表示、单时刻或时间段、正负方向、强度 r 定义
- 输出变化度量 D（未来波形差异 → 响应强度）
- 响应阈值 τ：须区分模型自然波动与扰动真实响应，应作为独立算法模块设计
- 临界强度搜索策略（逐步增加 / 区间搜索 / 数值优化）
- r* → 连接强度的最终映射

## 关键文档地图

- `docs/research/research-direction.md` — 研究框架权威文档（主题/两任务/方法框架/两子研究/术语表/核心逻辑）
- `.trellis/spec/research/` — 科研工作流规范六份（实验日志、路径隔离、可复现性、产物管理、评估口径、实验模板）
- `.trellis/research/environment.md` — 环境契约（写/跑代码前必读）
- **实验笔记目录与命名惯例**：随 Trellis 任务走 —— 每任务 `{TASK_DIR}/experiment.md`（创建任务时自动播种，格式见 spec/research/experiment-template.md）；跨任务独立实验记录目录未设立，如需要由用户指定

## AI 不得擅自触碰（须先问用户）

- 参考文献口径边界：不得把「临界扰动强度推断」表述为参考文献已给结论（框架文档明文区分）
- Stage 顺序：代理脑训练完成并固定参数（θ=θ*）之后才可做扰动实验；不得在未固定模型上做连接推断
- 数据集与预处理口径未拍板前，不得自行下载数据；项目数据一律存放于 `/root/autodl-tmp/`（用户 2026-10-10 指定），不得散落到工作区
- 以上前两条为框架文档自带的守护项，其余守护项（定版口径、不可覆盖产物目录等）逐项确认与增删以用户意见为准
