# latent-flow flow 出口 GNN 图混合变体设计（mild w500 衍生）

## Goal

基于任务 10-24-latent-flow-mild-w500 的架构（OCLatentFlow · mexhat-graph flow · mild 前缀窗
500 口径），设计轻量变体：**将 flow 出口处的零初始化逐通道 Linear（`flow.graph_out`）替换为
与入口一致的 GNN 图混合模块**（A_eff 消息传递 + 零初始化出口投影，保持 v≡0 恒等起步）。
本任务第一阶段仅产出方案设计供用户审阅；**用户确认通过后才进入代码实施**。

## 背景与动机

- 现行架构中全模型唯一的跨通道交互位 = flow 入口图混合（`LearnableMexHatMixer`）；出口为
  纯逐通道零初始化投影（mexhat 分支 `oc_latent_flow.py:598-601`，注释明言"不在出口做邻居
  混合以免入口带通结构被出口重新平均"）。
- 用户提出变体方向：出口也引入 GNN 图混合（与架构既有 GNN 模块一致的形式），检验出口侧
  邻居混合对 fc 生成质量/连接结构的作用。
- 前置事实：w500 口径 Run 003 显示 medoid/real std 已对齐（0.957/0.996），stage2 E150 顶格
  未收敛——本变体属结构性迭代实验，与"延长预算"正交，可组合。

## Requirements

- **R1 修改面（唯一结构性修改）**：`FlowVelocityNet` graph 模式（mexhat 分支）出口——
  `out_norm` 之后、输出 v 之前——插入 K 轮 A_eff 图消息传递块；末端的 H→D 投影**保留零初始化**
  （保 v≡0 恒等起步）。入口混合器、6 膨胀卷积块、时间嵌入、encoder/proj/decoder、损失函数、
  采样确定性全部不变。
- **R2 模块一致性**：出口图混合与入口 `LearnableMexHatMixer` 同构——同样的残差式
  `x += GELU(Linear(A_eff @ X))` 消息传递、同 A_eff 参数化（γ_M·MexHat(d;σ)+γ_W·W）；
  K_exit 默认 = 入口 graph_k = 2；激活 GELU；不新增归一化层（与入口消息循环一致，
  出口前已有 out_norm）。
- **R3 A_eff 归属（待用户拍板的决策点）**：默认方案 = **与入口共享 A_eff**（σ/W/γ 单份、
  两应用位、梯度双路累计、参数增量最小、EC 读出故事单一；先例 gpfm-v5 跨块共享）；
  备选 = 出口独立 A_eff（+16,385 参数，出口独立读出）。
- **R4 工程纪律**：包体 `latent_flow/` 零改动——变体以**子类**落变体层新文件
  （`FlowVelocityNetExitGNN(FlowVelocityNet)`，覆写 `_forward_graph` 尾段）；wrapper 增
  `--flow-exit-gnn` 旗标（默认关 = 原架构 bit 兼容）；viz/quality 装载路径同旗标透传（strict
  load 架构匹配）；stage2 本地 runner 的 flow 前缀白名单补 `flow.exit_`（VAE ckpt 装载面）。
- **R5 实施依赖**：本变体实施以 w500 任务的未提交改动为基座（train_mild_depress.py 参数化
  + 本地 runner 等）——**w500 提交计划确认落地后再动代码**。
- **R6 轻量约束（用户指定）**：改动范围最小（预计 1 个变体新文件 + wrapper/viz/quality 各
  少量旗标接线）；参数增量默认方案 **+33,024**（flow 488,897 → 521,921，+6.8%；全模型
  1,243,234 → 1,276,258，+2.7%）；备选方案 +49,409（+10.1% flow）。

## Acceptance Criteria

- [ ] **AC1** 方案文档（design.md）覆盖用户四项要求：修改方案（含维度对接）/完整参数配置/
      差异对比/风险评估，参数增量有逐项估算。
- [ ] **AC2**（实施阶段）变体子类落变体层、包体零 diff；`--flow-exit-gnn` 默认关闭时全链路
      行为与现行 bit 兼容。
- [ ] **AC3**（实施阶段）新口径冒烟通过：出口消息传递生效、出口投影零初始化（首个 batch
      v≡0 可验）、stage2 trainable 打印 = 903,458 + 出口增量。
- [ ] **AC4**（实施阶段）训练 Run（如获批）在 experiment.md 登记，产物落本任务新输出目录。

## 边界（本任务不做）

- 不改采样确定性、不改损失函数、不动入口混合器与膨胀卷积块。
- 不在本任务内决定是否 5-fold / 是否延长训练预算（另属拍板项）。
- 第一阶段（本轮）不写任何代码。

## 子研究路由（1.0b / 1.0c）

- **分类结果**：`study1` —— EEG 代理脑（Study I）模型结构变体设计（θ 候选产线迭代）。
- **依据**：交付物 = 代理生成模型的结构变体设计与（获批后）训练，无 EC 推断成分。
- **新路径/方法检测（1.0c）**：`已有路径：latent-flow（OC-LatentFlow，models/latent-flow-v1/）`
  —— 注册表既有路径内的变体级结构迭代（与 mexhat→dog、graph_k 等变体 knob 同类）；
  GNN 空间块先例见 gpfm-mamba-v5（另一模型线，跨块共享 A_eff），不构成新路径。
- **research 产物**：`无`（设计输入 = 本仓库代码与 w500 任务实测，见 design.md 引用）。
