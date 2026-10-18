# 虚拟扰动范式×响应度量低成本搜索（gpfm-mamba-v1-fmt θ*·fold0）

## Goal

在冻结 gpfm-mamba-v1-fmt fold0 θ* 上，用固定扰动强度 + 单次 CRN 批采样的**低成本两阶段搜索**，选出：输入端效果最好的虚拟扰动范式 P\*、输出端效果最好的响应度量 M\*。背景 = 前次扰动实验（10-17-perturb-ec-f0，v4a-fmt θ\*）表明模型对增益调制不敏感（跨通道传导弥散）；本任务**不追求复杂训练或大规模验证**，只做候选方案的快速筛选，为后续 Study II 流程重跑提供「算子 P + 度量 D」两要素的筛选依据。

## Requirements

- **R1 θ\* 冻结（只读）**：主模型 = `outputs/gpfm-mamba-fmt-f0/v1fmt-main-01/checkpoints/gpfm-mamba-v1-fmt-best.pt`（EMA，config_hash 校验；fmt 范式 fs=250 Hz / past 384 / future 128 / p=25 / 无生成侧低通）。本任务零训练、零微调、零模型侧改动。
- **R2 候选范围明确**：design.md 预注册候选清单——输入端虚拟扰动范式候选（增益/加性/偏置/瞬态/损毁式/带限选择性等类别，每类一个代表实现，含定义式与固定强度 r0）；输出端响应度量候选（时域/谱域/分带功率/包络/相关结构/时间形态等类别，每类一个代表实现）。各候选附一句话适用理由。
- **R3 评估与选择标准明确**：预注册 scorecard——敏感性（相对 CRN 噪声地板的分离度）、特异性（自响应/跨响应对比 + 空间衰减登记）、稳定性（跨受试者 split-half 秩相关）、可解释性（定性注记）、执行成本（实测耗时）。给出「效果最好」的显式判定规则与全候选不过线时的**如实阴性路径**（报最佳可得 + 缺口诊断，不宣称成功）。
- **R4 两阶段实验设计**：阶段一固定输入端扰动形式（预注册探针），扫全部输出端度量候选，选出 M\*；阶段二固定 M\*，扫全部输入端范式候选，选出 P\*。两阶段候选集、gates、选择规则一经主 Run 定版不得中途换。
- **R5 速度约束**：每候选只测**一个固定扰动强度 r0**（预注册，逐算子以自身自然单位标定）、每窗**一次**批采样推理（baseline 与全部变体并入同一 CRN 批）；固定小子集（≈3 受试者 × 8 窗 × K=16 样本）不动 5-fold、不做 EC 比较、不跑参考连接组。目标每阶段 GPU 主 Run 实测 ≤ 30 min（超出照登原因）。
- **R6 记录方式**：experiment.md 两阶段各一个 Run 块回填（候选表 / 参数 / scorecard 汇总表 / 筛选结论 + 一句话依据），明细落 `outputs/perturb-search-f0/` npz+csv；保持简洁，只登记汇总与结论级内容。
- **R7 简单性约束（全局）**：不引入不必要的复杂步骤——不做多种子、不做网格搜索、不做统计检验套餐；单测只锁 CRN/确定性契约与算子只动源通道三条。

## Constraints

- **θ\* 定位（已登记偏离，须 1.4 用户确认）**：v1-fmt 仍为前置验证级（fold0 · 单 seed · 指标1/3 过、指标2 人工判与 5-fold 未启动），本任务是用户 2026-10-17 定论「当前方法 EC 完全不可用、不在此 θ\* 上继续、优化留待后续任务」之后的既定后续：**先筛范式与度量，再谈重跑**；结论表述为「v1-fmt θ\* 上的筛选证据」，不宣称代理脑终局合格性。
- **路径隔离**（path-isolation.md）：新变体目录 `models/perturb-search-v1/`；只读复用 `models/gpfm-mamba-v1/`（sample/data 共享件）、`models/gpfm-mamba-v1-fmt/`（config + splits manifest）、`models/perturb-ec-v1/`（loader 配方 / CRN 距离叶子函数 / 增益算子）；outputs 根 `/root/npi/outputs/perturb-search-f0/`；禁写旧任务 outputs 根、旧变体目录、数据集本体。
- **口径纪律**：候选集 / r0 / gates / 探针选择在 design.md 定稿即预注册；主 Run 后最多一次「档位级」调整（如 K、窗数）并登记，候选定义不得中途改。CRN 契约沿 perturb-ec-pipeline.md（同窗一组 eps/eta 跨变体共享、r=0 逐位恒等、单次 sample() 调用纪律）。
- **滞后决策项不动**：fmt 条件 GC、比较器参考列接入、双变量 BIC 网格扩展、r\*→连接强度映射——本任务均不触碰。
- **环境**：environment.md 本地契约（RTX 4080S 32GB、有效 16 核 cgroup、BLAS 护栏惯例、nohup 长跑）；未记录环境场景先问用户。

## Acceptance Criteria

- [ ] AC1 预注册完成：design.md 定稿（候选清单含定义式与 r0、scorecard gates 与选择规则、两阶段流程、探针选择及备选）+ experiment.md 首个 Run 块登记实现口径（受试者/窗/K/seed/批形状/通道顺序）
- [ ] AC2 契约单测通过：r=0 输出逐位恒等；扰动仅改源通道输入（逐通道断言）；显式 eps/eta 下同输入两次调用逐位相同
- [ ] AC3 阶段一完成：全部度量候选 scorecard（双探针分别登记）落盘，M\* 显式选出并附判定依据；无候选过线时如实阴性 + 缺口诊断
- [ ] AC4 阶段二完成：全部范式候选在 M\* 下 scorecard 落盘，P\* 显式选出并附判定依据；无候选过线时如实阴性 + 缺口诊断
- [ ] AC5 记录完整简洁：两阶段 Run 块回填 + outputs npz/csv 落盘；每阶段实测耗时照登
- [ ] AC6 速度与简单性兑现：零训练；每阶段主 Run GPU 实测 ≤ 30 min 或照登超出原因；未引入多种子/网格/检验套餐
- [ ] AC7 收尾：P\*/M\*（或阴性结论）登记 progress.md；四要素口径更新（算子 P / 度量 D 两行）列入 3.3 spec 更新候选

## 子研究路由（1.0b / 1.0c）

<!-- trellis-research-routing -->

- **分类结果**：`study2`（EC 推断算法，框架 Study II；用户指定归属 Study 2）
- **依据**：任务目标 = 在冻结代理脑上搜索「虚拟扰动算子 P + 响应度量 D」两要素的最优组合，属框架 Study II 四要素的迭代筛选；代理脑仅作冻结输入，无训练/选型内容（非 study1）。
- **新路径/方法检测（1.0c）**：`已有路径：.trellis/research/context.md 技术路线表 Study II 行（Critical Perturbation Connectivity Inference）——本任务为该行「扰动算子 / 响应度量」两要素的低成本候选搜索（用户 2026-10-17 定论「优化留待后续任务」的既定后续，方向由用户本轮指定），非新方法引入；扰动-响应-搜索框架为 research-direction.md 既有设计`
- **research 产物**：`无`
