# fold0 虚拟扰动→EC有效路径推断流程设计验证（v4a-fmt 主模型）

## Goal

在 v4a-fmt（非 adaLN）fold0 固定代理脑上设计并验证『虚拟扰动→推断 EC 有效路径』实验流程：定义扰动算子/响应度量/阈值/临界强度搜索四要素并在 fold0 验证流程可跑通、能推断出 EC 有效路径。

## Requirements

- **R1 θ\* 冻结（只读）**：主模型 = `outputs/gpfm-mamba-fmt-f0/v4afmt-main-01/checkpoints/gpfm-mamba-v4a-fmt-best.pt`（best ep45，config_hash 05d9bb875a2c，EMA 权重）。本任务零训练、零微调、零模型侧改动。
- **R2 四要素设计落地**：扰动算子 P(X,A,r)、响应度量 D、响应阈值 τ（数据驱动独立模块，区分自然波动与真实响应）、临界强度搜索（r 网格）全部**预注册**（design.md 定稿 + experiment.md 首个 Run 块登记），备选方案与口径变更纪律显式。
- **R3 readout 三件套按已登记约束实现**（evaluation-calibers.md「Study II 扰动 EC 对比口径」）：主 = 低剂量斜率（r 网格低密度段加密），次 = AUC，降级 = r\*（右删失标注 + τ 阈值敏感性登记）。
- **R4 CRN 确定性契约**：baseline 与 perturbed 采样共享显式 eps/eta（`EulerSampler.sample` 已支持）；r=0 输出逐位恒等由单测锁定。
- **R5 fold0 端到端**：fold0 test 10 例（subject IDs [10,11,12,17,24,28,30,44,61,69]，与 fmt 任务逐 ID 一致）跑通，产出逐受试者 62×62 EC_pert 矩阵（行=源、列=靶）+ 有效路径集（r\*≤r_max 的有向边 A→B，A≠B）。
- **R6 结构收敛性对比**：EC_pert vs observational directed-connectivity references（GC/gPDC，fmt 同源 250→125 Hz 重算、沿用 ridge λ_rel=1e-2 / p=8 定标口径），主指标 = 逐被试**距离偏相关 Spearman ρ**（3782 有向边，通道三维欧氏距离偏出），检验 = 单样本 t（n=10）× Holm（跨参考法）+ 1000×受试者 bootstrap CI（seed 2026）；经 `models/baseline-ec-v1/compare.py::load_ec_pert` 契约进入比较器（slope/auc 天然大=强免翻转，rstar 调用方先翻转）。
- **R7 诊断证据链**（流程验证核心，非仅终点对比）：噪声地板 null 分布与 τ 标定、剂量-响应曲线形态（单调性/低剂量线性）、对角自响应 vs 跨通道响应量级、响应-通道距离衰减；可视化走 common/visualizer（visualization.md）。
- **R8 术语与口径纪律**：GC/gPDC 一律称 observational directed-connectivity references；结论表述为结构收敛性验证，不得称 ground-truth / 恢复真实因果。本任务参考侧 GC 为 **125 Hz 真实数据**计算，不触 fmt 评估侧「250 Hz genGC vs real GC 阻塞」口径（二者为不同对象，须在 experiment.md 显式区分登记）。

## Constraints

- **θ\* 定位（已登记偏离，须 1.4 用户确认）**：θ\* 为**前置验证级**（fold0·单 seed·指标1/3 已过、指标2 人工判待拍板、5-fold 未启动）。用户 2026-10-17 指示以该模型进入 Study II 流程验证；本任务**不宣称代理脑终局合格性**，扰动结论均为「流程验证·单模型·单折」性质。指标2 人工判与 5-fold 主实验独立推进，不构成本任务前置。
- **路径隔离**（path-isolation.md）：新变体目录 `models/perturb-ec-v1/`；只读复用 `models/gpfm-mamba-v1/`（sample/data/model/gpr_prior）、`models/gpfm-mamba-v4a-fmt/`（config 契约与 splits manifest）、`models/baseline-ec-v1/`（ec_refs 叶子函数）；outputs 根 `/root/npi/outputs/perturb-ec-f0/`；禁写旧任务 outputs 根、旧变体 splits、数据集本体。
- **口径变更纪律**：K / N_w / r 网格档位 smoke+bench 后**最多调整一次**并登记；四要素口径一经主 Run 定版不得中途换（旧 Run 原口径不动）。
- **环境**：environment.md 本地契约（RTX 4080S 32GB、有效 16 核 cgroup、BLAS 护栏 `_blas_guard` 惯例、nohup 长跑）；未记录环境场景先问用户。

## Acceptance Criteria

- [ ] AC1 四要素预注册完成：design.md 定稿 + experiment.md 首个 Run 块登记全部口径（含实现口径声明：seeds/通道顺序/对角处理/方向约定）
- [ ] AC2 CRN 契约单测通过：r=0 输出逐位恒等；扰动仅改源通道输入；显式 eps/eta 确定性复现（同输入两次调用逐位相同）
- [ ] AC3 端到端主 Run 在 fold0 test 10 例完成，产物完整落盘（窗级 D 曲线 npz、受试者级 EC 三 readout npz、load_ec_pert 契约 npz、有效路径集、诊断图）
- [ ] AC4 噪声地板 null 分布与 τ 标定完成（95% 分位主口径 + 90%/99% 敏感性），响应-地板分离度与剂量-响应形态诊断登记
- [ ] AC5 比较器完整运行：主指标偏相关 ρ + 未偏 ρ 上下文 + t×Holm + bootstrap CI + **split-half 天花板 fmt 10 例重算**（必做，review 对齐预登记口径）与 ρ̄/ceiling 比值登记
- [ ] AC6 结论判定明确：主 readout（低剂量斜率）对两参考法结构收敛性是否达初步证据门槛（ρ̄>0 且 Holm 校正 p<0.05）**如实判定**；未达门槛则交付缺口诊断与下一步选项，不宣称成功
- [ ] AC7 experiment.md Run 块完整回填；Study II 四要素首版流程规范沉淀列入 3.3 spec 更新候选

## 子研究路由（1.0b / 1.0c）

<!-- trellis-research-routing -->

- **分类结果**：`study2`（EC 推断算法，框架 Study II）
- **依据**：任务目标 = 在冻结代理脑上设计并验证「虚拟扰动→临界响应→连接推断」流程，即框架 Study II 的四要素首次定义与验证；代理脑仅作冻结输入，无训练/选型内容。
- **新路径/方法检测（1.0c）**：`已有路径：.trellis/research/context.md 技术路线表 Study II 行（Critical Perturbation Connectivity Inference，状态「四要素均未定义」——本任务即该行首次落地；扰动-响应-临界强度框架为 research-direction.md 既有设计，非新引入方法）`
- **research 产物**：`无`（参考论文 Luo et al. 的适配边界已由框架文档 §8 界定——参考文献仅给「代理脑 + perturb-and-record」，扰动协议/τ/搜索均为本研究自主设计；无需新 adaptation study）
