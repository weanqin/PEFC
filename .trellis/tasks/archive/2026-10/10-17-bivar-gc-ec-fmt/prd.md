# edra-fmt 数据集成对双变量 GC 计算 EC

## Goal

在 edra-fmt 数据集（/root/autodl-tmp/edra_processed_fmt，48 例 HC24/MDD24，250 Hz release 1–40 Hz 新预处理范式）上以成对、双变量 GC 计算 EC：逐通道对 (A,B) 拟合二通道 VAR → GC A→B，不条件化于其余通道；产出受试者级 62×62 有向 EC 矩阵（行=源/列=靶）与 split-half 噪声天花板登记。定位为 observational directed-connectivity reference 的新口径变体——既有参考侧（baseline-ec-v1 / perturb-ec-v1 refs_fmt）主口径均为全模型条件 GC，双变量口径为首次主轨落地。

## 任务配置（用户 2026-10-17 指定，三项硬约束）

1. **数据集 = edra-fmt**：`/root/autodl-tmp/edra_processed_fmt/`，契约 `.trellis/spec/data/edra-fmt.md`（48 例全 manifest status=ok；250 Hz；release 自带 1–40 Hz 带通，**不重复滤波**；montage 与 edra 共用 62 导联同序）。
2. **计算目标 = EC**：受试者级 62×62 有向 EC 矩阵，方向约定沿项目惯例**行=源、列=靶**，对角不定义（62×61=3782 有向边）。
3. **GC 方法 = 成对、双变量（pairwise bivariate）**：逐通道对 (A,B) **仅用 A、B 两通道**数据拟合二变量 VAR，计算 GC(A→B)；**不条件化于其余 60 通道**。与既有全模型条件 GC（baseline-ec-v1 主口径：单一全模型 Schur 读出）为不同口径，须在产物与文档中显式区分，不得混用。

## 已拍板口径（用户 2026-10-17 第二轮，六项全定）

| 开放项 | 拍板结果 |
|---|---|
| 受试者范围 | **48 例全量**（manifest 全部 status=ok） |
| fs 口径 | **统一使用降采样后的频率 = 125 Hz**（沿 refs_fmt 惯例：`resample_poly(x,1,2)` 抗混叠降采 250→125 Hz） |
| 模型选择 | **逐对 BIC 选阶**（候选网格 design 定稿，默认对齐 baseline-ec-v1 惯例 p∈{1..24}） |
| 读出指标 | **时域 ln(SSR ratio)**：GC(A→B) = ln(SSR_r/SSR_u)，restricted = B 仅回归自身过去，unrestricted = B 回归 A、B 共同过去；ln 比对逐通道缩放不变（z-score 下口径自洽） |
| 代码落点 | **默认设置**：新变体目录 `models/bivar-gc-ec-v1/`（路径隔离），只读复用 `baseline-ec-v1/ec_refs.py` / `resample_io.py` 叶子函数；outputs 根 `/root/npi/outputs/bivar-gc-ec-fmt/` |
| 交付边界 | 第四轮修订（用户 2026-10-17）：**任务不做 EC_pert 与 GC 的比较**——交付止于双变量 GC 参考产物自身（48 例 EC 矩阵 + 逐对阶数分布 + split-half 天花板）；比较器接入与 fmt 条件 GC 列（10 例复用 / 48 例重算 / 时点）相关决策一律滞后 |

## 规划补默认确认记录（用户 2026-10-17 第三轮）

1. 「逐对 BIC 落在无序对、一次拟合双向读出」——**认可**。
2. fmt 条件 GC——**先不做，决策滞后**（上表交付边界行已按此修订）。
3. 双变量列 split-half 天花板——**认可**（维持）。
4. （第四轮追加）**任务不做 EC_pert 与 GC 的比较**——比较器接入整体移出范围（交付边界行已按此修订）。

## Requirements

- **R1 数据口径**：48 例全量，edra-fmt 契约加载（npz dataloader 兼容；`pipeline` 溯源字段核对入产物元数据）；不重复滤波；`resample_poly(x,1,2)` 降采 250→125 Hz；逐通道全记录 z-score（ddof=0，沿参考侧惯例）；fs 原值/目标值入元数据。
- **R2 双变量 GC 主口径**：逐**无序**通道对 {A,B} 拟合一次二变量 VAR（BIC 逐对选阶），同一次拟合同时读出 GC(A→B) 与 GC(B→A)（行=源/列=靶，对角不定义）；阶数分布逐对落表登记。方向语义以**已知方向合成 VAR 单测锁定**（沿 baseline-ec-v1 R4 惯例）。
- **R3 产物**：48 例逐受试者 62×62 EC 矩阵 npz（元数据：fs/降采参数/BIC 网格/逐对阶数/z-score 口径/受试者清单）+ 汇总表；可视化走 `common/visualizer.py`（`ec_matrix` 图型）。
- **R4 split-half 噪声天花板（双变量列，用户第三轮认可）**：双变量 GC 参考自身的噪声天花板登记（后续任何比较可直接取用作 ρ̄/ceiling 分母）——按既有 split-half 惯例（奇偶交错分段分两半、同口径含逐对 BIC）计算，登记 mean±sd。
- **R5 可复现与纪律**：BLAS 线程护栏惯例（`_blas_guard`）；同输入逐位复现；experiment.md Run 块先写后跑（四要素）；算力预算评估先行（48 例 × 1891 对 × BIC 网格的 OLS 规模，16 核并行 + nohup 长跑惯例）。
- **R6 术语纪律**：结果一律称 **observational directed-connectivity reference**，不称 ground-truth / 恢复真实因果；本任务为真实数据直接计算，不涉代理脑/扰动（无 θ* 依赖）；与 fmt 评估侧「250 Hz genGC vs real GC 阻塞」口径为不同对象，experiment.md 显式区分登记。

## Constraints

- **路径隔离**（path-isolation.md）：新变体目录 `models/bivar-gc-ec-v1/`；`models/baseline-ec-v1/` 与 `models/perturb-ec-v1/refs_fmt.py` 仅作**口径惯例参照**（公式复刻，无运行时复用点）；outputs 根 `/root/npi/outputs/bivar-gc-ec-fmt/`；禁写旧任务 outputs 根与数据集本体。
- **环境**（environment.md）：本地契约（有效 16 核 cgroup、BLAS 护栏、nohup 长跑）；依赖现状 numpy/scipy 已装，**无新依赖预期**——若需新依赖先问用户。
- **口径变更纪律**：BIC 网格与主口径一经主 Run 定版不得中途换（旧 Run 原口径不动）。

## Acceptance Criteria

- [ ] AC1 数据按 edra-fmt 契约加载 48 例全量：125 Hz 抗混叠降采 + 逐通道 z-score + 溯源字段（pipeline/fs 原值与目标值/受试者清单）入产物元数据，无重复滤波
- [ ] AC2 双变量 GC 方向语义单测通过（已知方向合成 VAR：A→B / B→A 区分；对角处理与「一次拟合双向读出」实现口径登记）
- [ ] AC3 逐对 BIC 选阶完成：48 例 × 1891 对阶数分布落表；主 Run 产出 48 例 62×62 EC 矩阵（行=源/列=靶，ln(SSR) readout）+ 元数据完整落盘
- [ ] AC4 可复现兑现：同输入逐位复现；BLAS 护栏生效；experiment.md Run 块先写后跑
- [ ] AC5 双变量列 split-half 天花板完成并登记（mean±sd）
- [ ] AC6 口径区分与术语纪律：双变量 vs 全模型条件 GC 差异显式登记；observational reference 表述合规；不做 EC_pert 比较、条件 GC 滞后状态登记
- [ ] AC7 experiment.md Run 块完整回填；spec 更新候选（如双变量口径入 evaluation-calibers）列入 3.3

## 子研究路由（1.0b / 1.0c）

<!-- trellis-research-routing -->

- **分类结果**：`study2`（EC 推断算法，框架 Study II）
- **依据**：任务目标 = 在真实数据上直接以 GC 计算 EC（observational directed-connectivity reference 的新口径变体），无代理脑训练/选型内容；与 baseline-ec-v1（scope=study2）同类定位。
- **新路径/方法检测（1.0c）**：`已有路径：GC 观测参考（.trellis/research/context.md 技术路线表 Study II 参考侧）——本任务为该路径的双变量口径变体；成对双变量 GC 为经典标准方法，非新引入方法族，无需 adaptation study`
- **research 产物**：`无`
