# edra-fmt 250Hz 原始全长双变量 GC 真实参照（48例·不切窗·仅真实数据）

## Goal

在 edra-fmt 数据集上以 **250 Hz 原始采样率**（不降采、不切窗、60001 点全长连续）计算全部 48 例受试者的成对双变量 GC 观测参照矩阵，作为既有 125 Hz 参照（bivar-gc-ec-v1，任务 10-17-bivar-gc-ec-fmt）之外的原始采样率口径参照。仅真实数据：无生成侧、无模型依赖、无跨口径对比交付。

## Background

- 用户 2026-10-18 四项拍板：立项新任务 / 双变量逐对 GC / 48 例全量 / 不做对比交付。
- 动机语境：Study I 评估侧 GC 在 128 点窗式 future 段上计算（段长 × p 网格构成可分辨时标硬边界、名义 n 膨胀致选阶失真——2026-10-18 讨论登记）；本任务提供 250 Hz 原始全长、不切窗的真实数据参照。**本任务只出参照本体**，窗式 vs 全长的对比分析不在范围内。
- 方法登记：双变量 GC 观测参考为 Study II 已有路径（`.trellis/research/context.md` 技术路线表 Study II 行），本任务 = 同路径换采样率口径，非新路径、无 research 触发。

## Requirements

### R1 数据口径（250 Hz 原始全长，与 v1 唯一数据差异 = 不降采）

- 输入 `/root/autodl-tmp/edra_processed_fmt/` 48 例全量（manifest 全 status=ok；契约 `.trellis/spec/data/edra-fmt.md`），数据集只读。
- 字段断言同 v1：`filter=="1-40(release)"`、`pipeline=="EDRA_release_1-40_250Hz"`、`fs==250`、`truncated==False`、shape (62, 60001)。
- **无 resample 步骤**：250 Hz 原样进入拟合（v1 为 `resample_poly(x,1,2)` → 125 Hz）。
- z-score 逐通道全记录（ddof=0，σ 下限 1e-12）；float64 复原计算、float32 落盘（复现惯例同）。
- 主拟合布局 = 每例**单段全长连续**（不切窗），行集 `row_from=P_MAX` 对齐，n_eff = 60001 − P_MAX。

### R2 双变量 GC 口径（与 v1 同式，P_MAX 随 fs 翻倍）

- 方向约定全仓库统一：EC 矩阵**行=源 source，列=靶 target**，对角=0；有向边展平非三角（3782）。
- 逐无序对 {A,B}：全通道滞后 Gram 一次构建 + slice 二通道子问题逐阶求解 + BIC 选阶 + `ln(SSR_r/SSR_u)` 双向读出；OLS（非 ridge），**offdiag≥0 作为实现正确性断言**（负值即失败）。
- **P_MAX = 48**：物理时长对齐 v1（125 Hz@24 ≡ 250 Hz@48，最大滞后均 192 ms）——保「滞后范围」不变量，使两条参照的模型阶物理含义可比；候选 p ∈ {1..48}。
- **BIC 顶格风险预告**：惩罚项 4·p·ln(n_eff)/n_eff 在 n_eff≈6 万/例下 ≈0.035 量级，p\*=48 顶格为预期结局——**如实登记，不做调参干预**；网格扩展沿用 v1 遗留未决决策，不在本任务内处理。
- 核心实现复用：`bivar_gc.py` 自 `models/bivar-gc-ec-v1/` **只读 import**（纯 numpy 函数、p_max 全链参数化，v1 单测已锁定其正确性）；`split_half.py` 仅复用 `build_halves/half_ec/offdiag_flat` 等可传参函数——`ceiling_for_subject` 无 `seg` 参数且不透传（审查 Mandate A/B 一致核实），**不使用**，split 阶段 v2 本地编排；`data_io` 复刻 250 Hz 版（v1 版内嵌降采且绑定 v1 config）。

### R3 split-half 噪声天花板（参照自身质量登记，非跨口径对比）

- 段长 500 样本（2 s @ 250 Hz），其余段式同 v1：奇偶分半、尾段丢弃、段内设计不跨段、每半独立同口径（含逐对 BIC）。
- 交付 = 逐例 offdiag 3782 边 Spearman ρ → 天花板 mean±sd（ddof=1，48 例）。

### R4 范围边界（用户 2026-10-18 拍板）

- 仅真实数据：无生成侧、无 θ\*、无任何模型 checkpoint 依赖。
- **不做对比交付**：不与 125 Hz 参照、不与 Study I 窗式 GC 出对比表/图；口径隔离，产物与文档显式区分，**绝不写入 v1 目录与 `outputs/bivar-gc-ec-fmt/` 旧产物**。

### R5 运行纪律

- path-isolation：新变体目录 `models/bivar-gc-ec-v2/`（config 经唯一名 `bgec2_config` 装载，不占 `config` 模块名）；产物根 `/root/npi/outputs/bivar-gc-ec-fmt-250/`。
- run_id 序列：`smoke-01`（2 例）→ `refs-01`（48 例主 Run）→ `sh-01`（split-half）；任何 Run 先在 experiment.md 登记目的+配置，跑完同会话回填结果+结论。
- BLAS 护栏五变量 setdefault（numpy 导入前）+ 16 进程并行 + worker 单线程（environment.md 16 核 cgroup 契约）；长跑 nohup。
- 图经 `common/visualizer.py` 正典路线；既有图型不覆盖时纯增量扩展并在 check 白名单登记。

### R6 时间反转检验（GC 方向读数的有效性诊断，2026-10-18 用户追加）

- 对每例 z-score 后全长序列沿时间翻转（`X_rev = X_z[:, ::-1]`），以**完全相同口径**（不降采、不切窗、row_from=P_MAX=48、逐对 BIC p∈{1..48}、ln(SSR) 双向、OLS）计算反向矩阵 GC_rev（run `rev-01`）。
- 预登记对比指标（逐例 → 组级 mean±sd）：
  1. `edge_fwd_rev`：offdiag 3782 边 Spearman ρ(GC_fwd, GC_rev)——≈1 ⇒ 读数被对称滞后耦合主导（共同驱动/混合类伪影嫌疑；纯零滞后瞬时混合在本 readout 下不产生方向读数）；显著 <1 ⇒ 读数含时间方向成分；ρ 与幅值指标③联合判读（近零幅值边上 ρ 由数值噪声支配时不可单独解读）；
  2. `edge_fwd_mirrorrev`：ρ(GC_fwd, GC_revᵀ)——镜像对应检验（反转后方向互换假设）；
  3. 幅值：median offdiag GC_rev vs GC_fwd 及比值；
  4. 方向不对称度：mean|GC(A→B) − GC(B→A)|，fwd vs rev。
- **判读框架（预登记，非硬门限）**：二阶平稳高斯过程时间反演保持二阶结构不变、反向 VAR 系数不同 ⇒ GC_rev 与 GC_fwd 既不相等也不必归零——本检验为描述性指纹登记，不设"GC_rev≈0 才通过"式门限；结论落 experiment.md，供 EC 推断线引用。
- BIC 顶格预期与主 Run 相同——依据：惩罚 ≈0.035 极小 + n_eff 巨大 + 125 Hz 实测 100% 顶格（反转可改变 logdet 项，但量级对比不变）；p\* 分布以实测登记为准。

## Acceptance Criteria

- [ ] AC1 单测全绿（rng 2026）：方向单测（合成二通道 VAR，仅 A→B 耦合）、独立性零基线、offdiag 非负断言、Gram-slice↔naive 等价（rtol=1e-10，经 v2 import 路径验证）、同输入逐位复现、装载断言（fs≠250 / 长度≠60001 / truncated → 失败）
- [ ] AC2 `smoke-01`（2 例）通过：逐例 Gram 条件数登记、GC 量级人工核对（与 v1 同量级即可，不设硬阈值）
- [ ] AC3 `refs-01`：48 例 `refs_all.npz`（(48,62,62) float32，行=源/列=靶/对角=0）+ `refs_summary.json`（fs=250、P_MAX=48、n_eff、逐对 BIC 阶数分布**如实登记**）落盘；非负断言全通过
- [ ] AC4 `sh-01`：split-half 天花板 mean±sd 落盘
- [ ] AC5 experiment.md Run 块完整（目的/配置/结果/结论，无空块）；`evaluation-calibers.md` 增补 250 Hz 口径条目；v1 目录与旧产物零写入
- [ ] AC6 确定性：同输入重跑输出逐位一致
- [ ] AC7 `rev-01`：48 例 GC_rev 落盘（同口径非负断言通过）+ 四组预登记指标计算齐全 + 合成数据反转改变断言（单测 8）通过；判读按预登记框架写 experiment.md 结论

## 子研究路由（1.0b / 1.0c）

- **分类结果**：`study2` —— 双变量 GC 观测参考是 Study II EC 推断线的观测参照（context.md 技术路线表 Study II 行明载）
- **依据**：任务为真实数据 EC 参照计算，服务 EC 推断算法线，不涉及代理脑训练/选型/验收
- **新路径/方法检测（1.0c）**：`已有路径：双变量 GC 观测参考（bivar-gc-ec-v1 首版，125 Hz）` —— 本任务为同路径换采样率口径（250 Hz 原始全长），无新方法
- **research 产物**：`无`
