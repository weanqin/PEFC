# EC_pert v4 可视化（exit-gnn 无偏 EC 三图：逐被试热图 / 组均值×双 GC 并置 / MDD-HC 对照）

## 子研究路由（1.0b / 1.0c）

<!-- trellis-research-routing -->

- **分类结果**：`study2`（EC 推断产物可视化——归档 EC 任务的读图交付，不训练不推断重算）
- **依据**：纯可视化任务，消费 Study II 产物（EC_pert v4 矩阵 + GC refs）；无新方法无新路径。
- **新路径/方法检测（1.0c）**：`不涉及`（可视化路径 = 房内 visualization.md spec + nature-figure 技能既有契约）
- **research 产物**：`无`

## Goal

把归档任务 10-24-exitgnn-ec-unbiased 的 EC_pert v4（conn = −(ln σ*+0.5·ln2πe)+5/7，8 例
128×128 有向，行=源/列=靶）与双 GC 参考做成三张 nature-figure 风格图，支撑「该配置下
EC_pert 结构读数」的人工判读：个体结构、与观测 GC 的结构对照、组间（MDD/HC）形态。

## Requirements

- **R1 数据（只读）**：`outputs/10-24-exitgnn-ec-unbiased/recompute_v4/subject_{0..7}.json`
  （v4 conn 矩阵 + group 标签）；GC refs 两档各 8 例折叠（v1 fixed8/1–45 Hz、
  v2 AIC4-25/1–125 Hz，`refs/granger_bivariate_spectral/…/refs_s{sid}.npz`）。
- **R2 图 1 `ec_subjects_v4.png`**：2×4 逐被试热图（行 1 MDD s1,5,7,10 / 行 2 HC
  s21,25,27,30），**共享色标**（v4 值域），行=源/列=靶注记，sid+组标签。
- **R3 图 2 `ec_group_vs_gc.png`**：三联并置——组均值 EC_pert v4 × 组均值 GC v1 ×
  组均值 GC v2（各自对称色标注记，同向 行=源/列=靶；量纲不同不共标，显式注明）。
- **R4 图 3 `ec_mdd_hc.png`**：MDD 组均值 vs HC 组均值 + 差异面板（MDD−HC，发散色标）。
- **R5 风格与 QA**：nature-figure 契约（Python matplotlib 后端，已存偏好）；导出 PDF
  伴档并跑 collision audit + validate_figure；图路径登记 experiment.md Run 块。
- **R6 路径隔离**：脚本落 `models/latent-flow-v1/`（变体层，viz_ec_pert.py 单文件）；
  输出仅 `outputs/10-24-exitgnn-ec-viz/`；归档任务输出树与 refs 零写入。

## Acceptance Criteria

- [x] 三 PNG（+伴档 PDF）落 `outputs/10-24-exitgnn-ec-viz/`，逐图可读：色标/轴注记/
      组标签/sid 齐全，方向约定（行=源/列=靶）在图注或轴标明。
- [x] validate_figure.py 与 audit_figure_collisions.py 通过（preflight 18 pass/0 fail；
      alignment gate 三图 strict 通过；collision 三图 0 fail/0 warn——图 2 初版 2 FAIL
      经面板字母并入标题修复后 PASS，图 3 初版 visual-judge fail 经数据裁定为真实结构
      非渲染缺陷，诚实判读注记已入图）。
- [x] experiment.md Run 块（目的-配置-结果-结论）+ 图路径登记；研究记忆不涉及（无新事实，
      可视化读数在 Run 块登记）。
- [x] commit 仅本任务路径（脚本 + outputs 为 gitignore + 任务目录）。

## 读取的数值锚点（供图判读，不构成新分析）

- v4 值域 [+0.029, +3.90]（零点 0.4943 出网格 ⇒ 全正）；逐例 offdiag 均值 +0.63~+1.14
  （Run 003 在档）；靶主效应 0.907 / 跨被试 ρ 0.655；GC 对照 v1 Pearson +0.149 / v2
  −0.010（描述性）。
