# symw EC_pert × GC-EC 逐被试并排热力图 + 组均值对比可视化（秩归一同尺度 + 绝对值补充，nature-figure 三门 QA）

## 子研究路由（1.0b / 1.0c）

<!-- trellis-research-routing -->

- **分类结果**：`study2`（EC 推断算法，框架 Study II——已归档 EC 结果的对比可视化）
- **依据**：任务主体 = EC_pert 与 GC-EC 两类连接矩阵的可视化对比交付；不训练、不重算 EC（纯离线读档绘图）。
- **新路径/方法检测（1.0c）**：`已有路径：EC 结果可视化（10-24-exitgnn-ec-viz 已归档先例，nature-figure 契约三门）`——无新方法，research adaptation 不触发。
- **research 产物**：`无`
- **路由备注**：图结论为描述性（单 seed、provisional 代理脑），不构成代理脑资格判定。

## Goal

对 `10-25-symw-ec-unbiased` 归档的 EC 分析结果做系统可视化：8 被试（MDD s1/5/7/10、HC
s21/25/27/30）各一张 **EC_pert（conn v4）× GC-EC（v1 refs）并排热力图**（同 cmap、同色标
[0,1] = 矩阵内百分位秩、对角掩蔽、逐被试 Spearman ρ 注记、原始数值范围注标题），加**组均值
两版**（rank 版与绝对值补充版），输出 PNG(300dpi)+PDF 至 `outputs/10-25-symw-ec-viz/`，
过 nature-figure 三门 QA，并给出个体间/组水平可观察模式的文字总结。

## 背景与用户指令（2026-10-25，三项）

1. **个体层面**：每被试两类矩阵热力图并排、同 colormap 同色标范围便于对比方法差异；
2. **组层面**：组平均 EC_pert 与组平均 GC-EC 并排、与个体图一致配色；
3. **格式**：标注维度/节点标签（如有）/colorbar/数值范围；标题含被试 ID 与矩阵类型；
   说明数据文件路径与命名规则；输出图像文件 + 可观察模式简述。

## 关键决策（色标策略，问询未应答按最佳判断，透明在档）

- **实测**：EC_pert(v4) 值域 [0.029, 3.90]（全正，~6% 钳上界）；GC-EC(v1) 值域 [0, 0.29]
  （p99≈0.09）——**量级差 ~13×**，字面共用绝对色标会使 GC 面板仅占色标 ~7%（不可读）。
- **决策**：主图采用**秩归一同尺度**（两面板同 cmap、同 [0,1] 色标，颜色 = offdiag 矩阵内
  百分位秩；对角双侧掩蔽）——满足"同 cmap + 同色标范围"的可对比意图且两法均可读；原始
  数值范围注于各面板标题。**组级另附绝对值补充版**（各测自身稳健范围 vmax=p99.5(offdiag
  pooled)，同 cmap）保量级诚实。用户 AskUserQuestion 未应答，按推荐项执行（处置镜像
  10-24-exitgnn-ec-unbiased consent 先例：完整工作令 + 全部可逆 + 证据在档）。

## Requirements

- **R1（数据源与被试清单，只读）**：
  - EC_pert = `outputs/10-25-symw-ec-unbiased/recompute_v4/subject_{0..7}.json` 的
    `connectivity_matrix`（128×128，conn v4 交付口径；`subject_file` = s{1,5,7,10,21,25,27,30}，
    `group` MDD/HC）；
  - GC-EC = `refs/granger_bivariate_spectral/mild40-trial500-zscore-fs250-fixed8-f1-45-favg/
    refs_s{sid}.npz` 的 `ec`（mat[i,j] = GC(i→j)，1–45 Hz 固定阶 8 faverage；v2 refs 存在
    但本图以 v1 为 GC-EC 主参考——与 headline 相关口径一致，交付说明中注明）；
  - 方向约定：**行 = 源，列 = 靶**（spec perturb-ec-pipeline），两面板同约定；
  - 通道标签：`/root/autodl-tmp/Mild Depression/geometry_mild.pt` 的 `channel_names`
    （E1–E128，电极编号无解剖分区）——稀疏刻度每 16 通道（E1/E17/…/E113）+ 维度注记
    128×128；
  - ρ 注记数值：`outputs/10-25-symw-ec-unbiased/recompute_v4/gccorr_dual_summary.json`
    v1 块 per-subject `spearman_raw_aligned`。
- **R2（个体图 × 8）**：每被试一张 1×2 并排（左 EC_pert、右 GC-EC）：同 cmap（感知均匀
  序，如 viridis/magma 族）、同色标 [0,1] = offdiag 百分位秩、双侧对角掩蔽（白/细纹）；
  面板标题 = `s{id} ({group}) — EC_pert (conn v4) [min–max]` / `— GC-EC (1–45Hz fixed8)
  [min–max]`；单一共享 colorbar 注 "percentile rank (offdiag, within matrix)"；轴刻度
  E 系稀疏标签；图内注记该被试 Spearman ρ(aligned)。输出
  `outputs/10-25-symw-ec-viz/subject_pairs/subj_s{id}_{group}.{png,pdf}`（PNG 300dpi）。
- **R3（组级图 × 2 版）**：`group_means_rank`：8 被试秩图均值的 1×2 并排（同 R2 配色约定，
  注 group_mean ρ）；`group_means_abs`（补充）：8 被试**原始值**均值 1×2 并排，各测自身
  vmax = pooled offdiag p99.5、同 cmap、各自 colorbar 标注真实数值范围——量级诚实版。
  输出 `outputs/10-25-symw-ec-viz/group_means_*.{png,pdf}`。
- **R4（绘图脚本）**：`models/latent-flow-v1/viz_symw_ec_pairs.py`（新文件；BLAS 守卫 +
  OMP 前缀照房例；纯读档绘图零模型依赖；CLI 无参默认全量、`--subjects` 可子集冒烟）；
  py_compile 过。
- **R5（nature-figure 三门 QA）**：后端 = python（已存默认，独占）；按技能契约加载 core
  contract/stance + python fragment + qa-contract；`validate_figure.py` 过源；1×2 等宽面板
  `audit_panel_alignment`（容差 1.5pt，alignment JSON 随档）；导出 PDF 后
  `audit_figure_collisions`（FIX BEFORE DELIVERY 必须为零）；渲染 PNG 交 visual-judge
  审（每图 pass/fail + 问题证据，修复回环）。
- **R6（输出 README + 总结）**：`outputs/10-25-symw-ec-viz/README.md`（数据路径与命名
  规则、被试清单、色标策略与理由、复现命令、QA 结果）；交付消息含**个体间与组水平可观察
  模式的文字总结**（含与已知数值结论的一致性对照：s25 ρ −0.154 最负、s5/s10/s27 高端、
  GC 稀疏尖峰 vs EC 连续分布等）。
- **R7（路径隔离）**：改动仅 `models/latent-flow-v1/viz_symw_ec_pairs.py`（新）+
  `outputs/10-25-symw-ec-viz/` 新树 + 任务目录；旧任务输出树/refs 只读；latent_flow 包体
  零改动；预存在脏路径不擅动；**不占 GPU**（并行训练会话在跑，本任务纯 CPU）。

## Acceptance Criteria

- [ ] 8 张个体图 + 2 张组级图（rank + abs）落档，命名合规，PNG 300dpi + PDF 齐。
- [ ] 同被试两面板同 cmap 且共享同一 colorbar/范围（[0,1] 秩）；标题含 ID/类型/原始范围；
      对角掩蔽；稀疏通道刻度 + 128×128 维度注记；ρ 注记在档。
- [ ] 组级 abs 补充版各 colorbar 标注真实数值范围（量级差可见）。
- [ ] 三门 QA：validate / panel-alignment（JSON 在档）/ collision（PDF 审零 FIX）/
      visual-judge（全部图 pass 或问题修复后复判）。
- [ ] README 含数据路径、命名规则、被试清单、色标策略理由、复现命令。
- [ ] 路径隔离核查干净（git status 仅本任务路径 + 预存在脏路径原样）。
- [ ] 收尾三件套（2.2 check / 3.3 spec-update 判定 / 3.4 commit 仅本任务路径）。

## 约束

- **轻量任务声明**：PRD-only（design 不适用——规格已钉到图级；implement 并入本 PRD R1–R7
  执行序：R4 脚本 → 冒烟 → 全量 → QA 三门 → README/总结 → 收尾）。
- 环境：npi 本机 CPU（matplotlib；**GPU 禁占**——并行训练会话在跑）；python=
  /root/miniconda3/bin/python；BLAS 守卫照房例。
- 数据：全部只读归档档（见 R1 路径）；不重算 EC、不触 refs。
