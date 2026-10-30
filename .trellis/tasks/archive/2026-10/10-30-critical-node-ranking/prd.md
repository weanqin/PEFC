# 关键节点排序实验（噪声注入→全局影响→128通道排序→三σ稳健性，双线五折）

## Goal

Study II 新实验：**提出并验证「关键节点」（critical node）概念**——与 EC_pert 推理同型的虚拟扰动范式，
但从边级（源→靶连接）改为**节点级（源→全局影响）**。对源通道历史段注入固定方差高斯噪声，用其余
127 个通道未来段 MSE 之和度量该源通道的全局影响，对全部 128 个通道排序得关键节点排序；在三档
噪声方差（σ=0.3/0.5/0.7）下完整重复，检验排序的跨 σ 稳健性。用户 2026-10-30 直给四步设计
（噪声注入 → 影响度量 → 关键节点排序 → 稳健性检验）。

## Requirements

### R1 实验设计（用户直给，四步钉死）

1. **噪声注入**：选定某一源通道，向其输入数据（z-score 后、入模前）的历史段 x_obs 注入固定方差
   高斯噪声 σ·ξ（μ=0，iid；ξ~N(0,1)）；σ 三组 = 0.3 / 0.5 / 0.7。
2. **影响度量**：计算其余 127 个通道在输出未来段（forecast）上的 MSE（扰动生成 vs 未扰动基线，
   跨 trial×时间点平均），对 127 个 MSE 求和 ⇒ 该源通道的**全局影响值** I_c(σ)。
3. **关键节点排序**：对全部 128 个通道逐一执行上述步骤，按 I_c 从大到小排序 ⇒ 关键节点排序
  （rank 1 = 全局影响最大 = 最关键节点）。
4. **稳健性检验**：在 σ∈{0.3,0.5,0.7} 三档下完整执行，对比三档排序（一致性 / 相关性），
   评估关键节点排序是否稳健。

### R2 执行范围（用户 2026-10-30 拍板）

- **双线都跑**：MODMA·latent-flow 0705 五折（53 例：10/10/11/11/11）+ mild·latent-flow symw
  五折（40 例：8/折）；ckpt 一律取 `result/checkpoints/` 注册件（只读）。
- **五折全量**：逐被试排序 + 组平均排序，双线各自成档。
- **归档：仅 outputs/ 暂不注册 result/**（result-dir-policy：零写入，收尾登记"本轮未写 result/"）。

### R3 口径契约

- 复用 `bisect_sigma.SignalProbeSite`（μ=0 无偏 iid 噪声、确定性采样 z_obs=mu_obs、CRN =
  同源同 ξ 跨 σ 复用 ⇒ 三档 σ 天然配对）；seed 42、n_steps 10、batch 32、GPU bf16 autocast
  （与双线 EC_pert 主跑同轨）。
- MODMA 每被试 32 trial 子采样（`subsample_trials`，seed 42 + 被试序号键）；mild 每被试 30 trial
  全量；mild 窗 500（obs 375 + fc 125）。
- 影响值定义排除源自响应（源通道自身 MSE 不入和）；并列值按通道序号稳定排序。
- σ=0.5/0.7 超出 EC 网格上限 0.48 ⇒ 属剂量外推，caveat 随档登记（不预设结论）。

### R4 守卫（对拍 result/ 注册件）

- MODMA：fold{k} ckpt SHA256 对拍 `result/checkpoints/modma/latent_flow_0705/sha256.json`
  + ckpt 文件名 fold 前缀与 `--fold` 锁定。
- mild：fold{k} ckpt SHA256 对拍 `result/checkpoints/mild/.../sha256.json` + norm SHA 按 fold
  对拍 `ec_mild_unbiased.EXPECTED_NORM_SHA_BY_FOLD` + manifest SHA 对拍。
- 任一不符即拒跑；守卫负例（错误 ckpt）冒烟实测一次。

### R5 交付物

- 逐被试 json（影响矩阵 128×3 + 排序 128×3 + 诊断 + 口径元数据）→ `outputs/10-30-critical-node-ranking/{modma,mild}/fold{k}/`。
- 稳健性汇总 json + 表（两级：逐被试 / 组平均）。
- **study 2 实验文档 REPORT.md**（实验目的 / 方法描述 / 结果展示 / 稳健性分析四段齐全）+ 图件
  （≥3 张）→ `outputs/10-30-critical-node-ranking/`。
- 任务 `experiment.md` Run 块四要素齐全；`.trellis/research/progress.md` 同步。

## Acceptance Criteria

- [ ] **AC1** 双线五折全量跑批完成（modma 53 + mild 40 被试 × 3σ × 128 源），逐被试 json 落
      outputs（断点续跑 + 原子写可用）。
- [ ] **AC2** 守卫全过：modma ckpt SHA×5 + mild ckpt SHA×5 + mild norm SHA×5（按 fold）+ manifest
      SHA；负例拦截实测（错误 ckpt 拒跑）。
- [ ] **AC3** 稳健性分析两级齐备：逐被试级与组平均级的跨 σ 三对（0.3↔0.5 / 0.3↔0.7 / 0.5↔0.7）
      Spearman ρ + Kendall τ-b + top-10/20 交叠；σ=0 恒等断言逐被试过；单调性诊断
      （I(0.7)≥I(0.5)≥I(0.3)）如实登记违例数。
- [ ] **AC4** REPORT.md 四段齐全且数字与 json 可溯源；图件 ≥3 张（组平均影响谱三 σ 并置 /
      rank-rank 散点 / 逐被试 ρ 分布等）。
- [ ] **AC5** experiment.md Run 块无空块；progress.md 同步；result/ 零写入登记。
- [ ] **AC6** 复跑契约：σ=0 结构性恒等断言过；同 seed 冒烟复跑一致（GPU 独占口径；若漂移按
      10-29 教训降 CPU 单线程复算并如实登记）。

## Notes

- 预算：modma ≈40 min + mild ≈30 min GPU（EC 同管线 17 σ 点 ≈45 min/10 例折算，本实验 3 σ 点）；
  GPU 当前空闲（2026-10-30 查 0 MiB）。
- 概念定位：EC_pert（边级 r\*_A→B）⇒ 关键节点（节点级 Σ_b MSE）；同一「扰动→响应」范式，readout
  从临界强度换成固定剂量全局影响，供 Study II 叙事（连接推断 → 节点重要性排序）。

## 子研究路由（1.0b / 1.0c）

<!-- trellis-research-routing -->

- **分类结果**：`study2` —— EC 推断算法（框架 Study II：Critical Perturbation Connectivity Inference；
  冻结代理脑上的虚拟扰动与响应度量）
- **依据**：任务主体是在冻结 θ\*（双线注册 ckpt）上的扰动→影响度量→排序算法实验，与 EC_pert
  推理同型（SignalProbeSite 管线直用），属 Study II 算法研究非代理脑训练。
- **新路径/方法检测（1.0c）**：`已有路径：latent-flow EC_pert（bisect_sigma.SignalProbeSite
  无偏 μ=0 + global-τ 两阶段）→ 节点级聚合变体` —— 复用既有扰动算子/CRN/确定性契约，仅新增
  「固定 σ 全局影响求和 + 排序 + 跨 σ 一致性」readout 层；无新模型 / 无新技术路线 ⇒ 不触发
  trellis-research 适配研究。
- **research 产物**：`无`（复用先例口径；spec/research/perturb-ec-pipeline.md §1/§8 契约继续适用）
