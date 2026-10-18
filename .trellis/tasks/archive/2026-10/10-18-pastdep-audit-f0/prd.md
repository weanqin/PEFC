# 生成器 past 依赖审计（双通路分解 + 注入位置扫描 + 置换检验）

## Goal

检验「future 生成几乎不依赖 past」（用户 2026-10-18 波形图人工检验引出）：以 E1–E4 四件套区分三假说 H1 结构性短记忆（OU 先验主导）/ H2 敏感性≠信息依赖 / H3 future 几乎不依赖 past。

## Requirements

- R1 θ\* 只读（同 v1-fmt fold0 pin）；R2 24 窗（同 perturb-search-f0 窗集）；
- R3 E1 解析对拍（prior.A 包络 T50_prior + 子窗质量）+ 采样全 128 点衰减谱（62 源 × 剂量 {0.5,1,2}）；
- R4 E2 注入位置扫描（4×96 boxcar 子窗、RMS 子窗重归一、r=1.0）；R5 E3 四臂置换/消融能量距离（独立 seed）；R6 E4 剂量×自跨分解；
- R7 判据预注册（T50 对拍 ±20%；ρ 带 <1.2 / 1.2–2 / ≥2）；R8 GPU ≤ 10 min；CRN 契约同源。

## Acceptance Criteria

- [ ] E1a 解析包络与 T50_prior 落盘；E1b–E4 采样完成，curves/e3 npz 落盘
- [ ] summary.md 判据表（三假说显式判定）+ 诊断图过 QA 链
- [ ] 单测全绿；experiment.md Run 块跑前登记 + 跑后回填

## 子研究路由（1.0b / 1.0c）

- **分类结果**：`study2`（生成器/推断方法性质审计，不训练）
- **依据**：对 EC 推断框架的生成器 past 条件化性质做行为审计，属 Study II 方法面证据。
- **新路径/方法检测（1.0c）**：`不涉及`（复用既有扰动-CRN-装载管线，无新路径）
- **research 产物**：`无`
