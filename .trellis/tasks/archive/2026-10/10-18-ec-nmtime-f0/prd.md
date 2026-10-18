# prd — N×M-time 完整 EC 推断（v1-fmt θ\*·fold0 test；只出矩阵版）

## Goal

用 N×M-time（10-18-perturb-search-f0 的唯一 Gate B 过线组合，自跨比 2.17）在冻结 v1-fmt θ\* fold0 上执行**完整 EC_pert 生产流程**：N 剂量-响应采样 → 三 readout（slope 主 / AUC 次 / r\* 降级删失标注）→ 契约 npz 落盘 → 描述统计。**只出 EC 矩阵，不做 GC 对照与收敛判定**（用户 2026-10-18 拍板）。

## 背景

搜索任务定论：四门框架下本 θ\* 无可用 {P\*×M\*}，但 N×M-time 为 30 组合中唯一 Gate B 过线者（2.17/2.05 两阶段复现、置换 p<0.0002、全网格 r 稳定、56% 源通道自响应>2× 跨中位）；Gate D 30/30 全绿（|Δ| 单调，r\* 可定义）。10-17 的「EC 推断不可用」定论建立在 G×M-RMS 上——本任务产出 N×M-time 版矩阵，为后续对照任务备料。

## Requirements

- **R1 θ\* 只读**：同 v1-fmt fold0 best ckpt（hash 0dc0d5249c68 严格校验 + gen_lowpass_hz=None 断言）；装载配方沿 perturb-search-v1/loader.py。
- **R2 数据规模**：fold0 test 全 10 例 × 每例前 8 窗 = 80 窗（用户拍板）；win_idx = 受试者序×8 + 例内窗序 0..79。
- **R3 r 网格**：9 点含原点适配 N 尺度（用户拍板）R_GRID = {0, 0.03125, 0.0625, 0.125, 0.25, 0.5, 0.75, 1.0, 1.5}，低剂量段加密；**索引 0 ≡ r=0 ≡ D̄=0 由 CRN 恒等按构造不采样**（沿 perturb-ec-v1 readout 契约）；slope 段 = 前 5 点含原点；批 497 行/窗（1 + 62×8）。
- **R4 readout 三件套 + 契约**：slope（低 5 点 OLS 含原点，主）/ AUC（梯形/R_MAX）/ r\*（τ̃=0.25 相对自响应 D̃=D̄_{A→B}/D̄_{B→B}，右删失编码 R_NEXT、flip 后存）；npz 契约 = {ec (62,62), readout∈{slope,auc,rstar}, subject_ids|'global', orientation='larger_is_stronger'}——接口预留给后续对照任务，本任务不接比较器。
- **R5 描述统计（无对照）**：slope EC 自跨比、r\* 删失率（τ̃=0.25 主 + 0.5 敏感性）、有效路径数、Spearman(EC, 通道距离)、跨受试者 split-half 稳定性、dbar 单调占比（Gate D 同构 sanity）。**不做** GC/gPDC 对照、AC6 判定（用户拍板「只出 EC 矩阵」）。
- **R6 成本**：GPU ≤ 15 min；CPU readout 秒级。

## Acceptance Criteria

- [ ] 80 窗主 Run 完成，跨 Run CRN 逐位可复现（eps/eta sha256 落盘）
- [ ] 逐 subject（10）+ global 的 slope/AUC/r\* EC 矩阵按契约 npz 落盘（33 个矩阵文件）
- [ ] 描述统计六项齐全并落 summary.md；experiment.md Run 块跑前登记 + 跑后回填
- [ ] B0 单测全绿（N 算子契约/批构造/CRN/d_time 解析/readout 合成数据/loader 口径/输出级恒等）

## 约束与非目标

- 表述纪律：「v1-fmt θ\* 上的 EC_pert 产物」（前置验证级 θ\*）。
- N 波形常量与 perturb-search-v1 逐位一致（seed 2026 / 1–40 Hz / RMS=1 / 全局共享单波形）。
- 非目标：GC/gPDC 对照与收敛判定（后续任务基于契约 npz 直连）；gates 复评；多种子/多 K；神经学解读。

## 子研究路由（1.0b / 1.0c）

- **分类结果**：`study2`（EC 推断算法/产物生产，框架 Study II）
- **依据**：任务为 EC_pert 矩阵生产流程（扰动-响应-readout 全链），属 Study II 方法面；不训练模型。
- **新路径/方法检测（1.0c）**：`已有路径：N×M-time 组合 = 10-18-perturb-search-f0 搜索产物；readout 三件套沿 perturb-ec-pipeline.md 既定契约` —— 无新路径，research 未触发
- **research 产物**：`无`
