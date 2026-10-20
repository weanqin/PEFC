# NI-MSE-50-q0.90 执行（study2 EC_pert 路径）：v5-fmt-01 θ\* 信号空间噪声注入 → EC_pert × real GC refs 判读

## Goal

在冻结 θ\* **v5-fmt-01**（`outputs/gpfm-mamba-f0/v5-fmt-01/checkpoints/gpfm-mamba-v5-fmt-best.pt`，EMA shadow，config_hash `a64c0d416756`，best val 0.70242）上执行 **ni-mse-design.md v1.5 主实例化**：信号空间源通道加性零均值高斯（σ_p ≡ c = 0.5，标准化空间）× future 前 0.1s（n₀=25）逐靶 MSE 响应 × posterior null Q90 门控 × null 64×200 / 响应 32×50 窗（W_pert ⊂ W_null），产出 **EC_pert**（超限比 S + 门控 G，逐被试 + 组级）与 **× real GC refs**（fmt 48 例，只读）的 Pearson/Spearman/flip 检查/距离偏判定轨判读。

完整路径 = study1-gpfm-mamba-v5-fmt（已完成，归档任务 10-20-gpfm-mamba-v5-fmt-f0）× study2-NI-MSE-50-q0.90（本任务）。新路径目录 `models/ni-mse-v1/`；产物根 `outputs/10-20-ni-mse-v5fmt-f0/ni-mse-v5fmt-01/`。

## Requirements

1. **执行参数按 v1.5 钉定值，不可改**（变更须走口径变更纪律）：算子 `P(X,a,c): X'[a,:]=X[a,:]+c·ξ`（零均值加性高斯，标准化空间，注入 past 全段、位于 z-score 之后模型输入之前）；c 主读出 0.5，敏感性网格 {0.1, 0.3}，阴性对照 c=0.05；n₀=25（fmt@250）；B_ref=64（固定 ε 网格）；K_null=64 × W_null=200 窗/例；K=32 × W_pert=50 窗/例且 **W_pert ⊂ W_null**；τ_b = 逐靶通道 null **池化样本（W_null×K_null）** Q90（np.quantile 线性插值；k→w 中位聚合链只作用于响应 R̄ 与汇报层，不进 τ_b 估计）+ split-half 稳定性（>10% 触发 R3 加密）+ [Q90,Q95) 边际带旗标；种子树 master=2026（**ε_ref / ε_null / ξ_pert / η 四支**——η（σ_min 噪声）为审查补钉的独立派生支，缺省时 sample() 走全局 CUDA RNG 破坏逐位复现——+ win-null/win-pert 窗子集，全落 manifest）。
2. **θ\* 只读装载**：v5-fmt-01 ckpt/`channel_mean,std.npy`/config 只读（EMA shadow 回灌 + config_hash 断言）；走 v5-fmt 目录 sys.path 遮蔽惯例（v1 `sample.py` 复用 + v5-fmt `model.py`/`graph.py` 命中，spec 防坑矩阵约束）；EulerSampler 构造 **`gen_lowpass_hz=None`**（fmt 无低通口径；默认 40.0 brickwall 为已登记陷阱）。
3. **数据**：edra_processed_fmt fold0 test 10 例（manifest 只读 `models/gpfm-mamba-v1-fmt/splits/fold0_status_ok_seed42_val4.json`，34/4/10）；**test 名单 = [10,11,12,17,24,28,30,44,61,69]**（勘误登记：ni-mse-design.md §1.1 所列 [12,17,18,24,30,38,44,55,58,61] 为 v4 过渡口径名单，非 fmt manifest）；窗 384/128/stride 64，确定性 loader shuffle=False。
4. **refs 只读 + 判读口径**：`refs/granger_bivariate_spectral/edrafmt48-cont-zscore-fs250-fixed8-f1-125-nf65/`（48 例 `refs_s{sid}.npz`，sid=被试 ID；ec[a,b]=a→b 行=源）；**逐被试配对为主口径**（EC_pert_s × ec_refs_s，offdiag 3,782 对，n=10 汇报 mean±std/median/range），组级单次为副口径（10 例逐元素中位 vs 同 10 例 refs 均值；48 例全均值敏感性登记）；direct 主、flip（ec.T）方向性检查副；距离偏 Spearman（控制 = geometry pairwise_d）随附。**不对生成信号重算 GC**（v1.4 拍板）。
5. **异常规则 R1–R6 机器可执行**：R1 NaN/Inf 硬失败中止；**R2 c=0 逐位恒等断言必须通过**（同 ε/η 扰动前后逐位相等）；R3 split-half >10% → K_null 加密重跑；R4 退化窗排除留日志；R5 被试级离群旗标；R6 超敏边旗标。
6. **可复现**：三期断点续跑 CLI（`--stage {null|perturb|report}`）；manifest.json 记种子树/窗子集/参数/θ\* 指纹/环境快照；入口 BLAS guard（numpy 导入前五变量 setdefault）；cudnn.deterministic=True。
7. **隔离**：只新增 `models/ni-mse-v1/` 与 `outputs/10-20-ni-mse-v5fmt-f0/`；θ\*/refs/v5-fmt 与 v1 代码、既有 outputs 全部只读；不修改 gpfm-mamba 家族任何文件。
8. **登记契约**：experiment.md 按 Run 块跑前登记/跑后补记；勘误（含 test 名单）如实登记；交付为数值（json/npz/表），无图（判读期如需图另行走可视化规范）。

## Acceptance Criteria

- [ ] AC1 路径落地：`models/ni-mse-v1/` 模块 + 单测全绿（R2 恒等断言、窗子集确定性复现、τ_b 分位数单点、refs 被试映射；GPU 前向最小规模）
- [ ] AC2 主 run 三期完成：null（τ_b + split-half + Q95 旗标）→ perturb（c 臂 {0.05,0.1,0.3,0.5} × 62 源 × 10 例）→ report（S/G 矩阵逐被试+组级 + manifest）全部落盘 `outputs/10-20-ni-mse-v5fmt-f0/ni-mse-v5fmt-01/`
- [ ] AC3 R2 恒等断言通过 + split-half 数值登记（<10% 或 R3 处置如实登记）；R4–R6 旗标如有则登记
- [ ] AC4 判读四件套登记：①EC_pert × refs 逐被试 n=10 主口径（Pearson+Spearman mean±std）+ 组级副口径；②flip 检查（direct vs ec.T）；③距离偏 Spearman；④阴性对照 c=0.05（G 通过率 vs Q90 名义 10%、S 分布→1）与敏感性 c∈{0.1,0.3}
- [ ] AC5 结构预测判读（design §4.4）：A_eff 几何主导 ⇒ 预期强距离衰减/方向性弱/flip≈direct——按实测如实判读登记（非门槛）
- [ ] AC6 只读契约零违反（git status 复核：除新增目录与任务文件外零改动；refs/θ\* 校验和不变）
- [ ] AC7 experiment.md 全程契约 + 勘误登记完整（含 fmt test 名单勘误、epoch 编号以 ckpt 字段为准）

## Notes

- **权威设计源**：`.trellis/tasks/10-19-gpfm-mamba-v4b-gnn-f0/ni-mse-design.md`（v1.5，只读引用；本 prd 与本任务 design.md 仅记实例化落定与勘误，不复制全文）。
- 判读背景（非门槛）：v4a-fmt θ\* 上旧路径未获结构收敛证据（10-17 任务 AC6）；本任务 θ\* 为 v5-fmt-01，结果无论正负均如实登记。
- 计算量预算：单 c 臂扰动 ~17 min + null ~5 min；四 c 臂 + 装载/落盘合计 ~1.5 h 单卡量级（Run 007 实测速率外推）。

## 子研究路由（1.0b / 1.0c）

<!-- trellis-research-routing -->

> 任务创建时由 hook 自动播种。规划阶段必须填写以下四项；全部落定前不得进入 1.4 start 门禁（见 `.trellis/workflow.md` Phase 1.0b / 1.0c）。

- **分类结果**：`study2`
- **依据**：在冻结 θ\* 上做虚拟扰动（噪声注入）→ 响应度量（MSE）→ 门控（Q90 null）→ EC 对比（× real GC refs），属框架 Study II EC 推断；θ\* 为 study1 产物只读消费，本任务不含训练。
- **新路径/方法检测（1.0c）**：`已有路径：NI-MSE-50-q0.90 设计 v1.5 已定稿（ni-mse-design.md，2026-10-20 用户认可全方案并钉执行参数；context.md 2026-10-20 结论条目登记）——本任务为该设计的主实例化执行，无新方法路径`
- **research 产物**：`无`（设计文档即研究产物；五要素与执行参数均经用户逐项拍板）
