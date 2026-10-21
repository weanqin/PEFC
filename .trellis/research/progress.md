# 科研进度看板

> **维护规则**：SessionStart hook 只注入顶部「当前状态」区；AI 在节点完成/决策后当场更新（保持 ≤20 行）；历史细节追加到下方里程碑区（只增不改），完整过程见 journal。

## 当前状态

- **阶段**：study2 用户直给双方法 EC 探索完成（任务 10-21-v5main-ec-explore，2026-10-21）——方法1（噪声→靶熵失稳 σ*）**失败（用户宣告取消）**：4 例全删失（σ=0.48 时 lnVar 膨胀中位 0.014≪阈值 0.365，靶通道预测方差对源噪声不敏感）；方法2（源+0.5std 常量→靶预测均值响应）：10/10 例与 v4 GC refs **一致弱负相关**（ρ −0.126±0.019 全负、组均值 −0.188/r −0.317，方向信息≈零）；与增益算子/NI-MSE 阴性路径同向汇聚 ⇒ v5-main θ* 跨通道响应未达观测 GC 结构，重点指向 Study I 损失侧跨通道监督迭代。此前状态：edra_v4 重采样 GC 参考归档（10-21-oldlib-gc-edrav4）；NI-MSE 首执行阴性（10-20）——GC 计算主路径（原库谱 GC）× edra_process_v4 46 例全量（resample_poly(1,2)→250 Hz，其余口径同 edrafmt48 档）逐被试 EC 归档 `refs/granger_bivariate_spectral/edrav4-46-resample250-cont-zscore-fs250-fixed8-f1-125-nf65/`；与 edrafmt48 档重叠 45/46（v4 独有 s56），对照 ρ=0.2311±0.1026 / median 比 1.651±0.664（描述性登记，不做对比交付）；确定性逐位复现实测；此前状态：NI-MSE-50-q0.90 路径首执行阴性（10-20），两条 EC_pert 路径均阴性共同指向代理脑侧跨通道传导弱
- **Study I 候选现状**：gpfm 5-fold 候选集 {v1, v4a}（10-16 拍板）；v5/v5-fmt 轴分解交替变体完成（v5-main-01 0.73155 / v5-fmt-01 0.70242，θ\* 候选）；latent-flow edra_fmt θ 候选就绪（10-19 训练 run）但**三项合格性检验未做**；latent-flow MODMA 原生线本机首跑验证完成（10-21-nosc-check：geometry.pt 模板重建 + fold2 nosc probe CV 318–712% 超 JET 天花板，`N_CHANNELS` 缺陷登记未修）——三个候选族均未取得代理脑资格
- **Study II 状态**：两条 EC_pert 路径均阴性（perturb-ec-v1 增益算子 @v4a-fmt、NI-MSE 噪声注入 @v5-fmt-01），共同指向代理脑侧跨通道传导弱；流程/契约/规范（含 NI-MSE §7）可复用
- **暂停任务**：10-19-gpfm-mamba-v4b-gnn-f0（in_progress 暂停待恢复，未竟 = S4'-S6' 收尾 + Run 017 同口径评估）
- **下一任务（待建，先后待拍板）**：①Study I 迭代（损失侧跨通道监督——两条阴性路径的共同判据指向）；②latent-flow 合格性检验（θ 候选就绪）；③5-fold 主实验（范式/口径待拍板）
- **待拍板**：①下一任务先后关系；②指标2 视觉人工判；③GC 阻塞解除时点；④r\*→连接强度最终映射；⑤NI-MSE 若复跑的优化方案（D 预取零数值变化可直接用；A/B/C 数值口径变化须变更登记）；⑥10-19 暂停任务恢复时点；⑦probe `N_CHANNELS` 1 行 import 修复是否正式落包体（10-21-nosc-check wrapper 绕过，包体零改动纪律下留拍板）

## 里程碑（只增不改）

- 2026-10-10 · 研究框架文档定稿（用户提供）；Trellis 科研工作流装配完成；研究记忆初始化 · docs/research/research-direction.md
- 2026-10-16 · GPR+Mamba 可行性肯定（任务 10-12 归档：三检验通过、Q1 v2 最优 MAE 0.701 未超 GPR 基线）· 完整记录见 journal Session 2
- 2026-10-16 · H3 判「不支持」：混合不可局部移植为 GC 载体（接口/读出两臂证伪），5-fold 候选集 {v1, v4a}（任务 10-16 归档）· 完整记录见 journal Session 3
- 2026-10-17 · edra-fmt 新范式 fold0 前置验证通过（指标1/3 过线、v4a-fmt 全维度最优、GC 阻塞中；任务 10-16-gpfm-mamba-fmt-f0）· 完整记录见 journal Session 4
- 2026-10-17 · adaLN(t) 消融中性结果（全维度持平、fork_rng 契约兑现 epoch-0 逐位相等、排序不变；任务 10-17-gpfm-mamba-adaln-f0）· 完整记录见 journal Session 5
- 2026-10-17 · Study II 流程首验证：四要素首版 + 端到端跑通 + AC6 如实阴性（缺口=θ\* 跨通道弥散，四重证据）；流程规范首版沉淀 spec/research/perturb-ec-pipeline.md（任务 10-17-perturb-ec-f0）· 完整记录见 journal Session 6
- 2026-10-17 · 双变量 GC 观测参考落地：48 例 EC 矩阵 + 天花板 0.9243±0.0447（BIC 顶格 p\*=24 如实登记；不做 EC_pert 比较）· 完整记录见 journal Session 7
- 2026-10-18 · 双变量 GC 观测参考 250 Hz 原始全长档落地：refs-01 48 例 + 天花板 0.8981±0.1414 + rev-01 指纹 + AC6 raw bytes 级复现（条件数越限代决策续跑如实登记；不做跨口径对比）· 完整记录见任务 `.trellis/tasks/10-18-bivar-gc-fmt-250/experiment.md`
- 2026-10-18 · latent-flow 先前原型移植完成（study1 新候选路径）：OC-LatentFlow 代理 + bisection σ 归因代码面入 `models/latent-flow-v1/`，裁剪 A+bundle 备份、零源码改动、G1/G2 全过、trellis-check AC1–AC5 通过；MODMA 勘误（modma_resting_process.zip 可读，已解压在位）· 完整记录见任务 `.trellis/tasks/10-18-latent-flow-port/experiment.md`
- 2026-10-19 · latent-flow × edra_fmt 首个正式训练 run：npz shim + manifest 只读复用 + 两阶段 50 epoch 稳定收敛（stage1 1.5526 / stage2 mexhat CFM 0.5977），θ 候选产出、产物隔离、AC1–AC7 全过（D0/D1/D2/D4 四项用户拍板；S2.2 几何判定门如实触发后真实拍板 npy 基座）· 完整记录见任务 `.trellis/tasks/10-19-latent-flow-edra-fmt-train/experiment.md`
- 2026-10-20 · v5-fmt 变体完成（任务 10-20-gpfm-mamba-v5-fmt-f0）：v5 结构 × fmt 口径 + 早停 10（用户拍板），152,957 参数/单测 11/11，主 Run v5-fmt-01 ep39 早停 best 0.70242@ep29，A_eff/W 快读出随档，θ* 候选就绪待 study2 NI-MSE 接入
- 2026-10-20 · （10-19-gpfm-mamba-v4b-gnn-f0 会话补档）v5 变体落地 + v5-main-01 直训完成（best 0.73155@ep64）；GC 计算主路径拍板 = 原库 GC + refs 归档启用 `npi/refs/`（首批 fmt 48 例 EC 矩阵）；NI-MSE-50-q0.90 设计定稿 v1.5（用户五要素逐项拍板）
- 2026-10-20 · NI-MSE-50-q0.90 首执行（任务 10-20-ni-mse-v5fmt-f0）：v5-fmt-01 θ\* × c{0.05,0.1,0.3,0.5} × Q90 门控，流程契约全兑现（R2/split-half/续跑）；**阴性主读出**（G 全 0/3782、S~0.45、剂量全平、refs r≈0.05–0.10 flip≈direct）⇒ 该配置下 EC_pert 不可用；可交换 null 灵敏度权衡与 RNG 契约成本归因入 spec §7 · 完整记录见任务 `.trellis/tasks/10-20-ni-mse-v5fmt-f0/experiment.md`
- 2026-10-21 · 原库 GC × edra_v4 重采样参考归档（任务 10-21-oldlib-gc-edrav4）：v4 46 例全量（resample_poly(1,2)→250 Hz + fixed8/1–125 Hz 口径，15.4 min、SHA256 逐位复现）→ `refs/granger_bivariate_spectral/edrav4-46-resample250-.../`；与 edrafmt48 档重叠 45/46（s56 仅 v4），对照 ρ=0.231±0.103 / median 比 1.651±0.664 描述性登记；口径条目入 evaluation-calibers.md（含 GC 主路径登记）· 完整记录见任务 `.trellis/tasks/10-21-oldlib-gc-edrav4/experiment.md`
- 2026-10-21 · v5-main 双方法 EC 探索（任务 10-21-v5main-ec-explore）：方法1 熵失稳 σ* 失败宣告（4 例全删失证据在档）+ 方法2 常量偏移均值响应 10 例一致弱负（ρ −0.126±0.019）；产物 outputs/10-21-v5main-ec-explore/（未随 git，口径登记 README+experiment.md）· 完整记录见任务 `.trellis/tasks/10-21-v5main-ec-explore/experiment.md`
- 2026-10-21 · latent-flow MODMA no-sc ckpt 本机检查与首跑（任务 10-21-latent-flow-modma-nosc-check）：geometry.pt 按 MNE GSN-129 模板重建（vs ckpt 内嵌 pairwise_d corr=1.0000000000 / max diff 3.4e-08 m 逐位命中）落 `/root/autodl-tmp/modma/geometry.pt` + fold2 nosc（旧库 MODMA SOTA，自包含 state_dict）加载断言全过（W/pairwise_d 逐位一致、gamma_sc 弃用键按白名单丢弃）+ probe 首跑（fold2 test 1 例 169 窗 × 128 源 × 3σ，~17 min GPU）**CV 318.18%/587.12%/712.18%（σ=0.1/0.2/0.3）全超 JET v2 天花板**；probe `N_CHANNELS` 潜在 NameError（旧库 HEAD 同款，import 块漏 driver 定义）wrapper 注入绕过、包体零改动 · 完整记录见任务 `.trellis/tasks/10-21-latent-flow-modma-nosc-check/experiment.md`
