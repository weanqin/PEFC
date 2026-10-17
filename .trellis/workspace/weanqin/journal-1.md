# Journal - weanqin (Part 1)

> AI development session journal
> Started: 2026-10-10

---



## Session 1: baseline EC 观测参考管线（GC+gPDC）与扰动 EC 主轨比较器交付
<!-- trellis-session: v=2 fp=a93cf8635350ae72 -->

**Date**: 2026-10-16
**Task**: baseline EC 观测参考管线（GC+gPDC）与扰动 EC 主轨比较器交付
**Branch**: `main`

### Summary

任务 10-13-baseline-ec-v1 完成（已归档）：EDRA 46 例 125Hz 降采全量产出 GC（ridge 条件 ln SSR 比）与 gPDC(|gPDC|²) 两参考矩阵；关键定标=OLS 在 62ch 下 Gram 条件数 ~1e18、split-half 天花板≈0 → ridge λ_rel=1e-2 + 固定 p=8（ASCT 锚点），天花板救回 gc 0.900±0.030 / gpdc 0.734±0.039；mne-c 二元交叉核对 vs 条件 GC 0.069±0.080（estimand 分歧量化）。主轨比较器（距离偏相关 Spearman 主指标 + t×Holm + bootstrap + 天花板比值）以合成 EC_pert 联调通过；EC_pert readout 约束（低剂量斜率主/AUC 次/r* 降级删失）与术语纪律（observational references）登记入 evaluation-calibers/surrogate-qualification。TE 全系取消（用户拍板，16 核 cgroup 实测 S=50/100 需 35-70 天）；环境契约新增 cgroup 16 核事实 + _blas_guard 线程护栏惯例。真实 EC_pert 对比与三角诊断留待 Study II。

### Git Commits

| Hash | Message |
|------|---------|
| `d951ed9` | feat(baseline-ec-v1): 观测参考 EC 管线（GC+gPDC 两参考法）+ 扰动 EC 主轨比较器 + split-half 天花板 |
| `a034135` | docs(baseline-ec-v1): Run 002 配置块回填代码提交哈希 d951ed9（amend 后自引用修正） |

### Status

[OK] **Completed**


## Session 2: gpfm-mamba-edra-f0 收尾：三检验判定通过、H3 勘误与移交、任务归档
<!-- trellis-session: v=2 fp=7439ce763720d59a -->

**Date**: 2026-10-16
**Task**: gpfm-mamba-edra-f0 收尾：三检验判定通过、H3 勘误与移交、任务归档
**Branch**: `main`

### Summary

任务 10-12-gpfm-mamba-edra-f0 完成并归档（GPR+Mamba 可行性立项问题收肯定回答）。①三检验判定（用户 2026-10-16 拍板）：指标 1 FC 一致性通过（标准=组平均 r≥0.95，v1 0.9586/v2 0.9763）、指标 2 视觉 v1/v2 均过、指标 3 PSD 通过（标准=低频段 δ–β |Δlog10|≤0.5+目视；θ/β 补登 0.30–0.38，α 随边界漂移 0.12–0.18 但判定稳健）——通过标准回填 surrogate-brain-qualification.md 消除占位。②H3 勘误（用户更正定义）：H3=跨通道交互假设（v1 d16 瓶颈强制混合 vs v2 d62 通道对齐可独立映射，解释 GC 一致性 0.276>0.123）；三处历史误标勘误（Run 016 损失侧频带项误挂 H3、Run 012 tv-权重假设让位、Run 016 幅度括注训练期口径误用于生成侧——实测过冲 +23%/+29%）。③H3 移交下一任务：读 Tank 2021（cMLP 组件式+列组 group lasso）与 Li & Zhou 2025（sparse cMLP iEEG EZ 定位）两篇论文提取强制跨通道机制，写 seed（.trellis/research/h3-cross-channel-seed.md，v4 设计空间 A 输入瓶颈解耦/B 组件式输出头/C 混合块，推荐 A；论文存档 docs/research/papers/）。④5-fold 暂缓（用户拍板，候选集待 H3 变体扩充为 v1/v2/v4 一并取舍）。trellis-check 全量检查 PASS（1 Major 幅度表述+2 Minor 已修）。附带：前会话遗留 trellis 设施（1.0b/1.0c 路由门禁/data spec/nature-figure）补入库。

### Git Commits

| Hash | Message |
|------|---------|
| `aafb013` | chore(trellis): 科研工作流设施装配（前会话完成未入库）——1.0b/1.0c 子研究路由门禁（workflow.md+config.yaml hook+seed-research-routing 播种脚本）、data spec 接入（edra 数据集契约 index+data-pipeline.md）、nature-figure 技能 |
| `7a1ab8a` | docs(task-10-12): 前置验证三检验判定通过、H3 勘误、任务收尾——指标 1/3 通过标准回填 spec（组平均 r≥0.95 / 低频段 δ–β ≤0.5+目视，用户 2026-10-16 拍板）；H3 更正为跨通道交互假设（v1 d16 瓶颈强制混合 > v2 d62 通道对齐；三处历史误标勘误），seed+两篇 cMLP 论文存档移交下一任务；任务总结论：GPR+Mamba 可行性肯定（Q1 v2 最优 MAE 0.701 未超 GPR 基线、生成 std_t 过冲 +23%/+29% 勘误登记；Q2 三检验通过）；5-fold 暂缓（候选集待 H3 变体扩充） |
| `11f8a20` | chore(task): 归档 10-12-gpfm-mamba-edra-f0（GPR+Mamba 可行性任务收尾：三检验通过、H3 移交下一任务、5-fold 暂缓） |

### Status

[OK] **Completed**

### Next Steps

- 下一任务（待用户启动）：H3 跨通道交互假设变体探究（v2 基础 cMLP 式 v4，seed 就绪）；随后 5-fold 候选集拍板（v1/v2/v4）


## Session 3: 10-16-gpfm-mamba-h3-v4 全程与收尾：H3 判「不支持」、5-fold 候选集 {v1, v4a} 拍板、任务归档
<!-- trellis-session: v=2 fp=8c21e5f0b3a4d719 -->

**Date**: 2026-10-16
**Task**: H3 跨通道交互假设：gpfm-mamba v2 基础跨通道强制变体（v4）探究
**Branch**: `main`

### Summary

任务 10-16-gpfm-mamba-h3-v4 完成并归档。①v4-B 臂（cMLP 组件式读出头）：实现+主 Run（GC r 0.133）→ d_ch 1→8 剂量变更（用户方案 B）→ d8 主 Run（早停 55ep，GC r 0.163、指标 3 低频 0.519 超线、std_t 过冲 174%）——「容量必要而不充分」。②中途口径变更四次（均用户拍板，default-off 钩子落地）：fold0 内层 7/1/2=32/4/10、早停 patience=10、恒定 lr 1e-3（弃 cosine）、方向 v4-B→v4-A。③v4-A 臂（rank-16 纯线性输入瓶颈 + 共享 trunk d62）：v1 bottleneck_rank 钩子 + v4a 目录（八项单测 + 三变体 hash 零漂移）→ bench 1.95 GiB → 主 Run 早停 73ep（best val 0.77901）→ Run 013 正式口径五列评估：**质量五方最优**（MAE 0.69540 / CRPS 0.52392 / 逐通道 r 0.18317 首超 GPR 均值基线 0.16782）+ 指标 1 FC r 0.97561 过线 + 指标 3 低频 0.3425 过线 + R1 安静；**GC r 0.092 五方最低**（≈v2 CI 重叠）。④H3 判定（用户拍板收尾）：「**不支持**」侧——接口混合（v4a 0.092）与顶层显式读出（v4b 0.133/0.163）两臂均不回升（v1 0.2755），跨通道混合不可局部移植为 GC 载体；缺口共同指向损失侧（无跨通道时序监督）或 v1 优势载体「深度持续混合」（16 维贯穿 trunk 全深，原 C 方案域不追）；瓶颈位置语义=「门口窄」≠「发动机窄」。⑤**5-fold 候选集拍板 {v1, v4a}**（v1=GC 一致性最高构型、v4a=质量最优构型；v2/v4b-d1/d8 落选）；r 扫描 {8,32} 议项不采纳。⑥trellis-check 收尾检查 PASS with minors（收尾数字与落盘 JSON 逐位核对一致、v4a 八测 + v1 七测全数复跑通过；4 条措辞 Minor 已修：掉字/任务归属/32-x-x→7/1/2 口径/钩子链补 split_manifest_override）。spec 回填：data-pipeline fold0 内层 7/1/2 长期口径、reproducibility default-off config 钩子惯例；context/progress 同步。

### Git Commits

| Hash | Message |
|------|---------|
| `84e1dc3` | feat(task-10-16): v4-B 变体实现 + 1-epoch 基准过门禁待批（提交点 A/B） |
| `ed80e3e` | feat(task-10-16): 划分口径 7/1/2（32/4/10）落地 + Run 004 基准更新（用户 2026-10-16 拍板） |
| `0e89774` | feat(task-10-16): 早停 patience=10 落地（用户 2026-10-16 拍板，default-off 钩子） |
| `896fbb6` | feat(task-10-16): 取消 cosine 退火改恒定 lr 1e-3（用户 2026-10-16 拍板，default-off 钩子） |
| `ad1bb64` | feat(task-10-16): v4b 主 Run + 正式口径评估 + readout 诊断（提交点 C/D） |
| `acfc899` | feat(task-10-16): d_ch 1→8 口径变更（用户 2026-10-16 方案 B）+ Run 008 基准 |
| `0c8d0ee` | feat(task-10-16): v4b-d8 主 Run（早停 55ep）+ 正式评估 + readout（提交点 C/D） |
| `c329013` | feat(task-10-16): 方向变更 v4-B→v4-A（用户拍板）+ v4a 实现与基准（提交点 A'/B'） |
| `349aedd` | feat(task-10-16): v4a-r16 主 Run（早停 73ep）+ 正式评估（提交点 C'/D'） |
| `af5c8c1` | docs(task-10-16): H3 判定收尾（用户拍板）+ 5-fold 候选集 {v1, v4a} + spec 回填 |
| `e7b0406` | chore(task): 归档 10-16-gpfm-mamba-h3-v4 |

### Status

[OK] **Completed**

### Next Steps

- 下一任务（待用户启动）：5-fold 主实验——候选 {v1, v4a} × 5 折（口径：7/1/2 内层切分、早停 10、恒定 lr 1e-3、gen_lowpass 40 Hz；seed 数与评估范围 brainstorm 时定）


## Session 4: edra-fmt 新范式 fold0 前置验证（任务 10-16-gpfm-mamba-fmt-f0）全程与收尾
<!-- trellis-session: v=2 fp=d115160fb07340ae -->

**Date**: 2026-10-17
**Task**: edra-fmt 新范式 fold0 前置验证（任务 10-16-gpfm-mamba-fmt-f0）全程与收尾
**Branch**: `main`

### Summary

v1-fmt/v4a-fmt（LayerNorm/p=25/无低通/genGC-realGC 阻塞口径，48 例 34/4/10，fold0 单 seed）完成 smoke→bench→主 Run（92/55ep 早停）→正式评估：指标1 FC 0.970/0.981、指标3 1–30Hz mean|Δlog10| 0.317/0.301 全过线；v4a-fmt 质量全维度最优（MAE 0.70565/CRPS 0.53912/逐通道 r 0.12335 超 GPR-r 0.116），候选排序与旧范式一致（判读权重向 v4a 侧）；无低通下生成侧 >40Hz 功率占比≈真实 3.0×（≈95% 在带内）；v4a-fmt R1 触发定性口径效应。跨范式=多变量设计（fs/预处理/48例/p/低通/norm/纪律），仅方向性判读。spec 回填：data/edra-fmt 契约、指标3 频段 1–30/40–Nyquist 适配、evaluation-calibers fmt 口径、reproducibility 钩子链、data-pipeline 整数分配语义；context/progress 同步。收尾：提交 136223e/3905177/fa06f85 + 归档 b2a7c89（D7 勾选、三套单测全绿门禁）。5-fold 范式与 GC 阻塞解除待用户拍板。

### Git Commits

| Hash | Message |
|------|---------|
| `fa06f85` | docs(fmt-f0): 判读回填（experiment.md 全 Run 块+跨范式对比/差异清单/>40Hz 观察）+ spec 回填（edra-fmt 契约/指标3 频段/fmt 口径/钩子链/整数分配语义）+ research 记忆同步（提交点④） |
| `b2a7c89` | chore(task): 归档 10-16-gpfm-mamba-fmt-f0（edra-fmt fold0 前置验证通过：指标1/3 过线、v4a-fmt 全维度最优、GC 阻塞口径登记；5-fold 范式待拍板） |

### Status

[OK] **Completed**


## Session 5: v4a-fmt adaLN 调制归一化消融（任务 10-17-gpfm-mamba-adaln-f0）全程
<!-- trellis-session: v=2 fp=3869b1c605f7541e -->

**Date**: 2026-10-17
**Task**: v4a-fmt adaLN 调制归一化消融（任务 10-17-gpfm-mamba-adaln-f0）全程
**Branch**: `main`

### Summary

v4a-fmt 基座上把 5 个 LayerNorm 位换 adaLN（emb(t)→逐 site SiLU→Linear(16,124) 零初始化，A 案 scale/shift 用户拍板，参数 130366→140906）。流程：1.0c 适配研究（DiT 锚核）→ 双 Mandate review（共同 Major：nn.Linear 构建消耗全局 RNG→采纳 fork_rng 中性构造）→ 实现+六段单测+五套回归全绿 → smoke/bench → 主 Run 54ep 早停（best 44，val 0.66262）→ 正式评估（GC 阻塞口径）→ trellis-check 通过（6 处记录笔误修复）。结果：与 v4afmt-main-01 全维度持平（MAE 0.70571 vs 0.70565、FC 0.9797 vs 0.98147、指标3 0.30605 vs 0.30134 均✓，Δ≤0.005），中性结论——该规模/纪律下 t 条件调制无超出输入侧 temb 注入的增量收益；候选集 {v1,v4a} 排序不变。亮点证据：两 Run epoch-0 step-1 损失逐位相等（1.4787474870681763，fork_rng 契约 run 级兑现）。spec 回填：reproducibility RNG 中性构建惯例 + cond_adaln 钩子链、context 技术路线登记、progress 同步。提交 847c728/1b8b4ce/c235943。同 session 前段完成 fmt-f0 收尾（Session 4 已记）。

### Git Commits

| Hash | Message |
|------|---------|
| `c235943` | docs(adaln-f0): experiment.md Run 001–004 回填 + 对比表（中性结论·单变量）+ spec 回填（reproducibility RNG 中性构建惯例/钩子链、context 技术路线登记）+ research 记忆同步（提交点④） |

### Status

[OK] **Completed**

## Session 6: Study II 流程首验证（任务 10-17-perturb-ec-f0）全程
<!-- trellis-session: v=2 fp=pertur-ec-f0 -->

**Date**: 2026-10-17
**Task**: fold0 虚拟扰动→EC 有效路径推断流程设计验证（任务 10-17-perturb-ec-f0，study2 路由）
**Branch**: `main`

### Summary

以 v4a-fmt（非 adaLN）fold0 best ckpt 为冻结 θ\*（前置验证级偏离经用户 1.4 签认），完成 Study II 首任务全流程。规划：prd/design/implement + 双 mandate review（A-M1 先验通路事实更正：posterior_mean 逐通道、CRN 下跨通道仅速度网络单通路；B-Major1 gen_lowpass=40 陷阱、Major2 importlib sys.modules 先注册）。实现 B0-B6：perturb-ec-v1 变体（A 案增益算子用户拍板 + CRN 批采样 497 并批 + readout 三件套 + 子进程隔离参考侧 + 比较器）。关键节点：smoke 发现绝对 τ（独立样本对 q95≈1.82σ）下 100% 删失——CRN 确定性响应 vs 条件分布宽度材质性错配，τ v2 经用户拍板改相对自响应 τ̃=0.25（删失 3.7%、路径集交集 3201 边）；主 Run 10 例 27.7 min、跨 Run 逐位复现、100% 单调。AC6 如实判定：**该 θ\* 上未获结构收敛初步证据**（slope×GC ρ̄=−0.007 p=0.581、slope×gPDC ρ̄=+0.022 Holm p=0.112；ceiling 0.96、实际可检 ρ̄≈0.028 排除 power 不足），缺口四重证据（空间衰减平 −0.05、D̃ 近 r 无关 ~0.78、三 readout 同值、自/跨比 1.27×）定位 θ\* 跨通道传导弥散无特异性——为 Study I 损失侧迭代提供直接判据。spec 首版沉淀 spec/research/perturb-ec-pipeline.md + evaluation-calibers τ v2 登记 + context/progress 同步。trellis-check 通过（1 Major 数值笔误修复）。提交 5e66c88/2950630/37931dc/④。

### Status

[OK] **Completed**
