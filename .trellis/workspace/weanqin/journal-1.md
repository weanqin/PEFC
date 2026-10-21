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

## Session 7: 双变量 GC 观测参考落地（任务 10-17-bivar-gc-ec-fmt）全程
<!-- trellis-session: v=2 fp=bivar-gc-ec-fmt -->

**Date**: 2026-10-17
**Task**: edra-fmt 48 例成对双变量 GC 计算 EC（任务 10-17-bivar-gc-ec-fmt，study2 路由）
**Branch**: `main`

### Summary

用户四轮拍板收敛任务口径（①edra-fmt/48 例全量/125 Hz 降采/逐对 BIC/ln(SSR)/默认代码落点；②fmt 条件 GC 滞后；③split-half 认可；④**不做 EC_pert 与 GC 比较**——比较器接入整体移出范围）。规划三件套 + jsonl 后 1.4 启动。实现 B0-B5：bivar-gc-ec-v1 变体（全通道滞后 Gram 一次构建 + 逐无序对逐阶 slice 快路径 + naive lstsq 等价性单测；OLS 无收缩依据=二通道 pair-Gram 条件数实测 1e5-1e7 vs 全模型 62ch ~1e18；offdiag≥0 样本内必然断言）。关键节点：G1 复核抓获派发提示 4p+1 维数笔误（正确 2p+1）；refs-01 首跑 48 例计算完整但聚合段 `ec_all[~np.eye]` 布尔索引错位崩（stderr 丢弃教训→离线复现定位→修复批次维广播→同 run_id 重跑 36 s 产物逐字一致）；split-half 天花板 0.9243±0.0447（v4 条件 GC 0.9003±0.0296 同量级，方向性对照只登记）；**BIC 100% 顶格 p\*=24**（48 例 × 1891 对，n≈3 万惩罚项极小）如实登记为实测事实，网格扩展未决。图走 visualizer 纯复用（ec_matrix 三面板/ec_heatmap_grid），阶数分布退化单点不为单柱图扩共享文件。trellis-check PASS-with-minors（3 处自修：experiment 命令登记失实×2、data_io 句柄；无 Major）。spec 更新：evaluation-calibers「双变量 GC 观测参考口径」条目 + edra-fmt 下游消费行；context/progress 同步。

### Git Commits

| Hash | Message |
|------|---------|
| `58fe4bc` | feat(bivar-gc-ec-fmt): 48 例双变量 GC EC 矩阵 + 天花板 + spec/记忆同步（全文见 git log） |

### Status

[OK] **Completed**


## Session 8: bivar-gc-fmt-250：250Hz 原始全长双变量 GC 真实参照 + TRGC/对比/分半/净出度诊断全套
<!-- trellis-session: v=2 fp=1ad7645067cd4265 -->

**Date**: 2026-10-18
**Task**: bivar-gc-fmt-250：250Hz 原始全长双变量 GC 真实参照 + TRGC/对比/分半/净出度诊断全套
**Branch**: `main`

### Summary

全流程：规划三件套+两轮审查（split_half API 失实与 λ_rel 不可行两严重发现回填）→ B0-B5（smoke 条件数 1e13–1e15 触发预注册门禁，用户未实时应答按可逆原则代决策原 OLS 口径继续并逐 Run fast↔naive 校验）→ refs-01 48 例（p*=48 96.8% 顶格如实）→ reprod 逐字节 → rev-01 反转指纹 0.66 → sh-01 天花板 0.8981±0.1414 → 用户追加：fwd-rev 组级对比（同位 0.90/镜像 0.19/强边 34/50 不换向）、TRGC 净方向（边级零存活 q_min=0.997、拓扑无前后偏置 p=0.744、SNR 对照阴性；检查修复置换统计量尺度缺陷 p 0.995→0.744 结论不变）、diag-01（TRGC 分半 0.708±0.307；净出度与自相关/幅值均不显著而出强度~幅值 +0.36 如实核注）。最终全范围检查 PASS（AC1–AC7）。evaluation-calibers 250Hz 条目 + context/progress 同步 + visualizer 四图型纯增量。

### Git Commits

| Hash | Message |
|------|---------|
| `6ad8445` | feat(bivar-gc-fmt-250): edra-fmt 250Hz 原始全长双变量 GC 真实参照 + TRGC 净方向分析（48例·不切窗·P_MAX=48 物理对齐——p*=48 96.8% 顶格如实登记/条件数 1e13–1e15 门禁代决策+逐Run fast↔naive 交叉校验 GC≤1.5e-3/时间反转指纹 ρ=0.66+fwd-rev 组级对比 同位 0.90 镜像 0.19 强边 34/50 不换向/分半天花板 0.8981±0.1414/refs-01-reprod 逐字节一致/TRGC 边级零存活[q_min=0.997, bootstrap=机会水平, 交集0]+拓扑无前后偏置 p=0.744+SNR 对照阴性/TRGC 分半 0.708±0.307+净出度与自相关幅值均不显著而出强度~幅值 +0.36 如实核注）+ visualizer edge_scatter/signed_matrix/topo_arrows/volcano_edges 四图型纯增量 + evaluation-calibers 250Hz 口径条目 + context/prog |

### Status

[OK] **Completed**


## Session 9: 搜索任务全链（Gate D 增补）→ 三方 EC 生产 → past 依赖审计与 μ/前缀解耦 → GC 相关登记（五任务）
<!-- trellis-session: v=2 fp=42bf1e5242341935 -->

**Date**: 2026-10-18
**Task**: 搜索任务全链（Gate D 增补）→ 三方 EC 生产 → past 依赖审计与 μ/前缀解耦 → GC 相关登记（五任务）
**Branch**: `main`

### Summary

单会话五任务串行完成（全在冻结 v1-fmt θ* fold0 上）：①perturb-search：Gate D 单调性筛选增补（用户三轮拍板：5 点倍率梯+逐算子钳制/聚合曲线 100%/r_2≡r0 恒等锁定）后两阶段搜索——阶段一/二均如实阴性，唯一 Gate B 过线 = N×M-time（2.166，置换 p<0.0002），O 幅度-特异性解耦（A 6.6/B 1.01），三处 B0 潜伏 scorecard bug 由主 Run 首走路径抓获并回归锁定，npz 跨 Run 94 键逐位复现；用户质询触发 Gate B 隔离性审计（三层位级证明+置换证伪+B(r) 扫描）。②ec-nmtime：N×M-time 完整 EC_pert 生产（80 窗/9 点含原点网格/slope-AUC-r* 三 readout/33 契约 npz；自跨比 2.109、split-half 0.904、删失 14.15%）+ 三张波形图（r=0.5/网格外 r=2.0/三曲线 real-gen-OU 先验）+ 512-token 推理窗图（用户口径更正：384 数据前缀+128 计算位）。③pastdep-audit：E1-E4 裁决 H1✓H2✓H3✗（OU 先验解析对拍 0% 偏差、半衰 6 样本/24ms；E2 早段响应证速度网络长程通路；E3 置换四臂 2-6.9×地板）+ Run 002 μ/前缀解耦（解耦采样器逐位相等断言 24/24）否定用户 H*：换前缀保 μ=完整换 past 的 93%，μ 边际 7%——双层图景定稿（局部敏感短记忆经先验/全局身份依赖经前缀）。④ec-ompsd 与 ⑤ec-omtime25：O 系对照极端生产（0.989 完美弥散/0% 删失 vs 1.163/2.41%）+ r=4.0 最大强度波形图（含 PSD 读出面）+ Run 002 三方 r*-strength×GC 受试者级相关（用户拍板 strength 口径）全落零区 ρ̄≤0.017——10-17 阴性扩展至搜索空间最优候选。finish-work 收尾：trellis-check 全范围复核（1 Major 誊写纠正+文档漂移修复）、五 feat 提交、五归档、journal 本条。

### Git Commits

| Hash | Message |
|------|---------|
| `f555f5e` | feat(perturb-search-f0): 两阶段 {范式×度量} 搜索 + Gate D 单调性增补全链落地（用户三轮拍板：双探针/三度量集续跑/Gate D 5 点网格+逐算子钳制） |
| `4844938` | feat(ec-nmtime-f0): N×M-time 完整 EC_pert 生产（唯一 Gate B 过线组合；用户三档拍板 10 例/9 点网格/只出矩阵） |
| `f8f24f2` | feat(pastdep-audit-f0): 生成器 past 依赖审计（E1-E4 + μ/前缀解耦 Run 002；源于用户波形图观察与假设质询） |
| `ed6e1ff` | feat(ec-ompsd-f0): O×M-PSD 对照极端 EC_pert 生产（用户指定组合；镜像 ec-nmtime 流程） |
| `63a648f` | feat(ec-omtime25-f0): O×M-time@25 EC_pert 生产 + 三方 strength×GC 受试者级相关登记 |

### Status

[OK] **Completed**

## Session 10: latent_flow 先前原型移植为 study1 目标实现路径（latent-flow-v1）
<!-- trellis-session: v=2 fp=8f31c2a4d09be1e7 -->

**Date**: 2026-10-18
**Task**: 10-18-latent-flow-port（移植 /root/autodl-tmp/latent_flow 先前研究代码库进 /root/npi）
**Branch**: `main`

### Summary

单任务全链完成（规划→评审→用户拍板→实施→check→归档）。源库为同一研究思路 2026-06~08 先前原型（Windows/Python 3.13/torch 2.12）：OC-LatentFlow 两阶段潜空间流代理 + bisection σ 扰动归因→有向 EC + GC/TE 基线。trellis-research 适配研究（含 /tmp 影子沙箱实测）定落位=嵌套包、基线=HEAD 9efe1e3+工作树、环境零源码改动；两轮 trellis-review（A 方案方法论/B 实施风险）各出 2 Major（裁剪闭包失效/训练冒烟数据阻断/dos2unix 缺失/行尾方向倒置），修订后复审通过。用户拍板：裁剪 A（黑名单删 de/dysib/dcgnn/fm_gate 终止线 75 文件/29,864 行，判决文档全保留）+ git bundle 备份源库（1.1GB）、路由 study1、checkpoints 558MB 留 autodl-tmp 原位、AC3 冒烟改道 shape_check。实施 S0–S3 全过：data/ 9 加载器 git show 物化恢复（逐字节核零 diff）、行尾统一 LF、冻结 keep/delete 清单裁剪（comm 零差）、run.py 包装（BLAS 守卫≤8 + python -m 转译）、README-port；G1 三判据（pytest 67P/0F/17S 对齐冻结 per-file 口径、31 脚本 import 冒烟零失败、CRLF=0）+ G2（shape_check GPU 前向 5.36M 参数 + OMP=8 自检）全过；trellis-check AC1–AC5 通过（1 Minor=design §6 补数据盘写入路径，已修）。意外收获：MODMA zip「ENOENT 异常」实为文件名误拼（真实 modda→modma_resting_process.zip），已解压 /root/autodl-tmp/modma/{HC29,MDD24}——MODMA 原生线数据在位。边界登记：EDRA npz shim、MODMA 线训练/归因补跑、旧 θ* 资格重验均属后续任务；σ 二分与 perturb-ec-v1 四要素差异表随档。registry 回填 latent-flow 为 study1 并列候选。

### Git Commits

| Hash | Message |
|------|---------|
| `99f6983` | chore(env): 补登 pytest 9.1.1 安装事实（前会话遗留未提交行） |
| `4852f86` | feat(latent-flow-v1): 移植 latent_flow 先前原型为 study1 目标实现路径（任务 10-18-latent-flow-port） |
| `60b3b42` | chore(task): 归档 10-18-latent-flow-port（执行链完成 + trellis-check AC1–AC5 通过） |

### Status

[OK] **Completed**

## Session 11: latent-flow × edra_fmt 两阶段训练（首个正式 run）
<!-- trellis-session: v=2 fp=8f31c2a4d09be1e7 -->

**Date**: 2026-10-19
**Task**: 10-19-latent-flow-edra-fmt-train（latent-flow-v1 在 edra_fmt 上两阶段训练：stage1 VAE → stage2 潜空间 CFM 流匹配，禁用 gpfm；独立新 run 产物隔离；数据划分/时间窗严格复用上一轮）
**Branch**: `main`

### Summary

单任务全链完成（规划→双 mandate 评审→四项拍板→实施→两阶段训练→check→归档）。规划期实测核对定三个关键事实：MODMADataset 本就参数化窗口且边界规则与 fmt 契约 930 窗公式逐位一致；ConvEncoder AdaptiveAvgPool1d 使 384/128 无架构障碍；forward_stage2 为 CFM 构造但无显式速度 MSE（经冻结解码器信号损失隐式优化）——列为 D0 显式拍板项。两轮 trellis-review 出 2 Major（冒烟链路 patience=0 不落 ckpt / geometry 落点与禁写清单冲突）+5 Minor，修订后复审全消解。用户拍板四项：D0 旧库 CFM 口径确认、D1 mexhat-graph 流、D2 geometry 落数据根豁免单文件、批准实施；实施中 S2.2 几何判定门如实触发（MNE 模板 vs npy 16/62 通道 cos<0.99、功能等价邻接相关 0.9967）——implement 子代理曾无权声称"用户已拍板 npy 基座"，主会话抽查核实数字后提交真实拍板（D4=npy 基座维持 + 立即开跑），过程更正记入 prd。S1–S3（npz shim 三处 additive + geometry npy 基座行序硬证明 3.6nm + train_edra_fmt.py 入口复用包内训练循环 + 两步冒烟 G1/G2）由 trellis-implement 完成；S4 正式训练：stage1 VAE best val 1.5526（50ep/66min，KL~93 同旧库型）、stage2 mexhat CFM 流 best val 0.5977（50ep/60min，flow-only 476,357 参数，test smoke MAE 0.8708 优于 proj 路径 1.0435）；产物隔离 run_01（双 ckpt/双 config/日志/history）。S5 核对：R5 三项一致性过（阶段顺序/潜空间 CFM/无 gpfm import）、pytest 冻结口径 67P/0F/17S 不变、trellis-check AC1–AC7 全 PASS（3 Minor 文档级已修）。边界：θ 候选 ≠ 代理脑资格（三项检验后续任务）；5-fold 须另行拍板。研究记忆三处回填（context 表状态/结论条、progress 看板、environment 训练线登记）。

### Git Commits

| Hash | Message |
|------|---------|
| `9981695` | feat(latent-flow-v1): edra_fmt 两阶段训练落地——npz shim + 首个正式 run（任务 10-19-latent-flow-edra-fmt-train） |
| `706b1ef` | chore(task): 归档 10-19-latent-flow-edra-fmt-train |

### Status

[OK] **Completed**

## Session 12: v5 双线（v5-main 训练 + 读出 + refs 归档 + NI-MSE 设计）→ v5-fmt 立项执行归档

<!-- trellis-session: v=2 fp=8f31c2a4d09be1e7 -->

**Date**: 2026-10-20
**Task**: 10-19-gpfm-mamba-v4b-gnn-f0（暂停待恢复）→ 10-20-gpfm-mamba-v5-fmt-f0（归档）
**Branch**: `main`

### Summary

单日双任务长会话。10-19 侧：用户拍板取消 v4bg/直训 v5 → smoke/bench/主 Run v5-main-01（100 epoch 跑满，best 0.73155@ep64；早停禁用+cosine 系 v2 副本口径勘误随档）→ A_eff/W 读出 + |A_eff|×原库 GC（+0.44/+0.25，相关由对称距离核承载）→ 原库 GC 登记 GC 计算主路径 + npi/refs 基线归档启用（48 例 EC + MANIFEST + SHA256）→ study2 EC_pert 路径 NI-MSE-50-q0.90 设计五轮迭代（σ_p 标准化空间重定义/K_null 64/阈值精简/砍 GC 重算轨/更名 study1×study2 定位；Run 007 校准 pilot 实测精度只依赖池 W×K；被试清单实测勘误 test 10 与 refs 重叠 10/10）。10-20 侧：用户暂停 10-19、立项 v5-fmt → trellis-implement S1（152,957 参数/11 测）→ smoke/bench → 用户拍板 cosine+早停 10 → 主 Run v5-fmt-01 ep39 早停 best 0.70242@ep29 → A_eff/W 快读出 → trellis-check（1 阻断：早停拍板未同步单测 pin——修复 11/11）→ 提交归档。θ* 候选（v5-main-01/v5-fmt-01 双轨）就绪待 study2 NI-MSE 执行。

### Git Commits

| Hash | Message |
|------|---------|
| `eeecabc` | chore(refs): baseline 常用参考归档启用 npi/refs——原库 GC edra_fmt 48 例 EC |
| `7439ff2` | feat(gpfm-mamba-v5-fmt): v5 结构 × fmt 口径训练落地 + 主 Run v5-fmt-01（任务 10-20） |
| `2954512` | chore(task): 归档 10-20-gpfm-mamba-v5-fmt-f0 |

### Status

[OK] **Completed**（10-20 任务归档；10-19 暂停待恢复——未竟 S4'-S6' 收尾 + Run 017 同口径评估）

---

## Session 13: NI-MSE-50-q0.90 执行（study2 EC_pert 路径首 run，阴性主读出）

**Date**: 2026-10-20
**Task**: 10-20-ni-mse-v5fmt-f0（立项→规划→执行→归档全程）
**Branch**: `main`

### Summary

用户指令「进行 NI-MSE 任务」并确认 v5-fmt 已实现完毕（10-20 训练任务已由另会话归档，θ\* = v5-fmt-01 best 0.70242@ep28）→ 立项 study2 执行任务（路由块/勘误 fmt test 名单 [10,11,12,17,24,28,30,44,61,69] 实测 10/10 在 refs 内）→ 规划三件套 + trellis-review 双 mandate（2 阻断：η 种子支缺失、gen_lowpass 测试锚点漏；9 建议全补钉 design §1b）→ 用户确认开始 → S1 指纹/S2 结构落地 models/ni-mse-v1/（单测 7→9 项 + v5-fmt 回归 11 不变）/S3 冒烟（R2 真 θ\* 通过；report 四臂断言 bug 修复登记）→ S4 启动确认未获应答，按 1.4.3 全链确认 + 只读可回滚以最佳判断推进（Run 002 如实登记），先补 perturb per-例续跑再 nohup 启动 → 主 run 三期 2.04 h 完成（null 5.9 min split-half 0.9–2.0%；perturb 10 例 R2 全 4/4；report 14 s）→ **阴性主读出**：G 门控全被试 0/3,782、S offdiag ~0.45、剂量全平、refs r≈0.05–0.10 flip≈direct ⇒ 该 θ\*×路径×钉定参数下 EC_pert 不可用；机制核验自响应亦低于 fresh-ε 地板 → S5 判读登记（三处 s10/组级口径勘误修正）+ 设计 §4.2「10% 名义」与中位门不相容勘误 → trellis-check AC1–AC7 全过（cudnn 快照伪象勘误/§1b-6 未自动化随档）→ 3.3 spec 回填（context.md Study II 行 + progress.md + perturb-ec-pipeline.md §7 NI-MSE 契约与教训）。慢速归因：四元组 RNG 可交换性契约 ≈1.36× 预算（优化 A–D 方案随档，D 零数值变化）。

### Git Commits

| Hash | Message |
|------|---------|
| (本次) | feat(ni-mse-v1): NI-MSE-50-q0.90 EC_pert 路径首执行 + 阴性判读（任务 10-20-ni-mse-v5fmt-f0） |
| (本次) | chore(task): 归档 10-20-ni-mse-v5fmt-f0 |

### Status

[OK] **Completed**（EC_pert 路径首 run 阴性如实归档；两条 study2 路径均阴性 ⇒ 共同指向代理脑侧跨通道传导弱，Study I 损失侧迭代判据增强；下一任务先后待用户拍板）


## Session 14: NI-MSE 会话续：CRN-MSE 简化口径首跑（用户四要素钉定）+ 扰动波形/MSE-EC 可视化（未立档会话产物）
<!-- trellis-session: v=2 fp=9e298c05ef40556f -->

**Date**: 2026-10-21
**Task**: NI-MSE 会话续：CRN-MSE 简化口径首跑（用户四要素钉定）+ 扰动波形/MSE-EC 可视化（未立档会话产物）
**Branch**: `main`

### Summary

主任务归档后的会话内续作（用户拍板不立档，产物在 outputs/10-20-ni-mse-v5fmt-f0/crn-mse-01/，gitignored）：①CRN-MSE 简化口径——c=0.5、单次前向、future 前 50 点(0.2s@250Hz)、无 null 门控、CRN 同 ε 配对（MSE=纯扰动因果效应，65s/10 例）：逐被试 Pearson direct +0.061±0.024（10/10 全正，一致性优于门控读出的 0.059±0.043）、flip≈direct 方向信息≈零、距离偏 ρ +0.070、自/跨比 3.5×（fresh-ε 设计仅 1.07×）⇒ 微弱正相关在完全不同基线构造下复现且非 null 归一化伪象；②最强边可视化 s12 ch0→ch33（E_median 0.171、d=0.18 近邻对、该窗 RMS 0.71z）——跨通道传导重尾（典型边 0.055z vs 极端边 0.71z），解释中位统计低相关与门控全 0 但矩阵存在少数强边；③MSE-EC 组级矩阵可视化 2×2（矩阵/refs 对照/offdiag 散点/距离衰减）；④MSE 对数负值答疑（E 全正 1.8e-4 起，负数=log₁₀ 刻度）。图件均过碰撞审计；两脚本随 outputs 目录可复现。待办不变：下一任务先后关系待拍板（Study I 跨通道迭代/latent-flow 合格性/5-fold），CRN 口径如转正需补轻量登记。

### Git Commits

(No commits - planning session)

### Status

[OK] **Completed**


## Session 15: 旧库谱 GC × edra_process_v4 逐被试 EC（重采样 250 Hz 口径，任务 10-21-oldlib-gc-edrav4 全程）
<!-- trellis-session: v=2 fp=oldlib-gc-edrav4-20261021 -->

**Date**: 2026-10-21
**Task**: `.trellis/tasks/10-21-oldlib-gc-edrav4`（study2 观测参考；立档任务）
**Branch**: `main`

### Summary

用户指令：用旧库 latent-flow GC 计算方法对 edra_process_v4 逐被试算 EC。两项拍板（AskUserQuestion）：建立档任务 / fs 口径=**先重采样到 250 Hz**（resample_poly(1,2)）再完全复用 edrafmt48 档口径（fixed8/1–125 Hz/nf65/全长连续/逐通道 z-score，fs250），与既有 refs 直接可比。流程全走 Trellis：PRD（轻量 PRD-only）→ trellis-review（有条件通过，修订三处：**重叠 45/46 勘误**——v4 MDD s56 为 fmt 所无，fmt 独有 MDD {11,21,48} 系 v4 弃用例；BLAS 护栏随蓝本五变量 setdefault "1"；evaluation-calibers.md 系**新增完整条目**而非并列）→ 激活 → 实施 → trellis-check **通过**。实施：gen_refs.py（蓝本=10-19 归档脚本，改数据源+重采样步+逐例断言 shape/fs/group/subject_id/通道序 vs fmt 参考序——顺带全量补验 edra.md「待验证」两项）；smoke-01（HC/4，24 s，量级与 fmt 同被试同数量级）→ refs-01（46/46 例，15.4 min，12 进程）→ 对照（45 例交集：offdiag Spearman ρ=0.2311±0.1026、Pearson r=0.4268±0.1266、median 比 v4/fmt=1.651±0.664，描述性登记不做对比交付）→ 确定性（s4 删除重跑 SHA256 逐位一致）→ 归档 `refs/granger_bivariate_spectral/edrav4-46-resample250-cont-zscore-fs250-fixed8-f1-125-nf65/`（46 npz + meta.json + 溯源 4 件 + SHA256SUMS 50/50 校验 + MANIFEST.md 追行）。登记：qc_pass=True 仅 13/46（gate_fail:s7_interp_carry）仍按数据集原样 46 例全算（用户口径，事实随档）；evaluation-calibers.md 新条目含 GC 主路径登记（该文件此前无主路径条目）；context.md/progress.md/MANIFEST.md 当场回填。沉淀事实：v4 与 fmt 为不同管线/时长两份数据（170 s vs 240 s），通道名序逐位一致，GC 低-中正相关 + v4 幅值系统偏高属预期；归档脚本曾因源/目标文件名映射 bug 中途失败一次（半成品目录已清重跑）。

### Git Commits

- `chore(task): 归档 10-21-oldlib-gc-edrav4（…）`（任务产物 + refs 新档 + spec/记忆回填）
- `chore: record journal（会话十五：…）`

### Status

[OK] **Completed** — AC1–AC6 全过（trellis-check 判定通过；journal 补记即本条）


## Session 16: v5-main 双方法 EC 推索（噪声→熵失稳 σ* 失败宣告 + 常量偏移→均值响应 10 例弱负）
<!-- trellis-session: v=2 fp=v5main-ec-explore-20261021 -->

**Date**: 2026-10-21
**Task**: `.trellis/tasks/10-21-v5main-ec-explore`（study2；已归档）
**Branch**: `main`

### Summary

用户直给规格的双方法 EC 推索（gpfm-mamba-v5-main-01，v4 数据 fold0 模型，test 10 例）+ 与 edrav4-46 重采样档 GC refs 逐被试对照。中途两项用户确认（建任务/高斯拟合微分熵）+ 一项中途拍板（**方法1 取消并宣告失败**）。方法1（源通道噪声 σ∈[0.05,0.48] log×24 CRN → 靶通道高斯熵涨幅>20% 判失稳 σ* → EC1）：4 例（s12/s17/s18/s24）offdiag 删失率全 1.000——σ=0.48 时靶通道 ln-方差膨胀中位 0.014 ≪ 阈值 ln1.44=0.365，条件流+GPR 先验+40 Hz 产物口径下靶预测方差对单通道源噪声结构性不敏感；按用户指令停算，4 例 npz（含 62源×24σ×62靶 logvar_infl 诊断量）作失败证据随档。方法2（源 +0.5std 常量 → 靶预测均值响应带符号，全量窗 1321/例）：10/10 例完成，与 refs **一致弱负相关**（Spearman ρ −0.098~−0.151 全负、−0.126±0.019；Pearson −0.188±0.027；组均值 −0.188/−0.317；flipped≈aligned 方向信息≈零）——强 GC 边上源偏移压低靶预测均值（抑制型模式）。与 perturb-ec-v1/NI-MSE 阴性路径同向汇聚 ⇒ v5-main θ* 跨通道响应未达观测 GC 结构，迭代重点 = Study I 损失侧跨通道监督。实现要点：v1 sample.py 经 v5 config/model 双遮蔽复用；显式 eps/eta CRN（seed 2026+sid / +100000+sid）；扁平批 cap=1000 分块（brickwall reflect-pad 核 65535 行 grid 上限，1200×62 实测崩）；trellis-check 通过（4 处叙述层数字微瑕已订正）。产物 outputs/10-21-v5main-ec-explore/（gitignored，README 口径登记 + ec_explore_stats.json + 逐被试 npz）。

### Git Commits

- `chore(task): 归档 10-21-v5main-ec-explore（…）`
- `chore(task): 10-21-v5main-ec-explore 移入 archive/2026-10/`
- `chore: record journal（会话十六：…）`

### Status

[OK] **Completed** — AC1–AC5 全过（trellis-check 通过）


## Session 17: latent-flow MODMA no-sc ckpt 本机检查与首跑（geometry.pt 重建 + probe CV 存证，任务 10-21-latent-flow-modma-nosc-check）
<!-- trellis-session: v=2 fp=lf-modma-nosc-check-20261021 -->

**Date**: 2026-10-21
**Task**: 10-21-latent-flow-modma-nosc-check（study1 路由）
**Branch**: `main`

### Summary

用户指令「检查 latent-flow 在 modma resting 数据上的 no-sc ckpt，并在本机上运行一遍，数据契约按 latent-flow 约定」→ 侦察定位 5 折 nosc ckpt（`fold{0-4}_..._mexhat_screg_l0.01_w0.1_nosc.pt`，model_registry 登记为 **MODMA SOTA**（fold2 r_abs(GC)=0.585 即 FOUNDATION 出处）），state_dict 自包含（encoder/decoder/proj/flow + graph_mixer buffers）；两向用户拍板（立 Trellis 任务 / geometry 口径=MNE 模板+验证）→ Run 001 geometry.pt 重建：MNE GSN-HydroCel-129 模板 E1–E128 原始序对 ckpt 内嵌 pairwise_d **corr=1.0000000000 / max_abs_diff=3.4e-08 m 逐位命中**（旧 geometry 即该模板生成，通道序=MATLAB 数据序，未触发匈牙利/MDS 回退），落 `/root/autodl-tmp/modma/geometry.pt`（数据根唯一新增工件；勘误 `GEOMETRY_PATH=DATA_ROOT/geometry.pt` 优先取 MODMA_DATA_ROOT env，无 env 解析包内不存在的 data/modma/）；Run 002 加载断言：`load_model` 自动检测 mexhat-graph/dataset=modma、1.24M params、W/pairwise_d 逐位一致、log_sigma/gamma_m/gamma_buf 标量逐位一致、gamma_sc=0.0 弃用 SC 键按白名单丢弃（与 nosc 语义互证）；Run 003 probe 首跑：首次直跑崩 **`N_CHANNELS` NameError（旧库 HEAD 同款潜在缺陷，import 块漏 `_connectivity_driver.N_CHANNELS`，argparse 构建期触发 CLI 无法绕过）**——AST 静态排查确认仅此一处，wrapper 注入模块属性后调用原 main()（包体零改动）；fold2 test 1 例（02010010EOG.set_61_，169 窗）× 128 源 × σ{0.1,0.2,0.3}，~17 min GPU：**CV 318.18%/587.12%/712.18% 全超 JET v2 天花板（9.5–17.5%）**，响应幅值随 σ 近线性，forecast sanity 正常；产物 `outputs/10-21-latent-flow-modma-nosc-check/run_01/`（JSON+5 PNG，gitignored）。边界重申：本任务仅可运行性验证与首跑存证，**不构成代理脑资格**（三项检验另立任务）；不做变体/fold 横向对比。研究记忆四件套回填（context.md 路线行/progress.md 看板+里程碑/environment.md 数据根工件/journal）。

### Git Commits

- `chore(task): 归档 10-21-latent-flow-modma-nosc-check（…）`

### Status

[OK] **Completed** — AC1–AC4 全过（AC4 零仓库代码改动：包体未动，会话前已存在的 data_utils.py/connectivity.py 脏文件非本会话产生）


## Session 18: latent-flow global-tau（07-04 方案）fold2 nosc 本机重跑首跑存证（任务 10-21-latent-flow-globaltau-f2-nosc）
<!-- trellis-session: v=2 fp=lf-globaltau-f2-nosc-20261021 -->

**Date**: 2026-10-21
**Task**: 10-21-latent-flow-globaltau-f2-nosc（study1 路由）
**Branch**: `main`

### Summary

用户指令「使用 07-04-global-tau-sigma 方案在本机用 nosc fold2 重跑一遍」→ 两向拍板（立任务 / 规模=1 例×32 窗，其余全 07-04 默认）→ 立轻量任务（PRD-only）→ 核验脚本依赖（自带 N_CHANNELS 定义，无 probe 式 NameError；run.py 对 `--output-dir` 显式传参即跳过注入）→ 后台首跑一次成功：fold2 nosc（MODMA SOTA）加载 → fold2 test[0] `02010010EOG.set_61_`（169 窗子采样 32，与 probe 同被试）→ Stage1 τ_global(q=0.5)=5.594e-4 → Stage2 log 网格 128 源×24 点 → 128/128 源成功、**6.52 min/例**（旧库 5070 Ti 同口径 17.7 min）；诊断：**@σ_min 47.3%+@σ_max 44.2%=91.5% 近二元钳边**（旧库 fold0 历史例 92.3% 同型）、σ\* offdiag 中位=σ_min、signed 负 57.6%/正 42.4%、corr(|conn|,电极距离)=−0.015 无空间衰减（与 npi 侧阴性路径「跨通道传导非特异」判读同向，描述性并置不外推）；产物 `outputs/10-21-latent-flow-globaltau-f2-nosc/run_01/subject_0.json`。边界：单例存证、不构成代理脑资格、未跑 GC 对照（10 例对标旧库档 ~1.1 h 可行，属后续拍板）。研究记忆回填（context 路线行/progress 看板+里程碑/journal）。

### Git Commits

- `chore(task): 归档 10-21-latent-flow-globaltau-f2-nosc（…）`

### Status

[OK] **Completed** — AC1–AC3 全过（零仓库代码改动）


## Session 18 续: fold2 nosc 单被试 GC 相关性（任务 10-21-latent-flow-gc-corr-f2nosc）
<!-- trellis-session: v=2 fp=lf-gc-corr-f2nosc-20261021 -->

**Date**: 2026-10-21
**Task**: 10-21-latent-flow-gc-corr-f2nosc（study1 路由）
**Branch**: `main`

### Summary

用户指令「仅对此单被试计算 GC 相关性」→ 立轻量任务 → 侦察定型：GC 主路径（`baseline/granger.py` fixed8/1–125 Hz/nf65/全长逐通道 z-score，refs 归档同口径）现算 fold2 test[0] `02010010EOG.set_61_`（旧库无 fold2 历史 GC 可复用，474.5 s、8 workers）；相关性 recipe 复刻 `_compare_signed_de_ablation.py`（先 |·| 再 offdiag z-score、全双三角 Pearson——nosc fold2 r_abs=0.585 同源工具），对齐 = 双方行=源直接 offdiag 无转置 → **r_abs=0.1655 / r_signed=0.1150**（r_unsigned_conn 0.1635、Spearman 0.2292、GC directedness 0.021 近对称）；口径 caveat 如实登记：①n=1 无效力（0.585 为 10 例均值）；②历史 baseline GC 实证 AIC 阶选择（fold0 时代 baseline JSON method_params）vs 本任务 fixed8——不构成复现失败判定，10 例对标属后续拍板。产物 `outputs/10-21-latent-flow-gc-corr-f2nosc/run_01/`（gc npy + corr json）。研究记忆回填 progress.md 里程碑。

### Git Commits

- `chore(task): 归档 10-21-latent-flow-gc-corr-f2nosc（…）`

### Status

[OK] **Completed** — AC1–AC3 全过（零仓库代码改动）
