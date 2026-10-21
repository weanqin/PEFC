# gpfm-mamba-v5：v2 基线 × 轴分解交替 × GNN 先验矩阵融合（fold0 前置验证）

> **2026-10-19 二次转向（用户拍板，本 PRD 以此为准）**：弃用 v4b(ch1) 基座与"顶部单点图头"方案（v4bg S1/S2 产物停用留档）；改为**以 v2 为基线的新变体 v5**——d_model=C=62（通道即模型维、零折叠）、**时间块（Mamba）与空间块（GNN，A_eff 先验矩阵）逐块交替**、每块后 LayerNorm、以 v5 名称注册实例化、保持 v2 接口/config/调用流兼容。研究内核不变：融合 latent-flow 的 GNN 先验矩阵学习（A_eff = γ_m·M(d;σ) + γ·W）。对照列改 v1/v2/v5；训练/评估/R9 链路与判定读出处置沿用。详见 design.md（v5 版）与 implement.md。

## Goal

在 gpfm-mamba-v4b **d_ch=1**（用户指名 "v4b-ch1"，即 H3 的 v4b-d1 臂基座）之上，融合 latent-flow-v1 的 **LearnableMexHatMixer**（几何 mexhat 距离核先验 M(d;σ) + 可学习**有向**残差矩阵 W，`A_eff = γ_m·M + γ·W`）：以其图消息传递形态替换 v4b cMLP 顶层读出头的稠密跨通道混合，产出新的 Study I 变体并做 fold0 单 seed 前置验证（结构 + 单测 + 训练 + 正式口径评估 + A_eff/W 可解释读出）。

## 背景与动机

- H3（任务 10-16，已判）："跨通道混合可局部移植为 GC 载体"**不支持**——v4a rank-16 输入接口与 v4b cMLP 顶层读出（d1 0.133 / d8 0.163）两臂 GC 一致性均不回升（v1 0.2755）；v4b 落选 5-fold 候选集。
- Study II 首验证（任务 10-17-perturb-ec-f0，已判）：θ* 的缺口四重证据定位为**跨通道传导弥散无特异性**（空间衰减平、D̃ 近 r 无关、自/跨比仅 1.27×）——Study I 迭代需要能让传导"有结构"的混合载体。
- latent-flow 线（任务 10-19）在 edra_fmt 上的 stage2 流即以 LearnableMexHatMixer 为通道混合位（ricker 核、γ=0.1、γ_m=1.0、W 零初始化自由学习，未启用正则）。
- 本任务检验的假设（H4）：**几何先验约束的跨通道混合位（相对 H3 的自由稠密 cMLP）能否成为 GC 载体**——即固定距离核（中心激励+环绕抑制的带通先验）钉住"在哪里混"，可学习有向 W 只学"偏离几何的连接"，直接回应传导特异性缺口。

## Requirements

### R1 新变体结构落地（新目录，旧目录零改动）

- 新变体目录 `models/gpfm-mamba-v4b-gnn/`（命名可在评审时改）：config.py + model.py + tests + train/evaluate 双遮蔽包装（v4a 变体目录惯例）；mamba/data/gpr_prior/losses/train 循环只读复用 v1。
- 基座 = v4b d_ch=1 全套口径（CI 骨干 + 逐通道 RMSNorm + future 128 截取），**唯一结构变化 = `_head` 的稠密 cMLP（W1/W2 einsum，62 子 MLP）替换为 GNN 先验图头**（LearnableMexHatMixer 的适配移植，见 design.md）。
- 通道混合哲学保持"单一可解释混合位"：trunk 保持通道独立，跨通道混合只发生在图头。
- 图头要素：per-channel 输入投影 → K×{A_eff@X + per-channel Linear+GELU} → per-channel 输出投影；σ 可学习（ricker 核）；W 为 C×C 有向、零初始化（支持 w_init 注入）；γ/γ_m/K/kernel(ricker|dog) 为 config 字段；uniformity/sparsity 正则旋钮在 config 暴露、**默认 0（关闭）**——对齐 10-19 edra_fmt run 的实际口径。
- 电极几何：只读复用 `/root/autodl-tmp/edra_processed_fmt/geometry_edra_fmt.pt`（62ch 10-20 npy 基座，D2/D4 拍板产物；旧范式 edra 同蒙太奇，不新建几何文件）。latent-flow-v1 源码全程只读。

### R2 单测与等价性

- 组件级等价单测：图头 A_eff 的构造（同 coords/σ/γ/W 下与 latent-flow 源 `LearnableMexHatMixer.effective_adjacency` 数值一致——ricker 与 dog 两核）。
- 变体单测（v4b test_v4b.py 惯例迁移）：前向 shape / 因果性（future 截取） / 梯度有限 / 参数量精确登记 / config↔model 一致 + hash 稳定 / 双遮蔽与前缀守卫。
- 既有变体回归：v1（7 测）、v4（test_v4b 7 项）、v4a 全绿且 config_hash 零漂移（v1 `c6ff80ac1c43` / v2 `fc2a9ce14948` / 其余以现状为准登记）。

### R3 fold0 训练（R9 门禁链路）

- 口径沿用 fold0 累计纪律：32/4/10 划分（复用 v4a/v4b 既有 fold0 manifest，只读）、batch 256、seed 1234、EMA 0.999、早停 patience=10、恒定 lr 1e-3、gen_lowpass 40 Hz 正式口径。
- 链路：`--smoke` 冒烟 → `--bench` 1-epoch 基准（含图头新增参数量/显存/it/s 登记）→ **主 Run 须用户确认后启动**（R9 硬门禁，nohup 脱离式）。
- 数据：旧范式 `/root/autodl-tmp/edra_process_v4`（500 Hz、past=384/future=128、62ch）。

### R4 正式口径评估 + 对照

- `evaluate.py` + `fc_metric.py`（Run 017 完全同口径：Pass A K=16 seed=1234 / Pass B K=1 seed=1235 / GC p_max=10 bootstrap 1000 / PSD nperseg=512 / 空间 σ=0.080 / gen_lowpass 40 Hz）。
- 七维度多列对照：v1 main-01 / v4a-r16-main-01 / v4b-d1（v4b-main-01）/ **本变体**（test 折四方相同）；三检验（指标1 v2.1 / 指标3 分频段 / 指标2 轨迹图备好交用户目视）+ R1 附分子分母 + std_t 照登。

### R5 A_eff/W 可解释读出

- 产物：A_eff、M(d;σ*)、W 三个 (62,62) 矩阵热图（可视化统一 visualizer 规范）+ 学习后 σ 值登记。
- 对照读出（类比 v4b readout_diag）：W 与 A_eff 分别对 GC(gen)/GC(real) 的 Spearman ρ；与 v4b-d1 的 M[j,i] 读出（0.240/0.049）并列展示。

### R6 边界与不做

- **fold0 单 seed 前置验证 only**——5-fold 须另经用户同意（study1 训练顺序门禁）。
- 本任务不做扰动 EC 推断（Study II 侧）、不做代理脑合格性三项检验判定（θ 候选 ≠ 资格，另立任务）。
- 不改动 latent-flow-v1 / gpfm-mamba-v4 及其余变体目录任何文件；不启用 W 正则消融臂（旋钮落地即可，跑不跑另议）。
- 产物落 `outputs/gpfm-mamba-f0/<run_id>/`（沿用 f0 根、run_id 前缀隔离，H3 惯例）。

## Acceptance Criteria

- [ ] AC1 新变体目录落地（R1），pytest 全绿（新测 + v1/v4/v4a 回归），既有变体 config_hash 零漂移（R2）
- [ ] AC2 组件级等价单测通过：图头 A_eff 与 latent-flow 源实现在 ricker/dog 两核下数值一致（rtol=1e-6 级）（R2）
- [ ] AC3 fold0 训练链路按 R9 门禁走完：smoke/bench 指标与产物登记齐全，主 Run 经用户确认后启动并完成（早停或跑满），best ckpt + config + 日志齐备（R3）
- [ ] AC4 正式口径评估完成：七维度对照表（v1/v4a/v4b-d1/本变体）+ 三检验材料落盘（指标1/3 对拍板阈值逐项过/不过陈述）（R4）
- [ ] AC5 A_eff/M/W 读出产物齐备：EMA shadow 口径导出、行=源/列=靶方向约定（转置单点落位）、热图 + σ 登记、双轨对照（判定轨=对 GC(real) 距离偏 Spearman + t/bootstrap 显著性；描述轨=朴素 ρ + 双方向指纹，GC(gen) 侧明示循环性）（R5 + 评审处置）
- [ ] AC6 H4 三态判读结论如实落 experiment.md（GC 一致性是否回升 / 传导结构性是否改善——以 AC5 读出与 AC4 GC 双维为据，交用户拍板）
- [ ] AC7 trellis-check 全范围通过

## 拍板记录（1.4.3 同意门，2026-10-19）

用户答复：「对照不加入d8，开启」——显式同意启动；未逐条答复项按简报默认生效：

1. **基座口径 = d_ch=1**（随「开启」确认简报默认；阴性结果按"限 d1 基座不支持"表述）。
2. **判定读出修式采纳**（距离偏 Spearman 判定轨 + GC(gen) 描述性降级）。
3. **中间带处置 + δ=0.02 预注册采纳**。
4. **latent-flow-v1 三处未提交改动：悬置**——本任务全程不触碰；零改动核对以 S1.0 git 快照现状为基线（归属处置另议，不阻塞本任务）。
5. **热图渲染 = 变体目录自绘**（不依赖 latent_flow.viz 未提交函数）。
6. **对照列不加 v4b-d8（用户显式拍板）**——四列：v1 main-01 / v4a-r16-main-01 / v4b-d1 v4b-main-01 / v4bg。

## Notes

- 复杂任务：`design.md` + `implement.md` 随本 PRD 一并评审（2026-10-19 双 mandate trellis-review 已过——两份均「有条件通过」，全部发现已处置入 design/implement，拍板项见上），评审通过并经用户明确同意后方可 `task.py start`。

## 子研究路由（1.0b / 1.0c）

<!-- trellis-research-routing -->

> 任务创建时由 hook 自动播种。规划阶段必须填写以下四项；全部落定前不得进入 1.4 start 门禁（见 `.trellis/workflow.md` Phase 1.0b / Phase 1.0c）。

- **分类结果**：`study1`
- **依据**：EEG 代理脑（Study I）模型结构变体——v4b 基座融合图先验混合位的 fold0 前置验证（训练/评估/选型证据）。
- **新路径/方法检测（1.0c）**：`已有路径：Study I gpfm-mamba 变体线（context.md 技术路线表第 1 行，v4b 为已登记变体）+ Study I 候选 latent-flow（第 3 行，LearnableMexHatMixer 为其已登记组件）`——两法均已在册、代码均在库，属变体级融合（H3 判例：薄变体记已有路径）；3.3 时回填变体注记。
- **research 产物**：`无`
