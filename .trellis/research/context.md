# 科研方向档案（长期记忆）

> **维护规则**：SessionStart hook 把本文件完整注入每个新对话；重大节点/决策后当场更新；保持 ≤80 行，细节写入「关键文档地图」指向的文档。
> **双重身份**：「技术路线（当前活跃）」表同时是任务路由（workflow 1.0b）与新路径/方法检测（1.0c）的对照 registry；新方法经 research 采纳后须在 3.3 回填此表。
> **初始化**：2026-10-10，依据用户提供的《基于 EEG 数据的虚拟扰动推断研究框架》定稿填写。

## 研究主题

**基于 EEG 代理脑虚拟扰动的节点连接推断**：先训练「过去 EEG → 未来 EEG」的条件生成代理模型（surrogate brain），再在代理脑内做虚拟扰动实验，以「临界扰动强度」定义节点间有向连接。权威框架文档：[research-direction.md](../../docs/research/research-direction.md)。

直接参考文献：Luo et al., *Mapping effective connectivity by virtually perturbing a surrogate brain*, arXiv:2301.00148v3（NPI）。**口径边界**：参考文献只给出「代理脑 + perturb-and-record」框架；「过去 EEG→未来 EEG」任务定义与「临界扰动强度」推断方法均为本研究自主设计，不是参考文献已给结论。

## 技术路线（当前活跃）

| 路线 | 一句话方法 | 代码位置 | 状态 |
|---|---|---|---|
| Study I：EEG Conditional Future Generation | 训练代理脑 X_past→X_future（时序预测/条件生成/时间修补三视角），含 EEG 表征、未来生成、代理脑有效性三层次 | `models/gpfm-mamba-v1/`（+v2/v4b/v4a/v1-fmt/v4a-fmt/v4a-fmt-adaln 变体目录） | gpfm-mamba v1/v2/v4 已实现，fold0 单 seed 前置验证（任务 10-12）与 H3 消融（任务 10-16）完成；**5-fold 候选集已拍板 {v1, v4a}**；新范式 edra-fmt fold0 前置验证完成（任务 10-16-gpfm-mamba-fmt-f0，2026-10-17，LayerNorm/p=25/无低通/GC 阻塞口径）；adaLN(t) 消融中性结果（任务 10-17-gpfm-mamba-adaln-f0，2026-10-17，排序不变）；5-fold 主实验待启动 |
| Study II：Critical Perturbation Connectivity Inference | 固定 θ* 后扰动源节点 A→测目标节点 B 响应→搜临界强度 r*_A→B=inf{r:Δ(r)≥τ}→临界扰动矩阵→连接强度 | `models/perturb-ec-v1/`（θ\* 只读引用 Study I 产物） | **四要素首版已定义 + fold0 流程验证完成**（任务 10-17-perturb-ec-f0，2026-10-17）：CRN 确定性契约 + 增益算子 + τ 双口径 v2（操作性 τ̃=0.25 相对自响应）+ readout 三件套；v4a-fmt θ\* 上**未获结构收敛初步证据**（缺口=跨通道传导特异性，非流程问题）；流程规范首版见 spec/research/perturb-ec-pipeline.md |

Study I 是 Study II 的前置（先建立可信代理脑，再做虚拟扰动）。注意：**预测质量 ≠ Surrogate Brain Validity**，须作为两个独立问题评价。

## 既有结论（已验证）

尚无实验结论。已定案的是框架层事实（见框架文档）：两阶段流程（Stage I 训练并固定 → Stage II 扰动推断）、14 步完整流程、核心四问 Q1–Q4、术语表。

2026-10-12 定案：数据集为 **edra**（`/root/autodl-tmp/edra_process_v4/`，HC 24 / MDD 22 受试者级 npz，62 通道、500 Hz，契约 `.trellis/spec/data/edra.md`）；时间窗统一 `[past=384, future=128]`；5-fold 分层划分与 dataloader 统一接口见 `.trellis/spec/research/data-pipeline.md`，可视化统一 visualizer 见 `.trellis/spec/research/visualization.md`。

2026-10-13 消融结论（fold0 单 seed，Run 010）：**「d_model=16 跨通道瓶颈是 GC 偏弱主因」不成立**——d_model 16→62（参数 ×8.4 单变量）后 GC 一致性反降（r 0.282→0.231）、生成 GC 幅值更低（真实 40%→25%）；Q1 改善边际（MAE 0.704，未超 GPR 基线 0.687）；FC 一致性 0.976、空间 R̂~R 0.969 略优。GC 偏弱指向损失/训练目标（无跨通道时序监督）或数据，为下一消融方向。

2026-10-13 高频问题三连裁决（Run 012–016）：生成侧 >40 Hz 功率偏高**主因 = OU 时间核 1/ω² 谱尾**（iso 仅 ~1% 次因）、网络 40–100 Hz 去噪不充分、log 刻度+真实侧数值地板放大。**先验侧三条路线全部证伪**：Run 014 带限核（条件方差塌缩不可用）、Run 015/016 先采样后带限（数值可行但网络把 40–100 Hz 重新注入至 ~6.7%、幅度过冲 +36%、GC 塌缩 0.231→0.022，阴性）。**正式裁决（用户拍板）＝输出侧 40 Hz 低通为 gpfm-mamba 路径产物口径**（evaluation-calibers.md，Run 017 落地）；正式口径下意外修复 GC 幅值校准（40%→92%）。v3 不作 5-fold 候选。

2026-10-16 H3 判定（任务 10-16 归档，用户拍板）：**「跨通道混合可局部移植为 GC 载体」不支持**——v4a rank-16 输入接口（GC r 0.092，五方最低、≈v2 CI 重叠）与 v4b cMLP 顶层读出（d1 0.133 / d8 0.163）两臂均不回升（v1 0.2755），缺口共同指向**损失侧**（无跨通道时序监督）或 v1 优势载体「深度持续混合」（16 维贯穿 trunk 全深，原 C 方案域不追）。v4a 逐通道质量五方最优（MAE 0.69540 / CRPS 0.52392 / 逐通道 r 0.18317 首超 GPR 均值基线），GC 缺口与质量缺口解耦。**5-fold 候选集 = {v1, v4a}**（v2/v4b-d1/d8 落选）。fold0 累计口径：32/4/10 划分、早停 10、恒定 lr 1e-3、gen_lowpass 40 Hz。

2026-10-17 edra-fmt 新范式 fold0 前置验证（任务 10-16-gpfm-mamba-fmt-f0，单 seed，LayerNorm/p=25/无低通/**genGC vs real GC 阻塞**口径）：v1-fmt/v4a-fmt 均过指标1（FC 0.970/0.981）与指标3（1–30 Hz |Δlog10| 0.317/0.301）；**v4a-fmt 质量全维度最优**（MAE 0.70565 / CRPS 0.53912 / 逐通道 r 0.12335 超 GPR-r 0.116），候选排序与旧范式一致（判读权重向 v4a 侧——v1 侧跨范式叠加纪律变量）；无低通下生成侧 >40 Hz 功率占比 ≈ 真实 3.0×（≈95% 功率在带内，大体自行待在带内）；v4a-fmt R1 触发定性为口径效应（登记三值）。跨范式 = 多变量设计（fs/预处理/48例/p/低通/norm/纪律），仅方向性判读。5-fold（何范式/口径）待用户拍板。

2026-10-17 adaLN(t) 调制归一化消融（任务 10-17-gpfm-mamba-adaln-f0，v4a-fmt 基座 fold0 单 seed，A 案 scale/shift + fork_rng 恒等初始化）：**中性结果**——与 v4afmt-main-01 全维度持平（MAE 0.70571 vs 0.70565、FC 0.9797 vs 0.98147、指标3 0.30605 vs 0.30134 均 ✓，Δ≤0.005 单 seed 噪声框内），无维度方向性改善；过散/高频尾微加重（+44.5%→+47.0%、gen>40Hz 4.92%→5.19%）。两 Run 同 seed 初始权重逐位相同（fork_rng 契约，epoch-0 step-1 损失逐位相等 1.4787474870681763），分叉恰为处理效应——**该规模（140k）与纪律下，速度网络未从 norm 仿射 t 条件化获得超出输入侧 temb 注入的可用信息**。候选集 {v1, v4a} 排序不变；RNG 中性构建惯例已入 reproducibility.md。

2026-10-17 Study II 流程首验证（任务 10-17-perturb-ec-f0，v4a-fmt fold0 冻结 θ\*·前置验证级偏离经用户签认，10 例 CRN 批采样 27.7 min）：**流程可跑通且确定性契约全兑现**（r=0 同批逐位恒等、跨 Run 逐位复现、100% 单调剂量-响应）；四要素首版落地（增益算子/CRN-RMS 度量/τ 双口径 v2——绝对口径 q95≈1.82σ 全边 100% 删失系材质性标尺错配，操作性改相对自响应 τ̃=0.25 删失 3.7%/路径集交集 3201 边；r\* 近二元如实登记）；**AC6 如实判定：该 θ\* 上未获结构收敛初步证据**（slope×GC ρ̄=−0.007 p=0.581、slope×gPDC ρ̄=+0.022 Holm p=0.112；ceiling 0.96、实际可检 ρ̄≈0.028 排除 power 不足）——缺口四重证据定位 **θ\* 跨通道传导弥散无特异性**（空间衰减平 ρ=−0.05、D̃ 近 r 无关 ~0.78、三 readout ρ 同值、自/跨比仅 1.27×），为 Study I 迭代（损失侧跨通道监督）提供直接判据；流程/契约与实测基线沉淀 spec/research/perturb-ec-pipeline.md。

## 待用户拍板

- ~~扰动算子 P / 度量 D / 阈值 τ / 搜索策略~~ → **已首版定义**（2026-10-17 用户拍板 A 案增益算子 + τ̃=0.25 相对自响应；契约见 spec/research/perturb-ec-pipeline.md，变更须走口径变更纪律）
- r\* → 连接强度的最终映射（框架第 14 步，现仅候选 readout 三件套）
- Study II 缺口应对路径：换跨通道特异性更强的代理脑（Study I 损失侧迭代）后复跑流程 / B 案算子等口径变更 / 其他
- 5-fold 主实验（范式/口径）与 Study II 复跑的先后关系
- （沿既有）指标2 视觉人工判；GC 阻塞解除时点

## 关键文档地图

- `docs/research/research-direction.md` — 研究框架权威文档（主题/两任务/方法框架/两子研究/术语表/核心逻辑）
- `.trellis/spec/research/` — 科研工作流规范十份（实验日志、路径隔离、可复现性、产物管理、评估口径、实验模板、代理脑合格性检验、数据管道、可视化、**Study II 扰动流程**）
- `.trellis/spec/research/surrogate-brain-qualification.md` — Study I 代理脑合格性检验规范（任务路由到 `study1` 时经 1.0b/1.3 调用；固定 θ\* 前须完成三项检验）
- `.trellis/spec/research/perturb-ec-pipeline.md` — Study II 扰动→EC 推断流程规范首版（任务路由到 `study2` 时必读；CRN 契约/τ 双口径/隔离矩阵/实测基线）
- `.trellis/research/environment.md` — 环境契约（写/跑代码前必读）
- **实验笔记目录与命名惯例**：随 Trellis 任务走 —— 每任务 `{TASK_DIR}/experiment.md`（创建任务时自动播种，格式见 spec/research/experiment-template.md）；跨任务独立实验记录目录未设立，如需要由用户指定

## AI 不得擅自触碰（须先问用户）

- 参考文献口径边界：不得把「临界扰动强度推断」表述为参考文献已给结论（框架文档明文区分）
- Stage 顺序：代理脑训练完成并固定参数（θ=θ*）之后才可做扰动实验；不得在未固定模型上做连接推断
- 数据集与预处理口径未拍板前，不得自行下载数据；项目数据一律存放于 `/root/autodl-tmp/`（用户 2026-10-10 指定），不得散落到工作区
- 以上前两条为框架文档自带的守护项，其余守护项（定版口径、不可覆盖产物目录等）逐项确认与增删以用户意见为准
