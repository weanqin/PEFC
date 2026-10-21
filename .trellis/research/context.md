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
| Study I：EEG Conditional Future Generation | 训练代理脑 X_past→X_future（时序预测/条件生成/时间修补三视角），含 EEG 表征、未来生成、代理脑有效性三层次 | `models/gpfm-mamba-v1/`（+v2/v4b/v4a/v1-fmt/v4a-fmt/v4a-fmt-adaln/v5 变体目录） | gpfm-mamba v1/v2/v4 已实现，fold0 单 seed 前置验证（任务 10-12）与 H3 消融（任务 10-16）完成；**5-fold 候选集已拍板 {v1, v4a}**；新范式 edra-fmt fold0 前置验证完成（任务 10-16-gpfm-mamba-fmt-f0，2026-10-17，LayerNorm/p=25/无低通/GC 阻塞口径）；adaLN(t) 消融中性结果（任务 10-17-gpfm-mamba-adaln-f0，2026-10-17，排序不变）；**v5 变体结构落地**（任务 10-19-gpfm-mamba-v4b-gnn-f0，2026-10-19 用户拍板二次转向：v2 基线 + 轴分解交替 4×[Mamba→GNN 空间块(A_eff=γ_m·M(d;σ)+γ·W 跨块共享)→LayerNorm]、d_model=C=62、152,647 参数、单测 10/10 + A_eff 与 latent-flow 源位级等价；主 Run `v5-main-01` 已完成 2026-10-20——用户直训指令替代 R9 门，100 epoch 跑满（早停禁用/lr cosine 系 v2 副本口径勘误随档）、best val 0.73155@ep64）；**v5-fmt 变体完成**（任务 10-20-gpfm-mamba-v5-fmt-f0，2026-10-20 归档：v5 结构原样 × fmt 口径（layernorm/p=25/无低通/fs250）+ **用户拍板 cosine+早停 10**，152,957 参数、单测 11/11、主 Run `v5-fmt-01` **ep39 早停触发 best val 0.70242@ep29**（pin `a64c0d416756`），A_eff/W 快读出与 v5-main-01 对照随档——**θ\* 候选就绪，下游 = study2 NI-MSE-50-q0.90（ni-mse-design.md v1.5 主实例化）**）；5-fold 主实验待启动 |
| Study II：Critical Perturbation Connectivity Inference | 固定 θ* 后扰动源节点 A→测目标节点 B 响应→搜临界强度 r*_A→B=inf{r:Δ(r)≥τ}→临界扰动矩阵→连接强度 | `models/perturb-ec-v1/`（θ\* 只读引用 Study I 产物）；`models/bivar-gc-ec-v1/`（双变量 GC 观测参考）；`models/bivar-gc-ec-v2/`（同路径 250 Hz 原始全长档）；`models/ni-mse-v1/`（NI-MSE-50-q0.90 EC_pert 路径） | **四要素首版已定义 + fold0 流程验证完成**（任务 10-17-perturb-ec-f0，2026-10-17）：CRN 确定性契约 + 增益算子 + τ 双口径 v2（操作性 τ̃=0.25 相对自响应）+ readout 三件套；v4a-fmt θ\* 上**未获结构收敛初步证据**（缺口=跨通道传导特异性，非流程问题）；流程规范首版见 spec/research/perturb-ec-pipeline.md；**双变量 GC 观测参考已落地**（任务 10-17-bivar-gc-ec-fmt，2026-10-17，125 Hz 档）；**250 Hz 原始全长档参照已落地**（任务 10-18-bivar-gc-fmt-250，2026-10-18：`outputs/bivar-gc-ec-fmt-250/refs-01/` 48 例 + 天花板 0.8981±0.1414 + rev-01 时间反转指纹，两档口径隔离不做对比交付）；**原库 GC × edra_v4 重采样参考已归档**（任务 10-21-oldlib-gc-edrav4，2026-10-21：`refs/granger_bivariate_spectral/edrav4-46-.../` 46 例，与 edrafmt48 档重叠 45/46，口径隔离不做对比交付）；**NI-MSE-50-q0.90 路径首执行完成（阴性）**（任务 10-20-ni-mse-v5fmt-f0，2026-10-20：θ\*=v5-fmt-01 主实例化，G 门控 10 例全 0/3782、S offdiag 中位 ~0.45、剂量曲线全平、refs 相关 r≈0.05–0.10 flip≈direct ⇒ 该配置下 EC_pert 不可用；流程契约全兑现，实测基线与可交换 null 灵敏度教训沉淀 spec §7） |
| Study I 候选：latent-flow（OC-LatentFlow，先前原型移植） | 两阶段（stage1 VAE → stage2 潜空间流匹配）obs-conditioned 代理生成 + bisection σ 扰动归因→有向 EC（同一思路 2026-06~08 先前实现，移植自 latent_flow.7z） | `models/latent-flow-v1/`（嵌套包 + run.py 包装 + train_edra_fmt.py 训练入口；源锚点 `9efe1e3`；判决文档全保留，de/dysib/dcgnn/fm_gate 终止线已裁） | **edra_fmt 首个正式训练 run 完成**（任务 10-19-latent-flow-edra-fmt-train，2026-10-19，用户拍板 D0 旧库 CFM 口径/D1 mexhat/D2 geometry 数据根豁免/D4 npy 基座）：npz shim 落地（descriptor+reader+384/128/64 窗+训练折归一化，manifest 只读复用），fold0 两阶段各 50 epoch 稳定收敛（stage1 best 1.5526 / stage2 best 0.5977），θ 候选 ckpt 落 `outputs/10-19-latent-flow-edra-fmt-train/run_01/`；**未做代理脑合格性判定**（三项检验属后续任务）；MODMA 原生线可跑未跑；旧 θ\*（558 MB，autodl-tmp 原位）不继承代理脑资格 |

Study I 是 Study II 的前置（先建立可信代理脑，再做虚拟扰动）。注意：**预测质量 ≠ Surrogate Brain Validity**，须作为两个独立问题评价。

## 既有结论（已验证）

尚无实验结论。已定案的是框架层事实（见框架文档）：两阶段流程（Stage I 训练并固定 → Stage II 扰动推断）、14 步完整流程、核心四问 Q1–Q4、术语表。

2026-10-12 定案：数据集为 **edra**（`/root/autodl-tmp/edra_process_v4/`，HC 24 / MDD 22 受试者级 npz，62 通道、500 Hz，契约 `.trellis/spec/data/edra.md`）；时间窗统一 `[past=384, future=128]`；5-fold 分层划分与 dataloader 统一接口见 `.trellis/spec/research/data-pipeline.md`，可视化统一 visualizer 见 `.trellis/spec/research/visualization.md`。

2026-10-13 消融结论（fold0 单 seed，Run 010）：**「d_model=16 跨通道瓶颈是 GC 偏弱主因」不成立**——d_model 16→62（参数 ×8.4 单变量）后 GC 一致性反降（r 0.282→0.231）、生成 GC 幅值更低（真实 40%→25%）；Q1 改善边际（MAE 0.704，未超 GPR 基线 0.687）；FC 一致性 0.976、空间 R̂~R 0.969 略优。GC 偏弱指向损失/训练目标（无跨通道时序监督）或数据，为下一消融方向。

2026-10-13 高频问题三连裁决（Run 012–016）：生成侧 >40 Hz 功率偏高**主因 = OU 时间核 1/ω² 谱尾**（iso 仅 ~1% 次因）、网络 40–100 Hz 去噪不充分、log 刻度+真实侧数值地板放大。**先验侧三条路线全部证伪**：Run 014 带限核（条件方差塌缩不可用）、Run 015/016 先采样后带限（数值可行但网络把 40–100 Hz 重新注入至 ~6.7%、幅度过冲 +36%、GC 塌缩 0.231→0.022，阴性）。**正式裁决（用户拍板）＝输出侧 40 Hz 低通为 gpfm-mamba 路径产物口径**（evaluation-calibers.md，Run 017 落地）；正式口径下意外修复 GC 幅值校准（40%→92%）。v3 不作 5-fold 候选。

2026-10-16 H3 判定（任务 10-16 归档，用户拍板）：**「跨通道混合可局部移植为 GC 载体」不支持**——v4a rank-16 输入接口（GC r 0.092，五方最低、≈v2 CI 重叠）与 v4b cMLP 顶层读出（d1 0.133 / d8 0.163）两臂均不回升（v1 0.2755），缺口共同指向**损失侧**（无跨通道时序监督）或 v1 优势载体「深度持续混合」（16 维贯穿 trunk 全深，原 C 方案域不追）。v4a 逐通道质量五方最优（MAE 0.69540 / CRPS 0.52392 / 逐通道 r 0.18317 首超 GPR 均值基线），GC 缺口与质量缺口解耦。**5-fold 候选集 = {v1, v4a}**（v2/v4b-d1/d8 落选）。fold0 累计口径：32/4/10 划分、早停 10、恒定 lr 1e-3、gen_lowpass 40 Hz。

2026-10-17 edra-fmt 新范式 fold0 前置验证（任务 10-16-gpfm-mamba-fmt-f0，单 seed，LayerNorm/p=25/无低通/**genGC vs real GC 阻塞**口径）：v1-fmt/v4a-fmt 均过指标1（FC 0.970/0.981）与指标3（1–30 Hz |Δlog10| 0.317/0.301）；**v4a-fmt 质量全维度最优**（MAE 0.70565 / CRPS 0.53912 / 逐通道 r 0.12335 超 GPR-r 0.116），候选排序与旧范式一致（判读权重向 v4a 侧——v1 侧跨范式叠加纪律变量）；无低通下生成侧 >40 Hz 功率占比 ≈ 真实 3.0×（≈95% 功率在带内，大体自行待在带内）；v4a-fmt R1 触发定性为口径效应（登记三值）。跨范式 = 多变量设计（fs/预处理/48例/p/低通/norm/纪律），仅方向性判读。5-fold（何范式/口径）待用户拍板。

2026-10-17 adaLN(t) 调制归一化消融（任务 10-17-gpfm-mamba-adaln-f0，v4a-fmt 基座 fold0 单 seed，A 案 scale/shift + fork_rng 恒等初始化）：**中性结果**——与 v4afmt-main-01 全维度持平（MAE 0.70571 vs 0.70565、FC 0.9797 vs 0.98147、指标3 0.30605 vs 0.30134 均 ✓，Δ≤0.005 单 seed 噪声框内），无维度方向性改善；过散/高频尾微加重（+44.5%→+47.0%、gen>40Hz 4.92%→5.19%）。两 Run 同 seed 初始权重逐位相同（fork_rng 契约，epoch-0 step-1 损失逐位相等 1.4787474870681763），分叉恰为处理效应——**该规模（140k）与纪律下，速度网络未从 norm 仿射 t 条件化获得超出输入侧 temb 注入的可用信息**。候选集 {v1, v4a} 排序不变；RNG 中性构建惯例已入 reproducibility.md。

2026-10-17 Study II 流程首验证（任务 10-17-perturb-ec-f0，v4a-fmt fold0 冻结 θ\*·前置验证级偏离经用户签认，10 例 CRN 批采样 27.7 min）：**流程可跑通且确定性契约全兑现**（r=0 同批逐位恒等、跨 Run 逐位复现、100% 单调剂量-响应）；四要素首版落地（增益算子/CRN-RMS 度量/τ 双口径 v2——绝对口径 q95≈1.82σ 全边 100% 删失系材质性标尺错配，操作性改相对自响应 τ̃=0.25 删失 3.7%/路径集交集 3201 边；r\* 近二元如实登记）；**AC6 如实判定：该 θ\* 上未获结构收敛初步证据**（slope×GC ρ̄=−0.007 p=0.581、slope×gPDC ρ̄=+0.022 Holm p=0.112；ceiling 0.96、实际可检 ρ̄≈0.028 排除 power 不足）——缺口四重证据定位 **θ\* 跨通道传导弥散无特异性**（空间衰减平 ρ=−0.05、D̃ 近 r 无关 ~0.78、三 readout ρ 同值、自/跨比仅 1.27×），为 Study I 迭代（损失侧跨通道监督）提供直接判据；流程/契约与实测基线沉淀 spec/research/perturb-ec-pipeline.md。**用户定论（2026-10-17）**：当前方法推断出的 EC **完全不可用**，不在此 θ\* 上继续；优化留待后续任务（流程与四要素契约可复用，规范已沉淀）。

2026-10-17 双变量 GC 观测参考落地（任务 10-17-bivar-gc-ec-fmt，用户四轮拍板：48 例全量 / 125 Hz / 逐对 BIC / ln(SSR) / **不做 EC_pert 比较**）：edra-fmt 48 例成对双变量 GC EC 矩阵首版落盘（`models/bivar-gc-ec-v1/`、`outputs/bivar-gc-ec-fmt/refs-01/`，offdiag min/median/max=0.0007/0.0153/0.4356 ln 单位；split-half 天花板 0.9243±0.0447；OLS 无收缩非负断言 + Gram-slice↔naive 等价单测 rtol=1e-10、20 项全绿）；**BIC 100% 顶格 p\*=24**（n≈3 万惩罚项极小）为实测事实如实登记，网格扩展未决；与全模型条件 GC（baseline-ec-v1 主口径）平行**不混用**；fmt 条件 GC 与比较器接入均滞后决策；口径条目入 evaluation-calibers.md「双变量 GC 观测参考口径」。

2026-10-18 latent-flow 先前原型移植完成（任务 10-18-latent-flow-port，用户拍板：裁剪 A+bundle 备份/study1 路由/checkpoints 原位/AC3 冒烟改道）：OC-LatentFlow 两阶段潜空间流代理 + bisection σ 归因代码面落入 `models/latent-flow-v1/`（裁剪后 74 py/17,232 行 + 20 份判决笔记），当前环境（Python 3.12/torch 2.8）零源码改动可跑——G1 三判据（pytest 67P/0F/17S 对齐冻结 per-file 口径、31 脚本 import 冒烟零失败、CRLF=0）+ G2（shape_check GPU 前向 + BLAS 守卫）全过，trellis-check 通过（AC1–AC5）。边界：EDRA npz shim 与 MODMA 线补跑属后续任务；σ 二分与 perturb-ec-v1 四要素的算子/度量/τ/确定性机制差异表见任务 design §5；train --smoke 本机数据阻塞待 shim/数据后补。**MODMA 勘误**：真实文件名 `modma_resting_process.zip`（先前 ENOENT 系误拼），已解压 `/root/autodl-tmp/modma/{HC 29,MDD 24}`；源库 bundle 备份 `/root/autodl-tmp/latent_flow.bundle`。

2026-10-19 latent-flow × edra_fmt 首个正式训练 run 完成（任务 10-19-latent-flow-edra-fmt-train，用户四项拍板：D0 stage2=旧库 CFM 口径（线性插值路径+速度场+Euler，无显式速度 MSE，经冻结解码器信号损失隐式优化）/D1 mexhat-graph 流/D2 geometry 落数据根豁免单文件/D4 geometry 用 npy 基座——S2.2 判定门如实触发（MNE 模板 vs npy 16/62 通道 cos<0.99，功能等价邻接相关 0.9967）后主会话真实拍板）：EDRA npz shim 落地（`edra_fmt` descriptor + npz reader + 窗 512/64/384 每例 930 断言 + 训练折 34 例统计归一化，fold0 划分只读复用 gpfm-mamba-v1-fmt manifest 34/4/10 零重新划分）；两阶段各跑满 50 epoch 稳定收敛（stage1 VAE best val 1.5526/66min；stage2 mexhat CFM 流 best val 0.5977/60min，trainable=flow only 476,357，test smoke MAE 0.8708 优于 stage1 proj 路径 1.0435）；产物隔离 `outputs/10-19-latent-flow-edra-fmt-train/run_01/`（双 ckpt+双 config+日志+history）；pytest 冻结口径 67P/0F/17S 不变；trellis-check AC1–AC7 全过。**边界：θ 候选 ≠ 代理脑资格**（三项检验另立任务）；KL 原始值 ~93 与旧库 fold1-4 同型（beta 1e-4 贡献 ~0.009，旧库 08-01 曾以 1e-3 消融）。

2026-10-20 GC 计算主路径拍板 + baseline refs 归档（任务 10-19-gpfm-mamba-v4b-gnn-f0 会话，用户三项拍板）：①**GC 计算主路径 = 原库 GC**（`models/latent-flow-v1/latent_flow/baseline/granger.py`，成对双变量谱 GC，Geweke 1982/Dhamala 2008）——后续 |A_eff|/EC 类 × GC 对照一律用此实现（背景：npi 时域条件 GC 与原库谱 GC 此前实测近乎不相关 +0.036，实现选择即口径）；②GC 参考数据 = fmt；③**baseline 常用参考归档启用 `npi/refs/`**（组织规则 `refs/<方法名>/<参数配置>/` 独立子目录、保留原文件名、meta.json 记来源+口径+SHA256SUMS、不覆盖），首批归档原库 GC 在 edra_fmt 48 例全长上的 EC 矩阵（源 `outputs/10-19-latent-flow-edra-fmt-train/run_01/gc_oldlib_refs` → `refs/granger_bivariate_spectral/edrafmt48-cont-zscore-fs250-fixed8-f1-125-nf65/`，固定阶 8/1–125 Hz/fs 250/逐通道 z-score，ec[a,b]=a→b 行=源；对齐口径 |A_eff| vs ec.T）。同日续：study2 EC_pert 路径 **NI-MSE-50-q0.90** 设计定稿（v1.5，`任务目录/ni-mse-design.md`：信号空间无偏噪声 × MSE 响应 × 50 窗 × Q90 门控；完整路径 = study1-gpfm-mamba-v5 × study2 NI-MSE-50-q0.90；实测勘误 fold0 test 10 例与 refs 重叠 10/10 ⇒ 逐被试配对主口径）；**计划新增 v5-fmt-01 训练任务**（用户 2026-10-20：gpfm-mamba-v5 × edra_fmt/250 Hz，与 v5-main-01 并列，训后作 θ\* 接入 NI-MSE 路径——与 refs 同管线消除跨范式 caveat；**已立项 `10-20-gpfm-mamba-v5-fmt-f0` 执行，10-19 任务暂停待恢复**——未竟 = S4'-S6' 收尾 + Run 017 同口径评估）。

2026-10-20 NI-MSE-50-q0.90 首执行（任务 10-20-ni-mse-v5fmt-f0，θ\* = v5-fmt-01 主实例化即 fmt 管线 caveat 消除；null 64×200/响应 32×50⊂null/62 源/c 臂 {0.05,0.1,0.3,0.5}/n₀=25/种子树 2026 四支含 η；全程 2.04 h）：**流程效度全兑现**（R2 同批恒等 ×10 例×4 臂、split-half 0.914–2.000%≪10%、R1/R3/R6 干净、per-例断点续跑契约工作）；**结构结论阴性（主读出）**——G 门控全被试 0/3,782（Q95 带亦 0%）、S offdiag 中位 0.450±0.015（跨通道响应仅噪声地板 Q90 的 ~45%）、offdiag 剂量曲线 c∈[0.05,0.5] 全平（MSE 贡献 <1%）、自响应 S 对角亦仅 ~0.45→0.48（连自通道扰动都低于 fresh-ε 全后验变异性地板）；refs 相关微弱（逐被试 Pearson direct 0.059±0.043 / 组级 0.097；flip≈direct 方向信息≈0；距离偏后不降）⇒ **该 θ\* × 该路径 × 钉定参数下 EC_pert 不可用于连接推断**。方法层沉淀：①可交换 null（fresh-ε）以灵敏度换 Q90 概率语义——中位门需 Δμ²>null Q90−Q50≈0.59 MSE（Δμ≳0.77 z）；②ni-mse-design.md §4.2「阴性对照通过率→~10%」与 §1.5-F 中位聚合门数学上不相容（勘误随档，实测 0% 通过率恰为机器正确的表现）；③四元组 RNG 可交换性契约成本 ≈1.36× 预算（每例 ~29.8 万 RNG 流 + 16.1 亿高斯 CPU 物化；优化方案 A–D 随档：D 预取重叠零数值变化可直接用，A 向量化/B GPU 生成/C (a,c) 并批数值口径变化须变更登记）。

2026-10-21 原库 GC × edra_v4 重采样参考归档（任务 10-21-oldlib-gc-edrav4，用户两项拍板：立任务 / fs 口径=重采样 250 Hz）：旧库谱 GC（GC 计算主路径实现）× `edra_process_v4` 46 例全量（HC 24 + MDD 22，不做 qc_pass 筛选——qc_pass=True 仅 13/46 为登记事实）逐被试 EC 矩阵落盘归档（`refs/granger_bivariate_spectral/edrav4-46-resample250-cont-zscore-fs250-fixed8-f1-125-nf65/`，源 `outputs/10-21-oldlib-gc-edrav4/refs-01/`，15.4 min、同例重跑 SHA256 逐位复现）；唯一新增预处理 = `resample_poly(1,2)` 500→250 Hz（确定性），其余逐项同 edrafmt48 档口径（fixed8/1–125 Hz/nf65/全长连续/逐通道 z-score，ec[a,b]=a→b 行=源）；**与 edrafmt48 档重叠 45/46**（v4 MDD s56 为 fmt 所无，fmt 独有 MDD {11,21,48} 为 v4 弃用例）——下游两档对照一律取 45 例交集；实测对照（描述性登记不做对比交付）：offdiag Spearman ρ=0.2311±0.1026、Pearson r=0.4268±0.1266、median 比值 v4/fmt=1.651±0.664；口径条目入 evaluation-calibers.md「原库谱 GC 观测参考口径 · edra_v4 重采样 250 Hz」（该文件此前无 GC 主路径条目，本条目含主路径登记）。

2026-10-21 v5-main 双方法 EC 探索（任务 10-21-v5main-ec-explore，用户直给规格）：①方法1（源通道噪声 σ∈[0.05,0.48] log×24 → 靶通道高斯拟合微分熵涨幅>20% 判失稳 σ*，EC1=-(ln σ*+ln 2πe)+1）**失败（用户宣告取消）**——4 例全删失：σ=0.48 时靶通道 ln-方差膨胀中位 0.014≪ln(1.44)=0.365，条件流+GPR 先验+40 Hz 产物口径下靶预测方差对单通道源噪声结构性不敏感；②方法2（源 +0.5std 常量 → 靶预测均值响应带符号）与 v4 GC refs **10/10 例一致弱负相关**（ρ −0.126±0.019、组均值 −0.188/r −0.317；flipped≈aligned 方向信息≈零）——强 GC 边上源偏移压低靶预测均值（抑制型模式）。与 perturb-ec-v1/NI-MSE 两条阴性路径同向汇聚：v5-main θ* 跨通道响应未达观测 GC 结构，迭代重点 = Study I 损失侧跨通道监督。产物 `outputs/10-21-v5main-ec-explore/`。

## 待用户拍板

- ~~扰动算子 P / 度量 D / 阈值 τ / 搜索策略~~ → **已首版定义**（2026-10-17 用户拍板 A 案增益算子 + τ̃=0.25 相对自响应；契约见 spec/research/perturb-ec-pipeline.md，变更须走口径变更纪律）
- r\* → 连接强度的最终映射（框架第 14 步，现仅候选 readout 三件套）
- ~~Study II 缺口应对路径~~ → **已定方向（2026-10-17 用户）**：当前方法 EC 完全不可用，不在此 θ\* 上继续，优化留待后续任务（具体路径——代理脑迭代/算子变更/其他——届时再定）
- 5-fold 主实验（范式/口径）与 Study II 复跑的先后关系
- fmt 条件 GC（10 例复用 / 48 例重算 / 时点）与比较器参考列接入时点；双变量 BIC 网格是否扩展（{1..24} 顶格实测事实下）——均滞后决策（2026-10-17 登记）
- （沿既有）指标2 视觉人工判；GC 阻塞解除时点

## 关键文档地图

- `docs/research/research-direction.md` — 研究框架权威文档（主题/两任务/方法框架/两子研究/术语表/核心逻辑）
- `.trellis/spec/research/` — 科研工作流规范十份（实验日志、路径隔离、可复现性、产物管理、评估口径、实验模板、代理脑合格性检验、数据管道、可视化、**Study II 扰动流程**）
- `.trellis/spec/research/surrogate-brain-qualification.md` — Study I 代理脑合格性检验规范（任务路由到 `study1` 时经 1.0b/1.3 调用；固定 θ\* 前须完成三项检验）
- `.trellis/spec/research/perturb-ec-pipeline.md` — Study II 扰动→EC 推断流程规范首版（任务路由到 `study2` 时必读；CRN 契约/τ 双口径/隔离矩阵/实测基线）
- `.trellis/research/environment.md` — 环境契约（写/跑代码前必读）
- `refs/` — baseline 常用参考结果归档（2026-10-20 启用；`refs/<方法名>/<参数配置>/` 独立子目录 + MANIFEST.md；首批 = 原库 GC edra_fmt 48 例 EC 矩阵；2026-10-21 增 = 原库 GC edra_v4 重采样 46 例档）
- **实验笔记目录与命名惯例**：随 Trellis 任务走 —— 每任务 `{TASK_DIR}/experiment.md`（创建任务时自动播种，格式见 spec/research/experiment-template.md）；跨任务独立实验记录目录未设立，如需要由用户指定

## AI 不得擅自触碰（须先问用户）

- 参考文献口径边界：不得把「临界扰动强度推断」表述为参考文献已给结论（框架文档明文区分）
- Stage 顺序：代理脑训练完成并固定参数（θ=θ*）之后才可做扰动实验；不得在未固定模型上做连接推断
- 数据集与预处理口径未拍板前，不得自行下载数据；项目数据一律存放于 `/root/autodl-tmp/`（用户 2026-10-10 指定），不得散落到工作区
- 以上前两条为框架文档自带的守护项，其余守护项（定版口径、不可覆盖产物目录等）逐项确认与增删以用户意见为准
