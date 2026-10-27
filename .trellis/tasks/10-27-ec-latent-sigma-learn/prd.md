# EC 潜空间可学σ扰动（NoiseLayer·SGD·20epoch）首实现

## Goal

潜空间扰动 EC 新路径首实现：stage-2 入口 NoiseLayer（1 靶通道不加噪 + 127 通道逐通道可学 σ_i×iid 高斯）、单步构造（非 Euler10）、loss=λ(1−pearson(fc_pert,fc_origin))−H(σ) 修复型目标（λ 默认 1、pearson scope=target 用户拍板）、SGD×20epoch、σ*→v4 公式 EC；交付脚本+构造检查+GPU 冒烟，全量主跑另立。

## Requirements

### R1 用户直给规格（2026-10-27 设计稿 13 点，逐条为契约）

**模型与架构**
- R1.1 stage-1/2/3 全部参数冻结（requires_grad=False），唯一可训练模块 = NoiseLayer；冻结但前向图保留（梯度穿堂回传到 σ）。
- R1.2 NoiseLayer 插在 stage-2 入口、GNN 之前（mild 线变体图混合在 flow 内部 ⇒ 对 z_obs 条件先加噪即满足）。
- R1.3 stage-3 loss 在模型输出 fc 段上计算（vae→flow→decoder 最终输出；全窗解码器取 [:, obs_len:, :]，fc-only 解码器直出）。
- R1.4 stage-2 用训练/验证口径单步构造（oc_latent_flow.py:1111-1121），不用推理 Euler-10。

**NoiseLayer 行为**
- R1.5 输入 z_obs (B,128,16,64)（stage-2 确定性 mu，oc_latent_flow.py:1097）。
- R1.6 选定 1 靶通道不加噪；其余 127 通道各施加 σ_i × unit_noise。
- R1.7 unit_noise ~ N(0,1) 逐通道逐点独立采样（不共享）。
- R1.8 σ_i 全部初始化 0.01（= σ 盒下界，沿用现管线网格下界惯例）。

**优化目标**
- R1.9 loss = λ·(1 − pearson(fc_pert, fc_origin)) − H(σ)：修复型目标（字面「pearson−H」最大化/最小化均退化为盒角解，论证见 design §2；H 为 127 加噪通道平均微分熵，解析式 ½ln(2πeσ_i²)，∂H/∂σ_i = 1/(127σ_i)，纯解析直回传到叶参数 σ）。
- R1.10 最小化该 loss：保真项推 σ↓ / 容量项推 σ↑，平衡点 σ*_i ≈ 1/(127·λ·|∂P/∂σ_i|)，影响力强 ⇒ σ* 小 ⇒ EC 大（与 v4「σ* 小=连接强」同向）。

**优化流程**
- R1.11 SGD（无 momentum）优化逐通道独立 σ_i（单参数向量 elementwise 更新），epoch 固定 20。

### R2 用户拍板（2026-10-27 本任务立题指令）

- R2.1 立 Trellis 任务落库实现（本任务）。
- R2.2 pearson scope = **target**（只取靶通道 fc 段；逐样本时间维 pearson、batch 平均，仿 _pearson_corr_fc 口径；full 口径实现保留为 CLI 选项但非默认）。
- R2.3 λ 默认 = **1**（起步档，用户拍板）。⚠ research GPU 实测（2026-10-27，research §0-B/E）：|∂P/∂σ_i| 比熵梯度小 5–4 个数量级 ⇒ **λ=1 在任何 lr 下退化为零判别常行**（σ 全体均匀抬升或全钉上界），可行窗实测 λ≈100–300 × lr≈0.5–1.0。λ=1 保留为 CLI 默认（尊重拍板），但**冒烟强制附 λ 梯子 {1,100,300}×lr{5e-3,0.5} 对照表**；λ/lr 终值属主跑前拍板项（本任务内不搜索）。

### R3 口径继承（与现役信号空间 global-τ 管线的一致性）

- R3.1 σ 盒约束 [0.01, 0.48]（= 现管线 log 网格边界）。
- R3.2 σ*→EC 用线默认 v4 公式：−(ln σ*+0.5·ln 2πe)+5/7（10-27-md-latent-flow-config-fix 拍板）。
- R3.3 种子登记与确定性重建契约（种子+配方，四前提=同 seed/同形状/同设备/同 torch 版本）；CRN 语义 = origin/pert 同 t（潜空间单步构造唯一随机源为 t，t 逐样本固定后 fc_origin 可全程缓存）。
- R3.4 interior 率诊断沿用现管线口径（严格位于 (σ_min, σ_max) 的通道占比）。

### R4 交付边界

- R4.1 交付 = 脚本 + CPU 构造检查 + GPU 冒烟（2 例 × 小 epoch），**全量主跑另立任务**（沿用 10-27-ec-histblock-noise → 10-27-ec-histblock-symw-f0 两段式先例）。
- R4.2 产物落 `outputs/10-27-ec-latent-sigma-learn/`（路径隔离）；θ 代理脑默认 run_symw_fold0（「10-25-md-lf-anti0.2-gnnexit」现役，待 research 适配确认 ckpt 接口细节）。
- R4.3 数据 = Mild Depression fold0（w500 口径 375/125 前缀窗、z-score 同训练管线），冒烟 2 例从现管线 8 例惯例中取。

## Acceptance Criteria

- [ ] AC1 脚本 `models/latent-flow-v1/ec_latent_sigma_learn.py` 交付：NoiseLayer（R1.5–R1.8）+ 修复型 loss（R1.9）+ SGD×20epoch 循环（R1.11）+ CLI 参数化（θ ckpt / 靶通道 / λ / lr / epochs / n_mc / batch / seed / 输出目录）+ BLAS 守卫内置；latent_flow/ 包体零改动（若 research 判定必须动包体，改动面须在 design 记录并保持最小）
- [ ] AC2 CPU 构造检查全过（10 项 = 原七项 + research 补强三项）：①σ 盒投影（含边界值）②H 解析梯度 ∂H/∂σ_i=1/(127σ_i) vs autograd 对拍（fp64）③靶通道恒零噪声 + 靶位 σ 无梯度 ④CRN（t 固定）下 fc_origin 逐位复现恒等 ⑤pearson vs 逐样本参考（scipy.stats.pearsonr，fp64）+ 常数序列护栏 ⑥单步构造复刻 vs 手写公式逐位恒等 ⑦穿堂检查（非靶 σ.grad 有限非零、冻结参数 grad=None、fc_origin detach）⑧合成 P(σ) 上 λ(1−P)−H 驻点行为与方向验证（σ*∝1/√λ 斜率对拍）⑨NaN 注入拒跑路径 ⑩decoder 净/扰动 z_obs 靶读出逐位等价
- [ ] AC3 GPU 冒烟 2 例 × 3 epoch × 靶 {0,64} 跑通：σ 发生更新、逐 epoch 诊断（σ 分位/@lo/@hi/interior、|∂P/∂σ| 分位 vs 1/(127σ) 对照行、P/H、|σ−σ_prev|、逐靶 q0=0 掩码）落盘、EC_v4 落盘、无 NaN、显存峰值 <24 GiB、逐例 <5 min；**λ 梯子 {1,100,300}×lr{5e-3,0.5,1.0} interior 对照表落 experiment.md**（λ=1 预期零判别如实登记，为 §10 拍板供证据）
- [ ] AC4 产物与登记：payload json（种子/ε 配方（独立 Generator seed+偏移）/λ/lr/epochs/n_mc/batch/σ 边界/靶集合/scope/θ ckpt SHA/fold/norm SHA/orientation=行源列靶/z-std 诊断/method 串/复现四前提）+ `experiment.md` Run 块（目的/配置/结果/结论四项齐，含 λ 梯子对照表）+ 输出在声明变体目录内（路径隔离）
- [ ] AC5 spec 沉淀：`.trellis/spec/research/perturb-ec-pipeline.md` 增记潜空间可学 σ 变体契约（含修复型 loss 语义、联合边际 vs 逐源隔离 caveat、潜空间 σ 标度注记、EC 值读数边界三条：仅同 run 内序数意义 / 跨 λ 比较=ln λ 平移混淆 / 跨管线数值不可直比）
- [ ] AC6 复现契约：冒烟 Run 同 seed 重跑关键锚点逐位等价（σ 终值/EC_v4）
- [ ] AC7 不做：全量主跑、λ 扫描标定、多靶批打包扩展（后续任务/拍板项）

## Notes

- 复杂任务：`design.md`（技术设计：loss 良构化论证 + 架构接缝 + 并行/效率 + 风险）与 `implement.md`（执行清单）随本 PRD 交付。
- 主跑前须用户拍板项（本任务只暴露不决策）：λ 是否维持 1（视冒烟 interior 率）、lr 终值、n_mc、靶通道集合与多靶扩展。


## 子研究路由（1.0b / 1.0c）

<!-- trellis-research-routing -->

> 任务创建时由 hook 自动播种。规划阶段必须填写以下四项；全部落定前不得进入 1.4 start 门禁（见 `.trellis/workflow.md` Phase 1.0b / 1.0c）。

- **分类结果**：`study2` —— EC 推断算法（框架 Study II：扰动协议、响应度量、阈值、临界强度搜索、连接映射）
- **依据**：任务交付物为新的扰动→临界强度→EC 推断算法实现（潜空间扰动 + 可学 σ 梯度优化 + σ*→v4 映射），代理脑 θ 只读复用不训练，属 Study II 扰动推断方法面。
- **新路径/方法检测（1.0c）**：`新路径：ec-latent-sigma-learn → research 已触发` —— 对照 `.trellis/research/context.md` 技术路线表：Study II 现役条目 = 信号空间扰动（增益算子/无偏噪声）× 网格/二分 σ 搜索 × Euler10 采样；本任务的潜空间扰动 + 单步构造 + 逐通道可学 σ 梯度优化 + 保真-熵目标为 registry 外新方法（扰动算子/响应度量/搜索策略三要素同时更换）。
- **research 产物**：`research/ec-latent-sigma-adaptation.md`（适配结论 / 风险 / 可执行方案；trellis-research 派发于 2026-10-27，落盘后此处即为完成态）
