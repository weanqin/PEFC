# sym-W 变体代理脑无偏扰动归因 EC（global-τ 两阶段，q=0.5）+ 双 GC 逐被试相关对照，conn v4 公式交付

## 子研究路由（1.0b / 1.0c）

<!-- trellis-research-routing -->

- **分类结果**：`study2`（EC 推断算法，框架 Study II——固定 θ 候选代理脑做虚拟扰动推断连接）
- **依据**：任务主体是扰动归因 EC 计算与 GC 逐被试相关对照（推断算法侧）；代理脑（Study I 产物 sym-W 变体）仅作固定输入使用，本任务不训练、不做资格判定。
- **新路径/方法检测（1.0c）**：`已有路径：global-tau 无偏 EC 口径（10-24-mild-w500-ec-unbiased、10-24-exitgnn-ec-unbiased 两先例已归档；本任务 = 同口径第三跑，仅换代理脑 ckpt）`——无新方法，research adaptation 不触发。
- **research 产物**：`无`
- **路由备注**：EC 结论继承 symw 代理脑 provisional 状态 + **A_eff 无向 ⇒ EC 方向性按构造受限**（研究记忆待拍板⑩：用户已预告进 EC 线须按无向口径重设预期）。

## Goal

以 10-25-latent-flow-exit-gnn-symw stage2 best ckpt（`outputs/10-25-latent-flow-exit-gnn-symw/
run_symw_fold0/checkpoints/fold0_mild_flow_best_graph_mexhat.pt`，best val 0.8054@E144 早停
收敛，W torch.equal 级对称）为代理脑，**全默认口径**复用 `models/latent-flow-v1/ec_mild_unbiased.py`
无偏 μ=0 扰动归因（q=0.5、σ_ref=0.3、σ 网格 log[0.01,0.48]×24、fold0 test 8 例 × 30 trial
全量）主跑 EC_pert；交付公式 **v4：conn = −(ln σ* + 0.5·ln 2πe) + 5/7**；离线组级 summary；
EC×GC 描述性对照**双参考**（v1 mild40 1–45 Hz 固定阶 8 + v2 mild40 1–125 Hz AIC 4–25）逐被试
相关；**新增对称性诊断**（EC vs EC.T）量化无向性对方向判读的实际限制。

## 背景与动机

- 用户指令（2026-10-25，三项）：
  1. **代理脑**：本次训练流程最新产出 `model-md-gnn-exit-anti-uniform-0.2-对称变体`
     （= 10-25-latent-flow-exit-gnn-symw sym-W 变体），不用历史版本。
  2. **计算方法**：沿用此前任务的扰动归因流程与参数设置，不改归因算法逻辑。
  3. **输出**：EC_pert 结果 + 注明模型版本与计算配置（可与先前结果对比）+ 逐被试与
     GC-EC 的相关性。
- 口径事实：q=0.5 / σ_ref=0.3 / log[0.01,0.48]×24 / 无偏 μ=0 = `ec_mild_unbiased.py` 现行
  默认值，与 w500、exit-gnn 两先例**完全同参**——唯一实质差异 = 代理脑 ckpt。
- 适配事实（代码侧仅两处障碍，算法路径零触碰）：
  - symw 训练脚本直接 import `FlowVelocityNetExitGNN`（`flow_exit_gnn_symw.py:58`），
    SymWAdam 仅为 optimizer 尾部投影、零架构差异 ⇒ `--flow-exit-gnn` 的模型构建 +
    strict load 对 symw ckpt 直接可用；
  - 障碍①：ckpt 路径守卫只认 `10-24-latent-flow-exit-gnn` run_dir（`ec_mild_unbiased.py:142`）；
  - 障碍②：payload 元数据硬编码 exit-gnn 任务描述/caveat（`ec_mild_unbiased.py:272-295`）。
- v4 公式数学事实（两先例在档）：v4 = 房内原式 v1 严格常数平移 ⇒ **raw 口径
  Spearman/Pearson 逐位不变**；零点 σ*=0.4943 落在网格上界 0.48 之外 ⇒ 全正 ⇒ abs≡raw。
- **科学 caveat（本任务特有）**：SymWAdam 每步 W←(W+Wᵀ)/2 ⇒ flow 通道耦合严格对称 ⇒
  A_eff 无向 ⇒ EC_pert 方向性按构造受限；与**有向** GC 参考的相关上限须按此重设预期
  （对称性诊断 R7 给出实证）；λ0.2 与 sym-W 两因素同变（未消融）。
- 先例结果（并置参照）：w500：v1 块 ρ −0.099 / raw +0.118（基本不相关）；exit-gnn：v1
  +0.149±0.168（微弱正）、v2 −0.010±0.085（基本零）。

## Requirements

- **R1（脚本最小适配，算法零改动）**：`ec_mild_unbiased.py` 新增 `--symw` 旗标（默认
  False，旧调用面不变）：①架构侧等价 `flow_exit_gnn=True`（同 `_model_args`）；②ckpt 守卫
  改查 `10-25-latent-flow-exit-gnn-symw` run_dir（与 `--flow-exit-gnn` 互斥断言）；③payload
  的 method/surrogate_block/surrogate_caveat 走 symw 分支（task = 10-25-latent-flow-exit-gnn-symw
  run_symw_fold0、best 0.8054@E144、variant 描述含 SymWAdam 投影 + anti-uniform λ0.2、
  caveat 含 A_eff 无向方向性受限 + 两因素 + θ 待拍板）；④hparams 新增 `symw` 键。
  **σ*/τ/conn/probe/bisection 计算路径逐行零改动**（diff 审核确认）。
- **R2（等价性对拍，工程干跑存证）**：①旗标关（默认 w500 路径）同参 mini 跑（1 例 ×
  8 源 × 4 网格、seed 42）与本仓在档 `outputs/10-24-mild-w500-ec-unbiased/dryrun_mini/
  subject_0.json` 逐位对拍 **max|Δ|=0**（cudnn.deterministic；非零则登记实测差 + 停查根因，
  不触旧树）；②`--symw` 干跑（同 mini 口径）：strict load 成功 + 守卫/元数据/caveat 字段
  核对（标注非正式口径）。
- **R3（主跑，用户口径钉死）**：GPU 两阶段 global-τ EC：无偏 μ=0 扰动 σ·ξ
  （use_biased_noise=False，断言 μ_bias None）、q=0.5、σ_ref=0.3、log[0.01,0.48]×24、
  8 例（MDD 4：s1,5,7,10 / HC 4：s21,25,27,30）× 30 trial 全量、seed 42、bf16 autocast、
  断点续跑（json 存在即跳过）；输出 `outputs/10-25-symw-ec-unbiased/ec_main/`（逐例 json +
  曲线 npz，原子写）。凭据链完整：ckpt 路径 + config `symw:true` + git head + norm/manifest
  SHA 断言。τ 分位含对角口径照旧（房内先例）。
- **R4（v4 公式交付）**：复用 `recompute_conn_v4.py`（加 `--task-label` 参数，默认值 =
  现常量串，旧调用面不变；REFS 双参考常量不动）：三断言（raw Spearman 逐位不变 / 全正 /
  abs≡raw）8/8 PASS + 逐例 json 重映射档落 `outputs/10-25-symw-ec-unbiased/recompute_v4/`。
- **R5（组级 summary）**：`--mode summary --task-label` 离线（interior% / dyn_ratio / 方差
  分解 / 跨被试稳定性 / 重映射不变性断言），与 w500、exit-gnn 同口径**三方并置**。
- **R6（双 GC 逐被试相关对照 = 用户核心交付）**：EC v4 × {v1 refs, v2 refs}：offdiag
  16256 对、aligned/flipped × Spearman/Pearson，per-subject + group + group_mean 汇总
  （gccorr_dual_summary.json）；与 w500 / exit-gnn 先例并置描述（**跨代理脑描述性并置**
  注记——θ 不同，差异 = 代理脑差异 + 噪声，不作复现性判读）。
- **R7（对称性诊断，纯后处理）**：逐被试 EC v4 矩阵 vs 其转置：offdiag Pearson/Spearman +
  相对不对称度 ‖M−Mᵀ‖_F/‖M‖_F —— 量化 A_eff 无向对 EC 方向性的实际限制，支撑 R6 判读；
  独立小脚本（models/latent-flow-v1/ 新文件），不改归因算法。
- **R8（caveat 三处随档）**：A_eff 无向 ⇒ 方向性按构造受限 + λ/sym-W 两因素同变 + θ 候选
  判断待用户 + EC 结果 provisional 不构成代理脑资格判定——逐例 payload caveat 字段 +
  outputs README + experiment.md 三处登记。
- **R9（路径隔离）**：改动仅 `models/latent-flow-v1/ec_mild_unbiased.py` +
  `recompute_conn_v4.py`（--task-label）+ 对称性诊断新脚本 + 新 outputs 任务树
  `outputs/10-25-symw-ec-unbiased/` + 本任务目录；latent_flow 包体零改动；旧任务输出树 /
  refs 零触碰；**预存在脏路径不擅动**（models/ec-globaltau-v1/ 5 个未跟踪脚本 + 两个
  in_progress 任务目录）。

## Acceptance Criteria

- [ ] R1 适配 diff 审核通过：σ*/τ/conn 计算路径零改动；`--symw` 默认 False 时旧行为
      等价（运行时证据 = R2① 逐位对拍 max|Δ|=0）。
- [ ] R2 两项干跑证据在档：旗标关逐位对拍通过（或登记根因）；`--symw` strict load +
      payload 元数据/守卫核对通过。
- [ ] R3 主跑 8/8 例成功；逐例 json 凭据链完整（ckpt/旗标/SHA/seed/git head）；P(resp>τ)
      =0.500 逐例精确。
- [ ] R4 三断言 8/8 PASS；recompute_v4/ 逐例 json + 汇总落档。
- [ ] R5 summary json 落档 + 三方（w500/exit-gnn/symw）并置判读段。
- [ ] R6 gccorr 双参考 per-subject + group + group_mean 齐（aligned/flipped 照登）+
      先例并置注记。
- [ ] R7 对称性诊断落档（逐被试 EC vs EC.T 相关 + 不对称度）。
- [ ] experiment.md 各 Run 块齐备（目的-配置-结果-结论）；研究记忆回写（progress.md
      Study II 状态）；outputs README（配置/诊断摘要/复现命令/判读要点）。
- [ ] 收尾三件套完成（2.2 check 全项 PASS / 3.3 spec-update / 3.4 commit 仅本任务路径）。

## 代理脑 caveat（登记不阻塞）

- symw Run 000 质量事实：best val 0.8054@E144 三者最优（exit-gnn 0.8241@E116 / w500
  0.8356@E150）；medoid/real std 0.858/0.848（幅度收复，未达 w500 0.957/0.996）。
- **A_eff 无向（W torch.equal 级对称）⇒ EC_pert 方向性按构造受限**；与有向 GC 的相关
  判读须按无向口径重设预期（R7 对称性诊断提供实证）。
- λ0.2 与 sym-W 两因素同变（未消融）；θ 候选判断待用户拍板。
- 用户明确指定该模型为代理脑 ⇒ 照用；EC 产物不构成代理脑资格判定，标 provisional。

## 约束

- **轻量任务声明**：PRD + implement.md；design.md 不适用（100% 复用两先例已评审设计，
  唯一适配点 R1 已钉到行级）。1.4.1 review dispatch 按 1.4 规则可跳过（轻量），1.4.2/1.4.3
  briefing + 明确同意不豁免。
- 环境：npi 本机（RTX 4080 SUPER 32 GB，跑前查空闲）；GPU 主跑预算 ≈50 min（w500 实测
  46.9 min、exit-gnn 同量级参照）；BLAS 线程守卫脚本已内置（numpy 导入前）。
- 数据：Mild Depression（契约 spec/data/mild-depression.md）；w500 口径硬三元组
  500/375/125 断言 + norm SHA 对拍照旧（symw 训练同口径，复用无需改）。
