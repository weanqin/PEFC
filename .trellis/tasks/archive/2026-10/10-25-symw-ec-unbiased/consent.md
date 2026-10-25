# consent — 10-25-symw-ec-unbiased

## 1.4 同意记录（implementation consent gate）

- **时间**：2026-10-25（方案提交同日）
- **用户确认原话（同意链，按时间序）**：
  1. 原始指令（本会话首条消息，明确执行令）：「创建一个trellis任务，要求如下：1. 代理脑
     （Agent Brain）：使用刚刚训练完成的模型 `model-md-gnn-exit-anti-uniform-0.2-对称变体`，
     即本次训练流程中最新的模型产出，不要使用其他历史版本模型。2. 计算方法：沿用之前任务
     中使用的扰动归因（perturbation attribution）计算流程和参数设置，计算 `EC_pert` 指标，
     保持与此前实现一致，不要修改归因算法逻辑。3. 输出要求：输出计算得到的 `EC_pert`
     结果，并注明所使用的模型版本和计算配置，便于与之前的结果进行对比。输出EC_pert对
     逐被试上与GC-EC计算的相关性。」（任务创建同意令）
  2. 1.4.2 briefing（改动点/关键逻辑/潜在风险/路径与环境/验证方案/审查发现处理，含
     「轻量路径跳过 1.4.1 review dispatch」明示）呈现后，AskUserQuestion 问询框未应答；
     用户下一条消息直接回复：**「开始」**——对 briefing 与轻量路径的明确同意。
- **确认对象**：1.4.2 实施简报 + 两份规划产物（prd.md / implement.md；轻量任务声明在
  prd——design.md 不适用，1.4.1 双 mandate review dispatch 按 1.4 轻量规则跳过且在
  briefing 中向用户明示，用户以「开始」确认）。
- **流程凭据**：轻量路径（无 review dispatch）；implement.jsonl 6 条 / check.jsonl 4 条
  spec 上下文已配齐，`task.py validate` 通过。
- **关键口径钉死清单（执行不得偏离）**：
  - 代理脑 = 10-25-latent-flow-exit-gnn-symw run_symw_fold0 stage2 best（val 0.8054@E144，
    `outputs/10-25-latent-flow-exit-gnn-symw/run_symw_fold0/checkpoints/
    fold0_mild_flow_best_graph_mexhat.pt`），不用历史版本；
  - 归因口径与 w500/exit-gnn 先例逐项一致：无偏 μ=0 σ·ξ（use_biased_noise=False，断言
    μ_bias None）、q=0.5、σ_ref=0.3、log[0.01,0.48]×24、8 例（MDD s1,5,7,10 / HC
    s21,25,27,30）× 30 trial 全量、seed 42、bf16；τ 分位含对角；
  - **归因算法逻辑零改动**：`--symw` 旗标仅触碰 ckpt 守卫 + payload 元数据分支 +
    hparams 键，σ*/τ/conn/probe/bisection 计算路径逐行不动（diff 审 + 旗标关逐位对拍
    兜底）；
  - conn 交付 v4 = −(ln σ*+0.5·ln2πe)+5/7（主跑存 σ*+v1，离线重映射 + 三断言）；
  - 双 GC = v1 mild40（1–45 Hz 固定阶 8）+ v2 mild40（1–125 Hz AIC 4–25）：offdiag
    16256、aligned/flipped × Spearman/Pearson、per-subject + group + group_mean，描述性
    登记不构成代理脑资格判定；
  - 对称性诊断（R7）+ A_eff 无向 caveat 三处随档；EC 结果随代理脑标 provisional。
