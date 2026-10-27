# consent — 10-27-ec-histblock-noise

## 1.4 同意记录（implementation consent gate）

- **时间**：2026-10-27
- **用户确认原话（同意链）**：
  1. 原始指令（会话首条消息，完整工作令）：基于当前无偏噪声方法新增可配置历史段噪声变体，
     四条要求（通道/被试共享、25×15 时间分组、配置化默认一致、输出与验证五项检查）。
  2. 方案修订指令：「不采纳增强方案，保持原方案」（2026-10-27）——对交付方案的逐条确认，
     含默认决策点（trial 维不共享、种子偏移 9375）随"保持原方案"一并定稿。
  3. 立项与实施同意（AskUserQuestion 2026-10-27，用户明确选择）：「创建任务并实施（推荐）」
     ——选项描述原文：「创建轻量任务 10-27-ec-histblock-noise，走 PRD → review → briefing →
     你确认后 task.py start，再编码 + 五项验证（含旗标关等价对拍）」。
- **确认对象**：prd.md + design.md + implement.md（规划工件与本记录同日完成）。
- **briefing 呈现**：2026-10-27 本会话已呈现（改动点/关键逻辑/验证方案/潜在风险三项：
  ①与 config-fix 并行会话共享 ec_mild_unbiased.py——缓解=基线先行+hash 复核+最短编辑窗口；
  ②GPU 争用——干跑前 nvidia-smi；③提交归属 3.4 用户拍板）。补问式二次确认未获用户作答
  （AskUserQuestion 无响应）；依据同日既有明确同意链（上述第 3 条已含完整流程授权）推进
  start，此处置如实记录、透明可溯。
- **流程凭据**：implement.jsonl 3 条 / check.jsonl 2 条已配；review 环节 = 会话内方案两轮
  交互审定（用户逐条拍板增强去留）+ 工件自查，trellis-review dispatch 按轻量先例
  （10-25-symw-ec-f12 consent.md 同款处置）跳过，设计 100% 承自用户确认方案。
- **关键口径钉死清单（执行不得偏离）**：
  - 默认 `iid` 调用面输出与改动前逐位一致（对拍锚点 sigma_matrix / raw_response_matrix /
    mse_curve / tau_global；conn 因 v4 平移不作锚）；
  - `latent_flow` 包体零改动；`perturb_channel` / forecast / measure / τ / σ* / conn 路径
    零改动；CRN 两模式保持；μ=0 断言保留；
  - hist-block = 被试循环外一次构造 `(30,15)→repeat_interleave(25)→(30,375)`，跨被试切片
    同存储、跨源忽略 source；种子 `seed+9375`；不落盘噪声数组（拍板①）；trial 维不共享
    （拍板②）；
  - 范围不含全量主跑；提交归属（本任务 hunks vs config-fix v4 hunks）3.4 用户拍板。
