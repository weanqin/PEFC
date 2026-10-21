# 1.4 实施同意记录（consent）

- **时间**：2026-10-21 18:01 CST
- **用户确认原话**：「确认开始」（对 1.4.2 实施简报的回复）
- **简报要点**：
  1. 新增 3 个硬门 hook：gate-file-edits.py（无 in_progress 任务拦 Edit/Write 写码）、gate-task-start-consent.py（`task.py start` 前查本文件）、gate-wrapup-check.py（Stop 时 in_progress + 脏工作区则拦）
  2. 新增 inject-iron-rules.py（SessionStart 最先注入铁律）+ inject-research-context.py 速览化瘦身（~29KB → ≤10KB，context.md/environment.md 各补「速览」节）
  3. 修改 `.zcode/config.json` 注册以上；零 stock 管理文件改动
- **拍板项默认**（简报推荐项，用户未另作选择即按推荐执行）：
  - 10-19 遗留脏路径保留原位（Stop 门给 finish 暂停出路，不擅动）
  - Bash 写启发式拦截不进本任务，登记残余风险（design §5.2）
  - SessionStart matcher 纳入 `resume` 另行任务，本轮不动（design §5.8）
- **验证承诺**：28 项 stdin 场景矩阵 + py_compile + stock 零改动 diff + 本任务自身 E2E（含 S6 上电后真实 Stop 被拦记录），结果记 experiment.md
