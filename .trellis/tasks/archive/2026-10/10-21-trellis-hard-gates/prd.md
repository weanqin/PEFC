# Trellis 规范硬门（Edit/Write 拦截 + start 同意证据 + 收尾校验）与 SessionStart 瘦身重排

## Goal

落地诊断方案 1+4：把「无任务不改代码」「start 前必须用户同意」「收尾三件套」从软提醒升级为 hook 硬拦截；SessionStart 科研记忆摘要化并把工作流硬门提到注入最前面。原则：把「记得做」改造成「做不了别的」。

## Background

诊断结论（本任务来源）：现有规范 100% 走软提醒通道（SessionStart/UserPromptSubmit 注入文本），无任何硬拦截；三个历史症状（不路由、review 后不问、不呈现细节）均属「漏执行」而非「执行错」。`workflow.md:128-132` 维护注释自证 planning gate skip 与 3.4 commit skip 曾实际发生。

## Requirements

- **R1 编辑拦截门**：无 in_progress 任务（含无任务、planning 状态）时，Edit/Write 对 `.trellis/{tasks,research,workspace}/**` 之外路径的写入被 PreToolUse hook 阻断，阻断信息说明原因与补救路径（先分类/建任务/走 1.4 门）。`.trellis/{tasks,research,workspace}/**` 任何时候放行（规划产物/研究记忆/日志）；任务 in_progress 后全路径放行。
- **R2 start 同意证据门**：agent 经 Bash 执行 `task.py start` 时，若目标任务目录缺少 `consent.md`（记录 briefing 要点 + 用户明确同意的原话 + 时间），执行被阻断并给出写入指引；存在则放行。不改 task.py（受 trellis update 管理），走 PreToolUse Bash 拦截实现。
- **R3 收尾校验门**：Stop 事件时，若存在 in_progress 任务且 `git status --porcelain` 非空（排除 `.trellis/.runtime/`），Stop 被阻断并提示收尾三件套（2.2 check / 3.3 spec-update / 3.4 commit）；无活动任务或工作区干净则静默放行。平台上限 3 次续跑，天然防死循环。
- **R4 SessionStart 瘦身重排**：
  - 新增「铁律」hook，注册在 SessionStart 首位，最先注入（内容 = 三条硬门 + 逃生开关，≤15 行）；
  - 科研记忆瘦身：`context.md` 只注入「速览」节、`environment.md` 只注入「速览」节（各自无该节时回退注入全文，迁移安全），`progress.md` 维持现状；注入块附完整文件路径指针供按需 Read；
  - 三个源文件补写「速览」节：只提炼既有事实，不新增任何结论，原正文不删改。
- **R5 管理文件零改动**：全部新逻辑放项目本地新文件（`.zcode/hooks/` 新脚本 + `.zcode/config.json` 注册），不修改 trellis update 管理的 stock 文件（session-start.py、inject-shell-session-context.py、inject-subagent-context.py、inject-workflow-state.py、task.py、workflow.md）。
- **R6 逃生开关**：新 hook 尊重既有 `TRELLIS_HOOKS=0` / `TRELLIS_DISABLE_HOOKS=1` 约定。
- **R7 失败安全（不对称设计）**：编辑门判定异常时 fail-closed（宁可误拦，附逃生说明）；start 门与收尾门判定异常时 fail-open（避免锁死会话/提交流程），stderr 留警告。

## Acceptance Criteria

- [ ] **AC1** 手工 stdin 场景矩阵全绿（design.md §测试矩阵的每个场景 × 期望退出码/输出，逐条跑通）
- [ ] **AC2** 注入体积实测：SessionStart 研究记忆注入量从现状（context 20.5KB + env 6.9KB 全文）降到速览后目标（合计 ≤10KB），并给出前后对比数字
- [ ] **AC3** 既有流程不回归：planning 期间写 prd/design/implement.md、in_progress 期间改代码、有 consent.md 的 start、干净工作区的 Stop——全部不被误拦（AC1 矩阵覆盖 + 本任务自身全程作为真实验证）
- [ ] **AC4** `context.md` / `environment.md` 速览节为既有事实提炼，无新结论；原正文零删改（git diff 验证只有新增节）
- [ ] **AC5** stock 管理文件零改动（git diff 验证）；新文件均在 `.zcode/hooks/`，经解释器调用（不依赖可执行位），config.json 注册格式与既有条目一致
- [ ] **AC6** trellis-check 通过（含 hook 脚本 lint/语法检查、手动场景矩阵复跑）

## Out of Scope

- 面包屑（workflow-state 块）内容修改 = 诊断方案 3，本任务不做
- AGENTS.md 铁律写入 = 诊断方案 2，本任务不做
- task.py / workflow.md / stock hooks 的任何修改
- 科研记忆源文件除「新增速览节」外的任何内容变更
