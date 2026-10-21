# Implement — Trellis 规范硬门 + SessionStart 瘦身重排

> 执行前提：本文件经 1.4 review + briefing + 用户明确同意，且 `consent.md` 已写入后，方可 `task.py start`（consent 门 S2 上电后同样约束后续任务，本任务的 consent.md 即首个实例）。

## S1 gate-file-edits.py（R1）

- [ ] 新建 `.zcode/hooks/gate-file-edits.py`，按 design §2.1 实现：escape 开关 → root 解析 → 常显放行集（仅 `.trellis/{tasks,research,workspace}/**`；repo 外路径一律需 in_progress）→ **三级身份链路**（payload resolve_context_key → env 兜底 → 48h 新鲜单会话 fallback）→ 状态判定 → exit 2/fail-closed
- [ ] 顶层 try/except 保证退出码只出 0/2；stderr 含被拦路径 + 三级链路诊断 + 补救说明（措辞按 design §2.1，含「Bash 写码同样违反约定」与「TRELLIS_HOOKS=0 须用户在启动环境设置」）
- [ ] 复用既有惯例：stdin UTF-8 reconfigure、`_script_project_root()`、`tool_input`/`toolInput` spread、`common.active_task` 导入方式（参照 inject-shell-session-context.py）
- 验证：T1/T2/T2b/T2c/T5/T5b/T5c（无任务态可直接测）

## S2 gate-task-start-consent.py（R2）

- [ ] 新建 `.zcode/hooks/gate-task-start-consent.py`，按 design §2.2：shlex 扫全部 start 实例 → task_ref 向后扫描跳过 `-` 开头 token → **镜像 resolve_task_dir 全语义**（路径形态 ref 含 `/` 直接解析目录就地查 consent.md；名字形态走精确 → `*-<ref>` 后缀唯一；歧义/无命中 fail-open）→ consent.md 实质性检查（≥1 行非空非纯标题）→ 任一实例缺即 exit 2/fail-open
- [ ] 退出码只出 0/2；stderr 指引按 design §2.2 全文
- 验证：T6/T6b/T6c/T6d/T6e/T7/T7b/T8/T9（一律用临时假任务目录，不用本任务目录）

## S3 gate-wrapup-check.py（R3）

- [ ] 新建 `.zcode/hooks/gate-wrapup-check.py`，按 design §2.3：活动任务解析**复用 S1 三级链路**（含诊断缓冲）→ in_progress + porcelain 非空（runtime 已 gitignore 无需过滤）→ exit 2，stderr = 脏路径摘要（前 10）+ 3.4 两分组协议（未识别脏文件单列征求、不擅入）+ `task.py finish` 暂停出路；退出码只出 0/2，fail-open
- 验证：T10/T11（T12 留待 S8 后以真实 Stop 放行为证，取证时序按 design §4 T12）

## S4 inject-iron-rules.py（R4a）

- [ ] 新建 `.zcode/hooks/inject-iron-rules.py`（固定文本 ≤15 行，按 design §2.4 四条，含 Bash 写旁路宣告与 finish 暂停出路；hookSpecificOutput 格式同 inject-research-context.py）
- 验证：T13

## S5 inject-research-context.py 瘦身 + 源文件速览节（R4b/R4c）

- [ ] **执行时实测基线**（context.md 现值已非 20498 B，AC2 前后对比取同时点口径）并记录
- [ ] 修改 `inject-research-context.py`：context/environment 改「速览节 or 回退全文」注入 + 指针行
- [ ] `context.md` 顶部补 `## 速览`（只压缩既有事实；技术路线 registry 每路线一行）；`environment.md` 顶部补 `## 速览`（常驻规则 + 关键环境行）；两文件正文零删改（含 context.md 已失守的「≤80 行」行，不动，见 design §2.6）
- [ ] git diff 确认两源文件纯新增节（AC4）
- 验证：T14 + 注入字节前后对比（AC2）

## S6 config.json 注册

- [ ] 按 design §3 修改 `.zcode/config.json`：SessionStart 首位加 iron-rules；Bash 追加 consent 门；新增 `Edit|Write|MultiEdit|NotebookEdit|ApplyPatch` 与 Stop 条目
- [ ] `python3 -c "import json;json.load(open('.zcode/config.json'))"` + **对照 diagnosing-hooks 契约形式核对**：事件名七选一正确、matcher 正则大小写、timeout 单位为秒
- 注：S6 是三门的「上电时刻」，其后本任务全程受门管辖（S8 的 3.3/3.4 必须在 `finish` 之前完成，见 NOTE-5）

## S7 全量验证（2.2 收口）

- [ ] 测试矩阵 T1–T14（含 T2b/T2c/T4b–T4d/T5b/T5c/T6b–T6d/T7b）逐条跑，结果记入 `{TASK_DIR}/experiment.md` Run 块
- [ ] `python3 -m py_compile` 全部 4 个 hook 文件
- [ ] git diff 确认 stock 文件零改动：session-start.py / inject-shell-session-context.py / inject-subagent-context.py / inject-workflow-state.py / task.py / workflow.md（AC5）
- [ ] 本任务自身作为 E2E：in_progress 下 Edit/Write 未被误拦的记录、S6 后真实 Stop 被拦记录、首次真实 Edit/Stop 的身份解析链路来源，全部留档 experiment.md

## S8 收尾

- [ ] 3.3 spec update（**必须在 `task.py finish` 之前**——finish 后 spec 路径不在常显放行集，编辑门会拦）：「硬门三件套 + config 注册片段 + common/active_task 耦合点 + consent.md 契约 + Bash 写旁路残余风险」写入 `.trellis/spec/`（挂 backend 或 guides 下合适位置，读 index 后定）
- [ ] 清理测试用临时会话文件与临时假任务目录
- [ ] 3.4 分批 commit：AI 改动按逻辑分组出提交计划征求用户；10-19 遗留脏路径（若用户未另行处置）单列为 unrecognized 征求 include/exclude，不擅入
- [ ] T12 取证：预期结论已在 S7 写入 experiment.md；最终 commit 后本会话真实一次 Stop 的结果（放行 exit 0，或被拦且仅列外来脏路径）以会话记录为证，experiment.md 补记放 `finish` 之后随收尾 commit 落盘

## 回滚点

- S1–S4 任一步异常：删除对应新文件即可，无副作用
- S5 异常：`git checkout --` 三个文件
- S6 异常：`git checkout -- .zcode/config.json`
- 全量回滚见 design §6（4 个新文件 + spec 回退）
