# Workflow Hard Gates（Trellis 规范硬门契约）

> 2026-10-21 由任务 `10-21-trellis-hard-gates` 建立。把「无任务不改码 / start 前用户同意 / 收尾三件套」从软提醒升级为 ZCode hook 硬拦截。本文件是三个门的可执行契约：改 hook 行为前先改这里。

## 1. Scope / Trigger

- 触发：任何 agent 在本仓库经 ZCode 执行 Edit/Write 类工具、`task.py start`、或会话 Stop。
- 实现：`.zcode/hooks/gate-{file-edits,task-start-consent,wrapup-check}.py` + `.zcode/hooks/inject-iron-rules.py`（SessionStart 首位铁律）+ `inject-research-context.py` 速览化。全部为**项目本地文件**；stock 管理文件（session-start.py / inject-*-*.py / task.py / workflow.md）零改动，`trellis update` 不会覆盖。

## 2. Signatures

```jsonc
// .zcode/config.json 注册形态（timeout 单位=秒；重放注册时照抄）
"SessionStart": [ { "matcher": "startup|clear|compact", "hooks": [
  {"type":"command","command":"python3 \"<root>/.zcode/hooks/inject-iron-rules.py\"","timeout":15},      // 首位=注入顺序最前
  {"type":"command","command":"python3 \"<root>/.zcode/hooks/session-start.py\"","timeout":30},
  {"type":"command","command":"python3 \"<root>/.zcode/hooks/inject-research-context.py\"","timeout":15} ]}],
"PreToolUse": [
  {"matcher":"Agent|Task", "hooks":[ /* stock 注入，不动 */ ]},
  {"matcher":"Bash", "hooks":[
    {"type":"command","command":"python3 \"<root>/.zcode/hooks/inject-shell-session-context.py\"","timeout":5},
    {"type":"command","command":"python3 \"<root>/.zcode/hooks/gate-task-start-consent.py\"","timeout":10} ]},
  {"matcher":"Edit|Write|MultiEdit|NotebookEdit|ApplyPatch", "hooks":[
    {"type":"command","command":"python3 \"<root>/.zcode/hooks/gate-file-edits.py\"","timeout":10} ]}],
"Stop": [ { "hooks": [ {"type":"command","command":"python3 \"<root>/.zcode/hooks/gate-wrapup-check.py\"","timeout":15} ] } ]
```

- 阻断协议：**退出码 2 = deny（stderr 即阻断理由），0 = 放行；禁止输出其他退出码**（平台把非 0/2 记为 hook 错误）。不输出 JSON 决策（严格 schema 多一个键即失效）。
- 逃生开关：`TRELLIS_HOOKS=0` / `TRELLIS_DISABLE_HOOKS=1`（读 hook 进程自身 env；agent 无法为单次调用注入，只能由用户在启动环境设置）。SessionStart 系 hook 另镜像 `ZCODE_NON_INTERACTIVE=1` 跳过。

## 3. Contracts

### 3.1 编辑门 gate-file-edits.py
- 放行集（任何时候）：`.trellis/tasks/**`、`.trellis/research/**`、`.trellis/workspace/**`。其余（含 `.zcode/**`、`.trellis/{spec,scripts}/**`、repo 外绝对路径如 `/root/autodl-tmp/**`）一律要求活动任务 `status=in_progress`。
- 活动任务三级链路：① `resolve_context_key(hook_input, platform, allow_environment_context=False)`；② 同函数 `allow_environment_context=True`（函数内部读 `TRELLIS_CONTEXT_ID` 与平台 env 表，**不按变量名自查**，表含 UNVERIFIED 项）；③ fallback：`.trellis/.runtime/sessions/` 下**恰好一个** mtime ≤48h 且指向 `in_progress` 任务的会话文件。三链路全空按无任务处理。失败 fail-closed。
- **耦合点**：①② 导入 trellis 管理的 `common/active_task.py`——`trellis update` 若改其 API，编辑门会 fail-closed 全拦；修复方式：stderr 三级链路诊断定位 → 按 `common/active_task.py` 现签名适配。

### 3.2 start 同意门 gate-task-start-consent.py
- `task.py start` 执行前，目标任务目录必须有 `consent.md`：≥1 行非空、非纯 `#` 标题文本，内容 = briefing 要点 + **用户确认原话 + 时间**。
- task_ref 解析镜像 `common/task_utils.resolve_task_dir` 全语义：含 `/` 的路径形态直接解析目录就地查；名字形态精确 → `*-<ref>` 后缀唯一；跳过 `-` 开头 token 防 flag 前置绕过；多 start 实例任一缺即拦。歧义/无命中 fail-open（task.py 同语义报错）。
- 已知影响存量任务：恢复暂停任务（如 10-19）也须先补 consent.md。

### 3.3 收尾门 gate-wrapup-check.py
- 活动任务 in_progress 且 `git status --porcelain` 非空（`.trellis/.runtime/` 已 gitignore）→ exit 2：脏路径前 10 条 + 3.4 两分组协议（AI 改动分组提交征求确认；未识别脏文件单列 include/exclude 不擅入）+ 暂停出路 `task.py finish`（只清指针、任务保持 in_progress）。异常 fail-open 静默。
- 平台上限：Stop 续跑最多拦 3 次，之后会话照常结束（不锁死）。

### 3.4 研究记忆速览注入 inject-research-context.py
- `context.md` / `environment.md` 只注入 `## 速览` 节（至下一个 `## `），节缺失回退全文 + 各块附完整路径指针（细节按需 Read）；`progress.md` 维持「当前状态」节选。
- **维护规则同步义务**：事实/进度/环境变化回写源文件时，必须同步更新该文件速览节，否则 SessionStart 注入的是过期摘要。
- 已登记的正文失真（AC4 不许改，留后续任务）：context.md 首部「完整注入每个新对话」与「≤80 行」两句在速览化后失真。

## 4. Validation & Error Matrix

| 条件 | 行为 |
|---|---|
| 无任务/planning 写放行集外路径 | exit 2 + 路径 + 三级链路诊断 + 补救说明 |
| stdin 非法 JSON（编辑门） | fail-closed exit 2 + 逃生说明 |
| start 无 consent.md / consent.md 空或纯标题 | exit 2 + 1.4 门指引 |
| start ref 歧义/不存在/解析异常 | fail-open 放行（task.py 报错） |
| Stop 无任务/干净树/异常 | 静默放行 |
| `TRELLIS_HOOKS=0` | 三门全放行 |

## 5. Good/Base/Bad Cases

- Good：会话内任务 in_progress → 任意编辑放行；`start` 前 consent.md 就位 → 放行。
- Base：无身份键的子代理编辑 → fallback/env 链路命中本会话 in_progress 指针 → 放行。
- Bad（必拦）：无任务直接 Edit `models/x.py`；`start` 不写 consent.md；带着未提交改动反复 Stop。

## 6. Tests Required

- 全矩阵 33 项见任务 `10-21-trellis-hard-gates/research/test-gates.py`（可复现 harness）与 experiment.md Run 002；改动任一门后**必须全量重跑**。
- 断言点：退出码、stderr 关键要素（路径/链路诊断/两分组协议/finish 出路）、放行集正负例、三级链路逐级、fail-open/closed 不对称（同一非法 stdin 两门反向）。

## 7. Wrong vs Correct

#### Wrong（被门拦的典型）
```
# 无活动任务直接改模型代码
Edit models/foo.py                      → exit 2
# 想绕过 briefing 直接启动
Bash: python3 .trellis/scripts/task.py start 10-xx-task    → exit 2（无 consent.md）
# 用 Bash 重定向写码绕过编辑门
Bash: cat > models/foo.py <<EOF ...     → 不被技术拦截，但违反铁律第 1 条（已登记残余风险）
```

#### Correct
```
# 建任务 → 1.4 briefing → 用户「确认开始」→ 写 consent.md → start
Write .trellis/tasks/10-xx-task/consent.md   # 含用户原话与时间
Bash: python3 .trellis/scripts/task.py start 10-xx-task     → 放行
```

## 残余风险登记（v1 已接受）

1. Bash 重定向/heredoc/`sed -i` 写文件绕过编辑门（软约束兜底；启发式拦截留用户拍板）。
2. consent.md 可伪造、`task.json` status 可直改——门防「无意识遗漏」不防「故意绕过」，审计痕迹留 consent.md。
3. 48h fallback 窗口内的陈旧指针若恰好唯一且 in_progress 可借 fallback 获写权限。
4. SessionStart matcher 不含 `resume`（resume 会话无铁律/科研记忆注入；既有行为，另任务处理）。
5. `config.json` 若被 `trellis update` 重生成：按 §2 注册片段重放。
