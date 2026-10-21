# Design — Trellis 规范硬门 + SessionStart 瘦身重排

## 1. 架构总览

新增 3 个项目本地 hook 脚本 + 修改 1 个项目本地 hook + 编辑 2 个研究记忆源文件，全部注册/生效于 `.zcode/config.json`：

```
SessionStart（config 顺序即注入顺序）:
  ① .zcode/hooks/inject-iron-rules.py        [新增] 铁律块（最先注入）
  ② .zcode/hooks/session-start.py            [不动] workflow 摘要
  ③ .zcode/hooks/inject-research-context.py  [修改] 速览化注入
  ④ .zcode/hooks/inject-research-context.py 输出的研究记忆（瘦身）

PreToolUse:
  matcher "Edit|Write" → .zcode/hooks/gate-file-edits.py      [新增]
  matcher "Bash"       → 既有 inject-shell-session-context.py [不动]
                        + .zcode/hooks/gate-task-start-consent.py [新增，追加]

Stop:
  无 matcher → .zcode/hooks/gate-wrapup-check.py              [新增]
```

改动面：4 个 `.zcode/hooks/` 文件（3 新 1 改）、`.zcode/config.json`、`.trellis/research/context.md` 与 `environment.md`（仅新增速览节）。**零 stock 管理文件改动**（trellis update 管理会覆盖 session-start.py / task.py / workflow.md / 三个 stock hook——本设计全部绕开）。

## 2. 组件契约

### 2.1 gate-file-edits.py（R1 编辑拦截门）

- **注册**：PreToolUse，matcher `"Edit|Write|MultiEdit|NotebookEdit|ApplyPatch"`（逐个列全，不依赖未经证实的别名主张）。
- **输入**：stdin JSON，取 `tool_input.file_path`（兼容 `toolInput.filePath` 等大小写变体，沿 inject-shell-session-context.py 的 spread 惯例）。
- **判定顺序**：
  1. 逃生开关（`TRELLIS_HOOKS=0` / `TRELLIS_DISABLE_HOOKS=1`，读 hook 进程自身 env——命令串里设置无效）→ 放行；
  2. 解析 trellis root（从 `__file__` 向上找 `.trellis/`，同既有 hook）；
  3. 目标路径归一化为 repo 相对路径，命中常显放行集 `.trellis/tasks/**`、`.trellis/research/**`、`.trellis/workspace/**` → 放行；repo 之外与放行集之外路径一律需要 in_progress（含 `/root/autodl-tmp/**` 等绝对路径目标）；
  4. 解析活动任务，三级链路（每级结果记入诊断缓冲）：① `common.active_task.resolve_context_key(hook_input, platform)`，payload 身份键模式（`allow_environment_context=False`）；② 同函数 env 模式（`allow_environment_context=True`，函数内部读 `TRELLIS_CONTEXT_ID` 与平台 env 表——不按变量名自查，平台表含 UNVERIFIED 项）→ `.trellis/.runtime/sessions/<key>`；③ 单一新鲜会话 fallback：仅当 `.trellis/.runtime/sessions/` 下**恰好一个** mtime ≤48h 的会话文件且其任务 `status=in_progress` 时采用（覆盖子代理等拿不到父身份的形态；陈旧指针靠 mtime 窗口排除——注：现存 10-19 指针 mtime 2026-10-19 19:14，在当日 19:14 前仍可借 fallback 生效，仅影响拿不到身份的子代理，本任务 start 后会话文件数为 2、fallback 随即失活）；三链路全空 → 按无任务处理；
  5. `status == "in_progress"` → 放行；否则 **exit 2**，stderr 输出阻断说明（被拦路径 + 当前状态 + 三级链路诊断 + 补救路径：「先分类本轮请求；小改动也须立轻量任务并走 1.4 review→briefing→用户同意门；确认误拦时由用户在启动环境设 `TRELLIS_HOOKS=0` 重启会话」）。
- **失败安全**：顶层 try/except 兜住一切异常，退出码只出 0/2（其他非零会被平台记为 hook 错误）；任何异常 → fail-closed（exit 2 + 异常摘要 + 逃生说明）。理由：本门的存在意义就是无任务时不让代码写入发生；fail-open 会让「hook 坏了」变成永久旁路。
- **阻断机制**：退出码 2（diagnosing-hooks 契约：`0` 放行、`2` 对 PreToolUse 即 deny），不输出 JSON（严格 schema 多一个键即失效，退出码是唯一无歧义通道）。stderr 文本即阻断理由。

### 2.2 gate-task-start-consent.py（R2 start 同意证据门）

- **注册**：PreToolUse，matcher `"Bash"`，在既有 shell-session hook 之后追加（config 数组顺序）。
- **判定顺序**：
  1. 逃生开关 → 放行；
  2. shlex 解析 `tool_input.command`，找出全部 `task.py start` 实例（沿既有 `_extract_task_subcommands` 的扫描方式）；无 → 放行；
  3. 每个实例的 task_ref：从 `task.py start` 之后**向后扫描跳过以 `-` 开头的 token**（防 `--allow-empty-context` 前置形态取到 flag），取第一个位置参数；未取到 → 放行交由 task.py 自行报错；
  4. 每个实例的 task_ref 解析任务目录，**镜像 `resolve_task_dir` 全语义**（common/task_utils.py）：
     - ref 含 `/` 或为绝对路径（路径形态）→ 直接解析为目标目录（相对 repo root 或绝对），不落在 `.trellis/tasks/` 下或不存在 → fail-open 放行（task.py 同语义报错）；存在 → 就地查该目录 consent.md；
     - ref 为纯名字（名字形态）→ 先精确目录名匹配，失败后缀匹配 `*-<ref>`，唯一才命中；歧义或无命中 → fail-open 放行（task.py 会以同样语义报错，两侧行为一致）；
  5. 命中目录的 `consent.md` 存在且含 ≥1 行非空、非纯标题（`#` 开头）文本 → 该实例放行；缺失或空 → **exit 2**（多实例时任一不过即拦），stderr 指引：「1.4 门未满足：先派 trellis-review（复杂任务）→ 向用户呈现 briefing（改动点/关键逻辑/潜在风险/路径与环境/验证方案）→ 用户明确同意后把同意记录写入 {TASK_DIR}/consent.md（含用户确认原话与时间）→ 重试 start」。
- **不拦截** `create` / `finish` / `archive` 等；用户在自己终端手跑 `task.py start` 不经此 hook（hook 只拦 agent 的 Bash 工具）。
- **失败安全**：解析/IO 异常 → fail-open（放行 + stderr 警告），退出码只出 0/2。理由：start 本身还受 task.py 校验与会话身份约束，误拦会卡死合法启动；且本门可通过「故意破坏 hook」绕过，属可接受的残余风险（见 §5）。

### 2.3 gate-wrapup-check.py（R3 收尾校验门）

- **注册**：Stop 事件，无 matcher（全匹配；平台上限 3 次续跑防死循环）。
- **判定顺序**：
  1. 逃生开关 → 放行；
  2. 活动任务解析**复用 §2.1 同一三级链路**（含诊断缓冲），无任务或 `status != "in_progress"` → 静默放行；
  3. `git status --porcelain`（`-c core.quotepath=false`；`.trellis/.runtime/` 已被 gitignore，porcelain 天然不显示，无需额外过滤）非空且存在 runtime 之外的脏路径 → **exit 2**，stderr 内容：(a) 脏路径摘要（前 10 条）；(b) 复述 3.4 两分组协议——本会话 AI 改动 → 分批提交计划征求用户确认；未识别脏文件（他任务遗留/用户手改）→ **单列征求 include/exclude，不得擅入提交**；(c) 暂停/中止出路——「非本任务脏路径不要求本次处理；需暂停请 `task.py finish`（只清本会话指针，任务保持 in_progress 可恢复），之后 Stop 放行」；
  4. 过滤后无脏路径 → 放行。
- **失败安全**：顶层 try/except，任何异常 → fail-open 静默放行，退出码只出 0/2。理由：Stop 是会话出口，误拦的代价（会话卡住）大于漏提醒；退出码 1 会被平台记为 hook 错误。

### 2.4 inject-iron-rules.py（R4a 铁律前置注入）

- **注册**：SessionStart 数组**首位**（config 数组顺序 = 注入顺序，铁律块落在 workflow 摘要之前）。
- **内容**（≤15 行，固定文本，中文）：
  1. 无 in_progress 任务不得写入代码路径——Edit/Write 等文件工具已被硬拦（**经 Bash 重定向/heredoc/sed -i 写文件同样违反本条**，只是技术旁路不改变约定）；小改动也须立轻量任务；
  2. `task.py start` 前必须 review → briefing → 用户明确同意，consent.md 缺失已被硬拦；
  3. 有未提交变更时 Stop 会被拦——收尾三件套 check / spec-update / commit 完成后再停；非本任务脏路径不擅动，需暂停走 `task.py finish`（保状态清指针）；
  4. 误拦申诉：`TRELLIS_HOOKS=0` 须由用户在启动环境设置（agent 无法为自己的一次调用设置）；详细流程 `.trellis/workflow.md`，按需 `get_context.py --mode phase --step X.Y`。
- **输出**：`{"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": ...}}`（与 inject-research-context.py 同格式）。

### 2.5 inject-research-context.py 修改（R4b 瘦身）

- `context.md` / `environment.md` 改为**只注入「速览」节**（`## 速览` 到下一个 `## ` 之间；`_extract_section` 对缺失节返回空串，既有代码兼容）；节缺失时回退注入全文（迁移安全：未补速览的文件行为不变）。
- 每个注入块末尾附指针行：`(完整档案: .trellis/research/xxx.md；写/跑代码前按需 Read)`。
- `progress.md` 维持现状（已只注入「当前状态」节）。
- 未初始化守卫、失败静默 no-op、输出格式：全部保持不变。

### 2.6 研究记忆源文件补速览（R4c）

- `context.md` 顶部（维护规则行之后）新增 `## 速览`：研究主题一行、技术路线 registry 压缩表（每路线一行：名称/一句话/状态）、当前阶段一行、待拍板要点一行。**只压缩既有内容，零新增结论**；原文从「## 研究主题」起原样保留。context.md 现 77 行、在其维护规则自称的「≤80 行」预算**之内**——本任务新增速览节将**首次突破**该预算；同时首部「SessionStart hook 把本文件完整注入每个新对话」一句在 S5 后即失真。两处均为 AC4「正文零删改」不许改的正文，失真在此登记，留后续任务处理。
- `environment.md` 顶部新增 `## 速览`：常驻规则（未记录 env 必须先问）+ 硬件/Python/torch 关键行 + 数据盘约定。原正文保留。
- 速览与正文的一致性由既有维护规则覆盖（事实变化当场更新源文件时须同步速览）。

## 3. config.json 变更

```jsonc
{
  "hooks": {
    "enabled": true,
    "events": {
      "SessionStart": [
        {"matcher": "startup|clear|compact", "hooks": [
          {"type": "command", "command": "python3 \"<root>/.zcode/hooks/inject-iron-rules.py\"", "timeout": 15},   // 新增，首位
          {"type": "command", "command": "python3 \"<root>/.zcode/hooks/session-start.py\"", "timeout": 30},      // 不动
          {"type": "command", "command": "python3 \"<root>/.zcode/hooks/inject-research-context.py\"", "timeout": 15} // 不动
        ]}
      ],
      "PreToolUse": [
        {"matcher": "Agent|Task", "hooks": [ /* 不动 */ ]},
        {"matcher": "Bash", "hooks": [
          {"type": "command", "command": "python3 \"<root>/.zcode/hooks/inject-shell-session-context.py\"", "timeout": 5}, // 不动
          {"type": "command", "command": "python3 \"<root>/.zcode/hooks/gate-task-start-consent.py\"", "timeout": 10}     // 新增，追加
        ]},
        {"matcher": "Edit|Write|MultiEdit|NotebookEdit|ApplyPatch", "hooks": [
          {"type": "command", "command": "python3 \"<root>/.zcode/hooks/gate-file-edits.py\"", "timeout": 10}             // 新增
        ]}
      ],
      "Stop": [
        {"hooks": [{"type": "command", "command": "python3 \"<root>/.zcode/hooks/gate-wrapup-check.py\"", "timeout": 15}]} // 新增
      ]
    }
  }
}
```

## 4. 测试矩阵（AC1，全部用 `printf <json> | python3 <hook>` 手工驱动）

| # | hook | stdin 场景 | 期望 |
|---|---|---|---|
| T1 | gate-file-edits | 无活动会话，Edit `models/foo.py` | exit 2，stderr 含路径、三级链路诊断与补救说明 |
| T2 | gate-file-edits | 无活动会话，Edit `.trellis/tasks/x/prd.md` | exit 0 |
| T2b | gate-file-edits | 无活动会话，Edit `.trellis/spec/x.md`、`.zcode/hooks/x.py`、`.trellis/scripts/x.py` | exit 2（放行集负例：防误把整个 `.trellis/**` 放行） |
| T2c | gate-file-edits | 无活动会话，Edit `/root/autodl-tmp/x/out.png`（repo 外绝对路径） | exit 2 |
| T3 | gate-file-edits | 会话任务 planning，Write `models/foo.py` | exit 2 |
| T4 | gate-file-edits | 会话任务 in_progress，Edit `models/foo.py` | exit 0 |
| T4b | gate-file-edits | 子代理形态 stdin（payload 无身份键）+ env `TRELLIS_CONTEXT_ID` 指向 in_progress 会话 | exit 0（env 链路兜底） |
| T4c | gate-file-edits | 同 T4b 但 env 亦无，仅存一个 mtime≤48h 的 in_progress 会话文件 | exit 0（新鲜单会话 fallback） |
| T4d | gate-file-edits | 同 T4c 但会话文件 mtime>48h（陈旧指针） | exit 2（fallback 拒绝陈旧指针） |
| T5 | gate-file-edits | stdin 非法 JSON | exit 2（fail-closed）+ 逃生说明 |
| T5b | gate-file-edits | `TRELLIS_HOOKS=0` env，无任务 Edit `models/foo.py` | exit 0（逃生开关） |
| T5c | gate-file-edits | tool_name 变体（`MultiEdit`/`ApplyPatch`）无任务写码 | exit 2（判定与工具名无关） |
| T6 | gate-start-consent | `task.py start <临时假任务>`，无 consent.md | exit 2，stderr 含 consent.md 指引 |
| T6b | gate-start-consent | `task.py start --allow-empty-context <假任务>`（flag 前置形态），无 consent.md | exit 2（防取到 flag 绕过） |
| T6c | gate-start-consent | `task.py start trellis-hard-gates`（缺日期前缀 slug），无 consent.md | exit 2（后缀匹配语义） |
| T6d | gate-start-consent | `start A && start B` 双实例，A 有 consent.md B 无 | exit 2（任一缺即拦） |
| T6e | gate-start-consent | `task.py start .trellis/tasks/<假任务>`（路径形态 ref），无 consent.md | exit 2（路径形态直解析，防 find_task_by_name 拒 `/` 致 fail-open 绕过） |
| T7 | gate-start-consent | 同 T6，有 consent.md | exit 0 |
| T7b | gate-start-consent | consent.md 仅标题行/空文件 | exit 2 |
| T8 | gate-start-consent | `task.py list` 等非 start 命令 | exit 0 |
| T9 | gate-start-consent | stdin 非法 JSON | exit 0（fail-open）+ stderr 警告 |
| T10 | gate-wrapup-check | 无活动任务 | exit 0 无输出 |
| T11 | gate-wrapup-check | in_progress + 有脏路径（本任务执行中即天然满足） | exit 2，stderr 含两分组协议 + finish 暂停出路 |
| T12 | gate-wrapup-check | in_progress + 过滤后无脏路径 | exit 0（**取证时序**：预期结论先写入 experiment.md；真实放行以会话记录为证，experiment.md 补记放在 `finish` 之后随收尾 commit 落盘；**前提**：外来脏路径已按用户拍板处置，否则 Stop 被拦且仅列外来路径属 §5.6 预期行为，两者都算有效取证；不用 git stash 构造） |
| T13 | iron-rules | 任意 stdin | exit 0，stdout 合法 hookSpecificOutput JSON |
| T14 | inject-research-context | 现仓（速览节已补） | exit 0，注入文本仅含速览 + 指针，字节数记录（AC2） |

T3/T4/T4b–T4d/T11 用临时会话文件在 `.trellis/.runtime/sessions/` 下构造（mtimes 用 `os.utime` 调制，测后删除）；T6/T6b/T6c/T6d/T7b 用临时假任务目录构造（S8 清理）。`.trellis/.runtime/` 已被 gitignore，porcelain 不显示，T12 无需过滤构造。

**E2E 冒烟（AC2/AC3，硬性验收）**：本任务自身即是真实验证——in_progress 状态下全程 Edit/Write 不被误拦；S6 上电后至 S8 commit 前的每次 Stop 被拦属预期（R3），以本会话真实记录为准；**下一次新会话的 SessionStart 注入顺序（铁律最前）与体积**留档到 experiment.md。两项平台行为（PreToolUse exit-2 阻断理由对 agent 的可见性、Stop 续跑 3 次上限）仓内不可验证，留观并如实记入 experiment.md。S7 留档首次真实 Edit/Stop 的身份解析来源（三级链路哪级命中）。

## 5. 残余风险（已接受，如实登记）

1. **同意伪造**：agent 可虚构 consent.md 内容，或直接 Edit `task.json` 改 status（tasks/** 常显放行）。硬门的目标是把「无意识遗漏」变成「需故意绕过」；审计痕迹（consent.md 原话）留存。不追求防恶意。
2. **Bash 写文件旁路**：`cat > file`、heredoc、`sed -i`、`tee` 等完全绕过编辑门——且「忘记规则的 agent」同样不会记得改用 Edit/Write，故旁路与主症状同源、非故意绕过。v1 处置：铁律明文宣告「Bash 写码同样违反约定」（软约束）+ 本条登记；Bash 写启发式检测（重定向/heredoc 目标在放行集外即拦）有真实误报成本（`> out.log`、测夹具等），**是否上马留用户拍板**，不进本任务范围。
3. **hook 环境逃逸**：`TRELLIS_HOOKS=0` 可全关，且只能由用户在启动环境设置（agent 无法为单次调用注入）。与既有 hook 同约定，属用户级逃生口；副作用是「跳过 Trellis 的小改动」语义收窄为「立轻量任务」或「用户重启会话」——与 workflow.md Request Triage 的「用户说不就跳过」存在偏差，本任务不改 workflow.md，偏差在铁律文案中消化并在此登记。
4. **config.json 被 trellis update 重生成**：inject-research-context.py 先例表明项目本地注册可存活（该文件即如此注册）；若未来 update 覆盖 config.json，按 spec 沉淀的注册片段重放即可（3.3 落 spec）。
5. **resolve_context_key 依赖**：三级链路复用 `common/active_task.py`（trellis 管理文件）的导入；其 API 若在 update 中变更会破坏判定 → 编辑门 fail-closed 会放大为「全拦」。缓解：三级链路诊断随 stderr 输出、导入失败时 stderr 明示原因 + 逃生开关；3.3 在 spec 登记该耦合点。
6. **外来脏路径常态拦 Stop 属预期**：仓内现存 10-19 暂停任务遗留的 4 处脏路径（2 modified + 2 untracked 目录），任何 in_progress 会话的 Stop 都会被拦至 3 次上限或用户走 `finish` 暂停出路；stderr 已给两分组协议与暂停指引，不诱导错误提交。上电前是否先处置这批脏路径留用户拍板。
7. **Stop 单一会话新鲜度 fallback 的窗口**：48h 内的陈旧指针若恰好唯一且 in_progress，会借 fallback 获得写权限——与「子代理必须能工作」的权衡，接受并在此登记。
8. **SessionStart matcher 不含 `resume`**（既有行为，非本任务引入）：resume 会话既无铁律也无科研记忆；如要覆盖属用户级决定，另行任务。
9. **Stop 3 次上限内的体验**：连续 Stop 被拦最多 3 次，之后会话照常结束（平台上限），不会锁死；上限本身仓内不可验证，E2E 留观。

## 6. 回滚

单任务原子回滚：`git checkout -- .zcode/config.json` + 删除 **4 个**新 hook 文件（gate-file-edits / gate-task-start-consent / gate-wrapup-check / inject-iron-rules）+ `git checkout -- .zcode/hooks/inject-research-context.py .trellis/research/context.md .trellis/research/environment.md`；若已执行 S8，一并回退 `.trellis/spec/` 的新增条目。无数据迁移、无状态残留（测试用临时会话文件/假任务目录测后即删）。
