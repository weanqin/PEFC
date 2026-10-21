# 实验日志 — Trellis 规范硬门 + SessionStart 瘦身重排

> 每次「跑批/训练/评估/统计」追加一个 Run 块：跑前先写「目的+配置」，跑后同会话补「结果+结论」，不留空块。
> 格式契约见 `.trellis/spec/research/experiment-log.md`；产物只落本任务声明的变体目录（`.trellis/spec/research/path-isolation.md`）。

## Run 001 — SessionStart 科研记忆速览化瘦身（AC2/AC4/T14，2026-10-21）

### 目的与假设

- 假设：`inject-research-context.py` 改为「`## 速览` 节 or 回退全文」注入 + 指针行后，研究记忆注入量可从全文基线（执行时实测约 35 KB）降到 ≤10 KB，且两源文件改动为纯新增节（AC4）、`progress.md` 与未初始化守卫行为不变。

### 配置

- 代码：`.zcode/hooks/inject-research-context.py`（先测未修改版基线，再测 git 工作区修改版）；源文件 `.trellis/research/{context,environment}.md`（各新增 `## 速览` 节）
- 参数：注入字节 = stdout JSON 的 `hookSpecificOutput.additionalContext` 的 UTF-8 字节数；同时点口径（基线与瘦身在本次会话先后实测）
- 环境：本机 base（Python 3.12.3），对照 `.trellis/research/environment.md`
- 数据：`.trellis/research/` 现仓三档案（context.md 24,469 B / environment.md 7,461 B / progress.md 12,010 B）
- 命令：`printf '{}' | python3 .zcode/hooks/inject-research-context.py` + python 解析计字节

### 结果

- 指标：

  | 口径 | 基线（全文注入） | 瘦身后（速览+指针） | 变化 |
  |---|---|---|---|
  | research-context 块 | 24,488 B | 2,287 B | −90.7% |
  | research-progress 块（当前状态节，不动） | 3,343 B | 3,343 B | 0 |
  | research-environment 块 | 7,484 B | 947 B | −87.3% |
  | **合计 additionalContext** | **35,669 B** | **6,931 B** | **−80.6%** |

- 产物：速览节落 `.trellis/research/context.md`（+14 行）/ `environment.md`（+7 行）；测试 harness 落本任务 `research/test-gates.py`
- 图表：无（纯文本口径）
- 反证检查：瘦身后注入文本不含正文深处标记（「既有结论（已验证）」「派发惯例」均 absent）→ 只含速览 + 指针行

### 结论

- 支持：AC2 达成（35,669 B → 6,931 B，≤10 KB 目标达成）；AC4 达成（git diff 两文件合计 21 insertions / 0 deletions，纯新增节）；T14 PASS（exit 0，注入仅速览+指针）
- 登记（非本次改动）：context.md 维护规则自述「完整注入」「≤80 行」两处表述在速览化后失真——两者属 AC4 不许动的正文，按 design §2.6 留后续任务处理；速览与正文一致性由既有维护规则覆盖
- 下一步：S6 注册后下一次真实新会话的 SessionStart 注入顺序（铁律最前）与实际体积由主会话留档本文件

## Run 002 — 三门 + 铁律 stdin 场景矩阵 T1–T14（AC1/AC5，2026-10-21）

### 目的与假设

- 假设：按 design §2 判定顺序实现的 4 个新 hook，在 design §4 全矩阵（含 T2b/T2c/T4b–T4d/T5b/T5c/T6b–T6e/T7b 变体）上给出期望退出码与输出；编辑门 fail-closed、start 门与收尾门 fail-open 的不对称成立；stock 管理文件零改动（AC5）。

### 配置

- 代码：`.zcode/hooks/{gate-file-edits,gate-task-start-consent,gate-wrapup-check,inject-iron-rules}.py`（新增）+ `inject-research-context.py`（修改）；`.zcode/config.json` 注册（SessionStart 首位铁律、Bash 追加同意门、Edit|Write|MultiEdit|NotebookEdit|ApplyPatch 编辑门、Stop 收尾门）
- 参数：矩阵 T1–T14 全 33 项断言；会话夹具 = `.trellis/.runtime/sessions/` 临时会话文件（`os.utime` 调制 mtime，新鲜 ≤48h / 陈旧 96h），真实会话文件 mtime 临时 stash 后复原（内容零改动）；假任务 = `.trellis/tasks/zz-test-*`（planning/in_progress/有无 consent.md 各态），测后全删
- 环境：本机 base（Python 3.12.3）；harness 子进程 env 剥离全部会话身份变量后按场景注入 `TRELLIS_CONTEXT_ID` / `TRELLIS_HOOKS=0`
- 数据：真实脏工作区（本任务 11 条脏路径）作为 T11 与真实态探针输入
- 命令：`python3 .trellis/tasks/10-21-trellis-hard-gates/research/test-gates.py`（harness 以 `printf <json> | python3 <hook>` 同构方式驱动）；`python3 -m py_compile` 全部 5 个 hook 文件；`git diff` 六个 stock 文件

### 结果

- 指标：33/33 PASS（harness rc=0）。逐项：

  | 用例 | 期望 | 实测 | 判定 |
  |---|---|---|---|
  | T1 无会话 Edit `models/foo.py` | exit 2，stderr 含路径+三级链路+补救 | rc=2，①payload/②env/③fallback/补救全在 | PASS |
  | T2 无会话 Edit `.trellis/tasks/x/prd.md` | exit 0 | rc=0 | PASS |
  | T2b 放行集负例（spec/zcode/scripts 三路径） | exit 2 | 三例均 rc=2 | PASS |
  | T2c repo 外绝对路径 `/root/autodl-tmp/x/out.png` | exit 2 | rc=2 | PASS |
  | T3 会话→planning 任务 Write 码 | exit 2 | rc=2（state 含 planning） | PASS |
  | T4 会话→in_progress 任务 Edit 码 | exit 0 | rc=0 | PASS |
  | T4b 无身份 payload + `TRELLIS_CONTEXT_ID`→in_progress | exit 0（env 链路） | rc=0 | PASS |
  | T4c 无 env，唯一新鲜(≤48h)会话→in_progress | exit 0（fallback） | rc=0 | PASS |
  | T4d 同 T4c 但 mtime 96h（陈旧） | exit 2 | rc=2（③拒绝） | PASS |
  | T5 stdin 非法 JSON | exit 2 + 逃生说明 | rc=2，含 `TRELLIS_HOOKS=0` | PASS |
  | T5b `TRELLIS_HOOKS=0` 无任务写码 | exit 0 | rc=0 | PASS |
  | T5c MultiEdit/ApplyPatch 变体无任务写码 | exit 2 | 两例均 rc=2 | PASS |
  | T6 `start zz-test-consent-gate` 无 consent.md | exit 2 + 指引 | rc=2，含 consent.md/1.4/trellis-review/briefing/重试 start | PASS |
  | T6b `start --allow-empty-context <假任务>` | exit 2（防取 flag） | rc=2 | PASS |
  | T6c 字面 `start trellis-hard-gates`（后缀→真实任务，有 consent） | 见结论偏差注记 | rc=0（后缀唯一解析 + 放行） | PASS（见偏差①） |
  | T6c-block `start consent-gate`（后缀→假任务，无 consent） | exit 2 | rc=2 | PASS |
  | T6d `start A && start B`，A 有 B 无 consent | exit 2（任一缺即拦） | rc=2 | PASS |
  | T6e `start .trellis/tasks/<假任务>` 路径形态 | exit 2 | rc=2 | PASS |
  | T7 同 T6 但有实质 consent.md | exit 0 | rc=0 | PASS |
  | T7b consent.md 仅标题行 / 空文件 | exit 2 | 两例均 rc=2 | PASS |
  | T8 `task.py list` 非 start | exit 0 | rc=0 | PASS |
  | T9 stdin 非法 JSON | exit 0 + stderr 警告 | rc=0，stderr 警告非空 | PASS |
  | T10 收尾门无活动任务 | exit 0 静默 | rc=0，stdout/stderr 全空 | PASS |
  | T11 in_progress + 真实脏工作区 | exit 2 + 两分组协议 + finish 出路 | rc=2，四要素全在 | PASS |
  | T12 in_progress + 过滤后无脏路径 | exit 0 | 预测见结论；实测留 S8 | 留证 |
  | T13 铁律任意 stdin | exit 0 + 合法 hookSpecificOutput ≤15 行 | rc=0，SessionStart/5 行，Bash 旁路与 finish 出路文案在 | PASS |
  | T14 research hook 现仓 | exit 0，仅速览+指针 | rc=0，6,931 B | PASS |

- 产物：harness `research/test-gates.py`；py_compile 5 文件全过；`git diff` 六 stock 文件（session-start.py / inject-shell-session-context.py / inject-subagent-context.py / inject-workflow-state.py / task.py / workflow.md）**0 行差异**
- 真实态探针（S6 上电后）：收尾门对当前真实状态（③fallback 命中真实会话 → in_progress + 11 条脏路径）rc=2，stderr 列前 10 条脏路径 + 两分组协议 + finish 暂停出路——与 design §2.3 逐条一致；T1 式拦截样本 stderr 含被拦路径、三级链路诊断、补救全文（含「Bash 写码同样违反约定」与逃生开关说明）
- E2E（本任务自身）：S6 后本 implement 代理的每次 Edit/Write 均经编辑门——落 `.trellis/tasks/**`（experiment.md）时命中常显放行集放行，未发生误拦；本代理（子代理形态、payload 无身份键）未被拦本身即「允许集/回退链路」真实数据点

### 结论

- 支持：AC1 全绿（33/33）；AC5 达成（stock 零改动 + 新文件全在 `.zcode/hooks/` + config 注册契约核对通过：事件名合法、SessionStart 顺序 iron-rules→session-start→research、timeout 秒制）；fail-closed（T1/T5）与 fail-open（T9）不对称成立
- 偏差①（T6c，如实登记）：design 表面字面命令 `start trellis-hard-gates` 在本仓后缀唯一解析到**真实任务 10-21-trellis-hard-gates**（其 consent.md 已存在）→ 实测 rc=0 而非表注 rc=2；表注「无 consent.md」场景以补充用例 T6c-block（`start consent-gate` → 假任务）兑现，后缀匹配语义（解析+拦截两侧）均验证
- T12 预测（按 design §4 取证时序，先记于此）：S8 收尾 commit 落盘且无外来脏路径时，真实 Stop → 收尾门 exit 0 放行；若届时存在用户/他任务脏路径，则被拦且 stderr 仅列外来路径——两者均算有效取证；最终以 `task.py finish` 前后的会话记录为准，experiment.md 补记随收尾 commit 落盘
- 留观（仓内不可验证，如实登记）：PreToolUse exit-2 阻断理由对 agent 的实际可见性、Stop 续跑 3 次上限——均属平台行为，待真实会话观察后由主会话补记；下一次新会话 SessionStart 注入顺序与体积同批留档
- 下一步：S8（3.3 spec update 须在 `task.py finish` 之前）→ 分批 commit → 真实 Stop 取证补记

## Run 003 — trellis-check 独立复核：全矩阵复跑 + AC1–AC6 逐条验证（2026-10-21）

### 目的与假设

- 假设：check 代理独立复核下，design §2.1–2.4 判定语义逐条与四个 hook 实现一致，全矩阵复跑仍 33/33，临时夹具零残留、真实文件零改动，AC1–AC6 逐条可复核；Run 001 的 AC2 数字可复现且口径可澄清。

### 配置

- 代码：工作区当前版（4 个新 hook + `inject-research-context.py` 修改版 + `config.json`）；对照 design §2–§4、workflow 1.4/2.2/3.3/3.4、stock `common/task_utils.py` / `common/active_task.py` 源语义
- 参数：矩阵 T1–T14 全 33 项断言；另加 `_overview_or_full` 回退分支（无速览→全文+指针）单元验证、真实态探针、HEAD 版 hook 反事实基线复测
- 环境：本机 base（Python 3.12.3）
- 数据：真实脏工作区（探针时点 12 条脏路径）；真实会话文件 1 个（claude_session_6e89…，mtime 18:01:56）
- 命令：`python3 .trellis/tasks/10-21-trellis-hard-gates/research/test-gates.py`（前后 ls 比对夹具残留）；`python3 -m py_compile` ×4 hook；`git diff --stat` stock 六文件；`git show HEAD:.zcode/hooks/inject-research-context.py` 重跑基线并按块计字节

### 结果

- 指标：矩阵复跑 **33/33 PASS**（harness rc=0）；抽查关键项 T1 / T4d / T6e / T7b[heading-only+empty] / T11 / T14 全过，T14 注入 **6,931 B** 与 Run 001 精确一致
- 静态：py_compile 4 个新 hook 全过；stock 六文件 `git diff` **0 行差异**；`.zcode/config.json` 与 design §3 逐项一致（SessionStart 首位 iron-rules、Bash 追加 consent 门、`Edit|Write|MultiEdit|NotebookEdit|ApplyPatch`、Stop 无 matcher、timeout 秒制）
- 夹具与真实文件：sessions/ 目录前后清单一致、无 `zz-test-*` 残留；真实会话文件 mtime 复原至 18:01:56（与 consent.md 时间戳吻合）、内容零改动
- 真实态探针：编辑门对 `models/foo.py` rc=0（in_progress 放行，③fallback 命中唯一新鲜会话）；收尾门 rc=2 且 stderr 四要素齐全（脏路径前 10+余量、两分组协议、未识别 include/exclude、`task.py finish` 暂停出路）
- 反事实基线复测：HEAD 版 hook × 当前仓文件（已含速览节）= **38,711 B**（context 块 26,702 / progress 3,363 / environment 8,354）；Run 001 基线 35,669 B 与「速览节加入前」源文件（context 24,469 B / environment 7,461 B）经 git diff 新增量核算精确吻合——Run 001 基线口径 = 真实历史现状全文注入
- 回退分支单元验证：无 `## 速览` 时 `_overview_or_full` 返回全文+指针；有速览时深正文标记 absent——迁移安全成立

### 结论

- 支持：AC1–AC6 逐条复核通过，**无需代码修复**；Run 001 的 T14/瘦身后数字（6,931 B）精确复现
- 澄清登记（复核发现，非缺陷）：Run 001「同时点口径」仅对 hook 代码成立，基线数据取自速览节加入前的源文件（差值核算证实）；按当前仓反事实（不瘦身需注 38,711 B），瘦身降幅实为 −82.1%，AC2 结论方向不变且更保守
- 偏差②③补登记（与 Run 002 偏差①并列）：T4c/T4d 的 mtime 调制（`os.utime` + 真实会话 mtime stash/复原）即 design §4 原定夹具方法，非偏离；`inject-iron-rules.py` 的 `ZCODE_NON_INTERACTIVE=1` 跳过分支镜像 `inject-research-context.py` 既有约定（非交互会话不注入铁律、门不受影响），属保守默认
- 下一步：S8 照旧（3.3 spec update → 清理 → 分批 commit → T12 真实 Stop 取证补记）
