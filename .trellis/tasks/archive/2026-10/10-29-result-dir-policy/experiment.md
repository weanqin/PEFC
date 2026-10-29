# 实验日志 — 10-29-result-dir-policy

> 规约性小任务（无 GPU 跑批），Run 001 = 建目录 + 双层规约落档。

## Run 001 — 建立 result/ 目录与写入门（2026-10-30）

### 目的与假设

- 用户指令：项目根建 `result/`（论文最终结果专用），写入权限严格受控（两许可情形），中间结果禁入。

### 配置

- 用户指令原文见 consent.md；无代码/数据变更。

### 结果

- `mkdir /root/npi/result` ✓（空目录 + README）
- `result/README.md`：自描述规约（写入门两条许可、禁止条款、随档登记要求）✓
- `.trellis/spec/research/result-dir-policy.md`：硬规约落 spec（含 Quality Check 挂钩）✓
- `.trellis/spec/research/index.md`：规约表新增一行 + Quality Check 新增一项 ✓

### 结论

- 门已立：目录在位、规约双层（自描述 + spec hook 可见）、检查清单挂钩。后续任何 `result/` 写入须先核对许可情形并留同意证据。
