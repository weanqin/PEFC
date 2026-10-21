# implement — 执行计划

> 评审修订（2026-10-21）：θ\* 数值口径 = EMA swap + bf16 autocast（阻断 B1/B2 修法，briefing 拍板 2026-10-21）。

- S1 `models/ec-globaltau-v1/` 落地（adapter.py / probe_site.py / run_ec_globaltau.py + 单元断言小测）：
  加载 smoke（θ\* 加载 + **EMA swap 生效断言**：swap 后单窗输出与 EMA 正典一致、与 raw 权重输出不同 +
  单窗采样 shape/确定性 + **同输入同批形跨调用逐位一致**）→ 扰动语义断言（§2 三断言）。
- S2 冒烟：fold0 test 1 例（s12）× n_trials=briefing 拍板值，端到端产出 07-04 schema JSON。
- S3 确定性验证：同 seed 复跑冒烟例，σ\* 矩阵逐位 diff=0。
- S4 主跑：10 例（若 briefing 批准）+ 诊断汇总（τ/钳位/耗时）；≥1 h 量级时 nohup 后台化，
  日志落 `outputs/10-21-ec-globaltau-v5main/run_01/logs/`。
- S5 （若纳入）refs 对照：10 例配对 aligned/flipped ρ 描述性汇总。
- S6 收尾：experiment.md Run 登记 + **新 EC 口径条目入 `.trellis/spec/research/evaluation-calibers.md`**
  + 研究记忆回填 + 归档提交。

验收 = prd AC1–AC6。门禁：S1 断言不过不进 S2；S3 失败阻断 S4。
