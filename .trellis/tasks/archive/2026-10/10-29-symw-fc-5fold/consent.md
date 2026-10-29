# Consent — 10-29-symw-fc-5fold

## 1.0 任务创建同意

用户原话（2026-10-29 会话，立项确认问询）：
> "latent-flow-symw的5fold的ckpt都在result目录下，然后建立一个轻量任务"

- 同意创建轻量任务（PRD-only）
- ckpt 输入以 `result/checkpoints/mild/latent_flow_symw/` 注册件为准（非 outputs/ 原始路径）

## 1.4 实施同意（review → briefing → 明确同意）

Briefing 内容（改动点/关键逻辑/潜在风险）已在会话中完整呈现：
- 改动点：新增 `models/latent-flow-v1/fc_metric_mild_symw.py` + `outputs/10-29-symw-fc-5fold/` 产物 + 登记类文件（experiment.md / progress.md）
- 关键逻辑：镜像 fc_metric v2.1 口径（逐被试拼接→corrcoef→组平均→上三角 Pearson/Spearman + 逐被试 t 检验），w500 口径 obs375/fc125，确定性采样 K=1，四重守卫（manifest SHA / norm SHA×5 折 / ckpt SHA×5 折 / fold 前缀锁定）
- 风险与对策：折-权重错配守卫拒跑；z 域↔原始域等价数值断言（AC4）；fold0 确定性复跑（AC5）；指标 1 通过标准（组平均 r≥0.95）只作判定陈述不预设结论；result/ 全程只读

用户明确同意（2026-10-29 会话，AskUserQuestion 应答）：
> "同意，开始实施（推荐）"
