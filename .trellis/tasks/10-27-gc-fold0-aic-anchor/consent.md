# 1.4 启动同意记录

- **时间**：2026-10-27（用户在 OC↔GC 汇报后的会话回复）
- **用户确认原话**：「补全」
- **方案要素（上轮汇报已呈现，用户据此拍板）**：补全 10 例 OC↔GC = 现算 fold0 其余
  （含 subject_0 全量重算对拍）GC 基线；口径取 **AIC 档**（max 8 / 1–125 Hz / nf65 /
  fs250 / 全长 z-score = 旧库 07-05 时代残留 JSON 同参，锚点核对必须同口径；残留
  subject_0 可做管线对拍验证）；预计 ~9 min/例 × 10 ≈ 1.5 h CPU（16 workers）；
  复用移植版 runner `--gc-fixed-order 0`（落入 AIC sweep），零代码改动 + 一个只读
  并置核对脚本；产物落本任务 outputs。
- **前置关联任务**：10-27-lf-globaltau-f0-unbiased-repro（已归档，本任务补其
  NOT RECOMPUTABLE 锚点缺口）。
