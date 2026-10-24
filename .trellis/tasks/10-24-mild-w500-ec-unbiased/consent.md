# consent — 10-24-mild-w500-ec-unbiased

## 1.4 同意记录（implementation consent gate）

- **时间**：2026-10-24（方案提交与拍板同日）
- **用户确认原话**：「请完成以下信号处理任务：1. 将下载到 autodl-tmp 目录下的 mild GC
  （格兰杰因果）参考数据注册到 refs……AIC 阶数固定为 8，滤波频带为 1–45 Hz。
  2. 参数 q 保持默认值 0.5，无需测试或计算 q 的其他取值情况。
  3. 分析时窗采用 8×30 的设置，该配置经验证是合适的，请直接使用。
  请按上述要求执行，并确保各参数设置与说明完全一致。」
- **确认对象**：主会话上一轮提交的完整扰动实验方案（范式/参数/流程）+ 本轮三项修订：
  ① mild GC 参考注册入 refs（VAR 阶 8 固定、1–45 Hz、8×30 池化口径经勘察与用户说明
  逐项一致：gc_n_lags=8、fmin=1/fmax=45+faverage、trial 前 500 点、逐 trial z-score、
  30 trials 池化）；② q=0.5 钉死、不做任何其他取值测试（原 q=0.25 + ladder 校准门
  作废，改为 mini 工程干跑 + 曲线后置诊断照登）；③ 主跑 8 例 × 30 trial 直接使用。
  用户明确同意执行（「请按上述要求执行」）。
- **流程凭据**：复杂任务双 mandate trellis-review 已派发（Mandate A 方案与方法论：
  通过，2 建议全采纳——P(MSE@σ_ref>τ) 判读量、τ 含对角口径注明；Mandate B 实施与
  风险：通过，5 建议全采纳——500/375/125 硬断言 + norm SHA 对拍 d184e647…、
  autocast 责任显式化、逐例断点续跑、共享记忆文件提交处置、预算维持）；
  prd.md / design.md / implement.md 已按拍板与评审建议修订齐备；implement.jsonl（5 条）/
  check.jsonl（3 条）已登记。
- **关键口径钉死清单（执行不得偏离）**：q=0.5；σ_ref=0.3；σ 网格 log[0.01,0.48]×24；
  8 例 × 30 trial 全量；扰动 = σ·ξ 无偏 μ=0 加在 z-score 后入模前 obs 段源通道；
  conn = −(ln σ* + ln 2πe) + 1；GC refs 注册 sid=index+1（四锚点 bitwise 已验）；
  GC 对照协议对齐 gen_unbiased_gc_corr.py 先例（描述性登记不构成资格判定）。
