# consent — 10-25-latent-flow-exit-gnn-symw

## 1.4 同意记录（implementation consent gate）

- **时间**：2026-10-25（主会话，方案审阅同日）
- **用户确认原话**：「确认」
- **确认对象**：prd.md 方案 v1（上一轮会话全量 briefing 呈现）：
  ① 配置派生基座 = exit-gnn Run 001 全配置原样（w500 口径 500/375/125、batch 64、
  lr 1e-3、epochs 150、patience 30、seed 42、--unfreeze-decoder、--flow-exit-gnn、
  VAE 只读复用 w500 ckpt、fold0）；② 两项变更 = `--anti-uniform-lambda 0.2`（原 0.1）
  + 新旗标 `--sym-w`（每步 optimizer step 后投影 `W ← (W+Wᵀ)/2`，SymWAdam 变体层
  Adam 子类，目标 `model.flow.graph_mixer.W`，包体零改动）；③ 命名 = 任务目录
  `10-25-latent-flow-exit-gnn-symw` + 产物 `outputs/10-25-latent-flow-exit-gnn-symw/
  run_symw_fold0/`；④ 执行序 = 自验 → 冒烟 → 正式 Run → 对称断言 + 质量抽查/viz →
  experiment.md 登记（θ 判断留用户）；⑤ 科学含义登记（W 对称 ⇒ A_eff 对称 ⇒ EC
  方向性按构造受限）。
- **流程凭据**：轻量任务（正则/优化器级变体 + wrapper 旗标接线，PRD-only），review
  派发按 workflow 轻量条款豁免（同 exit-gnn 先例）；方案全量 briefing 已于上一轮
  会话呈现（含变更落点/命令/风险登记），本轮用户确认即过 1.4 门。
