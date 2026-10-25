# consent — 10-25-latent-flow-symw-f12

## 1.4 同意记录（implementation consent gate）

- **时间**：2026-10-25（主会话，方案 briefing + AskUserQuestion 同轮）
- **用户确认**：AskUserQuestion「fold1/fold2 双折训练是否创建 Trellis 轻量任务？」→
  选「**立轻量任务（推荐）**」——该问题前置消息已载明全量方案（模型 = symw 变体、
  stage2 与 Run 001 逐字同参、每折先 stage1、指标口径、三折对比）且明示「确认后即跑」，
  选择即对本方案与执行的确认。
- **方案要点（briefing 摘要）**：①模型 = 10-25 symw 变体（exit-gnn + λ0.2 + 每步
  W←(W+Wᵀ)/2），stage2 = Run 001 配置 verbatim（batch 64/lr 1e-3/epochs 150/
  patience 30/seed 42），两折仅 --fold/--run-dir/--vae-ckpt 不同；②仓库无 fold1/2
  VAE（已核查）⇒ 每折先 stage1（镜像 w500 fold0 stage1 **实际口径**：epochs 150/
  patience 30/lr 5e-4/batch 64——briefing 后修正：初稿误引旧库默认值 50/10/1e-3/32，
  实际对拍 w500 在档 stage1_config.cli 后以实际口径为准，协议不变）；③指标 = 生成式
  代理口径（val loss/分量/smoke/medoid·real std 双口径/coverage 结构性照登/W 对称
  断言），无准确率/F1（用户明示按任务类型自选）；④抽查被试每折 test 集 2 MDD+1 HC
  （fold1=2,12,22 / fold2=3,6,23）；⑤产物 `outputs/10-25-latent-flow-symw-f12/
  run_symw_fold{1,2}/`，fold0 只读参照。
- **流程凭据**：轻量纯跑批任务（零代码改动），review 派发按 workflow 轻量条款豁免
  （同 symw 先例）；数据路径/配置文件无需用户指定（manifest 固定 + 旗标内嵌，已向
  用户说明）。
