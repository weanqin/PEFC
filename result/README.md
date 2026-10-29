# result/ — 正式实验最终结果（论文数据专用）

> 本目录存放**最终用于论文写作的结果数据**。写入权限严格受控（用户 2026-10-30 指令，规约见
> `.trellis/spec/research/result-dir-policy.md`）。

## 写入门（必须满足其一，否则禁止写入）

1. **用户明确指令存入**——例如："请你将此次结果存放到 result 目录中"；
2. **Agent 主动询问并获用户明确同意**。

除上述两种情形外，**不得将任何文件写入本目录**。

## 约束

- 实验过程中的**中间结果 / 非正式结果**一律落 `outputs/<任务目录>/` 等位置，禁止混入本目录；
- 每次存入应随档登记：来源任务（`.trellis/tasks/...`）、产物路径、生成时间、口径摘要——保证论文数字可溯源；
- 本目录内容为正式交付物，存入后**不就地修改**；更新 = 新版本文件并排 + 登记说明（对齐 path-isolation「已定版不动」原则）。

## 当前状态

- 2026-10-30 建目录，规约生效。
- **2026-10-30 首次注册**（任务 `.trellis/tasks/10-29-result-ckpt-reg/`，写入许可情形 1）：mild · latent-flow symw 五折 ckpt ×10（5 flow + 5 VAE，fold0 VAE 借用 w500 基座）→ `checkpoints/mild/latent_flow_symw/fold{0..4}/`；逐文件 sha256 / 来源任务 / 口径摘要 / 不合格模型清单见 `checkpoints/MANIFEST.md`。modma·0705 差 fold0 EC（在途）未注册，待补。
- **2026-10-30 二次注册**（同任务，用户追加指令）：mild · latent-flow symw 五折 EC_pert 结果 + 逐被试/组平均 EC×GC 相关性 ×66 文件 → `ec_pert/mild/latent_flow_symw/`（fold0/1/2 仅 recompute_v4 版本、fold3/4 v4 直出；conn 公式全 v4 断言）；逐文件 sha256 见 `ec_pert/.../sha256.json`，口径与关键数字见同目录 `MANIFEST.md`。
- **2026-10-30 三次注册**（同任务，用户追加指令）：mild · latent-flow symw 五折训练逐 epoch 损失历史 ×10（5 折 stage2 + 4 折本折 stage1 + fold0 借用 w500 stage1）→ `loss_data/mild/latent_flow_symw/fold{0..4}/`；history 内 `val` 为 4 例验证划分曲线（非 8 例测试折——测试折无训练侧损失数据文件，3-batch smoke 行已抄录于 MANIFEST），口径与 best val 表见同目录 `MANIFEST.md`。
