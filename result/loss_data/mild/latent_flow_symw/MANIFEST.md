# result/loss_data/ — 训练/验证损失曲线注册清单（MANIFEST）

> **注册**：2026-10-30（任务 `.trellis/tasks/10-29-result-ckpt-reg/`，第三轮追加）。
> **写入许可**：用户追加指令点名注册入 result（result-dir-policy 情形 1；指令原文见任务 `consent.md`）。
> **逐文件 sha256 / size / 原始路径**：`sha256.json`（同目录，10 文件全量，复制后 `cmp` 逐位一致）。

## 注册对象

mild · latent-flow symw（`10-25-md-lf-anti0.2-gnnexit-symw`，与前两轮 ckpt / EC_pert 同款五折双全模型）五折训练的**逐 epoch 损失历史文件** ×10（5 折 × stage2 + 4 折本折 stage1 + fold0 借用 w500 stage1）。

## 数据语义（重要——「训练 / 测试」划分辨析）

- 每个 `*_history.json` = `{"train": [...], "val": [...]}`，逐 epoch 记录损失分量：
  - stage1（VAE）：`loss / recon / td / bd / corr / cons / kl / beta`；
  - stage2（flow）：`loss / recon2 / corr2 / cons2 / kl / beta / uni_reg`（含 anti-uniform λ0.2 项）。
- **`train` = 训练划分（28 例）训练时损失；`val` = 验证划分（每折 4 例）损失**——模型选择/早停所用曲线（`best val` 即源此），**不是 8 例测试折**。
- **测试折（每折 8 例）在训练管线中没有损失数据文件**：训练结束后仅有 3-batch smoke 检查（最终权重、控制台打印，非 best-ckpt 全量），五折数值抄录如下备查；测试折的正式评估产物 = EC_pert（已注册，见 `result/ec_pert/`）。

| 折 | smoke test MAE / std_ratio（3-batch，最终权重，训练日志行） |
|---|---|
| fold0 | 0.8974 / 0.8170 |
| fold1 | 0.8156 / 0.8748 |
| fold2 | 0.9087 / 0.7791 |
| fold3 | 1.0280 / 0.9032 |
| fold4 | 0.7925 / 0.6471 |

> 如需 best-ckpt 全量测试折损失（逐折 loss 数值文件），可用已注册 ckpt 现算生成——本轮未生成（超出「注册既有文件」范围），待指令。

## 文件清单与关键数字（10 文件，599,646 B）

| 折 | 文件 | 内容 | best val | 来源任务 · 原始路径 |
|---|---|---|---|---|
| 0 | `fold0/stage2_history.json` | stage2 train/val ×150 ep | **0.8054@E144**（150 顶格） | 10-25-latent-flow-exit-gnn-symw · `outputs/.../run_symw_fold0/logs/` |
| 0 | `fold0/stage1_history.json` | **借用 w500 基座** stage1 ×87 ep | 2.0992@E57 | 10-24-latent-flow-mild-w500 · `outputs/.../run_w500_fold0/logs/` |
| 1 | `fold1/stage1_history.json` | 本折 stage1 ×87 ep | 2.0358@E57 | 10-25-latent-flow-symw-f12 · `outputs/.../run_symw_fold1/logs/` |
| 1 | `fold1/stage2_history.json` | stage2 ×146 ep | **0.8264@E116** | 同上 |
| 2 | `fold2/stage1_history.json` | 本折 stage1 ×79 ep | 2.1210@E49 | 10-25-latent-flow-symw-f12 · `outputs/.../run_symw_fold2/logs/` |
| 2 | `fold2/stage2_history.json` | stage2 ×143 ep | **0.8433@E113** | 同上 |
| 3 | `fold3/stage1_history.json` | 本折 stage1 ×91 ep | 2.0326@E61 | 10-28-latent-flow-symw-f34 · `outputs/.../run_symw_fold3/logs/` |
| 3 | `fold3/stage2_history.json` | stage2 ×143 ep | **0.8358@E113** | 同上 |
| 4 | `fold4/stage1_history.json` | 本折 stage1 ×92 ep | 2.0580@E62 | 10-28-latent-flow-symw-f34 · `outputs/.../run_symw_fold4/logs/` |
| 4 | `fold4/stage2_history.json` | stage2 ×150 ep | **0.8769@E144**（150 顶格） | 同上 |

- stage2 五折 best val 均值 **0.8376**；训练口径：50 例窗划分 28/4/8（train/val/test，五折 test 互斥覆盖全 40 例）、窗 500/步 250、stage1 VAE 冻结 + flow-only（SymWAdam W←(W+Wᵀ)/2、anti-uniform λ0.2）、bf16 AMP。
- fold0 stage1 为 w500 基座借用件（与 ckpt 清单 #2 同源，config `cli/vae_ckpt` 指证），manifest 如实记来源任务。

## 维护约定

存入后不就地修改；更新 = 新版本并排 + 登记（result-dir-policy）。
