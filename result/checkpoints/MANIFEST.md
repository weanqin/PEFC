# result/checkpoints/ — 模型 checkpoint 注册清单（MANIFEST）

> **首次注册**：2026-10-30（任务 `.trellis/tasks/10-29-result-ckpt-reg/`）。
> **写入许可**：用户指令点名 result 目录（result-dir-policy 情形 1，同意凭据见任务 `consent.md`）。
> **维护约定**：存入后不就地修改；更新 = 新版本文件并排 + 本清单登记（result-dir-policy「已定版不动」）。

## 筛选标准与核查范围

- 标准：**五折训练 5/5 完成 且 五折 EC（扰动归因交叉评估）5/5 完成**，两者缺一不选；
- 范围：mild（Mild Depression）+ modma（MODMA）两数据集下全部模型族（outputs/ 全量核查，2026-10-30）；
- 结论：**唯一合格 = mild · latent-flow symw 五折**；不合格清单见文末。

## 合格注册：mild · latent-flow symw（五折，10 文件）

| 项 | 内容 |
|---|---|
| 模型变体 | `10-25-md-lf-anti0.2-gnnexit-symw`：两阶段 latent-flow（stage1 VAE 冻结 + stage2 flow-only），GNN 出口混合 + anti-uniform λ=0.2 + 每步强制对称化 W←(W+Wᵀ)/2（SymWAdam） |
| 数据集 | Mild Depression：250 Hz，128 ch，40 例（HC20/MDD20），窗 500 / 步 250（obs 375 + fc 125），逐被试逐通道 z-score |
| 五折训练（stage2 best val） | fold0 **0.8054@E144**（150 顶格）· fold1 **0.8264@E116** · fold2 **0.8433@E113** · fold3 **0.8358@E113** · fold4 **0.8769@E144**（150 顶格）；均值 **0.8376** |
| 五折 EC | 扰动归因 EC，conn 公式 **v4 = −(ln σ\*+0.5·ln2πe)+5/7**（q0.5 / σ_ref 0.3 / 网格 [0.01,0.48]×24 / P(resp>τ)=0.500 逐例精确），8 例/折；产物：fold0 `outputs/10-25-symw-ec-unbiased/`、fold1/2 `outputs/10-25-symw-ec-f12/`、fold3/4 `outputs/10-28-symw-ec-f34/` |
| GC 对照 | v1 refs（bivariate 谱 GC，1–45 Hz 固定阶 8）：五折组 Spearman **+0.134 / +0.159 / +0.176 / +0.2124 / +0.1822**（fold0→fold4） |
| W 对称性 | best ckpt W `torch.equal` 级对称断言五折全过（max|W| 1.24–1.59） |

### 文件清单（`cp -p` 只读复制 + `cmp` 逐位一致 + sha256 随档）

| # | 折 | 角色 | 文件（result 内路径） | sha256 | size (B) | 源 mtime | 来源任务 | 原始路径 |
|---|---|---|---|---|---|---|---|---|
| 1 | 0 | stage2 flow | `mild/latent_flow_symw/fold0/fold0_mild_flow_best_graph_mexhat.pt` | `43564dd326ad7b209ecba367f37b68362af1514f617b27b71da1c67893d6654d` | 5,219,587 | 2026-10-25 15:29 | 10-25-latent-flow-exit-gnn-symw | `outputs/10-25-latent-flow-exit-gnn-symw/run_symw_fold0/checkpoints/` |
| 2 | 0 | stage1 VAE（借用） | `mild/latent_flow_symw/fold0/fold0_mild_vae_best.pt` | `340cfb3147350533992a08f379ca94724da788a26da9a3e28275c792569ba944` | 21,495,207 | 2026-10-24 14:12 | 10-24-latent-flow-mild-w500 | `outputs/10-24-latent-flow-mild-w500/run_w500_fold0/checkpoints/` |
| 3 | 1 | stage2 flow | `mild/latent_flow_symw/fold1/fold1_mild_flow_best_graph_mexhat.pt` | `06eb59fe35c53693107cc06d3614cfa5698e9ce0c2cd616bafee3d029d97e4a0` | 5,219,587 | 2026-10-25 18:38 | 10-25-latent-flow-symw-f12 | `outputs/10-25-latent-flow-symw-f12/run_symw_fold1/checkpoints/` |
| 4 | 1 | stage1 VAE | `mild/latent_flow_symw/fold1/fold1_mild_vae_best.pt` | `98b707ae10e493e5a290e499e3673aa99407db232ba40063acb6506c30e54fd2` | 21,495,207 | 2026-10-25 18:26 | 10-25-latent-flow-symw-f12 | `outputs/10-25-latent-flow-symw-f12/run_symw_fold1/checkpoints/` |
| 5 | 2 | stage2 flow | `mild/latent_flow_symw/fold2/fold2_mild_flow_best_graph_mexhat.pt` | `4aaad72eab5a92433ac5674a8e9b005ea04428d355d0e6454b46b9e5f7f5303d` | 5,219,587 | 2026-10-25 18:57 | 10-25-latent-flow-symw-f12 | `outputs/10-25-latent-flow-symw-f12/run_symw_fold2/checkpoints/` |
| 6 | 2 | stage1 VAE | `mild/latent_flow_symw/fold2/fold2_mild_vae_best.pt` | `71ea8c6735f7e4f7dd3108ba6c326c0951f78bac7cb5bea83d5fd504ff9625da` | 21,495,207 | 2026-10-25 18:45 | 10-25-latent-flow-symw-f12 | `outputs/10-25-latent-flow-symw-f12/run_symw_fold2/checkpoints/` |
| 7 | 3 | stage2 flow | `mild/latent_flow_symw/fold3/fold3_mild_flow_best_graph_mexhat.pt` | `421edf87a191421a1babe3a8dce3cb7a4d78fff132a9e0fa32804767ddde6c44` | 5,219,587 | 2026-10-28 15:00 | 10-28-latent-flow-symw-f34 | `outputs/10-28-latent-flow-symw-f34/run_symw_fold3/checkpoints/` |
| 8 | 3 | stage1 VAE | `mild/latent_flow_symw/fold3/fold3_mild_vae_best.pt` | `85f62424cef7de3b94aa0f445e178e7c70bccbb7a83a368db29586a124bb2fd4` | 21,495,207 | 2026-10-28 14:48 | 10-28-latent-flow-symw-f34 | `outputs/10-28-latent-flow-symw-f34/run_symw_fold3/checkpoints/` |
| 9 | 4 | stage2 flow | `mild/latent_flow_symw/fold4/fold4_mild_flow_best_graph_mexhat.pt` | `3d8a4e974dfe40e2a462c71a0bbe1e86446f3fe9ebd6f8883c6dc005da2caa2c` | 5,219,587 | 2026-10-28 15:22 | 10-28-latent-flow-symw-f34 | `outputs/10-28-latent-flow-symw-f34/run_symw_fold4/checkpoints/` |
| 10 | 4 | stage1 VAE | `mild/latent_flow_symw/fold4/fold4_mild_vae_best.pt` | `3e6e7c7c7a786abf65de7199e9d333b8d888ee02031708d0486651a19b6c821a` | 21,495,207 | 2026-10-28 15:08 | 10-28-latent-flow-symw-f34 | `outputs/10-28-latent-flow-symw-f34/run_symw_fold4/checkpoints/` |

合计 10 文件，133,548,970 B（≈127 MB）。所有任务目录均在 `.trellis/tasks/`（10-25-latent-flow-exit-gnn-symw、10-24-latent-flow-mild-w500、10-25-latent-flow-symw-f12、10-28-latent-flow-symw-f34）。

### fold0 VAE 借用注记

symw fold0 的 stage1 沿用 w500 基座 ckpt（stage2 config `cli/vae_ckpt` 指证 → `outputs/10-24-latent-flow-mild-w500/run_w500_fold0/checkpoints/fold0_mild_vae_best.pt`），symw run 目录内无本折 VAE 副本；fold1–4 均为本折自训 VAE。上表 #2 即该借用件，来源任务相应记为 10-24-latent-flow-mild-w500。

## 不合格清单（未注册，2026-10-30 核查时点）

| 数据集 | 模型族 | 训练 | EC | 不合格原因 |
|---|---|---|---|---|
| mild | latent-flow w500（10-24-latent-flow-mild-w500） | 1/5 | 1/5（10-24-mild-w500-ec-unbiased） | 仅 fold0 |
| mild | latent-flow exit-gnn（10-24-latent-flow-exit-gnn） | 1/5 | 1/5（10-24-exitgnn-ec-unbiased） | 仅 fold0 |
| mild | latent-flow w500-no-reg-flow-only（10-28） | 1/5 | 1/5 | 仅 fold0 |
| mild | 10-21 三窗口旧线（10-21-latent-flow-mild-train，归档负结论） | 单 run | 0 | 未做五折 |
| modma | latent-flow symw（10-28-modma-symw-f0） | 1/5 | 1/5（10 例） | 仅 fold0，fold1–4 未训 |
| modma | latent-flow 0705 std graph-mexhat（10-29-modma-0705-f1234 + fold0 在途） | 5/5（ckpt 齐：fold0 `outputs/10-29-modma-ec-te-corr/fold0/run/` + fold1–4 `outputs/10-29-modma-0705-f1234/fold{1..4}/run/`） | **4/5**（fold1–4 完成，各 10/11/11/11 例；fold1 按设计 10 例，fold2–4 GC refs 仅覆盖 10/11） | **fold0 EC 未完成**：fold0 训练 2026-10-30 22:38 起在另一会话运行（活动任务 10-29-modma-ec-te-corr，链式 train→EC→DONE）——跑完即满足双全，届时补注册（须一并处理仓库外 stage1 VAE ×5：`/root/autodl-tmp/latent_ode/checkpoints/fold{0..4}_oc_latent_flow_vae_best.pt`，均已确认在位） |
