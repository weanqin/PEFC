# summary — 10-28-modma-symw-f0

> 任务级汇总；逐 Run 细节见 experiment.md（Run 001–004），决策链见 prd.md/consent.md。

## 交付物

- **代码（变体层新增，包体零改动）**：`models/latent-flow-v1/train_modma_symw.py`（MODMA fold0 × symw-exit-GNN stage-2 训练入口，mild 10-25 变体口径移植）、`models/latent-flow-v1/ec_modma_symw_v4.py`（v4 conn EC 主跑 + GC 相关双模式 driver）
- **训练产物**：`outputs/10-28-modma-symw-f0/run_01/`（ckpt `fold0_modma_symw_exitgnn_graph_mexhat.pt` + configs/stage2_config.json + logs/stage2_history.json + run_01_train.log）
- **EC 产物**：`outputs/10-28-modma-symw-f0/ec_main/subject_{0..9}.json`（10 例）+ `ec_main_run.log`
- **相关性行**：`outputs/10-28-modma-symw-f0/gccorr/gccorr_summary.json`
- **冒烟**：`outputs/10-28-modma-symw-f0/smoke/`（通路验证，含 trainable=936,482 与 mild symw fold0 逐位同值的移植硬校验）

## 关键数字

| 环节 | 结果 |
|---|---|
| 训练（150 ep 跑满，87.5 min） | best val loss **0.6532 @E139**（recon2 0.351/corr2 0.131/cons2 0.168/uni_reg 0.016）；trainable 936,482 |
| smoke（test） | MAE 0.8458 / std_ratio 0.4785（欠幅，同 mild 线已知缺陷方向） |
| EC（10/10，47.4 min） | τ ∈ [1.66e-4, 7.44e-4]；钳边 interior 52.6% / @σ_min 2.1% / @σ_max 45.3%；P(resp>τ)=0.500 逐例成立 |
| **EC×GC 逐被试** | Spearman \|EC\| aligned **−0.0260±0.0951**（10 例正负参半：+0.134 ~ −0.151） |
| **EC×GC 组平均** | 组均值矩阵 Spearman **−0.0245**（Pearson −0.0009；逐被试 Pearson 均值 −0.0009±0.0716） |

## 核心结论

1. **移植成功但相关性≈0**：symw-exit-GNN 变体在 MODMA fold0 上训练稳定收敛，但其 v4 EC 与观测 GC（AIC 口径）在两级读数上均无相关（−0.03 量级，散布内正负参半）。远低于 07-05 原版口径 OC↔GC +0.29 锚点——该差异是 exit-GNN 出口 + W 强制对称 + λ=0.2 正则 + MODMA 重训的复合效应，本任务未做拆解归因。
2. σ* 场以 @σ_max 弱端钳边为主（45.3%），与 07-05 原版近 50/50 平衡型不同；τ 比 07-05 复现值（~8e-5）高约 4 倍（扰动响应整体更敏感）。
3. flipped≈aligned（GC 近对称，无方向判别力，先例一致）。

## 边界

- 描述性并置登记，不构成代理脑资格判定；不外推。
- bf16 AMP 非逐位确定；新变体首训无对拍锚点。
- GC = 10-27 现算 AIC 口径（经旧库残留对拍逐位一致），与 07-05 原始 GC 锚点同口径。
