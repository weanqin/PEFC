# 实验日志 — 10-30-symw-ablation-mw

> 每次「跑批/训练/评估/统计」追加一个 Run 块：跑前先写「目的+配置」，跑后同会话补「结果+结论」，不留空块。
> 格式契约见 `.trellis/spec/research/experiment-log.md`；产物只落本任务声明的变体目录（`.trellis/spec/research/path-isolation.md`）。

## Run 001 — S1 冒烟：fold0 × A3 双移除臂 × 2 epoch（2026-10-30）

### 目的与假设

- 验证 γ 置零旗标链路端到端可跑（`--mexhat-gamma 0 --mexhat-gamma-m 0` 经 wrapper/守卫/训练/存档），并演练训后守卫断言（W 逐位零 / log_sigma 冻结 / A_eff≡0）。不产正式数。

### 配置

- 代码：`models/latent-flow-v1/run_ablation_mw.py`（本任务新增驱动）+ 既有 `train_mild_depress.py` 零改动
- 参数：argv 从基线 fold0 `stage2_config.json.cli` 逐键重建，仅改 run_dir→smoke/、γ 两键置 0、epochs=2、limit-subjects=2
- 环境：conda base @ 本地 4080S（environment.md 在档）
- 数据：Mild Depression 40 例 fold0 划分（mild_5fold_seed42.json）
- 命令：`python3 models/latent-flow-v1/run_ablation_mw.py --smoke`

### 结果

- rc=0（34 s，CUDA，2 epoch × 2 subjects）；跑前对拍 diff 键 = {epochs, limit_subjects, mexhat_gamma, mexhat_gamma_m, run_dir}（恰为冒烟允许集）；15 组正式组合 argv 预演对拍全过。
- 守卫实测：`W_abs_sum = 0.0`（逐位零）、`log_sigma_absdiff = 0.0`（=−2.034316062927246 恰为构造初值）、`A_eff_max_abs = 0.0` 精确、γ buffer 0/0、W 对称、uni_reg=1.000 常数（A3 by-design）。
- 附带复验：`--arm full --fold 0` 基线 ckpt 守卫通过（γ 1.0/f32(0.1)、W 训后非零对称）。
- 产物：`outputs/10-30-symw-ablation-mw/smoke/`

### 结论

- γ 置零链路端到端成立，scale-to-zero 硬移除的数学预期（W/log_sigma 逐位不动、A_eff≡0）全部实证；S2 放行。

## Run 002 — S2 正式批：A1/A2/A3 × fold0–4 共 15 跑 stage-2（2026-10-30）

### 目的与假设

- 量化 M/W 各自及联合贡献（Q-A/Q-B/Q-C）：三消融臂同折同 stage-1、除 γ 两键外与基线逐字同参训练；假设 M 先验与 W 残差均有正贡献，双移除退化最大。

### 配置

- 代码：同 Run 001（正式模式；argv 重建 + 跑前 CLI 对拍守卫，diff 键 ⊆ {run_dir, mexhat_gamma, mexhat_gamma_m}）
- 参数：基线同折逐键（epochs 150/lr 1e-3/batch 64/patience 30/seed 42/λ_anti 0.2/unfreeze-decoder/flow-exit-gnn/sym-w/obs 375/fc 125/window 500）+ 臂定义 γ：A1=(0, 0.1)、A2=(1.0, 0)、A3=(0, 0)
- 环境：conda base @ 本地 4080S 串行 fail-fast（nohup）
- 数据：同基线五折；vae_ckpt = 基线同折（fold0=w500、f1/2=f12、f3/4=f34，路径 resolve 绝对化）
- 命令：`nohup python3 models/latent-flow-v1/run_ablation_mw.py --stage2-batch > outputs/10-30-symw-ablation-mw/batch.log 2>&1 &`
- 训后守卫：`python3 models/latent-flow-v1/ablation_guards.py --all`（AC2）

### 结果

- 15/15 完成 rc=0（单跑 725–774 s），跑前 CLI 对拍 15/15（diff 键恰 = {mexhat_gamma, mexhat_gamma_m, run_dir}）；训后守卫 15/15 PASS + 基线 full 五折复验 5/5 PASS（guards/*.json 落档）。
- **M1 val datafit（recon2+corr2+cons2 @best epoch，五折）**：

| 臂 | f0 | f1 | f2 | f3 | f4 | mean±sd | best epoch（best/ran） |
|---|---|---|---|---|---|---|---|
| full | 0.8023 | 0.8234 | 0.8401 | 0.8328 | 0.8739 | **0.8345±0.0262** | 144/150 116/146 113/143 113/143 144/150 |
| noM | 0.5199 | 0.5201 | 0.5577 | 0.5212 | 0.6158 | **0.5469±0.0418** | 150/150 150/150 129/150 150/150 150/150 |
| noW | 0.8443 | 0.8347 | 0.8650 | 0.8360 | 0.8766 | **0.8513±0.0186** | 131/150 144/150 144/150 113/143 144/150 |
| noMW | 0.5301 | 0.5194 | 0.5778 | 0.4884 | 0.6010 | **0.5434±0.0455** | 116/146 150/150 116/146 140/150 150/150 |

- 产物：`run_{arm}_fold{k}/`（ckpt/history/config）+ `run_{arm}_fold{k}_stage2.log` ×15 + `guards/` ×20 + `batch.log`

### 结论

- **Q1 方向明确**：去掉 M（noM/noMW）后 val datafit 大幅更低（更优），去 W（noW）小幅更差；noM≈noMW ⇒ 预测拟合的收益全部来自移除 M，W 单独（noM 臂）即可达到。收敛 caveat：noM 4/5 折 150ep 顶格未早停（差距或仍被低估），noW/full 以早停为主。
- M1 层面 M 先验对预测拟合是**负贡献**；最终归因待 M2/M3/M4（代理脑质量）交叉——若 M 在 FC/幅度/GC 结构侧有正贡献，则构成「预测拟合 vs 结构有效性」的取舍证据。

## Run 003 — S3 M2 幅度抽查 ×15（2026-10-30）

### 目的与假设

- 各臂各折 test 代理脑幅度校准（std_ratio medoid/real）与基线并置；假设图混合移除损害幅度校准。

### 配置

- 代码：既有 `quality_check_mild.py` 零改动（--flow-exit-gnn --window-len 500 --obs-len 375 --forecast-len 125 --k 32 --seed 42；--subjects 逐折 triad：f0=1,7,21 f1=2,12,22 f2=3,6,23 f3=9,13,29 f4=4,15,24）
- 产物：`outputs/10-30-symw-ablation-mw/run_{arm}_fold{k}/quality/quality_meta.json`

### 结果

- 15/15 完成（quality_meta.json 齐备，window_len=500 口径核验由 summary 侧做）。
- **M2 std_ratio full_mean（→1 为佳）五折**：

| 臂 | f0 | f1 | f2 | f3 | f4 | mean | mean\|1−r\| |
|---|---|---|---|---|---|---|---|
| full | 0.858 | 1.011 | 0.886 | 1.038 | 0.680 | 0.895 | **0.125** |
| noM | 0.898 | 0.816 | 1.023 | 0.575 | 0.715 | 0.805 | 0.204 |
| noW | 0.865 | 0.804 | 1.047 | 1.201 | 0.749 | 0.933 | 0.166 |
| noMW | 1.230 | 0.741 | 0.899 | 0.733 | 1.033 | 0.927 | 0.178 |

### 结论

- M2 方向与 M1 **相反**：full 基线的幅度校准最接近 1（mean|1−r| 最小 0.125），noM 最差（0.204，fold3 掉到 0.575）；M/W 任一移除都损害幅度保真，且 noM > noW 损害（0.204 vs 0.166）。
- 与 M1 并置：M 先验损害 val 拟合但支撑幅度校准——「预测拟合 vs 代理脑质量」取舍格局在 M 侧成立；W 两侧均为正贡献。待 M3/M4 定结构侧。

## Run 004 — S4 M3 FC 一致性 ×15 + M4 A_eff↔GC ×15+5（2026-10-30）

### 目的与假设

- M3：gen-fc vs real-fc（指标 1 v2.1 镜像口径）四臂并置；假设图混合有贡献于 FC 结构保真。M4：A_eff 与组平均 GC（refs mild40 v1 档）结构一致性与否随臂变化；A2 臂以距离偏相关为主判读。

### 配置

- 代码：`fc_metric_mild_ablation.py`（镜像 fc_metric_mild_symw.py，显式 --ckpt + SHA 随档）、`aeff_gc_corr.py`（raw/|A| × aligned(=ec)/flipped(=ec.T) × plain/partial Spearman+Pearson；refs SHA256SUMS 全验）
- 数据：refs/granger_bivariate_spectral/mild40-trial500-zscore-fs250-fixed8-f1-45-favg/（只读）；M4 组平均主口径=折 test 8 例、副口径=全 40 例；A0 五折 ckpt 现提取并与 result 注册 SHA 对拍
- 产物：`outputs/10-30-symw-ablation-mw/fc/`、`outputs/10-30-symw-ablation-mw/aeff_gc/`

### 结果

- M3 FC ×20（CPU 确定性）+ M4 A_eff↔GC ×20（CPU threads=1）全部完成；full 臂五折 CPU 重算作并置基线，与 10-29 注册值（CPU 单线程先例口径）**逐折一致**：0.95636/0.77146/0.69556/0.86072/0.96251（fold0 锚点逐字段精确吻合，含逐被试样本 SHA）；设备选择（cuda vs cpu）对指标影响 ~1e-5 量级已随 fc json `determinism_note` 登记。
- **M3 组平均 Pearson r 五折 mean±sd**：full 0.8493±0.1163（2/5 折过 0.95 线）｜noM **0.4289±0.1392（0/5 过线）**｜noW 0.8761±0.1309（0/5 过线：4 折落 0.91–0.95 带、fold3 0.643 掉线）｜noMW **0.9711±0.0048（5/5 全过线，四臂最佳）**。
- **M4 Spearman（主口径 = 折 test 8 例组平均）**：raw_aligned full +0.497 / noW +0.507 / noM +0.033；**距离偏相关后全部归零**（full −0.037 / noW −0.019 / noM −0.078，|A_eff| 口径）⇒ full/noW 的 raw 相关完全由 M 的距离梯度中介；副口径全 40 例同构（raw +0.60/+0.61 → partial ≈ −0.05/−0.03，full/noW）。noMW NaN(by-design) 守卫生效。
- 产物：`fc/`×20 json+matrices、`aeff_gc/`×20 json（含 refs SHA 全验 + A_eff SHA/范数登记）

### 结论

- **M3：图混合是 FC 一致性的负因子**——完全无跨通道混合（noMW）反而最好（0.971，五折全过合格线），W 单独（noM）最差（0.429）；M 在场时部分抵消 W 的 FC 损害（full 0.849 / noW 0.876 ≈ 同档）。
- **M4：M/W 均不携带超出距离梯度的 GC 结构**——full/noW 的 raw +0.50 全由距离中介（partial 后 ≈0），W 单独学到近零结构。M4 为描述性并置。

## Run 005 — S5 汇总：四臂五折 mean±sd + 损失曲线并置（2026-10-30）

### 目的与假设

- AC3/AC4 交付：M1–M4 主表（mean±sd ddof=1）+ 逐折 Δ vs A0 + best_epoch/停止状态列 + datafit@argmin(datafit) 敏感性副列 + loss_curves.npz/png。

### 配置

- 代码：`ablation_summary.py`（本任务新增）；输入 = Run 002–004 产物 + 基线五折 history/quality/fc summary 复用

### 结果

- strict 模式全量收集通过（20 跑 × M1–M4 源，provenance 守卫无警告）；`summary/summary_5fold.json` + `README.md`（表 1 主表/表 2 逐折/结论段 + 6 条预登记 caveat）+ `loss_curves.npz`（80 数组：四臂五折 × loss/recon2/corr2/cons2 逐 epoch）+ `loss_curves.png` 落档。
- Q-A/Q-B/Q-C 五折配对 Δ（A0 − 臂）见 README 结论段；stage-2 三分量（要求 7）：M1 recon2 full 0.4216 / noM 0.2957 / noW 0.4305 / noMW 0.2898，corr2 0.2112/0.1016/0.2206/0.0973，cons2 0.2017/0.1497/0.2003/0.1562。

### 结论

- **Q-B（M 边际，A0 − A1）**：M1 datafit +0.288（损失型指标：M 抬高损失、强损预测拟合，三分量同向）但 M2 幅度 +0.089、M3 FC +0.420（M 强撑代理脑质量）——**「预测拟合 vs 代理脑有效性」取舍在 M 上成立**。
- **Q-A（W 边际，A0 − A2）**：M1 datafit −0.017（W 降低损失、小幅利拟合）、M3 −0.027（近中性）、M4 无结构增量——W 的贡献小而偏拟合侧。
- **Q-C（图混合总贡献，A0 − A3）**：M1 datafit +0.291（移除全部跨通道混合后损失更低）、M3 −0.122（**移除全部跨通道混合反而同时改善拟合与 FC**，幅度近中性 −0.033）——现实现下 GNN 图混合整体是代理脑质量的负因子，唯一正贡献在 M2 幅度（full 最佳 0.125 偏离）。
- 综合判读（单 seed/五折配对）：symw 线的 A_eff = M+W 是「幅度校准」与「FC 保真/预测拟合」之间的现行折中；noMW 臂首次五折全过 FC 合格线（0.95），提示通道独立 + 解码器路径已足以支撑 FC 结构，图混合的价值需要在新维度（如扰动 EC 有效性）上重新论证。后续决策同步 progress.md。
