# latent-flow fold2 nosc global-tau 单被试 GC 相关性（主路径现算）

## Goal

对 global-tau 首跑被试（fold2 test[0] = `02010010EOG.set_61_.mat`）现算原库谱 GC，按旧库判决工具的 r_abs/r_signed recipe 求与 global-tau signed connectivity 的相关，单被试存证。

## Background

- 上游：任务 10-21-latent-flow-globaltau-f2-nosc 已产出 `outputs/10-21-latent-flow-globaltau-f2-nosc/run_01/subject_0.json`（signed_connectivity/connectivity/raw response 等全矩阵）。
- **GC 计算主路径**（2026-10-20 拍板）：`latent_flow/baseline/granger.py` 成对双变量谱 GC；[a,b]=a→b 行=源，与 global-tau connectivity 行=源同向，offdiag 直接对齐（无需转置）。口径 = refs 归档同款：fixed_order=8 / 1–125 Hz / n_freq=65 / fs 250 / 全长连续 / 逐通道 z-score。
- 旧库无 fold2 历史 baseline GC 可复用（results/ 仅 fold0 时代 baseline_connectivity），故现算。
- **相关性 recipe**（`_compare_signed_de_ablation.py`，nosc fold2 r_abs=0.585 同源）：r_abs = Pearson(|target| offdiag z-score, |ref| offdiag z-score)（先取绝对值再 z-score）；r_signed = Pearson(target offdiag z, ref offdiag z)；offdiag = 全双三角去对角。

## Requirements

- R1 GC 现算：`granger_connectivity(data, fs=250, max_order=8, n_freq=65, fmin=1, fmax=125, fixed_order=8, n_jobs=8)`，data = MODMA .mat 全长逐通道 z-score。
- R2 相关性：r_abs / r_signed（recipe 见上）；另登记描述性诊断（Spearman、GC 方向性 directedness、|conn|-距离相关已有随档）。
- R3 产物落 `outputs/10-21-latent-flow-gc-corr-f2nosc/run_01/`（GC 矩阵 npy+json meta、相关性 json）。
- R4 零仓库代码改动。

## Acceptance Criteria

- [ ] AC1 GC 128×128 矩阵落盘且非负、对角为 0、确定性可复跑
- [ ] AC2 r_abs / r_signed 数值登记；描述性诊断随档
- [ ] AC3 零仓库代码改动；单被试口径不外推（n=1，无统计效力声明）

## 边界与不变式

- 单被试描述性存证，不做 10 例对标、不做 fold 间对比；不构成代理脑资格。
- GC 矩阵不写入 refs/（refs 归档口径为 edra 48/46 例档；如需 MODMA 档归档另拍板）。

## 子研究路由（1.0b / 1.0c）

- **分类结果**：`study1`（latent-flow Study I 候选线的读出对照）
- **依据**：global-tau σ\* 归因属 latent-flow 候选线组成部分，本任务为其单被试 GC 对照。
- **新路径/方法检测（1.0c）**：`已有路径：latent-flow + 原库谱 GC 主路径（均已登记）`
- **research 产物**：`无`
