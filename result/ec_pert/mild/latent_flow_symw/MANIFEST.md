# result/ec_pert/ — EC_pert 结果与 EC×GC 相关性注册清单（MANIFEST）

> **注册**：2026-10-30（任务 `.trellis/tasks/10-29-result-ckpt-reg/`，第二轮追加）。
> **写入许可**：用户追加指令点名注册入 result（result-dir-policy 情形 1；两轮指令原文与拍板见任务 `consent.md`）。
> **逐文件 sha256 / size / 原始路径**：`sha256.json`（同目录机器可读副档，66 文件全量，复制后 `cmp` 逐位一致）。
> **维护约定**：存入后不就地修改；更新 = 新版本并排 + 登记。

## 注册对象

mild · latent-flow symw（`10-25-md-lf-anti0.2-gnnexit-symw`，即 ckpt 清单（`result/checkpoints/`）同款五折双全模型）五折 EC_pert 结果 + 逐被试/组平均 EC×GC 相关性。共 **66 文件，123,934,584 B（118.2 MiB）**。

## 版本口径（重要）

- **EC_pert conn 公式统一 v4 = −(log σ\* + 0.5·ln2πe) + 5/7**——每折逐被试 json 内 `connectivity_formula` 字段断言一致（fold0/fold3 抽验 + manifest 生成时程序断言）。
- **fold0/1/2：仅注册 recompute_v4 版本**（用户 2026-10-30 追加指令）。被试 json 取各任务 `recompute_v4/` 目录（v4 公式重算版）；其原始 `ec_main/` v1 公式版（差常数平移 +1.1332，秩不变）、`*_curves.npz`、v1 口径 `summary/` 组汇总**均未注册**。
- **fold3/4：v4 直出原版**（`ec_main/` + `summary/` + `gccorr/`，10-28 任务产物，本就 v4）。
- EC 扰动口径（五折逐字同轨）：无偏 mu=0 高斯 σ\*ξ 扰动（z-score 后入模前源通道）、global-tau q=0.5 @ σ_ref=0.3、σ 网格 log[0.01,0.48]×24、seed 42、bf16 autocast、每折 test 8 例 × 30 trial。
- GC 对照：v1 refs（mne-connectivity 谱 GC，VAR 固定阶 8，1–45 Hz faverage，`refs/granger_bivariate_spectral/mild40-trial500-zscore-fs250-fixed8-f1-45-favg`）。fold0/1/2 的 `gccorr_dual_summary.json` 内含历史 v2 refs 块（该档已于 10-27 注销，仅作在档记录，正式读数取 v1 块）。
- 描述性并置登记（先例协议），不构成代理脑资格判定。

## 目录结构与来源

| 折 | 注册内容 | 来源任务 · 原始路径 |
|---|---|---|
| fold0 | `ec_main/subject_{0..7}.json`（v4 公式，含 σ\* 矩阵/conn 矩阵/τ/诊断）+ `ec_gc_corr/gccorr_dual_summary.json`（逐被试+组平均 EC×GC） | `outputs/10-25-symw-ec-unbiased/recompute_v4/` |
| fold1 | 同上结构 | `outputs/10-25-symw-ec-f12/fold1/recompute_v4/` |
| fold2 | 同上结构 | `outputs/10-25-symw-ec-f12/fold2/recompute_v4/` |
| fold3 | `ec_main/subject_{0..7}.json` + `subject_{0..7}_curves.npz`（原始扰动响应曲线，公式无关）+ `summary/{group_mean.npz, group_summary.json}`（组平均 EC 与方差分解）+ `ec_gc_corr/gccorr_summary.json` | `outputs/10-28-symw-ec-f34/fold3/{ec_main,summary,gccorr}/` |
| fold4 | 同 fold3 结构 | `outputs/10-28-symw-ec-f34/fold4/{ec_main,summary,gccorr}/` |
| 跨折 | `crossfold/fivefold_readme_10-28-symw-ec-f34.md`（五折并置表；f012 相关系数引 recompute_v4 读数，秩量纲版本不变） | `outputs/10-28-symw-ec-f34/README.md` |

来源任务目录：`10-25-symw-ec-unbiased`、`10-25-symw-ec-f12`、`10-28-symw-ec-f34`（`.trellis/tasks/archive/2026-10/` 或在档）。

## 关键数字（EC_pert × GC，v1 refs，offdiag 16256 对，aligned=行=源）

| 折 | 逐被试 Spearman（mean±std [min,max]，8 例） | 逐被试 Pearson | 组平均矩阵 Spearman / Pearson |
|---|---|---|---|
| fold0 | **+0.1344±0.1587** [−0.135, +0.330] | +0.1723±0.1787 | +0.1806 / +0.2338 |
| fold1 | **+0.1590±0.1164** [−0.086, +0.283] | +0.2102±0.1015 | +0.3805 / +0.3759 |
| fold2 | **+0.1757±0.1269** [−0.013, +0.310] | +0.1854±0.1148 | +0.2675 / +0.2494 |
| fold3 | **+0.2124±0.1073** [+0.092, +0.353]（8/8 全正） | +0.1881±0.0651 | +0.3503 / +0.3418 |
| fold4 | **+0.1822±0.0928** [+0.012, +0.307]（8/8 全正） | +0.1618±0.0817 | +0.2541 / +0.2134 |

- 「逐被试」= 每被试 128×128 offdiag EC vs GC 向量相关后在 8 例上汇总（mean±std）；「组平均矩阵」= 先对 8 例 EC 与 GC 各取组均值矩阵再相关。
- 相关文件字段名：f012 `gccorr_dual_summary.json → refs_blocks.v1_1_45_fixed8.{per_subject, group, group_mean}`；f34 `gccorr_summary.json → {per_subject, group, group_mean}`（entropy 即 v4 形式）。

## 明确未注册（口径原因）

- fold0/1/2 原始 `ec_main/subject_*.json`（v1 公式）、`*_curves.npz`、`summary/group_summary.json` + `group_mean.npz`（v1 口径组汇总）——按用户指令仅取 recompute_v4；
- `10-25-symw-ec-f12/crossfold/crossfold_summary.json`——混编 v1 口径 group summary 与已注销 v2 refs，非纯 recompute_v4 产物；
- fold0 `clamp030_gccorr.json`（探索性钳边对照）、`symmetry_diag/`、`dryrun*/`、`scratch_equiv/`、各 `*.log`、fold3/4 `viz_ec/` 图件——非本轮指令范围。
