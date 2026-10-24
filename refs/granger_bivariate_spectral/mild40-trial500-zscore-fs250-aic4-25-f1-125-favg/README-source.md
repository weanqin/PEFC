# EEG 格兰杰因果有效连接矩阵（subject 级）— v2

## 配置
- 频段积分：**1-125 Hz**（faverage 单频段均值）
- 阶数：AIC-selected per subject (4-25) — 逐被试 AIC 选阶（4–25，池化二元 VAR 完全模型 OLS-AIC，统一 500−25 截断；分布见 qc/gc_strength_group.png 右图与 qc/aic_curves.png；各被试 AIC 表在 interim/aic_subj_XX.npy）
- 其余同 v1：每 trial 前 500 点（2 s）、逐 trial 逐通道 z-score（原始数据未做）、
  30 trials 池化 multitaper CSD、mne-connectivity 状态空间谱 GC、仅 subject 级

## 数据
- `dataset/mild depression/EEG_ALL.mat`，变量 `EEG` [1200, 128, 1500] float64，
  40 被试 × 30 trials × 128 通道；**前 20 MDD，后 20 HC**
- 原始数据检查（2026-10-22）：有效带宽 ≈ 0.5–50 Hz（低通 ≈50 Hz），
  Cz 参考，未 z-score（本流程已补）；注意 51–125 Hz 频段在数据中已衰减
  −24 dB 以下，1–125 Hz 积分值主要由 50 Hz 以下贡献
- 任务态 evoked 瞬态的非平稳风险与建议的稳健性检查见
  `../EC_GC/task_eeg_gc_survey.md`（Wang/Chen/Ding 2008 警示），用户决定沿用原窗

## 文件
| 文件 | 说明 |
|---|---|
| `EC_GC_subject.npy` | [40, 128, 128] float64，mat[i,j] = GC(i→j) |
| `EC_GC_subject.mat` | 同上 + labels（MDD/HC）+ 参数元数据 |
| `gc_strength_per_subject.npy` | [40] 每被试平均 GC 强度 |
| `interim/gc_subj_XX.npy` | 逐被试矩阵（断点续跑） |
| `interim/aic_subj_XX.npy` | 逐被试 AIC 曲线与选中阶数 |
| `qc/*.png` | 矩阵热图 / 组间强度 / 选阶分布 / AIC 曲线 |

## QC 摘要
- sanity：对角线全 0、非负（0 截断）、无 NaN；mean=0.0475，max=0.568
- 全局强度（探索性）：MDD=0.0501, HC=0.0457,
  t=0.74, p=0.462

## 与 v1（1–45 Hz, p=8）结果对比
- 逐被试矩阵 Pearson 相关：median=0.148, min=-0.024, max=0.672（n=40）

## 局限
同 v1 README（pairwise 混淆、短窗任务态 evoked 风险、Cz 参考依赖）；可用 gc_tr 稳健性对照。

## 参考
Geweke 1982; Ding et al. 2000; Barnett & Seth 2015; mne-connectivity 文档;
Wang, Chen & Ding 2008（任务态警示，见 survey）
