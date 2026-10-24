# EEG 格兰杰因果有效连接矩阵（subject 级）

## 数据
- `dataset/mild depression/EEG_ALL.mat`，变量 `EEG` [1200, 128, 1500] float64，
  40 被试 × 30 trials × 128 通道 × 1500 点（250 Hz）；**前 20 被试 MDD，后 20 HC**
- 每 trial 取前 500 点（2 s）
- 原始数据检查结论（2026-10-22）：未做 z-score（已在本流程逐 trial 逐通道补做）；
  有效带宽 ≈ 0.5–50 Hz（低通 ≈50 Hz，非 40/45）；Cz 参考（非平均参考；数值排除平均参考，
  同系文献 Li et al. 2020 Front Neurosci 14:192 记载 "referenced to Cz"）

## 方法（全部 GC 数值由 mne-connectivity 计算，未自实现 GC 算法）
- `mne_connectivity.spectral_connectivity_epochs(method='gc')`：
  状态空间谱 Granger 因果（Barnett & Seth 2015），multitaper CSD 跨 30 trials 平均后
  经自协方差序列拟合 VAR（即 trial 池化）
- 参数：**gc_n_lags = 8（VAR 阶数固定）**，fmin=1 Hz，fmax=45 Hz，faverage=True
  （单频段均值；数据实测低通 ≈50 Hz，45 Hz 上界避开 50 Hz 滤波沿与工频残余）
- 预处理：每 trial 每通道 z-score（数据提供方未做）；无额外滤波/重参考
- 每被试 16,256 个有序通道对；mat[i,j] = GC(i→j)；对角线 0；理论非负
  （实测 0 个负值截断）
- 软件：Python 3.13（conda env `npi`），mne 1.11.0，mne-connectivity 0.9.0

## 文件
| 文件 | 说明 |
|---|---|
| `EC_GC_subject.npy` | [40, 128, 128] float64，第 s 片 = 被试 s 的 GC 矩阵 |
| `EC_GC_subject.mat` | 同上 + labels（MDD/HC）+ 参数元数据 |
| `gc_strength_per_subject.npy` | [40] 每被试平均 GC 强度 |
| `AIC_p8_per_subject.csv` | AIC(p=8) QC 诊断 见 `AIC_p8_per_subject.csv` |
| `interim/gc_subj_XX.npy` | 逐被试中间结果（断点续跑） |
| `qc/*.png` | QC 图（矩阵热图、组间强度对比、PSD 带宽） |
| `../..`/scripts/gc_ec/*.py | 计算脚本 |

## QC 摘要
- 全局 GC 强度（探索性，不作结论）：MDD=0.0152,
  HC=0.0139, t=1.13, p=0.264
- AIC(p=8) 为纯诊断（池化二元 VAR(8) 完全模型 OLS，取 16256 对平均），不参与 GC 数值

## 已知局限
- pairwise（二元）GC 无法区分直接/间接连接，共同输入可产生虚假双向因果
- 2 s 短窗对 GC 有偏倚风险（trial 池化 30×492 点可缓解；Wang, Chen & Ding 2008）
- GC 依赖参考电极（本数据为 Cz 参考；Trongnetrpunya et al. 2016）
- 后续稳健性对照可用 mne `gc_tr`（时间反转 GC）或 surrogate 检验

## 参考
Geweke 1982; Ding et al. 2000; Barnett & Seth 2015 (state-space GC);
Seth, Barrett & Barnett 2015; mne-connectivity 文档;
Li et al. 2020 Front Neurosci 14:192（同系数据预处理描述）
