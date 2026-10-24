# npi/refs — baseline 常用参考结果归档

> 目的：将已计算完成的参考结果（baseline refs）集中归档，供跨任务复用。
> 组织规则（2026-10-20 建档约定）：**`refs/<方法名>/<参数配置>/` 独立子目录**，不同方法/不同参数的结果分目录存放、不混用、不覆盖；保留原文件名；每档必含 `meta.json`（来源路径 + 方法/参数 + 文件清单 + 校验和文件）。
> 参考为只读：新增参考一律另建子目录并在本清单追加一行。

## granger_bivariate_spectral（latent_flow 原库 GC）

| 子目录 | 数据 | 关键参数 | 来源 | 归档日 |
|---|---|---|---|---|
| `edrafmt48-cont-zscore-fs250-fixed8-f1-125-nf65/` | edra_processed_fmt 48 例（HC 24/MDD 24）全长连续 60001 样本/例，逐通道 z-score | 固定阶 8、n_freq 65、1–125 Hz、fs 250、梯形积分（Geweke 1982 / Dhamala 2008） | `outputs/10-19-latent-flow-edra-fmt-train/run_01/gc_oldlib_refs`（任务 10-19-latent-flow-edra-fmt-train，2026-10-19 生成） | 2026-10-20 |
| `edrav4-46-resample250-cont-zscore-fs250-fixed8-f1-125-nf65/` | edra_process_v4 46 例（HC 24/MDD 22）全长连续，`resample_poly(1,2)` 500→250 Hz（85000→42500 样本/例），重采样后逐通道 z-score；不做 qc_pass 筛选（qc_pass=True 仅 13/46 事实随档） | 同上（固定阶 8、n_freq 65、1–125 Hz、fs 250、梯形积分）——与 fmt 档唯一差异 = 数据源 + 重采样步 | `outputs/10-21-oldlib-gc-edrav4/refs-01`（任务 10-21-oldlib-gc-edrav4，2026-10-21 生成） | 2026-10-21 |

- 内容：48 × `refs_s{id}.npz`（`ec` float32 (62,62)，**ec[a,b]=a→b 行=源/列=靶**，对角 0）+ `gen_refs.py`/`gen_refs.log`（生成溯源）+ `SHA256SUMS` + `meta.json`。
- 用途：原库 GC 登记（用户 2026-10-20 拍板）为 **GC 计算主路径**，本档为其在 fmt 数据上的现成基线参考；与 A_eff（行=靶/列=源）对照的方向对齐主口径 = |A_eff| vs `ec.T`。
- edrav4 档内容：46 × `refs_s{id}.npz`（格式同上）+ `gen_refs.py`/`gen_refs.log` + `compare_fmt.py`/`compare_fmt_summary.json`（与 fmt 档对照登记）+ `SHA256SUMS` + `meta.json`。
- edrav4 档用途与边界：同实现同参数 × v4 原管线数据口径的基线参考；**与 fmt 档重叠 45/46**（v4 MDD s56 为 fmt 所无），下游两档对照一律取 45 例交集；两档口径隔离**不做对比交付**（对照文件仅描述性登记：offdiag Spearman ρ=0.2311±0.1026、median 比 v4/fmt=1.651±0.664）。

## granger_bivariate_spectral（mne-connectivity 谱 GC）

| 子目录 | 数据 | 关键参数 | 来源 | 归档日 |
|---|---|---|---|---|
| `mild40-trial500-zscore-fs250-fixed8-f1-45-favg/` | Mild Depression EEG_ALL 40 例（MDD 20/HC 20，128ch Cz 参考），每 trial 前 500 点（2 s）× 30 trials 池化，逐 trial 逐通道 z-score（原数据未归一） | `gc_n_lags=8`（VAR 阶数固定）、fmin 1 / fmax 45 Hz + faverage 单频段均值、fs 250；状态空间谱 GC（Barnett & Seth 2015，mne-connectivity 0.9.0） | 用户 2026-10-24 提供 zip `/root/autodl-tmp/mild-depression-gc.zip`（注册任务 10-24-mild-w500-ec-unbiased S0；源 README 归档为 README-source.md） | 2026-10-24 |
| `mild40-trial500-zscore-fs250-aic4-25-f1-125-favg/` | 同上（同 EEG_ALL 数据、同 trial 前 500 点/逐 trial z-score/30 trials 池化口径） | **AIC 逐被试选阶（4–25，池化二元 VAR 完全模型 OLS-AIC，统一 500−25 截断）**、fmin 1 / fmax 125 Hz + faverage、fs 250；状态空间谱 GC（mne-connectivity） | 用户 2026-10-24 提供 zip `/root/autodl-tmp/EC_GC_1_125.zip`（注册任务 10-24-exitgnn-ec-unbiased S0/Run 000；源 README 归档为 README-source.md） | 2026-10-24 |

- 内容：40 × `refs_s{id}.npz`（`ec` float32 (128,128)，**ec[a,b]=a→b 行=源/列=靶**，对角 0，理论非负实测 min=0）+ `register_refs.py`/`register.log`（注册溯源，可重放）+ `SHA256SUMS` + `meta.json` + `README-source.md`。
- subject 映射：sid = EEG_ALL index + 1（前 20 MDD → sid 1–20，后 20 HC → sid 21–40）；四锚点（m_1/m_20/h_1/h_20）与 `models/latent-flow-v1/splits/mild_5fold_seed42.json` 文件 **bitwise 相等**已验（记录在 meta.json）。
- 用途与边界：latent-flow mild-w500 代理脑 EC_pert 的观测参考（对照协议对齐 unbiased 先例：|EC|/raw × aligned/flipped × Spearman/Pearson，描述性登记不构成代理脑资格判定）；与 latent_flow 原库 GC 两档**实现不同、口径隔离**，不做跨实现/跨数据集对比交付；GC 依赖 Cz 参考（Trongnetrpunya et al. 2016）等局限随源 README 在档。
- aic4-25 档（v2）内容：格式同 v1 档（40 npz + 注册溯源四件套）；**四锚点 bitwise 复验由注册脚本重做**（非沿用 v1 规划期验证，结果在 meta.json）；sanity 与源 README QC 对拍通过（mean=0.0475 / max=0.568）。
- aic4-25 档（v2）用途与边界：latent-flow mild 线代理脑 EC_pert 的第二观测参考（与 v1 档构成**双 GC 对照**，各自独立登记不做优劣判断）；caliber note（源 README 原文）：数据有效带宽 ≈0.5–50 Hz，51–125 Hz 已衰减 −24 dB 以下 ⇒ 1–125 Hz 积分主要由 <50 Hz 贡献；与 v1 档逐被试矩阵 Pearson median=0.148（结构差异大，双对照正当）；evoked 瞬态非平稳风险等局限随源 README 在档。
