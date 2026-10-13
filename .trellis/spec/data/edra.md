# 数据集契约 — edra（静息态 EEG · HC/MDD）

> 实测日期：2026-10-12。目录结构、文件数、npz schema、manifest 表头已核对（抽样 HC/13_rest_v4.npz 与 manifest 前两行）；标注「待验证」的条目尚未核对，逐项补测后去除标记。

## 1. 来源与版本

- 路径：`/root/autodl-tmp/edra_process_v4/`；版本标识 **v4**（文件名统一后缀 `_rest_v4`）
- 规模：**HC 24 例、MDD 22 例**，每受试者一个 npz（`HC/`、`MDD/` 目录各 24/22 个 `{subject_id}_rest_v4.npz`）
- 分组目录与文件内 `group` 字段一致性：抽查一致（待全量核对）

## 2. 组织与字段

- 每受试者一个 npz，核心数组：`data (62, 85000) float32`（62 通道 × 85000 采样点）、`channel_names (62,)`、`eog (85000,)`
- 元数据标量字段：`subject_id / group / sex / age / promis / fs / n_samples_raw / crop_start_sample / crop_len_samples / t_center_s`
- QC 与管线字段：`bad_channels / n_eog_removed / n_emg_removed / n_ica_removed / eog_ic_idx / emg_ic_idx / qc_pass / pipeline / ref_chain / filter / truncated`
- **采样率 500 Hz**（manifest 时长反推核证：93959 / 187.918 ≈ 500）；裁剪后每例统一 85000 点（=170 s；待全量核对是否所有受试者一致）
- 分组唯一依据：npz 内 `group` 字段（manifest.csv `group` 列同源）；目录位置仅作初筛，不作最终依据
- **montage / 通道空间信息**（数据集根目录）：`chan_coords_62.npy`（62 通道坐标）、`chan_forward_62.fif`；通道名以 npz `channel_names` 为准
- `manifest.csv`：受试者级汇总（subject_id, group, sex, age, promis, n_samples_raw, duration_s, n_bad_channels, bad_channels, clean_frac, qc_pass, truncated, notes, status, n_eog_removed, n_emg_removed, crop_start_sample, crop_len_samples, ref_chain 等）
- `qc_report.md` 与 `_qc/`（每受试者 `{id}.json`、`{id}_badsig.npy`、`{id}_icamix.npy`）：QC 明细

## 3. 预处理契约

- v4 管线已完成的处理随文件自带（`pipeline / ref_chain / filter` 及 ICA/EOG 去除计数等字段记录；具体参数以 `_qc/` 与 manifest 为准——待核对汇总）；**filter 字段实测（2026-10-13，受试者 1 抽样，Run 012）= `bp1-40+notch50(保真冗余)`**——1–40 Hz 带通 + 50 Hz 陷波；全量谱实测 >40 Hz 方差占比 ≈0.8%、>100 Hz ≈2e-9（数值地板）。**含义：下游任何谱类指标中，真实侧 >40 Hz 是滤波硬截止，不是物理内容；与生成侧比较时 log 谱在高频段被地板效应放大**（证据：任务 10-12 Run 012 诊断）
- **下游时间窗统一切分为 `[past, future]`：past 长度 384、future 长度 128（采样点）**；切分在 dataloader 内完成（见 [../research/data-pipeline.md](../research/data-pipeline.md)）
- 下游使用前核对 `ref_chain / filter` 字段，避免与 dataloader 内处理重复或冲突

## 4. 已知坑（实测沉淀）

- 暂无；后续实测发现逐条追加（每条附证据：哪次实测、什么现象、正确做法）。

## 5. 下游消费规则

- 受试者筛选允许按 manifest `qc_pass` / `status` 过滤；过滤口径须在 experiment.md 登记
- 5-fold 分层划分（MDD/HC、受试者级）按 [../research/data-pipeline.md](../research/data-pipeline.md) 执行
- 变更本契约任何口径 → 逐项排查对下游的影响并告知用户后方可生效
