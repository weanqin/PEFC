# edra_processed_fmt（新预处理范式）数据集事实卡

> 来源：`/root/autodl-tmp/edra_processed_fmt/README.md`（2026-10-16 转换时写入，本卡为任务内只读摘录备份，防外部 README 变动丢失口径）；转换脚本 `outputs/edra-paradigm-compare/convert_to_v4fmt.py`；身份审计 `outputs/edra-paradigm-compare/identity_audit.json`。

## 基本事实

- 源：`/root/autodl-tmp/EDRA_processed/`（release 自带预处理：1–40 Hz 带通 @250 Hz，48 例，每例 `data (62, 60001) float32`，无其他字段）
- **数据值零改动**：未重采样、未滤波、未去伪迹；250 Hz 原样保留；仅转 v4 dataloader 契约格式（`models/gpfm-mamba-v1/data.py` 零修改加载，冒烟已过：`X_past (B,62,384)` / `X_future (B,62,128)` float32）
- 布局：`HC/ 24 例`、`MDD/ 24 例`（文件名 `{sid}_rest_v4.npz` 为 dataloader 硬编码模式，**内容溯源看 npz `pipeline` 字段 = `EDRA_release_1-40_250Hz`，勿与 v4 管线混淆**）；`manifest.csv` 48 行全 status=ok；`chan_coords_62.npy` / `chan_forward_62.fif` 与 v4 同源共用
- manifest 缺失字段（sex/age/promis、坏导联、QC 计数、参考链）一律留空不造数 → **`subject_filter='qc_pass'` 口径在本数据集不可用**，只有 status_ok（48 例全量）

## 与旧范式 edra_process_v4 的差异清单（跨范式对比必随附）

| 维度 | edra_processed_fmt（新） | edra_process_v4（旧） |
|---|---|---|
| 预处理 | release 1–40 Hz 带通，**无 ICA/EOG/EMG 去伪迹、无 QC** | bp1-40+notch50 + ICA 等去伪迹 + QC 门禁 |
| 采样率 | **250 Hz**（Nyquist 125 Hz） | 500 Hz |
| 例数 | 48（HC24/MDD24） | 46（HC24/MDD22） |
| 每例点数 | 60001 | 85000 |
| 窗数/例（512 窗、stride 64） | 930 | 1322 |
| 窗口墙钟时长（512 采样点） | 2.048 s | 1.024 s |

> 勘注（2026-10-16 check 复核）：上表「窗数/例 1322」为外部 README 差异表原值，**实测 dataloader 口径 = 1321 窗/例**（(85000−512)//64+1 = 1321；旧 run 落盘复核：v1 main-01 44,914/34、v4a r16-main-01 42,272/32 均 = 1321/例）——README 值差一，以 1321 为准。

## 身份审计结论（identity_audit.json）

release .mat 与 v4 npz 为**同源录制的不同预处理/裁剪版本**：48 例 release 文件与 v4 逐例对齐（MDD_11/21/48 三例 v4 无 npz——v4 管线剔除，release 侧保留）；互相关分数 0.14–0.44、滞后 ~1e4 采样点（500 Hz 基准的裁剪偏移，属预期）。

## 训练接入注意（README 原文要点）

- v1 config 把 `data_root` 指向本目录即可；**勿复用 v1/v4a 已固化 splits manifest**（48≠46 例，`fixate_split` 防漂移检查按设计报错）——新 variant 目录首跑固化自己的 fold0 manifest
- 250 Hz vs 500 Hz：窗口按样本数计（512）墙钟时长不同，跨范式对比口径差异须显式拍板登记（本任务拍板：窗口保持样本数、kernel `period_p` 50→25 保物理时长、生成侧不滤 40 Hz）

## 250 Hz 连带口径（本任务新增拍板 D2/D3 的依据）

- 真实侧 1–40 Hz 硬带限（release 预处理）→ >40 Hz 无物理内容；生成侧不加低通可直接观察模型是否自行待在带内（旧范式 40 Hz 低通口径系针对 OU 谱尾+网络高频注入问题，新范式下先观察原始谱再定）
- `gpr_prior.py:94`：`tau_i = i·π/p` 为采样点网格 → fs 减半时 p 同步减半（50→25）保持先验物理时长（~0.1 s 特征尺度）不变
