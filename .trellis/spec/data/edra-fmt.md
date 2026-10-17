# 数据集契约 — edra-fmt（静息态 EEG · HC/MDD · 新预处理范式）

> 实测日期：2026-10-16（转换）/ 2026-10-17（训练接入实测，任务 10-16-gpfm-mamba-fmt-f0）。「待验证」条目逐项补测后去除。

## 1. 来源与版本

- 路径：`/root/autodl-tmp/edra_processed_fmt/`；版本标识 **fmt**（文件名沿用 `_rest_v4` 后缀系 dataloader 硬编码模式，**内容溯源看 npz `pipeline` 字段 = `EDRA_release_1-40_250Hz`**）
- 源：`/root/autodl-tmp/EDRA_processed/`（release 自带预处理），转换脚本 `outputs/edra-paradigm-compare/convert_to_v4fmt.py`；**数据值零改动**（未重采样/滤波/去伪迹），仅转 v4 dataloader 契约格式
- 同源身份：与 edra（v4 管线）为同源录制不同预处理（identity_audit.json 互相关 0.14–0.44、滞后 ~1e4 采样点）；release 48 例中 MDD 3 例（11/21/48）为 v4 管线剔除
- 规模：**HC 24 / MDD 24 共 48 例**；manifest 48 行全 status=ok

## 2. 字段与组织

- 每例 npz：`data (62, 60001) float32`（dataloader 唯一消费键）、`channel_names (62,)`、`fs=250`、`subject_id/group/suffix`、`pipeline/filter`（溯源标记）、`n_samples/truncated`
- manifest 缺失字段（sex/age/promis、坏导联、QC 计数、参考链）留空不造数 ⇒ **`subject_filter='qc_pass'` 不可用**，仅 status_ok
- montage：`chan_coords_62.npy` / `chan_forward_62.fif` 与 edra 共用（62 导联同序）

## 3. 预处理与采样率契约（与 edra 的关键差异）

- **250 Hz**（edra 500 Hz）：Nyquist 125 Hz；release 1–40 Hz 带通（无 ICA/EOG/EMG 去伪迹、无 QC 门禁；edra 为 bp1-40+notch50+ICA+QC）
- 时间窗仍按样本数 `[past=384, future=128]`、stride 64（README 契约）⇒ 每例 **930 窗**（(60001−512)//64+1）；**墙钟时长 2.048 s（edra 为 1.024 s）**——跨范式对比必随附
- 下游 fs 耦合参数（2026-10-16 用户拍板，任务 10-16-gpfm-mamba-fmt-f0）：GPR 时间核 `period_p` 50→**25**（保物理时长）；config `fs: 250` 经 getattr 透传 PSD/轨迹图

## 4. 已知坑（实测沉淀）

- **勿复用 edra 的固化 splits manifest**：48≠46 例，`fixate_split` 防漂移检查按设计报错——新 variant 目录首跑固化自己的（实测 fold0 内层 **34/4/10**，test [10,11,12,17,24,28,30,44,61,69]）
- 250 Hz 下重开生成侧低通**必须传 fs=250**（`brickwall_lowpass` 默认 fs=500）；本范式现行口径为**不滤波**（gen_lowpass_hz=None，观察模型是否自行待在带内）
- `diag_hf.py` / `gc_psd.py` 独立 main() 硬编码 fs=500 与旧 run id/旧根——fmt run 一律经变体 wrapper 的 evaluate.py 入口
- 外部 README 差异表「1322 窗/例」系 edra 侧笔误（实测 1321），以本契约为准

## 5. 下游消费规则

- dataloader 零修改加载（`models/gpfm-mamba-v1/data.py` 契约兼容，2026-10-16 冒烟通过）
- 变体目录：`models/gpfm-mamba-v1-fmt/`、`models/gpfm-mamba-v4a-fmt/`（LayerNorm/p=25/无低通口径）；产物根 `/root/npi/outputs/gpfm-mamba-fmt-f0/`
- 变更本契约任何口径 → 逐项排查下游影响并告知用户后方可生效
