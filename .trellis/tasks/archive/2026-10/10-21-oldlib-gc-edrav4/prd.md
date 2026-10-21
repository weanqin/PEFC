# 旧库谱 GC × edra_process_v4 逐被试 EC（重采样 250 Hz 口径）

## Goal

用旧库 latent-flow 成对双变量谱 GC（`latent_flow/baseline/granger.py`，Geweke 1982 / Dhamala 2008）对 `/root/autodl-tmp/edra_process_v4/` 全部 46 例（HC 24 / MDD 22）逐被试计算 EC 矩阵。fs 口径经用户拍板（2026-10-21）：**v4 500 Hz 原生数据先重采样到 250 Hz**，再完全复用既有 refs 口径（fixed_order=8 / 1–125 Hz / nf65 / 全长连续 / 逐通道 z-score），与 `refs/granger_bivariate_spectral/edrafmt48-cont-zscore-fs250-fixed8-f1-125-nf65/` 档直接可比。

## Background

- GC 计算主路径已拍板（2026-10-20）= 原库 GC（`granger.py`）；既有归档 `gen_refs.py`（任务 10-19 Run 005，edra_fmt 48 例，24.2 min @ 12 进程）为脚本蓝本，本任务仅换数据源 + 新增重采样步骤。
- v4 与 fmt 关系：不同管线/时长的两份数据（v4 = bp1-40+notch50 @500 Hz、170 s/例；fmt = EDRA_release_1-40_250Hz、240 s/例）；通道名序经实测逐位一致（2026-10-21，HC/4 与 HC/13 抽样）→ offdiag 逐边对照可比。
- **不做**对比交付：与 fmt 档的相关统计仅作质量对照登记（描述性），管线差异分析不在本任务范围。

## Requirements

### R1 数据口径（v4 → 重采样 250 Hz）

- 输入 `/root/autodl-tmp/edra_process_v4/{HC,MDD}/*.npz` 全量 **46 例**（HC 24 + MDD 22；manifest status=ok 46 例与 npz 目录一一对应），数据集只读。
- 逐例读 `data (62, 85000) float32`；断言 `fs==500`、shape (62, 85000)、npz `group` == 目录名、npz `subject_id` == 文件名 id；断言 `channel_names` 与 fmt 档参考序逐位一致（启动时读一例 fmt npz 作参考序；顺带全量补验 edra.md §1「待验证」的 shape/group 一致性）。
- **重采样**：`scipy.signal.resample_poly(x, 1, 2, axis=-1)` → 42500 点 @250 Hz（唯一新增预处理步骤，确定性无随机源；85000 整除 2 无尾差）。
- z-score 逐通道在重采样后进行（ddof=0，σ==0 置 1，同旧库 `load_subject_timeseries` 惯例）；float64 进入 GC 计算、float32 落盘。
- QC 事实登记：manifest 中 qc_pass=True 仅 13/46（33 例 gate_fail:s7_interp_carry 仍以 status=ok 发布）；按用户「数据集原样」口径 **46 例全算**，不做 qc_pass 筛选。

### R2 GC 口径（与 edrafmt48 档逐项一致，仅数据源不同）

- `granger_connectivity(data, fs=250.0, max_order=8, n_freq=65, fmin=1.0, fmax=125.0, fixed_order=8, n_jobs=12)`。
- 方向约定：**ec[a,b] = a→b（行=源，列=靶）**，对角=0；offdiag≥0 为实现正确性断言（GC 为方差 log-ratio，理论非负）。
- 每例全长单段连续（不切窗）；与 fmt 档唯一差异 = 数据源（重采样 v4，42500 点 < fmt 60001 点）。

### R3 与 edrafmt48 档对照（描述性登记，非硬门限）

- 重叠被试 **45/46**（HC 24/24 + MDD 21/22；trellis-review ID 集合实测：v4 MDD 含 **s56** 而 fmt 48 例无 56，fmt 独有 MDD {11, 21, 48} 为 v4 弃用例）。s56 处理 = GC 本体照算（R1 46 例全算不变），仅 fmt 对照跳过 s56 并登记。
- 逐重叠被试计算 offdiag 3782 边 Spearman ρ(GC_v4res, GC_fmt) 与幅值对照（offdiag median 比值）→ 逐例 + 组级 mean±sd 落 summary，如实登记，不设阈值。
- 预期非 ≈1：两档数据时长/管线不同（170 s vs 240 s），相关仅作口径落点合理性核对。

### R4 范围边界

- 仅真实数据：无生成侧、无 θ\*、无模型依赖。
- 零写入：v4 数据集、`refs/` 既有档、`outputs/` 既有产物一律只读；本任务产物只落 `outputs/10-21-oldlib-gc-edrav4/`。
- 不做 split-half 天花板 / 时间反转指纹（edrafmt48 档先例亦不含；如需另立任务）。

### R5 运行纪律

- Run 序列：`smoke-01`（1 例，**须选 fmt 重叠被试**如 HC/4，勿抽 s56）→ `refs-01`（46 例主 Run）；experiment.md 跑前写目的+配置、跑完同会话回填结果+结论。
- BLAS 护栏五变量（OMP/OPENBLAS/MKL/NUMEXPR/VECLIB_MAXIMUM）**setdefault 全 "1"**（numpy 导入前；随蓝本配置，12 loky workers × 1 BLAS 线程在 16 核 cgroup 内不超订，24.2 min 先例即该配置）；joblib loky n_jobs=12；断点续跑（已存在 npz 跳过）。
- 脚本工作副本落 `outputs/10-21-oldlib-gc-edrav4/gen_refs.py`（归档时复制入 refs 档）。
- 预计时长 ~16–25 min（42500 点 < fmt 60001 点，单例耗时 ≈0.71× fmt 档）。

### R6 refs 归档

- 归档目录：`refs/granger_bivariate_spectral/edrav4-46-resample250-cont-zscore-fs250-fixed8-f1-125-nf65/`（新子目录，不覆盖任何既有档）。
- 随档：46 × `refs_s{id}.npz` + `meta.json`（方法/口径/数据来源/subject 清单/alignment_convention，格式对齐 edrafmt48 档）+ `gen_refs.py` + `gen_refs.log` + 对照 summary + `SHA256SUMS`。

## Acceptance Criteria

- [ ] AC1 `smoke-01`（1 例）通过：重采样长度 42500 断言、z-score 后 mean≈0/std≈1 断言、EC shape (62,62) 且 offdiag≥0、与 edrafmt48 同被试量级对照核对（同数量级即可）
- [ ] AC2 `refs-01`：46/46 例 npz 落盘（ec float32 (62,62) 行=源/对角 0 + subject_id/group/params 齐全），非负断言全过；summary 含逐例耗时与 offdiag min/median/max
- [ ] AC3 对照登记：45 例重叠被试（s56 跳过并登记）逐例 offdiag Spearman ρ + 组级 mean±sd + 幅值比值落 summary，如实登记
- [ ] AC4 确定性：任抽 1 例删除后重跑，输出与首次逐位一致（SHA256 对比）
- [ ] AC5 归档完整：R6 目录齐全 + meta.json 格式对齐既有档 + SHA256SUMS 与任务 outputs 逐位一致；既有 refs 档与 v4 数据集零写入
- [ ] AC6 experiment.md Run 块完整（目的/配置/结果/结论，无空块）；`evaluation-calibers.md` **新增完整条目**「原库谱 GC × edra_v4 重采样口径」（实现指针 granger.py::granger_connectivity / fixed_order=8 / 1–125 Hz / nf65 / fs250 / resample_poly(1,2) 步骤 / 行=源约定 / 不做对比交付边界；2026-10-20 主路径拍板现状仅在 context.md 与 refs meta，evaluation-calibers.md 无既有条目可并列）；journal 记录本会话

## 子研究路由（1.0b / 1.0c）

- **分类结果**：`study2` —— 原库 GC 观测参考是 Study II EC 推断线的观测参照（2026-10-20 拍板 GC 计算主路径；context.md 技术路线表 Study II 行）
- **依据**：任务为真实数据 EC 参照计算（GC 计算主路径 × 新数据源口径），服务 EC 推断算法线，不涉及代理脑训练/选型/验收
- **新路径/方法检测（1.0c）**：`已有路径：原库 GC 计算主路径（granger_bivariate_spectral，edrafmt48 档）` —— 本任务为同方法换数据源口径（v4 + 重采样 250 Hz），无新方法
- **research 产物**：`无`
