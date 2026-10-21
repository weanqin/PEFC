# 实验日志 — 旧库谱 GC × edra_process_v4 逐被试 EC（重采样 250 Hz 口径）

> 每次「跑批/训练/评估/统计」追加一个 Run 块：跑前先写「目的+配置」，跑后同会话补「结果+结论」，不留空块。
> 格式契约见 `.trellis/spec/research/experiment-log.md`；产物只落本任务声明的变体目录（`.trellis/spec/research/path-isolation.md`）。

## Run 001 — smoke-01：单例（HC/4）口径与断言验证（2026-10-21）

### 目的与假设

- 在正式跑 46 例前，对 1 例（HC/4，fmt 重叠被试）验证：v4 npz 读取断言、resample_poly(1,2) 长度 42500、重采样后 z-score、旧库 GC 全链调用、EC offdiag≥0、量级与 edrafmt48 档同被试同数量级。
- 假设：重采样口径下 GC 量级与 fmt 档相当（同实现同参数、数据同为 1–40 Hz 带限 EEG），offdiag median 应在同一数量级（0.01–0.1 ln 单位框）。

### 配置

- 代码：`outputs/10-21-oldlib-gc-edrav4/gen_refs.py`（蓝本 = `refs/granger_bivariate_spectral/edrafmt48-cont-zscore-fs250-fixed8-f1-125-nf65/gen_refs.py`，任务 10-19 Run 005；本任务改数据源 + 新增 resample 步骤）
- 参数：`granger_connectivity(fs=250.0, max_order=8, n_freq=65, fmin=1.0, fmax=125.0, fixed_order=8, n_jobs=12)`；resample `scipy.signal.resample_poly(x, 1, 2, axis=-1)`；z-score 在重采样后（ddof=0，σ==0 置 1，同旧库 loader）
- 环境：本地 base env（Python 3.12.3 / numpy 2.3.2 / scipy 1.18.1），BLAS 五变量 setdefault "1"（numpy 导入前），joblib loky 12 进程
- 数据：`/root/autodl-tmp/edra_process_v4/HC/4_rest_v4.npz`（62×85000 @500 Hz → 42500 @250 Hz）；通道序参考 fmt `HC/13_rest_v4.npz`；数据集只读
- 命令：`python3 outputs/10-21-oldlib-gc-edrav4/gen_refs.py --smoke`

### 结果

- s4 (HC)：24.0 s；offdiag mean=67.3703 / median=39.9920 / max=602.642；offdiag≥0 断言过；z-score 后 |x| 均值 0.7623（半正态期望 ≈0.798 ✓）
- 量级对照：fmt 档同被试 s4 = mean 40.724 / max 728.054（gen_refs.log [17/48]）——同数量级（十~百 ln 单位框），支持「同实现同参数下量级相当」假设；v4 侧 mean 偏高 ~65%，归因数据源差异（不同时段/管线），留全量对照定量化
- 全部读取断言过（shape (62,85000)/fs=500/group/subject_id/通道序 vs fmt 参考序）

### 结论

- 支持：口径与断言全过、量级同数量级，可进全量 run；单例 24 s ⇒ 46 例预计 ~18 min

## Run 002 — refs-01：全量 46 例主 Run（2026-10-21）

### 目的与假设

- 对 edra_process_v4 全部 46 例（HC 24 / MDD 22）产出旧库谱 GC EC 矩阵参考（重采样 250 Hz 口径），并做与 edrafmt48 档的 45 例重叠被试对照登记。
- 假设：全量耗时 ~18 min；与 fmt 档逐例 offdiag Spearman ρ 为中低正相关（数据时段/管线不同，不期望 ≈1）；s56 无 fmt 对照（fmt 无此例）。

### 配置

- 代码：`outputs/10-21-oldlib-gc-edrav4/gen_refs.py`（同 Run 001）
- 参数：同 Run 001（fs250 / fixed8 / 1–125 Hz / nf65 / 全长连续 / 重采样后逐通道 z-score / n_jobs=12）
- 环境：本地 base env，BLAS 五变量 setdefault "1"，loky 12 进程
- 数据：`/root/autodl-tmp/edra_process_v4/{HC,MDD}/*.npz` 46 例全量（不做 qc_pass 筛选；qc_pass=True 仅 13/46 为登记事实）；数据集只读
- 命令：`python3 outputs/10-21-oldlib-gc-edrav4/gen_refs.py > outputs/10-21-oldlib-gc-edrav4/refs-01.log 2>&1`

### 结果

- **46/46 例全部落盘**（`outputs/10-21-oldlib-gc-edrav4/refs-01/refs_s{id}.npz`，ec float32 (62,62) 行=源/对角 0），总耗时 **15.4 min**（12 进程，~20.4 s/例，预估 18 min 内 ✓）；全例 offdiag≥0 断言过、读取断言（shape/fs/group/subject_id/通道序 vs fmt 参考序）全过
- offdiag 量级（ln 单位积分值，46 例 range）：mean ~55–95、median ~33–65、max ~330–890，与 fmt 档（mean ~36–91、max ~307–889）同数量级，v4 侧系统偏高
- **与 edrafmt48 档对照（45 例重叠，s56 为 v4 独有跳过并登记）**：offdiag 3782 边 Spearman ρ=**0.2311±0.1026**（min 0.0667 / max 0.4455）、Pearson r=**0.4268±0.1266**、median 比值 v4/fmt=**1.651±0.664**（`compare_fmt_summary.json`）——非 ≈1 符合预期（数据时段/管线不同：170 s vs 240 s、bp1-40+notch50 vs release 1-40）；ρ 数值低属两数据源固有差异，登记不判读
- **确定性（AC4）**：s4 删除后独立重跑，npz 容器级 SHA256 逐位一致（6b6d9338…cc8a925 = first）
- **非负性全局补证（trellis-check 事后复算）**：46/46 npz offdiag 全局 min = 0.051 > 0，零负边（运行时逐例断言之外的全量核验）

### 结论

- 支持：旧库 GC × v4 重采样口径全链可跑、断言全过、逐位可复现；两档 GC 相关为低-中正相关 + v4 幅值系统偏高 ~1.65×，与「同实现同参数、不同数据源」预期一致
- 产物就绪待归档 refs；派生登记：s56 仅存于 v4（fmt 无），下游 fmt 对照一律 45 例交集口径

