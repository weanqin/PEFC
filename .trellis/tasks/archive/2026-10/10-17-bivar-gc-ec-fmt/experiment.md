# 实验日志 — 10-17-bivar-gc-ec-fmt（edra-fmt 成对双变量 GC 计算 EC）

> 每次「跑批/训练/评估/统计」追加一个 Run 块：跑前先写「目的+配置」，跑后同会话补「结果+结论」，不留空块。
> 格式契约见 `.trellis/spec/research/experiment-log.md`；产物只落本任务声明的变体目录（`.trellis/spec/research/path-isolation.md`）。
> 任务级声明：**不做 EC_pert 与 GC 的比较**（用户 2026-10-17 第四轮拍板）；本任务 GC 一律 125 Hz 真实数据侧 observational directed-connectivity reference，与 fmt 评估侧「250 Hz genGC vs real GC 阻塞」口径为不同对象。

## Run 001 — smoke-01 真实数据装载/条件数/量级冒烟（2026-10-17）

### 目的与假设

- B2 全量前置（G1 门素材）：真实 fmt 数据上核对①装载字段断言与 z-score 复核、②pair-Gram 条件数（OLS 数值安全依据，对照全模型 62ch ~1e18 情形）、③真实 GC 统计量量级（对合成预演的 sanity）、④耗时标定。控制台输出，不建 run 目录不落盘。

### 配置

- 代码：`models/bivar-gc-ec-v1/` 首版（config/data_io/bivar_gc/split_half/run/tests；单测 20 项全绿，T3 Gram-slice↔naive 等价 rtol=1e-10）
- 参数：P_MAX=24、BIC p∈{1..24}（行集 row_from=24 嵌套）、FS 250→125（resample_poly up=1/down=2）、逐通道全记录 z-score ddof=0 σ≥1e-12；manifest 升序前 2 例
- 环境：miniconda3 base @ 本地（environment.md 契约；BLAS 五变量护栏 setdefault）
- 数据：`/root/autodl-tmp/edra_processed_fmt/`（48 例 status=ok，只读）
- 命令：`python3 models/bivar-gc-ec-v1/run.py --stage smoke --subjects 2`；另以同 2 例跑 `bivar_gc_matrix` 全流程控制台预演（真实 GC 统计量，不落盘）

### 结果

- `--stage smoke --subjects 2`（s1 MDD / s4 HC）：字段断言全过（pipeline/filter/fs/truncated）；形状 (62,30001) ✓；z-score 复核 max|mean|≤3.2e-17、max|σ−1|≤2.2e-16；n_eff=29977。
- **pair-Gram 条件数（8ch 子集 28 对 @p=24）**：s1 min=4.1e4 / median=1.0e5 / max=9.6e6；s4 min=2.5e5 / median=7.3e5 / max=5.9e6——与全模型 62ch ~1e18（baseline 需 ridge 情形）差 ~11 个量级，OLS 数值安全，预注册对角加载备选不启用。
- **真实 GC 统计量预演（全流程）**：s1 min=0.0012 / median=0.0085 / max=0.0911 / mean=0.0105；s4 min=0.0019 / median=0.0134 / max=0.1045 / mean=0.0160（ln(SSR) 单位；offdiag≥0 断言过）。
- 耗时：装载+Gram 1.6–1.7 s/例；全流程 8.8–9.5 s/例（与合成预演 9.5 s 一致）。
- **阶数实测事实：BIC 全部对顶格 p\*=24**（s1/s4 的 min=median=mode=max=24）——n≈3 万下惩罚项 K²p·ln(n)/n≈3.4e-4·4p 极小、似然增益持续占优所致；按预注册网格 p∈{1..24} 执行，阶数分布退化为边界值如实登记，网格扩展属口径变更留后续任务决策。

### 结论

- G1 门通过：断言全过、条件数安全、量级合理、耗时标定成立 → 允许 refs-01 全量。无代码修正（Run 002 即本首版代码）。

## Run 002 — refs-01 主 Run：48 例成对双变量 GC EC 矩阵（2026-10-17）

### 目的与假设

- 产出主交付物：48 例受试者级 62×62 双变量 GC EC 矩阵（行=源/列=靶/对角=0，ln(SSR) readout）+ 逐对 BIC 阶数分布 + refs_stats 登记。假设：offdiag≥0 必然成立（样本内 OLS 单调性断言，负值=实现错误即刻失败）；单例耗时与合成预演（9.5 s）同量级。

### 配置

- 代码：同 Run 001 首版（如 Run 001 触发修正则以修正版为准并在此登记）
- **修复登记（首跑失败→重跑）**：首跑 48 例计算与逐例落盘完整，但聚合段 `ec_all[~np.eye(62)]` 对 (48,62,62) 用 (62,62) 掩码直接索引 → IndexError，refs_summary/config_snapshot 未写出（stderr 当时被丢弃，后经已落盘产物离线复现定位）。修复为 `ec_all[:, ~np.eye(62)]` 批次维广播；计算路径未变，同 run_id 重跑产物确定性一致；stderr 此后落 `logs/nohup.log`。
- 参数：同 Run 001；16 进程 spawn 并行、worker 单线程 BLAS（cgroup 有效 16 核契约）
- 环境：同 Run 001
- 数据：全 48 例（manifest 全 status=ok；与 EXPECTED_IDS 逐位断言）
- 命令（修复后重跑版）：`nohup python3 models/bivar-gc-ec-v1/run.py --stage main --run-id refs-01 > outputs/bivar-gc-ec-fmt/refs-01/logs/nohup.log 2>&1 &`（stdout/stderr 落 nohup.log，见修复登记；首跑为 `> /dev/null 2>&1` 致 stderr 丢弃。run.log 为 append 模式，双跑轨迹留存）
- 产物：`outputs/bivar-gc-ec-fmt/refs-01/{refs_all.npz, refs/refs_s*.npz, data/resample_meta.json, refs_summary.json, config_snapshot.json}`

### 结果

- 重跑完成（修复后）：48/48 例，wall 36 s（16 进程），逐例统计与首跑逐字一致（跨 Run 复现证据，另 T4 单测逐位复现）。
- **refs_all.npz**：bivar_gc (48,62,62) f32（行=源/列=靶/对角=0 ✓）；offdiag **min=0.0007 / median=0.0153 / max=0.4356**（ln(SSR) 单位）；HC 24 / MDD 24。
- **阶数分布（48 例 × 1891 对全量）**：p\*=24 占 100%（pooled 1152=24×48/对）——Run 001 的边界顶格事实在全量成立，如实登记。
- 产物齐全：refs/refs_s*.npz ×48（ec/orders/n/subject_id）、data/resample_meta.json、refs_summary.json、config_snapshot.json、logs/{run.log,nohup.log}。
- 图表（B4，经 common/visualizer 正典路由，纯复用零改动）：`figs/bivar_gc_group_means.png`（ec_matrix 三面板：HC n=24 / MDD n=24 / 全体 n=48 组均，Blues 非负、对角掩膜；组均量级 median=0.019 / max=0.076，目检通过）；`figs/bivar_gc_subjects_sample.png`（ec_heatmap_grid 等距 6 例，行内共享 vmax）。**阶数分布图裁量**：分布退化为单点（100% p\*=24），以 refs_summary.json 登记呈现，不为单柱图新增 visualizer 图型（共享文件零改动）。
- 断言全过：全有限 / 对角=0 / offdiag≥0（样本内 OLS 单调性实现正确性断言）。

### 结论

- 主交付物达成：48 例双变量 GC EC 矩阵完整落盘，口径元数据完备（observational reference / 非负断言 / 与 ridge 条件 GC 口径区分已写入 caliber）。BIC 顶格为实测事实（登记，不构成失败）；网格扩展（如 p∈{1..48}）属口径变更，留后续任务决策。→ 允许 B3 split-half。

## Run 003 — split-half 噪声天花板（sh；产物落 refs-01/split_half.json）（2026-10-17）

### 目的与假设

- 双变量 GC 参考自身的噪声天花板登记（后续任何比较可直接取用作 ρ̄/ceiling 分母）：奇偶 2 s 段分半、每半同口径（含逐对 BIC）、逐例 offdiag 3782 边 Spearman → mean±sd（ddof=1）。实现选择：split 产物与 main 同 run_id 落 `refs-01/split_half.json`（沿 baseline-ec-v1 split_half.json 同 run 目录惯例；design §3「sh-01」独立 run_id 命名据此微调，特此登记）。

### 配置

- 代码/参数：同 Run 002；段长 250 样本（2 s@125 Hz）、偶A/奇B/尾弃、段内设计不跨段
- 命令：`nohup python3 models/bivar-gc-ec-v1/run.py --stage split --run-id refs-01 >> outputs/bivar-gc-ec-fmt/refs-01/logs/nohup.log 2>&1 &`

### 结果

- 48/48 例（wall 55 s，16 进程；n_seg=120/例 = 60 段/半 × 250 样本）。
- **天花板 = 0.9243 ± 0.0447**（mean±sd，ddof=1）；逐例 min=0.7895 / max=0.9878。
- 产物：`refs-01/split_half.json`（caliber/逐例 ρ/段口径/wall）。

### 结论

- 双变量 GC 参考自身噪声天花板就位（后续任何比较可直接取用作 ρ̄/ceiling 分母）。与 v4 侧条件 GC 天花板（0.9003±0.0296，46 例）同量级——**方向性对照只登记不判读**（跨范式多变量差异，edra-fmt.md 契约纪律）。B2/B3 全部完成，无 TBD 残留。
