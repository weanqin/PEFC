# implement.md — 执行计划（10-18-bivar-gc-fmt-250）

> 执行序 B0→B5；每块验证命令就地给出；回滚 = 弃 run_id 重跑，旧产物零写入。
> 环境契约：`.trellis/research/environment.md`（16 核 cgroup、BLAS 五变量护栏、CPU-only）。Run 前置纪律：experiment.md 先登记目的+配置。

## B0 装载/环境冒烟

- [ ] 确认数据集只读可访问：48 例 npz 逐例存在、字段断言样本抽查（fs==250、shape (62,60001)）。
- [ ] `PYTHONDONTWRITEBYTECODE=1 python3 -c` 冒烟：v2 目录 sys.path 装配（v2 居首、v1 仅追加）下 `import bivar_gc, split_half`（解析到 v1）可装载、`sys.modules` 含 `bgec_config` 且 `bgec2_config` 名未被占用（实例独立断言归 B1 单测 6）。
- 验证：脚本退出码 0；无任何写入（数据集与 v1 目录 `*.py` mtimes 不变）。

## B1 代码落地 + 单测

- [ ] `models/bivar-gc-ec-v2/config.py`：FS=250、P_MAX=48、SEG_SAMPLES=500、OUTPUTS_ROOT=`/root/npi/outputs/bivar-gc-ec-fmt-250`、EXPECTED_IDS（同 v1 48 例清单）、SEED=2026、N_PROC=16；唯一名 `bgec2_config` 四行配方。
- [ ] `models/bivar-gc-ec-v2/data_io.py`：复刻 v1 同式（字段断言 + z-score ddof=0 σ≥1e-12 + float64/f32 口径），**删降采步**；BLAS 护栏置于 numpy 前。
- [ ] `models/bivar-gc-ec-v2/run.py`：`--stage {smoke,main,reverse,split,figs}`，16 进程并行（worker 单线程 BLAS + 模块顶部 sys.path 装配复刻 v1 run.py 式），产物落 OUTPUTS_ROOT/<run_id>/{data,refs,logs,figs}/。**smoke = 主流程限 2 例 + 同式落盘产物**（v1 的 smoke 为纯控制台不落盘，不照抄）；条件数登记 = `gram_pair_cond` 全 1891 对 @p=48 的 min/median/max 入 refs_summary.json。
- [ ] `models/bivar-gc-ec-v2/tests/`：design §4 八项单测（方向/零基线/非负/slice↔naive rtol=1e-10/逐位复现/装载与归属断言/split 段长锁定/时间反转改变断言）。
- 验证：`python3 models/bivar-gc-ec-v2/tests/run_all.py` 全绿（或逐脚本退出码 0）；`git status` 确认 v1 目录零改动。

## B2 smoke-01（2 例）

- [ ] experiment.md 登记 Run smoke-01（目的=链路冒烟+条件数登记；配置=2 例、P_MAX=48）。
- [ ] `nohup python3 models/bivar-gc-ec-v2/run.py --stage smoke --run-id smoke-01 &`；完成后人工核对：GC 量级与 v1 同量级、offdiag 非负、条件数登记落表、p\* 分布如实记录。
- 验证：refs_summary.json 存在且含 2 例矩阵；无 NaN；回填 experiment.md 结果+结论。

## B3 主 Run refs-01（48 例）

- [ ] experiment.md 登记 Run refs-01（目的=250 Hz 全长参照主交付；配置=48 例全量、不降采、不切窗、P_MAX=48、OLS）。
- [ ] `nohup python3 models/bivar-gc-ec-v2/run.py --stage main --run-id refs-01 &`（预期 10–15 min 级）。
- 验证：`refs_all.npz` (48,62,62) f32 + 对角=0 断言 + offdiag≥0 全通过；`refs_summary.json` 含 BIC 阶数分布（顶格与否如实）+ caliber 注记（顶格 ⇒ 实效固定阶 VAR(48)）；AC6 逐位复现抽查：重跑入 scratch run_id `refs-01-reprod`，逐字节比对对应 per-subject npz（imap_unordered + subject_id 排序堆叠 ⇒ refs_all.npz 逐位可比）。

## B4 split-half sh-01

- [ ] experiment.md 登记 Run sh-01（目的=参照噪声天花板；配置=段 500 样本、奇偶分半、每半独立 BIC）。
- [ ] **v2 本地编排（审查 Mandate A/B 一致发现：不调 v1 `ceiling_for_subject`——其无 `seg` 参数且不透传，缺省会静默用 v1 config 的 250 样本段）**：`split_half.build_halves(x, seg=500)` → `half_ec(·, p_max=48)` 两半 → `offdiag_flat` + Spearman；**运行时口径锁**：断言 n_seg==120、每半行数==27120、`seg_samples=500` 写入 summary。sh-01 独立 run_id，**不复制** v1 stage_split 的「同 run_id refs_summary.json 前置断言」（那是 v1 同 run_id 惯例产物）。
- [ ] `nohup python3 models/bivar-gc-ec-v2/run.py --stage split --run-id sh-01 &`。
- 验证：逐例 ρ 落表、天花板 mean±sd 落 `refs_summary.json`（或独立 summary）；回填 experiment.md。

## B4.5 时间反转 rev-01

- [ ] experiment.md 登记 Run rev-01（目的 = GC 方向读数时间序敏感性指纹；配置 = 同主 Run、仅数据沿时间翻转）。
- [ ] run.py `--stage reverse`：装载链复用主链，`X_rev = X_z[:, ::-1]`，逐例同管线 → 四组预登记指标（edge_fwd_rev / edge_fwd_mirrorrev / 幅值比 / 方向不对称度）入 `rev-01/refs_summary.json`（fwd 基准读 refs-01，缺失即报错）。**GC_rev 落盘规格**：逐例 `refs_s<id>.npz`（ec=GC_rev）+ `refs_all.npz` 同 main 布局（含 `reversed=true` 标记）落 `rev-01/refs/`。
- [ ] `nohup python3 models/bivar-gc-ec-v2/run.py --stage reverse --run-id rev-01 &`（≈分钟级）。
- 验证：48 例 GC_rev 非负断言通过；四组指标齐全；判读按 design §2.4 预登记框架写 experiment.md 结论（描述性，无硬门限）。

## B5 图 + 登记 + check + 提交

- [ ] `--stage figs`：48 例 EC 矩阵图（visualizer 既有 ec_matrix / ec_heatmap_grid 路线，预期零改动）；BIC 阶数分布**退化（顶格）不出图、只入 summary**（沿 v1 先例）；缺口才纯增量并登记白名单。
- [ ] experiment.md 全部 Run 块回填完整（目的/配置/结果/结论，无空块）。
- [ ] `evaluation-calibers.md` 增补「双变量 GC 观测参考 · 250 Hz 原始全长口径」条目（与 125 Hz 条目并列、显式区分）。
- [ ] `.trellis/research/context.md` 技术路线表 Study II 行增补一句产物登记；`progress.md` 当前状态同步。
- [ ] 派发 `trellis-check`（全范围末轮）；lint/类型检查（纯 numpy 脚本任务以 tests 全绿 + py_compile 为准）。
- [ ] Phase 3.4 提交（工作提交先行，归档/journal 随后）。

## 回滚点

- 任一 Run 失败/数值异常 → 弃该 run_id，修复后重跑；已归档 Run 不覆写。
- 条件数越限 → **暂停 + 登记实测分布 + 请示用户拍板**（λ_rel 加载与只读复用/OLS 非负断言不兼容，禁止实施中自行切换）。
- 弃 run_id → 产物移该 run 目录内 `archive/` 子目录（不删除，artifacts-management.md 惯例）；重跑用新 run_id。
- 单测红 → 回 B1，禁止带红跑 Run。
