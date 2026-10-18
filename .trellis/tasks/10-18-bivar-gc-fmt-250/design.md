# design.md — edra-fmt 250 Hz 原始全长双变量 GC 真实参照（48 例 · 不降采 · 不切窗）

> 口径总源：prd.md（用户 2026-10-18 四项拍板）+ 本文件 + evaluation-calibers.md。方向约定全仓库统一：**EC 矩阵行 = 源 source，列 = 靶 target**；有向边展平非三角（3782）。
> 定位：双变量 GC 观测参考（Study II 已有路径）的**原始采样率口径**第二条参照；与 125 Hz 首版（`models/bivar-gc-ec-v1/`，任务 10-17-bivar-gc-ec-fmt）平行，绝不混用、绝不覆写。

## 0. 范围与边界

- **输入**：`/root/autodl-tmp/edra_processed_fmt/` 48 例全量（manifest 全 status=ok；契约 `.trellis/spec/data/edra-fmt.md`）。数据集只读。
- **产出**：48 例受试者级 62×62 双变量 GC EC 矩阵（行=源/列=靶/对角=0，ln(SSR) readout）+ 逐对 BIC 阶数分布 + split-half 噪声天花板。
- **不做对比交付（用户 2026-10-18 拍板）**：不与 125 Hz 参照、不与 Study I 窗式 GC 出任何对比表/图；比较类工作届时另立任务。
- **无生成侧/模型依赖**：不触任何 checkpoint、不触 Study I 代码运行时。

## 1. 模块布局（path-isolation 薄变体 v2）

```
models/bivar-gc-ec-v2/
├── config.py            # 全部口径常量（FS=250、P_MAX=48、SEG_SAMPLES=500、OUTPUTS_ROOT、EXPECTED_IDS）；唯一名 bgec2_config
├── data_io.py           # 复刻 250 Hz 版：字段断言同 v1 + **无降采** + z-score 同式（复刻不 import：v1 版内嵌 resample 且绑定 bgec_config）
├── run.py               # CLI：--stage {smoke,main,split,figs}；主流程 16 路进程并行 + nohup
└── tests/               # 独立脚本单测（断言惯例，rng 2026）
```

- **只读 import 复用 v1**（sys.path：v2 目录居首、v1 目录仅追加，不复制）：
  - `bivar_gc.py` — 纯 numpy 函数、`p_max` 全链参数化（`build_gram` / `pair_fit` / `bivar_gc_matrix` / `naive_*` / `gram_pair_cond`，v2 全程显式传参），Gram-slice 核心保持单一源，v1 单测已锁定其正确性；
  - `split_half.py` — **API 事实（审查 Mandate A/B 一致核实）**：`build_halves(x, seg=)` / `half_ec(segs, p_max=)` / `offdiag_flat` 等可显式传参；**`ceiling_for_subject(x, p_max=)` 无 `seg` 参数且内部 `build_halves(x)` 不透传**（缺省落 v1 config `SEG_SAMPLES=250`，静默漂移面）——**v2 split 阶段不调用它**，改在 v2 本地编排：`build_halves(x, seg=500)` + `half_ec(·, p_max=48)` + `offdiag_flat` + Spearman（约 5 行，v1 保持零改动）。其顶层经唯一名 `bgec_config` 装载 v1 config，与 v2 `bgec2_config` 不冲突（单测断言两实例独立）。
- `data_io.py` 与 v1 **同名不同实现**：v2 目录居 sys.path 首位 ⇒ `import data_io` 恒解析到 v2 版；被复用的 v1 模块（bivar_gc、split_half）均不 import data_io，v1 版永不被装载。spawn 并行 worker 的重导入依赖模块顶部 sys.path 装配（复刻 v1 run.py 同式）。**import 解析归属断言**（`bivar_gc.__file__`/`split_half.__file__` 指向 v1、`data_io.__file__` 指向 v2）入单测 6。
- **对共享文件的已知写入点**：`common/visualizer.py` 仅当既有图型不覆盖时**纯增量**扩展（visualization.md 正典路线；check 阶段路径扫描以此声明为准）。
- **BLAS 护栏**：全部入口（run.py / tests / 并行 worker）在 numpy 导入前 setdefault 五变量；并行 worker 显式单线程/进程（16 进程 cgroup 上限，environment.md 契约）。

## 2. 口径（与 v1 差异逐条；未列条目 = 与 v1 逐字同式）

### 2.1 数据准备

- 字段断言同 v1：`filter=="1-40(release)"`、`pipeline=="EDRA_release_1-40_250Hz"`、`fs==250`、`truncated==False`、shape (62, 60001)——异常即 AssertionError。
- **无降采**（本任务唯一数据差异）：不调 `resample_poly`，250 Hz 原样进入拟合；元数据记 `fs=250`、`resample="none"`。
- z-score 逐通道全记录（ddof=0，σ 下限 np.maximum(sd, 1e-12)）；float64 计算、float32 落盘缓存。

### 2.2 拟合与选阶（复用 v1 bivar_gc，参数换档）

- **行集对齐惯例**：`row_from = P_MAX = 48`（全候选阶共用行集，t=48..60000，n_eff = 59953/例）。
- **候选网格**：p ∈ {1..48}。**P_MAX=48 的依据**：v1 口径 125 Hz@24 的最大滞后 = 192 ms；250 Hz 下保持同一物理滞后范围 ⇒ p 翻倍。这使两条参照的「模型阶」物理含义一致，是采样率口径变更下唯一保不变量的选择（备选 P_MAX=24@96 ms 会同时改变滞后范围与采样率两个变量，违背单变量变更原则，不采用）；与本仓库既有先例同构（fs 变更时 GPR `period_p` 50→25 保物理时长，evaluation-calibers.md）。
- 逐无序对 {A,B}：从全通道滞后 Gram（G (1+62·48)²=2977² 上三角，`build_gram` 按行块累积）slice 出二通道子问题，`pair_fit` 逐阶求解：
  - BIC(p) = ln|Σ̂(p)| + (K²·p)·ln(n_eff)/n_eff，K=2；截距跨阶同数不进 argmin。选 p\*({A,B})，并列取最小 p。
  - p\* 下双向读出：GC(A→B) = ln(SSR_r^B / SSR_u^B)，GC(B→A) 对称；对角按构造填 0。
- **OLS 非负断言**：offdiag≥0 作为实现正确性断言（负值即刻失败）。
- **BIC 顶格预告（如实登记）**：惩罚 4·48·ln(59953)/59953 ≈ 0.035，p\*=48 顶格为预期结局；登记口径 = 实测分布照报，不做调参干预；网格扩展沿用 v1 遗留未决决策，不在本任务处理。p\* 顶格 ⇒ 实效估计量为**固定阶 VAR(48) 双变量 GC**，refs_summary 的 caliber 字段显式注记，向下游言明估计量性质。
- **条件数**：二通道 unrestricted 2p+1 ≤ 97 维（restricted p+1 = 49）、n_eff≈6 万行，数值余量优于 v1（2p+1=49 / 3 万行）；smoke 阶段以 `gram_pair_cond` 登记 1891 对条件数 min/median/max 入 refs_summary.json。**数值兜底 = halt 请示（审查 Mandate A/B 一致结论）**：250 Hz 过采样下相邻滞后共线性升高，条件数预期高于 v1 实测带 ~1e5–1e7；若 smoke 实测显著越限，**暂停、登记实测分布、请示用户拍板**——λ_rel 对角加载在「只读复用 + OLS 非负断言」约束下不可实施（v1 无加载入口；ridge 下 offdiag≥0 不再必然，须口径变更重定义断言），不存在「登记后自行切换」的执行空间。

### 2.3 split-half 噪声天花板

- 段长 500 样本（2 s @ 250 Hz）；偶段→半 A、奇段→半 B、尾段丢弃；段内设计不跨段（段内滞后行 t≥48..499，段间行堆叠，每半 n_eff = 60×452 = 27120 行）。
- **运行时口径锁**（防 v1 `ceiling_for_subject` 缺省段长 250 的静默漂移）：断言 n_seg==120、每半行数==27120、`seg_samples=500` 写入 sh-01 summary。
- 每半独立同口径（含逐对 BIC，P_MAX=48）→ 两半 offdiag 3782 边 Spearman ρ（逐例）→ 天花板 = mean±sd（ddof=1，48 例）。

### 2.4 时间反转检验（R6，2026-10-18 用户追加）

- **检验逻辑**：有向度量的方向读数应依赖时间先后；序列沿时间翻转后以完全相同口径重算，度量对"时间序"的敏感性即暴露——fwd/rev 高度一致 ⇒ 读数主要来自对称耦合（瞬时相关/体积传导类伪影），显著改变 ⇒ 读数含时间方向成分。
- **口径**：`X_rev = X_z[:, ::-1]`（逐例，z-score 后翻转——z-score 为逐通道时不变统计，翻转次序无影响）；其余与主 Run 逐字同口径（单段全长、row_from=48、逐对 BIC p∈{1..48}、ln(SSR) 双向、OLS 非负断言）。BIC 顶格预期同主 Run——依据惩罚项极小（≈0.035）+ n_eff 巨大 + 125 Hz 实测 100% 顶格（反转可改变 logdet 项，量级对比不变）；p\* 分布以实测登记为准。
- **预登记指标**（逐例 → 组级 mean±sd）：① `edge_fwd_rev` = Spearman ρ(offdiag GC_fwd, offdiag GC_rev)；② `edge_fwd_mirrorrev` = ρ(offdiag GC_fwd, offdiag GC_revᵀ)；③ 幅值 = median offdiag 比值（rev/fwd）；④ 方向不对称度 = mean|GC(A→B)−GC(B→A)|，fwd vs rev。
- **判读框架（预登记，非硬门限）**：二阶平稳高斯过程时间反演下二阶结构不变、反向 VAR 系数不同 ⇒ GC_rev 与 GC_fwd 既不相等也不必归零（文献中此检验用作方向读数的时间序敏感性指纹）；本检验为描述性诊断，结论落 experiment.md 供 EC 推断线判读引用。
- **实现**：run.py 新增 `--stage reverse`（装载链复用主链 + 翻转一行 + 逐例同管线）；GC_rev 与四组指标落 `rev-01/refs_summary.json`（fwd 基准引用 refs-01）。

## 3. 数据与运行规模（environment.md 契约内）

| 项 | 规模 | 预算 |
|---|---|---|
| Gram 构建 | G 2977²≈71 MB f64/例；Z 行块瞬时 ≤8192×2977×8≈195 MB | 单例秒级（BLAS dsyrk 级） |
| 主 Run 求解 | 48 例 × 1891 对 × 48 阶 ≈ 4.4M 小系统（2p+1≤97 维）+ 48 次 Gram | 基线 = v1 refs-01 实测 36 s（非估算）；本档阶数×2、行数×2、Gram 宽×4，外推分钟级 ≈1–3 min；nohup 惯例照旧 |
| split-half | 数据减半 × 2 半 × 48 例 | 与主 Run 同量级 ×2 |
| 时间反转 rev | 与主 Run 同量级（仅数据翻转，管线不变） | ≈1–3 min |
| GPU | 不使用（CPU-only numpy/scipy） | — |

- 产物布局：`/root/npi/outputs/bivar-gc-ec-fmt-250/<run_id>/{data,refs,logs,figs}/` + `refs_all.npz` + `refs_summary.json` + `config_snapshot.json`（artifacts-management.md）。
- run_id 序列：`smoke-01`（2 例：EXPECTED_IDS 前 2 例）→ `refs-01`（48 例主 Run）→ `refs-01-reprod`（AC6 逐位复现抽查）→ `rev-01`（时间反转）→ `sh-01`（split-half）；rev-01 与 sh-01 名义顺序可互换（均仅依赖 refs-01）。无比较器 run（用户拍板）。

## 4. 单测（tests/ 独立脚本，rng 2026；同 v1 六项，参数换 250 Hz 档）

1. **方向单测**：合成二通道 VAR，仅 A→B 耦合 → 断言 GC(A→B) > GC(B→A) 且 GC(B→A)≈0（经 v2 import 的 v1 `bivar_gc` 路径，P_MAX=48）。
2. **独立性零基线**：独立白噪二通道 → 双向 GC≈0（|GC|<阈值）。
3. **样本内非负断言**：真实数据任意对读出 offdiag≥0。
4. **Gram-slice 等价性**：小规模随机数据，slice 路径 vs naive `lstsq`：SSR/BIC/GC 逐位对齐（rtol=1e-10）——同时锁定 v2 的 v1 import 链路无遮蔽漂移。
5. **逐位复现**：同输入两次运行输出逐位相同（确定性契约）。
6. **装载与归属断言**：fs≠250 / 长度≠60001 / truncated 标记 → AssertionError（data_io 契约）；`bivar_gc.__file__`/`split_half.__file__` 指向 v1 目录、`data_io.__file__` 指向 v2 目录；v1 `bgec_config` 与 v2 `bgec2_config` 实例独立。
7. **split 段长锁定**：合成 T=1500、seg=500 → n_seg==3（锁定 v2 编排路径，防 v1 缺省 250 漂移面）。
8. **时间反转改变断言**（增量复审实证校准：二通道 offdiag 仅 2 边、Spearman 简并为 ±1，改语义直写）：强单向合成 VAR（仅 A→B）翻转后断言 `GC_rev(B→A) > GC_rev(A→B)` 且 `|GC_rev(A→B) − GC_fwd(A→B)|` 超显式阈值、GC_rev 有限非负（防"翻转被实现无视"的静默失败，如时间轴索引错置；T 与 p_max=48 入断言锁定）。

## 5. 已知风险与对策

| # | 风险 | 对策 |
|---|---|---|
| 1 | BIC p\*=48 顶格（预期内） | 如实登记实测分布；不调参凑数；网格扩展属 v1 遗留未决决策，另行走口径变更 |
| 2 | v1 模块 import 链路遮蔽/漂移 | v2 不定义同名核心模块；单测 4 经 v2 路径重锁等价性；`bgec_config`/`bgec2_config` 唯一名互不干扰（单测 6） |
| 3 | Gram-slice 索引错位（p_max=48 新档位） | 复用 v1 已验证代码 + 单测 1/4；smoke 2 例人工核对统计量量级 |
| 4 | OLS 数值条件（250 Hz 过采样相邻滞后共线性 ↑） | smoke 登记 1891 对条件数 min/median/max；越限 → halt + 登记实测分布 + 请示用户拍板（λ_rel 加载与只读复用/非负断言不兼容，禁止自行切换） |
| 5 | 16 进程 × BLAS 超订 | worker 显式单线程 + 入口五变量护栏（environment.md 教训） |
| 6 | visualizer 图型覆盖 | `common/visualizer.py` 已含 ec_matrix / ec_heatmap_grid（审查核实，B5 大概率零改动）；BIC 阶数分布沿 v1 先例——**退化（顶格）不出图、只入 summary**；缺口才纯增量并登记 check 白名单 |

## 6. 执行序与回滚

B0 装载/环境冒烟 → B1 config/data_io/run 落地 + 单测全绿 → B2 smoke-01 → B3 主 Run refs-01（nohup）→ B4 split-half sh-01（v2 本地编排）→ B5 图（阶数分布非退化才出图，沿 v1 先例）+ experiment.md 回填 + evaluation-calibers 条目 + trellis-check + 提交。各块产物独立落 run_id，回滚 = 弃 run_id（产物移该 run 内 `archive/` 子目录，不删除）重跑；v1 目录与 `outputs/bivar-gc-ec-fmt/` 旧产物零写入（口径隔离）。详见 implement.md。
