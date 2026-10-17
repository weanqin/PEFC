# design.md — edra-fmt 成对双变量 GC 计算 EC（48 例 · 125 Hz · 逐对 BIC · ln(SSR)）

> 口径总源：prd.md（含用户 2026-10-17 三轮拍板）+ 本文件 + evaluation-calibers.md。方向约定全仓库统一：**EC 矩阵行 = 源 source，列 = 靶 target**；有向边展平非三角（3782）。
> 定位：observational directed-connectivity reference 的**双变量口径**首次主轨落地；与既有全模型条件 GC（baseline-ec-v1 主口径）平行，绝不混用。

## 0. 范围与边界

- **输入**：`/root/autodl-tmp/edra_processed_fmt/` 48 例全量（manifest 全 status=ok；契约 `.trellis/spec/data/edra-fmt.md`）。数据集只读。
- **产出**：48 例受试者级 62×62 双变量 GC EC 矩阵（行=源/列=靶/对角=0，ln(SSR) readout）+ 逐对 BIC 阶数分布 + split-half 噪声天花板。
- **不做 EC_pert 与 GC 的比较（用户 2026-10-17 第四轮）**：比较器接入整体移出范围、决策滞后；fmt 条件 GC（10 例复用 / 48 例重算 / 时点）同样滞后。现有 EC_pert 仅 10 例且用户已判定该 θ\* 完全不可用——后续若比较，届时另定任务。
- **口径隔离**：本任务 GC 在 **125 Hz 真实数据**侧计算，与 fmt 评估侧被阻塞的「250 Hz genGC vs real GC 一致性」为不同对象；experiment.md 显式登记该区分。

## 1. 模块布局（path-isolation 薄变体）

```
models/bivar-gc-ec-v1/
├── config.py            # 全部口径常量（FS/P_MAX/SEG_SAMPLES/SEEDS/OUTPUTS_ROOT/EXPECTED_IDS）
├── data_io.py           # fmt manifest + prepare_subject（复刻 refs_fmt 同式：字段断言/1:2 降采/z-score ddof=0 σ≥1e-12）
├── bivar_gc.py          # 核心：全通道滞后 Gram 一次构建 + 逐无序对逐阶 slice 求解 + BIC 选阶 + ln(SSR) 双向读出（numpy-only）
├── split_half.py        # 奇偶 2 s 段分半（复刻 baseline 段式）→ 每半同口径（含逐对 BIC）→ 逐例 offdiag Spearman
├── run.py               # CLI：--stage {smoke,main,split,figs}；主流程 16 路进程并行 + nohup
└── tests/               # 独立脚本单测（断言惯例，rng 2026）
```

- **只读依赖（不改写，无运行时复用点）**：`/root/autodl-tmp/edra_processed_fmt/`（数据 + chan_coords_62.npy）；`models/baseline-ec-v1/` 与 `models/perturb-ec-v1/refs_fmt.py` 仅作**口径惯例参照**（z-score 同式、row_from=p_max 嵌套行集、行=源/列=靶/对角=0——公式复刻，不 import；原 compare.py 子进程隔离复用随第四轮「不做比较」移出而取消）。
- **对共享文件的已知写入点**：`common/visualizer.py` 仅当既有图型不覆盖时**纯增量**扩展（visualization.md 正典路线；check 阶段路径扫描以此声明为准）。
- **config 唯一名装载**：本目录 config 一律经 importlib 唯一名 `bgec_config` 装载（四行配方，先注册后 exec），不占据 `config` 模块名。
- **BLAS 护栏**：全部入口（run.py / tests 启动器 / 并行 worker）在 numpy 导入前 setdefault 五变量；主流程并行 worker 显式单线程/进程（16 进程 cgroup 上限，避免 BLAS 超订——environment.md 有效 16 核事实）。

## 2. 双变量 GC 口径（核心，用户三轮拍板落定）

### 2.1 数据准备（与 refs_fmt.prepare_subject 逐式同构）

- 字段断言：`filter=="1-40(release)"`、`pipeline=="EDRA_release_1-40_250Hz"`、`fs==250`、`truncated==False`、shape (62, 60001)。
- `resample_poly(x, 1, 2, axis=-1)` → (62, 30001) @125 Hz（抗混叠 FIR 内置；不重复滤波——release 已带限 1–40 Hz）。
- 逐通道全记录 z-score（ddof=0，σ 下限 1e-12）；float32 落盘缓存 / float64 复原计算（resample_io 往返惯例）。
- fs 原值/目标值、降采参数、z-score 口径全部入产物元数据。

### 2.2 拟合与选阶（逐无序对 · 一次拟合双向读出）

- **行集对齐惯例**：BIC 全候选阶共用行集 `row_from = P_MAX = 24`（t=24..30000，n_eff≈29977/例）——与 baseline-ec-v1「对齐行集 row_from=p_max」同惯例，消除各阶样本数差异。
- **候选网格**：p ∈ {1..24}（对齐 baseline BIC 惯例；含截距列）。
- **逐无序对 {A,B}**：从全通道滞后 Gram（(2·62·24+1)² 上三角，一次 dsyrk 构建，~数百 MB f64）slice 出该对的二通道子问题，逐阶求解 multivariate OLS：
  - **BIC(p) = ln|Σ̂(p)| + (K²·p)·ln(n_eff)/n_eff**，K=2，Σ̂=2×2 残差协方差；截距跨阶同数不进 argmin 比较。选 p\*({A,B})。
  - p\* 下读出双向：unrestricted（二通道 VAR(p\*)）得 SSR_u^A、SSR_u^B；restricted（**同阶** p\* 单通道 AR）得 SSR_r^A、SSR_r^B；
  - **GC(A→B) = ln(SSR_r^B / SSR_u^B)**，GC(B→A) 对称；对角按构造填 0。
- **OLS 而非 ridge 的依据**：二通道 4p+1≤49 回归元、n_eff≈3 万行，与全模型 62 通道 ~1e18 条件数情形（baseline 需 ridge 的原因，Run 001 证据）不同构；smoke 阶段逐例登记 Gram 条件数实测，若出现数值问题再按口径变更纪律升级（预注册备选：对角加载 λ_rel=1e-4 档）。
- **样本内必然非负**：OLS 残差 SSR 单调性 ⇒ offdiag GC≥0 必然——**作为实现正确性断言**（负值=实现错误，即刻失败；与 ridge 条件 GC 允许微负不同，须在产物 caliber 注明区分）。
- **Gram-slice 快路径**：逐对阶求解全为 ≤49 维小系统（slice+solve，~45k 次/例）；与 naive `lstsq` 直接拟合的等价性由单测锁定（小规模随机数据逐位对齐 rtol=1e-10）。
- ln 比对逐通道缩放不变（z-score 下口径自洽）；矩阵存 float32 落盘（统计在 float64）。

### 2.3 split-half 噪声天花板（R4，用户第三轮认可）

- 段式复刻 baseline：段长 250 样本（2 s@125 Hz）；偶段→半 A、奇段→半 B、尾段丢弃；**段内设计不跨段**（段内滞后行 t≥24..249，段间行堆叠，n_eff≈226×60/半）。
- 每半**独立同口径**重跑双变量 GC（含逐对 BIC）→ 两半 offdiag 3782 边 Spearman ρ（逐例）→ 天花板 = mean±sd（ddof=1，48 例）。定位 = 参考自身的噪声天花板登记，后续任何比较可直接取用作 ρ̄/ceiling 分母。

## 3. 数据与运行规模（environment.md 契约内）

| 项 | 规模 | 预算 |
|---|---|---|
| 主 Run | 48 例 × 1891 对 × 24 阶 ≈ 2.2M 小系统求解 + 48 次 Gram dsyrk | 单例 ~1 min 内（Python 循环开销主导）；16 进程并行 ≈ 5 min；nohup 长跑惯例 |
| split-half | 数据减半 × 2 半 × 48 例 | 与主 Run 同量级 ×2 |
| 显存/内存 | Gram ~数百 MB f64/例 ×16 进程 | ≪503 GB，无压力 |
| GPU | 不使用（CPU-only numpy/scipy 路径） | — |

- 产物布局：`/root/npi/outputs/bivar-gc-ec-fmt/<run_id>/{data,refs,logs,figs}/` + `refs_all.npz` + `refs_summary.json` + `config_snapshot.json`（artifacts-management.md）。
- run_id 序列：`smoke-01`（2 例）→ `refs-01`（48 例主 Run）→ `sh-01`（split-half）。无比较器 run（第四轮拍板）。

## 4. 单测（tests/ 独立脚本，rng 2026）

1. **方向单测**：合成二通道 VAR，仅 A→B 耦合（B 方程含 A 滞后、A 方程不含 B）→ 断言 GC(A→B) > GC(B→A) 且 GC(B→A)≈0（<小阈值）；已知方向转置事故防线。
2. **独立性零基线**：独立白噪二通道 → 双向 GC≈0（|GC|<阈值）。
3. **样本内非负断言**：真实数据任意对读出 offdiag≥0（实现正确性，§2.2）。
4. **Gram-slice 等价性**：小规模随机数据，slice 路径 vs naive `lstsq` 直接拟合：SSR/BIC/GC 逐位对齐（rtol=1e-10）。
5. **逐位复现**：同输入两次运行输出逐位相同（确定性契约）。
6. **装载断言**：fmt 字段不匹配/截断标记 → 断言失败（data_io 契约）。

## 5. 已知风险与对策

| # | 风险 | 对策 |
|---|---|---|
| 1 | Gram-slice 切片/索引错位（快路径核心风险） | 单测 4 等价性锁定；smoke 先 2 例人工核对统计量量级 |
| 2 | 方向转置事故 | 单测 1 + 行=源/列=靶断言 + 对角=0 断言（沿 baseline-ec-v1 R4 惯例） |
| 3 | OLS 数值条件问题 | smoke 逐例登记条件数；预注册备选 λ_rel=1e-4 对角加载（口径变更纪律走 experiment.md 登记，主 Run 前最多调一次） |
| 4 | BIC 逐对异阶使跨对可比性受质询 | 口径本质（逐对模型选择）+ 阶数分布落表登记；split-half 同口径自洽 |
| 5 | 16 进程 × BLAS 超订打满 128 可见核 | worker 显式单线程 + 入口五变量护栏（environment.md 教训） |
| 6 | visualizer 既有图型不覆盖（阶数分布等） | 优先复用 ec_matrix；缺口按纯增量扩展并在 check 白名单登记 |

## 6. 执行序与回滚

B0 装载/环境冒烟 → B1 核心+单测全绿 → B2 主 Run refs-01（smoke-01 先行）→ B3 split-half sh-01 → B4 图+登记回填 → B5 trellis-check + spec 更新 + 提交。各块产物独立落 run_id，回滚=弃 run_id 重跑，旧 Run 原口径不动（口径变更纪律）。详见 implement.md。
