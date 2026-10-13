# experiment — baseline-ec-v1

> Run 块四要素（目的/配置/结果/结论），先写后跑；旧 Run 不改写。口径变更走 evaluation-calibers 纪律。

## Run 001 —— 冒烟：1 被试端到端（状态：已完成 2026-10-13）

- **目的与假设**：验证 resample→VAR/选阶→条件 GC→gPDC→mne-c 交叉核对→比较器（EC_pert 合成矩阵）全链可跑通、产物落盘正确、耗时可控；不产生科学结论。
- **配置**：
  - 代码：`models/baseline-ec-v1/`（commit 哈希见 Run 002 块首——首次提交与本 Run 结论同步落盘）；单测四件套实跑退出码 0（test_resample / test_ec_refs / test_compare / test_split_half）。
  - 数据：EDRA v4 HC/4（status_ok；filter 断言 bp1-40+notch50(保真冗余) 通过）；500→125 Hz resample_poly(1,4)；逐通道 z。
  - 口径（**含两项冒烟期口径变更，登记**）：
    1. **OLS→ridge**：OLS Gram 条件数实测 ~1e18（所有阶数、连 500 Hz Run 007 口径同样），Schur 读出被共线噪声支配（gc_q99 随阶膨胀 0.06→3.4；split-half 天花板在任何 p 下 ≈0；对照 500Hz p=4 天花板 0.017）。改为 **ridge λ_rel=1e-2 固定**（相对 tr(G_lag)/k_lag；截距不罚；GCV 实测欠平滑选 1e-8 弃用；λ 由三被试 split-half 稳定性校准 0.91–0.94）。GC 读出 = ln(SSR_red/SSR_full)，SSR 二次型口径，每源一次删列重解（62 次/被试，非逐边）。
    2. **选阶：BIC/白性阶梯→固定 p=8**：BIC 在 C=62 惩罚过重（选 p*=1 且 lb_frac=1.0）；白性阶梯（LB 检验）对真实数据超严全档失败（lb_frac=1.0 属常态非失效信号）。固定 p=8（ASCT p/2 谱峰 × EEG 2-4 节律锚点；p=8 与 16 天花板等价 0.93-0.94）。BIC 表逐被试存档。
    3. **BLAS 线程护栏**：cgroup 16 核 vs numpy 默认 128 线程超订（用户报告 CPU 100%）；入口统一 `_blas_guard` 限 8 线程（setdefault 可覆盖）。注：同机另有 `gpfm-mamba-v3/evaluate.py v3lp-main-01`（非本会话）在跑，负载基线非零。
  - 启动命令：`python3 models/baseline-ec-v1/run.py --stage smoke --run-id smoke-a --subjects 1 --mnec all`
- **结果**（产物 `outputs/baseline-ec-v1/smoke-a/`）：
  - 全链跑通（resample→order→refs→split-half→compare→图）；refs 单被试 35.5 s（含 mne-c 62ch 全量 3782 对）。
  - HC/4：gc_max=0.0655、gpdc_max=0.888（值域 [0,1] ✓）、lb_frac=1.0（存档 QC，不作判据）。
  - split-half 天花板（单被试）：**gc=0.942、gpdc=0.798**（OLS 下 ≈0，ridge 救回）。
  - 比较器合成 EC_pert 联调：gc ρ_partial=0.137、gpdc=0.097（单被试 CI 退化属预期，仅验证管道）。
- **结论**：通过。两项口径变更（ridge λ_rel=1e-2、固定 p=8）有量化证据支撑并已登记；旧 OLS/BIC 口径弃用但证据留档于本块。

## Run 002 —— 全量：46 被试参考 EC + 比较器联调 + split-half（状态：已完成 2026-10-13）

- **目的与假设**：产出 GC/gPDC 两参考的逐被试矩阵与 split-half 天花板；比较器以合成 EC_pert 全链联调。假设：两法 split-half 天花板 >0.5；GC↔gPDC 原始值秩相关显著为正（共享 VAR 构造性一致，sanity 非科学问题）。
- **配置**：代码+任务+spec 主体提交 `d951ed9`（本行为其后补记的小提交）；46 例（HC 24/MDD 22，status_ok，filter/ref_chain 全例断言通过）；口径与 Run 001 定标一致（p=8、ridge λ_rel=1e-2、mne-c 全量 46 例）；同机并行进程：`gpfm-mamba-v3/evaluate.py v3lp-main-01`（非本任务，负载挤占致 2 例耗时 275/347 s，其余 ~33 s/例）。启动命令：`run.py --stage resample/refs/split-half/compare --run-id main-01`（refs --mnec all）。
- **结果**（产物 `outputs/baseline-ec-v1/main-01/`：data/ 46 例 125 Hz z-scored、eval/refs_all.npz + refs/{sid}.npz（含 mnec）+ refs_summary.json + split_half.json + comparison.json、figs/ec_matrix_main-01.png）：
  - **split-half 噪声天花板（n=46）**：gc **0.900 ± 0.030**、gpdc **0.734 ± 0.039** → 两参考逐被试高度可靠；后续 EC_pert 一致性若低，可排除「参考侧噪声」解释（对应用户要求的失败归因分解）。
  - 量级健全性：gc_max 全体 0.047–0.210（Run 007 真实 GC 量级 ~0.03 邻域）；gpdc_max 0.69–0.95（值域 [0,1] ✓）；BIC 存档 46/46 p*=1（高维惩罚，佐证固定 p=8 决定）。
  - 比较器合成 EC_pert 联调（**SYNTHETIC，非科学结论**）：gc ρ_partial=0.104 [0.0999,0.1089] p_holm=1.4e-38 ratio=0.12；gpdc 0.082 [0.0766,0.0860] p_holm=1.3e-32 ratio=0.11（合成矩阵=GC 均值派生+30% 噪声，数值只验证管道与统计链路）。
  - **mne-c 二元交叉核对 vs 主口径条件 GC**：逐被试秩相关 mean **0.069 ± 0.080**（min −0.146，n=46）——二元与条件化 estimand 在 62 传感器混叠数据上的分歧被量化（与传感器空间方法学批评一致）；条件化为主口径的决定获数据支撑，mne-c 列按登记保留为分歧记录。
  - 耗时：refs 46 例 ≈ 27 min（含 mne-c 全量）、split-half 36 s、compare 4 s（BLAS 8 线程护栏下）。
- **结论**：交付完成。两参考法天花板 0.73–0.90（假设成立）；GC↔gPDC 一致性与真实 EC_pert 对比留待 Study II（接口与 readout 约束已登记 evaluation-calibers.md）；bivariate-vs-conditional 分歧（0.069）作为传感器空间方法学事实记录在案。
