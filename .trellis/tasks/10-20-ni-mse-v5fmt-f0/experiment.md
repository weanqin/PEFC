# 实验日志 — NI-MSE-50-q0.90 执行（θ\* = v5-fmt-01）

> 每次「跑批/训练/评估/统计」追加一个 Run 块：跑前先写「目的+配置」，跑后同会话补「结果+结论」，不留空块。
> 格式契约见 `.trellis/spec/research/experiment-log.md`；产物只落本任务声明的变体目录（`.trellis/spec/research/path-isolation.md`）。

## Run 000 — S1 基线登记 + S2 结构落地（2026-10-20）

### 目的与假设

- S1：主 run（Run 002）启动前的基线登记——git 快照基线、θ\*（v5-fmt-01）与 refs 只读指纹固化，任何后续只读契约复核以此为参照。
- S2：`models/ni-mse-v1/` 结构落地（design.md §2 八文件 + 七项单测），验证装载遮蔽契约与 R2 恒等断言机械正确；**本 Run 不跑数据侧主链路**（冒烟属 Run 001、主 run 属 Run 002）。

### 配置

- 代码：`models/ni-mse-v1/`（全新目录，本 Run 落地）+ 只读依赖 `models/gpfm-mamba-v1/`（sample/data/gpr_prior/mamba）、`models/gpfm-mamba-v5-fmt/`（config/model/graph）、θ\* 产物 `outputs/gpfm-mamba-f0/v5-fmt-01/`、refs `refs/granger_bivariate_spectral/edrafmt48-cont-zscore-fs250-fixed8-f1-125-nf65/`。
- git 快照基线（S1.1，2026-10-20 登记，如实）：
  - HEAD = `07ff5033bdf1049d0458d7f3145b1f9087fb9920`（main，2026-10-20 19:06:38 +0800）；
  - 脏路径（既有，本任务不 commit、不改动）：`M .trellis/tasks/archive/2026-10/10-19-latent-flow-edra-fmt-train/experiment.md`、`M models/latent-flow-v1/latent_flow/baseline/data_utils.py`、`M models/latent-flow-v1/latent_flow/viz/connectivity.py`、`?? .trellis/tasks/10-19-gpfm-mamba-v4b-gnn-f0/`、`?? .trellis/tasks/10-20-ni-mse-v5fmt-f0/`、`?? models/gpfm-mamba-v4b-gnn/`、`?? models/gpfm-mamba-v5/`（`models/gpfm-mamba-v5-fmt/` 已入库，工作树干净）。
- θ\* 指纹（S1.1，只读）：
  - ckpt `outputs/gpfm-mamba-f0/v5-fmt-01/checkpoints/gpfm-mamba-v5-fmt-best.pt` SHA256 = `d5ec13716225c1983ca4de66f6a71aa853d070d482837b30e71e6f914c8446b1`；config_hash = `a64c0d416756`（代码 pin 同值）；epoch = 28 / best_val = 0.7024236700547634（ckpt 字段为准，experiment.md 旧记「ep29」为 1 基显示差异）；EMA shadow 73 tensors（decay 0.999）。
  - 同目录 `channel_mean.npy` SHA256 = `fa607b59188510c20b12288323ea1285c2f57e2c376c0d73ec41d31d5bc0b113`、`channel_std.npy` SHA256 = `8e668cf1e0136aa73764b00a690984e23dbd85add3c06ce5362909222c4ca58e`、`config.yaml` SHA256 = `c1f539d722ef9f277ffc6b41fa60cf523fac785a43888598a6f2d60ed77d7455`。
- refs 指纹（S1.1，只读）：`refs/granger_bivariate_spectral/edrafmt48-cont-zscore-fs250-fixed8-f1-125-nf65/SHA256SUMS` 复核 `sha256sum -c` 全量通过（50/50 OK，退出码 0）。
- 数据：fold0 test 10 例 = `[10,11,12,17,24,28,30,44,61,69]`（manifest `models/gpfm-mamba-v1-fmt/splits/fold0_status_ok_seed42_val4.json` 只读，34/4/10；勘误已登记——ni-mse-design.md §1.1 的 v4 过渡名单作废）；数据根 `/root/autodl-tmp/edra_processed_fmt`（250 Hz，60001 样本/例 → 930 窗/例 @384/128/stride64）。
- 种子树（master=2026，分支整数映射落 manifest）：ε_ref=10 / ε_null=11 / ξ_pert=12 / η=13 / win-null=20 / win-pert=21。
- 环境：python 3.12.3 @ autodl-container-csa5j6rngg-c17e0ff7（本地 GPU 容器）；torch 2.8.0+cu128 / cudnn 91002 / numpy 2.3.2 / scipy 1.18.1 / RTX 4080 SUPER 32GB；BLAS guard 五变量 setdefault=8。
- 命令（占位，S3/S4 启动时按 implement.md 执行）：
  - 单测：`cd /root/npi/models/ni-mse-v1 && python3 -m pytest tests/ -q`
  - 冒烟（Run 001）：`python3 /root/npi/models/ni-mse-v1/run.py --stage null --run-id ni-mse-v5fmt-smoke --config w_null=8 w_pert=4 k_resp=4 k_null=4 b_ref=4 c_arms=0.5 subjects=10`（+ 同参数 perturb/report 两期）
  - 主 run（Run 002）：`python3 /root/npi/models/ni-mse-v1/run.py --stage {null|perturb|report} --run-id ni-mse-v5fmt-01`（三期断点续跑，nohup + logs/ 重定向）

### 结果

- S2 结构落地（2026-10-20 完成）：`models/ni-mse-v1/` 八文件齐——`_blas_guard.py` / `thostar.py`（预占 sys.modules[config/model/graph]→import sample 七项 `__file__` 断言 + config_hash pin + epoch/best_val pin + EMA shadow 装载 + `gen_lowpass_hz=None` 断言）/ `windows.py` / `noise.py` / `stages.py` / `report.py` / `run.py` / `tests/test_ni_mse.py`（七项）。
- 单测：`python3 -m pytest tests/ -q` **7 passed**（含 GPU 段：真 θ\* 装载 + R2 同批逐位恒等 + `sampler.gen_lowpass_hz is None`）。
- 回归：`models/gpfm-mamba-v5-fmt` tests **11 passed** 不变（只读零改动复核）。
- 实现钉定登记（design §1b 未尽细节的本实现口径，判读与复跑沿用）：
  1. ε_ref 固定网格 = **逐被试一份** `spawn_key=(10, sid)` → (B_ref,C,F) 全窗复用（「固定 ε 网格」= Ŷ_ref 为 x_past 的确定性泛函，跨窗可比）；η_ref 同构 `(13, sid)`。
  2. ε_null/η_null = 逐窗 `(11, sid, w)` / `(13, sid, w)` 一次物化 (K_null,C,F)（k 经抽取次序，不进 spawn_key——§1b-1 仅钉 ξ/η 四元组）；响应侧 fresh ε = `(11, sid, w, a, k)`（ε_null 支，无 c ⇒ 跨 c 臂 CRN 配对）。
  3. ξ/η（扰动侧）严格按 `(12, sid, w, a, k)` / `(13, sid, w, a, k)` 四元组——**行组织 = (w,k) 展平进 B、K=1**（x_past 每 (w,k) 行独立注入 ξ，past 条件化不变量成立；B·K=50×32=1,600 ≤ 8,192）。
  4. R2 断言 = 每例首窗、a=0、**每 c 臂**各一次（首调用行 0/1 = P(X,a,0)·扰动路径 vs 干净基线，同批同 eps/eta，`torch.equal`；行 0/1 落同一 chunk_bk=1024 分块内）。
  5. τ_b split-half = 池化 (W×K) 样本 w-major 展平后**奇偶二分**各算 Q90；触发统计量 = median_b(|Δτ_b|/τ_b)（>10% 触发 R3），max_b 同步登记。
  6. R4 幅度判据在 **z 空间**执行：训练折 z-score 后逐通道 std ≡ 1（dataloader 口径构造使然）⇒ 阈值 = 10×1.0；`channel_std.npy`（物理单位 2.32–5.14）由 thostar 装载并随 manifest 登记供 σ_p^phys 汇报导出，不进判据（design §1.2：ŝ_a 仅物理换算非算子成分）。
  7. 被试窗口全局索引（0..929，loader shuffle=False 确定性序）为种子元组中 w 的取值；窗子集索引与映射表全量落 manifest。
- git 复核：除 `models/ni-mse-v1/`、`.trellis/tasks/10-20-ni-mse-v5fmt-f0/` 外零新改动（既有脏路径清单见上，未动）。

### 结论

- S1/S2 完成：θ\* 与 refs 指纹固化、装载契约（七项 `__file__` + hash pin + `gen_lowpass_hz=None`）与 R2 同批恒等在真 θ\* 上机器验证通过，结构就绪。
- 下一步：S3 冒烟（Run 001，小参数全链路 + 速率实测）→ S4 主 run 三期（Run 002）。

## Run 001 — S3 冒烟 `ni-mse-v5fmt-smoke`（【跑前登记】2026-10-20 20:16）

### 目的与假设

- 小参数全链路三期（null→perturb→report）在真 θ\*（v5-fmt-01）+ 真数据（s10）上顺序跑通，暴露 S2 单测覆盖不到的管线级 bug（续跑断言/产物契约/manifest 字段/CLI 键名）。
- 假设：R2 同批恒等断言在管线内嵌路径（每例首窗、a=0、每 c 臂）逐位通过；R1 无 NaN/Inf；τ_b (62,)/S,G (62,62)；refs_s10.npz 逐被试配对命中；manifest.json 种子树/窗索引/参数/θ\* 指纹齐全。
- 速率实测：null/perturb 期每（窗·条）耗时 → 外推主 run 预算，与 design §5 的 ~1.5 h 对照（偏差 >2× 需原因分析）。

### 配置（跑前钉定）

- 代码：`models/ni-mse-v1/`（HEAD `07ff5033bdf1`，S2 落地态未动）；θ\*/refs/家族代码只读（指纹见 Run 000）。
- 冒烟参数（全部经 `--config` 覆盖，**不改代码缺省值**）：1 例 = s10（fold0 test 首例）× W_null=8 × W_pert=4 × K_resp=4 × K_null=4 × B_ref=4 × c 臂 {0.5}。
- 产物根：`outputs/10-20-ni-mse-v5fmt-f0/ni-mse-v5fmt-smoke/`（`null/ perturb/ report/ manifest.json logs/`）；run-id `ni-mse-v5fmt-smoke`。
- 环境：本地 GPU 容器（RTX 4080 SUPER 32GB，torch 2.8.0+cu128），前台直跑 + tee 日志。
- 命令（三期顺序，各期退出码与墙钟随跑后补记）：
  - `mkdir -p outputs/10-20-ni-mse-v5fmt-f0/ni-mse-v5fmt-smoke/logs`
  - `python3 /root/npi/models/ni-mse-v1/run.py --stage null --run-id ni-mse-v5fmt-smoke --config w_null=8 w_pert=4 k_resp=4 k_null=4 b_ref=4 c_arms=0.5 subjects=10`
  - `python3 /root/npi/models/ni-mse-v1/run.py --stage perturb --run-id ni-mse-v5fmt-smoke --config w_null=8 w_pert=4 k_resp=4 k_null=4 b_ref=4 c_arms=0.5 subjects=10`
  - `python3 /root/npi/models/ni-mse-v1/run.py --stage report --run-id ni-mse-v5fmt-smoke --config w_null=8 w_pert=4 k_resp=4 k_null=4 b_ref=4 c_arms=0.5 subjects=10`

### 结果（跑后补记 2026-10-20 20:23）

**三期执行与退出码**（首次执行 → 修复后重跑，日志齐全于 `logs/`）：

| 期 | 首次 | 重跑（修复后） | 关键日志原文 |
|---|---|---|---|
| null | exit 0，墙钟 13s | exit 0（s10 已存在，跳过——**null 续跑契约实测生效**） | `[null] s10 完成 \| 窗 8 \| R4 排除 0 \| 池/通道 32 \| split-half median 15.620% max 48.730% \| R3 触发 \| 1s` |
| perturb | exit 0，墙钟 16s | exit 0，compute 3.7s | `[perturb] s10 完成 \| R2 断言 1/1 通过 \| R4 排除 0 (a,窗) 项 \| 4s` |
| report | **exit 1（冒烟暴露 bug，见下）** | exit 0，墙钟 6s | 失败原文（`logs/report.fail1.log` 保留）：`AssertionError: report 前置断言：c 臂 0.05 不在 perturb 参数 [0.5]` |

**冒烟暴露 bug 与修复（如实登记，非静默）**：
1. **report 四臂强制断言 vs 单臂冒烟配置冲突**：`report.py::report_stage` 原版 `for c in [C_MAIN, C_NEG, *C_SENS]: assert c in c_arms` 使 run.py docstring/Run 000 登记的单臂冒烟（`c_arms=0.5`）report 期必挂。修复 = 前置断言只强制主读出臂 0.5；阴性对照/敏感性臂缺场时对应节段在 report.json 显式登记 `{"skipped": "c=… 臂不在本次 c_arms …（缩参冒烟口径，节段缺场）"}`（不静默）；**四臂全在场（主 run）时各节段数值与原版逐字段一致**。未改 Params 缺省值。修复后单测 `7 passed`（11.0s）重跑全绿。
2. **stages.py 续跑粒度 docstring 与实现不符**（重跑时实测暴露：perturb 未跳过已存在 `10_c0.5.npz` 而是整例重算 4s）：docstring 原文「perturb 按 {sid}_c{c}.npz 存在即跳过（§1b-7）」与实现不符——修正 docstring 为实际行为（null 按例跳过；perturb 整例确定性重算覆写，CRN 下逐位一致正确性无虞；§1b-7 本只约束 manifest 原子写/report 前置/R2 拒跑，均成立）。仅注释改动零逻辑变化，单测复跑 `7 passed`。**S4 前可选项**：若主 run 想要 perturb 断点续跑（崩例重算代价 ~1.7 min/例/臂），需补 per-例跳过+有限性校验——留主会话拍板，本次不动。

**检查项逐条结论**（证据原文/直接核验）：
- **R2 同批恒等（真 θ\*，管线内嵌）**：✅ 日志 `R2 断言 1/1 通过`；`perturb/10_c0.5.npz` 字段 `r2_checked=1`；产物根 `.r2_failed` 不存在（核验 False）。断言位置 = 每例首窗 a=0 每 c 臂，P(X,a,0) 扰动路径 vs 干净基线同批同 eps/eta 逐位 `torch.equal`。
- **R1 无 NaN/Inf**：✅ R1 硬失败守卫全程未触发（触发即非零退出）；产物直接核验全有限：`N (4,62,8)`、`tau_b (62,)`、`tau_q95 (62,)`、`y_ref_wpert (4,62,128)`、`R_bar (62,62)`、`R_medk (62,4,62)`、`S (1,1,62,62)` finite=True 全过。
- **形状**：✅ `tau_b shape=(62,)`、`R_bar/S (62,62)`、`G_main (1,62,62) bool`（report_matrices.npz 直读核验）。
- **refs 逐被试配对**：✅ report.json `per_subject_main_c` 键 = `['10']`，refs_s10.npz `ec (62,62)` finite 命中（对角 0 / offdiag ≥ 0 由单测④+report 消费路径双重核验）。
- **manifest.json 完整**：✅ 九区齐全——params（含 `cli_overrides` 与 resolved 区分、n0=25）/ seed_tree（master=2026、分支整数映射 6 支、9 条 spawn_key 公式、w 语义、c 无关性）/ theta_star（ckpt SHA256 `d5ec1371…` 与 Run 000 指纹一致、config_hash `a64c0d416756`、epoch 28 / best_val 0.70242…、channel mean/std SHA、split manifest、test 10 名单）/ env（host/python 3.12.3/torch 2.8.0+cu128/BLAS 8/`cudnn_deterministic: false`——**勘误（trellis-check 建议1，2026-10-20）**：该 false 为快照时序伪象——env_snapshot 在 thostar 装载前捕获 torch 进程缺省值；thostar.py 实际按 cfg 设 `cudnn_deterministic=True`（v5-fmt config 缺省即 True），**实际执行态比 manifest 记录更严格**，与 R2 逐位通过自洽；首期伪快照经 setdefault 固化至三期，产物无数值影响）/ windows（wnull 8 + wpert 4 全局索引，wpert ⊆ wnull 核验 True）/ stages 三期 done。注：重跑后 `stages.null.elapsed_s=0.0`（跳过路径计时），null 真实 compute 1s 只在首跑 `logs/null.log`。
- **冒烟数值登记（缩参口径，不作任何判读）**：`s_offdiag_median=0.507`、`g_pass_rate=0.000529`（2/3782 边）、pearson_direct −0.0348 / spearman_direct −0.0182 / flip ≈ direct（−0.032/−0.018）、dist_partial −0.0130；阴性对照/敏感性节段 = 缺场登记（单臂口径）。**R3 触发（sh_median 15.6% > 10%）为冒烟尺度伪迹**：池化 32 样本/通道（4K×8W）下 Q90 由次序统计量承载、split-half 各半仅 16 样本——主 run 池化 12,800/通道无此问题，主 run 的 R3 旗标不受冒烟观察先验影响。
- **窗子集抽样子证**：s10 wnull = `[8390,8405,8440,8451,8568,8811,8826,9088]`（ds 受试者序 = 分层划分置换序，s10 全局基座 ≈ 8370）、wpert = `[8390,8405,8568,8826]` ⊂ wnull，升序原序保持。

**速率实测与主 run 外推（S3.2，算术全文）**：
- 实测：null 8 调用（B·K=8 行）/ 1s → **0.125 s/调用**；perturb 62 调用（B·K=18 行）/ 3.7s（manifest elapsed_s）→ **0.0597 s/调用**；每期固定开销（θ\* 装载+loader）≈ 墙钟−compute ≈ 12s/期；report 墙钟 6s（纯 CPU）。
- 外推下界（每调用成本持平的开销地板模型）：null 2,000 调用（10×200，128 行）× 0.125s ≈ **4.2 min**（design ~5 min，比值 0.83×，吻合）；perturb 2,480 调用（10×62×4，1,602 行）× 0.0597s ≈ **2.5 min**（design 4×17=68 min，比值 0.04×）；合计 ≈ 0.12 h —— **比 design 1.5 h 快 12.4×（>2× 偏差，登记原因）**。
- 偏差原因分析：冒烟单次调用 8/18 行为 kernel 启动/固定开销主导，每调用地板无法承载主 run 89×（perturb）/16×（null）行的计算增长；反之每行模型（3.32 ms/行 × 2,480×1,602 行 ≈ **3.7 h**，慢 2.5×）因 18 行摊薄开销而每行成本虚高 89×——**两模型分别在快/慢两个方向失真 ≥2×，design 1.5 h 落在两者之间，冒烟尺度不可再细化**。佐证：design 内涵满行每调用成本 null 0.15 s@128 行（= 冒烟 1.2×，null 侧 GPU 开销主导、缩放温和，可信）；perturb 1.645 s@1,602 行（= 冒烟 27.6×，处于「89× 行 → 27.6× 时」的亚线性并行区间，可信）；另计 numpy 侧物化（主 run 每 a 2×(1600,62,128) normals ≈ 0.7s × 62×10 ≈ 7 min）后 perturb ≈ 75 min，全链 ≈ 1.3–1.4 h，与 design ≈1.5 h 自洽。
- 处置：**design §5 的 1.5 h 保留为主 run 规划预算**（冒烟无证伪力）；S4 开跑后以首例完成日志行（s10：null 200 调用 + perturb 248 调用）为活体速率检查点——若首例 perturb 单臂耗时 >2× design 内涵（>3.4 min/例/臂）即停下排查再续跑。

### 结论（跑后补记 2026-10-20 20:23）

- S3 冒烟**通过**：三期 exit 0 全链跑通，R2 真 θ\* 同批恒等、R1 全有限、τ_b (62,)/S,G (62,62)、refs s10 命中、manifest 九区齐全——管线机械正确，主 run（Run 002）无阻塞。
- 冒烟暴露并处置 2 项：report 四臂断言 bug（已修+单测全绿+重跑通过）；stages.py 续跑粒度 docstring 失实（已修正为实际行为；perturb 断点续跑为 S4 前可选项，留主会话拍板）。
- R3 触发为冒烟尺度伪迹（池化 32/通道），不预示主 run；冒烟数值（|r|≈0.03、S 中位 0.51 等）无判读意义，仅证管线产出有限数值。
- 速率：冒烟无法证伪 design 1.5 h（开销地板 0.12 h / 每行模型 3.7 h 双向失真包夹）；按 design 预算执行 S4，首例日志为活体检查点。
- 下一步：S4 主 run `ni-mse-v5fmt-01` 三期（Run 002，启动确认沿 v5-main 先例——执行期用户确认项，非本次派发范围）。

## Run 002 — S4 主 run `ni-mse-v5fmt-01`（【跑前登记】2026-10-20）

### 目的与假设

- 在真 θ\*（v5-fmt-01）+ fold0 test 10 例上执行 NI-MSE-50-q0.90 主实例化全参数三期（null → perturb → report 链式），产出 τ_b/split-half、四 c 臂 R̄/S/G 与 refs 判读四件套——AC2/AC3 的主产物 Run。
- 假设：R2 每例硬门通过（真 θ\* 已在单测⑤与冒烟管线内嵌两度逐位验证）；R1 无 NaN/Inf；split-half <10%（R3 不触发——主 run 池化 12,800/通道，冒烟 15.6% 为池 32 的尺度伪迹）；总墙钟 ~1.5 h 量级。

### 配置（跑前钉定）

- **主参数（全部缺省钉定值，不带任何 `--config` 覆盖）**：W_null=200 窗/例 × K_null=64（池 12,800/通道）｜W_pert=50 ⊂ W_null × K=32｜B_ref=64｜c 臂 {0.05, 0.1, 0.3, 0.5}（0.5 主读出 / 0.05 阴性对照 / {0.1,0.3} 敏感性）｜n₀=25（fmt@250）｜10 例 = [10,11,12,17,24,28,30,44,61,69]｜种子树 master=2026（ε_ref=10/ε_null=11/ξ_pert=12/η=13/win-null=20/win-pert=21，Run 000 登记）。
- **θ\* 指纹（只读，Run 000 固化）**：`outputs/gpfm-mamba-f0/v5-fmt-01/checkpoints/gpfm-mamba-v5-fmt-best.pt` SHA256 `d5ec13716225c1983ca4de66f6a71aa853d070d482837b30e71e6f914c8446b1`；config_hash `a64c0d416756`；epoch 28 / best_val 0.7024236700547634；EMA shadow 装载；`gen_lowpass_hz=None`。
- 产物根：`outputs/10-20-ni-mse-v5fmt-f0/ni-mse-v5fmt-01/`（`null/ perturb/ report/ manifest.json logs/`）；run-id `ni-mse-v5fmt-01`。
- **启动方式**：bash 驱动脚本 `logs/driver.sh` 三期**链式**（每期显式重定向 `> logs/{stage}.log 2>&1`、退出码检查；任一期非零退出即停并写产物根 `.stage_failed` 标记），整体 `nohup bash driver.sh > logs/driver.log 2>&1 &` 后台。
- 等价启动命令（driver.sh 内逐期形态）：
  - `python3 /root/npi/models/ni-mse-v1/run.py --stage null    --run-id ni-mse-v5fmt-01 > .../logs/null.log 2>&1`
  - `python3 /root/npi/models/ni-mse-v1/run.py --stage perturb --run-id ni-mse-v5fmt-01 > .../logs/perturb.log 2>&1`
  - `python3 /root/npi/models/ni-mse-v1/run.py --stage report  --run-id ni-mse-v5fmt-01 > .../logs/report.log 2>&1`
- **处置背景（如实登记，不得省略）**：
  1. **启动确认未获应答**：S4 启动前经 AskUserQuestion 向用户确认未获应答；主会话按 workflow 1.4.3 以「立项时『确认开始』已覆盖 S1–S6 全链」+ 本 Run 只读 θ\*/refs、产物自隔离可删目录回滚的性质，以最佳判断推进启动。
  2. **perturb 续跑补丁先于启动落地（主会话择「先补再启」）**：Run 001 遗留可选项「perturb per-例断点续跑」于本次启动前补齐——`stages.py` perturb 期新增 per-例跳过：该例全部 c 臂 `perturb/{sid}_c{c}.npz` 存在且 R_bar 有限、k_resp/窗子集一致、`r2_checked == |c 臂|` ⇒ 整例跳过（对齐 null 期既有 skip）。**R2 与跳过的取舍**：R2 为每例硬门，本实现中 R2 在例内 perturb 循环 a=0 处**先于全部产物落盘执行**并随产物落 `r2_checked` 标记 ⇒ 产物存在且含完整标记即证明该例 R2 已通过，已含标记的例整例跳过（重装进程后未完成/无标记例仍全跑含 R2）；任何校验不满足（缺臂/NaN/参数漂移/r2 不完整）⇒ 整例重算。缺省参数零改动。验证：单测新增 2 项（伪造产物 → 二次调用跳过、产物逐字节不变、零数据集/零模型访问 + 守卫面 5 负例）合计 **9 passed**；真 θ\* 端到端复核 = 冒烟产物副本二次调用 `exit 0` + 日志「已存在…跳过」+ 产物 SHA256 逐位不变（`99208acc…`，scratch 目录 `ni-mse-v5fmt-resumecheck` 用毕即删）。
- **预算**：design §5 保留 ~1.5 h 单卡（四 c 臂 ~4×17 min + null ~5 min + 装载/IO）；**活体检查点阈值 = 首例 perturb 单臂 3.4 min/例/臂**（design 内涵 1.7 min 的 2×，Run 001 结论沿用）——超阈且外推超预算 2× 即停下排查（不杀进程，交主会话决策）。
- 环境：同 Run 000/001（本地 GPU 容器 RTX 4080 SUPER 32GB，torch 2.8.0+cu128；启动时 GPU 空闲 1 MiB/0%）。

### 结果（活体检查点补记 2026-10-20 20:56——run 仍在进行，S4.3 report 由 driver 链式自动续跑，终局数值留后续补记）

- **启动**：20:35:35 `nohup bash logs/driver.sh > logs/driver.log 2>&1 &`，**driver PID 26548**；日志 = `logs/{driver,null,perturb,report}.log`；启动行回显参数全缺省钉定值（无任何 `--config` 覆盖）、subjects=[10,11,12,17,24,28,30,44,61,69]。
- **null 期完成**：20:35:35 → 20:41:26，**rc=0**，墙钟 ≈ 5.9 min（design ~5 min，吻合）。逐例 33–35s（s10 首例 34s）。10 例数值摘要：
  - split-half `median_b(|Δτ_b|/τ_b)` 逐例 = s10 1.019% / s11 1.145% / s12 0.914% / s17 2.000% / s24 1.825% / s28 1.069% / s30 0.915% / s44 1.042% / s61 1.360% / s69 1.552%（跨被试 mean 1.284%、range 0.914–2.000%）；`max_b` 全部 ≤ 5.736%——**R3 全 10 例未触发**（触发线 10%；design 预计 ~0.67–1.5% 兑现，冒烟 15.6% 确为池 32 尺度伪迹）。
  - R4 退化窗排除：s12 1 窗（池 12,736/通道）、s61 4 窗（池 12,544），其余 8 例 0 排除（排除+日志政策内）。
  - τ_b 全有限（s10 示例：min 0.477 / median 1.062 / max 1.467）；R1 全程未触发。
- **perturb 期活体检查点（判停门核验）**：20:41:26 开始；**首例 s10 完成 697s**，日志原文 `[perturb] s10 完成 | R2 断言 4/4 通过 | R4 排除 0 (a,窗) 项 | 697s`，四臂产物 `10_c{0.05,0.1,0.3,0.5}.npz` 落盘。
  - **折算 2.90 min/例/臂 < 检查点阈值 3.4 min/例/臂（design 内涵 1.7 min 的 1.71×）——未触发判停**。
  - 外推：perturb 全期 ≈ 10×697s ≈ 116 min，加 null 5.9 min + report ~1 min ≈ **全程 ~2.0–2.1 h** = design 1.5 h 的 **~1.4×**——判停门要求「>3.4 min/例/臂 **且** 外推 >2× 预算」两条件同时满足，此处仅预算单条件超出 1.4×，**不判停**，偏差如实登记（原因待跑后从 manifest elapsed 逐例数核；冒烟双模型包夹结论中 design 落两模型之间，实测落在其上界侧）。
  - R2 硬门：s10 4/4 通过；产物根无 `.r2_failed` / `.stage_failed` 标记（20:54 核验）。
- **状态**（20:54:57 最后轮询）：s11 计算中（GPU 100%），driver/日志链正常；主 run 预计 ~22:40 前后三期完毕。

### 结果（终局补记 2026-10-20 22:50——三期 22:37 由 driver 链式自动完毕，无人工干预）

**三期完成时刻线**（`logs/driver.log` 原文核验）：

| 期 | 起止（driver.log） | 墙钟 | manifest elapsed |
|---|---|---|---|
| null | 20:35:35 → 20:41:26 | 5.9 min | 338.9 s |
| perturb | 20:41:26 → 22:37:35 | ~116 min | 6,954.5 s（= 115.9 min；10 例逐例 691–699 s，首例 s10 697 s） |
| report | 22:37:35 → 22:37:49 | 14 s | 0.4 s（纯 report 计算；其余为装载） |

终局行原文：`[driver] 2026-10-20 22:37:49 all stages done (null/perturb/report)`——三期 rc=0 链式完毕。

**完整性核验（R1–R6 逐条，产物与日志直接复核）**：
- **R2（硬门）**：10 例 × 4 c 臂全过——`logs/perturb.log` 恰 10 处「R2 断言 4/4 通过」（s10–s69 逐例各一处）；产物根无 `.r2_failed` / `.stage_failed`（核验不存在）。
- **产物齐全**：perturb 40 npz（10 例 × `{sid}_c{0.05,0.1,0.3,0.5}.npz`）+ null 10 npz + `report/report.json`（generated 2026-10-20T22:37:48）+ `report/report_matrices.npz`（S (4,10,62,62) / G_main / G_q95_main / tau_b / tau_q95 / S_group）+ manifest.json 三期 done。
- **R1**：未触发（null/perturb/report 三日志无 NaN/Inf 记录；report 前置断言全有限通过）。
- **R3**：全 10 例未触发——split-half median 逐例 0.914–2.000%（跨被试 mean 1.284%），`max_b` 全部 ≤ 5.736%（与活体检查点段登记及 report.json `flags.r3_splithalf` 逐字段一致；`r3_any_triggered=false`）。
- **R4**：null 期 s12 排除 1 窗（池 12,736/通道）、s61 排除 4 窗（池 12,544），其余 8 例 0；perturb 期 s61 排除 249 (a,窗) 项、其余例 0（manifest `r4_count` 与日志逐例一致；排除项 NaN 进 R_medk、经 nanmedian 链不进 R̄）。
- **R5**：旗标被试 {11,12}（`r5_subject_outliers=[11,12]`；s_offdiag_median 极值 s11 0.4705（max）/ s12 0.4132（min），组中位 0.4512，偏离 >3×MAD）——旗标不排除，判读段（Run 003）知悉登记。
- **R6**：零超敏边（`r6_hypersensitive_edges.count=0`）。

**预算核对与偏差归因（终局）**：
- perturb 实测 ~116 min；全程 20:35:35→22:37:49 = 2 h 02 min ≈ **2.04 h = design §5 预算 1.5 h 的 1.36×**。
- 判停门复核：逐例 691–699 s ⇒ 2.90 min/例/臂 < 阈值 3.4 min/例/臂（条件一不满足）；预算超幅 1.36× < 2×（条件二不满足）——两条件不同时满足，**未判停为门规则的正确执行**。
- 偏差归因（主会话会话内详细分析，此处登记结论）：纯生成速率与 pilot 一致——null 期 1.17 ms/条 vs Run 007 pilot 1.04 ms/条（+13%，同量级）；差额来自**可交换性契约的 per-(w,a,k) 四元组 RNG 成本**：每例 ξ/η/fresh-ε 三流 × (50 窗 × 62 源 × 32 重复) = 297,600 ≈ 29.8 万 RNG 流构造 + ~16.1 亿高斯 CPU 抽样（GPU 空转等待 numpy 物化）+ 248 次 per-(a,c) 调用固定开销（62 源 × 4 c 臂）。

### 结论（Run 002 终局）

- S4 三期全链 rc=0 完毕：R2 硬门 10 例 × 4 臂全过；R1/R3/R6 干净；R4 政策内处置（null 5 窗 + perturb s61 249 项）；R5 {11,12} 旗标如实登记。主产物（τ_b + split-half、四 c 臂 R̄/S/G、refs 判读四件套、manifest 完整化）齐全落 `outputs/10-20-ni-mse-v5fmt-f0/ni-mse-v5fmt-01/`。
- 预算 2.04 h（design 的 1.36×）如实登记：归因于可交换性契约的 RNG 构造成本（生成速率本体与 pilot 一致），非模型前向劣化；判停门两条件不同时满足，续跑正确。
- 判读（AC4 判读四件套 + AC5 结构预测，正负结果同权）留 Run 003 块登记。

## Run 003 — S5 判读汇总：NI-MSE-50-q0.90 判读（c 主臂 0.5 × θ\* v5-fmt-01 × fmt fold0 test 10 例）（2026-10-20）

### 目的与假设

- S5.1：对 Run 002 主产物按 design §4 / prd AC4–AC5 口径落档判读四件套 + 结构预测（数值表 + 如实判读，正负结果同权登记）。
- S5.2：与 v4a-fmt 旧路径（10-17 任务 AC6 未获结构收敛）对照讨论一段（θ\* 不同、路径不同，只作背景非等价比较）。
- 本 Run 为**纯登记 Run**：无新计算、无新种子、无代码改动；读出材料只读。

### 配置（读出材料）

- 数据源（只读）：Run 002 report 期产物 `outputs/10-20-ni-mse-v5fmt-f0/ni-mse-v5fmt-01/report/report.json`（generated 2026-10-20T22:37:48）+ `report/report_matrices.npz`（S ≡ R̄/τ_b 超限比，(4 c 臂, 10 例, 62, 62)；G_main/G_q95_main；S_group）+ null/perturb npz（数值复核用）。
- 口径：逐被试配对主口径 n=10（offdiag 3,782 对；Pearson/Spearman direct 主、flip（ec.T）副、距离偏 Spearman = pairwise_d OLS 残差化）+ 组级副口径（10 例逐元素中位 vs 同 10 例 refs 均值；vs 48 例全均值敏感性）；refs 只读复用；不对生成信号重算 GC（v1.4 拍板）。

### 结论（阴性，主读出）

该 θ\* × 该路径 × 钉定参数（c≤0.5、前 0.1s 窗）下，**未获 EC 结构收敛证据；EC_pert 不可用于连接推断**——G 门控 10 例被试全部 0/3,782 通过（Q95 带亦 0%），跨通道响应中位数仅为噪声地板的 ~45%（S offdiag median 0.450±0.015）。

### 结果 — 判读四件套（AC4）

**逐被试数值表**（report.json `per_subject_main_c`，c=0.5 主读出臂；G 通过率与 Q95 带份额 10 例全部 0.0000）：

| sid | Pearson direct | Spearman direct | Pearson flip | Spearman flip | 距离偏 Spearman | S offdiag 中位 |
|---|---|---|---|---|---|---|
| 10 | 0.0150 | −0.0215 | 0.0216 | −0.0197 | −0.0124 | 0.4446 |
| 11 | 0.0016 | 0.0176 | 0.0016 | 0.0211 | 0.0072 | 0.4705 |
| 12 | 0.1467 | 0.1306 | 0.1451 | 0.1300 | 0.1371 | 0.4132 |
| 17 | 0.0331 | 0.0039 | 0.0321 | 0.0007 | 0.0262 | 0.4533 |
| 24 | 0.0820 | 0.0662 | 0.0814 | 0.0650 | 0.0575 | 0.4576 |
| 28 | 0.0512 | 0.0641 | 0.0519 | 0.0607 | 0.0596 | 0.4488 |
| 30 | 0.0771 | 0.0733 | 0.0778 | 0.0734 | 0.0826 | 0.4621 |
| 44 | 0.0297 | 0.0292 | 0.0326 | 0.0376 | 0.0148 | 0.4491 |
| 61 | 0.0930 | 0.0861 | 0.0898 | 0.0840 | 0.0790 | 0.4487 |
| 69 | 0.0591 | 0.0372 | 0.0591 | 0.0352 | 0.0425 | 0.4546 |

1. **逐被试主口径**：Pearson direct mean 0.0588±0.0428（range 0.0016–0.1467）/ Spearman direct mean 0.0487±0.0443（range −0.0215–0.1306）；组级副口径 vs 同 10 refs 均值 Pearson 0.0966 / Spearman 0.0752，vs 48 例全均值 0.0993 / 0.0850——微弱正相关（s12 为唯一 |r|>0.1 被试，正负同权登记不剔除）。
2. **flip 检查**：Pearson flip 0.0593 vs direct 0.0588（差 <0.001）；direct−flip≈0 ⇒ **方向信息≈零**，相关由对称（列结构/靶侧地板异质）分量承载——AC5 结构预测「flip≈direct」兑现。
3. **距离偏 Spearman** mean 0.0494（vs 直接 0.0487）——微弱相关**非**几何距离承载（与 |A_eff|×refs Run 006 的距离承载模式不同）；但幅度太低，不构成结构证据，仅登记。
4. **阴性对照与敏感性**：c=0.05 与 c=0.5 的 S offdiag median 0.447 vs 0.450、G 均全 0；c∈{0.1,0.3} Spearman 0.046 / 0.047——**offdiag 剂量-响应全平**（跨通道传导在 0.1s 窗的 MSE 贡献 <1%：raw offdiag R̄ 组级池化中位 0.4855→0.4887 即 +0.66%；report.json 逐被试均值口径 0.4470→0.4503 即 +0.73%；s10 单例 +0.23%——各聚合口径均 <1%，勘误：初稿「≤0.2%」系 s10 单例口径）。

### 结果 — 机制性核验（自响应）

- S 对角中位 c=0.05→0.5 仅 0.447→0.476（逐被试均值口径；raw 自响应 MSE 组级 +7.3–7.5%，s10 单例 +8.7%；勘误：初稿「0.442→0.477 / +8.5%」系 s10 单例口径）——注入经 R2 证明逐位进入条件路径，但 GPR 条件先验对 past 加性白噪声的投影衰减 + 速度网络弱响应，使 0.1s 内**连自通道扰动都低于 fresh-ε 全后验变异性构成的地板**。
- 与 perturb-ec-v1（增益算子 + CRN RMS、自响应显著、v4a-fmt θ\*）不矛盾：本路径把全后验采样变异性嵌入响应度量——**可交换 null 的代价是灵敏度**（Δμ² 须超过 null Q90−Q50 ≈ 0.59 MSE 即 Δμ≳0.77 z 单位才能触发门）。

### 结果 — 设计勘误登记（重要，如实登记）

- ni-mse-design.md §4.2「c→0.05 时 G 通过率 → ~10%（Q90 名义水平）」与 §1.5-F 的中位聚合门（R̄ = median_k median_w 后再比 τ_b）**数学上不相容**——中位统计量越过同分布池 Q90 的 H₀ 概率≈0，10% 名义属「单样本 vs Q90」读法。
- 实测阴性对照 0% 通过率正是可交换 null 机器正确工作的表现（零假阳性 + 剂量平坦），但「~10%」预期线本身是设计文档内部不一致，AC4-④ 的该子项按勘误口径判读（非实现失败）。
- 此勘误不改 run 任何数值，只改判读预期线的归属。

### 结果 — 与 v4a-fmt 旧路径对照（S5.2，背景非等价比较）

- 旧路径（10-17 任务）在该 θ\* 家族上 slope×GC ρ̄≈0（弥散传导无特异性）；本路径阴性更强（门控不触发），并显性化「灵敏度-有效性权衡」——两个 θ\*、两种算子/度量，共同指向代理脑侧跨通道传导弱，而非某一 readout 的实现问题。

### 结果 — 边界

- c≤0.5 为钉定范围（更高 c 属训练分布外外推，未测）；0.1s 窗漏检慢传导（v1.1 用户拍板接受）；θ\* 为候选非合格代理脑（三项检验另立）；单 seed 读出（master 2026）。

## 收尾处置登记（2026-10-20，trellis-check 后）

- 检查结果：AC1–AC7 全过、无阻断；三项建议处置——①cudnn 快照伪象勘误已补 Run 001（见上，实际执行态 True 比 manifest 记录更严格）；②design §1b-6「R3 加密落 null-k128/」的**重跑机制未自动化**（触发检测/登记链完整——stages.py r3_triggered → npz/manifest/report 三处落档；K_null 加倍重跑需届时手工执行）——主 run 全 10 例未触发（sh_median 0.914–2.000% ≪ 10%），该 contingent 路径未行使，无产物影响，作为已知限制随档；③Run 002 启动确认未获应答按 1.4.3 最佳判断推进——Run 002 已如实登记处置依据，归档知悉即可。
- 单测终态：ni-mse-v1 `9 passed`（7 设计钉定 + 2 续跑补丁）；v5-fmt 回归 `11 passed` 不变。
