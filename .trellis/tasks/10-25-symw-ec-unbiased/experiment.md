# 实验日志 — 10-25-symw-ec-unbiased

> 每次「跑批/训练/评估/统计」追加一个 Run 块：跑前先写「目的+配置」，跑后同会话补「结果+结论」，不留空块。
> 格式契约见 `.trellis/spec/research/experiment-log.md`；产物只落本任务声明的变体目录（`.trellis/spec/research/path-isolation.md`）。

## Run 000 — 脚本最小适配 + 静态检查（S0，算法零改动）（2026-10-25）

### 目的与假设

- prd R1/R4：为 symw 代理脑 EC 主跑做**最小脚本适配**——`ec_mild_unbiased.py` 加
  `--symw` 旗标（默认 False，旧调用面逐行不动）、`recompute_conn_v4.py` 加
  `--task-label`（默认 = 现常量串）。假设 = symw 训练直用
  FlowVelocityNetExitGNN（`flow_exit_gnn_symw.py:58`，零架构差异）⇒ 复用
  `--flow-exit-gnn` 的模型构建 + strict load 链路即可，σ*/τ/conn/probe/bisection
  计算路径零改动。

### 配置

- 代码：`models/latent-flow-v1/ec_mild_unbiased.py`（+58/−5 行）、
  `models/latent-flow-v1/recompute_conn_v4.py`（+7/−1 行）
- `ec_mild_unbiased.py` 四落点：①argparse `--symw`（action=store_true 默认关）；
  ②`_caliber_asserts` 守卫：与 `--flow-exit-gnn` 互斥断言 + symw 须配
  `10-25-latent-flow-exit-gnn-symw` run_dir ckpt；③模型构建
  `_model_args(a.flow_exit_gnn or a.symw)`（架构侧等价，SymWAdam 仅 optimizer
  投影不进 ckpt）；④payload `elif a.symw:` 元数据分支（method_str 注明 sym-w
  variant surrogate；surrogate_block task/ckpt_stage2_best 0.8054@E144/variant 含
  SymWAdam 投影+λ0.2 二因素/`"symw": True` 键；surrogate_caveat 含 A_eff 无向
  方向性构造受限 + λ/sym-W 两因素 + medoid/real std 0.858/0.848 + θ 待拍板 +
  PROVISIONAL 四要素）+ hparams `"symw": bool(a.symw)` 键；另 summary/gccorr 的
  caveat_note/surrogate_name 字符串条件化（元数据，计算零改动）
- `recompute_conn_v4.py`：`--task-label`（默认 = 原常量串 "10-24-exitgnn-ec-
  unbiased Run 003 (v4 remap + dual descriptive EC x GC)"，旧调用面等价）；v4
  公式/REFS 双参考/断言组逐行零改动
- 环境：npi 本机 conda（`/root/miniconda3/bin/python`），静态检查无需 GPU

### 结果

- `python -m py_compile` 两脚本 PASS；`--help` 出现 `--symw` 与 `--task-label`。
- diff 自审：`--symw` 仅出现在 argparse/守卫/模型构建落点/元数据分支/hparams 五段
  （+docstring 增量注记）；probe（SignalProbeSite）/bisection 两阶段循环/
  compute_global_tau/sigma_star_with_global_tau/conn 公式/npz 落盘**逐行零改动**；
  旗标关时 `a.flow_exit_gnn or a.symw` ≡ `a.flow_exit_gnn`，行为逐位等价
  （运行时证据 = Run 000b 对拍）。

### 结论

- S0 完成，适配面收敛在元数据/守卫层，计算路径未触碰；放行 Run 000b 干跑双证据。
- 下一步：Gate G1（strict load + 逐位对拍）通过后进 Run 001 主跑。

## Run 000b — 工程干跑双证据：旗标关逐位对拍 + --symw 链路干跑（Gate G1）（2026-10-25）

### 目的与假设

- ①旗标关等价对拍（运行时证据）：S0 改后默认 w500 路径与在档 w500 先例干跑
  **逐位等价**；②`--symw` 干跑：symw ckpt 经 FlowVelocityNetExitGNN strict load
  全链路可用、守卫/元数据/断言链生效。任一失败 ⇒ Gate G1 停查根因，不进主跑。

### 配置

- 代码：`models/latent-flow-v1/ec_mild_unbiased.py`（Run 000 改后版本）
- 口径：非正式干跑 1 例（s1）× 8 源 × log[0.01,0.48]×4 网格，其余默认
  （q=0.5/σ_ref=0.3/seed 42/30 trial 全量/bf16/cudnn.deterministic）
- 两跑：①旗标关 + 默认 w500 ckpt → `outputs/10-25-symw-ec-unbiased/scratch_equiv/`
  （与 w500 在档 `outputs/10-24-mild-w500-ec-unbiased/dryrun_mini/subject_0.json`
  对拍 max|Δ|=0 逐位断言）；②`--symw --task-label 10-25-symw-ec-unbiased` + symw
  ckpt（`outputs/10-25-latent-flow-exit-gnn-symw/run_symw_fold0/checkpoints/
  fold0_mild_flow_best_graph_mexhat.pt`）→ `outputs/10-25-symw-ec-unbiased/
  dryrun_symw/`；附守卫负例（exit-gnn 旧 ckpt + --symw 应被拦）
- 命令：`OMP_NUM_THREADS=8 /root/miniconda3/bin/python models/latent-flow-v1/
  ec_mild_unbiased.py --mode main --subjects 1 --sources 8 --n-grid 4 --seed 42
  --out-dir outputs/10-25-symw-ec-unbiased/scratch_equiv`（②加 `--symw
  --task-label 10-25-symw-ec-unbiased --ckpt <symw best>`，out-dir 换 dryrun_symw）

### 结果

- ①旗标关等价对拍（scratch_equiv vs w500 在档 dryrun_mini，subject_0）：六矩阵
  （sigma/connectivity/raw_response/mean_diff/sign/signed_connectivity，8×128）+
  tau_global（1.6919e-04）+ q/σ_ref + diagnostics **全部逐位相等 max|Δ|=0.0
  （bitwise_equal=True）**；hparams 仅增申报新键 flow_exit_gnn:False / symw:False，
  顶层 task 字段未写（空串语义正确），surrogate.task 保持 w500 原值——运行时证据
  落档：S0 改动旗标关时行为逐位等价（AC-R1 证据）。
- ②--symw 干跑（dryrun_symw）：FlowVelocityNetExitGNN 构建生效（1,276,258 参数 =
  exit-gnn 同架构面，w500 基座 1,243,234），strict load 通过（无 missing/unexpected
  keys 报错）；τ=1.317e-04、interior 52.2%、@min 14.9%、@max 32.8%、
  P(resp>τ)=0.500 精确；payload 核对 18/18 PASS（task=10-25-symw-ec-unbiased、
  surrogate 块五键含 symw:true、caveat 四要素齐、hparams.symw=true、norm
  d184e647…/manifest 47987909… SHA 断言过、ckpt_path 指向 symw run_dir）；
  curves.npz 落盘 (8,4,128) 无 NaN。
- 守卫负例两枚：exit-gnn 旧 ckpt + `--symw` ⇒ AssertionError「--symw 须配 symw
  run_dir 的 ckpt」（真实 exit=1，断言先于 mkdir 无残留目录）；`--symw +
  --flow-exit-gnn` 同置 ⇒ 互斥断言拦下。
- 产物：`outputs/10-25-symw-ec-unbiased/{scratch_equiv,dryrun_symw}/subject_0.json
  (+subject_0_curves.npz)`。

### 结论

- Gate G1 通过：适配正确、默认面运行时逐位等价、symw strict load + 守卫/元数据/
  断言链全生效；放行 Run 001 主跑。
- 下一步：Run 001 正式口径主跑（8 例 × 30 trial × 24 网格 × 128 源全量）。

## Run 001 — GPU 主跑：无偏 μ=0 global-τ EC × symw 代理脑（正式口径）（2026-10-25）

### 目的与假设

- 用户指令主实验：以 symw stage2 best（0.8054@E144 早停收敛，W torch.equal 级
  对称）为代理脑，无偏 μ=0 σ·ξ 扰动 + 两阶段 global-τ（q=0.5@σ_ref=0.3，
  log[0.01,0.48]×24）计算 fold0 test 8 例 × 30 trial 的 σ*/conn（128×128）。
- 假设（对照 w500 46.9 min / exit-gnn 51.2 min 单例 6.40 min / interior 60.3% /
  52.2% / dyn 2.62×）：同架构面前向开销 ⇒ 单例 ≈6.4 min 参照、总预算 ≈50 min；
  **A_eff 无向 ⇒ EC 方向判读按无向口径重设预期**（对称性诊断 S4 实证）。

### 配置

- 代码：`models/latent-flow-v1/ec_mild_unbiased.py`（Run 000b 验证版，git head 随 payload）
- 代理脑：`outputs/10-25-latent-flow-exit-gnn-symw/run_symw_fold0/checkpoints/
  fold0_mild_flow_best_graph_mexhat.pt`（best val 0.8054@E144，W 严格对称）
- 参数：**全默认**——q=0.5、σ_ref=0.3、log[0.01,0.48]×24、n_steps 10、batch 32、
  seed 42、bf16 autocast、μ_bias 关闭（断言 μ_bias None）、τ 分位含对角（房内
  先例口径）、断点续跑（json 存在即跳过）；旗标 `--symw --task-label
  10-25-symw-ec-unbiased`
- 环境：npi 本机 RTX 4080 SUPER 32 GB（启动前空闲已验 1 MiB/32760 MiB）；预算 ≈50 min
- 数据：Mild Depression fold0 test 8 例（MDD s1,5,7,10 / HC s21,25,27,30）× 30 trial
  全量；norm/manifest SHA 断言链照旧（d184e647…/47987909…）
- 命令：`cd /root/npi && OMP_NUM_THREADS=8 nohup /root/miniconda3/bin/python
  models/latent-flow-v1/ec_mild_unbiased.py --mode main --symw --task-label
  10-25-symw-ec-unbiased --ckpt /root/npi/outputs/10-25-latent-flow-exit-gnn-symw/
  run_symw_fold0/checkpoints/fold0_mild_flow_best_graph_mexhat.pt --out-dir
  /root/npi/outputs/10-25-symw-ec-unbiased/ec_main >
  /root/npi/outputs/10-25-symw-ec-unbiased/ec_main.log 2>&1 &`（后台；断点续跑语义：
  中断后原命令重发）

### 结果

- 启动确认（2026-10-25 17:0x 后台起跑，本块跑后回填最终结果）：python PID 8073
  存活、GPU 2946 MiB / util 100%；log 首屏口径行
  `[main] paradigm: global-tau UNBIASED (mu=0, sigma*xi) | q=0.5 sigma_ref=0.3
  grid=log[0.01,0.48]x24 | subjects=[1, 5, 7, 10, 21, 25, 27, 30] | device=cuda`；
  manifest/norm SHA 断言通过（执行已越过两断言到达 paradigm 打印）、
  FlowVelocityNetExitGNN 1,276,258 参数 strict load 无报错；log =
  `outputs/10-25-symw-ec-unbiased/ec_main.log`。
- **8/8 全部成功**（断点续跑未触发；总时长 **52.1 min**，单例 6.51–6.52 min，与预算
  ≈50 min 参照一致），逐例 stdout（log = `outputs/10-25-symw-ec-unbiased/ec_main.log`）：

  | # | 被试 | 组 | tau | interior | @min | @max | P(resp>τ) | elapsed |
  |---|------|----|-----|----------|------|------|-----------|---------|
  | 0 | s1  | MDD | 1.322e-04 | 54.2% | 15.2% | 30.6% | 0.500 | 6.52 min |
  | 1 | s5  | MDD | 6.093e-05 | 66.4% | 2.2%  | 31.4% | 0.500 | 6.51 min |
  | 2 | s7  | MDD | 6.603e-05 | 56.2% | 12.8% | 30.9% | 0.500 | 6.51 min |
  | 3 | s10 | MDD | 8.578e-05 | 68.8% | 2.9%  | 28.4% | 0.500 | 6.51 min |
  | 4 | s21 | HC  | 8.655e-05 | 71.1% | 1.6%  | 27.4% | 0.500 | 6.51 min |
  | 5 | s25 | HC  | 7.129e-05 | 55.9% | 12.4% | 31.7% | 0.500 | 6.51 min |
  | 6 | s27 | HC  | 5.732e-05 | 71.7% | 1.6%  | 26.7% | 0.500 | 6.51 min |
  | 7 | s30 | HC  | 6.396e-05 | 69.0% | 1.8%  | 29.1% | 0.500 | 6.51 min |

- **P(resp>τ)=0.500 逐例精确**（8/8，q=0.5 分位构造校准通过）；interior 54.2–71.7%
  （粗均值 ≈64.2%），高于 w500 先例 interior 均值 60.3%（区间 42.3–66.5%）且下尾
  明显抬升（本跑最低 54.2% vs w500 最低 42.3%）；τ 量级 5.7e-05–1.3e-04 与两先例
  同阶。产物 `ec_main/subject_{0..7}.json(+_curves.npz)` 逐例落档。

### 结论

- 主跑完成（AC-R3）：8/8 例成功、P(resp>τ)=0.500 精确、凭据链随 payload
  （ckpt/旗标/SHA/seed/git head）；symw 代理脑同架构面前向开销与 exit-gnn 同量级
  （单例 6.51 min vs exit-gnn 6.40 min 参照），52.1 min 在预算内。
- interior 分布较 w500 抬升、与 exit-gnn（40.6–64.1%，均值 53.5%）相比整体更高——
  σ* 解更多落在网格内部而非钳边，精确组级判读（dyn_ratio/方差分解）交 Run 002。
- 下一步：Run 002 离线组级 summary（三方并置）。

## Run 002 — 离线组级 summary（S3）（2026-10-25）

### 目的与假设

- prd R5：对 Run 001 主跑 8 例离线汇总——重映射不变性断言 8/8、跨被试稳定性矩阵、
  方差分解（行源/列靶/残差）、dyn_ratio、诊断组均，产出 summary json+npz；与
  w500 / exit-gnn 两先例同口径**三方并置**判读。假设：symw 代理脑 interior 更高 ⇒
  σ* 分布更居中，dyn_ratio 与方差分解形态介于/异于两先例，数值如实登记。

### 配置

- 代码：`models/latent-flow-v1/ec_mild_unbiased.py` `--mode summary --symw
  --task-label 10-25-symw-ec-unbiased`（summary 走与 main 相同的离线读取路径；
  --symw 仅条件化 surrogate_name/caveat_note 元数据，计算零改动）
- 输入：`outputs/10-25-symw-ec-unbiased/ec_main/subject_{0..7}.json`（Run 001 产物）
- 输出：`outputs/10-25-symw-ec-unbiased/summary/`（group_summary.json + group_mean.npz）
- 环境：npi 本机 CPU 离线；对照（只读）：w500
  `outputs/10-24-mild-w500-ec-unbiased/summary/`、exit-gnn
  `outputs/10-24-exitgnn-ec-unbiased/summary/`
- 命令：`cd /root/npi && OMP_NUM_THREADS=8 /root/miniconda3/bin/python
  models/latent-flow-v1/ec_mild_unbiased.py --mode summary --symw --task-label
  10-25-symw-ec-unbiased --main-dir /root/npi/outputs/10-25-symw-ec-unbiased/ec_main
  --out-dir /root/npi/outputs/10-25-symw-ec-unbiased/summary`

### 结果

- 跑通（stdout：`n=8 | interior 64.2% | dyn_ratio_med 2.5x | P(resp>tau) 0.500 |
  row_share 0.031 | stab rho 0.425`）；产物
  `outputs/10-25-symw-ec-unbiased/summary/group_summary.json`（+ group_mean.npz）。
- **重映射不变性断言 8/8 PASS**（1/(σ*+1e-4) Spearman 逐例 PASS）；P(resp>τ)=0.500
  组内 std=0（8/8 精确）；nonmono_pair_frac ≈0.9922（与两先例同值——网格口径共性）；
  σ 中位 offdiag 0.3023（先例均 0.303，同阶）。
- 组级：interior **64.2%±7.4**（54.2–71.7，三者最高）；@min 6.3%±6.0、@max
  29.5%±1.9；dyn_ratio_median **2.55±0.41**（1.98–3.14）；conn_mean_offdiag
  **−0.378**±0.175、conn_std_offdiag 0.897±0.310；方差分解（group-mean 矩阵）
  **行源 0.031 / 列靶 0.753 / 残差 0.216**；跨被试稳定性 offdiag Spearman 均值
  **0.425**（min 0.074 / max 0.723）。
- **三方并置**（同口径；w500 / exit-gnn 先例只读对照，θ 不同属跨代理脑描述性并置）：

  | 指标 | w500 | exit-gnn | **symw（本跑）** |
  |------|------|----------|------------------|
  | interior 均值 | 60.3% | 53.5% | **64.2%** |
  | @min / @max 均值 | 6.0% / 33.7% | 12.3% / 34.2% | **6.3% / 29.5%** |
  | dyn_ratio_median | 2.62 | 1.79 | **2.55** |
  | 方差分解 行/列/残差 | 0.037/0.641/0.322 | 0.010/0.907/0.083 | **0.031/0.753/0.216** |
  | 跨被试稳定性 ρ̄ | 0.576 | 0.655 | **0.425** |
  | conn_mean_offdiag | −0.357 | −0.119 | **−0.378** |
  | conn_std_offdiag | 0.907 | 1.245 | **0.897** |

### 结论

- Run 001 主跑组级判读（AC-R5）：symw 形态**最贴近 w500**（dyn 2.55 vs 2.62、conn
  均值 −0.378 vs −0.357、@min 低钳边占比同低），方差分解居两先例之间且残差份额
  0.216 高于 exit-gnn（0.083）——σ* 仍以列靶全局感受性为主，行源份额低（0.031）。
- 差异点：跨被试稳定性 0.425 为三者最低（w500 0.576 / exit-gnn 0.655）且被试间
  方差分解形态分化（HC s21/s27 残差 0.37/0.40 vs MDD s1/s7 残差 0.06/0.08）——
  EC 模式跨被试一致性弱于两先例；单 seed provisional，不作组间判读。
- 下一步：Run 003 v4 重映射 + 双 GC 对照 + 对称性诊断（方向性 caveat 实证）。

## Run 003 — conn v4 重映射 + 双 GC 逐被试对照 + 对称性诊断（S4，纯离线）（2026-10-25）

### 目的与假设

- prd R4/R6/R7 三件：①`recompute_conn_v4.py` 对 Run 001 8 例做 v4 重映射
  （conn = −(ln σ* + 0.5·ln 2πe) + 5/7）+ 三断言（raw Spearman 逐位不变 / 全正 /
  abs≡raw）8/8；②EC v4 × 双 GC 参考（v1 mild40 1–45 Hz 固定阶 8 + v2 mild40
  1–125 Hz AIC 4–25）offdiag 16256 对 aligned/flipped × Spearman/Pearson 逐被试
  + group + group_mean（= **用户核心交付**）；③新写对称性诊断
  `symmetry_diag_symw.py`：逐被试 conn v4 矩阵 M vs Mᵀ（offdiag Pearson/Spearman +
  相对不对称度 ‖M−Mᵀ‖_F/‖M‖_F）——实证 A_eff 无向对 EC 方向性的构造限制。
- 假设：SymWAdam 每步 W←(W+Wᵀ)/2 ⇒ flow 通道耦合严格对称 ⇒ EC 矩阵近对称
  （corr(M,Mᵀ)→1、不对称度极小）⇒ 方向判读按构造坍缩为无向，与有向 GC 的相关
  上限受限；aligned≈flipped（方向信息≈零）。

### 配置

- 代码（**两者本批零改动/仅新增**）：`models/latent-flow-v1/recompute_conn_v4.py`
  （上批已加 `--task-label`；REFS 双参考常量不动）+ 新增
  `models/latent-flow-v1/symmetry_diag_symw.py`（独立后处理小脚本）
- ①命令：`cd /root/npi && OMP_NUM_THREADS=8 /root/miniconda3/bin/python
  models/latent-flow-v1/recompute_conn_v4.py --main-dir
  /root/npi/outputs/10-25-symw-ec-unbiased/ec_main --out-dir
  /root/npi/outputs/10-25-symw-ec-unbiased/recompute_v4 --task-label
  10-25-symw-ec-unbiased`；输出 recompute_v4/subject_*.json +
  gccorr_dual_summary.json
- ②对称性诊断：读 ec_main/subject_*.json，σ* float64 重算 conn v4（M）与 v1 房内
  口径两档；offdiag 16256 对 corr(M[off], Mᵀ[off])（Pearson+Spearman）+
  ‖M−Mᵀ‖_F/‖M‖_F（v4 为主读出，v1 比值并置 + 口径注记：常数平移改变 ‖M‖ 不改
  变 ‖M−Mᵀ‖ ⇒ 两口径比值不同、相关系数不受影响）；输出
  `outputs/10-25-symw-ec-unbiased/symmetry_diag/symmetry_diag_summary.json`
- 参考对照（只读）：exit-gnn 先例
  `outputs/10-24-exitgnn-ec-unbiased/recompute_v4/gccorr_dual_summary.json`
  （v1 +0.149±0.168 / v2 −0.010±0.085）；w500 先例 v1 ρ −0.099 / raw +0.118
  ——并置为**跨代理脑描述性并置**（θ 不同，非复现性判读）
- 环境：npi 本机 CPU 离线（OMP_NUM_THREADS=8）

### 结果

- **①v4 重映射**：跑通 1.4 s；断言 **40/40 PASS**（8 例 × [v4↔v1 raw Spearman
  单调 1e-9 / 全正 / abs≡raw 逐位] + 8 × 2 双参考 Pearson 平移不变性）——即
  三断言 **8/8 PASS**；v1 json float32 舍入 max|Δ| = 2.69e-07（精度伪影随档登记，
  先例同量级）。产物 `recompute_v4/subject_{0..7}.json` + `gccorr_dual_summary.json`。
  （溯源注记：summary 的 `naming` 串为脚本先例常量（exit-gnn variant surrogate
  措辞，本批 recompute_conn_v4.py 零改动）；symw 权威凭据 = `task` 字段
  10-25-symw-ec-unbiased + 逐例 json 随档拷贝的 surrogate 块（symw:true）。）
- **追加修正（2026-10-25 同会话，用户指令解除 recompute_conn_v4.py 本批零改动
  限制、仅此一处）**：`naming` 串参数化——新旗标 `--surrogate-name`（默认 =
  先例常量串，旧调用面零影响），本任务传 "sym-w variant surrogate (SymWAdam
  W←(W+Wᵀ)/2, λ0.2; best 0.8054@E144)"，与交付要求③（模型版本注明）对齐；
  随后**同命令 + 新参数重跑本离线步**：断言 40/40、8 份逐例 json 与
  refs_blocks/formula_facts/asserts/task 全部 deep-equal（仅 timestamp/naming
  元数据差）——纯元数据改动数值逐位不变；diff 自审 = argparse 常量 + naming
  f-string 两段，计算路径零触碰（py_compile PASS）。
- **②双 GC 逐被试相关（用户核心交付；abs≡raw 逐位相同，下表 raw 口径）**：

  v1_1_45_fixed8（1–45 Hz 固定阶 8），offdiag 16256 对：

  | 被试 | 组 | ρ aligned | r aligned | ρ flipped |
  |------|----|-----------|-----------|-----------|
  | s1  | MDD | +0.0099 | +0.0711 | +0.0730 |
  | s5  | MDD | +0.3052 | +0.3752 | +0.2575 |
  | s7  | MDD | +0.0659 | +0.1033 | −0.0009 |
  | s10 | MDD | +0.2594 | +0.2322 | +0.0299 |
  | s21 | HC  | +0.1269 | +0.2329 | +0.1612 |
  | s25 | HC  | −0.1347 | −0.1592 | +0.1383 |
  | s27 | HC  | +0.3304 | +0.3925 | +0.0909 |
  | s30 | HC  | +0.1125 | +0.1307 | +0.0620 |
  | **group** | | **ρ +0.1344±0.1587** | **r +0.1723±0.1787** | ρ +0.1015±0.0822 |
  | **group_mean** | | **ρ +0.1806** | **r +0.2338** | — |

  v2_1_125_aic4_25（1–125 Hz AIC 4–25）：

  | 被试 | 组 | ρ aligned | r aligned | ρ flipped |
  |------|----|-----------|-----------|-----------|
  | s1  | MDD | +0.0429 | −0.0333 | −0.0378 |
  | s5  | MDD | −0.0060 | −0.0141 | −0.3445 |
  | s7  | MDD | +0.0468 | +0.0403 | −0.2158 |
  | s10 | MDD | +0.0411 | +0.0105 | +0.0367 |
  | s21 | HC  | +0.0797 | +0.0759 | +0.0281 |
  | s25 | HC  | −0.1860 | −0.2166 | +0.1590 |
  | s27 | HC  | +0.0640 | −0.0019 | −0.3437 |
  | s30 | HC  | +0.0441 | −0.0550 | −0.2598 |
  | **group** | | **ρ +0.0158±0.0851** | **r −0.0243±0.0879** | ρ −0.1222±0.1928 |
  | **group_mean** | | **ρ +0.1363** | **r +0.0510** | — |

- 先例并置（**跨代理脑描述性并置**，θ 不同非复现性判读）：v1 块 symw
  **+0.1344±0.1587** vs exit-gnn +0.149±0.168（几乎同位）vs w500 ρ −0.099 /
  raw +0.118（基本不相关）；v2 块 symw **+0.0158±0.0851** vs exit-gnn
  −0.010±0.085（同位近零）——symw 与 exit-gnn（同 GNN 架构面）GC 相关形态
  高度一致，弱正 v1 / 近零 v2。
- **③对称性诊断**（`symmetry_diag_symw.py` 跑通 0.3 s；py_compile PASS；
  产物 `symmetry_diag/symmetry_diag_summary.json`）——逐被试 conn v4 M vs Mᵀ：

  | 被试 | Pearson(M,Mᵀ) | Spearman(M,Mᵀ) | asym v4 | asym v1 |
  |------|----------------|-----------------|---------|---------|
  | s1  | −0.1372 | −0.1811 | 1.1709 | 1.4759 |
  | s5  | −0.0043 | +0.0333 | 1.0046 | 1.1672 |
  | s7  | −0.0718 | −0.0202 | 1.1311 | 1.4221 |
  | s10 | +0.0275 | +0.1123 | 0.9852 | 1.1363 |
  | s21 | +0.0365 | +0.0813 | 0.8938 | 0.9479 |
  | s25 | −0.1936 | −0.3032 | 1.2070 | 1.4952 |
  | s27 | +0.2363 | +0.3178 | 0.8158 | 0.9057 |
  | s30 | +0.0895 | +0.1771 | 0.9381 | 1.0517 |
  | **组均值** | **−0.0021±0.1349** | **+0.0271±0.1971** | **1.0183** | **1.2002** |

  （组均值矩阵：asym v4 0.6792；两口径差矩阵范数逐位相等断言 PASS；
  v1/v4 比值差异 = 常数平移改 ‖M‖ 不改 ‖M−Mᵀ‖，口径注记随 json 档。）
- 补充只读探针（三代理脑同诊断，旧树只读）：corr(M,Mᵀ) 均值 w500 +0.052 /
  exit-gnn −0.015 / symw −0.002；asym 均值 0.989 / 1.077 / 1.018——**三跑同型**。

### 结论

- **AC-R4/R6 达成**：三断言 8/8 PASS、双 GC per-subject + group + group_mean 齐。
  用户核心交付判读：symw EC 与 v1 GC 呈**微弱正 aligned 相关**（ρ +0.1344±0.1587，
  8 例中 6 正，s5/s10/s27 驱动，s25 负；group_mean ρ +0.1806）；与 v2 GC **近零**
  （+0.0158±0.0851）。与 exit-gnn 先例几乎同位（v1 +0.149 / v2 −0.010）⇒ GC
  相关形态由 GNN 架构面承载、对 W 对称化不敏感；描述性口径（单 seed、无多重
  比较控制），v1 块名义 t≈2.4 不作确证性宣称。
- **AC-R7 达成，且判读锚点取反向分支**：**未观察到**「corr→1 且 asym 极小」——
  实测 corr(M,Mᵀ)≈0（−0.002±0.135）、asym≈1.02，EC 方向性**未**坍缩为无向；
  三代理脑同型（探针）⇒ 不对称是本口径（σ* 列靶主导 + 逐通道编码/出口投影
  非对称）的量级典型，非 symw 特有效应。**A_eff 无向 caveat 实证收窄**：W 对称
  只约束 GNN 消息传递通路，EC 的方向结构由非对称逐通道编码/出口 + 数据依赖
  σ* 承载，仍存在（v1 aligned +0.134 > flipped +0.102，方向特异信号弱但非零）。
  与有向 GC 相关弱的瓶颈在**整体对齐弱**（ρ~0.1–0.2），不是对称性坍缩。
- provisional 状态不变（λ/sym-W 两因素同变未消融、θ 待拍板、单 seed）；
  下一步：S5 README + 研究记忆回写（对称性结论按本 Run 收窄表述回写）。
