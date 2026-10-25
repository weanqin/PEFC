# 实验日志 — <任务名>

> 每次「跑批/训练/评估/统计」追加一个 Run 块：跑前先写「目的+配置」，跑后同会话补「结果+结论」，不留空块。
> 格式契约见 `.trellis/spec/research/experiment-log.md`；产物只落本任务声明的变体目录（`.trellis/spec/research/path-isolation.md`）。

## Run 000 — 脚本元数据级适配 + 静态检查（S0，算法零改动）（2026-10-25）

### 目的与假设

- prd R1：为 f12 双折（fold1/fold2）symw 代理脑 EC 主跑做**元数据级三处适配**——
  `ec_mild_unbiased.py` 单文件。假设 = f12 训练直用 FlowVelocityNetExitGNN +
  `--symw` 既有链路（10-25-symw-ec-unbiased 先例），仅 norm SHA 折键控 / 守卫扩展 /
  payload f12 分支三落点需动，σ*/τ/conn/probe/bisection 计算路径逐行零改动。

### 配置

- 代码：`models/latent-flow-v1/ec_mild_unbiased.py`（+104/−32 行，git diff 随档）
- 三落点：①`EXPECTED_NORM_SHA` → `EXPECTED_NORM_SHA_BY_FOLD = {0: d184e647…,
  1: 0f10b153…, 2: a2c27a2d…}`（值 = f12 训练侧 configs/stage{1,2}_config.json
  norm_stats.mean_std_sha256 登记，fold1/2 两 stage 一致已核；断言处按 `a.fold`
  取值，缺 fold 键拒跑；payload provenance norm sha 同步按 fold 填断言值）；②
  `--symw` 守卫扩展：f12 run_dir（`10-25-latent-flow-symw-f12`）分支断言 ckpt
  文件名 `fold{k}_` 前缀 + fold∈{1,2} 与 `--fold` 锁定（防折-权重错配）；先例
  run_dir（`10-25-latent-flow-exit-gnn-symw`）收窄为仅 `--fold 0`（fold0 ckpt 配
  `--fold 1` 在装载前被拦）；③payload surrogate f12 分支（task =
  `10-25-latent-flow-symw-f12 run_symw_fold{k}`；ckpt_stage2_best fold1
  0.8264@E116 真早停 E146 / fold2 0.8433@E113 真早停 E143；variant 注明 f12
  双折训练 symw Run 001 同参跨折；caveat 含 fold-transfer θ 待用户 + 跨折并置
  描述性（各折 θ 不同实例）+ A_eff 无向注记引用 fold0 对称性诊断收窄结论
  （corr(M,Mᵀ) −0.002±0.135 / fro_asym≈1.02，非坍缩）+ PROVISIONAL）+ hparams
  显式 `"fold": int(a.fold)`（仅 f12 分支，fold0/旗标关 payload 键面不变）
- 环境：npi 本机 conda（`/root/miniconda3/bin/python`），静态检查无需 GPU

### 结果

- `python -m py_compile` PASS；`--help` 正常（新增调用面 `--fold {1,2}` + `--symw`
  + f12 ckpt，参数面零变化）。
- diff 自审：改动全部位于常量定义/`_caliber_asserts` 守卫/norm 断言取值/payload
  元数据字符串五段（+docstring 增量注记）；SignalProbeSite 构建、两阶段扰动循环、
  compute_global_tau、sigma_star_with_global_tau、conn 公式、npz 落盘**逐行零改动**；
  fold0 symw else 半支内容逐字保留（仅缩进+1）；旗标关时新逻辑不进任何分支。
- f12 norm SHA 值三方核对：本文件常量 = f12 configs 登记（fold1/2 × stage1/2 四处
  一致）= PRD 关键事实，逐位相等。

### 结论

- S0 完成，适配面收敛在元数据/守卫层，计算路径未触碰；放行 Run 000b 干跑双证据。
- 下一步：Gate G1（旗标关逐位对拍 + fold1 链路干跑）通过后 Run 001 主跑。

## Run 000b — 工程干跑双证据：旗标关 fold0 逐位对拍 + --fold 1 --symw 链路干跑（Gate G1）（2026-10-25）

### 目的与假设

- prd R2：S0 改动不触计算路径的**运行时证据**。①旗标关 + 默认 w500 ckpt + fold 0
  mini 跑，与在档 `outputs/10-24-mild-w500-ec-unbiased/dryrun_mini/subject_0.json`
  六矩阵 + tau_global 逐位对拍 max|Δ|=0；②`--fold 1 --symw` + f12 fold1 ckpt mini
  干跑：strict load + fold1 norm SHA 断言通过 + payload f12 元数据核对 +
  P(resp>τ)=0.500；③守卫负例：fold0 symw ckpt 配 `--fold 1` 须被守卫拦截。

### 配置

- 代码：`models/latent-flow-v1/ec_mild_unbiased.py`（Run 000 改后版本）
- 口径：非正式干跑 1 例 × 8 源 × log[0.01,0.48]×4 网格，其余默认（q=0.5/
  σ_ref=0.3/seed 42/30 trial 全量/bf16/cudnn.deterministic）；②fold1 test 首例
  s2 × f12 fold1 ckpt（`outputs/10-25-latent-flow-symw-f12/run_symw_fold1/
  checkpoints/fold1_mild_flow_best_graph_mexhat.pt`）
- 命令：①`OMP_NUM_THREADS=8 /root/miniconda3/bin/python models/latent-flow-v1/
  ec_mild_unbiased.py --mode main --subjects 1 --sources 8 --n-grid 4 --seed 42
  --out-dir outputs/10-25-symw-ec-f12/scratch_equiv`；②同参 + `--fold 1 --symw
  --task-label 10-25-symw-ec-f12 --ckpt <f12 fold1 best>`，out-dir 换 dryrun_f1
- 环境：npi 本机 conda @ RTX 4080S（4080S bf16 路径同 seed 逐位可复现 = w500/
  symw-fold0 两先例实测）

### 结果

- **①旗标关等价对拍**（scratch_equiv vs w500 在档 dryrun_mini，subject_0）：六矩阵
  （sigma/connectivity/raw_response/mean_diff/sign/signed_connectivity，8×128）+
  tau_global（1.691863e-04）**全部逐位相等 max|Δ|=0.0（bitwise_equal=True）**；
  q/σ_ref 相等——运行时证据落档：S0 守卫/norm 键控/元数据分支改动旗标关时行为
  逐位等价（R2 证据 1）。
- **②`--fold 1 --symw` f12 fold1 ckpt 干跑**（dryrun_f1）：FlowVelocityNetExitGNN
  构建生效（1,276,258 参数 = exit-gnn 同架构面，w500 基座 1,243,234），strict load
  通过（无 missing/unexpected keys 报错）；manifest fold1 = 28/4/8 断言过；到达
  paradigm 打印 ⇒ **fold1 norm SHA 断言通过**（0f10b153… 各折键控生效）；s2（fold1
  test 首例）τ=1.001e-04、interior 80.1%、@min 0.8%、@max 19.1%、**P(resp>τ)=0.500
  精确**；payload 核对 **20/20 PASS**（task=10-25-symw-ec-f12、surrogate.task=
  10-25-latent-flow-symw-f12 run_symw_fold1、ckpt_stage2_best=0.8264@E116 真早停
  E146、variant 含 f12 dual-fold training、caveat 四要素齐[fold-transfer θ 待用户/
  跨折 DESCRIPTIVE/A_eff 无向引用 fold0 对称性诊断收窄/PROVISIONAL]、hparams
  fold:1 + symw:true、norm/manifest SHA 断言过、ckpt_path 指向 f12 run_dir）；
  curves.npz 落盘 (8,4,128) 无 NaN。
- **③守卫负例两枚**（均真实 exit=1 + AssertionError，断言先于 mkdir 无残留目录）：
  fold0 symw ckpt（exit-gnn-symw run_dir）+ `--fold 1` ⇒ 「--symw 须配 symw run_dir
  的 ckpt（exit-gnn-symw 仅限 --fold 0；--fold 1 须配 …f12 的 fold1_ 前缀 ckpt）」；
  f12 fold2 ckpt + `--fold 1` ⇒ 「f12 ckpt 须配 --fold {1,2} 且文件名以 fold1_ 前缀
  开头（折-权重错配防护）」——前缀断言拦截生效。
- 产物：`outputs/10-25-symw-ec-f12/{scratch_equiv,dryrun_f1}/subject_0.json
  (+subject_0_curves.npz)`。

### 结论

- **Gate G1 通过**：适配正确、默认面运行时逐位等价、f12 fold1 strict load + 折键控
  SHA 断言 + 守卫/元数据链全生效；放行 Run 001 主跑。
- 下一步：Run 001 fold1 主跑（8 例 × 30 trial × 24 网格 × 128 源全量）；fold2 由
  主会话在 fold1 完成后启动（Run 002）。

## Run 001 — GPU 主跑：无偏 μ=0 global-τ EC × f12 fold1 symw 代理脑（正式口径）（2026-10-25）

### 目的与假设

- prd R3（折 1/2）：在 f12 fold1 symw 代理脑上以与 fold0 逐字相同的无偏噪声
  global-τ 口径计算 fold1 test 8 例 128×128 有向 EC（σ*/conn v1 落例、v4 后置
  Run 003 重映射）。假设 = 同口径跨折迁移可计算且逐例 P(resp>τ)=0.500 构造性
  自洽（阴性亦如实登记，负证据链惯例）。

### 配置

- 代码：`models/latent-flow-v1/ec_mild_unbiased.py`（Run 000 版本，git head 随
  payload）
- 代理脑：`outputs/10-25-latent-flow-symw-f12/run_symw_fold1/checkpoints/
  fold1_mild_flow_best_graph_mexhat.pt`（stage2 best 0.8264@E116，真早停 E146
  收敛；θ 候选判断待用户——EC 结论 provisional 随其联动）
- 参数：`--fold 1 --symw`、q=0.5（钉死）、σ_ref=0.3、σ 网格 log[0.01,0.48]×24、
  seed 42、Euler 10 步、batch 32、bf16 autocast、μ_bias 关闭、τ 分位含对角
- 数据：manifest `splits/mild_5fold_seed42.json`（SHA 47987909… 单文件全折共用）
  fold1 test 8 例 s{2,12,14,17,22,32,34,37} × 30 trial 全量；归一化 = fold1 train
  28 例前 500 点统计（SHA 对拍 0f10b153… 通过即跑，折键控）
- 环境：conda npi @ 本机 RTX 4080S 32GB；预算 ≈52 min（8 例 × ~6.5 min）
- 命令：`cd /root/npi && OMP_NUM_THREADS=8 nohup /root/miniconda3/bin/python
  models/latent-flow-v1/ec_mild_unbiased.py --mode main --fold 1 --symw
  --task-label 10-25-symw-ec-f12-fold1 --ckpt /root/npi/outputs/10-25-latent-
  flow-symw-f12/run_symw_fold1/checkpoints/fold1_mild_flow_best_graph_mexhat.pt
  --out-dir /root/npi/outputs/10-25-symw-ec-f12/fold1/ec_main >
  /root/npi/outputs/10-25-symw-ec-f12/fold1_ec_main.log 2>&1 &`（nohup 后台，
  json 逐例原子写 = 断点续跑）

### 结果

- 起跑确认（implement 会话返回时点，2026-10-25）：真实 python 子进程 **PID 16805**
  （nohup bash 包装 16803 之子，GPU 计算进程）；GPU 2936 MiB / util 92%；日志
  `/root/npi/outputs/10-25-symw-ec-f12/fold1_ec_main.log` 已达 paradigm 行
  （subjects=[2, 12, 14, 17, 22, 32, 34, 37]、q=0.5、log[0.01,0.48]×24、
  device=cuda）⇒ manifest SHA + **fold1 norm SHA（0f10b153…）断言通过**、
  FlowVelocityNetExitGNN（1,276,258 参数）构建 + strict load 完成。
- **跑后回填（2026-10-25，S3 会话）**：**8/8 例成功，52.2 min**（单例
  6.51–6.53 min 稳定，与 fold0 symw 6.51 min 逐字同量级）；**P(resp>τ)=0.500
  逐例精确**（q=0.5 构造性自洽）；逐例表（interior 范围 **67.8–79.5%**）：

  | 被试 | 组 | tau | interior | @min | @max | P(resp>τ) |
  |------|----|-----|----------|------|------|-----------|
  | s2 | MDD | 8.803e-05 | 75.1% | 0.8% | 24.2% | 0.500 |
  | s12 | MDD | 6.786e-05 | 75.9% | 0.8% | 23.3% | 0.500 |
  | s14 | MDD | 6.425e-05 | 75.0% | 0.8% | 24.2% | 0.500 |
  | s17 | MDD | 9.044e-05 | 73.7% | 2.9% | 23.4% | 0.500 |
  | s22 | HC | 1.067e-04 | 69.9% | 0.8% | 29.3% | 0.500 |
  | s32 | HC | 7.867e-05 | 67.8% | 3.4% | 28.8% | 0.500 |
  | s34 | HC | 1.154e-04 | 76.1% | 0.8% | 23.1% | 0.500 |
  | s37 | HC | 1.366e-04 | 79.5% | 0.8% | 19.7% | 0.500 |

- 凭据链核对（S3 会话逐例）：8/8 payload `task=10-25-symw-ec-f12-fold1`、
  `hparams.fold=1` + `symw=true`、`provenance.norm.mean_std_sha256=0f10b153…`
  （fold1 键控断言值）、manifest SHA 47987909…、ckpt_path 指向 f12 run_symw_fold1；
  产物 `outputs/10-25-symw-ec-f12/fold1/ec_main/subject_{0..7}.json(+_curves.npz)`。

### 结论

- fold1 主跑全绿：同口径（q=0.5 无偏 μ=0 global-τ）在 f12 fold1 代理脑上可计算且
  逐例自洽；interior 67.8–79.5% 高于 fold0 symw 组均 64.2%（网格内解析度高）。
  EC 数值结论 provisional（θ 候选判断待用户）。放行 Run 003/004 离线后处理。

## Run 002 — fold2 主跑（2026-10-25，主会话启动）

- **目的**：fold2（test = [3,6,8,11,23,26,28,31]）同口径 EC 主跑，fold2 best ckpt
  （0.8433@E113）；命令 = Run 001 同款仅 `--fold 2` + fold2 ckpt + fold2 输出树；
  预算 ≈52 min。跑前登记（本块）。

### 配置

- 同 Run 001 仅三处替换：`--fold 2`、ckpt =
  `outputs/10-25-latent-flow-symw-f12/run_symw_fold2/checkpoints/
  fold2_mild_flow_best_graph_mexhat.pt`（best 0.8433@E113 真早停 E143）、
  输出树 `outputs/10-25-symw-ec-f12/fold2/ec_main`；日志
  `outputs/10-25-symw-ec-f12/fold2_ec_main.log`；norm SHA 断言值 = fold2 键控
  `a2c27a2d…`；manifest 单文件全折共用（SHA 不变）。

### 结果

- 跑后回填（2026-10-25，S3 会话）：**8/8 例成功，52.2 min**（单例 6.51–6.52 min）；
  manifest fold2 = 28/4/8 断言过 + **fold2 norm SHA（a2c27a2d…）断言通过**（日志
  paradigm 行 subjects=[3, 6, 8, 11, 23, 26, 28, 31]）；**P(resp>τ)=0.500 逐例
  精确**；逐例表（interior 范围 **62.5–77.9%**，s6 @min 13.2% 为两折唯一
  >10% 下钳边例）：

  | 被试 | 组 | tau | interior | @min | @max | P(resp>τ) |
  |------|----|-----|----------|------|------|-----------|
  | s3 | MDD | 6.749e-05 | 75.5% | 3.3% | 21.3% | 0.500 |
  | s6 | MDD | 6.681e-05 | 62.5% | 13.2% | 24.2% | 0.500 |
  | s8 | MDD | 9.304e-05 | 69.8% | 3.0% | 27.2% | 0.500 |
  | s11 | MDD | 8.784e-05 | 70.7% | 3.9% | 25.4% | 0.500 |
  | s23 | HC | 7.049e-05 | 71.4% | 3.1% | 25.4% | 0.500 |
  | s26 | HC | 1.092e-04 | 77.9% | 2.5% | 19.6% | 0.500 |
  | s28 | HC | 6.564e-05 | 74.4% | 3.3% | 22.3% | 0.500 |
  | s31 | HC | 7.598e-05 | 72.3% | 3.0% | 24.7% | 0.500 |

- 凭据链核对：8/8 payload `task=10-25-symw-ec-f12-fold2`、`hparams.fold=2` +
  `symw=true`、norm SHA = a2c27a2d…、ckpt_path 指向 f12 run_symw_fold2；产物
  `outputs/10-25-symw-ec-f12/fold2/ec_main/subject_{0..7}.json(+_curves.npz)`。

### 结论

- fold2 主跑全绿：与 fold1 同口径同自洽（P=0.500 全精确）；interior 62.5–77.9%
  与 fold1 67.8–79.5% 同量级。EC 数值结论 provisional（θ 候选判断待用户）。
  放行 Run 003/004 离线后处理。

## Run 003 — 每折组级 summary（离线，S3）（2026-10-25）

### 目的与假设

- prd R5：fold1/fold2 各 8 例 ec_main 的组级汇总——interior/dyn_ratio/方差分解
  （行源/列靶/残差）/跨被试稳定性 ρ̄ + 重映射不变性 8/8。假设 = 汇总形态与
  fold0 symw 同型（列靶主效应主导、跨被试中低稳定），阴性亦如实登记。

### 配置

- 代码：`models/latent-flow-v1/ec_mild_unbiased.py --mode summary`（Run 000 版本，
  零代码改动；summary 模式 task 串硬编码「Run 002」后缀——脚本沿用先例字面量，
  元数据注记非本任务 Run 编号，README/本表已注明）
- 命令（每折同款，k∈{1,2}）：`OMP_NUM_THREADS=8 /root/miniconda3/bin/python
  models/latent-flow-v1/ec_mild_unbiased.py --mode summary --fold k --symw
  --main-dir /root/npi/outputs/10-25-symw-ec-f12/fold{k}/ec_main --out-dir
  /root/npi/outputs/10-25-symw-ec-f12/fold{k}/summary --task-label
  10-25-symw-ec-f12-fold{k}`（纯离线 CPU，读例档 + curves npz）
- 环境：conda npi @ 本机 CPU；数据 = Run 001/002 逐例档（MDD 4/HC 4 每折）

### 结果

- **fold1**（`fold1/summary/group_summary.json` + group_mean.npz）：interior 组均
  **74.1±3.7%**（@min 1.4±1.1% / @max 24.5±3.2%）；dyn_ratio 中位 **3.59×±0.89**
  （q05 均值 1.88；nonmono 0.9921）；**方差分解 行/列/残差 = 0.142/0.381/0.477**
  （残差近半，三折最边特异形态）；跨被试稳定 ρ **0.585**（min 0.320 max 0.777）；
  τ 中位 8.924e-05（6.43e-05–1.37e-04）；σ* 中位 offdiag 0.3016±0.0010；
  conn offdiag mean **−0.554±0.052** / std 0.543±0.137（v1 口径）；P(resp>τ)
  0.500±0.000；**重映射不变性 8/8 PASS**。
- **fold2**（`fold2/summary/group_summary.json`）：interior 组均 **71.8±4.6%**
  （@min 4.4±3.6% / @max 23.8±2.5%）；dyn_ratio 中位 **2.79×±0.46**（q05 均值
  1.52）；**方差分解 0.094/0.762/0.144**（列靶主导，与 fold0 0.031/0.753/0.216
  同型）；跨被试稳定 ρ **0.588**（min 0.223 max 0.837）；τ 中位 7.323e-05；
  σ* 中位 0.3011±0.0011；conn offdiag mean **−0.462±0.127** / std 0.774±0.199；
  P=0.500±0.000；**重映射不变性 8/8 PASS**。

### 结论

- prd R5 落档：两折汇总自洽（P=0.500 全精确、σ* 中位三折几乎一致 ~0.301）；
  interior/dyn_ratio 均高于 fold0 symw（64.2%/2.55×）；fold2 方差分解与 fold0
  同型（列靶主导）、fold1 形态分化（残差 0.477 近半 + 行源 0.142 最高）——跨
  代理脑实例差异如实登记，描述性口径。EC 结论 provisional（θ 待拍板）。

## Run 004 — 每折 v4 重映射 + 双 GC 对照（离线，S3）（2026-10-25）

### 目的与假设

- prd R4：每折 conn v4 = −(ln σ*+0.5·ln2πe)+5/7 离线重映射 + v1/v2 双 GC
  per-subject/group/group_mean 对照（refs 全 40 被试覆盖，各折 test 被试不同）。
  假设 = 三断言 8/8（严格单调/全正/abs≡raw）+ 双参考 Pearson 平移不变 16/16，
  GC 相关形态与 fold0 symw（v1 微弱正/v2 近零）同位。

### 配置

- 代码：`models/latent-flow-v1/recompute_conn_v4.py`（零代码改动，--task-label/
  --surrogate-name 新调用面先例已备）
- 命令（每折同款，k∈{1,2}）：`OMP_NUM_THREADS=8 /root/miniconda3/bin/python
  models/latent-flow-v1/recompute_conn_v4.py --main-dir /root/npi/outputs/
  10-25-symw-ec-f12/fold{k}/ec_main --out-dir /root/npi/outputs/10-25-symw-ec-f12/
  fold{k}/recompute_v4 --task-label 10-25-symw-ec-f12-fold{k} --surrogate-name
  "sym-w f12 fold{k} surrogate (SymWAdam W←(W+Wᵀ)/2, λ0.2; best 0.8264@E116
  fold1 / 0.8433@E113 fold2)"`（按折写准确值）
- 参考：v1_1_45_fixed8（1–45 Hz 固定阶 8）+ v2_1_125_aic4_25（1–125 Hz AIC
  4–25）；offdiag 16256 对；描述性登记不构成代理脑资格判定
- 环境：conda npi @ 本机 CPU

### 结果

- **两折三断言全过：asserts_passed=40/40**（8/8 重映射严格单调 + 8/8 全正 +
  8/8 abs≡raw 逐位 + 双参考 Pearson 平移不变 16/16，|Δ|<1e-9；json v1 float32
  舍入差 max fold1 2.68e-07 / fold2 2.63e-07 随档登记）；refs 全 40 被试覆盖，
  16 个新被试对照无缺。逐被试 + group + group_mean 完整数值（offdiag 16256 对，
  raw≡abs 逐位相同；Spearman 主读数 ρ + Pearson r + flipped）：

- **fold1**（`fold1/recompute_v4/`）：
  - **v1 refs（1–45 Hz 固定阶 8）**：per-subject ρ = s2 +0.1188 / s12 +0.2834 /
    s14 +0.1552 / s17 +0.2393 / s22 +0.2533 / s32 +0.1876 / s34 +0.1209 /
    s37 −0.0864（7 正 1 负）；**group ρ +0.1590±0.1164 / r +0.2102±0.1015**；
    flipped ρ +0.0988±0.0512；**group_mean ρ +0.3805 / r +0.3759**
    （aligned>flipped 保持，方向特异弱但非零）
  - **v2 refs（1–125 Hz AIC 4–25）**：per-subject ρ = s2 +0.0132 / s12 +0.0947 /
    s14 −0.0114 / s17 +0.0913 / s22 +0.1752 / s32 +0.0809 / s34 +0.0247 /
    s37 +0.1495（7 正 1 微负）；**group ρ +0.0772±0.0655 / r +0.0559±0.0861**；
    flipped ρ −0.0893±0.2108；**group_mean ρ +0.1467 / r +0.1130**（近零偏正）

- **fold2**（`fold2/recompute_v4/`）：
  - **v1 refs**：per-subject ρ = s3 +0.1849 / s6 −0.0128 / s8 +0.3100 /
    s11 +0.2860 / s23 +0.2301 / s26 −0.0079 / s28 +0.2720 / s31 +0.1437
    （6 正 2 微负）；**group ρ +0.1757±0.1269 / r +0.1854±0.1148**；flipped ρ
    +0.0506±0.0638；**group_mean ρ +0.2675 / r +0.2494**（aligned>flipped 保持）
  - **v2 refs**：per-subject ρ = s3 −0.0566 / s6 +0.1932 / s8 +0.1227 /
    s11 +0.0966 / s23 −0.0386 / s26 −0.1863 / s28 +0.0976 / s31 −0.1524
    （4 正 4 负）；**group ρ +0.0095±0.1379 / r −0.0180±0.0837**；flipped ρ
    −0.1167±0.1876；**group_mean ρ +0.0288 / r −0.0262**（近零）

### 结论

- prd R4 落档：**双折 GC 形态与 fold0 同型**——v1 微弱正（fold0 +0.134 /
  fold1 +0.159 / fold2 +0.176，三折 aligned>flipped）+ v2 近零（+0.016/
  +0.077/+0.010，flipped 皆负）⇒ fold0 判读「GC 相关形态由 GNN 架构面承载、
  对 W 对称化不敏感」在三折并置下**描述性稳定出现**（各折 θ 不同实例，
  非复现性判读）。描述性登记不构成代理脑资格判定；EC 结论 provisional。

## Run 005 — 跨折汇总（fold0/1/2 三折对照，用户核心交付③，S4）（2026-10-25）

### 目的与假设

- prd R6：fold0（在档 10-25-symw-ec-unbiased 只读）vs fold1 vs fold2 三折并置
  ——best val/test 被试集/interior/dyn_ratio/τ 中位/conn offdiag mean±std/方差
  分解三元组/跨被试稳定 ρ̄/v1·v2 GC ρ（group mean±std + group_mean + 逐被试
  ρ 明细 24 例三折全景）。假设 = 三折 GC 相关形态同型（v1 微弱正、v2 近零）；
  跨折并置为**描述性口径**（各折 θ 不同代理脑实例，单 seed，不作复现性判读）。

### 配置

- 代码：`models/latent-flow-v1/crossfold_summary_f12.py`（**本任务新增小脚本**，
  读档汇总纯后处理，py_compile 先行；读 fold0 = `outputs/10-25-symw-ec-unbiased/
  {summary/group_summary.json, recompute_v4/gccorr_dual_summary.json}` 只读 +
  fold1/2 = 本任务对应档）
- 输出：`outputs/10-25-symw-ec-f12/crossfold/crossfold_summary.json`（必含
  caliber_note 三折同口径逐字 + 跨折 DESCRIPTIVE 注记 + 单 seed 标注 +
  provenance 各折 ckpt/best-val/来源任务名）
- 环境：conda npi @ 本机 CPU，零 GPU、零随机源

### 结果

- `crossfold_summary_f12.py` py_compile PASS；读档守卫全过（三折 task 串/8 例/
  asserts=40/重映射 8/8/24 例被试无重叠）；产物
  `outputs/10-25-symw-ec-f12/crossfold/crossfold_summary.json`（caliber_note +
  descriptive_only + model_version + provenance×3 + per_fold×3 + 逐被试 GC
  明细 24×2）。三折对照关键行：

  | 行（口径见 caliber_note） | fold0 | fold1 | fold2 |
  |---|---|---|---|
  | stage2 best val | 0.8054@E144 | 0.8264@E116 | 0.8433@E113 |
  | test 8 例 | s1,5,7,10,21,25,27,30 | s2,12,14,17,22,32,34,37 | s3,6,8,11,23,26,28,31 |
  | interior 组均 | 64.2±7.4% | **74.1±3.7%** | 71.8±4.6% |
  | dyn_ratio 中位 | 2.55× | **3.59×** | 2.79× |
  | τ 中位（被试间） | 6.87e-05 | 8.92e-05 | 7.32e-05 |
  | conn offdiag mean±std（v1 口径） | −0.378±0.175 | −0.554±0.052 | −0.462±0.127 |
  | 方差分解 行/列/残差 | 0.031/0.753/0.216 | 0.142/0.381/**0.477** | 0.094/0.762/0.144 |
  | 跨被试稳定 ρ̄ | 0.425 | 0.585 | 0.588 |
  | v1 GC ρ（group mean±std） | +0.134±0.159 | +0.159±0.116 | +0.176±0.127 |
  | v1 GC r（Pearson） | +0.172±0.179 | +0.210±0.102 | +0.185±0.115 |
  | v1 group_mean ρ / r | +0.181 / +0.234 | **+0.381 / +0.376** | +0.268 / +0.249 |
  | v2 GC ρ | +0.016±0.085 | +0.077±0.066 | +0.010±0.138 |
  | v2 GC r | −0.024±0.088 | +0.056±0.086 | −0.018±0.084 |
  | v2 group_mean ρ / r | +0.136 / +0.051 | +0.147 / +0.113 | +0.029 / −0.026 |

- **跨折形态一致性判读（描述性）**：①GC 形态三折同型——v1 微弱正且
  aligned>flipped 三折全保持、v2 近零且 flipped 三折皆负（−0.12/−0.09/−0.12）
  ⇒ fold0「GC 相关形态由 GNN 架构面承载、对 W 对称化不敏感」结论在三折并置
  下稳定；②方差分解 fold2 与 fold0 同型（列靶主导 0.76），fold1 实例分化
  （残差 0.477 + 行源 0.142 最高 + dyn 3.59×）——边特异/源全局结构份额跨
  θ 实例可变，如实登记不归因；③σ* offdiag 中位三折几乎一致（0.302/0.302/
  0.301）；④三折均单 seed、被试集互斥 ⇒ **不作复现性判读**（caliber_note
  落档）。

### 结论

- prd R6 落档：三折对照表 + 逐被试 24 例 ρ 明细 + caliber/provenance 全档；
  跨折 GC 形态一致性为描述性支持（非复现性判读）。EC 结论 provisional（θ
  候选判断待用户，待拍板⑩联动）。
