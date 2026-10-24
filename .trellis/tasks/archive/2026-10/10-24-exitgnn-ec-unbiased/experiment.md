# 实验日志 — 10-24-exitgnn-ec-unbiased

> 每次「跑批/训练/评估/统计」追加一个 Run 块：跑前先写「目的+配置」，跑后同会话补「结果+结论」，不留空块。
> 格式契约见 `.trellis/spec/research/experiment-log.md`；产物只落本任务声明的变体目录（`.trellis/spec/research/path-isolation.md`）。

## Run 000 — v2 GC 参考注册（EC_GC_1_125 → refs）（2026-10-24）

### 目的与假设

- 用户拍板双 GC 对照之新参考：注册 `/root/autodl-tmp/EC_GC_1_125.zip`（1–125 Hz、AIC
  逐被试选阶 4–25）为 refs v2 档，供 Run 003 双对照消费；假设 subject 映射 sid=index+1
  与 v1 相同（四锚点 bitwise 复验重做）。

### 配置

- 代码：`refs/granger_bivariate_spectral/mild40-trial500-zscore-fs250-aic4-25-f1-125-favg/
  register_refs.py`（镜像 v1 先例 + 四锚点复验内置 + README QC 量级对拍断言）
- 参数：fmin 1 / fmax 125 Hz faverage、AIC per-subject 4–25（500−25 截断）、fs 250、
  trial 前 500 点 × 30 池化、逐 trial 逐通道 z-score；源 zip SHA256 随 meta
- 环境：npi 本机（CPU）
- 数据：zip 内 `EC_GC_subject.npy` (40,128,128) float64；锚点 EEG_ALL.mat vs
  m_1/m_20/h_1/h_20.mat（manifest splits/mild_5fold_seed42.json）
- 命令：`cd refs/.../aic4-25-f1-125-favg && /root/miniconda3/bin/python register_refs.py 2>&1 | tee register.log`

### 结果

- 四锚点 bitwise 复验：idx0=m_1(sid1) / idx19=m_20(sid20) / idx20=h_1(sid21) /
  idx39=h_20(sid40) **全部 maxabsdiff=0.0**；组别交叉断言 40/40 一致。
- sanity：diag_max=0、min=0（非负）、无 NaN、mean=0.0475 / max=0.5684 与源 README QC
  （0.0475/0.568）对拍通过（容差 2e-3/5e-3）。
- 产物：40 × refs_s{sid}.npz + SHA256SUMS + meta.json（含 caliber_note：数据有效带宽
  ≈0.5–50 Hz，51–125 Hz 衰减 −24 dB 以下；与 v1 逐被试 Pearson median=0.148）+
  README-source.md + register.log；`refs/MANIFEST.md` 追加 v2 行 + 内容/边界注记。
- 耗时 11.9 s。

### 结论

- v2 参考注册完成，映射假设成立；双 GC 对照的参考侧就绪。

## Run 000b — 工程干跑 + 旗标关等价性对拍（非正式口径）（2026-10-24）

### 目的与假设

- S1 适配（--flow-exit-gnn/--task-label/原子写）后验证：①exit-gnn ckpt 经
  build_model_mild 变体子类 strict load 全链路可用；②旗标关时与 w500 先例行为**逐位等价**
  （评审 Mandate A 建议的运行时证据）。

### 配置

- 代码：`models/latent-flow-v1/ec_mild_unbiased.py`（S1 改后）
- 参数：非正式干跑口径 1 例（s1）× 8 源 × log[0.01,0.48]×4 网格，其余默认
  （q=0.5/σ_ref=0.3/seed 42/30 trial 全量）
- 两跑：①旗标开 + exit-gnn ckpt → dryrun_mini/；②旗标关 + 默认 w500 ckpt →
  scratch_equiv/（与 w500 在档 dryrun_mini 同参对拍）
- 命令：`OMP_NUM_THREADS=8 /root/miniconda3/bin/python models/latent-flow-v1/
  ec_mild_unbiased.py --mode main --flow-exit-gnn --task-label
  10-24-exitgnn-ec-unbiased --ckpt <exit-gnn best> --subjects 1 --sources 8
  --n-grid 4 --out-dir outputs/10-24-exitgnn-ec-unbiased/dryrun_mini`（②去旗标换默认
  ckpt，out-dir scratch_equiv）

### 结果

- ①旗标开：FlowVelocityNetExitGNN 替换生效（+33,024 参数，1,276,258 全模型），
  strict load 通过；τ=7.331e-05、interior 40.1%、P(resp>τ)=0.500 精确；payload 含
  task/surrogate(flow_exit_gnn:true)/surrogate_caveat 字段（核对无缺）。
- ②旗标关等价对拍：sigma_matrix / connectivity_matrix / raw_response_matrix /
  mean_diff_matrix 四矩阵 + tau_global 与 w500 在档 dryrun **全部逐位相等
  （max|Δ|=0.0）**；hparams 新增 flow_exit_gnn:False（声明的新键）；顶层 task 字段
  未写（空串语义正确）；surrogate.task 保持 w500 原值。
- 产物：`outputs/10-24-exitgnn-ec-unbiased/{dryrun_mini,scratch_equiv}/subject_0.json(+curves.npz)`。

### 结论

- Gate G2 通过：适配正确、默认面运行时逐位等价（AC2 证据落档）；放行 Run 001。

## Run 001 — GPU 主跑：无偏 μ=0 global-τ EC × exit-gnn 代理脑（2026-10-24）

### 目的与假设

- 用户指令主实验：以 exit-gnn stage2 best（0.8241@E116）为代理脑，无偏 μ=0 σ·ξ 扰动 +
  两阶段 global-τ（q=0.5@σ_ref=0.3，log[0.01,0.48]×24）计算 fold0 test 8 例 × 30 trial
  的 σ*/conn（128×128 有向）。
- 假设（对照 w500 先例 46.9 min / interior 60.3% / dyn 2.62×）：出口消息传递使前向略增、
  单例耗时 ≥5.85 min 参照；响应结构（interior/dyn_ratio）量级待实测。

### 配置

- 代码：`models/latent-flow-v1/ec_mild_unbiased.py`（Run 000b 验证版）
- 参数：**全默认**——q=0.5、σ_ref=0.3、log[0.01,0.48]×24、n_steps 10、batch 32、seed 42、
  bf16 autocast、τ 分位含对角（房内先例口径）；旗标 --flow-exit-gnn +
  --task-label 10-24-exitgnn-ec-unbiased
- 环境：npi 本机 RTX 4080 SUPER（GPU 独占窗口，启动前空闲已验）
- 数据：Mild Depression fold0 test 8 例（MDD s1,5,7,10 / HC s21,25,27,30）× 30 trial
  全量；norm/manifest SHA 断言链照旧（d184e647…/47987909…）
- 命令：`OMP_NUM_THREADS=8 /root/miniconda3/bin/python models/latent-flow-v1/
  ec_mild_unbiased.py --mode main --flow-exit-gnn --task-label
  10-24-exitgnn-ec-unbiased --ckpt /root/npi/outputs/10-24-latent-flow-exit-gnn/
  run_exitgnn_fold0/checkpoints/fold0_mild_flow_best_graph_mexhat.pt --out-dir
  /root/npi/outputs/10-24-exitgnn-ec-unbiased/ec_main`（后台 + log；断点续跑语义）

### 结果

- 8/8 例成功，总 51.2 min；**单例 6.40 min 全程稳定**（w500 参照 5.85 min ⇒ +9.4%
  出口消息传递前向开销，登记为迁移成本事实）；GPU 100% util / 2.9 GB。

| 例 | sid | 组 | τ | interior% | @min% | @max% | P(resp>τ) |
|---|---|---|---|---|---|---|---|
| 0 | s1 | MDD | 7.598e-05 | 40.6 | 23.2 | 36.1 | 0.500 |
| 1 | s5 | MDD | 5.024e-05 | 57.7 | 8.5 | 33.8 | 0.500 |
| 2 | s7 | MDD | 5.225e-05 | 52.6 | 12.7 | 34.7 | 0.500 |
| 3 | s10 | MDD | 5.638e-05 | 52.6 | 13.1 | 34.3 | 0.500 |
| 4 | s21 | HC | 4.975e-05 | 51.4 | 12.2 | 36.3 | 0.500 |
| 5 | s25 | HC | 3.208e-05 | 44.6 | 20.1 | 35.3 | 0.500 |
| 6 | s27 | HC | 5.173e-05 | 64.1 | 5.0 | 30.9 | 0.500 |
| 7 | s30 | HC | 4.566e-05 | 64.0 | 3.8 | 32.2 | 0.500 |

- P(resp>τ)=0.500 逐例精确（q=0.5 构造性自洽）；凭据链全过（manifest/norm SHA、
  ckpt 路径断言、surrogate_caveat/task 字段落档）。
- 产物：`ec_main/`（8 json + 8 curves.npz，31 MB）+ `ec_main_run.log`。

### 结论

- 工程假设成立：exit-gnn 代理脑全链路可跑，成本 +9.4%/例；interior 40.6–64.1%
  （组均 53.5% < w500 60.3%），钳边结构以 @max 为主（组均 34.2% vs @min 12.3%）。
- EC 读数移交 Run 002/003。

## Run 002 — 离线组级 summary（2026-10-24）

### 目的与假设

- 与 w500 Run 002 同口径对照：σ*/conn 分布结构、方差分解（源行/靶列/边残差）、
  跨被试稳定性、重映射不变性。

### 配置

- 命令：`--mode summary --flow-exit-gnn --task-label … --main-dir ec_main --out-dir
  summary`（离线无 GPU）。

### 结果

- interior 组均 **53.5%**（w500 60.3%）；**dyn_ratio 中位 1.79×**（w500 2.62×——更平）；
  方差分解 行 0.010 / **列 0.907** / 残差 0.083（w500 0.037/0.641/0.322——靶主效应
  更主导、边特异残差份额更小）；跨被试稳定 ρ=**0.655**（w500 0.576）；σ* offdiag
  中位 0.303；重映射（1/(σ*+1e-4)）Spearman 不变 8/8 PASS。
- 产物：`summary/group_summary.json` + `group_mean.npz`。

### 结论

- 响应结构较 w500 **更弱**：动态范围更平 + 靶主效应更主导（0.91）+ 边残差更小
  （0.08）——与 exit-gnn 出口「再平均」幅度欠校准 caveat 同向（跨通道传导结构更弱）；
  跨被试稳定性反而略高（0.655 vs 0.576，或反映结构趋同/靶效应主导的简化）。

## Run 003 — conn v4 重映射 + 双 GC 对照（2026-10-24，用户拍板交付公式）

### 目的与假设

- 用户指定 v4 = −(ln σ*+0.5·ln2πe)+5/7 交付；双参考（v1 1–45 Hz 固定阶 8 / v2
  1–125 Hz AIC 4–25）描述性对照。

### 配置

- 代码：`models/latent-flow-v1/recompute_conn_v4.py`（正式入库；先例 Run 007 为临时
  脚本未入库）；纯离线。
- 断言组：同 σ* float64 路径 Spearman 严格单调（|Δρ|<1e-9）+ 全正（min>0）+ abs≡raw
  逐位 + 每参考 Pearson 平移不变性（|Δr|<1e-9）——**40 项全过**。
- 实现注记（口径登记）：json 的 v1 conn 为主跑 float32 舍入值，与 float64 重算存在
  ~1e-13 量级 Spearman 尾差（舍入引入新并列，s5 实测 1−ρ=7.3e-13——精度伪影非口径
  漂移）；不变性断言因此取同 σ* float64 重算路径，json v1 舍入差 max 2.7e-7 随档
  （summary.formula_facts.v1_json_f32_rounding_maxabs）。

### 结果

- v4 分布：全正（断言），raw≡abs（协议两键保留、逐位一致断言在档）。
- **双 GC（offdiag 16256，raw aligned）**：

| 参考 | Pearson | Spearman | flipped Spearman | 组均值矩阵 ρ / r |
|---|---|---|---|---|
| v1 1–45 Hz 固定阶 8 | **+0.1486±0.1677** [−0.13,+0.35] | +0.1196±0.1805 | +0.1071±0.0721 | +0.1934 / +0.1993 |
| v2 1–125 Hz AIC 4–25 | **−0.0102±0.0847** [−0.18,+0.08] | +0.0260±0.0762 | −0.1479±0.1693 | +0.1059 / +0.0555 |

- v1 块跨代理脑并置（w500 Run 007 先例 +0.1674±0.1661 / 组均值 +0.2155）：同量级
  微弱正相关、本跑略低——θ 不同，描述性并置非复现性判读。
- v2 块：aligned 基本零相关；flipped 组均 −0.15（与 ec.T 反向的弱结构，描述性登记）。
- 产物：`recompute_v4/`（8 例重映射 json + gccorr_dual_summary.json）。

### 结论

- v4 交付就绪（全正无 |·| 折叠歧义，后续网络化/阈值切边可直接用数值）。
- GC 对照阴性读数同先例方向且更弱：v1 参考微弱正相关（≈0.12–0.15）、v2 参考
  基本零——**该配置（exit-gnn + 无偏 μ=0 + q=0.5）下 EC_pert 不携带可用有向连通
  信号**；与代理脑幅度欠校准/靶主效应主导（Run 002）同向汇聚。
- 单次运行（fold0 单 seed，bf16 AMP）；EC 结果 provisional（随 exit-gnn θ 判断
  拍板联动改判）。

