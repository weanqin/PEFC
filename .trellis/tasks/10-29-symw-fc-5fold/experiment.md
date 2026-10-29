# 实验日志 — 10-29-symw-fc-5fold

> 每次「跑批/训练/评估/统计」追加一个 Run 块：跑前先写「目的+配置」，跑后同会话补「结果+结论」，不留空块。
> 格式契约见 `.trellis/spec/research/experiment-log.md`；产物只落本任务声明的变体目录（`.trellis/spec/research/path-isolation.md`）。

## Run 001 — symw 五折 test 集 gen-fc vs real-fc Pearson 一致性（指标 1 v2.1 正式核算）（2026-10-29）

### 目的与假设

- 问题：mild-latent-flow-symw-anti0.2-gnn-exit（symw）代理脑生成的未来段 EEG 与真实数据在 Pearson 功能连接上的一致性如何？对照指标 1 v2.1 通过标准（组平均 Pearson r ≥ 0.95，辅助：逐被试 t 检验 p<0.05）给出五折逐折与汇总判定（Surrogate Brain Validity / Q2）。

### 配置

- 代码：`models/latent-flow-v1/fc_metric_mild_symw.py`（本任务新增；复用 load_manifest / compute_mild_train_norm / MildTrialDataset / build_model_mild / load_flow_ckpt / sample_forecast；指标纯函数镜像 gpfm-mamba-v1/fc_metric.py，p 值 Lentz 连分式镜像 gc_psd.py）
- 数据：Mild Depression（契约 spec/data/mild-depression.md），manifest `splits/mild_5fold_seed42.json`（SHA 断言），每折 test 8 例 × 30 trial；w500 口径 obs 375 / fc 125（trial 第 375–500 点，x_f 字段不参与）
- 模型：symw 五折 stage2 best = `result/checkpoints/mild/latent_flow_symw/fold{k}/`（注册件只读，SHA×5 对拍 MANIFEST 登记值）；FlowVelocityNetExitGNN strict 装载
- 指标口径：逐被试 30 段拼接 (128,3750) float64 → np.corrcoef；组平均上三角 8128 边 Pearson r + Spearman ρ；逐被试 r_s/ρ_s 单样本双侧 t 检验（df=7/折，汇总合并 40 例 df=39）
- 采样：确定性 K=1（全确定性管线，K 条逐位相同），seed 42；FC 计算域 = z 域（与原始 μV 域仿射等价，每折首被试双阈值断言 <1e-12/<1e-6）
- 设备：**CPU 单线程**（BLAS/torch threads=1）。GPU 前向经探针证不可逐位复现：同卡并发训练进程改变显存分配 → 指针对齐 → cuBLASLt 核选择漂移，CUBLAS_WORKSPACE_CONFIG=:4096:8 + torch.use_deterministic_algorithms(True) 双开后仍逐次不同；CPU 单线程跨进程逐位一致（探针 2026-10-29，/tmp 会话存证）。device 对指标数值影响 ~1e-5
- 命令：`for k in 0 1 2 3 4; do python3 models/latent-flow-v1/fc_metric_mild_symw.py --mode fold --fold $k --out-root /root/npi/outputs/10-29-symw-fc-5fold; done` + `--mode determinism --fold 0` + `--mode summary`
- 环境：npi 环境 @ 本机（.trellis/research/environment.md；PyTorch 2.8.0+cu128，本跑 CPU）

### 结果

- 指标（五折并置，组平均点估计 + 逐被试 r_s±sd，df=7/折）：

| fold | 组 Pearson r | 组 Spearman ρ | 被试级 r_s | t | p |
|---|---|---|---|---|---|
| 0 | 0.95636 | 0.94361 | 0.90544±0.03848 | 66.5515 | 4.545e-11 |
| 1 | 0.77146 | 0.76199 | 0.74921±0.06514 | 32.5296 | 6.713e-09 |
| 2 | 0.69556 | 0.69456 | 0.67624±0.08445 | 22.6486 | 8.283e-08 |
| 3 | 0.86072 | 0.81606 | 0.81583±0.07112 | 32.4445 | 6.837e-09 |
| 4 | 0.96251 | 0.98280 | 0.91965±0.01089 | 238.8300 | 5.956e-15 |
| **5-fold 平均** | **0.84932** | **0.83980** | 合并 40 例 0.81327±0.10953（df=39） | 46.9625 | 6.113e-36 |

- 守卫：manifest/norm×5/ckpt×5 SHA 全对拍过；z↔原始域等价断言五折全过（float64 重算 ~1e-15 / float32 管线 ~2–3e-9）；AC5 确定性 fold0 复跑逐位一致（8/8 gen 样本 SHA + 全数值字段）
- 产物：`outputs/10-29-symw-fc-5fold/`（fold{k}/fc_metric.json + fc_matrices.npz、summary_5fold.json、README.md、run.log）
- 图表：无（数值交付）

### 结论

- **部分达标（2/5 折过线，均值未过）**：fold0（0.95636）与 fold4（0.96251）达到指标 1 通过标准组平均 Pearson r ≥ 0.95，fold1/2/3（0.771/0.696/0.861）未达；五折均值 0.84932 < 0.95。辅助口径全过：五折逐被试 t 检验 p ∈ [5.96e-15, 8.28e-08] 全部 ≪ 0.05，合并 40 例 r_s = 0.8133±0.1095（t=46.96，p=6.1e-36）——FC 结构正向且高度显著，但幅度未达 0.95 线。跨折散布大（0.696–0.963）为显著特征，fold2 最低同时被试间方差最大（±0.084）
- 按规范口径：本结果属 Surrogate Brain Validity（Q2）指标 1 单项核算；指标 2（人工视觉判）/指标 3（PSD）未在本任务范围。指标 1 单项不构成「代理脑合格」完整判定
- 下一步 / 派生决策项：①symw θ 候选判断（待拍板⑩）现可并入指标 1 数据（2/5 折过线、均值 0.849 的部分达标形态）；②环境学习已沉淀：共享 GPU 并发下前向不可逐位复现（cuBLASLt 核选择漂移），小评估任务逐位复现走 CPU 单线程（spec/reproducibility.md 增记 + environment.md 回填）

