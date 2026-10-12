# 实验日志 — gpfm-mamba-edra-f0

> 每次「跑批/训练/评估/统计」追加一个 Run 块：跑前先写「目的+配置」，跑后同会话补「结果+结论」，不留空块。
> 格式契约见 `.trellis/spec/research/experiment-log.md`；产物只落本任务声明的变体目录（`.trellis/spec/research/path-isolation.md`）。

## Run 000 —— fold0 单 seed 完整训练（预登记 · 未启动 · 用户门禁）

- **状态**：未启动。implement.md §5 完整训练须待 4.5 基准摘要经用户确认放行后方可执行；本块为 Run-0 预登记（implement.md 0.2），启动时在本块补齐 resolved config 全量要素与 run_id。
- 目的：fold0 单 seed 前置验证训练（Study I 代理脑，gpfm-mamba-v1 变体）。
- 配置（已定，跑前 dump resolved config 到 run 目录）：
  - 划分：受试者级 5-fold 分层（MDD/HC）split_seed=42，fold0；qc_pass=True 过滤（实测 **13 例：HC 5 / MDD 8**）；内部验证 2 例（HC 1 / MDD 1，val_subject_seed=42）；训练折 8 例 10568 窗 / 内部验证 2 例 2642 窗 / 测试 3 例 3963 窗（stride=64，每例 1321 窗）
  - z-score：逐通道，仅训练折 8 例估计
  - 先验：K_t=OU γ=1 τ=i·π/p p=50 iso=1e-2（折叠进时间核）pp_jitter=1e-4；K_s=Ricker3D σ=0.080 m + jitter_s=1e-3
  - 模型：Mamba d_model=16 d_state=16 d_conv=4 expand=2 ×4 层 + 预 RMSNorm 残差；EMA 0.9999
  - 损失：1·mae+1·cons+0.1·tv+0.1·corr（future 128 列，一步生成 x̂1=x_t+(1−t)v_θ）
  - 训练：seed=1234、AdamW lr=1e-3 wd=1e-4 cosine、clip 0.5、batch 256、epochs 100；t=sigmoid(N(−0.8,0.8²)) clamp [1e-4,1−1e-4]
  - best 判据：内部验证总 loss（固定 t 网格 {0.1,0.3,0.5,0.7,0.9}+固定 eps=seed 2026 确定性复算）
  - 主指标（§6）：测试折 MAE（ODE 终点样本均值口径）

## Run 001 —— 训练折核诊断复跑（2026-10-12）

### 目的与假设

- 在 fold0 训练折**正式 z-score 口径**下复跑 GPR 核诊断（适配研究 Q1 仅 2 被试、逐被试 z-score 代理，为指示性）：确认 OU 时间核数值稳定（R²(μ)/post_std/条件数）与 Ricker3D 空间核 σ=0.080 默认档结论（min_eig/邻位相关/负瓣/有效秩）；若诊断推翻默认档，改档属口径变更须登记。

### 配置

- 代码：`models/gpfm-mamba-v1/diagnose_kernel.py`（本任务首建，commit 见 Run 块结论补记）
- 参数：数据 = fold0 训练折 8 例（qc_pass=True、z-score 仅训练折、窗 384/128 stride 64，正式管道）；诊断窗 = 8 例 × 40 窗（确定性等距抽样）；先验 = OU γ=1 p=50 iso=1e-2 pp_jitter=1e-4 + Ricker3D σ_s∈{0.045,0.080,0.137} jitter_s=1e-3；先验采样 400 条/窗（诊断专用 seed=1234）
- 环境：base conda @ 本地云服务器（.trellis/research/environment.md）
- 数据：`/root/autodl-tmp/edra_process_v4/`（只读）
- 命令：`python3 models/gpfm-mamba-v1/diagnose_kernel.py`

### 结果

- 时间侧（OU，fold0 训练折正式口径）：R²(μ)=+0.0495（适配研究指示值 0.01–0.03，量级一致 ≈0）；post_std mean 0.971（min 0.369 接缝处 / max 1.005）；Σ_pp 条件数 7.56e2（iso=1e-2 下 ~1e2–1e3，数值稳定）；GT future 粗度 0.127，先验样本粗度 0.30（≈2.4× GT，介于适配研究 iso=1e-4 的 1.5–2.2× 与 iso=1e-1 的 2.2–3.8× 之间，符合 iso=1e-2 折中预期）；GT future 窗内 std 0.833。
- 空间侧（Ricker3D 三档全 PSD + jitter 后 Cholesky 成功）：

| σ_s | min_eig | +jitter 条件数 | 邻位相关（解析/经验）| 负对占比 | 过零 vs 阵列 | 有效秩 exp(H)/Σλ/λmax |
|---|---|---|---|---|---|---|
| 0.045 | +9.92e-4 | 2.3e3 | +0.664 / +0.674 | 77.4% | 0.078 < 0.190 | 27.0 / 13.4 |
| **0.080（默认）** | **+3.98e-7** | **1.4e4** | **+0.881 / +0.874** | **37.1%** | 0.139 < 0.190 | 10.1 / 4.5 |
| 0.137 | +2.09e-10 | 3.3e4 | +0.958 / +0.948 | 0% | 0.237 > 0.190 | 4.3 / 1.9 |

- 产物：`.trellis/tasks/10-12-gpfm-mamba-edra-f0/research/kernel-re-diagnosis.md`（含电极几何：最近邻 mean 0.031 m、最大间距 0.190 m）

### 结论

- **支持（默认档维持）**：σ=0.080 档复现适配研究全部关键数值——min_eig +3.98e-7（真实余量）、邻位 +0.88、负瓣 −0.055 / 37%、+jitter 条件数 1.4e4（float64 预计算安全）、过零 0.139 m 在阵列内；时间侧 OU 数值稳定且后验展宽匹配 z-score 尺度。**不构成口径变更，σ=0.080 默认档维持。**
- 三档语义确证：0.045 强中心-周边（77% 负对）、0.080 平衡、0.137 全局相干（无负瓣、过零出阵列、有效秩 ~2–4）。
- 口径备注：design 所引「有效秩 7.5/62」定义未注明；本次报告 exp(H)=10.1 与 Σλ/λmax=4.5 两口径，排序与定性结论不变（已记入诊断 md）。
- 派生事项：无（无推翻默认档证据）。

## Run 002 —— 小子集冒烟（2026-10-12）

### 目的与假设

- implement.md 4.4：~4 受试者小子集 1 epoch，验证训练循环机械正确性——四项损失均有限且下降、吞吐（it/s）记录、metrics.jsonl/best/last checkpoint 就位。

### 配置

- 代码：`models/gpfm-mamba-v1/train.py`（commit 见结论补记）
- 参数：`--smoke`（训练折每类取前 2 例 = 4 例，z-score 统计随子集重算——冒烟口径；epochs=1）；其余全默认（batch 256、lr 1e-3、权重 {1,1,0.1,0.1}、t=sigmoid(N(−0.8,0.8²))、seed=1234）
- 环境：base conda @ 本地云服务器
- 数据：`/root/autodl-tmp/edra_process_v4/` fold0（只读）
- 命令：`python3 models/gpfm-mamba-v1/train.py --smoke --run-id smoke-a`
- 产物：`/root/npi/outputs/gpfm-mamba-f0/smoke-a/`（config.yaml / env_snapshot.txt / logs/{train.log,metrics.jsonl,summary.json} / checkpoints/{gpfm-mamba-f0-best.pt,last.pt} / channel_{mean,std}.npy）

### 结果

- 训练折小子集 4 例（每类前 2：HC 27,39 / MDD 1,7）5284 窗，20 step/epoch；四项损失全程有限且下降：step1→step20 total 1.5128→1.3262；epoch 均值 mae 0.7365 / cons 0.5776 / tv 0.2150 / corr 0.5581 / total 1.3914。
- 内部验证（EMA 权重、固定 t 网格 {0.1,0.3,0.5,0.7,0.9}+固定 eps=2026 确定性复算）total 5.5964——EMA decay 0.9999 在 20 步下影子权重仍≈初始化（0.9999²⁰≈99.8% 初始权重），val≫train 属该机制的预期表现而非缺陷。
- t 分桶 sanity：桶计数 [195,932,1326,1033,817,487,227,92,11,0]（logit-normal 集中低中段、t>0.9 零样本，符合设计）；分桶 mae b0 1.057 → b8 0.202 随 t 单调降（初始化时 (1−t) 缩放效应，符合预期）。
- 吞吐：9.2 it/s（含首 step kernel 预热，稳态 ~23 it/s）、2437 窗/s；epoch 墙钟 2.2 s；peak 显存 0.728 GiB；‖v‖ 42.7；x̂1 std_t 0.695。
- 产物齐备：metrics.jsonl 1 行全字段、best/last checkpoint 各 341 KB、resolved config + pip freeze 快照落盘。代码 commit 3f02d36。

### 结论

- **通过**：训练循环机械正确（四项损失有限下降、checkpoint/metrics/复现三要素就位）。冒烟口径（小子集 z-score、EMA 滞后）不用于科学结论。
- 观察项（完整训练关注）：EMA 0.9999 时间常数 ~10⁴ 步 > 100 epoch×41 step=4100 步，val-best 判据在前期将主要反映 EMA 追赶过程——基准摘要（Run 004）向用户一并汇报。

## Run 003 —— 断点续传冒烟（2026-10-12）

### 目的与假设

- implement.md 4.4：kill -9 中断 → `--resume` 续跑，验证 R9 断点续传——metrics.jsonl 无缺口、loss 连续、epoch 计数/RNG/optimizer/scheduler 状态恢复正确（对照不间断同配置运行的 epoch 指标一致性）。

### 配置

- 代码：同 Run 002
- 参数：`--smoke --epochs 6`（冒烟口径 4 受试者）；B 组中途 kill -9（epoch 0 完成后、后续 epoch 进行中）→ `--resume <last.pt>` 续至 6 epoch；A 组不间断跑完 6 epoch 作对照
- 环境：同 Run 002
- 命令：A `python3 models/gpfm-mamba-v1/train.py --smoke --epochs 6 --run-id resume-ref`；B `python3 models/gpfm-mamba-v1/train.py --smoke --epochs 6 --run-id resume-kill` → kill -9 → `python3 models/gpfm-mamba-v1/train.py --smoke --epochs 6 --run-id resume-kill --resume /root/npi/outputs/gpfm-mamba-f0/resume-kill/checkpoints/gpfm-mamba-f0-last.pt`
- 产物：`/root/npi/outputs/gpfm-mamba-f0/{resume-ref,resume-kill}/`

### 结果

- B 组 kill -9 时机：epoch 2（0-based 1）step 19/20（last.pt 为 epoch 0 产物，metrics.jsonl 仅 epoch 0 一行——真实中断场景）。
- 首次 resume 尝试暴露并修复一处缺陷：EMA 影子权重随 checkpoint 以 map_location='cpu' 载入后未回 GPU，`ema.update` 设备不匹配报错（修复：`EMA.to(device)` + update 内部 cast，commit 见 Run 004 前的 train 提交）。
- 修复后 resume 成功：epoch 0→1、global_step 20、best_val 5.5964、RNG(python/numpy/torch/cuda)+optimizer+scheduler 恢复、metrics.jsonl 截齐至 epoch 0。
- **A/B 对照（6 epoch 全程）**：train total 逐 epoch 相对差 **0.0**（1.3914/1.3291/1.3040/1.3011/1.2947/1.2898 完全一致）；val total 最大相对差 1.8e-6（float 打印舍入级）；lr 轨迹逐 epoch 一致（9.3e-4→7.5e-4→…→0）；epoch 5 四项损失逐一相同；metrics.jsonl 0..5 无缺口。
- 结论性事实：当前配置（cudnn.deterministic + epoch 种子化 dataloader + 全量 RNG 状态保存）下，kill -9 中断→resume 与不间断运行**逐位一致**（Mamba CUDA kernel 在本规模下未观测到非确定性）。

### 结论

- **通过**：R9 断点续传全部验证点满足——metrics 无缺口、loss 连续、epoch 计数/RNG/optimizer/scheduler 状态恢复正确；续跑按同一 Run 延续登记（本块，不另开 Run）。
- 派生事项：无（EMA 设备缺陷已在冒烟阶段拦截，正式 Run 不受影响）。

## Run 004 —— 全量 1-epoch 基准测试（2026-10-12）

### 目的与假设

- implement.md 4.5（用户要求，含确认门禁）：全量训练折 1 epoch——peak 显存、epoch 墙钟、it/s、样本吞吐、100-epoch 外推、四项损失走势、t 分桶 sanity；**基准摘要汇报用户，未确认不得启动 §5 完整训练**。

### 配置

- 代码：同 Run 002
- 参数：`--epochs 1 --bench`（全量训练折 8 例 10568 窗，正式 z-score 口径；batch 256；seed=1234）
- 环境：同 Run 002
- 命令：`python3 models/gpfm-mamba-v1/train.py --epochs 1 --bench --run-id bench-1ep`
- 产物：`/root/npi/outputs/gpfm-mamba-f0/bench-1ep/`

### 结果

- 数据：全量训练折 8 例 10568 窗（41 step/epoch，batch 256）+ 内部验证 2 例 2642 窗；数据加载 1.1 s。
- **peak 显存**：0.728 GiB（`torch.cuda.max_memory_allocated()`；32 GB 卡余量 ~31 GB）。
- **epoch 墙钟**：3.12 s（含 epoch 末内部验证 + checkpoint 写盘）；整体 13.1 it/s、3383 窗/s（含 val/IO；纯训练 step 稳态 ~23 it/s）。
- **100-epoch 外推**：0.087 h ≈ **5.2 min**（3.12 s × 100；数据加载与启动 ~2 s 另计）。
- **四项损失走势**（每 5 step 采样，41 step 全程下降）：total 1.5122 → 1.3483（斜率 −4.2e-3/step）；mae 0.7973→0.7080、cons 0.6320→0.5642、tv 0.2353→0.2007、corr 0.5940→0.5608；epoch 均值 total 1.3925。无 NaN（NaN guard 未触发）。
- **t 分桶 sanity**：桶计数 [412,1931,2646,2180,1599,1015,504,187,22,0]——logit-normal(−0.8,0.8) 形状正确（低中段集中、t>0.9 零样本、t∈[0.8,0.9) 22 样本 ≈0.2%）；分桶 mae 随 t 单调下降 b0 1.009 → b8 0.208（近初始化权重下 (1−t) 缩放的预期效应，非病态）。
- 监控量：‖v‖ 均值 38.2、x̂1 std_t 0.686（诊断 GT future 窗内 std 0.833，量级正确）；内部验证（EMA 近初始化）total 5.5782。
- 代码 commit d2b01a3；config_hash 339a75571462；复现三要素（命令+env_snapshot.txt+resolved config.yaml）齐备。

### 结论

- **基准通过，建议直接进入 §5 fold0 单 seed 完整训练**：显存占用 0.73/32 GB（2.3%）、100 epoch 总时长 ~5 min——资源与时长均无瓶颈，无优化必要性的证据。
- 观察项（不阻塞，完整训练时验证）：①EMA 0.9999 时间常数 ~1e4 步 > 全程 4100 步，val-best 前期主要反映 EMA 追赶（Run 002/004 的 val≫train 即此机制）；②高 t 桶样本极少（设计使然），高 t 欠约束监控交给 §6 采样终点误差口径。
- **用户门禁决策：待用户确认**（本摘要已转呈用户：确认进入完整训练 / 先优化模型——未确认不启动 §5，决策后补记本块）。

> **勘误（2026-10-12，质检 Major-1）**：Run 002/003/004 的内部验证 total 为「t 网格求和」口径（×5）——Run 002 的 5.5964 与 Run 004 的 5.5782 对应登记口径（网格均值）≈1.119 / ≈1.116；同因子单调，best 选取与 A/B resume 一致性结论不受影响。修复（train.py 除以网格长度）与复验见 Run 005；本块原数值保留不改。

## Run 005 —— 质检修复复验（2026-10-12）

### 目的与假设

- trellis-check Major-1/Minor-2/Minor-4 修复后复验：①内部验证改「网格均值」口径（val total 应回到 ~1.1 量级）②EMA 权重下 x̂1 std 幅度监控字段生效 ③四组单测不受影响。

### 配置

- 代码：`models/gpfm-mamba-v1/train.py`（修复 commit 见结论）；单测 4 脚本
- 命令：单测逐个实跑 + `python3 models/gpfm-mamba-v1/train.py --epochs 1 --bench --run-id bench-1ep-fix`
- 环境：base conda @ 本地云服务器

### 结果

- 单测 4/4 PASS（test_data / test_gpr / test_model / test_losses 逐个实跑，退出码 0）。
- `bench-1ep-fix`：val **total 1.1156**（网格均值口径；= Run 004 的 5.5782 / 5 ✓），mae 0.5873 / cons 0.4678 / tv 0.2040 / corr 0.4010；**EMA 幅度监控字段生效**：xhat1_std_ema 0.7546 vs x1_std 0.8381；epoch 3.07 s / peak 0.728 GiB / 13.35 it/s，与 Run 004 一致（修复无性能影响）；config_hash 339a75571462 不变。

### 结论

- **通过**：质检 Major-1（网格均值）、Minor-2（EMA 幅度监控）、Minor-4（死变量）修复确认；Minor-3（implement.md σ 陈旧值）已勘误。新口径 val 数值自本 Run 起生效，与 Run 002–004 的 val 不可直接比较（见 Run 004 勘误注）。
