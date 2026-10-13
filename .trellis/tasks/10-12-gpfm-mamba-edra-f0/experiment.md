# 实验日志 — gpfm-mamba-edra-f0

> 每次「跑批/训练/评估/统计」追加一个 Run 块：跑前先写「目的+配置」，跑后同会话补「结果+结论」，不留空块。
> 格式契约见 `.trellis/spec/research/experiment-log.md`；产物只落本任务声明的变体目录（`.trellis/spec/research/path-isolation.md`）。

## Run 000 —— fold0 单 seed 完整训练（用户门禁已放行 · 2026-10-12 启动）

- **状态**：**已放行启动**（Run 004 门禁决策：进入 §5；数据口径 **(b) status_ok**；划分**固化**；EMA 0.999——AI 建议值，向用户解释后默认采纳，如需 0.9999 改一行 config 重跑）。run_id=`main-01`（nohup 后台），跑后在本块补结果。
- 目的：fold0 单 seed 前置验证训练（Study I 代理脑，gpfm-mamba-v1 变体）。
- 配置（已定，resolved config 随 run 落盘；config_hash 05f7a30c448d）：
  - 划分：受试者级 5-fold 分层（MDD/HC）split_seed=42，fold0；**subject_filter=status_ok（46 例：HC 24 / MDD 22，用户 2026-10-12 拍板 b；33 例 s7_interp_carry 软门失败者保留——管线作者标 ok）**；内部验证 2 例（25,40）；**训练折 34 例 44914 窗 / 内部验证 2642 窗 / 测试折 10 例（HC5/MDD5）13210 窗**（stride=64，每例 1321 窗）；**划分固化 manifest：`models/gpfm-mamba-v1/splits/fold0_status_ok_seed42.json`（此后漂移即报错）**
  - z-score：逐通道，仅训练折 34 例估计
  - 先验：K_t=OU γ=1 τ=i·π/p p=50 iso=1e-2（折叠进时间核）pp_jitter=1e-4；K_s=Ricker3D σ=0.080 m + jitter_s=1e-3
  - 模型：Mamba d_model=16 d_state=16 d_conv=4 expand=2 ×4 层 + 预 RMSNorm 残差；**EMA 0.999**（口径变更登记：论文 0.9999 的 τ≈10⁴ 步在本程 17.5k 步下前半程影子权重严重滞后，0.999 的 τ≈10³ 步≈5.7 epoch 匹配短程训练；观察项见 Run 002/004）
  - 损失：1·mae+1·cons+0.1·tv+0.1·corr（future 128 列，一步生成 x̂1=x_t+(1−t)v_θ）
  - 训练：seed=1234、AdamW lr=1e-3 wd=1e-4 cosine、clip 0.5、batch 256、epochs 100；t=sigmoid(N(−0.8,0.8²)) clamp [1e-4,1−1e-4]
  - best 判据：内部验证总 loss（固定 t 网格 {0.1,0.3,0.5,0.7,0.9}+固定 eps=seed 2026 确定性复算，均值口径）
  - 主指标（§6）：测试折 MAE（ODE 终点样本均值口径）
  - 命令：`nohup python3 models/gpfm-mamba-v1/train.py --run-id main-01 > /root/npi/outputs/gpfm-mamba-f0/main-01/logs/train.log 2>&1 &`（run 目录由脚本先建，日志重定向）

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
- **用户门禁决策（2026-10-12）**：**进入 §5 完整训练**；数据口径选 **(b) status_ok（46 例）**并**固化划分**；EMA 经解释后由 AI 定为 0.999（如用户后续要求 0.9999，属口径变更、重跑登记）。决策依据：Run 004 摘要 + manifest QC 勘察（33 例挂 s7_interp_carry 软门但 status=ok）+ 13 例口径测试折仅 3 例统计力弱。

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

## Run 006 —— 46 例口径 1-epoch 基准（2026-10-12）

### 目的与假设

- 用户门禁决策 (b)（status_ok 46 例）+ 划分固化 + EMA 0.999 落地后，重跑 1-epoch 基准锁定新口径数字（窗数/步数/时长/显存/外推），作为 Run 000 启动前提。

### 配置

- 代码：`models/gpfm-mamba-v1/{config,data,train}.py`（subject_filter='status_ok'、fixate_split=True、ema_decay=0.999；commit 见结论）
- 命令：`python3 models/gpfm-mamba-v1/train.py --epochs 1 --bench --run-id bench-1ep-46subj`
- 环境：base conda @ 本地云服务器

### 结果

- 划分（固化 manifest `splits/fold0_status_ok_seed42.json`）：train **34 例 44914 窗**（175 step/epoch）/ val 2 例 2642 窗 / test **10 例（HC5/MDD5）** 13210 窗；数据加载 4.2 s；z-score 仅训练折 34 例。
- epoch 墙钟 **8.78 s**、19.9 it/s、5117 窗/s；peak 显存 **0.728 GiB**；**100-epoch 外推 0.244 h ≈ 14.6 min**。
- 1 epoch 后：val total（均值口径）1.1320（mae 0.5816 / cons 0.4911 / tv 0.1916 / corr 0.4011）；x̂1 std(EMA) 0.741 vs x1 std 0.837；无 NaN。
- config_hash 05f7a30c448d；test_data.py 新口径断言全过（46/24/22、fold 平衡 [4,5,5,5,5]/[4,4,4,5,5]、固化 manifest 二次构建一致）。

### 结论

- **通过，Run 000（main-01）按此口径启动**。相对 13 例口径：窗数 ×4.25、步数 4100→17500（EMA 0.999 τ≈5.7 epoch 匹配）、时长 5.2→14.6 min，资源仍极宽裕（显存 2.3%）。

### Run 000 结果（2026-10-12 训练完成，评估见 Run 007）

- 100 epoch 全程无 NaN；mean 8.06 s/epoch，总时长 ~13.4 min；peak 显存 0.728 GiB；21.7 it/s；复现三要素齐备（config_hash c6ff80ac1c43、commit 45791bd、env_snapshot/pip freeze、启动命令在块首）。
- 训练曲线：train total 1.3254（ep1）→ 0.9918（ep100）；**val total（EMA、网格均值口径）1.1320 → 0.80226，best=epoch 100（cosine 末端）**；四项 val：mae 0.458 / cons 0.298 / tv 0.152 / corr 0.315。
- 幅度监控：x̂1 std(EMA) 0.757 vs x1 std 0.837（90.4%，tv 平滑压制轻微）。
- **R1 触发检查：高 t 桶 [0.8,1] mae / 中 t 桶 [0.4,0.7] mae = 0.33（阈值 1.5）→ 未触发**；分桶 mae 随 t 单调降（b0 0.764 → b9 0.098），高 t 段无欠约束迹象（logit-normal t 采样 + 一步生成的预期行为）。
- checkpoints：gpfm-mamba-f0-best.pt（ep100）+ gpfm-mamba-f0-last.pt；metrics.jsonl 100 行无缺口。

### Run 000 结论（训练部分）

- **训练收敛通过**：损失单调下降、best 判据工作正常（末 epoch 最优）、R1/幅度监控无异常、无 NaN。模型质量与代理脑有效性结论待 Run 007 采样评估与前置验证三指标（分开陈述）。

## Run 007 —— 前置验证评估（采样/MAE/GC/PSD/空间归因 · **待主会话执行**）

- **状态**：评估管线已实现并通过单测（tests/test_sample.py、tests/test_gc_psd.py 实跑退出码 0；既有 4 组单测回归通过；visualizer 5 类图合成数据渲染冒烟 + gc_psd.run_all 合成迷你数据端到端冒烟通过）；**未启动真实评估**（implement 6.1–6.4 实现完成即停止，真实评估由主会话执行下述命令，跑后在本块补结果/结论）。
- 目的：fold0 单 seed 前置验证（design §6）——Q1 预测质量（test-MAE 主指标 + CRPS/逐通道 r 次指标 + 持久性/GPR 均值双基线 + R1 触发监控）与 Q2 三检验（GC 连接一致性 / 人工视觉检验 / PSD 相似性）+ 指标 4 空间结构归因（R̂ vs K_s vs R），Q1/Q2 结论分开陈述。
- 配置：
  - 代码：`models/gpfm-mamba-v1/{sample,evaluate,gc_psd}.py` + 共享 `common/visualizer.py`（仓库共享区首建，nature-figure 契约风格、300 dpi PNG；实验代码内无散落绘图）；commit 见本 Run 结论补记
  - 采样：EMA 权重（best ckpt 的 `ema` state + config_hash 严格校验），Euler ODE t:0→1 共 16 步（cfg.euler_steps），从 GPR 条件先验 x0（+σ_min·ε）出发；每窗 K=16 条独立样本（seed=cfg.seed=1234）；GC/PSD 生成侧 = 每窗固定 seed 单样本（seed=1235，K=1 独立确定性 pass，顺序由测试折 loader 决定）
  - 主指标：test-MAE（**ODE 终点样本 x̂(1) 口径**：每窗 16 样本均值作点预测，非训练期 x̂1 公式）；次指标：CRPS（NRG 成对近似 mean|x−y| − 0.5·mean|x−x′|，分母 K² 含对角零，排序恒等式向量化已对 brute-force 逐点核验 max|Δ|≈9e-16）+ 逐通道 Pearson r 均值（近平坦守护与训练 corr 同口径）；基线同口径：持久性（past 末值复制 128）与 GPR 后验均值 μ=A·y^p
  - R1 触发：高 t 桶 [0.8,1]/中 t 桶 [0.4,0.6] mae ≥1.5（metrics.jsonl 末期 5 epoch 计数加权），或终点 MAE ≥ 训练末期 x̂1-MAE ×1.5（口径注记：test vs train 折不同数据，按 design §9.1 执行并如实标注）→ eval.log 告警行
  - GC：future-only，两组参数完全一致；全模型 VAR(p) OLS（受试者级 Gram 聚合，避免整折设计矩阵物化）；**BIC 真实组选 p∈{1..10} 两组共用**（对齐行集 t∈[p_max,F) 嵌套子矩阵口径）；成对条件 GC = 删列额外平方和（块 Schur 补：SSR_r[i,i] = SSR_full[i,i] + B[cols_j,i]ᵀ(G⁻¹)[cols_j,cols_j]⁻¹B[cols_j,i]，不做 3782 次全量重拟合）；GC 量 = ln(SSR_r/SSR_full)，F 检验 raw p（numpy 手写 betainc，已对 7 个教科书 F/t 临界值核验 |Δ|<5e-3）→ BH-FDR（numpy 排序实现；各组 3782 有向边 q<0.05 数 + 两组 Jaccard）；矩阵一致性 Pearson r + Spearman ρ（秩变换后 Pearson）+ raw p + 95% CI——**bootstrap 单元 = 受试者级**（1000 次、seed 2026、两组配对同重采样索引；对 design「对窗 bootstrap」的登记偏离：逐窗 Gram 存储需 ~20 GB 不可行，且 stride=64 相邻窗高度相关、受试者才是独立单元）
  - PSD：两组完全相同分段与拼接——互不重叠 future 段（窗 start ≡ 0 mod 2·stride）逐受试者拼接、丢前导残段对齐绝对 512 点网格 → Welch（Hann，nperseg=512，fs=500 → 0.977 Hz≈0.98 Hz）
  - 空间归因（指标 4）：R̂（生成单样本全折零滞后相关）vs K_s（Ricker3D σ=0.080 加 jitter 前原始核）vs 真实 R：Frobenius ×3 + 上三角 Pearson/Spearman ×3 + 数值表
  - 产物：`outputs/gpfm-mamba-f0/main-01/` 下 `logs/eval.log` + `logs/eval_metrics.jsonl`（R9 留存）+ `eval/`（summary.json、gc_input.npz（gen/real (N,62,128) f32+窗元数据）、gc_{real,gen}.npy、gc_F_p_*.npz、psd.npz、spatial_corr.npz、gc_psd_results.json）+ `figs/`（轨迹叠加/PSD/GC 矩阵/训练曲线/空间三方共 5 类，统一 visualizer）
  - 命令：`python3 models/gpfm-mamba-v1/evaluate.py --run-id main-01`

### Run 007 结果（2026-10-12 评估完成；checkpoint=main-01 best（ep100，EMA））

- **Pass A（Q1 预测质量，主指标 test-MAE=ODE 终点样本均值口径，13210 窗 × K=16）**：model **0.7109** | persistence 0.9527 | **gpr_mean 0.6873**；CRPS(model) 0.5293；逐通道 Pearson r：model 0.1421 / gpr_mean 0.1678 / persistence 0。——模型显著优于持久性，但 MAE 与 r 均略逊于 GPR 后验均值基线（差 3.4%）；与适配研究预期一致（EEG 线性可预测上限低，GP 先验=结构化基分布）。单 seed 初步观察。
- **R1 触发检查（§6.2 口径）**：高/中 t 桶比 0.377（阈值 1.5）、终点/训练末期比 1.248（阈值 1.5）→ **未触发**。
- **指标 1（GC 一致性，future-only，BIC 选 p*=4 两组共用，受试者级 bootstrap 1000 次 seed=2026）**：Pearson r=**0.2820**（95%CI 0.193–0.336）| Spearman ρ=**0.1932**（0.130–0.248）；显著边（BH-FDR q<0.05）真实 3782/3782 vs 生成 3777（Jaccard 0.999，该维度判别力弱）；生成 GC 幅值范围约为真实一半（max 0.0124 vs 0.0306）。
- **指标 3（PSD，future 拼接 Welch nperseg=512，0.98 Hz，两组同分段）**：峰频率一致（均 0.98 Hz）；全通道平均谱分频段 |Δlog10|：delta 0.09 / alpha 0.15（主节律段高度相似）/ theta 0.30 / beta 0.34 / 30–60 Hz 2.86 / 60–120 Hz 9.0（60–250 Hz 全带 10.4；勘误：原记「>60 Hz」，未定义上界，据质检复算补边界）——**低频段谱形匹配良好，>30 Hz 生成功率显著偏高**（真实信号高频已被 v4 滤波压至近零，log 地板放大高段差异；白噪地板来自先验 iso 与网络未完全去噪）。
- **指标 4（空间结构归因）**：R̂~R r=**0.952** > K_s~R 0.891 > R̂~K_s 0.809；Frobenius R̂−R 10.79 < K_s−R 11.11——生成的零滞后空间相关比先验 K_s **更接近**真实 R：网络在先验基础上把空间结构向真实重塑（非先验复读；K_s 与 R 本身 0.891 相关，归因部分共享）。
- **指标 2（人工视觉检验）**：轨迹图 `figs/eval_trajectories_main-01.png`（多窗真实 vs 生成多轨迹叠加）——**待用户判定，AI 不代判**。
- 图产物（统一 visualizer，300 dpi）：eval_trajectories / eval_training_curves / eval_gc_matrix / eval_psd / eval_spatial_corr；数值 json+npy 落 eval/；eval.log + eval_metrics.jsonl 落 logs/。

### Run 007 结论

- **前置验证数值与图全部落盘**（fold0 单 seed）；Q1 与 Q2 分开陈述：Q1 模型可预测性中等（优于持久性、未超线性 GPR 基线）；Q2 指标 1 中等正相关（r≈0.28）、指标 3 低频相似/高频偏高、指标 4 显示网络重塑空间结构、指标 2 待人工检验。**指标 1/3 通过标准为占位，由用户结合数值与图判定；未通过判定前不宣称「代理脑合格」，不进入 5-fold、不固定 θ\***（门禁：surrogate-brain-qualification.md §判定流程 2/3）。
- 已知局限（下一轮候选优化项，均须新变体/新 Run 登记）：生成高频功率偏高（tv 权重/先验 iso/网络去噪）；GC 幅值整体偏低约一半（生成动态平滑）；Q1 未超 GPR 基线（采样均值口径 vs 确定性线性外推）。

## Run 008 —— 指标 1 口径变更 v2：Pearson FC 矩阵一致性（2026-10-12 用户指定）

### 目的与假设

- 用户定义指标 1 采用经典无向 FC：用 **Pearson 相关**计算 FC 矩阵（真实组、生成组各一、完全相同流程），再对两个 FC 矩阵（上三角展平）计算 **Pearson r 与 Spearman ρ**。Run 007 的 GC 结果保留为旧口径补充参考（evaluation-calibers 变更纪律：旧 Run 原口径不动）。

### 配置

- 代码：`models/gpfm-mamba-v1/fc_metric.py`（本 Run 首建，commit 见结论）
- 输入：Run 007 `eval/gc_input.npz`（生成侧 = gc_seed 1235 确定性 K=1 样本——与 GC 口径同输入，保证「同输入、不同度量」可比）
- 口径：逐受试者 FC（该受试者全部 future 段拼接 → 62×62 Pearson）→ 组内 10 例平均 → 每组一个 FC 矩阵；上三角 1891 项（对角剔除）→ Pearson/Spearman；受试者级**配对** bootstrap 1000 次 seed=2026 → 95% CI
- 命令：`python3 models/gpfm-mamba-v1/fc_metric.py --run-id main-01`

### 结果

- **Pearson r = 0.95791**（95%CI 0.94082–0.96094）；**Spearman ρ = 0.97033**（95%CI 0.95250–0.97360）；10 受试者 / 13210 窗 / 1891 上三角边；耗时 4.9 s；产物 `eval/fc_metric.json` + `eval/fc_matrices.npz`（FC_real/FC_gen）。
- 交叉验证：与 Run 007 指标 4 的池化口径 R̂~R（r=0.952 / ρ=0.968）量级一致（口径差异 = 池化 vs 受试者平均，预期内）。

### 结论

- **v2 口径下 FC 一致性很高（r≈0.96、ρ≈0.97，CI 稳定）**；通过标准仍由用户判定。归因语境必须随附：零滞后 FC 结构**部分来自 Ricker 空间先验**（指标 4：K_s~R=0.891）——生成侧 FC 相似不能全部归功网络；有向动力学保留的证据见 GC 补充参考（r=0.282 [0.193,0.336]）。口径变更已回填 design §6 与 surrogate-brain-qualification.md（v2 注记）。

## Run 009 —— 指标 1 显著性口径 v2.1：bootstrap CI → t 检验（2026-10-12 用户指定）

### 目的与假设

- 指标 1 的置信度口径由「受试者级配对 bootstrap 95% CI」（v2，Run 008）改为 **t 检验显著性 p**（v2.1）：逐受试者一致性值 r_s/ρ_s → 单样本双侧 t 检验（H0: 均值=0，df=S−1=9）。点估计不变（组平均矩阵的 r/ρ）。

### 配置

- 代码：`models/gpfm-mamba-v1/fc_metric.py`（复用 gc_psd.t_two_sided_p，Lentz incomplete beta）；输入同 Run 008（gc_input.npz，生成侧 seed 1235 K=1）
- 命令：`python3 models/gpfm-mamba-v1/fc_metric.py --run-id main-01`

### 结果

- 点估计不变：r=0.95791 / ρ=0.97033（组平均矩阵，1891 上三角边）。
- 逐受试者 r_s：mean **0.9163 ± 0.0307**（范围 0.845–0.955，10/10 受试者均强正）；ρ_s：0.9292 ± 0.0333。
- **单样本双侧 t 检验**：Pearson t=94.40，df=9，**p=8.5e-15**；Spearman t=88.35，df=9，**p=1.5e-14**。
- 产物：`eval/fc_metric.json`（覆盖为 v2.1 口径，含逐受试者值）+ `fc_matrices.npz`；耗时 4.2 s。

### 结论

- **指标 1（v2.1 口径）：r=0.958 / ρ=0.970，t 检验高度显著（p≈1e-14/1e-15，df=9）**——10 例受试者逐例一致性均强正，结论稳健。v2 的 bootstrap CI 数值保留于 Run 008 原口径不动。口径变更已回填 spec（指标 1 v2.1 注记）与 design §6。通过标准仍由用户判定。

## Run 010 —— v2 d_model=62 消融（薄变体 · 2026-10-12 用户指定）

### 目的与假设

- **单变量消融**：d_model 16→62（=通道数），验证「16 维跨通道瓶颈是否为 GC 偏弱主因」（Run 007：GC r=0.282 [0.193,0.336]、生成 GC 幅值约为真实一半；design §9.7 / adaptation R6 预案）。假设：若瓶颈是主因，v2 的 GC 一致性（补充参考）与有向幅值应显著改善；若 GC 仍偏弱，则瓶颈假说不成立，原因在损失/训练目标（四项损失均逐通道分解、无跨通道时序监督）或数据本身。
- **单变量原则**：唯一变化 = d_model=62（config 其余字段与 v1 逐字段一致：n_layers=4、d_state=16、d_conv=4、expand=2、time_emb_dim=16、初始化方式、EMA 0.999、损失权重 {1,1,0.1,0.1}、t~logit-normal(−0.8,0.8)、训练超参、评估口径）；不做恒等初始化等额外变量。参数量 15,856 → 132,682（×8.4）。

### 配置（跑前登记；结果见下方 Run 010 结果）

- 代码：`models/gpfm-mamba-v2/`（薄变体：config.py 完整副本改 d_model=62 与 variant_name/run 命名 + train.py/evaluate.py 入口包装——sys.path 令 v2 config 遮蔽 sys.modules['config']，data/gpr_prior/mamba/model/losses/train/sample/evaluate/gc_psd **只读复用 v1**，v1 代码零改动；design §7 v2 行）；评估侧 config_hash 校验经包装内 build_config（v2 默认=训练落盘 resolved config）严格通过，config_hash `8a26fb38cc66`（v1 默认 05f7a30c448d，不同 ✓）
- 划分：与 v1 同 subject_filter=status_ok / split_seed=42 / fold0 / 内部验证 (25,40) ⇒ data.fixate_split 校验到 v1 的 `models/gpfm-mamba-v1/splits/fold0_status_ok_seed42.json`（跨变体共享只读，不重写）；z-score 仅训练折 34 例（与 v1 同统计量）
- 训练：seed=1234、AdamW lr 1e-3 wd 1e-4 cosine、clip 0.5、EMA 0.999、batch 256、epochs 100；run 目录前缀强制 `v2d62-`（包装守卫），checkpoint `gpfm-mamba-v2d62-{best,last}.pt`
- 冒烟（v2d62-smoke-a）：小子集 4 例 1 epoch，四项损失有限下降（total 1.4803→1.2836，20 step），val total 1.2234，无 NaN——机械正确性通过；v1 既有 6 组单测回归全 PASS
- 1-epoch 基准（v2d62-bench-1ep，全量训练折 34 例 44914 窗 175 step）：epoch 墙钟 **9.45 s**（v1 8.78 s，+7.6%）、**18.53 it/s**（v1 19.9）、**peak 显存 1.942 GiB**（v1 0.728，仍 32 GB 卡的 5.9%）、**100-epoch 外推 0.262 h ≈ 15.7 min**（v1 14.6 min）；1 epoch 后 val total（均值口径）1.1075（v1 同期 1.1320）
- 环境：base conda @ 本地云服务器（4080S 单卡独占）
- 数据：`/root/autodl-tmp/edra_process_v4/`（只读）
- 命令：`nohup python3 models/gpfm-mamba-v2/train.py --run-id v2d62-main-01 > /root/npi/outputs/gpfm-mamba-f0/v2d62-main-01/logs/train.log 2>&1 &`
- 评估（训练完成后同口径）：`python3 models/gpfm-mamba-v2/evaluate.py --run-id v2d62-main-01`（Pass A/B + GC + PSD + 空间归因 + 全部图）+ `python3 models/gpfm-mamba-v1/fc_metric.py --run-id v2d62-main-01`（指标 1 v2.1；fc_metric 硬编码共享 outputs 根 + run_id 形参，天然兼容 v2 run）
- 指标口径：与 Run 007/009 完全一致——主指标 test-MAE（ODE 终点 K=16 样本均值）、CRPS/逐通道 r、持久性/GPR 均值基线、指标 1 v2.1（FC Pearson r/ρ + 逐受试者单样本 t 检验）、GC 旧口径补充参考、PSD future-only Welch、空间归因三方对比；Q1 与 Q2 分开陈述

### Run 010 中止记录（2026-10-12）

- 完整训练 `v2d62-main-01` 进行至 **epoch 56/100 时被用户指令中止**（进程已终止、GPU 已释放）。中止时 train total ≈0.79（v1 同期 ep56 ≈1.02）、无 NaN、metrics.jsonl 无缺口；`gpfm-mamba-v2d62-{best,last}.pt` 保留在 epoch 55 附近，**可随时 `--resume` 续跑**（R9 断点续传已验证）。
- 本块结果/结论留空待续：若后续恢复训练则在本块补齐（同一 Run 延续）；若放弃 v2 消融，则在本块补记放弃决定与理由后关闭。评估未执行。

### Run 010 续跑登记（2026-10-13 · 用户指令「继续当前任务」）

- 断点核对：last.pt = epoch 54 末（global_step 9625、best_val 0.7417），metrics.jsonl 完整至 epoch 54、无缺口；epoch 55 step 131/175 处被 kill（train.log 尾行佐证）。
- 续跑：`--resume` 自 epoch 55 续至 100（同一 Run 延续；metrics.jsonl 由脚本截齐至 epoch 54 后续写；train.log **追加**不覆盖）。预计 ~8 min（45 epoch × 9.45 s + 启动/内部验证）。
- 命令：`nohup python3 models/gpfm-mamba-v2/train.py --run-id v2d62-main-01 --resume /root/npi/outputs/gpfm-mamba-f0/v2d62-main-01/checkpoints/gpfm-mamba-v2d62-last.pt >> /root/npi/outputs/gpfm-mamba-f0/v2d62-main-01/logs/train.log 2>&1 &`
- 跑后流程（本块补齐）：训练检查（loss/无 NaN/best 判据/R1）→ `python3 models/gpfm-mamba-v2/evaluate.py --run-id v2d62-main-01` → `python3 models/gpfm-mamba-v1/fc_metric.py --run-id v2d62-main-01`（与 Run 007/009 完全同口径）→ v1/v2 对比结论。

### Run 010 结果（训练 · 2026-10-13 续跑完成）

- 续跑 45 epoch（ep55–99，1-based 56–100）无中断：mean 8.27 s/epoch（段总 ~6.2 min）、21.16 it/s、peak 显存 1.942 GiB（与基准一致）；无 NaN（guard 未触发）；metrics.jsonl **100 行无缺口**（0–99 连续）；config_hash `fc2a9ce14948`（best/last/checkpoint 三方一致；Run 010 配置块登记的 8a26fb38cc66 为 bench(--epochs 1) 口径 hash，本行为 resolved epochs=100 口径，勘误注记）；code_commit a168d7d。
- 训练曲线：train total 中止前 ep55 ≈0.79 → ep99 0.7734（微降平台）。
- **内部验证（EMA、网格均值）在 ep53（0-based）触底 0.74167 → best=epoch 53**；ep54–99 缓慢单调回升至 0.7471（**轻微过拟合拐点**）。与 v1 对照：v1 val 单调下降 best=ep100（0.80226）——v2 收敛更快（早 ~45 epoch 触底）、内部验证 loss 低 7.5%，但 100 epoch 对 v2 已过头（后续若采纳 v2，epochs 可减或加正则，属新变体决策不在此展开）。
- best/last checkpoint 就位（best=ep53 EMA，评估按 best 口径）。评估见下方 Run 010 评估结果。

### Run 010 评估结果（2026-10-13 · checkpoint=best ep53 EMA；与 Run 007/009 完全同口径）

- **Pass A（Q1，13210 窗 × K=16）**：model test-MAE **0.70375**（v1 0.71089，−1.0%）| persistence 0.95271 | gpr_mean 0.68729；CRPS 0.52947（v1 0.52926，持平）；逐通道 r 0.14885（v1 0.1421，+4.7%；GPR 0.16782）。——Q1 改善边际，**仍未超 GPR 线性基线**。
- **R1 触发检查：触发**（高/中 t 桶比 0.403<1.5 未触发；**终点/训练末期比 1.532 ≥1.5 触发**，eval.log 告警行已落）。口径注记：分子=test 折 ODE 终点、分母=train 折末期 x̂1-MAE（不同折数据）；触发主因是 v2 大模型（参数 ×8.4）把训练折拟合得更深（train 末期 mae 0.4593 vs v1 0.5698，分母变小），而终点 MAE 本身略优于 v1（分子未变差）——**属「大模型 train 拟合更深」的口径效应，非采样质量劣化证据**；按 §6.2 如实报告并提请用户知悉，不单独构成否决。
- **指标 1（v2.1）**：Pearson r=**0.97572** / Spearman ρ=**0.97826**（v1 0.95791/0.97033）；逐受试者 r 0.9373±0.0170（v1 0.9163±0.0307，离散更小）；t 检验 p=3.4e-17 / 5.1e-17（v1 8.5e-15/1.5e-14）。
- **GC（旧口径补充参考，p*=4 两组共用同 v1）**：Pearson r=**0.2314**（95%CI 0.172–0.260）| Spearman ρ=0.1994——**低于 v1 的 0.2820 [0.193,0.336]**；显著边 3782/3781（Jaccard 1.000）；生成 GC 幅值 max 0.0078（真实 0.0306 的 25%；v1 为 40%）——生成动态进一步平滑。
- **指标 3（PSD 同分段）**：峰频率一致 0.98=0.98 Hz；频段 |Δlog10|（本次与 v1 同口径复算，v1 复算值与 Run 007 登记一致；alpha 单点 0.12 vs 登记 0.15 为频段边界取法差异）：delta 0.121 / theta 0.315 / alpha 0.123 / beta 0.380 / 30–60 Hz 2.949 / 60–120 Hz 9.145 / 60–250 Hz 10.422——与 v1（0.090/0.297/0.124/0.349/2.901/9.023/10.356）**各频段几乎一致**：低频谱形匹配良好、>30 Hz 生成功率偏高的问题 d_model 不敏感。
- **指标 4（空间归因）**：R̂~R r=**0.969**（v1 0.952）、Frobenius R̂−R **6.712**（v1 10.79）——空间结构显著更贴近真实；R̂~K_s 0.860（v1 0.809）同步升高（大容量下先验印记与真实结构更难区分，归因共享不变）。
- **指标 2（人工视觉检验）**：轨迹图 `figs/eval_trajectories_v2d62-main-01.png`——**待用户判定，AI 不代判**（v1 对照图 `main-01/figs/eval_trajectories_main-01.png`）。
- 产物：`eval/summary.json、gc_input.npz、gc_{real,gen}.npy、psd.npz、spatial_corr.npz、fc_metric.json、gc_psd_results.json` + `figs/` 5 类图；`logs/eval.log`（含 R1 告警行）+ `eval_metrics.jsonl`（8 行）。

### Run 010 结论

- **瓶颈假说不成立（本消融的单变量证据）**：d_model 16→62（参数 ×8.4，其余逐字段不变）后，GC 一致性不升反降（r 0.282→0.231，CI 不重叠）、生成 GC 幅值进一步压低（真实幅值的 40%→25%）——「16 维跨通道瓶颈是 GC 偏弱主因」被否定；GC 偏弱更可能源于**损失/训练目标**（四项损失逐通道分解、无跨通道时序监督）或数据本身（design §9.7 备择解释，未检验）。
- 次要发现：①Q1 改善边际（MAE −1%、逐通道 r +4.7%，未超 GPR 基线）；②FC（指标 1）与空间结构（指标 4）略优且更稳；③v2 收敛更快但 100 epoch 过头（best ep53 后轻微过拟合）；④R1 触发属 train 拟合加深的口径效应（见上方注记）。
- 单 seed fold0 初步结论，不外推；指标 1/3/4 的通过标准与「v1 还是 v2 进 5-fold（或均不进、先改损失）」由用户判定——**未判定前不宣称代理脑合格、不进入 5-fold、不固定 θ\***。

## Run 011 —— 轨迹图可视化口径变更：medoid 代表轨迹 + 90% 预测区间（2026-10-13 用户指定）

### 目的与假设

- 原 trajectories 图把 K' 条生成轨迹全部叠加（红细线 α=0.45），波形严重重叠，难以观察典型动态与不确定性。用户指定新口径：①**不再画全部 K 条**，也**不用逐时间点均值**当代表——从 K 个实际生成样本中按完整多通道轨迹 pairwise RMSE 选 **medoid** k\*=argmin_k(1/(K−1))Σ_{j≠k}d(x_k,x_j)，只画该实际样本；②**90% predictive interval**：全部 K 样本逐 (通道,时间点) 的 Q05–Q95 半透明阴影带（颜色同 medoid 红、低透明度；不用 min-max envelope）；③ground-truth future 蓝线保留；版式/轴/配色不变。**这是指标 2（人工视觉检验）载体的口径变更，不改变任何数值指标。**

### 配置

- 代码：`common/visualizer.py::_draw_trajectories`（新数据契约 gen_medoid/gen_q05/gen_q95，版式不变）+ `models/gpfm-mamba-v1/evaluate.py`（新增 `trajectory_summary`：全 K=16 样本·**全通道**轨迹 pairwise RMSE 选 medoid（RMSE=Euclidean/√(CF)，argmin 与常数无关）+ 沿 K 维逐点 Q05/Q95；图窗收集改存摘要；新增 `--figures-only` 确定性重采样仅重渲染轨迹图，不写 eval_metrics/summary）；v2 经薄变体包装自动同口径。TRAJ_SAMPLES 常量删除（不再截取 K' 条）
- 核验（跑前完成）：medoid k\* 与 numpy brute-force 逐例一致；medoid 输出与被选实际样本逐位相等（非重构波形）；q05/q95 与 np.percentile(linear) allclose；合成数据渲染烟雾通过；既有 6 组单测回归全 PASS
- 重渲染命令（同 seed=1234 同 loader 顺序 → 与 Run 007/010 Pass A 样本逐位一致）：
  - `python3 models/gpfm-mamba-v1/evaluate.py --run-id main-01 --figures-only`
  - `python3 models/gpfm-mamba-v2/evaluate.py --run-id v2d62-main-01 --figures-only`
- 旧图保留：`eval_trajectories_{run}.png` → `eval_trajectories_{run}.k-overlay.png`（旧口径不覆盖销毁，本块结论后以新图为指标 2 判定载体）

### Run 011 结果（2026-10-13）

- v1 `main-01` 与 v2 `v2d62-main-01` 均重渲染完成（同 seed 确定性采样 13210 窗：v1 67 s / v2 233 s，纯采样无指标累积）；eval.log 追加 `[figures-only]` 行，eval_metrics.jsonl/summary 未触碰。
- 新图：`main-01/figs/eval_trajectories_main-01.png`、`v2d62-main-01/figs/eval_trajectories_v2d62-main-01.png`；旧叠加口径图保留为同目录 `*.k-overlay.png`。
- 目视 QA（两图逐项）：36 面板 medoid 红线与 90% PI 阴影带全部清晰、蓝色真实曲线可辨、图例 4 项完整（past / real future / generated medoid (K=16) / 90% predictive interval）、ch/win 标签正常、无空白/重叠/缺线缺陷；阴影带宽度随窗变化（不确定性有区分度）。

### Run 011 结论

- **轨迹图口径切换完成**：全部 K 叠加 → medoid 实际样本（全通道 pairwise RMSE 选取）+ 逐点 Q05–Q95 阴影带；数值指标零改动，仅指标 2 判定载体更新。指标 2 人工视觉检验请以两张新图为对象（v2 图可见生成动态与不确定性区分度），判定仍归用户。

## Run 012 —— 诊断分析：生成 EEG 高频（40 Hz+）功率偏高根因（2026-10-13 用户提问驱动）

### 目的与假设

- 现象：真实侧 v4 滤波后 >40 Hz 功率近零，生成侧 30–60/60–120/60–250 Hz |Δlog10| ≈ 2.9/9.1/10.4（Run 007/010，v1/v2 几乎相同）。**候选假设**：H1 GPR 条件先验 iso=1e-2 白噪折叠 → x0 自带平坦噪声底（Run 001 已证 post_std≈0.97、条件化不降方差）；H2 网络流未充分去噪/额外引入高频（需 x0 vs 终点对比）；H3 tv 权重 0.1 抑制不足（损失结构论证，无重训不可直接检验）；H4 Euler 16 步离散化伪影（步数敏感性检验）；H5 log 刻度放大（真实为数值地板，非生成绝对功率大——需方差占比定量）。
- 判据设计：逐环节 PSD 分解（real / 后验均值 μ / x0(iso=1e-2) / x0(iso=0) / ODE 终点@{8,16,32,64} 步）→ 高频方差占比与频段 PSD；全量 gc_input 生成信号 30/40 Hz 低通后 MAE 变化（高频成分是否纯噪声）。

### 配置

- 代码：`models/gpfm-mamba-v1/diag_hf.py`（本 Run 首建；只读复用 sample/evaluate 装载函数；诊断 seed=2027；不改变体 config）
- 输入：v1 `main-01` best ckpt（EMA）+ 测试折前 512 窗（确定性 loader）；v1/v2 `eval/gc_input.npz` 全量 13210 窗（K=1 生成侧，低通检验）
- 口径：逐 (窗,通道) 128 点 Hann 周期图平均（分辨率 3.906 Hz，各元素同法处理内部可比；与 Run 007 拼接 Welch 口径不同、只做机制归因不做指标比较）；频段 {40–100, 100–250, >40}；方差占比 = 频段功率/全带功率
- 低通检验：rFFT 砖墙滤波（>30/>40 Hz 置零）后与真实 future 的 MAE，对照未滤波 MAE（v1 0.7109 / v2 0.70375 为锚）
- 数据事实核对：edra v4 npz `filter` 字段实读（契约中「待核对」项）+ 真实谱拐点经验估计
- 命令：`python3 models/gpfm-mamba-v1/diag_hf.py --run-id main-01`；产物 `outputs/gpfm-mamba-f0/diag-hf/`

### Run 012 结果（2026-10-13）

- **数据事实（契约回填）**：v4 滤波字段实读 = **`bp1-40+notch50(保真冗余)`**（受试者 1 抽样；已回填 `.trellis/spec/data/edra.md` §3）——真实数据 1–40 Hz 带通 + 50 Hz 陷波；全量实测：真实 >40 Hz 方差占比 **0.6%（512 窗）/ 0.8%（全量）**、>100 Hz 占比 **2e-9**（数值地板，非物理内容）；真实谱拐点经验值 ≈47 Hz（40 Hz 截止的滚降裙边）。
- **逐环节 PSD 分解**（512 窗，Hann 周期图，>40 Hz 方差占比）：

| 环节 | >40 Hz 占比 | 40–100 占比 | 100–250 占比 | 解读 |
|---|---|---|---|---|
| real | 0.58% | 0.58% | 2e-9 | 带通后近零 |
| μ（GPR 后验均值） | ~0（方差总量仅为真实 0.2%） | ~0 | ~0 | 线性部分与现象无关 |
| **x0（先验样本，iso=1e-2）** | **10.4%** | 6.3% | 4.2% | **主源** |
| x0（iso=0，仅诊断构造） | 9.7% | 6.2% | 3.5% | OU 尾巴本身≈9%；iso 白噪底贡献≈1% |
| ODE 终点 ep16 | 7.6% | 6.3%（持平） | 1.3%（−70%） | 流清除了 >100 Hz 的 70%，40–100 几乎未动 |
| ep8 / ep32 / ep64 | 7.9 / 7.5 / 7.5% | — | — | 步数不敏感（H4 否定） |
| gen_v1（全量 K=1） | 6.9% | 5.8% | 1.1% | 与 ep16 一致 |
| gen_v2（全量 K=1） | 8.0% | 6.7% | 1.3% | 与 v1 同量级（架构无关） |

- **低通检验（全量 13210 窗，K=1 生成侧）**：30 Hz 砖墙低通后 MAE v1 −4.8%（1.0498→0.9997）/ v2 −5.4%（1.0720→1.0140）；40 Hz 低通 −2.1%/−2.5%——高频成分对点预测**净有害**（是噪声而非信号）。
- OU 时间核谱宽理论核验：k(d)=e^{−γ|d|}、γ=1、τ 网格 Δτ=π/50 ⇒ Lorentzian 角频率 ≈5 Hz、1/ω² 尾巴 ⇒ 理论 >40 Hz 方差 ≈4%（实测 6.2%，含条件协方差接缝效应，量级吻合）。

### Run 012 结论（根因归因，按贡献排序）

1. **主因（H1′，修正版）：GPR 先验时间核本身是宽带过程**——OU(γ=1) 的 1/ω² 谱尾巴在 >40 Hz 保留 ~9% 方差（iso=0 依旧），与真实数据 40 Hz 硬截止的口径不对称；原假设「iso 白噪底是主因」被 iso=0 对照否定（iso 仅贡献 ~1%，次要）。
2. **次因（H2b）：网络流去噪不完全**——终点对 >100 Hz 清除 70%（训练 MAE 目标确实推动去噪）、但 40–100 段几乎原样保留（psd 10.2→11.2 持平）；且 v1/v2 同量级（容量无关）、Euler 8→64 步不敏感（H4 离散化伪影否定）——残余反映训练压力不足而非数值机制。
3. **放大器（H5）：log 刻度 + 真实侧数值地板**——生成 >100 Hz 绝对量仅 ~1.2% 方差，但真实侧 1e-9 级地板使 |Δlog10| 达 7–10，视觉冲击远大于绝对量。
4. **μ 无关**；**σ_min=1e-4 端点噪声（1e-8 方差）可忽略**。
5. **H3（tv 权重 0.1 抑制不足）**：间接证据支持（网络在无数据方差频段仍保留先验底），但严格检验须重训消融，未执行——列为后续变体方向。
- **缓解方向（供用户决策，均属口径/变体变更须登记）**：①生成侧后处理 40 Hz 低通（立得 MAE −5%，零训练成本）；②先验侧换 Matérn-3/2（尾巴 ω⁻⁴，注册表已备）或频域截止 40 Hz 的带限核（口径对齐数据）；③iso 降至 1e-3（收益 ~1%，小）；④tv 权重↑/显式频带损失（须重训）。

## Run 013 —— 生成侧 40 Hz 低通 + 轨迹图重渲染（2026-10-13 用户指定）

### 目的与配置

- 用户决策采纳 Run 012 缓解方向①：**轨迹图生成侧样本先 40 Hz 低通、再选 medoid/算 90% PI**，重渲染 v1/v2 轨迹图（仅可视化口径，不改数值指标）。
- 实现：`evaluate.py --gen-lowpass 40`（`lowpass_last_dim`：末维 rFFT 砖墙 >40 Hz 置零，**reflect 填充 ±64 点后滤波再裁剪**——抑制 DFT 循环卷积段边界振铃，实测边缘/中段偏差比 1.56→1.09、移除方差 7.0%；与 Run 012 无填充诊断口径的差异为填充本身）；visualizer 图例 medoid 项追加 ", lowpass 40 Hz" 注记；滤波在 medoid 选取与分位计算**之前**作用于每窗全部 K 条样本。
- 核验（跑前）：合成信号 >40 Hz 频点相对幅值 3.7e-8（置零生效）；真实 gc_input 子集 MAE 0.9619→0.9389（低通净收益保持）；py_compile 通过。
- 命令（--figures-only 确定性采样 + 滤波）：
  - `python3 models/gpfm-mamba-v1/evaluate.py --run-id main-01 --figures-only --gen-lowpass 40`
  - `python3 models/gpfm-mamba-v2/evaluate.py --run-id v2d62-main-01 --figures-only --gen-lowpass 40`
- 图产物保全：未滤波 medoid+PI 版保留为 `eval_trajectories_{run}.unfiltered.png`，滤波版写 canonical 名。

### Run 013 结果（2026-10-13）

- v1/v2 均重渲染完成（确定性采样 + 40 Hz 低通，v1 68 s / v2 233 s）；canonical 名 = 滤波版：`main-01/figs/eval_trajectories_main-01.png`、`v2d62-main-01/figs/eval_trajectories_v2d62-main-01.png`；figs/ 内并存三个版本（滤波 / `.unfiltered` / `.k-overlay`），eval.log 追加 `[figures-only]` 行。
- 目视 QA（两图逐项）：medoid 曲线平滑（与 past 段高频形成可见对比，滤波生效）、位于 90% PI 带内、蓝色真实曲线清晰、**future 段起止处无振铃/突变伪影**（reflect 填充生效）、图例 4 项含 "generated medoid (K=16), lowpass 40 Hz" 注记（大图小字号易读漏，裁剪放大核实）、无空白/缺陷面板。

### Run 013 结论

- 轨迹图（指标 2 判定载体）现为 **40 Hz 低通版**：medoid 展示的是滤波后样本中按滤波轨迹 pairwise RMSE 选出的实际样本，90% PI 带为滤波样本逐点 Q05–Q95；数值指标口径未动。若后续采纳「生成侧低通」为正式产物口径，须按口径变更回填 evaluation-calibers 并重跑全量评估；当前仅为可视化侧变更。

## Run 014 —— 设计可行性探针：频域硬截止 40 Hz 带限核（2026-10-13 用户提问驱动；只读探针、无产物落盘、不修改生产代码）

### 目的与配置

- 用户问「若采用频域硬截止 40 Hz 带限核应怎么修改」。跑前探针回答两个问题：①数值可行性（Toeplitz 条件数/Cholesky/有效秩）；②先验行为是否退化（post_std、GPR 均值 MAE、x0 谱）。口径：与 gpr_prior 同构网格（T=512/τ=iπ/50，θ_c=2π·40/500）；平顶砖墙 sinc 核、训练折 8 例实测谱匹配核（LP40 与 BP1-40 两版）、谱匹配+观测噪声条件化（σ_n² 加在 pp 侧）四方案对照；GPR 均值 MAE 用测试折前 3 例（3963 窗，main-01 z 统计）；OU+iso 现行口径为对照。

### 结果（探针数字，采纳时须在变体内按 Run 001 口径正式复跑）

| 方案 | eff_rank(pp) | post_std seam/far | GPR 均值 MAE | x0 >39 Hz 功率 | chol |
|---|---|---|---|---|---|
| OU+iso=1e-2（现行） | — | 0.369/≈1.0（mean 0.971） | **0.6287** | ~10% | OK |
| 平顶砖墙 sinc（j_pp=1e-4） | 75/384 | 0.022/0.990 | **2.6190** ✗ | 1.2e-3 | 需 jitter≥1e-10 |
| 谱匹配 LP40（噪声折进核=0） | 70/384 | **0.015/0.015**（塌缩） | 0.8665 | ~0 | K_post 需 +1e-8 |
| 谱匹配 BP1-40 | 70/384 | 0.014/0.014 | 3.5933 ✗ | ~0 | 同上 |
| **谱匹配 LP40 + σ_n² 条件化** | 70/384 | σ_n²=1e-2：0.074/0.074（mean 0.169） | **0.6204** ✓ | 4.5e-3（↓20×） | OK |

### 结论（结构性发现 + 推荐设计）

- **字面带限不可直接用**：带限过程在 384 点全观测过去下被解析延拓决定——条件方差塌缩至 jitter 水平（0.015–0.022 vs OU 0.97），x0 失去随机性、流匹配噪声源退化；平顶版还把 GPR 均值基线打崩（0.63→2.62）；BP1-40 更差（3.59，不建议含高通）。
- **推荐设计（探针验证可行）**：谱形匹配（训练折实测平均谱、>40 Hz 硬置零）+ **观测噪声放条件化一侧**（pp 块 +σ_n²·I、不进先验谱，σ_n² 替代现行 iso 的折叠角色）——MAE 0.6204 优于现行、x0 高频功率降 20×、数值全通过。
- **遗留标定问题**：先验散布（post_std≈0.17）显著低于真实条件不确定度（GPR 残差水平 ~0.6–0.8）——需通带白化混合（S=w·谱形+(1−w)·平顶，w 为散布-保形滑杆）或论证流可学扩张映射；采纳前须在变体内标定。实施协议（v3 薄变体、谱文件落盘、gpr_prior/config 修改面、tests、Run 001 式诊断复跑）见会话答复，未实施。

## Run 015 —— 设计可行性探针：先采样后滤波（x0 频域带限，2026-10-13 用户提问驱动；只读探针、无产物、不改生产代码）

### 目的与配置

- 用户问「直接对先验采样的数据滤波、数据进训练前就滤波可行吗」。方案 = 保留 OU+iso 条件化（满秩、不塌缩），对采样出的 x0（及推理出发点）做 reflect-pad 砖墙 40 Hz 低通（Run 013 同实现）——数学上等价于把**条件先验分布**带限化（协方差 B·K_post·Bᵀ），而非 Run 014 那样把先验**过程**带限化后再条件化。探针口径同 Run 014（测试折前 3 例 3963 窗、main-01 z 统计、K=16 逐窗子采样 330 窗）。

### 结果

| 量 | 未滤波（现行） | 滤波后 | 真实对照 |
|---|---|---|---|
| 同窗 K=16 样本散布 std | 0.925 | **0.882（保留 95%）** | — |
| >39 Hz 功率占比 | 6.9% | **7.1e-4（↓~100×）** | 1.05% |
| GPR 均值 MAE | 0.6318 | 0.6320（μ 本就平滑，滤波不动它） | — |
| post_std seam/far/mean | 0.369/1.005/0.971 | 同（条件化不变） | — |

### 结论

- **可行且为三路线最优**：①散布保留 95%（Run 014 带限核塌缩至 0.015–0.17）；②高频功率 ↓100× 且低于真实裙边；③GPR 基线数值不动（跨变体可比性保持）；④数值零风险（无秩亏求逆，纯 FFT）。Run 014 的塌缩根源在「带限过程 + 全观测条件化 ⇒ 解析延拓」；本方案条件化在带限**之前**用 OU 完成，只投影样本，绕开整个问题。
- 实施面（未实施）：`gpr_prior.py::GPRPrior.sample()` 末端 + `sample.py::EulerSampler.sample()` x0 装配末端各加同一 `bandlimit()` 助手（train.py:112 经 prior.sample 自动覆盖；posterior_mean 不滤，基线口径不动）；config 增 `prior_bandlimit_hz`（v3 薄变体，run 前缀 `v3lp-`）；tests 增滤波谱断言与训练/推理同滤一致性断言；重训 100ep + diag_hf 复跑（测网络端点重新引入的带外残余）+ 全套评估。
- 预期：生成侧 >40 Hz 随先验归零大幅下降（网络 40–100 Hz 保留主要来自先验，Run 012）；MAE 略降（流不再消耗容量搬运带外方差）；CRPS 略降（带外样本方差是纯惩罚）；指标 3 高频段 |Δlog10| 塌缩。风险：端点或有小量带外回漏（diag_hf 实测定夺，必要时叠加 Run 013 输出滤波作对照）。

## Run 016 —— v3 变体：先验采样后 40 Hz 带限（2026-10-13 用户批准「建立变体并评估」）

### 目的与配置（跑前登记；2026-10-13 用户修正：**基线 = v2（d_model=62）**）

- **单变量消融（基线 v2）**：唯一变化 = `prior_bandlimit_hz: None → 40.0`（d_model=62 与 v2 一致；相对 v1 为两处变化——带限效应归因对照 = v2）。设计依据 Run 015 探针（散布保留 95%、高频 ↓100×、GPR 基线不动、无塌缩）。
- 代码（共享库扩展 + 薄变体，v1/v2 行为与 config_hash 逐位不变——已验证 v1 默认 hash=c6ff80ac1c43 与 main-01 ckpt 一致）：
  - `gpr_prior.py`：GPRPrior 增 `bandlimit_hz/fs/bandlimit_pad`（默认关）与 `bandlimit()`（末维 reflect-pad 砖墙，Run 013 实现）；`sample()` 末端启用；`posterior_mean` 不滤
  - `train.py:213` / `sample.py::build_eval_prior`：`bandlimit_hz=getattr(cfg, "prior_bandlimit_hz", None)` 透传；`EulerSampler` x0 核心装配末端 `prior.bandlimit`（σ_min·η 不滤，与训练路径 t→0 口径一致）
  - `models/gpfm-mamba-v3/{config,train,evaluate}.py`：薄变体（config 遮蔽 + run 前缀守卫 `v3lp-`，variant_name=gpfm-mamba-v3lp，config_hash `b05544549a60`）；划分读 v1 固化 manifest（只读共享）
- 测试：新增 `tests/test_bandlimit.py` 4/4 PASS（None 逐位回归 / >40Hz 0.0679→7.42e-04 / 散布保留 95% / posterior_mean 不受开关影响）；既有 6 组单测回归全 PASS
- 划分/训练：与 v1 完全同（status_ok 46 例、fold0、seed=1234、100ep、EMA 0.999、batch 256）
- 命令：smoke `--smoke --run-id v3lp-smoke-a`；bench `--epochs 1 --bench --run-id v3lp-bench-1ep`；训练 `nohup python3 models/gpfm-mamba-v3/train.py --run-id v3lp-main-01 > /root/npi/outputs/gpfm-mamba-f0/v3lp-main-01/logs/train.log 2>&1 &`
- 评估（训练完成后）：`python3 models/gpfm-mamba-v3/evaluate.py --run-id v3lp-main-01` + `python3 models/gpfm-mamba-v1/fc_metric.py --run-id v3lp-main-01` + `python3 models/gpfm-mamba-v1/diag_hf.py --run-id v3lp-main-01`（端点带外回漏测量）——与 Run 007/009/010 完全同口径，Q1/Q2 分开陈述
