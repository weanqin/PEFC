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
