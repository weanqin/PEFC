# 实验日志 — 新预处理范式 v1/v4a fold0 训练（edra_processed_fmt，任务 10-16-gpfm-mamba-fmt-f0）

> 每次「跑批/训练/评估/统计」追加一个 Run 块：跑前先写「目的+配置」，跑后同会话补「结果+结论」，不留空块。
> 格式契约见 `.trellis/spec/research/experiment-log.md`；产物只落本任务声明的变体目录（`.trellis/spec/research/path-isolation.md`）。
> 本任务产物根：`models/gpfm-mamba-v1-fmt/`、`models/gpfm-mamba-v4a-fmt/`、`/root/npi/outputs/gpfm-mamba-fmt-f0/`（禁写旧根 `/root/npi/outputs/gpfm-mamba-f0/`）。
> 口径总登记（每 Run 随附适用项）：**GC 对比阻塞（D1，--skip-gc-only）**、**生成侧无 40 Hz 低通（D2）**、**fs=250（PSD 频轴/轨迹图标秒）**、**period_p=25（D3）**、**norm=LayerNorm（D4）**、fold0 单 seed 1234。

## Run 001 — v1-fmt 冒烟（每类 2 例 1 epoch，dataloader 契约核对）（2026-10-16）

### 目的与假设

- 验证 v1-fmt 薄变体在新数据集 `edra_processed_fmt`（250 Hz、48 例）上的训练通路端到端可跑（划分固化 → z-score → LayerNorm(d16) 模型 → 1 epoch 训练/验证 → ckpt 落盘）；
- 核对 dataloader 契约：每例 930 窗、`data (62,60001)` float32、`X_past (B,62,384)` / `X_future (B,62,128)`、z-score 输出有限（B5 review 盲区回填）。

### 配置

- 代码：`models/gpfm-mamba-v1-fmt/`（config 0dc0d5249c68 口径；epochs 覆写 1 → hash 6fd4c101f263）+ v1 共享模块 default-off 钩子（norm_type/skip_gc/fs/--outputs-root）
- 参数：`--smoke`（训练折每类 2 例 → 4 例小子集，epochs=1）；其余 = v1-fmt config 默认（p=25、无低通、LayerNorm、恒 lr 1e-3、早停 10〔1ep 不触发〕、seed 1234、batch 256）
- 环境：miniconda base Python 3.12 @ 本地 AutoDL（RTX 4080 SUPER 32GB，16 核 cgroup，BLAS 护栏 OMP=8；environment.md 契约，零依赖安装）
- 数据：`/root/autodl-tmp/edra_processed_fmt/`（status_ok 48 例 HC24/MDD24；只读）；划分 fold0 manifest 已由单测首建固化（34/4/10，见 Run 001 结果）
- 命令：`python3 models/gpfm-mamba-v1-fmt/train.py --smoke --run-id v1fmt-smoke-01`

### 结果

- 指标：train 4 例 3,720 窗（每例 930 ✓）、val 4 例 3,720 窗、1 epoch 完成（1.8 s，7.8 it/s，峰值显存 0.698 GiB）；epoch1 train total 1.3262 / val total 1.1331（新 best）；resolved config_hash 6fd4c101f263（= 单测 pin）
- 契约核对（独立脚本逐项断言，跑后同会话）：`data (62,60001)` float32 ✓；train/val/test 每例 930 窗 ✓（test 10 例 9,300 窗）；X_past (62,384) / X_future (62,128) float32、首/末窗全有限 ✓（三折均核）；run 目录 z-score 统计与重算 allclose ✓，std>0 全通道 ✓，|z|>10 占比 2.7e-5（正常幅度）
- 产物：`/root/npi/outputs/gpfm-mamba-fmt-f0/v1fmt-smoke-01/`（config.yaml/env_snapshot.txt/channel_mean.npy/channel_std.npy/logs/{train.log,metrics.jsonl,summary.json}/checkpoints/{best,last}.pt）
- 图表：无（冒烟不评估）

### 结论

- 支持：v1-fmt 变体 + 五处 default-off 钩子在新数据集上通路成立；dataloader 契约与 README 登记逐项一致（930 窗/例、形状/类型/有限性），可进 bench。划分固化补充登记：manifest 实际首建发生在本 Run 前的单测首跑（同参数决定性同内容），smoke 运行经 fixate_split 校验一致通过——整数分配 **34/4/10**（train 34 / val 4 = [25,35,37,64] 分层 2HC+2MDD / test 10 = [10,11,12,17,24,28,30,44,61,69]，与 review 实测模拟一致）。
- 下一步 / 派生决策项：Run 002 bench（全量 34 例 1 epoch，显存/时长外推 100ep）。

## Run 002 — v1-fmt bench（全量 34 例 1 epoch，显存/时长基准）（2026-10-16）

### 目的与假设

- 测定 v1-fmt（d16 + LayerNorm，15,936 参数）在全量 fold0 训练折（34 例）上的单 epoch 时长、it/s 与峰值显存，外推 100 epoch 主 Run 时长。

### 配置

- 代码：同 Run 001（v1-fmt；epochs=1 覆写 → resolved hash 6fd4c101f263）
- 参数：`--bench --epochs 1`（全量 34 例训练折 + val4；batch 256、恒 lr 1e-3、seed 1234）
- 环境：同 Run 001
- 数据：同 Run 001（训练折全量 34 例 31,620 窗）
- 命令：`python3 models/gpfm-mamba-v1-fmt/train.py --bench --epochs 1 --run-id v1fmt-bench-1ep`

### 结果

- 指标：train 34 例 31,620 窗 1 epoch = 6.94 s（17.73 it/s，4,557.6 samples/s）；数据加载 6.7 s；峰值显存 **0.698 GiB**；epoch1 train total 1.2757 / val total 1.0871；resolved config_hash 6fd4c101f263
- 100ep 外推：**0.193 h ≈ 11.6 min**（summary.json extrapolation_100_epochs_h；早停 patience=10 生效时更短）
- 产物：`/root/npi/outputs/gpfm-mamba-fmt-f0/v1fmt-bench-1ep/`（logs/summary.json 全量登记 + best/last ckpt）
- 图表：无

### 结论

- 支持：v1-fmt 资源口径远低于 32 GB 显存上限（0.7/32 GiB）；100ep 主 Run 约 12 min 量级，与早停 10 组合无资源障碍，可排主 Run（本段派发止于 bench，主 Run 不启动）。
- 下一步 / 派生决策项：v4a-fmt 冒烟（Run 003）→ bench（Run 004）。

## Run 003 — v4a-fmt 冒烟（每类 2 例 1 epoch）（2026-10-16）

### 目的与假设

- 验证 v4a-fmt 薄变体（r16 瓶颈 + trunk d62 + LayerNorm，130,366 参数）在同一新数据集上通路成立；划分与 v1-fmt 复用同一 fold0 manifest（逐字节一致已单测断言）。

### 配置

- 代码：`models/gpfm-mamba-v4a-fmt/`（config 05d9bb875a2c 口径；epochs 覆写 1）
- 参数：`--smoke`（每类 2 例，epochs=1）；`--rank` 不传（config 默认 16）
- 环境：同 Run 001
- 数据：同 Run 001；划分 = v4a-fmt splits/ 自备 manifest（与 v1-fmt 逐字节一致）
- 命令：`python3 models/gpfm-mamba-v4a-fmt/train.py --smoke --run-id v4afmt-smoke-01`

### 结果

- 指标：train 4 例 3,720 窗（每例 930 ✓）、val 3,720 窗、1 epoch（1.88 s，7.5 it/s，峰值显存 1.828 GiB）；epoch1 train total 1.2835 / val total 1.1436；模型参数 **130,366**（= v4a 130,056 + 310 LayerNorm bias，[model] 行核对）；resolved config_hash bdab1f15c58b（= 单测 epochs=1 pin）；fixate_split 校验通过（与 v1-fmt manifest 逐字节一致）
- 产物：`/root/npi/outputs/gpfm-mamba-fmt-f0/v4afmt-smoke-01/`（config.yaml/env_snapshot/channel_mean.npy/channel_std.npy/logs/checkpoints/{best,last}.pt）
- 图表：无（冒烟不评估）

### 结论

- 支持：v4a-fmt 变体通路成立（r16 瓶颈 + d62 trunk + LayerNorm 同构装载）；划分复用与 v1-fmt 完全一致的 fold0（34/4/10，test 10 例同 ID），两变体 test 指标可比性就位。
- 下一步 / 派生决策项：Run 004 bench。

## Run 004 — v4a-fmt bench（全量 34 例 1 epoch，显存/时长基准）（2026-10-16）

### 目的与假设

- 测定 v4a-fmt（d62 trunk）全量单 epoch 时长/显存，外推 100 epoch；与 v1-fmt bench 对比资源口径。

### 配置

- 代码：同 Run 003（v4a-fmt；epochs=1 覆写）
- 参数：`--bench --epochs 1`（全量 34 例 + val4；batch 256、恒 lr 1e-3、seed 1234；r=16 默认）
- 环境：同 Run 001
- 数据：同 Run 003（训练折全量 34 例 31,620 窗）
- 命令：`python3 models/gpfm-mamba-v4a-fmt/train.py --bench --epochs 1 --run-id v4afmt-r16-bench-1ep`

### 结果

- 指标：train 34 例 31,620 窗 1 epoch = 7.30 s（16.85 it/s，4,332.8 samples/s）；数据加载 6.8 s；峰值显存 **1.828 GiB**；epoch1 train total 1.1773 / val total 1.0834；resolved config_hash bdab1f15c58b
- 100ep 外推：**0.203 h ≈ 12.2 min**（summary.json extrapolation_100_epochs_h；早停 patience=10 生效时更短）
- 产物：`/root/npi/outputs/gpfm-mamba-fmt-f0/v4afmt-r16-bench-1ep/`（logs/summary.json 全量登记 + best/last ckpt）
- 图表：无

### 结论

- 支持：v4a-fmt 资源口径充裕（1.8/32 GiB，约为 v1-fmt 的 2.6×——d62 trunk 所致）；100ep 主 Run 约 12 min 量级，与早停 10 组合无资源障碍（本段派发止于 bench，主 Run 不启动）。两变体 bench 对比：epoch 时长 6.94 vs 7.30 s、峰值显存 0.698 vs 1.828 GiB。
- 下一步 / 派生决策项：主 Run（v1fmt-main-01 / v4afmt-r16-main-01）+ 评估四步（--skip-gc-only / fc_metric --outputs-root 新根 / 指标3 预登记口径补算 / 视觉图）——留待后续派发。

## Run 005 — v1-fmt 主 Run（fold0 单 seed，早停 10，≤100ep）（2026-10-16）

### 目的与假设

- 完成 v1-fmt 在新预处理范式 edra_processed_fmt 上的 fold0 前置验证主训练（B8），产出 best/last checkpoint 与 metrics 曲线供正式评估（B9）。

### 配置

- 代码：`models/gpfm-mamba-v1-fmt/`（config 0dc0d5249c68 完整口径，无 CLI 覆写）+ v1 共享模块 default-off 钩子（commit 4ba2e8e 之上工作区状态）
- 参数：epochs ≤100、**早停 patience=10**（best 判据 = val4 总 loss 固定 t 网格确定性复算/EMA）、**恒定 lr 1e-3**（无 cosine）、**单 seed 1234**（fold0 单 seed 前置验证，5-fold 须用户同意门禁）、batch 256、EMA 0.999；架构 d16 贯穿 trunk + **LayerNorm**（5 norm 位，15,936 参数）；**period_p=25**（fs 250 保物理时长）；**生成侧无低通**（gen_lowpass_hz=None）；划分 34/4/10（fold0 manifest 固化，与 v4a-fmt 逐字节一致）
- 环境：miniconda base Python 3.12 @ 本地 AutoDL（RTX 4080 SUPER 32GB，16 核 cgroup，BLAS 护栏 OMP=8；environment.md 契约，零依赖安装）
- 数据：`/root/autodl-tmp/edra_processed_fmt/`（status_ok 48 例 HC24/MDD24，只读）；train 34 例 31,620 窗 / val 4 例 3,720 窗
- 命令：`nohup python3 models/gpfm-mamba-v1-fmt/train.py --run-id v1fmt-main-01 > /root/npi/outputs/gpfm-mamba-fmt-f0/v1fmt-main-01/logs/nohup.out 2>&1 &`

### 结果

- 指标：**92 epochs（早停触发于 ep92，patience 10）**；best epoch **82**，best val total **0.69424**；末期（last-5）train mae 均值 0.50176；mean epoch 5.79 s、峰值显存 0.698 GiB；resolved config_hash 0dc0d5249c68（= 单测 pin）；best/last ckpt + metrics.jsonl + summary.json 落盘
- 产物：`/root/npi/outputs/gpfm-mamba-fmt-f0/v1fmt-main-01/`（checkpoints/gpfm-mamba-v1-fmt-{best,last}.pt、logs/、config.yaml、env_snapshot.txt、channel_{mean,std}.npy）
- 图表：无（训练期不产图；评估见 Run 007）

### 结论

- 支持：v1-fmt 主训练收敛正常（val total 1.087@ep1 → 0.694@ep82 后平台 10ep 早停——早停口径按设计生效，未跑满 100ep）；资源口径与 bench 预测一致。R1 监控留待评估侧（evaluate.py r1_monitor 自动核对）。
- 下一步 / 派生决策项：Run 007 正式评估（--skip-gc-only + fc_metric + 指标3 补算）。

## Run 006 — v4a-fmt 主 Run（fold0 单 seed，早停 10，≤100ep）（2026-10-16）

### 目的与假设

- 完成 v4a-fmt（r16 瓶颈 + trunk d62 + LayerNorm）同款主训练（C7）；与 v1-fmt 同划分/同纪律 → 两变体 test 折可比；对照纯净度注记（vs 旧 v4a 差异仅 数据+p+低通+norm，无训练纪律变量叠加）。

### 配置

- 代码：`models/gpfm-mamba-v4a-fmt/`（config 05d9bb875a2c 完整口径，无 CLI 覆写；r=16 = config 默认，不传 --rank）
- 参数：同 Run 005（epochs ≤100、早停 10、恒 lr 1e-3、单 seed 1234、batch 256、EMA 0.999、p=25、无低通、fs=250）；架构 r16 低秩瓶颈 + trunk d_model=62 + **LayerNorm**（130,366 参数 = v4a + 310 bias）
- 环境：同 Run 005
- 数据：同 Run 005（划分同 manifest 逐字节一致）
- 命令：`nohup python3 models/gpfm-mamba-v4a-fmt/train.py --run-id v4afmt-main-01 > /root/npi/outputs/gpfm-mamba-fmt-f0/v4afmt-main-01/logs/nohup.out 2>&1 &`

### 结果

- 指标：**55 epochs（早停触发于 ep55，patience 10）**；best epoch **45**，best val total **0.6627**；末期（last-5）train mae 均值 0.44490；mean epoch 6.27 s、峰值显存 1.828 GiB；resolved config_hash 05d9bb875a2c（= 单测 pin）；best/last ckpt + metrics.jsonl + summary.json 落盘
- 产物：`/root/npi/outputs/gpfm-mamba-fmt-f0/v4afmt-main-01/`（checkpoints/gpfm-mamba-v4a-fmt-{best,last}.pt、logs/、config.yaml、env_snapshot.txt、channel_{mean,std}.npy）
- 图表：无（评估见 Run 008）

### 结论

- 支持：v4a-fmt 收敛快于 v1-fmt（best ep45 vs 82；best val 0.663 vs 0.694——内部验证口径，4 例小样本仅方向性参考）；早停口径两 Run 均按设计生效。R1 监控留待评估侧。
- 下一步 / 派生决策项：Run 007/008 正式评估（--skip-gc-only + fc_metric + 指标3 补算 + GC 阻塞验收）。

## Run 007 — v1-fmt 正式评估（B9：Q1 + 指标1 FC + 指标3 PSD/空间，GC 阻塞）（2026-10-16）

### 目的与假设

- 对 v1fmt-main-01 best ckpt（ep82）完成前置验证四步评估：Q1 质量（MAE/CRPS/逐通道 r）、指标1 FC 一致性、指标3 PSD（fs=250 预登记口径补算）+ 空间结构归因；**GC 对比阻塞（D1）**——产物不得出现任何 GC 对比文件。

### 配置

- 代码：`models/gpfm-mamba-v1-fmt/evaluate.py`（单遮蔽 + v1 evaluate.py 钩子）+ `models/gpfm-mamba-v1-fmt/fc_metric.py`（薄包装注入 --outputs-root 新根）
- 参数：`--run-id v1fmt-main-01 --skip-gc-only`（GC 阻塞；PSD fs=250 nperseg 512 → 0.488 Hz 分辨率；空间归因照跑）；K=cfg.n_samples=16、euler 16、sample_seed=1234（GC 单样本 seed 1235）；**生成侧无低通**（gen_lowpass_hz=None，D2）；fc_metric `--run-id v1fmt-main-01`（口径 v2.1：点估计=组平均矩阵，显著性=逐受试者单样本 t 检验）
- 环境：同 Run 005
- 数据：test 折 10 例 9,300 窗（fold0 manifest test IDs 同 v4a-fmt）；z-score = run 目录训练折统计（重算 allclose 校验）
- 命令：`python3 models/gpfm-mamba-v1-fmt/evaluate.py --run-id v1fmt-main-01 --skip-gc-only`；`python3 models/gpfm-mamba-v1-fmt/fc_metric.py --run-id v1fmt-main-01`

### 结果

- Q1（best ckpt ep82/0-based 81，test 10 例 9,300 窗）：MAE model **0.71266** / 持久性 0.95278 / GPR 均值 0.66349；CRPS **0.54329**；r(model) **0.08929**（逐通道 62/62 为正，0.061–0.112；r(gpr) 0.11604 对照）；幅度 gen_std_t 1.15817 vs x1_std_t 0.81738（**+41.7%** 过散）；R1 未触发（高/中 t 桶比 0.4282、终点比 1.4203 均 <1.5）
- 指标1 FC（v2.1）：Pearson r **0.96982** / Spearman ρ 0.97189；逐受试者 r 均值 0.89578（t=99.19，df=9，p=5.5e-15）、ρ 均值 0.89829
- 指标3 PSD（fs=250，nperseg 512 → 0.488 Hz，1,160 段）：**1–30 Hz mean |Δlog10| = 0.31734（≤0.5 通过，预登记口径）**；40–125 Hz = 7.84747（观察段照登）；>40 Hz 功率占比 real 1.6246% vs **gen 4.8207%**；峰频率 real 1.46 Hz vs gen 1.95 Hz；全频段 mean_abs_logdiff(ln) 12.5456（口径注记：不作判读依据——无低通+带限地板主导）
- 空间归因：R̂~R 上三角 r **0.96611**（R̂~K_s 0.82114、K_s~R 0.883）
- GC 阻塞验收：eval/ 与 figs/ 逐文件核验**无任何 GC 对比文件**（gc_real/gc_gen/gc_F_p_*/gc 矩阵图全无）；summary/gc_psd_results 均记 `"gc": "skipped(blocked)"`、arrays 仅 psd/spatial；eval_metrics.jsonl stage=gc 记 skipped
- 产物：`/root/npi/outputs/gpfm-mamba-fmt-f0/v1fmt-main-01/eval/{summary.json,gc_psd_results.json,psd.npz,spatial_corr.npz,gc_input.npz,fc_metric.json,fc_matrices.npz}` + `figs/{eval_trajectories,eval_training_curves,eval_psd,eval_spatial_corr}_v1fmt-main-01.png`
- 图表：轨迹图 4 幅（时间轴 fs=250 标秒——指标2 人工判材料，待用户）；PSD/空间图各 1

### 结论

- 支持：v1-fmt 前置验证四步全通——Q1 三基线排序正常（model < persistence，略劣于 GPR 均值，与旧范式 v1 同型）；FC 一致性强（r 0.970）；指标3 预登记口径通过（0.317 ≤ 0.5）；空间结构 R̂~R 0.966；R1 未触发。单次运行（fold0 单 seed）。
- 观察：无低通口径下生成侧 >40 Hz 功率占比 4.82% ≈ 真实侧（1.62%）的 3.0×——模型大体待在带内（95% 功率 <40 Hz）但高频尾偏重（D3 观察项，与过散 +41.7% 同向）；判读见 D 节注记。
- 下一步 / 派生决策项：与 v4afmt（Run 008）对比 + D 节跨范式判读；5-fold 启动留待用户拍板（门禁）。

## Run 008 — v4a-fmt 正式评估（C8：同 Run 007 四步）（2026-10-16）

### 目的与假设

- 对 v4afmt-main-01 best ckpt（ep45）完成同款四步评估；与 v1-fmt 同 test 折逐 ID 一致 → 两变体指标直接可比。

### 配置

- 代码：`models/gpfm-mamba-v4a-fmt/evaluate.py` + `models/gpfm-mamba-v4a-fmt/fc_metric.py`
- 参数：`--run-id v4afmt-main-01 --skip-gc-only`；其余同 Run 007（K=16/euler 16/seed 1234/无低通/fs=250）
- 环境：同 Run 005
- 数据：同 Run 007（test 10 例 9,300 窗，同 manifest）
- 命令：`python3 models/gpfm-mamba-v4a-fmt/evaluate.py --run-id v4afmt-main-01 --skip-gc-only`；`python3 models/gpfm-mamba-v4a-fmt/fc_metric.py --run-id v4afmt-main-01`

### 结果

- Q1（best ckpt ep45/0-based 44，test 同 v1-fmt 逐 ID 一致 10 例 9,300 窗）：MAE model **0.70565** / 持久性 0.95278 / GPR 均值 0.66349；CRPS **0.53912**；r(model) **0.12335**（逐通道 62/62 为正，0.099–0.139；r(gpr) 0.11604 对照）；幅度 gen_std_t 1.18074 vs x1_std_t 0.81738（**+44.4%** 过散）
- **R1 触发（终点分支）**：endpoint_vs_train_late = 1.5861 ≥ 1.5（高/中 t 桶比 0.4418 未触发）。按 evaluation-calibers R1 解读规则做定性流程：分子 test 终点 MAE **0.70565**（与同折对照 v1fmt 0.71266 相比**略优**）；分母 train 末期 x̂1-MAE **0.44490**（显著低于 v1fmt 同折 0.50176，−11.3%——v4a-fmt 训练折拟合更深）⇒ 触发机械来自**分母收缩（口径效应）**而非终点劣化，与登记先例（v2=1.532/v3=1.555）同型；「高 t 欠约束风险」不成立（本 Run 口径）
- 指标1 FC（v2.1）：Pearson r **0.98147** / Spearman ρ 0.98320；逐受试者 r 均值 0.91309（t=82.73，df=9，p=2.8e-14）、ρ 均值 0.91484
- 指标3 PSD（fs=250，1,160 段）：**1–30 Hz mean |Δlog10| = 0.30134（≤0.5 通过）**；40–125 Hz = 7.81150（照登）；>40 Hz 功率占比 real 1.6246% vs **gen 4.9210%**；峰频率 real 1.46 Hz = gen 1.46 Hz；全频段 mean_abs_logdiff(ln) 12.4834（不作判读依据）
- 空间归因：R̂~R 上三角 r **0.98072**（R̂~K_s 0.83963）
- GC 阻塞验收：同 Run 007 逐文件核验**无任何 GC 对比文件**；json 全记 `"gc": "skipped(blocked)"`
- 产物：`/root/npi/outputs/gpfm-mamba-fmt-f0/v4afmt-main-01/eval/{summary.json,gc_psd_results.json,psd.npz,spatial_corr.npz,gc_input.npz,fc_metric.json,fc_matrices.npz}` + `figs/` 四图（同 Run 007 清单）
- 图表：轨迹图（fs=250 标秒，指标2 人工判材料，待用户）；PSD/空间图各 1

### 结论

- 支持：v4a-fmt 前置验证四步全通，且在全部可判维度优于 v1-fmt（MAE 0.7057 vs 0.7127、CRPS 0.5391 vs 0.5433、r 0.1234 vs 0.0893、FC r 0.9815 vs 0.9698、指标3 0.3013 vs 0.3173、空间 0.981 vs 0.966）——候选排序与旧范式一致（v4a 侧最优）；R1 触发经定性流程判为口径效应（分母收缩），登记分子/分母/对照三值。单次运行（fold0 单 seed）。
- 观察：与 v1-fmt 同向——>40 Hz 占比 gen 4.92% ≈ real 1.62% 的 3.0×（大体在带内、高频尾偏重）；过散 +44.4%。
- 下一步 / 派生决策项：D 节跨范式判读；5-fold 启动留待用户拍板（门禁）。

## 判读登记（D1–D3，2026-10-16；全部数值来源：本文件 Run 005–008 + 旧范式 outputs/gpfm-mamba-f0/{main-01,v4a-r16-main-01}（只读）；补算脚本 research/psd_band_metric3.py）

### D2 跨范式对比表（**单次运行**，fold0 单 seed 1234；方向性观察，不做归因强主张）

| 指标 | v1 旧范式(main-01) | **v1-fmt(Run 005/007)** | v4a 旧范式(r16-main-01) | **v4a-fmt(Run 006/008)** |
|---|---|---|---|---|
| 训练 | 100ep 满（无早停） | 92ep 早停，best ep82，best val 0.69424 | 73ep 早停，best ep63，best val 0.77901 | 55ep 早停，best ep45，best val 0.66270 |
| Q1 MAE(model) | 0.70834 | 0.71266 | 0.69540 | **0.70565** |
| Q1 MAE(persistence / GPR) | 0.95271 / 0.68729 | 0.95278 / 0.66349 | 0.96063 / 0.69297 | 0.95278 / 0.66349 |
| CRPS | 0.52629 | 0.54329 | 0.52392 | 0.53912 |
| r(model) 逐通道均值 | 0.14371 | 0.08929 | 0.18317 | 0.12335 |
| r(gpr) 对照 | 0.16782 | 0.11604 | 0.16782 | 0.11604 |
| gen_std_t / x1_std_t | 0.97183 / 0.78808（+23%） | 1.15817 / 0.81738（+42%） | 1.09690 / 0.79459（+38%） | 1.18074 / 0.81738（+44%） |
| 指标1 FC r（ρ） | 0.95856（0.97126） | 0.96982（0.97189） | 0.97561（0.98203） | **0.98147（0.98320）** |
| FC 逐受试者 r 均值 | 0.91707（p=7.6e-15） | 0.89578（p=5.5e-15） | 0.93571（p=6.0e-16） | 0.91309（p=2.8e-14） |
| 指标3 1–30 Hz mean\|Δlog10\| | 0.27714 ✓ | 0.31734 ✓ | 0.34248 ✓ | **0.30134** ✓ |
| 指标3 40–125 Hz 照登（四列同段可比） | 7.2148 | 7.8475 | 7.2633 | 7.8115 |
| >40 Hz 功率占比 real→gen | 0.98% → 1.59%（有 40 Hz 低通） | 1.62% → 4.82%（**无低通**） | 0.97% → 1.65%（有低通） | 1.62% → 4.92%（**无低通**） |
| 空间 R̂~R 上三角 r | 0.95322（旧 run gc_psd_results.json 照登） | 0.96611 | 0.96944（同左） | 0.98072 |
| R1 | 未触发（1.2432） | 未触发（1.4203） | 未触发（1.3372） | **触发**（1.5861；定性=口径效应，分母收缩 0.4449 vs 对照 0.50176，分子 0.70565 反略优） |

表注（check 复核回填）：①指标3 照登行四列均为 **40–125 Hz 同段**补算值（psd_band_metric3.py 统一频段，跨范式可比；旧范式自身「40 Hz–Nyquist(250)」口径照登 = 9.1199（v1 旧）/ 9.1665（v4a 旧），与本表不同段不可直比）；②旧范式空间 R̂~R 系 40 Hz 低通口径（生成侧滤波后相关矩阵），fmt 侧无低通——方向性参考。

方向性观察（均**单次运行**，多变量设计不做单变量归因）：

1. Q1 量级跨范式大体持平（MAE 0.70–0.71、CRPS 0.52–0.54）；r(model) 与 r(gpr) 同步下移（0.144→0.089 / 0.183→0.123；GPR 0.168→0.116）——两基线同降说明 **r 下移主要来自数据侧（250 Hz、窗口物理时长 ×2、不同 test 折）而非模型退化**；model-vs-GPR 的相对差在 fmt 侧收窄（v1 侧反超 GPR：0.089 vs 0.116；v4a 侧 0.123 vs 0.116 持平反超边缘）。
2. FC 一致性跨范式不降反微升（v1 0.959→0.970；v4a 0.976→0.981）——慢波动 FC 结构在新范式下保持强复现。
3. 指标3 预登记口径（1–30 Hz ≤0.5）四 Run 全过；fmt 侧 0.30–0.32 与旧范式 0.28–0.34 同水平。
4. 候选排序不变：**v4a(-fmt) 在全部维度优于 v1(-fmt)**（跨范式两侧一致）。
5. 过散与 >40 Hz 尾（见 D3）同向加重：无低通 + 带限数据下 gen_std/x1_std 比从 +23%/+38% 升至 +42%/+44%。

### D2 差异清单（跨范式对比必随附；多变量设计）

fs 500→250、预处理（release 1–40 Hz 带通/无 ICA·QC vs bp1-40+notch+ICA+QC）、例数 46→48（fold0 整数分配：v1 侧 34/2/10→34/4/10、v4a 侧 32/4/10→34/4/10，test 折不同受试者）、period_p 50→25（先验与数据同时变）、生成侧 40 Hz 低通→无（D2）、norm RMSNorm→LayerNorm（D4）、**v1-fmt 侧额外叠加训练纪律变量（早停 10/恒定 lr/val4——旧 v1 为满 100ep+cosine+val2）**、GC 对比阻塞（D1，fmt 无 GC 指标）。

### D2 三条判读注记（review 预登记）

1. **同 epochs ≠ 同步数**：fmt 每例 930 窗 vs 旧 1321 窗（fmt 31.6k 窗/epoch vs 旧 v4a 42.3k / 旧 v1 44.9k）；EMA τ=1000 步 ≈ 8.1 epoch（fmt）vs 6.1（旧 v4a）/ 5.7（旧 v1）epoch——EMA 权重更新节奏不同，早停 epoch 数不可直接跨范式比较。
2. **窗口物理时长 ×2**（512 样点 = 2.048 s @250 Hz vs 1.024 s @500 Hz）：Euler/TV 平滑的物理尺度与逐窗自相关结构同步变化，r 类指标口径不同。
3. **对照纯净度不对称**：v4a-fmt vs 旧 v4a 较干净（差异 = 数据 + p + 低通 + norm，训练纪律同款）；v1-fmt vs 旧 v1 额外叠加纪律变量——**跨范式判读权重向 v4a 侧倾斜**（review Mandate A）。

### D3 生成侧 >40 Hz 功率占比观察（D2 拍板观察项；数据源 eval/psd.npz，脚本补算）

- 真实侧（release 1–40 Hz 带通 skirt）：**1.62%**（两 fmt Run 同 test 折同值）；
- 生成侧（**无 40 Hz 低通**，D2 口径）：v1-fmt **4.82%**、v4a-fmt **4.92%**——≈ 真实侧 3.0×；
- 结论：**模型大体自行待在带内**（≈95% 功率 <40 Hz；峰频率与真实侧对齐：v4a-fmt 1.46=1.46 Hz，v1-fmt 1.95 vs 1.46 Hz），但高频尾一致性偏重（v1-fmt +3.2pp / v4a-fmt +3.3pp），与过散 +42%/+44% 同向——无低通口径的已知代价，方向性可接受；若 5-fold 主实验要求更干净的带内谱，可议重开低通（**250 Hz 下必须传 fs=250**，config 已登记该坑）。旧范式对照（有低通）：gen 1.59%/1.65% vs real 0.98%/0.97%（比值 1.6×，低通压制后残余为滤波裙边）。

### 指标2 视觉材料（D4 预备，待用户人工判）

- `figs/eval_trajectories_{v1fmt,v4afmt}-main-01.png`（medoid + 90% PI，时间轴 fs=250 正确标秒——A2③ 钩子生效）；PSD/空间/训练曲线图同目录。
