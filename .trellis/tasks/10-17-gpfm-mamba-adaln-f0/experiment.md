# 实验日志 — 10-17-gpfm-mamba-adaln-f0（v4a-fmt adaLN 调制归一化变体 fold0 前置验证）

> 每次「跑批/训练/评估/统计」追加一个 Run 块：跑前先写「目的+配置」，跑后同会话补「结果+结论」，不留空块。
> 格式契约见 `.trellis/spec/research/experiment-log.md`；产物只落本任务声明的变体目录（`.trellis/spec/research/path-isolation.md`）。
> 全部 Run 标签：**前置验证 · fold0 · 单 seed · 单次运行**（不越 5-fold 门禁）。

## 单变量差异声明（全部 Run 随附）

- 基线 = **v4afmt-main-01**（MAE 0.70565 / CRPS 0.53912 / FC r 0.98147 / 指标3 1–30Hz 0.30134 / 逐通道 r 0.12335）。
- 唯一实验变量：norm LayerNorm→**adaLN(t)**（A 案 scale/shift，5 个 norm 位 = 4×MambaBlock pre-norm + head norm
  全部换 `modulate(LN(x), s(t), b(t)) = (1+s)⊙LN(x)+b`；(s,b) 由 time_embedding(t,16) 经逐 site
  Sequential(SiLU, Linear(16,124)) 零初始化生成，fork_rng 内构建——同 seed 共享权重与基线逐位相同，
  epoch-0 前向与基线逐位一致）。
- 参数量登记：130,366 → **140,906（+10,540 / +8.08%）**。
- 训练纪律逐字段一致：fold0 manifest 逐字节同（34/4/10）、seed 1234、batch 256、恒 lr 1e-3、
  早停 10、val4、EMA 0.999、≤100ep。
- 机制登记（design §2 预登记）：①t 双注入不对称（输入侧仅 future 位、调制侧全 512 位——DiT 同款）；
  ②EMA 影子滞后：31,620 窗 / batch 256 drop_last = **123 step/epoch**，EMA τ≈1000 step ≈ **8.1 epoch**
  ——阴性预案：若结论「无改善」且 best epoch ≲16，追加非 EMA last.pt 补充评估（补充 Run，不替换正式口径）；
  ③t_loc=−0.8 偏态下 (s,b)(t) 在稀有 t>0.7 区为有界外推（判读留意 Euler 末端步）；④weight_decay
  对调制 W 同等对待（向恒等的轻微先验）。
- 判读规则（预登记，design §3）：单 seed 不设数值阈值，差异量级与 fmt-f0 跨变体单 seed 差异同框做方向性判读。

## Run 001 — adaln 变体冒烟（dataloader 契约核对 + manifest 固化）（2026-10-17）

### 目的与假设

- 验证 cond_adaln="scale_shift" 钩子在真实训练链路（数据→模型→损失→ckpt）机械正确：每类 2 例 1 epoch
  跑通且有限；核对 dataloader 契约与 v4a-fmt 完全一致（930 窗/例、data (62,60001) float32、
  X_past (B,62,384)/X_future (B,62,128)、z-score 有限）；首跑固化本目录 manifest 并与 v4a-fmt 逐字节一致。

### 配置

- 代码：`models/gpfm-mamba-v4a-fmt-adaln/`（config 唯 4 差异字段 + 三 wrapper）；共享钩子
  `models/gpfm-mamba-v1/{mamba,model,train,sample}.py`（git cd1ea4b + 工作树 cond_adaln 钩子，commit ① 待提交）
- 参数：`--smoke`（每类 2 例、epochs 1、run_id v4afmt-adaln-smoke-01）；其余 = config 默认
  （cond_adaln="scale_shift"、r=16、d_model=62、LayerNorm、seed 1234、batch 256、恒 lr 1e-3）
- 环境：base conda python3 3.12.3 @ 本机 RTX 4080S 32GB（.trellis/research/environment.md）
- 数据：`/root/autodl-tmp/edra_processed_fmt/`（48 例 status_ok；smoke 子集 = 训练折每类前 2 例）
- 命令：`python3 models/gpfm-mamba-v4a-fmt-adaln/train.py --smoke --run-id v4afmt-adaln-smoke-01`

### 结果

- 冒烟跑通：train 4 例（每类 2）×930 = 3720 窗 / 14 step（batch 256 drop_last），1ep 2.25s，
  全程损失有限（total 1.4479→1.2866，末步 loss_trend），内部验证 total 1.1435（新 best）；peak 显存 1.952 GiB。
- **dataloader 契约核对（与 v4a-fmt 口径逐项一致）**：npz `data (62,60001) float32` 有限、fs=250、
  pipeline=EDRA_release_1-40_250Hz；**930 窗/例**（(60001−512)//64+1）；dataset 输出
  `X_past (62,384)/X_future (62,128) float32` 有限；全折 train 31,620 / val 3,720 窗
  （31,620/256 drop_last = **123 step/epoch**，与 design §2.3 EMA 锚点一致）；run 目录
  channel_mean/std (62,) 有限。
- **manifest 固化**：本目录 splits/fold0_status_ok_seed42_val4.json 首跑写入（1,334 字节），与
  v4a-fmt manifest **逐字节一致**（独立核验 + 单测双重断言）；test 10 例 [10,11,12,17,24,28,30,44,61,69]。
- resolved config.yaml 落盘（cond_adaln: "scale_shift"、outputs_root=本任务新根、epochs=1 覆写 →
  hash 8bd4ab0aae50）；ckpt `gpfm-mamba-v4a-fmt-adaln-{best,last}.pt` 含全部 10 调制键 +
  EMA shadow 同 10 键（泛型覆盖成立）。
- 产物：`/root/npi/outputs/gpfm-mamba-adaln-f0/v4afmt-adaln-smoke-01/`（config.yaml/env_snapshot/
  channel_{mean,std}.npy/logs/train.log+metrics.jsonl+summary.json/checkpoints/）

### 结论

- 支持：adaLN 钩子在真实训练链路机械正确（有限损失、ckpt/EMA 键齐、dataloader 契约与基线一致），
  可进 bench。派生决策：无（无契约偏差须修）。

## Run 002 — adaln 变体 1-epoch 基准（显存/时长外推）（2026-10-17）

### 目的与假设

- 全量训练折 1 epoch 基准：登记 adaLN 分支真实显存峰值与 epoch 时长，外推 100ep（早停 10）墙钟可行性。

### 配置

- 代码：同 Run 001
- 参数：`--bench --epochs 1 --run-id v4afmt-adaln-bench-1ep`；其余 = config 默认（epochs 覆写 1 →
  config_hash 8bd4ab0aae50，smoke/bench 覆写口径 pin）
- 环境：同 Run 001
- 数据：同 Run 001（全量训练折 34 例）
- 命令：`python3 models/gpfm-mamba-v4a-fmt-adaln/train.py --bench --epochs 1 --run-id v4afmt-adaln-bench-1ep`

### 结果

- 全量训练折 34 例 31,620 窗 / **123 step**（batch 256 drop_last，与 design §2.3 EMA 锚点一致）；
  train total 1.1721，val total 1.0827（新 best），全程有限无 NaN guard 触发。
- **显存/时长**：peak 1.952 GiB（RTX 4080S 32GB，余量充裕）；epoch 训练段 8.95 s（13.74 it/s，
  3,532.8 samples/s）+ 内部验证段（3720 窗 × 5 t 网格，未计入 epoch_time）；
  **100ep 外推 ≈ 0.249 h 训练段（~15 min）**，加验证段按 ~2–3× 估 ≤0.8 h——早停 10 下墙钟无压力。
- 产物：`/root/npi/outputs/gpfm-mamba-adaln-f0/v4afmt-adaln-bench-1ep/`（summary.json 记
  extrapolation_100_epochs_h=0.249；config_hash 8bd4ab0aae50）。

### 结论

- 支持：adaLN 分支训练吞吐/显存与基线同量级（调制仅为 (B,16)→(B,124) 小 MLP + 逐元素广播，
  开销可忽略），主 Run 全程 ≤1 h 量级，可启动。派生决策：主 Run nohup 后台即可，无须特殊资源安排。

## Run 003 — adaln 主 Run（fold0 单 seed 正式训练，≤100ep 早停 10）（2026-10-17）

### 目的与假设

- 正式单变量对比 Run：adaLN(t) 调制归一化 vs v4afmt-main-01 纯 LayerNorm（同 manifest/seed/纪律）。
  假设：t 条件调制至少不损害（恒等初始化 ⇒ epoch-0 与基线逐位一致，其后分叉恰为处理效应）。

### 配置

- 代码：同 Run 001（commit 以 ckpt 内 code_commit 登记为准）
- 参数：config 默认全量（epochs 100 上限、早停 patience 10、seed 1234、batch 256、恒 lr 1e-3、
  EMA 0.999、val4；cond_adaln="scale_shift"）——无任何 CLI 覆写（config_hash 9f0af705696c）
- 环境：同 Run 001（长跑 nohup 后台）
- 数据：`/root/autodl-tmp/edra_processed_fmt/` 全量（train 34 例 / val 4 例，manifest 逐字节同 v4a-fmt）
- 命令：`nohup python3 models/gpfm-mamba-v4a-fmt-adaln/train.py --run-id v4afmt-adaln-main-01 > /root/npi/outputs/gpfm-mamba-adaln-f0/v4afmt-adaln-main-01/nohup.log 2>&1 &`
  （run 目录 = `<outputs_root>/v4afmt-adaln-main-01`，train.py 实际 outputs 结构，无 run_ 前缀）

### 结果

- **训练完成（2026-10-17，早停）**：epoch 54/100 触发早停（val 总 loss 连续 10ep 无新 best，判据 = 固定 t 网格确定性复算/EMA）；**best epoch 44，best val total 0.66262**；config_hash 9f0af705696c，code_commit cd1ea4b；54 ep × mean 7.97 s/ep（15.43 it/s，peak 1.952 GiB）；best/last ckpt + metrics.jsonl 落盘完整（末行附 `"early_stop": true`）。
- 训练节奏与基线 v4afmt-main-01 几乎同拍（基线 55ep 早停 vs 本 Run 54ep）——总步数量级一致，EMA 影子滞后锚点（8.1 ep）对两 Run 同等适用；best epoch 44 > 2×8.1，阴性预案触发条件（best ≲16）不满足。
-（正式评估见 Run 004）

### 结论

- 训练侧：adaLN(t) 变体收敛正常（无 NaN/尖峰），早停节奏与基线同量级——A 案在现有纪律下训练稳定性成立（design §4「失稳」风险未触发）。质量结论待 Run 004 同框判读。

## Run 004 — adaln 主 Run 正式评估（Q1 + 指标1/2/3，GC 阻塞口径）（2026-10-17）

### 目的与假设

- 对 Run 003 best ckpt（EMA）做与 v4afmt-main-01 完全同口径正式评估：Q1 质量 + 指标1 FC + 指标2 视觉 + 指标3 PSD（1–30 Hz 判读 / 40–125 Hz 照登）+ >40 Hz 功率占比观察；genGC vs real GC 阻塞延续（`--skip-gc-only`，产物零 GC 对比文件）。
- 假设：若 t 条件调制有效，Q1 相对基线不劣化且至少一项改善；方向性判读（单 seed 不设阈值，与 fmt-f0 跨变体差异同框）。

### 配置

- 对象：`/root/npi/outputs/gpfm-mamba-adaln-f0/v4afmt-adaln-main-01/`（best ckpt，EMA；config_hash 严格校验）
- 命令①：`python3 models/gpfm-mamba-v4a-fmt-adaln/evaluate.py --run-id v4afmt-adaln-main-01 --skip-gc-only`（Q1 + PSD fs=250 + 空间归因，无 GC 产物）
- 命令②：`python3 models/gpfm-mamba-v4a-fmt-adaln/fc_metric.py --run-id v4afmt-adaln-main-01`（wrapper 注入新 outputs 根；指标1）
- 命令③：指标3 频段补算（1–30 Hz mean |Δlog10| + 40–125 Hz 照登 + >40 Hz 占比），数据源 eval/psd.npz，脚本 = 本任务 `research/psd_band_metric3.py`（自归档任务复制改 run 路径）
- 环境：同 Run 001；确定性口径同基线

### 结果

- **Q1 质量**：test MAE（ODE 终点样本均值口径）**0.70571** | CRPS **0.54033** | r(model) 逐通道 **0.12148** | r(gpr) 同折对照 0.11604 | 持久性 MAE 0.95278（r=0）。幅度：gen std_t 1.2019 vs 真实 0.8174（**+47.0%**，样本间展宽 1.1560）。
- **指标1 FC**：Pearson r **0.9797** / Spearman ρ **0.98146**；逐受试者 r 均值 0.90939（sd 0.02597，p=2.0e-15；df=9）。
- **指标3 PSD（fs=250）**：判读窗 **1–30 Hz mean |Δlog10| = 0.30605 ✓（≤0.5 过线）**；40–125 Hz 照登 7.84633；0–1 Hz 地板段 0.39587 照登；全频段 mean_abs_logdiff(ln) 12.54597（不作判读依据）；峰频率 real 1.46 Hz = gen 1.46 Hz 对齐。
- **>40 Hz 功率占比**：real 1.6246% → gen **5.1875%**（≈3.19×，模型大体自行待在带内）。
- **空间结构**：R̂~R 上三角 r **0.978**（R̂~K_s 0.831，K_s~R 0.883）。
- **R1 告警触发**：终点 MAE 0.7057 ≥ 训练末期 x̂1-MAE 0.4390 × 1.5——与基线 v4afmt-main-01 同款**口径效应**（分母 train 折一步 0.4390 比基线 0.4449 更小，分子 0.70571 ≈ 基线 0.70565 持平；按 evaluation-calibers R1 登记规则附分子分母两值定性）。
- **GC 阻塞核验（逐文件）**：eval/ 仅 {fc_matrices, fc_metric, gc_input, gc_psd_results, psd, spatial_corr, summary}——无 gc_real/gc_gen/gc_F_p_*；gc_psd_results.json 记 `"gc": "skipped(blocked)"` 且 arrays/figures 无 gc 引用。✓
- **指标2 视觉材料**：figs/ 四图（轨迹 medoid+90% PI 250 Hz 标秒 / 训练曲线 / PSD / 空间）——待用户人工判。

### 结论（前置验证 · fold0 · 单 seed · 单次运行；方向性判读）

**adaLN(t)（A 案 scale/shift）与纯 LayerNorm 基线全面持平——中性结果**：

- 全部维度差异在 ±0.005 量级（见下对比表），处于单 seed 噪声框内，无任何维度出现方向性改善或劣化；Q1 MAE 差 0.00006（四位数内相同）。
- 过散（+44.5%→+47.0%）与 >40 Hz 尾（gen 4.92%→5.19%）微加重——与调制自由度增加同向，但幅度小。
- EMA 影子滞后不构成混杂解释：best epoch 44 ≫ 8.1 ep（EMA 时间常数），调制在 best 判据时点已充分进入 EMA 权重；阴性预案（best ≲16）未触发。
- 机制判读：恒等初始化起点下两 Run 从**逐位相同的初始权重**出发（fork_rng 契约 + config_hash 分别锁定），训练后全维度收敛到同一水平——**在该规模（140k）与纪律下，速度网络未从「norm 仿射 t 条件化」中获得超出输入侧 temb 注入的可用信息**；调制分支亦未损害质量（无失稳、R1 同款口径效应）。
- 对候选集的含义（登记，不改判）：adaLN 不改变 {v1, v4a} 排序判断，v4a(-fmt) 基线口径维持；若后续 5-fold 需要细分「t 条件化 vs 容量」归因，按 review Mandate A 预案另议加臂。

## 对比表 — v4afmt-adaln-main-01 vs v4afmt-main-01（C2；单变量：norm LayerNorm→adaLN(t)，参数 130,366→140,906 = +10,540/+8.08%）

| 指标 | 基线 v4afmt-main-01 | 本 Run v4afmt-adaln-main-01 | Δ（本−基线） |
|---|---|---|---|
| 早停（best epoch） | 55 ep（best 45） | 54 ep（best 44） | 同拍 |
| test MAE（ODE 终点） | 0.70565 | 0.70571 | +0.00006 |
| CRPS | 0.53912 | 0.54033 | +0.00121 |
| r(model) 逐通道均值 | 0.12335 | 0.12148 | −0.00187 |
| r(gpr) 同折对照 | 0.11604 | 0.11604 | 0（确定性同折） |
| model−GPR 相对差 | +0.00731 | +0.00544 | 反超保持、微收窄 |
| gen std_t / x1_std_t | 1.18074/0.81738（+44.5%） | 1.2019/0.8174（+47.0%） | 过散 +2.6pp |
| 指标1 FC r（ρ） | 0.98147（0.98320） | 0.9797（0.98146） | −0.0018（−0.0017） |
| FC 逐受试者 r 均值 | 0.91309（p=2.8e-14） | 0.90939（p=2.0e-15） | −0.0037 |
| 指标3 1–30 Hz（判读） | 0.30134 ✓ | 0.30605 ✓ | +0.0047 |
| 指标3 40–125 Hz（照登） | 7.8115 | 7.84633 | +0.035 |
| >40 Hz 占比 real→gen | 1.62% → 4.92% | 1.62% → 5.19% | 尾 +0.27pp |
| 空间 R̂~R 上三角 r | 0.98072 | 0.978 | −0.002（0.97829−0.98072） |
| R1（终点/分母） | 触发（0.70565/0.4449=1.586） | 触发（0.70571/0.4390=1.607） | 同口径效应 |
| 峰频率 real vs gen | 1.46 vs 1.46 Hz | 1.46 vs 1.46 Hz | 对齐 |
| 参数量 | 130,366 | 140,906 | +10,540（调制 MLP 5×2,108） |

判读规则（预登记）：单 seed 不设数值阈值，差异量级与 fmt-f0 跨变体单 seed 差异同框方向性判读。随附差异声明：见本文件顶部「单变量差异声明」——除 cond_adaln 外全部逐字段一致（同 manifest 逐字节/同 seed 1234/同纪律），初始权重经 fork_rng 契约逐位相同。
