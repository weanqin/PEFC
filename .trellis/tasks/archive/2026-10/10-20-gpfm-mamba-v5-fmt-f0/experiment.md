# 实验日志 — 10-20-gpfm-mamba-v5-fmt-f0（gpfm-mamba-v5 × edra_fmt/250 Hz，产出 θ\* 候选 v5-fmt-01）

> 每次「跑批/训练/评估/统计」追加一个 Run 块：跑前先写「目的+配置」，跑后同会话补「结果+结论」，不留空块。
> 格式契约见 `.trellis/spec/research/experiment-log.md`；产物只落本任务声明的变体目录（`.trellis/spec/research/path-isolation.md`）。

## S1 结构落地登记（2026-10-20，非 Run 块——结构/单测/零改动核对的实测登记）

### S1.0 git 快照（零改动基线，2026-10-20 v5-fmt 动工前）

`git status --porcelain` =
```
 M .trellis/research/context.md
 M .trellis/tasks/archive/2026-10/10-19-latent-flow-edra-fmt-train/experiment.md
 M models/latent-flow-v1/latent_flow/baseline/data_utils.py
 M models/latent-flow-v1/latent_flow/viz/connectivity.py
?? .trellis/tasks/10-19-gpfm-mamba-v4b-gnn-f0/
?? .trellis/tasks/10-20-gpfm-mamba-v5-fmt-f0/
?? models/gpfm-mamba-v4b-gnn/
?? models/gpfm-mamba-v5/
?? refs/
```
（悬置未提交改动 diffstat：context.md +5−1 / archive experiment.md +188 / data_utils.py +4 / connectivity.py +56——含 10-19 任务与 10-20 主会话 registry 回填、refs 归档与 10-19/10-20 任务目录、v5/v4bg 变体目录现状，本任务一律不触碰。）完成后对比口径：新增仅 `models/gpfm-mamba-v5-fmt/` 与本任务目录内文件。

### S1.1–S1.3 结构与单测（2026-10-20）

**新变体目录 `models/gpfm-mamba-v5-fmt/`（工作树新增）**：`config.py`（178 行，v5 副本 + fmt 口径字段）/ `model.py`（211 行，自 v5 复制 + 钩子断言放行一处）/ `graph.py`（242 行，自 v5 复制 + fmt 数据根/例数口径）/ `train.py`（139 行）/ `evaluate.py`（115 行）（双遮蔽包装，前缀守卫 `v5-` 沿用）/ `tests/test_v5_fmt.py`（755 行，十一项 = v5 十项 fmt 实例 + 划分固化只读复用）。

**config 字段差异表（v5 → v5-fmt；逐字段对照 `models/gpfm-mamba-v1-fmt/config.py` 先例）**：

| 字段 | v5 | v5-fmt | 依据 |
|---|---|---|---|
| data_root | /root/autodl-tmp/edra_process_v4 | **/root/autodl-tmp/edra_processed_fmt** | fmt 数据（48 例 HC24/MDD24 实测） |
| variant_name | gpfm-mamba-v5 | **gpfm-mamba-v5-fmt** | 三处注册一致（checkpoint 主名 `gpfm-mamba-v5-fmt-{best,last}.pt`） |
| n_val_subjects | 2 | **4** | fmt 34/4/10（manifest 相关——fixate 校验所需） |
| split_manifest_override | （无字段） | **v1-fmt splits/fold0_status_ok_seed42_val4.json（只读）** | S1.2 零重新划分 |
| period_p | 50 | **25** | fmt D3 保物理时长 |
| norm_type | （无字段） | **"layernorm"** | fmt 线 D4 |
| fs | （无字段） | **250.0** | fmt 口径（PSD/轨迹图透传） |
| gen_lowpass_hz | （无字段） | **None** | fmt D2 无低通（v1-fmt 字段现状） |
| outputs_root | gpfm-mamba-f0 | 不变 | PRD R1 同根并列 |
| lr_schedule / early_stop_patience | （无字段） | **（不移植，见偏离 3）** | 训练纪律字段非 fmt 专有 |

graph 七字段与 d_model=62 逐字段不变；geometry_path 本就 fmt 原生 `geometry_edra_fmt.pt`。

**model.py/graph.py 复制与适配（偏离登记见后）**：model.py 可执行代码差异**仅钩子断言一处**——`norm_type not in ("rmsnorm", "layernorm")` 即报错（恰好额外接受 layernorm，其余非默认仍报错；v5 断言收窄的 fmt 最小放开）；另同步更新 4 处描述该断言/norm 位的 docstring 行（如实描述，避免过时文案）。graph.py 机制/公式零改动，仅 `DEFAULT_DATA_ROOT` 改指 fmt 数据根 + `EXPECTED_GEOMETRY_SUBJECTS` 46→48（含 3 处注释/报错文案同步）。

**v1-fmt 训练循环依赖核验（简报要求「先核验」项）**：v1-fmt **无独立训练循环**——其 train.py 亦为 importlib 加载 v1 train.py 的包装（仅 config 单遮蔽）；fmt 口径全部经 v1 train.py 的 getattr 钩子（norm_type/lr_schedule/early_stop_patience）+ config 字段（data_root/period_p/gen_lowpass_hz/fs）+ data.py 钩子（split_manifest_override）生效。故 v5-fmt 包装链 = **v5 双遮蔽（config+model+graph）+ fmt config 字段，训练循环同为 v1 train.py**，钩子同点生效，无需接 v1-fmt 侧任何循环差异。v1-fmt 的 `--skip-gc-only` 为评估期 CLI 使用口径（D1 GC 阻塞），已写入 v5-fmt evaluate.py docstring 用法行，属 S3 执行项。

### 实测登记值

- **参数量（精确）**：v5-fmt 全模型 **152,957** = v5 rms 分支 **152,647** + **310**（fmt 5 norm 位 rmsnorm→layernorm 各 +62 bias：4×MambaBlock pre-norm + head_norm）；明细 seg 2 + in_proj 4,960 + blocks **4×31,000** + graph 3,845 + spatial 15,624 + lns 496 + head_norm **124** + head 3,906。**并列实测（复制零结构改动证据）**：子进程隔离装载 v5 model.py = 152,647，v5-fmt rms 分支（norm_type="rmsnorm"）state_dict 键集与 v5 **逐位一致**；fmt 增量键恰 5 个 norm bias（`blocks.{i}.norm.bias`×4 + `head_norm.bias`）。
- **config_hash pin**：v5-fmt 默认 **`82acf24340e3`**；epochs=1（smoke/bench 覆写）**`f4be07288e9b`**。零漂移复验（子进程隔离复算）：v1 `c6ff80ac1c43` / v2 `fc2a9ce14948` / v5 `1ac433fdd3d0` / **v1-fmt `0dc0d5249c68`**（简报要求从其 config 实算登记的新 pin 复验——与其 2026-10-16 首建 pin 逐位一致）。
- **fmt 口径字段生效核验**：`build_config()` → variant_name **gpfm-mamba-v5-fmt** / d_model==n_channels==62 / norm_type layernorm / fs 250.0 / period_p 25 / gen_lowpass_hz None / n_val_subjects 4；dump_yaml 66 行（header gpfm-mamba-v5-fmt，各 fmt 字段值如实落 yaml）；checkpoint 主名经 v1 train.py `f"{cfg.variant_name}-best/last.pt"` 自动派生为 `gpfm-mamba-v5-fmt-{best,last}.pt`（与 v5 的 `gpfm-mamba-v5-*` 天然隔离）；run 目录 `/root/npi/outputs/gpfm-mamba-f0/v5-fmt-01`。
- **A_eff 等价（实测强于断言口径）**：ricker（σ=0.05 非默认）与 dog（4 参数全非默认）两核下与 latent-flow-v1 源 `effective_adjacency` **max|Δ| = 0.000e+00**（断言口径 allclose rtol=1e-6；pairwise_d 逐位一致）。
- **交替结构（前向钩子序）**：`[mamba_i, spatial_i, ln_i] ×4` 严格交替；msg/LN 逐块独立；全模型恰一个 MexHatAdjacency（A_eff 跨块共享）；**fmt 口径 5 norm 位全 LayerNorm(62)**（4 pre-norm + head_norm，D4 口径实测）。
- **几何门禁（fmt 数据根）**：真 fmt 数据根全 **48/48** 例 channel_names 与 `geometry_edra_fmt.pt` 顺序全等（lru_cache 按 (geometry, data_root) 命中）；名序不一致报错 + 例数不符（1≠48）报错反例均过；σ init = median_d = **0.120868**（与 v5 同几何一致——geometry 文件同源）。
- **划分固化（S1.2）**：fmt 48 例（HC24/MDD24 实测）→ 同参数划分 train **34** / val **4**（2HC+2MDD，[25,35,37,64]）/ test **10**，三折互斥、并集 48；`fixate_split` 幂等校验通过 **v1-fmt 固化 manifest 且文件逐字节不变**（零重新划分，只读复用）；default-off 钩子核验：无 override 字段的 config 仍回 v1 原 manifest 路径。
- **GPU 冒烟（单测①）**：B=256 fwd+bwd 44 ms，peak **2.25 GiB**（与 v5 smoke 单测 2.38/2.315 GiB 同量级——LayerNorm 换入无显存影响）。
- **单测**：`python3 -m pytest models/gpfm-mamba-v5-fmt/tests/test_v5_fmt.py -q` = **11 passed**（271.08 s）；回归（⑩ 子进程独立执行）：v1 七测（test_bandlimit/test_data/test_gc_psd/test_gpr/test_losses/test_model/test_sample）+ v4 test_v4b + v4a test_v4a + **v5 test_v5** 全绿（exit 0）。
- **包装链可跑（不启动训练）**：`python3 models/gpfm-mamba-v5-fmt/train.py --help` exit 0（三模块遮蔽 + 运行时一致性断言 + v1 train.py importlib 装载全链路）；前缀守卫反例：train `--run-id v2d62-hijack`、evaluate 缺 --run-id、evaluate `--run-id main-01` 均 exit 1 拦截（报错前缀 `[v5-fmt wrapper]`）。

### 与简报/PRD 的偏离登记（逐条，无静默变通）

1. **参数量 +310（5×62）而非简报预期 +4×62=152,895**：简报预估口径漏了 head_norm（第 5 个 norm 位）——v1 train.py 透传的 norm_type 同时作用于 `build_mamba_backbone`（4 pre-norm）与 `head_norm=build_norm(d_model, norm_type)`；v1-fmt 先例「容量 +80（5×16）」同口径佐证。按简报自身「以实测为准写死断言」规则钉 **152,957**（单测⑤ ±0 断言）。
2. **model.py docstring 同步面**：可执行代码差异仅断言一处（偏离简报预告的「一处适配」本体），另更新 4 处描述该断言/norm 位的 docstring/注释行（模块头、结构描述 2 行、参数注释）——保留 v5 原文案将事实性错误（称「仅接受 rmsnorm」），如实更新并在此登记。
3. **lr_schedule="constant" / early_stop_patience=10 未自 v1-fmt 移植**：两字段系 v1-fmt 侧「训练纪律长期口径」（v4 系同款、非 fmt 数据口径专有）；v5 config 家族（v2 副本）无此二字段，PRD R4「早停按 config 家族默认如实登记」⇒ v5 家族默认 = **cosine 退火 + 早停禁用**，与 v5-main-01 同训练纪律（v5-fmt-01 vs v5-main-01 对照差异集收窄为数据范式 + fmt 口径字段）；与 v1-fmt/v4a-fmt 线跨线对照须随附该纪律差异（对齐 10-16 任务「对照纯净度不对称」注记）。单测⑥断言两字段不存在。
4. **包装函数/报错文案命名**：`_shadow_v5` → `_shadow_v5_fmt`、`[v5 wrapper]` → `[v5-fmt wrapper]`（v1-fmt 包装先例惯例），逻辑逐行同 v5；train/evaluate docstring 补 fmt 口径（--skip-gc-only 用法行、钩子生效点说明）。
5. **前缀守卫 `v5-` 沿用的口径内边界（如实登记，非新增偏离）**：`--run-id v5-main-01` 前缀合法（PRD R1 拍板的隔离层 = checkpoint 主名）——误传会向 v5 的 run 目录追加 train.log（ckpt 主名不同不覆盖）；缓解 = 包装 docstring 注记 + 本变体 run 一律 `v5-fmt-*` 命名纪律（Run 块登记口径约束）。

### S1 完成时零改动核对

git 快照对比：与 S1.0 基线**逐条相同**（4 处悬置未提交文件 diffstat 不变：context.md +5−1 / archive experiment.md +188 / data_utils.py +4 / connectivity.py +56），新增仅 `?? models/gpfm-mamba-v5-fmt/` 与本任务目录内文件；**outputs/ 零新目录**（未启动任何训练/评估 run——S2 训练链不在本次派发范围）。v1/v1-fmt/v2/v4*/v5/v4bg/latent-flow-v1/common 只读（经 ⑩ 回归 + 4 个 config_hash pin 复验佐证）。

## Run 001 — v5-fmt 冒烟（smoke：限批全链路）（2026-10-20）

### 目的与假设

- 验证 v5-fmt 变体（v5 结构 × fmt 口径）经训练循环的全链路机械正确性：双遮蔽装载、前向/反向、损失有限、val 内部验证、best/last checkpoint 落盘。不做任何质量判读。

### 配置

- 代码：`models/gpfm-mamba-v5-fmt/`（S1 落地版；train.py 三模块双遮蔽 + 运行时一致性断言）
- 参数：全部默认（d_model=62、4 交替块、graph ricker/γ=0.1/γ_m=1.0/W 零初始化/λ=0、norm_type=layernorm（fmt 口径）、period_p=25、无低通；batch 256、seed 1234、EMA 0.999、lr cosine（v5 家族默认，S1 偏离登记 3）；划分 34/4/10，manifest 只读复用 v1-fmt 固化）；`--smoke` = 每类 2 例（合计 4 例）1 epoch
- 环境：base conda（Python 3.12.3 / torch 2.8.0+cu128 / mamba_ssm 2.3.1）@ 本地云服务器（RTX 4080 SUPER 32 GB，16 核 cgroup，BLAS 限 8 线程）
- 数据：`/root/autodl-tmp/edra_processed_fmt`（48 例 @250 Hz，fold0 train 折冒烟子集 4 例）；几何 `/root/autodl-tmp/edra_processed_fmt/geometry_edra_fmt.pt`（只读，fmt 原生）
- 命令：`python3 models/gpfm-mamba-v5-fmt/train.py --smoke --run-id v5-fmt-smoke-a`

### 结果

- 指标：train 4 例 3,720 窗 14 step，全损失有限（total 1.331→1.308）| val 4 例 3,720 窗 total **1.13006**（新 best；mae 0.6057 / cons 0.4515 / tv 0.3092 / corr 0.4190；x̂1 std(EMA) 0.8124 vs x1 std 0.9114）| it/s 收尾 7.7 | peak **2.19 GiB** | config_hash `f4be07288e9b`（= S1 登记的 epochs=1 覆写 pin，逐位一致）
- 产物：`outputs/gpfm-mamba-f0/v5-fmt-smoke-a/`（config.yaml + env_snapshot.txt + channel_mean/std.npy + logs/{train.log, metrics.jsonl, summary.json} + checkpoints/{gpfm-mamba-v5-fmt-best,last}.pt 八类齐）
- 图表：无（冒烟不做图）

### 结论

- 支持（机械正确性）：v5-fmt 经训练循环全链路（双遮蔽装载/前向反向/损失有限/val 确定性复算/best+last 落盘）无异常；显存余量充裕。

## Run 002 — v5-fmt 1-epoch 基准（bench：34 例全量）（2026-10-20）

### 目的与假设

- 主 Run 前材料登记：参数量、峰值显存、it/s、val 内部验证总 loss、100-epoch 外推时长、W/σ 可学性点值（best ckpt 事后提取 raw + EMA 双口径，v5-main Run 004 同款）。不做质量判读。

### 配置

- 代码：同 Run 001
- 参数：同 Run 001 唯 `--epochs 1 --bench`；训练折全量 34 例
- 环境：同 Run 001
- 数据：同 Run 001（train 34 例全量 + val 4 例内部验证）
- 命令：`python3 models/gpfm-mamba-v5-fmt/train.py --epochs 1 --bench --run-id v5-fmt-bench-1ep`

### 结果

- 指标：params **152,957** | train **34 例 31,620 窗 123 step** | train total 1.2114 | **val total 1.07710**（新 best；mae 0.5798 / cons 0.4251 / tv 0.2977 / corr 0.4237；x̂1 std(EMA) 0.7794 vs x1 std 0.8630）| **8.33 s/epoch（train 段）**、it/s 14.76（收尾 18.1）| **peak 2.19 GiB** | 100-epoch 外推 **≈0.231 h**（train 段；加逐 epoch val/checkpoint 实际 ~15–25 min 量级）| config_hash `f4be07288e9b`
- **W/σ 可学性点值（best ckpt 事后提取，raw + EMA shadow 双口径）**：W (62,62) 零初始化 → 1 epoch（123 step）后 raw **‖W‖_F = 1.0274** / EMA **0.0821**（decay 0.999 滞后量级自洽）/ raw max|W| = 0.0615；σ init=median_d 0.120868 → raw **0.123800** / EMA **0.121074**（微调中）——与 v5-main bench 同期水平（raw ‖W‖ 1.2356/σ 0.1262）同量级，W 学得动（同款证据口径）
- 产物：`outputs/gpfm-mamba-f0/v5-fmt-bench-1ep/`（八类齐；best ckpt epoch 0 / step 123 / best_val 1.07710）
- 图表：无（bench 不做图）

### 结论

- 支持（主 Run 材料齐备）：机械/资源可行（显存 2.2 GiB、~15 it/s、主 Run 预计 ~15–25 min）；W 可学性成立；1-epoch val 1.077 不作质量判读。
- 下一步：**主 Run `v5-fmt-01` 启动确认（PRD 钉的执行期用户确认项）**——材料汇报用户、明确确认后启动（100 epoch 上限 + nohup 脱离式 + 显式重定向）。

## 用户拍板登记（2026-10-20，主 Run 启动前）

- **早停 10 启用 + 主训练启动（用户明示）**：「cosine+早停10开启，仅修改这个配置，其余不变，无需子agent测试，修改配置后进行主训练」——配置变更 = **仅新增 `early_stop_patience: int = 10` 字段**（v4-B 口径字段，v1 train.py getattr 透传；cosine 为 v5 家族默认本就在用，无改动）；按用户指示**未走子代理测试**，以最小验证替代（字段装载实读 + 新 pin 派生 + 包装链 `--help` exit 0）。S1 偏离登记 3 的「早停禁用」表述自本条起作废（lr_schedule 仍不移植，cosine 不变）。
- **config_hash pin 更新（口径变更必登记）**：默认 `82acf24340e3` → **`a64c0d416756`**；epochs=1 `f4be07288e9b` → `603188ccc0a1`（S1 登记的旧 pin 对应无早停字段版本，留档对照）。
- **对照差异集更新**：v5-fmt-01（cosine + **早停 10**）vs v5-main-01（cosine + 早停禁用跑满 100）——跨线对照须随附。

## Run 003 — v5-fmt 主 Run（main：100 epoch 上限 + 早停 10）（2026-10-20）

### 目的与假设

- 产出 fmt 管线的 θ\* 候选：v5-fmt（v5 结构 × fmt 口径）fold0 单 seed 完整训练——收敛曲线、best val、早停轮次；训练后接 A_eff/W 快读出（对齐 v5-main Run 006 协议，与 v5-main-01 对照）。R9 确认 = 本节上方用户拍板（早停 10 + 直启主训练）。

### 配置

- 代码：`models/gpfm-mamba-v5-fmt/`（S1 落地版 + 本日早停字段变更；git 22afbc5 工作树）
- 参数：全部默认（= Run 001 口径 + **early_stop_patience=10**；epochs 100、batch 256、seed 1234、EMA 0.999、lr cosine（v5 家族默认）、无低通 fmt 口径；无 --smoke/--bench 覆写，config_hash 应回新默认 pin **`a64c0d416756`**）
- 环境：同 Run 001
- 数据：同 Run 002（train 34 例全量 + val 4 例内部验证，test 10 例留评估）
- 命令：`mkdir -p outputs/gpfm-mamba-f0/v5-fmt-01/logs && nohup python3 models/gpfm-mamba-v5-fmt/train.py --run-id v5-fmt-01 > /root/npi/outputs/gpfm-mamba-f0/v5-fmt-01/logs/train.log 2>&1 &`（脱离式 + 显式重定向）

### 结果

- **早停按拍板口径触发**：**epoch 39/100 停**（连续 10 epoch 无新 val best，metrics.jsonl 本行附 `"early_stop": true`）；**best epoch 29（0 基 28）、best val total 0.70242**；总墙钟 ~5.5 min（39 epoch × mean 7.22 s train 段，17.03 it/s，peak 2.19 GiB，commit 22afbc5）
- config_hash **`a64c0d416756`**（= 早停字段变更后新 pin，逐位一致）
- 收敛轨迹（val total）：ep0 1.07711 → ep9 0.89326 → ep19 0.74141 → **ep28 0.70242（best）** → ep38 0.71886（best 后回升 → 触发早停）；train total 1.2095→0.5751；末段 val 分量 mae 0.4067 / cons 0.2645 / tv 0.2167 / corr 0.2608；x̂1 std(EMA) 0.7135 vs x1 std 0.8630
- 产物：`outputs/gpfm-mamba-f0/v5-fmt-01/`（八类齐：config.yaml + env_snapshot + channel_mean/std + logs/{train.log, metrics.jsonl, summary.json} + checkpoints/{gpfm-mamba-v5-fmt-best,last}.pt）
- 图表：无（训练 run 不做图）

### 结论

- 支持（θ\* 候选产出）：稳定收敛、早停机制按拍板口径工作（best@29 + 10 轮耐心即停，省 61 epoch）；best ckpt（ep29）为下游 study2 NI-MSE-50-q0.90 的 θ\* 候选（EMA shadow 口径）。
- 边界：val 内部验证 ≠ 质量/代理脑合格性判定；与 v5-main-01 对照受**训练纪律差异**调制（早停 10 vs 跑满 100——本 run best 出现更早（ep29 vs ep64）且未经历长平台段）。

## Run 004 — v5-fmt A_eff/W 快读出（EMA shadow，best ckpt ep29；与 v5-main-01 对照）（2026-10-20）

### 目的与假设

- PRD R5：θ\* 落地即取 A_eff/W 快读出（σ/W 范数、M vs W 分解、结构统计——对齐 v5-main Run 006 协议的紧凑版，**不做图**；图与 ×GC 相关属 study2 NI-MSE 执行期）。

### 配置

- 代码：内联提取（graph.py 自 v5-fmt 目录装载，shadow graph.\* 子字典 strict 回灌；几何 choke point fmt 48 例断言生效）
- A_eff 源：`outputs/gpfm-mamba-f0/v5-fmt-01/checkpoints/gpfm-mamba-v5-fmt-best.pt` EMA shadow（raw 对照值随记）
- 统计公式：对齐 v5-main Run 006（offdiag CV/行和 CV/Fro 不对称度/中心-环绕/距离衰减/M vs W 分解）
- 命令：python 内联（结果落 `readout_aeff_quick.json` 随 run 目录）

### 结果

- 指标（EMA shadow；**括号 = v5-main-01 对照**，注意其训练纪律差异）：σ **0.12977**（raw 0.13214；σ/中位距 **1.074**，v5-main 1.174）；‖W‖_F raw **2.798** / EMA **2.339**（v5-main 5.090/4.945——训练时长短 61% 所致，raw 仍在增长段）；**W 占比 γ·W/(γ_m·M) = 0.85%**（Fro；v5-main 1.7%）——几何先验主导更强；A offdiag CV **2.79**（1.686）/ 行和 CV **0.468**（0.307）/ 不对称度 **1.2%**（2.4%）/ offdiag 正占比 **0.542**（0.652）；**中心-环绕：σ内 +0.452 / σ外 −0.223**（v5-main +0.446/−0.181——环绕净抑制更深）；|A_eff|×距离 Pearson **−0.698**（−0.841）；对角均值 0.998（0.994）
- 产物：`outputs/gpfm-mamba-f0/v5-fmt-01/readout_aeff_quick.json`（全量统计随档）

### 结论

- 支持（结构先验同型成立）：与 v5-main-01 同构的带通几何结构（局部兴奋/环绕抑制/距离衰减/行和异质），且**环绕抑制更深、几何主导更强（W 占比更低）**——fmt 管线 + 更短训练下 W 残差更小；σ 学习收窄（1.074 中位距）。
- 边界：快读出 ≠ 完整 A_eff 读出（无图、无 ×GC 相关——属 study2 NI-MSE 执行期，θ\* = 本 run best ckpt）；与 v5-main 对照受早停纪律差异调制（如上）。
- 收尾处置登记（2026-10-20，trellis-check 后）：①检查阻断项修复——test_v5_fmt.py 三处同步早停拍板（line 417 断言改 `early_stop_patience == 10` + lr_schedule 仍不存在；pin 常量改现行 `a64c0d416756`/`603188ccc0a1`，旧 pin 留档注释），全量复跑 **11 passed**（240 s）；②检查建议项处置——本文件文末种子模板骨架删除；Run 004 内联读出脚本落档 `outputs/gpfm-mamba-f0/v5-fmt-01/readout_aeff_quick.py`（复现性三要素补齐，Run 004 配置节命令行同步指向）。
