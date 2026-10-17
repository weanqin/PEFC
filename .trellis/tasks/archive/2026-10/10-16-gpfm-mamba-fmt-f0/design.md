# design.md — 新预处理范式 v1/v4a fold0 训练（edra_processed_fmt）

## 1. 架构与边界

遵循薄变体协议（`.trellis/spec/research/path-isolation.md`）：两个新薄变体目录，基线代码（v1 目录全部模块）只读复用、零默认行为改动；新能力一律 default-off 钩子经 `getattr(cfg, 字段, 默认值)` 透传。

### 1.1 路径清单（path-isolation 契约）

**新建（写）**：

| 路径 | 内容 |
|---|---|
| `models/gpfm-mamba-v1-fmt/` | config.py（完整副本+新字段）+ train.py/evaluate.py 包装 + fc_metric.py 包装 + splits/（首跑固化）+ tests/ |
| `models/gpfm-mamba-v4a-fmt/` | 同上（config 基于 v4a 副本；包装含 --rank 透传，接口对齐 v4a） |
| `/root/npi/outputs/gpfm-mamba-fmt-f0/` | 本任务 outputs 根（两变体共享，run_id 前缀隔离：`v1fmt-` / `v4afmt-`） |

**只读依赖（不改写）**：`models/gpfm-mamba-v1/`（model/data/gpr_prior/mamba/losses/train/sample/evaluate/gc_psd/fc_metric/visualizer 全部模块）、`models/gpfm-mamba-v4a/`（config 参照物）、`/root/autodl-tmp/edra_processed_fmt/`（数据集）、`common/`。

**禁写**：旧任务 outputs 根 `/root/npi/outputs/gpfm-mamba-f0/`、旧变体 splits manifest、`/root/autodl-tmp/edra_processed_fmt/` 本体。

**例外改动（default-off 钩子，须旧变体 hash 零漂移单测）——review 回填（1.4.1）+ 用户 norm 变更（D4）后共五处**：

1. `gc_psd.py::run_all` 加 `skip_gc: bool = False`：True 时跳过阶段 1/2（BIC 选阶 + 两组 VAR 条件 GC + 一致性/bootstrap），**照跑 PSD 与空间结构归因**（二者只依赖 `gc_input.npz` 的 gen/real/win 元数据，与 GC 阶段无数据依赖——review 已核 `gc_psd.py:470-530`）；summary 与日志显式记 `"gc": "skipped(blocked)"`，且不落任何 GC 产物（含 `gc_real/gc_gen/gc_F_p_*` npy 与结果 json 的 gc 数组/图引用）。默认 False = 原行为逐位不变；docstring 频率注释（fs=500→0.977 Hz）顺带改为 fs 参数化表述。
2. `evaluate.py` 三点（同一文件，一处 fs 假设面）：①`run_all` 调用点 `fs=500.0` 改 `fs=float(getattr(cfg, "fs", 500.0))`；②新增 `--skip-gc-only` 旗标——与既有 `--skip-gc` **argparse mutually exclusive group**（语义互斥：整段跳过 vs 仅跳 GC，同传报错）；③`_traj_fig_data` 的 `"fs": 500`（~292 行）改 `float(getattr(cfg, "fs", 500.0))`——轨迹图时间轴（指标2 人工判材料）在 250 Hz 下须正确标秒（review Major，双 mandate 一致发现）。
3. `fc_metric.py`：`--outputs-root` 参数，默认值 = 现硬编码 `/root/npi/outputs/gpfm-mamba-f0`（默认路径零漂移）；fmt 调用传新根。
4. **norm 类型钩子（用户 2026-10-16 拍板 D4：RMSNorm→LayerNorm）**：`mamba.py::MambaBlock/build_mamba_backbone` 与 `model.py`（head_norm）加 `norm_type: str = "rmsnorm"` 参数，默认分支 = `nn.RMSNorm` 原结构逐位一致（state_dict 键不变）；`train.py:278`/`sample.py:52` 构造点经 `getattr(cfg, "norm_type", "rmsnorm")` 透传（bottleneck_rank 同款先例，两处构造点已核）；fmt 两变体 `norm_type="layernorm"`（**5 个 norm 位全换**：4×MambaBlock pre-norm + 1×head norm——仓库仅此两处 RMSNorm 构建点，已 grep 核尽）。LayerNorm 含 bias ⇒ 新变体 state_dict 多 `*.norm.bias`/`head_norm.bias` 键（仅 fmt ckpt，旧变体零影响）；mamba.py/model.py docstring 的 RMSNorm 表述顺带参数化。
5. （不改码，登记）`gc_psd.py` 独立 `main()`（~550 行）硬编码旧 outputs 根/旧 data_root/`--fs` 默认 500——**fmt run 一律经 evaluate.py 入口，不走该独立入口**（误用为响亮 FileNotFoundError）。

**已知坑预登记**（review 汇总）：
- `sample.py` 低通经 `brickwall_lowpass(out, cut_hz)` 用默认 `fs=500`——本任务 `gen_lowpass_hz=None`（关闭）不触发；**日后在 250 Hz 重开低通必须传 fs=250**（config 注释 + experiment.md 登记）
- `diag_hf.py` 三处 fs=500 + 硬编码旧 run id——**fmt 禁用**；D3 的 >40 Hz 功率占比改从 `eval/psd.npz`（fs=250 正确频轴）计算，落 experiment.md
- `gpr_prior.py:42,86` bandlimit 路径 fs=500——fmt 恒不触发；`evaluate.py:172` lowpass_last_dim 为无调用方休眠 shim
- 其余 fs 耦合参数逐项核毕无需改（review Mandate A）：iso/pp_jitter/sigma_s/euler_steps/t 网格/损失权重均非 fs 耦合；**p=25 换算核验正确**（OU e-folding = p/(π·fs) 秒，50@500 ≡ 25@250 严格等价）

### 1.2 变体 config 差异（相对各自基线的全部字段差异）

**v1-fmt**（基线 = v1 config，差异字段）：

| 字段 | v1 | v1-fmt | 依据 |
|---|---|---|---|
| data_root | edra_process_v4 | `/root/autodl-tmp/edra_processed_fmt` | R1 |
| outputs_root / variant_name | gpfm-mamba-f0 / gpfm-mamba-f0 | `/root/npi/outputs/gpfm-mamba-fmt-f0` / `gpfm-mamba-v1-fmt` | R1 |
| period_p | 50 | **25** | D3 保物理时长 |
| gen_lowpass_hz | （无字段，getattr 默认 40） | **None**（显式字段） | D2 不滤 |
| fs | （无字段，getattr 默认 500） | **250**（显式字段） | PSD 口径 |
| n_val_subjects | 2 | 4 | 长期口径 7/1/2 |
| split_manifest_override | （无字段） | 本目录 splits/ 自备 manifest 路径 | 48 例须新固化 |
| early_stop_patience | （无字段=禁用） | 10 | 长期口径 |
| lr_schedule | （无字段=cosine） | "constant" | 长期口径 |
| norm_type | （无字段=rmsnorm） | **"layernorm"** | D4 用户拍板 |

**v4a-fmt**（基线 = v4a config）：同上表全部字段（v4a 已有 override/早停/恒 lr 字段，改值即可）+ `bottleneck_rank=16`、`d_model=62` 保持。checkpoint 名 = `gpfm-mamba-{v1,v4a}-fmt-{best,last}.pt`，与历史 run 天然隔离。

其余字段（K_s/K_t 其他参数、损失权重、batch 256、epochs 100、lr 1e-3、seed 1234、EMA 0.999、euler/n_samples 等）与基线逐字段一致。

### 1.3 数据流与划分契约

- dataloader 零修改：`EdraWindowsDataset` 读 `data (62,60001)` → 930 窗/例（512=384+128，stride 64）→ z-score 逐通道、仅训练折统计（data-pipeline.md 契约不变）
- fold0 划分：`stratified_subject_folds(48 例, 5, seed=42)` → fold0 测试折 + 训练折内 `n_val_subjects=4`（每类 2）→ 预期整数分配 ~34/4/10（**以首跑固化 manifest 实际值为准登记**；7/1/2 为长期口径语义，非精确比例）
- 两变体 manifest：同参数生成 ⇒ 逐字节一致；单测断言 test 折逐 ID 一致（v4a 先例：与 v4b manifest 逐字节一致断言）。v1-fmt 首跑固化后，v4a-fmt 首跑校验复用同划分
- `subject_filter='status_ok'`（fmt manifest 48 行全 ok；qc_pass 口径不可用——字段留空）

### 1.4 评估流（每变体主 Run 后）

1. `evaluate.py`（经各自包装）：Pass A Q1 质量（MAE/CRPS/逐通道 r 等，fs 无关）+ Pass B 落 `gc_input.npz`（生成侧**未滤波**样本）+ `--skip-gc-only`：PSD（fs=250，nperseg 512 → 0.488 Hz 分辨率）+ 空间结构归因照跑；**GC 一致性不出现在任何产物**
2. `fc_metric.py --outputs-root <新根>`：指标1 FC 一致性
3. 指标2 视觉：trajectory/PSD 图经 common/visualizer 落各自 run figs/（时间轴 fs=250 正确标秒），用户人工判
4. **指标3 PSD 判读口径预登记（review Major 回填，跑前落 experiment.md，evaluation-calibers「指标定义先于运行」）**：
   - 通过判据计算式 = 组平均谱在 **1–30 Hz**（δ–β；下沿 1 Hz 对齐 release 带通——真实侧 <1 Hz 为滤波地板，不得入口径窗）的 mean |Δlog10| ≤ 0.5，数据源 `eval/psd.npz`，变体侧小脚本补算（不动共享模块）
   - 高频观察段适配为 **40–125 Hz**（Nyquist 125 Hz；无阈值，照登数值）+ >40 Hz 功率占比（D3 观察项：模型是否自行待在带内）
   - **全频段 `mean_abs_logdiff`（summary.json 现有统计量）在新口径下不作判读依据**——生成侧未滤波 + 真实侧 1–40 Hz 地板，该量被地板区主导
   - D7 回填时同步修 evaluation-calibers / surrogate-brain-qualification 的指标3 频段定义（原 60–250 Hz 表述系 500 Hz 耦合）
5. 口径登记：experiment.md 每 Run 块记录 GC 阻塞、无低通、fs=250、p=25；判读注记三条（review）：①同 epochs ≠ 同步数（~31.6k 窗/epoch vs 旧 42.3k，EMA τ=1000 步 ≈8 vs 6 epoch）；②窗口物理时长×2（2.048 s）⇒ Euler/TV 平滑物理尺度同步×2；③**对照纯净度不对称**——v4a-fmt vs 旧 v4a 较干净（差异=数据+p+低通），v1-fmt vs 旧 v1 额外叠加训练纪律变量，跨范式判读权重向 v4a 侧倾斜

## 2. 关键权衡

- **共享 outputs 根 + 前缀隔离**（沿 v4a 先例）而非每变体独立根：同任务内对比方便，fc_metric 只需一个新根参数；风险 = 前缀守卫失守 → wrapper 双重校验（run_id 前缀 + checkpoint 名含 variant_name）
- **skip_gc 钩子而非拆分 run_all**：最小改动面（一个分支参数），PSD/空间归因代码路径零变化；代价 = run_all 内 GC 阶段代码保留不删（阻塞是暂时性，D1）
- **p=25 + 不滤 + norm 换 LayerNorm + 数据多变量同时变**：跨范式对比非单变量——判读一律随附差异清单（fs/预处理/48例/p/低通/**norm**/训练纪律），只做方向性观察，不做归因强主张。norm 换的容量影响：LayerNorm 多 bias 参数（v1-fmt +80 = 5×16、v4a-fmt +310 = 5×62），相对 33.8k/129.8k 可忽略但登记；语义变化 = 均值中心化归一（RMSNorm 仅尺度缩放）

## 3. 运行与回滚

- 运行环境 = environment.md 本地契约（RTX 4080S 32GB、16 核 cgroup、BLAS 护栏、nohup 长跑）；跑前核对无环境变更
- 顺序（每变体）：smoke（1ep 小子集）→ bench（1ep 全量，显存/时长外推）→ main（≤100ep，早停 patience 10，nohup）→ 评估四步
- experiment.md 先登记目的+配置再跑（research 纪律）；结果+结论同会话回填
- 回滚点：①共享模块钩子若致旧变体 hash 漂移/单测红 → 立即 revert 钩子重设计；②smoke/bench 发现 dataloader 契约不匹配 → 停在数据接入层修复，不进训练；③主 Run 异常 → run 目录保留（不覆盖），换 run_id 重跑
