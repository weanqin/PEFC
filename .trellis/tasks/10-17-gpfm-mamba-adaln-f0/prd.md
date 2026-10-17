# v4a-fmt adaLN 调制归一化变体 fold0 前置验证（emb(t) 条件调制）

## Goal

在 v4a-fmt 实验设置（edra_processed_fmt，fold0 34/4/10）上，把速度网络的 5 个 LayerNorm 位替换为 adaLN：调制参数（scale/shift）由流时间嵌入 `emb(t)` 经 MLP 生成，对归一化后特征做 `scale ⊙ LN(x) + shift` 调制。fold0 单 seed 训练 + 正式评估，与 v4a-fmt 基线（v4afmt-main-01）单变量对比，结果记入 experiment.md。

## Background（勘察确认的事实）

- **对照基线已就绪**：v4afmt-main-01（55ep 早停；MAE 0.70565 / CRPS 0.53912 / FC r 0.98147 / 指标3 1–30Hz 0.30134 / 逐通道 r 0.12335）；fold0 manifest 已固化（48 例 34/4/10，`models/gpfm-mamba-v4a-fmt/splits/`）
- **t 的现状**（`models/gpfm-mamba-v1/model.py`）：t 为每样本标量；训练时 `t = sigmoid(z), z ~ N(t_loc=-0.8, t_scale²=0.8)`（logit-normal）；采样时 Euler 网格逐步调用 `v_θ(x_t, t, X_past)`。`time_embedding(t, 16)` 正弦嵌入已存在，当前仅拼接进输入投影（future 128 位置广播、past 段为零）——**未参与任何调制**
- **norm 位共 5 个构建点**：4×MambaBlock pre-norm（`x + mamba(norm(x))`，mamba.py）+ head_norm（model.py）；仓库无其他 norm 构建点（fmt 任务 D4 时已核尽）
- **惯例约束**（`.trellis/spec/research/reproducibility.md`）：default-off config 钩子 + `getattr` 透传，默认分支与原结构逐位一致（state_dict 键不变、旧变体 config_hash 零漂移，单测必测）；薄变体目录 path-isolation（config 完整副本 + wrapper 三要素 + run_id 前缀守卫 + 独立 outputs 根）
- **评估口径延续 fmt 任务**：genGC vs real GC 阻塞（`--skip-gc-only`）、fs=250 全链透传、`gen_lowpass_hz=None`、`period_p=25`、指标3 判读窗 1–30 Hz（40–125 Hz 照登）
- **门禁**（surrogate-brain-qualification.md）：fold0 单 seed 前置验证，不越 5-fold 门禁

## Requirements

- **R1 共享模块 default-off 钩子**（v1 `mamba.py`/`model.py` + `train.py`/`sample.py` 构造点透传）：adaLN 由新 config 字段启用（命名 design 定稿）；默认关闭分支与现状**逐位一致**（结构/state_dict 键/前向，单测断言旧变体 config_hash 零漂移）；adaLN 分支 = 5 个 norm 位全部换成 `LN + emb(t) 调制`，恒等初始化（初始等价于纯 LayerNorm，具体形式待 Q1 拍板后 design 定稿）
- **R2 新薄变体目录**（`models/gpfm-mamba-v4a-fmt-adaln/`，命名 design 定稿）：v4a-fmt config 完整副本 + 唯差异字段；splits manifest 与 v4a-fmt **逐字节一致**（断言）；outputs 落新根 `/root/npi/outputs/gpfm-mamba-adaln-f0/`；run_id 前缀守卫；wrapper 复用 v1 入口（无 --rank 需求，r=16 固定）
- **R3 训练纪律与 v4a-fmt 逐字段一致**：fold0 单 seed 1234、早停 10、恒定 lr 1e-3、val4、EMA 0.999、batch 256——与基线构成**单变量对比**（唯一差异 = norm LayerNorm→adaLN(t) + 调制 MLP 参数量增量，登记精确数值）
- **R4 评估**：与 v4a-fmt 同口径全量（Q1 质量 + 指标1 FC + 指标2 视觉图（250 Hz 标秒）+ 指标3 PSD（1–30 Hz 判读 + 40–125 照登）+ >40 Hz 功率占比观察）；GC 阻塞延续（产物零 GC 对比文件，逐文件核验）
- **R5 记录**：experiment.md 逐 Run 登记（目的/配置/结果/结论四区，无空块；Run 块标签「前置验证 · fold0 · 单 seed · 单次运行」）；跨变体对比表随附单变量差异声明；方向性判读规则预登记（design §3，不设数值阈值）

## Acceptance Criteria

- [ ] 单测全绿：①钩子默认关闭 = 旧变体（v1/v2/v3/v4a/v1-fmt/v4a-fmt）config_hash 零漂移 + 结构/state_dict 逐位一致（v4/v4b 经 v1 mamba.py 构建骨干，含于影响面——由既有 test_v4b 回归覆盖）；②adaLN 分支断言（5 个调制位、state_dict 键、**fork_rng 契约下**恒等初始化前向与 LayerNorm 基线逐位一致——design §1.1）；③manifest 逐字节一致断言；既有测试回归通过
- [ ] 变体目录 + 单测入库；smoke（dataloader 契约核对）→ bench → 主 Run（早停或满 100ep，ckpt+metrics 落盘）全流程完成
- [ ] 正式评估产物齐全（Q1/指标1/指标2/指标3）且**无任何 GC 对比文件**
- [ ] 与 v4afmt-main-01 对比表（Q1 全维度/FC/PSD/>40Hz 占比/参数量）+ 单变量差异声明随附
- [ ] experiment.md 无空 Run 块；结论「单次运行」标注；不越 5-fold 门禁

## Out of Scope

- v1 侧 adaLN（用户指定 v4a 基座）；v2/v4b 等落选变体
- 5-fold 主实验 / 多 seed（须另行拍板）
- genGC vs real GC 对比（阻塞延续）；低通重开；r 扫描
- 训练纪律变更（超参搜索等）

## Open Questions（待用户拍板）

- 无（Q1 已拍板；D2/D3 为技术设计决策，design 阶段依据 research 输出定稿）

## Key Decisions（拍板后回填）

- **D1 adaLN 形式 = 纯 scale/shift**（用户 2026-10-17 拍板）：5 个 norm 位全部 `out = (1+s) ⊙ LN(x) + b`，调制参数 `(s,b)` 由 `emb(t)` 经逐 site MLP 生成；**MambaBlock 残差结构不动**（不加门控）；恒等初始化（MLP 末层零初始化 ⇒ 初始严格等价纯 LayerNorm）——与 v4a-fmt 基线构成最干净的单变量对比
- **D2 调制 MLP = 逐 site `SiLU→Linear(16, 2×62)`（零初始化）×5**（design 定稿，research §2）：与共享单 MLP 参数量/函数类完全等价（+10,540），逐 site 取 DiT 先例与键布局（`blocks.{i}.mod.1.*` + `head_mod.1.*`）；调制信号复用 `time_embedding(t,16)` 原始特征不加投影；per-sample 全局调制广播全 512 位置（含 past 段，DiT 同款语义，采样自洽性 research §1.2 已核）
- **D3 变体目录 = `models/gpfm-mamba-v4a-fmt-adaln/`，RUN_PREFIX = `v4afmt-adaln-`**（design 定稿）：无 --rank（r=16 固定）；钩子字段 = `cond_adaln: str = "off"`（枚举字符串，lr_schedule 先例；否决 norm_type="adaln"——正交维度）

## 子研究路由（1.0b / 1.0c）

- **分类结果**：`study1` —— 代理脑速度网络构型消融（norm 条件化）的 fold0 训练与检验（`task.py set-scope` 已执行）
- **依据**：任务主体是 gpfm-mamba 候选构型 v4a 的结构变体在 Study I 门禁下的训练与评估
- **新路径/方法检测（1.0c）**：`新路径：adaln → research 已触发` —— adaLN（流时间条件调制归一化）不在 context.md 技术路线表已登记变体内（表内变体：v1/v2/v4b/v4a/v1-fmt/v4a-fmt，均无 t 条件 norm 机制），trellis-research 适配研究已完成（2026-10-17）
- **research 产物**：`research/adaln-adaptation.md`（适配结论/风险/可执行方案，7 项研究问题逐项结论 + 内外部锚点；研究结论已采纳进 design）
