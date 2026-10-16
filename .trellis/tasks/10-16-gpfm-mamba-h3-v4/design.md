# Design — H3：v4-B 通道独立骨干 + cMLP 组件式混合读出（fold0 单 seed）

> 需求见 `prd.md`；架构由用户 2026-10-16 指定（弃 A 用 B，保留通道身份、无低秩混合）。seed（`.trellis/research/h3-cross-channel-seed.md`）的 A 方案框架作废存档。

## 1. 假设与对照框架（改写：混合位置 × 显式性）

v1/v2 的 GC 差异（0.2755 vs 0.1227，正式口径）在「结构是否鼓励跨通道滞后依赖」上。v4-B 引入第三种结构姿态：

| 臂 | trunk 跨通道混合 | 混合层可辨识性 | GC 一致性（Run 017） |
|---|---|---|---|
| v1（d16） | 隐式、全程（瓶颈强制叠加） | 不可辨识（分散于各层） | **0.2755** |
| v2（d62） | 可选可学（接口/块内投影线性层） | 不可辨识，且可退化通道对齐 | 0.1227 |
| **v4-B** | **trunk 严格零混合（通道独立）** | **唯一混合层在顶层，按源通道列组显式可辨识（cMLP）** | 待测 |

**判读逻辑（单 seed，判定权在用户）**：
- **H3 支持**：v4-B GC r 回升至 v1 带（0.2+ 量级，cMLP 结构理论上限更高）且五维度（MAE/CRPS/逐通道 r/FC r/空间）不显著塌缩 ⇒ 显式组件式跨通道读出是 GC 的有效载体。
- **H3 部分支持**：GC r 明显高于 v2（≥0.18）但未达 v1 带。
- **H3 不支持**：GC r ≈ v2 水平（≈0.12）⇒ GC 偏弱主因不在结构侧，归因转向损失/训练目标（无跨通道时序监督）。**不对称守护**：回升 ⇒ 显式混合结构充分；不回升 ≠ 结构无关被证伪（残留解释：表征容量、损失侧监督缺失、数据），登记后移交。
- **归因局限（必随结论登记）**：v4-B vs v2 非单变量（CI trunk + cMLP head 两处变化）；若需归因分解（CI 代价 vs head 收益），后续可加「CI trunk + 原 head」「共享 trunk + cMLP head」分解变体——条件后续，不在本任务。
- 三态阈值跑前预注册；落点 0.15–0.22 邻域时预设「提请用户加 seed 再判」。
- **过拟合竞争解释预登记（评审补充）**：v4-B 参数/样本比 22:1，若 best epoch ≪ 100 或 val 总 loss 明显劣于 v2，「五维度不塌缩」条件的失败可能由过拟合（而非结构）驱动——判读材料必须随附 best epoch 与 train/val gap 数值，过拟合列为竞争解释后再下结构结论。

**滞后语义对应（判读材料的表述基准）**：v_θ(t,i) = g_i(h_{t,1..62})，head 混合虽发生在同一时间步（point-wise），但 h_{t,j} 是通道 j 全部因果历史（past 384 + future 前缀）经独立 scan 的表征 ⇒ 有向滞后依赖**结构上可表达**；且 W^{(i),1}_{:,j} 按源通道分块 ⇒ **可辨识**。这正是 Tank 2021「g_i 输入 = 全部通道滞后历史」的表征版移植（输入由原始滞后值换为各通道历史表征，列组语义保持）。

## 2. v4-B 架构（用户指定规格的逐条落地）

### 2.1 张量流

```
输入 token 序列 L=512 = past 384（条件前缀）+ future 128（流变量位置）——与 v1/v2 相同
逐 token 通道轴拆分：x_{t,j}（B, 62, 512）
   │  每通道 j 拼接共享辅助特征：[x_{t,j}(1) ‖ seg(1) ‖ t_emb(16)] = 18 维
   │  （t_emb 仅 future 位置非零，广播到全部通道——与 v1 同源）
   ▼
[1] 逐通道独立输入投影  Linear(18 → d_ch=16)        —— 权重跨通道共享（AI 默认，见 §3）
    输出 (B, 62, 512, d_ch) ≡ (B×62, 512, d_ch)      —— 通道身份保留为独立数据流
   ▼
[2] 4 × MambaBlock(d_ch=16, d_state=16, d_conv=4, expand=2)   —— 通道独立 selective scan
    逐通道沿时间轴因果递归；块结构 = 预 RMSNorm + mamba_ssm.Mamba + 残差（复用 v1 mamba.py 零改动）
    得 h_{t,j} ∈ R^{d_ch}
   ▼
[3] 截取 future 128 位置 + 逐通道 RMSNorm(d_ch)
    同时间步汇聚：h_t = [h_{t,1}, …, h_{t,62}] ∈ R^{62·d_ch=992}     (B, 128, 992)
   ▼
[4] 顶层 cMLP Head —— 唯一跨通道混合层（62 个独立子 MLP，逐时间步 point-wise）
    z_{t,i}  = σ( Σ_{j=1..62} W^{(i),1}_{:,j} h_{t,j} + b¹_i )     W^{(i),1} ∈ R^{d_z×992}
    v_θ(t,i) = W^{(i),2} z_{t,i} + b²_i                            W^{(i),2} ∈ R^{1×d_z}
   ▼
v_θ ∈ (B, 62, 128)
```

### 2.2 模块定义与结构事实

- **[1] 通道独立编码**：每通道特征 18 = 1（本通道信号值；future 位置即流变量 x_t 该通道）+ 1（段类型可学习标量，past/future 共 2 个参数）+ 16（全局 t 正弦嵌入，`model.py::time_embedding` 复用）。**无任何跨通道算子**。
- **[2] CI 骨干**：官方 `mamba_ssm.Mamba` 在 (B×62, 512, d_ch) 上直接运行（通道折进 batch 维）；块内 conv depthwise、SSM 逐特征、块内 in_proj/out_proj 全线性——这些「跨特征」操作全部发生在**单通道内部**的 d_ch 维表征上，不跨 EEG 通道。4 层、残差、预 RMSNorm 与 v1 相同。**EEG 通道间在整个编码+骨干段零信息流动**。
- **[3] 逐通道 Norm**：RMSNorm(d_ch) 权重跨通道共享（与 [1][2] 一致的 CI 共享口径）。
- **[4] cMLP Head**：62 个子 MLP 参数独立（i=1..62）；第一层权重按源通道列分块 `W^{(i),1} = [W^{(i),1}_{:,1} | … | W^{(i),1}_{:,62}]`，块形状 (d_z, d_ch)；σ=SiLU（默认）；第二层线性。实现为堆叠张量 + einsum（W¹∈R^{62×d_z×992}，W²∈R^{62×d_z}），无逐通道 Python 循环。
- **初始化**：全部 PyTorch 默认（与 v1/v2 原则一致，不做恒等初始化等额外变量）。
- **EMA**：结构无关的权重平均，沿用 decay 0.999（类定义随 v4/model.py 复制，与 v1 逐字一致）。

### 2.3 与用户规格的逐条对照（登记）

| 用户规格条目 | 本设计落地 | 备注 |
|---|---|---|
| [x_{t,j}(1)+seg(1)+t_emb(16)]=18 维 → Linear(18→d_ch) | §2.1 [1] | d_ch 默认 16 |
| 4× CI Mamba，逐通道沿 L=512 scan | §2.1 [2] | 权重跨通道共享（§3 决策 1） |
| 截取 future 128 + 逐通道 Norm；汇聚 h_t∈R^{62·d_ch} | §2.1 [3] | RMSNorm 共享 |
| z_{t,i}=σ(Σ_j W^{(i),1}_{:,j}h_{t,j}+b¹)；v_θ=W²z+b² | §2.1 [4] | σ=SiLU、d_z=16 默认 |
| 「唯一跨通道混合层」 | ✓ 全网络仅在 [4] 第一层发生通道间信息流动 | CI 隔离性单测守护 |

## 3. 超参默认与参数量核算

### 3.1 决策默认值（全部入 config 并登记，评审/用户可改）

| 超参 | 定值 | 说明 |
|---|---|---|
| d_ch（config 字段沿用 `d_model`，见 §4.3） | **1（用户拍板 2026-10-16）** | 每通道独立表征宽度 = 1 个标量；AI 原默认 16 作废；d_state=16/d_conv=4/expand=2 沿用。**特例风险登记 §9.10：RMSNorm(1) 退化为 sign(x)·w、trunk 容量塌缩** |
| d_z（子 MLP 隐宽） | **16（用户拍板）** | head 输入 = 62·d_ch = 62 维 |
| σ | **SiLU（用户拍板「其余默认」）** | 与 Mamba 内激活一致 |
| CI 权重共享 | **共享（用户拍板）** | PatchTST 式 channel-independent |
| group lasso | **不加（用户拍板）** | 损失与 v1/v2 逐字一致；免费粗 readout |
| 其余一切（损失/t/EMA/优化器/seed/数据） | 与 v2 逐字段一致 | 唯一变化 = 速度网络结构 |

### 3.2 参数量（**d_ch=1, d_z=16**，用户拍板口径）

| 模块 | v1 | v2 | **v4-B (d_ch=1)** |
|---|---|---|---|
| 输入投影 | 1280 | 4960 | **19**（Linear(18→1)，共享） |
| trunk（4×MambaBlock(1)） | 13,504 | 123,504 | **484**（每块 121：RMSNorm(1)×1 + Mamba(1)×120；共享） |
| head 前逐通道 RMSNorm(1) | — | — | **1** |
| head（62 子 MLP，输入 62·1=62 维） | 1,054 | 3,968 | **63,550**（每子 MLP 1025 = W¹ 16·62 + b¹ 16 + W² 16 + b² 1；×62） |
| **总计** | ≈15.9k | ≈132k | **≈64.1k（64,056；0.48× v2；head 占 99.2%）** |

- head 单项：W¹ 62×(16·62)=61,504 + b¹ 992 + W² 992 + b² 62 = 63,550。
- 参数/样本比：v4-B ≈1.4:1 vs v2 ≈3:1（训练折 44,914 窗）——**过拟合风险低**（与 d_ch=16 方案的 22:1 相反）；主要风险反转为 **trunk 容量塌缩**（§9.10）。
- 训练侧监控仍登记 train/val gap（防反向问题：欠拟合）。

### 3.3 计算形态与资源预算（bench 硬门禁依据）

- scan 等效序列数 ×62：(B×62, 512, d_ch=1)，B=256 时 15,872 条 vs v2 的 256 条；d_model=1 的 kernel 效率不确定（§9.11）。
- 训练时长预期：v2 实测 8.78 s/epoch → v4-B(d_ch=1) **不可按 FLOPs 直觉估，bench 实测为准**；显存预期温和（激活 (B×62,512,1) 很小）。
- **评估侧显存/时长（评审补充，同样入 bench 门禁）**：sampler chunk × CI 形态单次前向峰值——bench 阶段跑 **eval 前向探针**（装载 v4 model 按 sampler 路径跑一个 batch 量峰值与耗时）；eval 侧预案 = `evaluate.py --batch-size` 降档（默认 64→32/16，登记口径）。
- OOM/过慢/欠拟合预案（均属口径变更，须用户确认）：训练侧 batch 256→128、d_z 降档、**d_ch 升档（1→8/16，欠拟合第一预案，§9.10）**；d_z/d_ch 变更须 config+model 双处同改并重跑单测（§4.3）。检查点：bench 摘要汇报时给出建议项。

## 4. 实现路径（v1 代码零改动 + config/model 双遮蔽）

### 4.1 v1 侧：零改动

A 方案的 v1 default-off 钩子随 A 弃用**作废**——`models/gpfm-mamba-v1/` 全部文件不动，v1/v2/v3 config_hash 逐位不变（回归 = 跑 v1 既有 7 单测，无需新增钩子单测）。

### 4.2 v4 目录

```
models/gpfm-mamba-v4/
├── config.py      # 完整 config 副本（v2 基础）：d_model=16（语义=逐通道宽度 d_ch，见 §4.3）
│                  #   + d_z=16 + head_act="silu" + gen_lowpass_hz=40.0 + variant_name="gpfm-mamba-v4b"
│                  #   + docstring（用户架构规格、别名「口头 v3/v4-B」、CI 共享口径）
├── model.py       # VelocityNetwork（v4-B 语义：CI 编码+骨干+cMLP head，docstring 注明）
│                  #   + time_embedding/EMA 复制（docstring 注明「复制自 v1 model.py，勿单独演化」）
│                  #   `from mamba import build_mamba_backbone`（复用 v1 mamba.py）
├── train.py       # v2 wrapper 扩展：sys.path 遮蔽 config 与 model 两模块
├── evaluate.py    #   （占据 sys.modules 后再 importlib 加载 v1 train/evaluate）
└── tests/         # 脚本式单测（§4.4）
```

- **双遮蔽机制**：v1 `train.py` 的 `from model import EMA, VelocityNetwork` 与 `sample.py::load_eval_model` 的同源 import，在 v4 wrapper 先占据 `sys.modules["model"]`（v4/model.py）后命中新结构——与 v2 遮蔽 config 同款手法，v1 代码零改动。
- run_id 前缀守卫 `v4b-`；checkpoint 主名 `gpfm-mamba-v4b-{best,last}.pt`（state_dict 键与 v1/v2 完全不同，config_hash 校验天然防串用）。

### 4.3 构造签名兼容与运行时契约（关键适配点，评审补全）

v1 train.py/sample.py 以显式 kwargs 构建：`VelocityNetwork(n_channels, past_len, future_len, d_model, n_layers, d_state, d_conv, expand, time_emb_dim, use_fast_path)`。v4 的类**保持同一签名**（全部接受），内部语义映射：`d_model` → 逐通道独立宽度 d_ch（config 注记「v4-B 中 d_model 即 d_ch」）；v4 特有参数 `d_z`/`head_act` 作为**类默认参数**（16/"silu"），与 config 字段同值。

**运行时契约四件套（冒烟即验证，缺一 ImportError/AttributeError）**：
1. 构造 kwargs 同 v1 签名（如上）；
2. `forward(x_past, x_f, t, return_all=False)` 输入输出形状与 v1 相同（(B,C,P)+(B,C,F)+(B,) → (B,C,F)）；
3. `num_params()` 方法（v1 train.py L230 调用）；
4. **符号导出**：模块必须按名导出 `VelocityNetwork` 与 `EMA`（v1 train.py L38 / sample.py L34 按名导入——类名沿用 `VelocityNetwork`，v4-B 语义写 docstring，不起别名新名）。

**d_z/d_ch 双处一致性防线（评审前移）**：d_z/d_ch 同时存在于 config 字段（进 hash）与 model 类默认两处，漂移时 state_dict 形状自洽、resume/eval 均不报警——三重防线：① 单测断言 cfg.d_z == model 构建默认（§4.4.6）；② **v4 wrapper 在加载 v1 train/evaluate 前运行时断言** `model 模块类默认 d_z/head_act == cfg 对应字段`，不一致即退出；③ 任何 d_z/d_ch 变更属口径变更（须用户确认），且变更流程明文要求 config.py 与 model.py 两处同改并重跑 §1 全部单测。

### 4.4 单测清单（tests/，脚本式零依赖）

1. **CI 隔离性**：扰动输入通道 j 的值，`h_{t,k}`（k≠j）逐位不变；v_θ 通道 k 输出可变（经 head，合法）。
2. **因果性**：扰动 future 位置输入，past 位置的 h 逐位不变（return_all 口径）。
3. **head 结构**：W¹ 形状 (62, d_z, 992)、按 (62, d_z, 62, d_ch) reshape 列组视图正确；62 个子 MLP 参数独立（梯度不串）。
4. **前向形状/梯度**：(B,62,128) 输出、backward 梯度有限、无 NaN。
5. **参数量断言**：num_params ≈ 1.00M（±1%，随默认值登记）。
6. **config/model 一致性**：cfg.d_z、cfg.head_act 与模型构建默认一致；config_hash 稳定。
7. **遮蔽与前缀守卫冒烟**：config/model 双遮蔽命中 v4、`v4b-` 前缀正反例。

## 5. 训练口径（与 v2 逐字段一致；差异仅结构相关三字段）

seed 1234 · epochs 100 · batch 256 · AdamW lr 1e-3 wd 1e-4 cosine · clip 0.5 · EMA 0.999 · t~logit-normal(−0.8,0.8) · 损失 {1,1,0.1,0.1} · best=内部验证总 loss（固定 t 网格确定性复算）· R9 全套监控/checkpoint/断点续传 · fold0 status_ok 46 例 · 划分固化 manifest 只读校验。

- **1-epoch 基准（硬门禁）**：`--epochs 1 --bench --run-id v4b-bench-1ep`，登记 peak 显存/墙钟/it/s/四项损失走势/t 分桶 sanity + 100-epoch 外推；**摘要（含 OOM/时长预案建议）汇报用户确认后**才启动全程。
- 主 Run：`nohup … --run-id v4b-main-01`（先 mkdir run 目录）；断点续传冒烟留痕。
- head 参数大 → 监控 train/val 总 loss 差（过拟合观察项，§9.2），metrics.jsonl 照登。

## 6. 评估口径（Run 017 同口径 + 诊断副产品）

命令序列与默认参数与 Run 017 逐项一致：

```bash
python3 models/gpfm-mamba-v4/train.py --epochs 1 --bench --run-id v4b-bench-1ep    # 门禁
python3 models/gpfm-mamba-v4/train.py --run-id v4b-main-01                          # 100-epoch
python3 models/gpfm-mamba-v4/evaluate.py --run-id v4b-main-01                       # 七维度+图
python3 models/gpfm-mamba-v1/fc_metric.py --run-id v4b-main-01                      # 指标 1 v2.1
```

七维度并列（对照 Run 017：v1 / v2）+ 三检验（指标 1 FC r≥0.95 / 指标 3 低频 δ–β ≤0.5 + 目视 / 指标 2 用户目视）+ R1 附分子分母 + std_t 幅度过冲照登（真实 0.7881，v1 +23%/v2 +29%）。

**诊断副产品（仅诊断，不参与主判据）**：训后从 best（EMA）权重算内部有向 readout 矩阵 **`M[j, i] = ‖W^{(i),1}_{:,j}‖_F`（(62,62)，行=源 j、列=靶 i——与全仓库方向约定一致（evaluation-calibers.md L29：EC 矩阵行=源、列=靶；gc_psd.py L256 同），评审勘正：此前草稿行/列相反会与外部 GC 呈转置比较）**，热图 + 与外部 GC 矩阵的相关（Spearman，作参考）——**无监督项参与训练**，损失与 v1/v2 逐字一致；group lasso 正则登记为条件后续。

## 7. 路径清单（path-isolation 契约）

| 类型 | 路径 | 说明 |
|---|---|---|
| 修改 | **无**（v1 零改动） | v1 既有 7 单测回归通过即可 |
| 新建（变体） | `models/gpfm-mamba-v4/` | config/model/train/evaluate + tests |
| 只读复用 | `models/gpfm-mamba-v1/`（mamba.py、data/gpr_prior/losses/sample/evaluate/gc_psd/fc_metric、splits manifest）、`common/visualizer.py` | 双遮蔽契约 |
| 只读数据 | `/root/autodl-tmp/edra_process_v4/` | 契约 `.trellis/spec/data/edra.md` |
| 新建输出 | `/root/npi/outputs/gpfm-mamba-f0/v4b-*/`（已拍板沿用 f0 根） | run_id 前缀隔离 |
| 登记 | `{TASK_DIR}/experiment.md` | Run 块四要素 + 产物互链 |

## 8. 决策点状态

| 决策 | 状态 | 值 |
|---|---|---|
| 变体方案 | **用户拍板（2026-10-16）：v4-B（用户自定架构），A 弃用** | 无低秩、保留通道身份、cMLP 顶层混合 |
| v4 编号/别名 | 已定 | 目录 v4；用户口头「v3」「v4-B」均登记别名 |
| CI 权重共享 | **已定（用户 2026-10-16）**：共享 | PatchTST 式 |
| d_ch（config 字段 d_model） | **已定（用户 2026-10-16）：1** | 特例风险 §9.10（RMSNorm 退化/容量塌缩）；欠拟合第一预案 = 升 d_ch |
| d_z / σ | **已定（用户 2026-10-16）**：16 / SiLU | — |
| group lasso | **已定（用户 2026-10-16）**：不加 | 免费 readout 诊断；正则为条件后续 |
| 训练/数据/评估口径 | 已定（继承 v2/Run 017） | §5/§6 |
| outputs 根 | **已拍板**：沿用 f0 根 + `v4b-` 前缀 | 例外登记 experiment.md |
| 启动 | **已批准（用户 2026-10-16）** | task.py start |
| 5-fold / θ* | 门禁保留 | 本任务不启动 |

## 9. 风险与缓解

1. **head 过参数过拟合**（1.0M vs 44,914 窗，22:1）：监控 train/val 总 loss 差与 best 判据走向；预案（须用户确认）：d_z 16→8、或加 weight decay 局部（口径变更登记）。
2. **CI trunk 容量疑虑**（每通道仅 16 宽独立编码）：v1 全局 16 宽已支撑 GC 0.276，逐通道 16 宽容量应更足；若 MAE 明显劣化，d_ch 备选 32（参数 +2k trunk，可忽略——head 输入维度变 62·d_ch 需同步）。
3. **训练时长/显存**（scan 批 ×62）：bench 硬门禁 + OOM 预案（batch/d_z/d_ch 三级，均口径变更须确认）；100 epoch 预估 2.5–6.5 h。
4. **归因非单变量**：v4-B vs v2 两处变化（CI trunk + cMLP head）；结论表述限定「结构组合效应」，分解变体登记条件后续——判读材料必附此声明。
5. **「同刻混合」质疑**：head point-wise 混合发生在同一时间步，滞后依赖经 h_{t,j} 的历史编码表达——判读材料按 §1 滞后语义对应表述，避免误写「零滞后混合 ⇒ 无法表达 GC」。
6. **单 seed 局限**：同 A 框架——预注册阈值 + 邻域加 seed 升级路径；结论标注「单次运行」。
7. **R1 误读**：按已登记解读规则报告（分子分母+对照）。
8. **双遮蔽脆弱性**：sys.modules 遮蔽对 import 顺序敏感——wrapper 内置遮蔽自检（解析路径断言），单测覆盖。
9. **EMA/最优判据交互**：无结构耦合（权重平均）；τ≈10³ 步口径不变，无需动作。
10. **d_ch=1 特例风险（用户拍板口径，2026-10-16 登记）**：① **RMSNorm(1) 数学退化**——对 1 维向量归一化 ≈ sign(x)·w，块预归一化把幅度信息二值化，幅度仅经残差路径保留；② **trunk 表征容量塌缩**——每通道每时间步仅 1 个标量 h_{t,j}，跨通道时序建模负担几乎全部移到 head；SSM 状态虽 16 维但输出投影回 1 维。**判读竞争解释预登记**：若 GC 不回升或五维度劣化，「trunk 太弱（d_ch=1）」列为与「结构假设不成立」并列的竞争解释，bench 损失走势（对照 v2 同期）为第一证据；欠拟合（train loss 高原远差于 v2）⇒ 第一怀疑 d_ch，门禁汇报建议升 d_ch 重跑（口径变更流程）。
11. **d_model=1 的 CUDA kernel 效率**：mamba_ssm 在极窄宽度下 kernel 利用率低，时长预估不确定度大——bench 实测为准（可能快于也可能慢于按 FLOPs 的直觉）。

## 10. 回滚

- v4-B = 删 `models/gpfm-mamba-v4/` 与 `outputs/gpfm-mamba-f0/v4b-*/`。
- v1 零改动 ⇒ 无回滚项；任一阶段失败直接回退 commit。
- 里程碑提交：v4 目录+单测 / 冒烟+bench / 主 Run / 评估判读。
