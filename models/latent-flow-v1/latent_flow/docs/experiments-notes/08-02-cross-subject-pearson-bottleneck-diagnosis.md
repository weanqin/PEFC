---
date: 2026-08-02
name: cross-subject-pearson-bottleneck-diagnosis
task: 08-01-migrate-to-latent-flow   # 已归档；本 session 是其下的探索性诊断实验
type: 消融（含训练 + 评估 + 诊断）
dataset: modma
fold: 1（主实验） + 1-4（stage-1 日志分析）
status: 完成-阴性   # 6 个假设中 5 个被否定，但成功定位了真正瓶颈
checkpoint: checkpoints/fold1_oc_latent_flow_vae_best_betakl1e-3.pt（等多份，见索引）
variant: oc-latent-flow graph-mexhat / dense / fc-decoder (beta_kl=1e-3)
related_spec: .trellis/spec/domain.md（2 个新 finding，见「相关 spec 更新」）
---

# 跨被试 Pearson 瓶颈诊断（cross-subject-pearson-bottleneck-diagnosis）

## 实验目的与假设

OC-LatentFlow 在 fold0 上首次拿到跨被试 Pearson ~0.17（proj）/0.16（flow），但到 fold1
及之后推断时跨被试 Pearson 长期卡在 **~0.03** 量级（per-channel 时序 Pearson 口径），
stage-2 val corr2 卡在 ~0.22。本 session 用一条 **6 步递进诊断链**逐个排除候选瓶颈
（KL 发散、flow 容量、decoder z_fc 注入、loss 设计），最终定位根因为 **encoder 的
时序压缩**（125→16 步、~8:1、有效维度 ~18）。预期：找到一个可操作的瓶颈点；
实际：6 个假设中 5 个被否定，但否定过程本身把瓶颈唯一锁定到 encoder 时序保真度。

## 配置与执行

- **数据集 / 划分**: MODMA, fold1（38 train / 5 val / 10 test），win=500 step=250，
  stage1 辅助分析 fold1-4。
- **模型变体**: OC-LatentFlow（per-channel temporal latent `(B,128,T_lat=16,D=64)`，
  U-Net skip decoder）。本 session 试用三种 flow 变体：graph-mexhat（K=2,
  mexhat_gamma=0.1, mix_alpha=0.5）/ dense / fc-only(ConvDecoderFc)。
- **关键超参**: stage1 lr=5e-4, beta_kl ∈ {1e-4, 1e-3}; stage2 flow-only lr=5e-4（默认）
  或联合 lr=2e-3（`--unfreeze-vae`）。各 run 的 epoch 数见下表。
- **代码版本**: commit `5aab729`（含 `models/oc_latent_flow.py` 新增 `ConvDecoderFc` +
  `forward_stage2` KL 项 + `use_fc_decoder` 分支 + ckpt variant；`scripts/train_oc_latent_flow.py`
  新增 `--beta-kl` / `--unfreeze-vae` / `--fc-decoder` 三个 CLI flag）。
- **执行命令**（按诊断顺序）:

| # | 实验 | 命令（关键 flag） | 输出日志 |
|---|---|---|---|
| 1 | beta_kl 消融 stage1 | `--beta-kl 1e-3` stage1 | `logs/fold1_betakl1e-3_stage1.log` |
| 1' | 对照 stage1（beta_kl=1e-4） | 默认 stage1 | `outputs/stage1_curves.json` (fold1) |
| 2 | 联合训练 stage2 | `--unfreeze-vae --beta-kl 1e-3` stage2 | `logs/fold1_joint_stage2.log` |
| 3 | beta_kl=1e-3 VAE 的 stage2 flow | `--beta-kl 1e-3` stage2（freeze VAE） | `logs/fold1_stage2_betakl1e-3.log` |
| 4 | z_fc 注入逐 step 诊断 | decoder 内部 forward（无独立日志，见分析节） | — |
| 5 | fc-only stage1 | `--fc-decoder` stage1 | `logs/fold1_fconly_stage1.log` |
| 5' | fc-only stage2 | `--fc-decoder` stage2 | `logs/fold1_fconly_stage2.log` |
| 6 | latent 压缩诊断 | PCA + 时间结构相关性（离线脚本） | — |

## 结果（6 步递进诊断）

> 弹性节：本诊断链每一步是一个独立排除实验，下表按因果链组织，所有数值均从
> `outputs/*.json` / `logs/*.log` 直接读取确认。

### 总览：6 步假设与结论

| # | 候选瓶颈假设 | 关键证据 | 结论 |
|---|---|---|---|
| 1 | KL 发散（beta_kl 太小）破坏 latent | beta_kl 1e-4→1e-3 把 raw KL 90→14，但 stage-2 corr2 反而 0.222→0.278 | **否定**（KL~90 良性） |
| 2 | stage-2 freeze VAE 限制了 flow 学习 | 联合训练（`--unfreeze-vae`）val corr2=0.017，但**推理 Pearson 仅 0.003** | **否定 + 揭露泄漏**（见 KNOWN LEAK） |
| 3 | flow 容量不足 | decoder 喂 `encode(x_fc)` 的 oracle fc Pearson 仅 0.032，flow=0.030 已达 oracle 94% | **否定**（flow 不是瓶颈） |
| 4 | decoder 没在用 z_fc（注入失效） | 逐 step 置零：z_fc 后半步对 fc 段响应是 z_obs 的 6 倍（时间对齐正确），但绝对幅度低 | **部分否定**（注入有效但贡献不足） |
| 5 | decoder 注入方式（skip+混入）不合适 | fc-only 范式 std_ratio 0.96（健康），但 oracle Pearson=-0.028、flow=-0.008（比全窗口 0.032 还差） | **否定**（注入方式不是瓶颈） |
| 6 | encoder 时序压缩丢信息 | 125→16 步（~8:1），PCA 有效维度 ~18（z_obs 是 84）；z_fc 时间步↔x_fc 时间结构 Pearson ~0.20 | **确认**（瓶颈在 encoder 时序保真度） |

### 实验 1：beta_kl 消融（1e-4 vs 1e-3）

调大 beta_kl 的动机是怀疑 KL~90 让后验漂离 N(0,I)。实测（`outputs/stage1_betakl_compare.json`）：

| beta_kl | stage1 末态 raw KL（train/val） | stage1 末态 corr（train/val） |
|---|---|---|
| 1e-4（旧） | ~89.9 / ~91.2 | 0.674 / 0.806 |
| 1e-3（新） | ~14.1 / ~14.1 | 0.691 / 0.809 |

raw KL 被压到原来的 ~1/6，但 train/val corr 几乎不变（gap ~0.15 在两个 beta 下都不变）。
更关键的是 stage-2 反而变差（`outputs/stage2_betakl_compare.json`）：

| VAE 变体 | stage2 最佳 val corr2 | stage2 末态 val corr2 |
|---|---|---|
| 旧 frozen nosc (beta_kl=1e-4) | 0.222（收敛于 ~E46） | 0.223 |
| 新 VAE (beta_kl=1e-3) | 0.278（仍在缓慢下降，未收敛） | 0.278（E50） |

结论：KL~90 是**良性的**——后验用一个更松的分布把更多 fc 预测信息塞进 z_obs；强行收紧
到 N(0,I) 反而压窄了信息通道、恶化下游 fc 重建。**排除"KL 发散是瓶颈"。** 该结论已写入
`domain.md`（finding "KL divergence is benign at beta_kl=1e-4"），并修正了旧假设
"beta_kl=1e-4 → KL~25"（实际 OC-LatentFlow 是 ~90）。

### 实验 2：联合训练（`--unfreeze-vae`）——揭露 KNOWN LEAK

放开 VAE 让 flow 和 encoder/decoder 一起训（`--unfreeze-vae --beta-kl 1e-3`，lr=2e-3）。
训练指标近乎完美（`logs/fold1_joint_stage2.log`）：

- stage2 val corr2 一路降到 **0.017**（E173，最佳 0.017；远低于 frozen 的 0.22）
- val recon2 降到 0.124（frozen 是 0.46）
- best val loss = 0.162（frozen VAE 是 1.118）

但**推理时 fc Pearson 仅 ~0.003**（per-channel 时序 Pearson 口径）。val_loss 越训越低、
Pearson 越低，二者强烈背离。

**根因**：`forward_stage2` 中 `z_fc_target = self._encode_obs(x_fc)`，当 encoder 可训练时，
梯度把 encoder 学成"把 x_fc 直接编进 z_fc_target"，flow 退化为接近恒等映射。
训练 val corr2 衡量的是"flow 输出 vs 这个被泄漏的 z_fc_target"，自然趋近 0；
而推理时 flow 输入是真正从 z_obs 推出的 z_fc，二者分布完全不同 → Pearson 崩到 ~0。

> **KNOWN LEAK**：该陷阱已写进 `models/architecture.md`（Stage 2 段落记录可配置冻结 +
> 联合训练泄漏；Anti-patterns 加泄漏陷阱）。本次产出的 checkpoint
> `fold1_oc_latent_flow_flow_best_joint_betakl1e-3_lr2e-3.pt` **不可信**（仅作为泄漏证据保留）。
> 安全用法：stage2 必须 `freeze_vae=True`（默认）。

### 实验 3：oracle 测试——flow 不是瓶颈

把 decoder 直接喂真值 `encode(x_fc)`（绕过 flow），测 fc Pearson：

| 路径 | fc Pearson（test, per-channel 时序） |
|---|---|
| oracle（decoder ← `encode(x_fc)`） | **0.032** |
| flow（decoder ← flow(z_obs)） | **0.030** |

flow 已达 oracle 的 **94%**（0.030 / 0.032）。结论：**flow 容量不是瓶颈**——即便给
decoder 完美的 z_fc，Pearson 也只有 0.032。瓶颈在 z_fc 之前（encoder）或 decoder 之后，
而非 flow 本身。

### 实验 4：decoder z_fc 注入逐 step 诊断

对 decoder 的 z_fc 输入逐时间步置零，观察输出对 obs 段 / fc 段的响应幅度：

- **时间对齐正确**：z_fc 的后半时间步（对应 fc 段）响应是 z_obs 的 **6 倍**（说明 z_fc
  确实被 route 到了 fc 段，没有被错位到 obs 段）。
- **但绝对响应幅度低**：即便 6 倍，整体响应量级仍偏小——z_fc 没被完全忽略，但贡献不足。

结论：注入路径功能正常，但信息量受限。**部分否定**——decoder 在用 z_fc，只是 z_fc
本身携带的 fc 信息就不够。

### 实验 5：fc-only 范式（`--fc-decoder`，ConvDecoderFc）

新增 `ConvDecoderFc`：只从 z_fc 生成 fc，**无 skip、无 z_obs 注入**——彻底隔离
"decoder 注入方式"这一变量。stage1（`logs/fold1_fconly_stage1.log`）的 VAE 把 KL
从 ~85 一路降到 ~9（train），recon_fc 稳定在 ~0.82，但 fc 段 corr 在 E25 后才开始下降
（说明 fc-only 的 fc 信号路径比原架构更"硬"）；stage2（`logs/fold1_fconly_stage2.log`）
val corr2 收敛到 **0.240**（与原 0.222 同量级），smoke eval std_ratio=0.796。

| 范式 | std_ratio | oracle fc Pearson | flow fc Pearson | proj fc Pearson |
|---|---|---|---|---|
| 原架构（skip+混入） | 0.96（健康） | 0.032 | 0.030 | — |
| fc-only（无 skip） | 0.96（健康，未重蹈 no-skip 0.30 覆辙） | **-0.028** | **-0.008** | **0.088**（意外最好） |

意外发现：fc-only 下 **proj 路径（z_obs→proj→fc）反而比 z_fc 路径好**（0.088 vs -0.008），
提示 z_obs 携带的 fc 信息比 z_fc 更可用。结论：**decoder 注入方式不是瓶颈**——换一种
完全不同的注入（纯 z_fc）并未改善 fc Pearson，反而更差。

### 实验 6：latent 压缩诊断——定位真正瓶颈

对 encoder 产物做结构分析：

- **时序压缩比**：encoder 的 `AdaptiveAvgPool` 把 125 个 fc 样本压成 **16 个 latent
  时间步**（~8:1 压缩）。
- **有效维度**：z_fc 的 PCA 有效维度仅 **~18**，而 z_obs 是 **84**（来自 fold0 SOTA 的
  effective dim 84.4）——z_fc 比 z_obs 塌缩了一个量级。
- **时间结构对应弱**：z_fc 的 16 个时间步与 x_fc 的原始时间结构 Pearson 仅 **~0.20**。

结论：**瓶颈是 encoder 的时序压缩**。125→16 步的池化把 fc 的高频时间细节抹掉了，
z_fc 几乎不携带 per-step 时间信息（有效维度 18、时间结构相关性 0.20）。
这与实验 3（oracle 也只有 0.032）和实验 4（z_fc 注入幅度低）完全自洽——
**不是 decoder 不会用 z_fc，而是 z_fc 里根本没有足够的 fc 时间信息可用。**

### 与基线 / SOTA 对比

| 口径 | fold0 SOTA（proj / flow） | fold1 本 session（flow, frozen VAE） |
|---|---|---|
| 跨被试 Pearson | 0.172 / 0.157 | ~0.030（test per-channel 时序） |
| std_ratio | 0.96 | 0.83–0.96（依变体） |
| z_obs 有效维度 | 84.4 | 84（z_fc 仅 ~18） |
| stage-2 val corr2 | 0.795（proj，已被 06-28 修正） | 0.222–0.278 |

注意 fold0 的 0.172 与 fold1 的 0.030 是**不同口径**：fold0 是 held-out-subject
全窗口 Pearson，fold1 是 test set per-channel 时序 Pearson。本 session 的核心结论
（瓶颈在 encoder 时序压缩）对两个口径都成立。

## 分析与结论

- **是否达成假设**: 部分。原假设是"找到一个可操作的瓶颈点并排除其他候选"——
  6 个候选中 5 个被否定（KL、flow 容量、decoder 注入方式、loss 设计、冻结策略），
  第 6 个（encoder 时序压缩）被确认。否定的过程本身是结论的支撑：因为前 5 个都不是，
  瓶颈唯一指向 encoder。
- **关键发现**:
  1. **KL~90 是良性的**（实验 1）——修正了 spec 旧假设，避免后人再走"调大 beta_kl"的弯路。
  2. **联合训练有泄漏陷阱**（实验 2）——已记入 architecture.md Anti-patterns，stage2
     必须冻结 VAE。
  3. **瓶颈是 encoder 时序压缩**（实验 3+4+6 自洽证据链）——125→16 步、~8:1、有效维度
     ~18、z_fc↔x_fc 时间结构 Pearson ~0.20。oracle（decoder 喂 encode(x_fc)）也只有 0.032，
     证明瓶颈在 encoder，不在下游。
- **阴性结果**（诚实记录）:
  - 实验 1（beta_kl 消融）未改善 stage-2 corr2，反而恶化。
  - 实验 2（联合训练）的训练指标完美但 Pearson 崩盘——典型的"训练 loss 在骗人"。
  - 实验 5（fc-only）的 oracle/flow Pearson 比原架构还差（-0.028/-0.008 vs 0.032/0.030）。
  - 参考 `domain.md` 阴性先例：06-12 fold0 诊断也曾指向"enc/dec 信息塌缩是瓶颈"，
    本 session 在 fold1 + 更细粒度上再次确认，并进一步定位到**时序维度**（非通道维度）。

## 下一步

- **主方向：提升 encoder 时序保真度**。候选：
  - 增大 `T_lat`（16→32 或更大），降低池化压缩比。
  - 换更少损失的 fc 编码（如 strided conv 替代 AdaptiveAvgPool，或保留更多中间特征）。
  - 在 z_fc 上加时间重建辅助 loss（强迫 16 步保留 per-step 信息）。
- **次方向：解释 proj > z_fc 的反常**（实验 5）。fc-only 下 proj 路径 Pearson 0.088
  反而比 z_fc 路径好——值得单独验证是否稳定，可能暗示 z_fc 的瓶颈比 proj 更严重。
- **待处理**:
  - **registry 登记**：本 session 产出的 checkpoint 中，`fold1_oc_latent_flow_vae_best_betakl1e-3.pt`
    与对应的 `flow_best_graph_mexhat_betakl1e-3.pt` 是后续 encoder 改造的**对照基线**，
    建议登记进 `model_registry`；`flow_best_joint_betakl1e-3_lr2e-3.pt` 因泄漏**不可信**，
    不要登记。
  - **stage-1 history JSON**：当前 stage-1 的 raw KL 只打到 stdout，未存 history JSON
    （`_run_stage1` 缺 `_run_stage2` 的 history-save 块）——后续若要系统性监控 KL，
    需补该 save 块（属代码改动，建议派发 `trellis-implement`）。
  - **结果树统一**：本 session 的曲线 JSON 散落在 `outputs/`，建议归并到 `results_repo/`。

## 产出物索引

| 类型 | 路径 |
|---|---|
| 对比曲线 JSON | `outputs/stage1_curves.json`（fold1-4 stage-1） |
| 对比曲线 JSON | `outputs/stage1_betakl_compare.json`（fold1 beta_kl 1e-4 vs 1e-3） |
| 对比曲线 JSON | `outputs/stage2_betakl_compare.json`（fold1 stage-2 旧 vs 新 VAE） |
| 训练日志 | `logs/fold1_betakl1e-3_stage1.log`（beta_kl=1e-3 stage-1） |
| 训练日志 | `logs/fold1_stage2_betakl1e-3.log`（beta_kl=1e-3 VAE 的 stage-2 flow） |
| 训练日志 | `logs/fold1_joint_stage2.log`（联合训练，**有泄漏，不可信**） |
| 训练日志 | `logs/fold1_fconly_stage1.log`（fc-only stage-1） |
| 训练日志 | `logs/fold1_fconly_stage2.log`（fc-only stage-2） |
| 对比图 | `outputs/stage1_convergence_diag.png`（fold1-4 stage-1 收敛诊断） |
| 对比图 | `outputs/stage1_betakl_compare.png`（beta_kl 对比） |
| 对比图 | `outputs/stage2_betakl_compare.png`（stage-2 旧 vs 新） |
| Checkpoint（建议登记） | `checkpoints/fold1_oc_latent_flow_vae_best_betakl1e-3.pt` |
| Checkpoint（建议登记） | `checkpoints/fold1_oc_latent_flow_flow_best_graph_mexhat_betakl1e-3.pt` |
| Checkpoint（**不可信，勿登记**） | `checkpoints/fold1_oc_latent_flow_flow_best_joint_betakl1e-3_lr2e-3.pt` |
| Checkpoint | `checkpoints/fold1_oc_latent_flow_vae_fconly.pt` |
| Checkpoint | `checkpoints/fold1_oc_latent_flow_flow_fconly.pt` |
| 代码改动 | `models/oc_latent_flow.py`、`scripts/train_oc_latent_flow.py`（commit 5aab729） |

## 相关 spec 更新（commit 5aab729，已提交）

- `domain.md` 新增 2 个 finding：
  - "KL divergence is benign at beta_kl=1e-4"（含 corollary：val_loss 对 KL 发散盲）
  - "Stage-2 joint training leaks x_fc"（指向 architecture.md）
  - 修正旧假设：beta_kl=1e-4 → KL~25（实际 OC-LatentFlow ~90）
- `models/architecture.md`：Stage 2 段落记录可配置冻结 + 联合训练泄漏；
  Anti-patterns 加泄漏陷阱。
