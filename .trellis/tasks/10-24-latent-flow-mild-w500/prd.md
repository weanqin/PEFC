# latent-flow × Mild Depression 前缀窗 500（375/125）口径训练 fold0

## Goal

按用户 2026-10-24 拍板口径在 mild-depression 数据集上重训 latent-flow-v1（OC-LatentFlow 两阶段）：
每 trial **前 500 时间点单窗**（obs x_p 375 / forecast x_f 125，无滑窗无多窗）、**仅 fold 0**、
batch 64、lr 5e-4、**stage2 decoder 与 flow 联合训练（仅冻结 encoder）**、graph 流 anti-uniform
λ=0.1、**归一化改为训练折 trial 前 500 点统计**、**采样确定性不变**；训练后按
evaluation-calibers 口径做生成质量数值抽查（coverage 照登并注明确定性采样的结构性 0）。

## 背景与动机

- 上一轮（任务 10-21-latent-flow-mild-train，归档）三窗口口径（3:1/5:1/9:1 @1500 点窗）训练
  健康收敛但生成质量阴性：逐点 Q05–Q95 coverage 一律 0.00%、medoid/real std 主体 0.5–0.74、
  末端 Euler 发散；定论「结构性缺陷」，不指定 θ 候选。
- 本轮主会话方案审阅已向用户披露：coverage=0 有相当成分系 `sample()` 全确定性（z_obs=μ +
  确定性 Euler + 确定性 decoder，K 条"样本"逐位相同）的结构性结果；用户拍板**沿用确定性采样**，
  质量判读以 medoid/real std 比 + 波形人工判为主，coverage 照登并注明结构性。
- 新口径动机（方案审阅通过）：500 点窗 = 包内 MODMA 原生口径（obs 375/fc 125，23.4 样本/潜步），
  恢复与旧线时间压缩比可比性；forecast 段（1.5–2.0 s）远离数据尾段 ~5.8 s 幅度衰减区，
  「末端 Euler 发散 × 尾段衰减叠加」伪影不再适用。
- 用户另拍板三项训练侧偏离（相对上轮）：batch 64、lr 5e-4、stage2 decoder 联合训练 +
  anti-uniform 0.1——属对负证据链（生成幅度不足/分布塌缩）的针对性迭代实验。

## Requirements

- **R1 数据口径**：1200 trial 全集描述性形状 [1200,128,1500]；每 trial 取前 500 点（0–499，
  完整落在数据文件 x_p 字段内部，x_f 字段不参与），窗内切 x_p_new=[:375]、x_f_new=[375:500]；
  单窗口、无滑窗、无多窗采样；划分只读复用钉死 manifest `splits/mild_5fold_seed42.json`
  （SHA256 契约不变），本轮仅 fold 0（train 28 例/840 trial、val 4/120、test 8/240）。
- **R2 训练配置（用户拍板，全部为硬要求）**：
  - batch size 64；
  - lr 5e-4（Adam，常值，两阶段同 lr）；
  - stage2 图流损失启用 anti-uniform 项，权重 0.1；
  - stage2 decoder 与 flow 一起训练，仅冻结 encoder（proj 不在 stage2 计算图内、无梯度，
    实施面随 encoder 一并冻结，行为无差）；
  - 沿用采样确定性（推断链路零随机性，不开放后验采样）；
  - 归一化 = 训练折 trial **前 500 点**逐通道 mean/std（wrapper 内 float64 累加、严格
    held-out、std==0→1.0 护栏不变）。
- **R3 沿用项（方案审阅默认，用户未否决）**：epochs 150（两阶段）、早停 patience 30、
  seed 42、β_KL 1e-4（KL 线性退火前 20% epochs，仅 stage1）、mexhat-graph 流（K=2、γ_W=0.1、
  γ_M=1.0）、全窗 decoder（use_fc_decoder=False）、AMP 开启 + max_grad_norm 1.0、
  Euler 10 步推断；模型架构零改动（OCLatentFlow 原样，stage2 构建 1,243,234 总参数；
  decoder 联合训练后可训练面 = flow 488,897 + decoder 部分）。
- **R4 训练后质量抽查**：按 `.trellis/spec/research/evaluation-calibers.md` 数值抽查口径
  （test 3 例 × 30 trial × K=32 seed 42：逐点 Q05–Q95 coverage + medoid/real std 比，全程与
  去末端双口径——去末端口径为**定点沿用登记的 25 点**（非按 fc 长度缩放；占 fc 窗比例由
  上轮 6.7% 变为 20%，数值与上轮不可比已由窗口变更覆盖））+ viz v2.1 波形渲染；
  **数值抽查代码须持久化为变体层脚本**（上轮为内联脚本无落盘，本轮修正）。
- **R5 工程纪律**：包体 `latent_flow/` 零改动（全部改动落变体层 `models/latent-flow-v1/`
  wrapper）；旧口径默认行为 bit 兼容（新 CLI 参数默认值 = 旧行为）；产物落本轮声明的新输出
  目录，不覆盖上轮 Run 001–006 任何存档；experiment.md 每 Run 块跑前登记目的+配置、跑后回填
  结果+结论。

## Acceptance Criteria

- [ ] **AC1 口径参数化落地且默认兼容**：`train_mild_depress.py` 支持 `--window-len`（默认
      1500=旧行为不变）、`--anti-uniform-lambda`（默认 0）、`--unfreeze-decoder`（默认关）；
      新口径 = `--window-len 500 --obs-len 375 --forecast-len 125` 全链路断言（obs+fc==window、
      window≤1500、manifest 全量断言重放、计数 840/120/240）通过。
- [ ] **AC2 stage1 训练完整**：fold0 stage1（VAE）150 epochs 上限内收敛，best ckpt +
      history + config json 落盘于本轮新输出目录。
- [ ] **AC3 stage2 冻结面正确**：stage2 打印 trainable 参数面 = decoder + flow（encoder/proj
      冻结），anti-uniform λ=0.1 生效（loss 字典含 uni_reg）；训练完整收敛，产物落盘齐全。
- [ ] **AC4 质量抽查数字落盘**：数值抽查脚本产出 coverage（双口径，注明确定性采样结构性）
      + medoid/real std 比（双口径）+ viz v2.1 PNG + meta，数字回填 experiment.md。
- [ ] **AC5 路径隔离**：全部产物在 `outputs/10-24-latent-flow-mild-w500/` 声明目录内；
      `outputs/10-21-latent-flow-mild-train/` 零触碰；包体 `latent_flow/` 零 diff。
- [ ] **AC6 复现凭据**：config json 记录 window/归一化范围/冻结面/anti-uniform/全部超参 +
      manifest SHA256 + norm stats hash + git head；同 seed 复跑产物可追溯。

## 边界（本轮不做）

- 不做代理脑合格性三项检验（surrogate-brain-qualification.md 门禁另立任务；本任务产物不继承资格）。
- 不做 5-fold 全量训练（fold0 单 seed 前置验证；扩 5 折须用户另行同意——代理脑规范训练顺序门禁）。
- 不改采样确定性、不动模型架构与包体、不做 θ 候选指定（除非质量抽查反转上轮阴性，仍须用户判读拍板）。
- 数据契约 `spec/data/mild-depression.md` 的窗口消费口径变更在任务收尾时回填登记（口径变更告知义务）。

## 子研究路由（1.0b / 1.0c）

- **分类结果**：`study1` —— EEG 代理脑（Study I）条件生成模型训练任务：latent-flow-v1 ×
  Mild Depression 训练与生成质量抽查，无 EC 推断成分。
- **依据**：任务核心交付 = 代理生成模型训练 + 生成质量评估（θ 候选产线），属 Study I
  「模型选型与训练、代理脑有效性」层次。
- **新路径/方法检测（1.0c）**：`已有路径：latent-flow（OC-LatentFlow，models/latent-flow-v1/）`
  —— 技术路线表已登记该路径及其 Mild Depression 线条目（含 10-21 负结论）；本轮为同路径
  新数据口径 + 训练侧参数迭代，无新方法引入，无需适配研究。
- **research 产物**：`无`（已有路径；上轮负证据链与口径文档即本任务直接输入，
  见 design.md 引用清单）。
