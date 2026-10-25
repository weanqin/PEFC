# latent-flow 出口 GNN 变体训练：W 强制对称化 + anti-uniform 0.2（exit-gnn 衍生）

## Goal

基于任务 10-24-latent-flow-exit-gnn 正式 Run 001 配置派生轻量训练变体，两项配置变更：

1. **anti-uniform 权重系数 0.1 → 0.2**（`--anti-uniform-lambda 0.2`）；
2. **W 强制对称化**：训练每一步 optimizer step 后将权重矩阵投影到对称子空间
   `W ← (W + Wᵀ) / 2`，确保 W（从而 A_eff）严格对称。

其余超参数全部不变。本轮先交付本方案待用户确认；**用户确认后才进入代码实施与训练**。

## 背景与动机

- 基座 exit-gnn Run 001（best val 0.8241@E116，150ep 预算内真收敛）但幅度校准回归
  （medoid/real std 全程 0.645 / 去末 0.581，较 w500 的 0.957/0.996 回落，"出口再平均"
  机制登记在案）。
- 现行 `A_eff = γ_M · M(d) + γ_W · W`（`LearnableMexHatMixer.effective_adjacency`，
  oc_latent_flow.py:426-441）：M 为对称距离核（ricker/DoG），W 为可学习残差且
  **刻意不做对称化**（oc_latent_flow.py:417 注释明言 directed）——即 A_eff 的全部
  有向自由度都在 W 上。
- 用户变体假设：提高 anti-uniform λ 至 0.2 + 每步强制 W 对称，检验对 fc 生成质量
  （幅度校准 / 收敛性）与连接结构的作用；W 对称 ⇒ A_eff 全对称 ⇒ 消息传递退化为
  无向图混合（该变体的有意设计，非副作用）。
- **科学含义登记**：对称 A_eff 下 EC 读出的方向性信息按构造受限（对 Study II 的含义
  属训后判读/拍板项，不在本任务展开）。

## Requirements

- **R1 配置派生基座**：exit-gnn Run 001 全配置原样（w500 口径 `--window-len 500
  --obs-len 375 --forecast-len 125`、`--batch-size 64 --lr 1e-3 --epochs 150
  --patience 30 --seed 42`、`--unfreeze-decoder`、`--flow-exit-gnn`、VAE 只读复用
  w500 `run_w500_fold0/checkpoints/fold0_mild_vae_best.pt`、fold0）。仅两项变更：
  `--anti-uniform-lambda 0.2`（原 0.1）+ 新旗标 `--sym-w`。
- **R2 强制对称化语义（用户指定字面语义）**：每个 optimizer step 后投影
  `W ← (W + Wᵀ)/2`。实施形态 = **变体层 Adam 子类 `SymWAdam`**（`step()` 内
  `super().step()` 后对受管参数原位投影）——`BaseTrainer` 的 optimizer 为外部注入
  （trainer.py:180-181），逐步投影经 optimizer 子类实现即无需复制训练循环；
  包体 `latent_flow/` 零改动（延续 exit-gnn R4 纪律）。
- **R3 W 定位（唯一投影目标）**：`model.flow.graph_mixer.W`（128×128 nn.Parameter，
  oc_latent_flow.py:414-424）。exit-gnn 变体下入口/出口共享单份 A_eff ⇒ 恰一个 W；
  gaussian 分支（`GraphChannelMixer`）与 dense 分支无该参数 ⇒ 与 `--dense-fallback`、
  gaussian kernel 互斥拒跑。
- **R4 wrapper 接线（train_mild_depress.py）**：新增 `--sym-w` 旗标（默认关 = 现行
  行为 bit 兼容）；守卫同 exit-gnn 惯例：仅 `--stage 2` + mexhat/dog 流 +
  `--unfreeze-decoder`（本地 runner `_run_stage2_mild` 路径——包内 `_run_stage2`
  不可注入自定义 optimizer）；口径守卫将 `sym_w` 纳入新口径特征（须显式 `--run-dir`）；
  resolved config 凭据新增 `sym_w` 字段 + `flow_variant` 标签 `...-exit-gnn-symw` +
  `anti_uniform_lambda: 0.2`。
- **R5 对称性凭据**：stage2 flow 键重初始化 ⇒ W 零初始化（对称）起步，每步投影维持
  严格对称；①runner 起跑断言 W 对称（fail-fast）；②训后 ckpt W 对称断言（验收项）；
  ③smoke 与正式 Run 均执行②。
- **R6 评估链路零改动**：架构与 `FlowVelocityNetExitGNN` 完全一致（对称化仅训练期
  参数投影，不改前向、不改 state_dict 键面）⇒ `quality_check_mild.py` / `viz_mild.py`
  以现行 `--flow-exit-gnn` 旗标 strict load 即可，零接线改动。

## Acceptance Criteria

- [ ] **AC1** 方案经用户确认后实施；变体层新文件 + wrapper 旗标接线，包体 `latent_flow/`
      零 diff；`--sym-w` 默认关闭时 wrapper 行为与现行 bit 兼容。
- [ ] **AC2** 新变体文件 CPU 直跑自验：随机非对称 W 经一步 SymWAdam 后严格对称；
      `effective_adjacency()` 对称；参数量与基线逐位一致（投影不增参数）。
- [ ] **AC3** 冒烟 Run（2 epoch / --limit-subjects 2）：旗标链路通、uni_reg 以 λ=0.2
      生效、训后 W 对称断言通过、config 凭据 `sym_w: true`、口径守卫拒跑路径在档。
- [ ] **AC4** 正式 Run（fold0 全量）：训练完成（早停或预算顶格）、产物落
      `outputs/10-25-latent-flow-exit-gnn-symw/run_symw_fold0/{configs,checkpoints,logs}`、
      训后 ckpt W 对称断言通过、experiment.md 登记（与 exit-gnn Run 001 同口径对照：
      val 曲线/best/smoke）。
- [ ] **AC5** 质量抽查 + viz（同 w500 Run 003 / exit-gnn Run 002 口径：--k 32 --seed 42
      --subjects 1,7,21，全程 125 / 去末 25 双口径 medoid/real std + coverage 照登），
      结果登记 experiment.md；θ 候选判断留用户拍板。

## 边界（本任务不做）

- 不改 `FlowVelocityNetExitGNN` 架构、不改损失定义、不动包体、不改采样确定性。
- 不做 5-fold、不延长预算、不做 lr 消融（exit-gnn 遗留两因素归因另属拍板项）。
- 不在本任务内判定 θ 候选资格（质量数值 + 波形人工判读属用户）。

## 子研究路由（1.0b / 1.0c）

- **分类结果**：`study1` —— EEG 代理脑（Study I）模型正则/训练变体（θ 候选产线迭代）。
- **依据**：交付物 = 代理生成模型训练变体（exit-gnn 基座 + 正则/优化器级 knob），无
  EC 推断成分。
- **新路径/方法检测（1.0c）**：`已有路径：latent-flow（models/latent-flow-v1/）`——
  注册表既有路径内的正则/优化器级变体（与 w500 口径 knob、exit-gnn 结构 knob 同类），
  无新路径/新方法 ⇒ 无需 research 适配研究。
- **research 产物**：`无`（设计输入 = 本仓库代码 + exit-gnn/w500 任务实测）。
