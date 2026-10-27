# 实验日志 — 10-24-latent-flow-exit-gnn

> 每次「跑批/训练/评估/统计」追加一个 Run 块：跑前先写「目的+配置」，跑后同会话补「结果+结论」，不留空块。
> 格式契约见 `.trellis/spec/research/experiment-log.md`；产物只落本任务声明目录（path-isolation）。

## Run 000 — 变体接线冒烟（stage2 全链路，新口径 + --flow-exit-gnn）（2026-10-24）

### 目的与假设

- 出口 GNN 变体在真实 wrapper 全链路成立：旗标开启构建/trainable 936,482（w500 冻结面 +
  33,024）/loss 含 uni_reg/ckpt+history+config 落盘/守卫拒跑；旗标关闭默认路径不受影响。

### 配置

- 代码：`flow_exit_gnn.py`（新）+ train/viz/quality 旗标接线（S1–S3 后工作区）
- 命令：stage2 `--fold 0 --stage 2 --epochs 2 --patience 1 --window-len 500 --obs-len 375
  --forecast-len 125 --limit-subjects 2 --unfreeze-decoder --anti-uniform-lambda 0.1
  --flow-exit-gnn --vae-ckpt <w500 Run 001 ckpt> --run-dir .../smoke/r000 --save-path
  .../smoke/r000/ckpt_smoke_s2.pt`
- 环境：npi @ 4080S；VAE 只读复用 w500 `fold0_mild_vae_best.pt`（best 2.0992@E57）

### 结果

- 全链路通：旗标开启构建 `graph-mexhat-exit-gnn`，config 凭据 `flow_exit_gnn: true` /
  `stage2.trainable_params: 936482`（= w500 冻结面 903,458 + 33,024，与 design 逐位一致）；
  VAE 装载面白名单过滤正常（w500 ckpt 只读复用）；loss 含 `uni_reg`；2 epoch val 2.3917，
  ckpt（自描述 format v1）+ history + config 落 `smoke/r000/`；wall 26.7 s。
- 实施期静态自验（implement agent）：初始 v≡0 实测成立（随机张量前向全零）、
  state_dict 无 `graph_out` 有 `exit_msg_layers/exit_out`、backward 后 msg 层与共享 W
  均有梯度、包体零 diff、`--help` 三处旗标在列。

### 结论

- 接线假设成立：变体经真实 wrapper 全链路可用，恒等起步与 A_eff 共享语义落地。
- 放行 Run 001 正式训练。

## Run 001 — stage2 正式 · 出口 GNN 变体（lr 1e-3，w500 VAE 复用）（2026-10-24）

### 目的与假设

- 用户假设方向：出口邻居混合（与入口共享 A_eff）在不破坏恒等起步的前提下，对 fc 段生成
  质量（medoid/real std、corr2）与连接结构的作用——相对 w500 Run 002（出口纯逐通道，
  lr 5e-4，best 0.8356@E150 顶格未收敛）的同口径对照。
- 变量：出口结构（+exit GNN）、lr（5e-4→1e-3）两因素同变——**与 w500 Run 002 的差异为
  二因素组合，单变量归因须谨慎（登记）**。

### 配置

- 命令：`--fold 0 --stage 2 --window-len 500 --obs-len 375 --forecast-len 125 --batch-size 64
  --lr 1e-3 --epochs 150 --patience 30 --seed 42 --unfreeze-decoder --anti-uniform-lambda 0.1
  --flow-exit-gnn --vae-ckpt /root/npi/outputs/10-24-latent-flow-mild-w500/run_w500_fold0/
  checkpoints/fold0_mild_vae_best.pt --run-dir outputs/10-24-latent-flow-exit-gnn/run_exitgnn_fold0`
- 模型：flow 521,921（净增 33,024）/ 全模型 1,276,258 / stage2 trainable 936,482
  （encoder+proj 冻结 339,776）；A_eff 与入口共享
- 产物：`outputs/10-24-latent-flow-exit-gnn/run_exitgnn_fold0/{configs,checkpoints,logs}`

### 结果（Run 001 训练）

- **预算内真收敛：早停 E146，best val 0.8241@E116**，wall 706.8 s；trainable 936,482、
  A_eff 共享生效（uni_reg 0.013→0.016，贡献量级 0.1×0.016 同 w500）。
- best-epoch 分量：recon2 0.416 / corr2 0.203 / cons2 0.204 / uni_reg 0.016。
- 对照 w500 Run 002（lr 5e-4 · 出口纯逐通道）：best 0.8356@**E150 顶格未收敛** → 本变体
  val 略优且真收敛；但**出口结构与 lr 两因素同变，优势归因不唯一（登记）**。
- smoke：MAE=0.8874 / std_ratio=0.7757（w500 为 0.9581/0.8947——std_ratio 预警幅度回落）。
- 产物齐；ckpt 文件名沿用 `fold0_mild_flow_best_graph_mexhat.pt`（wrapper suffix 未含
  exit-gnn 区分；本任务独立 run_dir 无覆盖风险，变体身份凭据 = config `flow_exit_gnn:
  true`——化妆性限制登记）。

### 结论（Run 001）

- 训练动力学假设成立：出口 GNN + lr 1e-3 在 150ep 预算内收敛且 val 略优于 w500；
  收敛性问题（w500 遗留拍板项）在本配置下消解。
- 质量判读移交 Run 002（幅度预警已现）。

## Run 002 — 质量数值抽查 + viz v2.1（出口 GNN 变体 best ckpt）（2026-10-24）

### 目的与假设

- 与 w500 Run 003 同口径对照：出口 GNN（+lr 1e-3）下 medoid/real std 与 coverage（结构性）
  是否保持/改善；波形人工判读。

### 配置

- ckpt：Run 001 `fold0_mild_flow_best_graph_mexhat.pt`（best val 0.8241@E116，exit-gnn 变体，
  config 凭据 flow_exit_gnn: true）
- 命令：`quality_check_mild.py --fold 0 --window-len 500 --obs-len 375 --forecast-len 125
  --k 32 --seed 42 --subjects 1,7,21 --flow-exit-gnn --out-dir run_exitgnn_fold0/quality`；
  `viz_mild.py` 同参（--flow-exit-gnn 透传）
- 口径：同 w500 Run 003（全程 125 / 定点去末 25 双口径；确定性采样 coverage 结构性照登）

### 结果（Run 002 抽查）

| sid | 组 | 全程 mean (min/max) | 去末 25 mean (min/max) | w500 对照（全程/去末） |
|---|---|---|---|---|
| 1 | MDD | 0.601 (0.53/0.67) | 0.496 (0.44/0.57) | 1.016 / 1.053 |
| 7 | MDD | 0.696 (0.48/0.94) | 0.662 (0.51/1.00) | 0.884 / 0.904 |
| 21 | HC | 0.637 (0.45/0.78) | 0.584 (0.45/0.74) | 0.973 / 1.032 |
| ALL | | **0.645 (0.45/0.94)** | **0.581 (0.44/1.00)** | **0.957 / 0.996** |

- coverage 双口径 0.0000（确定性采样结构性，照登）。
- viz v2.1 三 PNG + meta 落 `run_exitgnn_fold0/viz/`（--flow-exit-gnn 透传 strict load 正常）。

### 结论（Run 002）

- **幅度欠校准回归：medoid/real std 全程 0.645/去末 0.581，较 w500（0.957/0.996）显著
  回落至接近 10-21 旧线区间（0.53–0.74）**；val corr2 持平（0.203 vs 0.199）而幅度回落
  ——与风险登记的"出口再平均"机制一致（邻居混合平滑 v → 生成幅度收缩；零初始化门控被
  训练打开后该效应显现）。二因素 caveat：lr 1e-3 与出口结构同变，单变量归因待消融
  （如需：exit-gnn + lr 5e-4 补一 Run，属拍板项）。
- **θ 候选判断：本变体不如 w500 基座**（幅度校准显著劣化，收敛性改善不足以抵偿）；
  波形人工判读与是否消融归因留用户拍板。单次运行（fold0 单 seed，bf16 AMP）。
