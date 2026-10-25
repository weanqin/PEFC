# latent-flow sym-W 变体 fold1/fold2 双折训练（10-25 symw 任务线衍生）

## Goal

symw 变体（任务 10-25-latent-flow-exit-gnn-symw 交付：exit-gnn 基座 + anti-uniform
λ0.2 + 每步 W←(W+Wᵀ)/2 强制对称投影）在 **fold1 与 fold2** 两个数据划分上分别完整训练
（每折 stage1 VAE + stage2 symw 流），报告训练配置一致性凭据、每折评估指标、两折
（含 fold0 参照）对比总结。**纯跑批任务：零代码改动**（代码 = 已交付并归档的
`flow_exit_gnn_symw.py` + `--sym-w` 接线，check 在档）。

## 背景与必要性

- symw fold0（Run 001）best val 0.8054@E144、medoid/real std 0.858/0.848——单折单
  seed；用户指令扩至 fold1/fold2 检验跨折稳定性。
- **仓库无任何 fold1/fold2 stage1 VAE ckpt**（已核查 outputs/ 全树）——历代 mild 线
  均只训 fold0；且跨折复用 fold0 VAE 违反 5-fold 隔离（归一化统计/训练集均须逐折
  held-out）⇒ **每折必须先跑本折 stage1**。
- 数据/配置零新指定：manifest 固定 `splits/mild_5fold_seed42.json`（40 例 28/4/8 ×5
  折互斥覆盖，入口全量断言重放），`--fold 1/2` 自动选折；模型配置 = 已在旗标里的
  symw 变体，无外部配置文件。

## Requirements

- **R1 stage2 配置（两折逐字一致，= symw Run 001 verbatim）**：`--fold N --stage 2
  --window-len 500 --obs-len 375 --forecast-len 125 --batch-size 64 --lr 1e-3
  --epochs 150 --patience 30 --seed 42 --unfreeze-decoder --anti-uniform-lambda 0.2
  --flow-exit-gnn --sym-w --vae-ckpt <本折 stage1 产物> --run-dir <本折目录>`；两折
  CLI 差异仅 `--fold/--run-dir/--vae-ckpt` 三项。
- **R2 stage1 配置（两折逐字一致，= w500 fold0 stage1 实际口径镜像）**：`--fold N
  --stage 1 --window-len 500 --obs-len 375 --forecast-len 125 --epochs 150
  --patience 30 --batch-size 64 --lr 5e-4 --seed 42 --run-dir <本折目录>`（β_kl 默认
  1e-4；w500 在档 best 2.0992@E57/87ep 早停；**注意 stage1 lr=5e-4 与 stage2 lr=1e-3
  不同**——分属两阶段各自的同款镜像，非笔误）。
- **R3 指标口径（生成式代理任务，无准确率/F1；沿用 evaluation-calibers.md）**：
  ①stage1 best val + 早停形态；②stage2 best val + 分量（recon2/corr2/cons2/uni_reg）
  + 收敛形态（早停 epoch / 顶格）；③smoke MAE/std_ratio；④质量抽查 medoid/real std
  双口径（全程 125 / 去末 25，k=32/seed=42）+ coverage 结构性照登（确定性采样无证伪
  力）；⑤训后 best ckpt W 对称断言（torch.equal 级）。
- **R4 抽查被试（各折 test 集 2 MDD + 1 HC，fold0 的 1,7,21 同构规则）**：fold1 →
  `2,12,22`（MDD,MDD,HC；test=[2,12,14,17,22,32,34,37]）；fold2 → `3,6,23`
  （MDD,MDD,HC；test=[3,6,8,11,23,26,28,31]）。跨折不可复用 fold0 的 1,7,21（非本折
  test）。
- **R5 产物隔离**：`outputs/10-25-latent-flow-symw-f12/run_symw_fold{1,2}/
  {configs,checkpoints,logs,quality,viz}` + 每折训练日志；fold0 参照只读引用
  `outputs/10-25-latent-flow-exit-gnn-symw/run_symw_fold0/`。
- **R6 顺序与纪律**：单 GPU 串行（fold1 s1 → fold1 s2 → fold2 s1 → fold2 s2，链式
  fail-fast）；不重训不改动任何在档产物。

## Acceptance Criteria

- [ ] **AC1** 两折 stage2 CLI 凭据一致：resolved config `cli` 对拍仅 fold/run_dir/
      vae_ckpt 三键不同，其余逐键相等（脚本对拍，非目测）。
- [ ] **AC2** 两折 stage1 均健康收敛（早停或顶格，best val 与 loss 曲线登记）。
- [ ] **AC3** 两折 stage2 均完成训练并落盘；best ckpt W 对称断言（torch.equal）双折
      通过；收敛形态登记（早停/顶格 + best epoch）。
- [ ] **AC4** 质量抽查 + viz 双折完成（R4 被试/口径），coverage 结构性照登；三折
      （fold0 参照 + fold1 + fold2）并置对比表入 experiment.md。
- [ ] **AC5** experiment.md Run 块（跑前目的+配置 / 跑后结果+结论）齐全；跨折结论
      （val/幅度校准稳定性）如实登记，θ 判断留用户。

## 边界（本任务不做）

- 零代码改动（含包体/变体层/wrapper）；不做 EC 消费、不做 θ 候选判定、不做 fold3/4。
- 不重复 fold0 训练（在档只读引用）。

## 子研究路由（1.0b / 1.0c）

- **分类结果**：`study1` —— EEG 代理脑（Study I）θ 候选产线的跨折训练验证。
- **依据**：交付物 = 代理生成模型多折训练与生成质量评估，无 EC 推断成分。
- **新路径/方法检测（1.0c）**：`已有路径：latent-flow（models/latent-flow-v1/）`——
  注册表既有路径内已交付变体的多折重训，无新方法/新路径 ⇒ 无需 research 适配。
- **research 产物**：`无`。
