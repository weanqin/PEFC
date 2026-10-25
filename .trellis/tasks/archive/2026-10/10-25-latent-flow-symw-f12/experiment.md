# 实验日志 — 10-25-latent-flow-symw-f12

> 每次「跑批/训练/评估/统计」追加一个 Run 块：跑前先写「目的+配置」，跑后同会话补「结果+结论」，不留空块。
> 格式契约见 `.trellis/spec/research/experiment-log.md`；产物只落本任务声明目录（path-isolation）。

## Run 001 — fold1 全流程：stage1 VAE（w500 口径镜像）+ stage2 symw（2026-10-25）

### 目的与假设

- symw 变体跨折稳定性检验第一折：fold1（train 28/val 4/test 8 = [2,12,14,17,22,32,34,37]）
  上完整复现 fold0 Run 001 流程——假设 best val 与幅度校准（medoid/real std ~0.85 档）
  在 fold1 复现，W 对称约束全程成立。

### 配置

- 代码：零改动（10-25 symw 任务交付版：`flow_exit_gnn_symw.py` + `--sym-w` 接线）。
- stage1（= w500 fold0 stage1 实际口径镜像）：`--fold 1 --stage 1 --window-len 500
  --obs-len 375 --forecast-len 125 --epochs 150 --patience 30 --batch-size 64 --lr 5e-4
  --seed 42 --run-dir outputs/10-25-latent-flow-symw-f12/run_symw_fold1`。
- stage2（= symw Run 001 verbatim）：`--fold 1 --stage 2 --window-len 500 --obs-len 375
  --forecast-len 125 --batch-size 64 --lr 1e-3 --epochs 150 --patience 30 --seed 42
  --unfreeze-decoder --anti-uniform-lambda 0.2 --flow-exit-gnn --sym-w --vae-ckpt
  <run_symw_fold1/checkpoints/fold1_mild_vae_best.pt> --run-dir .../run_symw_fold1`。
- 环境：npi @ 4080S；数据 manifest `splits/mild_5fold_seed42.json`（入口全量断言重放）。

### 结果

- **stage1**：best val 2.0358@E57，早停 E87（形态与 w500 fold0 同款：同为 87ep 早停、
  best 同在 E57；值略优 2.0992）。
- **stage2**：best val **0.8264@E116**，**真早停 E146**（patience 30 触发，预算内给出
  收敛信号——优于 fold0 的 150ep 顶格未早停形态）；wall 706.5 s。best-epoch 分量：
  recon2 0.418 / corr2 0.202 / cons2 0.204 / uni_reg 0.0149。
- smoke：MAE 0.8156 / std_ratio 0.8748（fold0 0.8974/0.8170——幅度比更优）。
- **W 对称断言**：best ckpt torch.equal 级（max|W| 1.251 实质对称结构）。
- 质量抽查（test 被试 2,12,22 = MDD,MDD,HC；k=32/seed=42/双口径）：
  sid2 0.938/0.958 · sid12 0.986/1.033 · sid22 1.110/1.138 ·
  **ALL 1.011 (0.73/1.49) / 1.043 (0.69/1.60)**——≈ 完美幅度校准；coverage 双口径
  0.0000（结构性照登）。viz 三 PNG + meta 落 `run_symw_fold1/viz/`。

### 结论

- fold1 假设成立且强于预期：val 0.8264（fold0 0.8054 同域）、幅度校准 1.011/1.043
  超出 fold0（0.858/0.848）与 w500 fold0（0.957/0.996）；对称约束与 λ0.2 的组合在
  该折全面复现并改善。

## Run 002 — fold2 全流程：stage1 VAE（w500 口径镜像）+ stage2 symw（2026-10-25）

### 目的与假设

- 同 Run 001，第二折：fold2（test = [3,6,8,11,23,26,28,31]）——两折 + fold0 参照并置
  给出跨折稳定性判读。

### 配置

- 同 Run 001 逐字，仅三处不同：`--fold 2`、`--run-dir .../run_symw_fold2`、
  `--vae-ckpt .../fold2_mild_vae_best.pt`（stage1/stage2 各自镜像）。
- 环境：同 Run 001（串行同机同日，单 GPU 链式 fail-fast）。

### 结果

- **stage1**：best val 2.1210@E49，早停 E79。
- **stage2**：best val **0.8433@E113**，**真早停 E143**（patience 30 触发）；wall
  692.0 s。best-epoch 分量：recon2 0.430 / corr2 0.205 / cons2 0.205 / uni_reg 0.0156。
- smoke：MAE 0.9087 / std_ratio 0.7791。
- **W 对称断言**：best ckpt torch.equal 级（max|W| 1.309）。
- 质量抽查（test 被试 3,6,23 = MDD,MDD,HC）：sid3 0.791/0.787 · sid6 0.938/0.952 ·
  sid23 0.928/0.948 · **ALL 0.886 (0.56/1.23) / 0.896 (0.55/1.47)**；coverage 双口径
  0.0000（结构性照登）。viz 三 PNG + meta 落 `run_symw_fold2/viz/`。

### 结论

- fold2 假设成立：val 0.8433、幅度校准 0.886/0.896（fold0 档位），真早停收敛；
  跨折幅度均值 ~0.92 档成立。

## Run 003 — 三折对比总结（fold0 参照 + fold1 + fold2，AC4/AC5）（2026-10-25）

### 目的与假设

- 跨折稳定性判读：symw 变体的 val 与幅度校准是否跨折稳健。

### 配置

- 数据 = manifest 同一划分文件逐折选择（CLI 对拍：三折 stage2 cli 剔除
  fold/run_dir/vae_ckpt 三键后**逐键相等**，脚本对拍非目测——AC1 凭据）；
  fold0 = 在档 Run 001 只读参照。

### 结果

| 指标 | fold0（参照） | fold1 | fold2 |
|---|---|---|---|
| stage1 best val | 2.0992@E57（87ep 早停） | 2.0358@E57（87ep） | 2.1210@E49（79ep） |
| stage2 best val | 0.8054@E144（150ep **顶格**） | 0.8264@E116（146ep 早停） | 0.8433@E113（143ep 早停） |
| smoke MAE / std_ratio | 0.8974 / 0.8170 | 0.8156 / 0.8748 | 0.9087 / 0.7791 |
| medoid/real std 全程 | 0.858 | **1.011** | 0.886 |
| medoid/real std 去末25 | 0.848 | **1.043** | 0.896 |
| W 对称（torch.equal） | ✓ | ✓ | ✓ |
| 抽查被试（各折 test） | 1,7,21 | 2,12,22 | 3,6,23 |

- 跨折统计：best val 0.805–0.843（均值 0.825，折间极差 0.038）；幅度校准全程
  0.858–1.011（均值 0.918）——**幅度回升跨折稳健**（对照 exit-gnn 单折 0.645 的
  回归被稳定修复，均值逼近 w500 单折 0.957）。
- 收敛形态：fold1/2 真早停（E146/E143），fold0 顶格——预算 150ep 边缘形态，三折
  best epoch 113–144 均贴预算尾部（残余改进空间 caveat 照登）。

### 结论

- **跨折稳定性成立**：val 三折 0.81–0.84 同域、幅度校准三折 0.86–1.01（exit-gnn
  的幅度回归在全部三折被对称化修复）、W 对称断言三折全过、配置一致性脚本对拍全等。
- 单折单 seed caveat 仍在（每折一次运行、bf16 AMP 非逐位路径）；θ 候选判断/波形
  人工判读（`run_symw_fold{0,1,2}/viz/`）留用户；fold3/4 未跑（用户未指令，属拍板项）。

