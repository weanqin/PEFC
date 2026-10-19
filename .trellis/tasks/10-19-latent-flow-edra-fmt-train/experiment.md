# 实验日志 — 10-19-latent-flow-edra-fmt-train（latent-flow-v1 × edra_fmt 两阶段训练）

> 每次「跑批/训练/评估/统计」追加一个 Run 块：跑前先写「目的+配置」，跑后同会话补「结果+结论」，不留空块。
> 格式契约见 `.trellis/spec/research/experiment-log.md`；产物只落本任务声明的变体目录（`.trellis/spec/research/path-isolation.md`）。

## Run 001 — edra_fmt 首个 latent-flow 两阶段正式训练（stage1 VAE → stage2 mexhat 潜空间 CFM）（2026-10-19）

### 目的与假设

- 在 edra_fmt（48 例 @250 Hz）上以 latent-flow-v1 完成 study1 候选路径的首个正式训练 run：stage1 单独训 VAE，stage2 冻结 VAE 训潜空间 CFM 流匹配（mexhat-graph 变体，D1 拍板）。
- 假设：数据 shim（npz Descriptor + manifest 只读复用 + 384/128/64 窗）与两阶段训练循环在满量（34/4/10 例 × 930 窗）上可稳定收敛（val loss 下降、早停或达 50 epoch 上限）；产出 θ 候选供后续代理脑合格性检验。
- 本 run **不**回答代理脑有效性问题（预测质量 ≠ Surrogate Brain Validity，两问题独立评价）。

### 配置

- 代码：`models/latent-flow-v1/`（git 基线 3aec527 + 本任务 S1–S3 改动：config/datasets/modma 三处 additive + `train_edra_fmt.py` + `build_edra_fmt_geometry.py`）
- 数据：`/root/autodl-tmp/edra_processed_fmt/`（48 例 npz，HC 24 / MDD 24）；划分只读复用 `models/gpfm-mamba-v1-fmt/splits/fold0_status_ok_seed42_val4.json`（fold0 内层 34/4/10，test=[10,11,12,17,24,28,30,44,61,69]）
- 时间窗：window 512（obs 384 + fc 128）/ stride 64 / 边界丢尾，每例 930 窗（入口断言）
- 归一化：训练折 34 例逐通道 mean/std（float64），val/test 复用（`norm_stats` 严格 held-out 路径）
- 模型：`OCLatentFlow(n_channels=62, latent_dim=64, t_lat=16, obs_len=384, forecast_len=128)`；stage2 流 = mexhat-graph（`graph_k=2, out_mix_alpha=0.5, mexhat_gamma=0.1, gamma_m=1.0`，geometry = npy 基座 `geometry_edra_fmt.pt`，D4 拍板），无 A_eff 正则
- 超参（两阶段同）：epochs 50 / patience 10 / lr 1e-3 恒定 / batch 32 / seed 42 / beta_kl 1e-4（stage1 KL 退火 20% epochs）/ Adam / AMP bf16 / max_grad_norm 1.0
- 环境：base @ 本机（Python 3.12.3 / torch 2.8.0+cu128 / RTX 4080 SUPER 32GB），对照 `.trellis/research/environment.md`
- 命令（可复现）：
  - `nohup python3 /root/npi/models/latent-flow-v1/train_edra_fmt.py --stage 1 > …/run_01/logs/stage1.log 2>&1 &`
  - `nohup python3 /root/npi/models/latent-flow-v1/train_edra_fmt.py --stage 2 --vae-ckpt …/run_01/checkpoints/fold0_edra_fmt_vae_best.pt > …/run_01/logs/stage2.log 2>&1 &`
- 产物根：`/root/npi/outputs/10-19-latent-flow-edra-fmt-train/run_01/`（configs / checkpoints / logs）

### 结果

- **两阶段均完成**（各跑满 50 epoch 上限、未触发早停；train/val 曲线见 history JSON）：
  | 阶段 | wall | best val loss（best epoch） | 末 epoch val 分量 | test smoke（384/128 口径） |
  |---|---|---|---|---|
  | stage1 VAE | 3956.4 s（66 min） | **1.5526**（E50/50） | E50: recon 0.353 / corr 0.694 / cons 0.477 | MAE 1.0435, std_ratio 1.3377（proj bootstrap 路径） |
  | stage2 CFM 流（mexhat-graph） | 3616.3 s（60 min） | **0.5977**（E49/50） | E50: recon2 0.301 / corr2 0.109 / cons2 0.195 | **MAE 0.8708, std_ratio 1.0431**（flow 路径） |
- **阶段顺序证据**：stage2 config `checkpoints.stage1_vae` = run_01 内 stage1 产物路径；日志「flow keys re-initialized fresh: 15 missing, 4 unexpected（graph 流预期重初始化，非 flow 键零失配）」；「Trainable params (flow only): 476,357」= VAE 全冻结。
- **数据口径证据**（入口断言实测输出）：manifest 48/34/4/10 OK，test=[10,11,12,17,24,28,30,44,61,69]，HC 24/MDD 24，48 例 npz 字段+形状全过，每例窗数==930；windows 31620/3720/9300（train/val/test），norm=训练折统计（files_hash=e276f424b3df）。
- **产物**：`outputs/10-19-latent-flow-edra-fmt-train/run_01/`——checkpoints/{fold0_edra_fmt_vae_best.pt 12.8MB, fold0_edra_fmt_flow_best_graph_mexhat.pt 5.0MB}、configs/{stage1,stage2}_config.json（含 git HEAD 3aec527、manifest sha、窗口/归一化记录）、logs/{stage1,stage2}.log + history JSON（stage1 wrapper 自写 / stage2 复制自包内 logs/edra_fmt/）+ 冒烟 history 证据；geometry_edra_fmt.pt（npy 基座，D4）落数据根。
- **回归与隔离**：全量 pytest **67 passed / 0 failed / 17 skipped**（终态复跑 10.3s，冻结口径不变）；`git status` 仅 5 改（三包内 additive + .gitignore + README-port）+ 2 新（两 wrapper）+ 任务目录；包内无 checkpoints/；数据根仅新增豁免单文件 geometry_edra_fmt.pt；gpfm 文本命中 3 处均为只读引用（manifest 路径常量 + docstring + README-port 说明段），`import.*gpfm` 零命中、无代码依赖。
- KL 观察：两阶段 KL 原始值 ~93（beta=1e-4 贡献 ~0.009）——与旧库 fold1-4 已登记行为同型（08-01 任务曾以 beta=1e-3 消融），如实登记不改判。

### 结论

- **支持**：latent-flow-v1 × edra_fmt 两阶段管线在满量 34/4/10 例 × 930 窗上稳定可跑且收敛（两阶段 val loss 均单调下降至 50 epoch 上限，无发散/NaN/OOM）；数据划分与时间窗严格复用上一轮固化口径（断言全过），产物完全隔离可复现。θ 候选（stage2 flow ckpt + stage1 VAE）已产出。
- 流路径 test smoke（MAE 0.8708 / std_ratio 1.0431）优于 stage1 proj bootstrap（1.0435 / 1.3377），方向符合「流学了动力学而非复制 proj」（旧库 decision-B 修复的动机）；但**这只是 smoke 级 sanity，不是代理脑合格性判定**——按 surrogate-brain-qualification.md，固定 θ\* 前须另走三项检验（属后续任务）。
- 下一步 / 派生决策项：①评估与代理脑合格性检验（另立任务）；②σ 二分归因在本 θ\* 上的小规模冒烟（对照 perturb-ec 口径差异表）；③5-fold 扩展须用户另行拍板（study1 门禁）。
