# 实验日志 — 10-24-latent-flow-mild-w500

> 每次「跑批/训练/评估/统计」追加一个 Run 块：跑前先写「目的+配置」，跑后同会话补「结果+结论」，不留空块。
> 格式契约见 `.trellis/spec/research/experiment-log.md`；产物只落本任务声明的变体目录（`.trellis/spec/research/path-isolation.md`）。

## Run 000 — 口径改造冒烟（旧口径兼容 A 段 + 新口径两阶段 B 段）（2026-10-24）

### 目的与假设

- A 段（兼容假设）：`--window-len/--anti-uniform-lambda/--unfreeze-decoder` 全默认（= 旧口径
  特征）时 wrapper 走原路径不炸、产物落显式 run-dir——默认 bit 兼容承诺的执行面验证。
- B 段（接线假设）：新口径 `--window-len 500 --obs-len 375 --forecast-len 125
  --unfreeze-decoder --anti-uniform-lambda 0.1` 两阶段全链路成立——Dataset window=500、
  norm 为训练折前 500 点统计、stage2 trainable = 总−encoder−proj（decoder+flow）、
  stage2 loss 字典含 uni_reg、ckpt/history/config 齐落盘、口径守卫生效。
- 冒烟仅为接线验证，数字不作任何质量/口径比较。

### 配置

- 代码：`models/latent-flow-v1/train_mild_depress.py`（S1–S3 改造后，git 未提交工作区；
  改动面限 train_mild_depress.py / viz_mild.py / quality_check_mild.py）
- 环境：npi @ 本机 4080S（Python 3.12 / torch 2.8，对照 environment.md）
- 数据：`/root/autodl-tmp/Mild Depression/` manifest 只读复用 fold0，`--limit-subjects 2`
  截断（manifest 断言仍全量 40 例重放）
- 命令：
  - A 段：`python3 models/latent-flow-v1/train_mild_depress.py --fold 0 --stage 1 --epochs 1
    --patience 0 --limit-subjects 2 --run-dir outputs/10-24-latent-flow-mild-w500/smoke/compat_s1`
    （其余全默认）
  - B 段 stage1：`--fold 0 --stage 1 --epochs 2 --patience 1 --window-len 500 --obs-len 375
    --forecast-len 125 --limit-subjects 2 --run-dir .../smoke/r000 --save-path
    .../smoke/r000/ckpt_smoke_s1.pt`
  - B 段 stage2：同口径 + `--unfreeze-decoder --anti-uniform-lambda 0.1 --vae-ckpt
    .../smoke/r000/ckpt_smoke_s1.pt --save-path .../smoke/r000/ckpt_smoke_s2.pt`
  - 守卫验证：新口径特征 + 不传 `--run-dir` → 预期 AssertionError 拒跑

### 结果

- A 段（兼容）：旧口径路径原样跑通——Dataset 打印 window=1500/obs=1125/fc=375、Params
  5,363,361（与上轮存档逐位一致）、27.3 s、产物落 `smoke/compat_s1/`。
- B 段 stage1：window=500/375/125 口径跑通，2 epoch val 2.7277，ckpt/history 落
  `smoke/r000/`，smoke MAE=0.8315。
- B 段 stage2：`Trainable params (decoder+flow; encoder/proj frozen): 903,458 (total
  1,243,234, frozen 339,776)`——trainable = 总 − encoder − proj 成立（flow 488,897 +
  decoder 414,561）；`anti-uniformity regularizer: lambda=0.1` 生效，loss 字典含
  `uni_reg=0.013`；wrapper 本地 runner 标题/ckpt（自描述 format v1）/history
  （`_history_name` 约定）齐全。
- 守卫实测：新口径特征（仅 window_len=500）不传 `--run-dir` → AssertionError 拒跑
  （信息含受保护目录路径与处置指引）。
- config 凭据核对：`window_len=500`、norm source = `train fold first-500 samples`、
  stage2 块（runner/unfreeze_decoder/frozen_modules/anti_uniform_lambda/trainable_params）
  齐全。

### 结论

- 兼容与接线假设全部成立：S1–S3 改造按 design 落地，新口径两阶段全链路可用，默认路径
  bit 兼容（A 段参数量与上轮逐位一致），路径守卫从事后核对升级为事前拦截。
- 放行正式 Run 001（stage1）/ Run 002（stage2）。
- 补记：trellis-check 就地修复 2 处（_run_stage2_mild 补 sparse_lambda 绑定对齐包内；
  viz 注释勘误），复验通过（py_compile / 包体零 diff / 守卫实跑 / 1500 bit 等价数值证明 /
  参数面计数 1,243,234 = 339,776 冻结 + 903,458 可训）。

## Run 001 — stage1（VAE）正式 · fold0 前缀窗 500（2026-10-24）

### 目的与假设

- 新口径下 stage1 VAE 收敛假设：obs 375 / forecast 125 / batch 64 / lr 5e-4 / 150ep 上限
  / patience 30 下 val 稳定收敛（风险 3 门禁：平台化过早/发散 → 中止回报）。

### 配置

- 代码：`models/latent-flow-v1/train_mild_depress.py`（trellis-check 后工作区）
- 命令：`--fold 0 --stage 1 --window-len 500 --obs-len 375 --forecast-len 125
  --batch-size 64 --lr 5e-4 --epochs 150 --patience 30 --seed 42 --run-dir
  outputs/10-24-latent-flow-mild-w500/run_w500_fold0`
- 环境/数据：npi @ 4080S；manifest 只读复用 fold0（train 28 例/840 trial 全量）；
  norm = 训练折前 500 点统计
- 产物：`run_w500_fold0/{configs/stage1_config.json, checkpoints/fold0_mild_vae_best.pt,
  logs/stage1_history.json, logs/stage1.log}`

### 结果

- 训练完整：87 epoch（早停触发，patience 30），**best val 2.0992@E57**，wall 418.7 s。
- 轨迹健康：val 2.571（E1）→ 2.243（E10）→ 2.104（E50）→ best 2.0992（E57）→ 平台缓升
  （E75 2.128）→ E87 早停；lr 5e-4 × batch 64 无平台化过早/发散（风险 3 门禁通过）。
- KL 原始值 ~83–93（β=1e-4 贡献 ~0.009）——与 MODMA/EDRA/上轮 mild 各线同型（登记口径）。
- smoke（test 少批）：MAE=0.8616 / std_ratio=0.6621。
- 产物齐：`configs/stage1_config.json`、`checkpoints/fold0_mild_vae_best.pt`、
  `logs/stage1_history.json`、`logs/stage1.log`。

### 结论

- stage1 收敛假设成立：新口径 + lr 5e-4 + batch 64 下 VAE 稳定收敛，早停机制按设计工作。
- 放行 Run 002（stage2 依赖 ckpt 已就位）。

## Run 002 — stage2（decoder+flow 联合训练）正式 · fold0（2026-10-24）

### 目的与假设

- 用户拍板口径核心假设：stage2 冻结 encoder+proj、decoder 与 flow 联合训练 + anti-uniform
  λ=0.1，相对上轮「flow-only 冻结 VAE」能否在收敛健康的前提下改善 fc 段生成
  （观察 val corr2/cons2 走向；质量判读留 Run 003）。

### 配置

- 命令：`--fold 0 --stage 2 --window-len 500 --obs-len 375 --forecast-len 125
  --batch-size 64 --lr 5e-4 --epochs 150 --patience 30 --seed 42 --unfreeze-decoder
  --anti-uniform-lambda 0.1 --run-dir outputs/10-24-latent-flow-mild-w500/run_w500_fold0`
- 依赖：Run 001 ckpt（`fold0_mild_vae_best.pt`）；trainable 903,458 / frozen 339,776
- 产物：`run_w500_fold0/{configs/stage2_config.json,
  checkpoints/fold0_mild_flow_best_graph_mexhat.pt, logs/stage2_history.json,
  logs/stage2.log}`

### 结果

- 训练完整跑满 150 epoch（未触发早停），**best val 0.8356@E150（顶格仍在改善，未收敛）**，
  wall 709 s；trainable 903,458 / frozen 339,776，anti-uniform λ=0.1 生效。
- 轨迹：val 2.067（E1）→ 1.526（E10）→ 1.060（E60）→ 0.976（E120）→ 0.836（E150）——
  单调下降未见平台；best-epoch 分量 recon2 0.424 / corr2 0.199（⇒ val fc 逐通道
  Pearson ≈ 0.80）/ cons2 0.210 / uni_reg 0.030（贡献 0.1×0.030=0.003，A_eff 行和方差
  缓升 = 反均匀项在起作用且未主导）。
- smoke（test 少批）：MAE=0.9581 / std_ratio=0.8947（stage1-only 路径为 0.8616/0.6621）。
- 产物齐：`configs/stage2_config.json`、`checkpoints/fold0_mild_flow_best_graph_mexhat.pt`
  （自描述 format v1）、`logs/stage2_history.json`、`logs/stage2.log`。

### 结论

- 收敛假设部分成立：训练全程健康无发散、decoder+flow 联合训练收敛明显慢于上轮 flow-only
  （上轮 5:1/9:1 真收敛 E122/E125，本轮 E150 仍在下降）——150ep 预算对本冻结面偏低，
  **是否延长训练预算属口径变更，留用户拍板**（本 Run 按审阅配置如实执行）。
- E150 best ckpt 可用，放行 Run 003 质量抽查。

## Run 003 — 生成质量数值抽查 + viz v2.1（E150 best ckpt）（2026-10-24）

### 目的与假设

- 上轮阴性（medoid 欠幅 0.5–0.74 / coverage 0）在新口径 + decoder 联合训练 + anti-uniform
  下是否改善：medoid/real std 比（主判读）、coverage（结构性 0 照登）、波形人工判读。
- 单次运行（fold0 单 seed，bf16 AMP），结论按 evaluation-calibers 纪律标注。

### 配置

- 代码：`models/latent-flow-v1/quality_check_mild.py` + `viz_mild.py`（v2.1）
- ckpt：Run 002 `fold0_mild_flow_best_graph_mexhat.pt`（best val 0.8356@E150）
- 命令：`quality_check_mild.py --fold 0 --window-len 500 --obs-len 375 --forecast-len 125
  --k 32 --seed 42 --subjects 1,7,21 --out-dir run_w500_fold0/quality`；
  `viz_mild.py --fold 0 --window-len 500 --obs-len 375 --forecast-len 125 --subjects 1,7,21
  --out-dir run_w500_fold0/viz`（同 triad，与上轮 Run 004–006 判读连续）
- 口径：coverage/medoid 双口径（全程 125 / 定点去末 25，占比 20%）；确定性采样 ⇒ K 条
  逐位相同，coverage 结构性 0

### 结果

- 执行修复（主会话就地，run 纪律拦出）：`quality_check_mild.py` / `viz_mild.py` 的
  fold test 被试解析原用文件名 stem（`m_/h_` 组内编号，h_i↔subject_id 20+i 错位），
  改用 `load_manifest` 返回的 fold 记录 `test`（真实 subject_id）；修复后全链路通
  （py_compile 过；上轮 viz 靠显式 --subjects 绕过同坑，本轮连自动挑选分支一并修复）。
- **medoid/real std 比（主判读，确定性单轨迹幅度比）**：

  | sid | 组 | 全程 mean (min/max) | 去末 25 mean (min/max) |
  |---|---|---|---|
  | 1 | MDD | 1.016 (0.92/1.17) | 1.053 (0.95/1.23) |
  | 7 | MDD | 0.884 (0.67/1.11) | 0.904 (0.62/1.24) |
  | 21 | HC | 0.973 (0.79/1.18) | 1.032 (0.83/1.29) |
  | ALL | | **0.957 (0.67/1.18)** | **0.996 (0.62/1.29)** |

- **coverage 双口径 = 0.0000（90 trial × 128ch）**——确定性采样结构性（K 条逐位相同），
  不携带质量信息，照登不判读（用户 2026-10-24 拍板）。
- viz v2.1：`viz/wave_subj{01,07,21}_trial00.png` + `viz_meta.json`（2×4 通道面板，
  real 蓝线 + gen medoid 红线 + Q05–Q95 带；时间轴 1.5–2.0 s forecast 段）——待用户判读。
- 产物：`run_w500_fold0/quality/quality_meta.json`（含口径/norm/manifest/seed/K 凭据）。

### 结论

- **上轮 medoid 欠幅缺陷（主体 0.53–0.74）在本口径下基本消除**：ALL 全程 0.957 / 去末
  0.996，逐例均值 0.88–1.05——新口径（前缀 500 窗）+ decoder 联合训练 + anti-uniform 0.1
  的组合下，确定性代表轨迹幅度与真实对齐（sid 7 略欠幅 0.88–0.90 为最弱例）。
- coverage 结构性 0 如预期；**条件分布逐点方差维度本轮口径下不可检验**（须另轮开放
  后验采样）。
- 「波形是否通过」留用户判读；**stage2 E150 顶格未收敛**——延长预算是否另立 Run 留用户
  拍板。单次运行（fold0 单 seed，bf16 AMP）。
