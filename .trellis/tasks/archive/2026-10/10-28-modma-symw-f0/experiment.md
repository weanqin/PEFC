# 实验日志 — 10-28-modma-symw-f0

> 每次「跑批/训练/评估/统计」追加一个 Run 块：跑前先写「目的+配置」，跑后同会话补「结果+结论」，不留空块。
> 格式契约见 `.trellis/spec/research/experiment-log.md`；产物只落本任务声明的变体目录（`.trellis/spec/research/path-isolation.md`）。

## Run 001 — 冒烟：训练通路（7 例×2 ep）+ EC mini（1 例×8 源）（2026-10-28）

### 目的与假设

- 验证移植链路：mild symw 变体口径在 MODMA fold0 数据管线上可训练、可存档；EC driver 对 symw-exit-GNN ckpt strict 装载 + v4 conn 全通路无错。

### 配置

- 代码：`models/latent-flow-v1/train_modma_symw.py` + `ec_modma_symw_v4.py`（本任务新增，py_compile PASS）
- 参数：训练 `--epochs 2 --limit-subjects 7 --save-path smoke/ckpt_smoke.pt`（5 训 + 2 val，_build_loaders 5 例 held-out 语义——limit 须 >5）；EC `--subjects 1 --sources 8`
- 环境：base conda @ 本机 GPU 4080 SUPER，`MODMA_DATA_ROOT=/root/autodl-tmp/modma`
- 数据：MODMA resting fold0（SubjectFoldSplitter 5 折 seed42），500 窗/step250/obs375/fc125，逐被试 z-score；VAE = 旧库 `fold0_oc_latent_flow_vae_best.pt`（只读）
- 命令：见 --run-dir `outputs/10-28-modma-symw-f0/smoke/`

### 结果

- 训练：VAE 装载 ✓（flow 键重初始化 19 missing/4 unexpected，graph 流预期）；**trainable 936,482 = mild run_symw_fold0 config 逐位同值**（架构移植硬校验）；anti-uniform λ=0.2 + SymWAdam（W←(W+Wᵀ)/2）生效；2 ep loss 2.43→2.22；smoke test MAE 0.9651/std_ratio 0.8564；38.9s。
- EC mini：strict 装载 ✓；τ=2.199e-5，P(resp>τ)=0.500 构造性校验 ✓；conn v4 落档 ✓；0.31 min（8 源）。
- GC 匹配：10-27 GC json 的 subject_file 与 fold0 test 10 例全对上（集合相等）。
- 产物：`outputs/10-28-modma-symw-f0/smoke/`（ckpt_smoke.pt + configs/logs；ec_mini 验后即删，不占存档）。

### 结论

- 支持：全链路可跑，正式训练放行。规模外推：~1s/step ⇒ 正式 38 训例 ≈ 99+13 步/epoch ≈ 65–75s/epoch ⇒ 150 ep ≈ 2.5–3h（高于 briefing 估时 ~90min，如实修正）。

## Run 002 — 正式训练：MODMA fold0 × symw-exit-GNN stage-2（150 ep）（2026-10-28）

### 目的与假设

- mild 10-25-symw-gnn-exit 变体口径在 MODMA resting fold0 上取得可用代理脑 ckpt（用户拍板口径）。

### 配置

- 代码：同 Run 001（git_head 随 config 落档）
- 参数：lr 1e-3 / 150 ep / batch 64 / patience 30 / seed 42 / anti-uniform λ=0.2 / symw ON / unfreeze-decoder（= mild run_symw_fold0 configs/stage2_config.json 逐字段对齐）
- 环境：同 Run 001
- 命令：`OMP_NUM_THREADS=8 ... MODMA_DATA_ROOT=/root/autodl-tmp/modma python3 train_modma_symw.py --run-dir /root/npi/outputs/10-28-modma-symw-f0/run_01`

### 结果

- **AC1 兑现**：150 ep 跑满（无早停），best val loss **0.6532 @E139**（组分 recon2 0.351 / corr2 0.131 / cons2 0.168 / uni_reg 0.0155）；train loss 终值 0.608；wall 5248.7s（87.5 min）；VAE 装载 ✓（19 flow 键重初始化，graph 流预期）；trainable 936,482 = mild symw fold0 同值；SymWAdam + λ=0.2 全程生效。
- smoke（test 少量批，375/125）：MAE 0.8458 / std_ratio 0.4785（生成分布欠幅，同 mild 线已知结构性缺陷方向，如实登记）。
- 产物：`outputs/10-28-modma-symw-f0/run_01/{checkpoints/fold0_modma_symw_exitgnn_graph_mexhat.pt, configs/stage2_config.json, logs/stage2_history.json}` + 控制台日志 `run_01_train.log`。

### 结论

- 移植训练成功（AC1）：mild symw 变体口径在 MODMA fold0 全量数据（38 训例 × 6392 窗）上稳定收敛、无发散；val 轨迹单调下降后趋平。
- 边界：bf16 AMP 非逐位确定；新变体首训无对拍锚点；不构成代理脑资格判定。

## Run 003 — EC 主跑（symw ckpt × v4 conn × 07-05 参数，10 例）（2026-10-28）

### 目的与假设

- 对 Run 002 ckpt 以 v4 公式计算 fold0 test 10 例逐被试 EC（无偏 μ=0、两阶段 global-τ）。

### 配置

- 代码：`ec_modma_symw_v4.py --mode main`；ckpt = Run 002 产物（训练侧同构建函数 strict 装载）
- 参数：q=0.5 / σ_ref=0.3 / grid log[0.01,0.3]×16 / 每例 32 窗 / n_steps=10 / batch=32 / seed=42 / conn v4 = −(lnσ*+½ln2πe)+5/7
- 命令：`OMP_NUM_THREADS=8 ... MODMA_DATA_ROOT=/root/autodl-tmp/modma python3 ec_modma_symw_v4.py --mode main --ckpt <run_01 ckpt> --out-dir outputs/10-28-modma-symw-f0/ec_main`

### 结果

- **AC2 兑现**：10/10 例成功（源失败 0）、47.4 min 总耗时（4.7 min/例）；τ_global ∈ [1.66e-4, 7.44e-4]（均值 3.54e-4）；钳边型态（10 例均值）：interior 52.6% / @σ_min 2.1% / @σ_max 45.3%（逐例见 ec_main_run.log）；**P(resp>τ)=0.500 逐例精确成立**（构造性自洽校验）；payload 含 method/hparams/conn 公式 v4/noise 契约/checkpoint 名。
- 产物：`outputs/10-28-modma-symw-f0/ec_main/subject_{0..9}.json`（10 例）+ `ec_main_run.log`。

### 结论

- EC 主跑成功（AC2）；symw-exit-GNN 变体的 σ* 场以 @σ_max 钳边为主（弱连接端钳边 ~45%，与 07-05 原版口径的近 50/50 平衡型不同）。

## Run 004 — gccorr：v4 EC × 10-27 现算 GC（离线）（2026-10-28）

### 目的与假设

- 逐被试 + 组平均两级相关性对照（描述性并置协议）。

### 配置

- 代码：`ec_modma_symw_v4.py --mode gccorr --main-dir <ec_main> --out-dir <gccorr>`；GC = `outputs/10-27-gc-fold0-aic-anchor/gc_aic/gc`（AIC max8/1-125Hz/nf65/fs250/全长，经旧库残留对拍逐位一致）
- 协议：offdiag 16256 对，|EC|/raw × aligned/flipped × Spearman/Pearson + 组均值矩阵

### 结果

- 逐被试 Spearman |EC| aligned（entropy v4）：s59 +0.065 / s67(25) −0.028 / s60 −0.048 / s49 −0.151 / s68 −0.022 / s67(23) −0.061 / s57 +0.088 / s87 −0.127 / s84 −0.110 / s98 +0.134
- **组级：逐被试均值 −0.0260±0.0951 | Pearson −0.0009±0.0716 | 组均值矩阵 Spearman −0.0245**
- flipped ≈ aligned（GC 近对称，无方向判别力，先例同）；inv 读数与 entropy 逐位同秩（v4 单调等价自洽）
- 产物：`outputs/10-28-modma-symw-f0/gccorr/gccorr_summary.json`

### 结论

- **AC3 兑现**；如实登记：该 symw-exit-GNN 变体在 MODMA fold0 上的 v4 EC 与观测 GC 相关性≈0（组均值 −0.03，散布 ±0.10 内正负参半）——远低于 07-05 原版口径的 OC↔GC +0.29 锚点（不同 ckpt/不同变体，不可归因于单一因素：exit-GNN 出口 + W 强制对称 + λ=0.2 正则 + 新数据集训练的复合差异）。
- 边界重申：描述性并置，不构成代理脑资格判定；bf16 首训无对拍锚点。


