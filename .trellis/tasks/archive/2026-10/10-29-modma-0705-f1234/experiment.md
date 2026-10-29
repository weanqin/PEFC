# 实验日志 — 10-29-modma-0705-f1234

> 每次「跑批/训练/评估/统计」追加一个 Run 块：跑前先写「目的+配置」，跑后同会话补「结果+结论」，不留空块。
> 格式契约见 `.trellis/spec/research/experiment-log.md`；产物只落本任务声明的变体目录（`.trellis/spec/research/path-isolation.md`）。

## Run 001 — 冒烟：std-mexhat 训练通路（fold1 限例×2ep）+ EC mini（v4 网格）（2026-10-29）

### 目的与假设

- 验证 07-05 口径（标准 graph-mexhat、flow-only、无正则）在 MODMA fold1-4 管线上的训练/存档与 EC（v4 网格 [0.01,0.48]×24）全通路无错。

### 配置

- 代码：`models/latent-flow-v1/train_modma_0705.py`（本任务新增）+ `ec_modma_symw_v4.py --arch std-mexhat`（本任务增开关，默认旧行为不变）；py_compile PASS
- 参数：训练 `--fold 1 --epochs 2 --limit-subjects 7 --save-path smoke/ckpt_smoke_f1.pt`；EC `--subjects 1 --sources 8 --sigma-max 0.48 --n-grid 24`
- 环境：base conda @ 本机 GPU 4080 SUPER，`MODMA_DATA_ROOT=/root/autodl-tmp/modma`
- 数据：MODMA resting fold1（SubjectFoldSplitter 5 折 seed42），VAE = 旧库 `fold1_oc_latent_flow_vae_best.pt`（只读）

### 结果

- 训练：flow-only 可训 **488,897**（= 07-05 架构 flow 参量，与 fold0 原版同构）；2 ep loss 2.60→2.50，存档/历史复制 ✓；17.6s。
- EC mini：std-mexhat strict 装载 ✓；grid log[0.01,0.48]×24、τ=2.72e-5、P(resp>τ)=0.500 ✓、conn v4 落档（hparams 含 arch 字段）；0.42 min（8 源）。
- 产物：`outputs/10-29-modma-0705-f1234/smoke/`（ec_mini 验后即删）。

### 结论

- 支持：链路放行；规模外推 训练 ~30min/折、EC ~7min/例 ⇒ 全链 ~7h。

## Run 002 — 正式链：fold1-4 训练（50ep/b64/p10/lr1e-3）→ EC（v4 网格）顺序执行（2026-10-29）

### 目的与假设

- 用户拍板口径的 fold1-4 全量训练与 v4 EC，汇总各折结果。

### 配置

- 代码：同 Run 001（git_head 随 config 落档）
- 参数：训练 50ep/batch64/patience10/lr1e-3/seed42（用户逐字指定）；EC q=0.5/σ_ref=0.3/grid log[0.01,0.48]×24/32 窗/seed42/conn v4/--arch std-mexhat；fold1 10 例、fold2-4 11 例
- 命令：顺序链 `for F in 1 2 3 4: train_modma_0705.py --fold $F --run-dir fold$F/run && ec_modma_symw_v4.py --mode main --arch std-mexhat --ckpt <fold ckpt> --out-dir fold$F/ec_main --fold $F --subjects 11 --sigma-max 0.48 --n-grid 24`（失败即断；日志 fold$F/{train,ec}.log）

### 结果

- （进行中，跑后补）

### 结论

- （跑后补）

## Run 003 — fold1 EC_pert × GC 相关（用户提供旧库原版 GC 矩阵）（2026-10-29）

### 目的与假设

- 用户令（中途）：暂停 fold1 GC 现算，改用其下载到 /root/autodl-tmp/modma-gc-ec.zip 的旧库原版矩阵（fold0-4 全折 × gc/gpd/te，07-18/07-23 生成）。

### 配置

- EC 源：`outputs/10-29-modma-0705-f1234/fold1/ec_main`（10/10，v4 网格 0.48×24）
- GC 源：解包至 `/root/autodl-tmp/modma-gc-ec/baseline_connectivity_fold1/gc`（AIC max8/1-125Hz/nf65/fs250/全长）
- 身份验证：zip fold0/gc/subject_0 与旧库残留 `results/baseline_connectivity/gc/subject_0.json` 及 10-27 现算三方**逐位一致**；fold1 被试集合与 EC 精确匹配（10/10）
- 协议：offdiag 16256，aligned/flipped × Spearman/Pearson + 组均值矩阵

### 结果

- 逐被试 Spearman aligned：**+0.2568±0.2198**（8/10 正；明细 +0.54~−0.15，见 gccorr_summary.json）
- **组平均矩阵 Spearman +0.4630 / Pearson +0.4552**
- flipped≈aligned（GC 近对称，先例同）
- 产物：`outputs/10-29-modma-0705-f1234/fold1/gccorr_summary.json`

### 结论

- fold1 的 07-05 口径模型（v4 网格 EC）与观测 GC 呈中强正相关，组平均读数（+0.46）在 fold0 锚点（+0.38，不同网格口径）量级之上——跨折可比对照待 fold3/4 EC 完成后并入总表（fold2-4 GC 已由 zip 全覆盖，另含 gpd/te 两方法备用）。

## Run 002 结果补记（全链完成，2026-10-30）

- **AC1/AC2 兑现**：4/4 折训练 50ep 跑满（无早停触发），EC fold1 10/10、fold2-4 11/11 全成功、源失败 0、P(resp>τ)=0.500 逐例成立。
- 训练 best val：fold1 1.0014@E48 / fold2 1.1253@E49 / fold3 1.1151@E47 / fold4 1.2039@E48。
- EC 钳边型态（4 折一致）：interior 77–86% / @σ_min <1% / @σ_max 14–22%（弱端钳边为主）；τ 量级跨折差 ~3x（6.6e-5 ~ 1.8e-4）。
- GPU 链 wall ≈ 6.5h（GC 现算争用解除后 EC 提速 ~5x）。

## Run 004 — fold2-4 相关性 + 跨折汇总（用户 zip 原版 GC）（2026-10-30）

### 配置

- 同 Run 003 协议；GC = zip `baseline_connectivity_fold{2,3,4}/gc`（每折 10 例——旧库当年只算 10 例/折，fold2-4 test 有 11 例 ⇒ 相关取 EC∩GC 10 例交集，缺例登记：fold2 02020014 / fold3 02020022 / fold4 02020025）

### 结果

| fold | 逐被试 Spearman | 组平均矩阵 Spearman |
|---|---|---|
| 1 | +0.2568±0.2198 | +0.4630 |
| 2 | +0.3757±0.0751 | **+0.6144** |
| 3 | +0.2582±0.0984 | +0.4478 |
| 4 | +0.2143±0.1828 | +0.3700 |
| **4 折均值** | **+0.2763** | **+0.4738** |

- 产物：各折 `fold{N}/gccorr_summary.json` + `cross_fold_summary.json`

### 结论

- **AC3 兑现**：07-05 口径架构损失 × 新训（50ep）× v4 网格 EC 在 fold1-4 全部与观测 GC 呈稳定中强正相关——组平均 +0.37~+0.61（均值 +0.474），逐被试 +0.21~+0.38（均值 +0.276）；跨折方向/量级一致，无 fold 异常。
- 与 fold0 参照：07-05 原版（0.3×16 网格）组平均 +0.379 在本表量级区间内（跨网格口径，描述性并置）。
- 边界重申：bf16 非逐位确定；描述性登记不构成代理脑资格判定。
