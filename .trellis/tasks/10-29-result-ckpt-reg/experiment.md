# 实验日志 — result/ 五折双全模型 ckpt 注册（mild·latent-flow symw 五折）

> 每次「跑批/训练/评估/统计」追加一个 Run 块：跑前先写「目的+配置」，跑后同会话补「结果+结论」，不留空块。
> 格式契约见 `.trellis/spec/research/experiment-log.md`；产物只落本任务声明的变体目录（`.trellis/spec/research/path-isolation.md`）。

## Run 001 — 五折双全筛选 + result/ ckpt 注册（2026-10-30）

### 目的与假设

- 要回答：mild + modma 全量模型族中，哪些「五折训练 + 五折 EC」双全？双全者的 ckpt 注册入 `result/`（用户指令点名 result，写入许可情形 1）。

### 配置

- 代码：无代码改动（纯文件操作：`cp -p` / `cmp` / `sha256sum`）。
- 参数：筛选标准 = 训练 5/5 折 ckpt 齐且 EC 5/5 折完成；注册物 = mild·latent-flow symw（`10-25-md-lf-anti0.2-gnnexit-symw`）10 ckpt。
- 环境：本机（Ubuntu 22.04，/root/npi），对照 `.trellis/research/environment.md`。
- 数据：不涉新数据；核查对象 = `outputs/` 全量（mild/modma 两数据集）。
- 命令：`cp -p <src> result/checkpoints/mild/latent_flow_symw/fold{0..4}/` + `cmp` 逐位 + `sha256sum` 随档（详见 summary.md）。

### 结果

- 指标：**唯一双全 = mild·latent-flow symw**（五折训练 val 0.8054/0.8264/0.8433/0.8358/0.8769 均值 0.8376；五折 EC 8 例/折 ×5；GC v1 refs 组 Spearman 五折 +0.134/+0.159/+0.176/+0.2124/+0.1822）。注册 10 ckpt（5 flow + 5 VAE，fold0 VAE 借用 w500 基座）→ `result/checkpoints/mild/latent_flow_symw/fold{0..4}/`，`cmp` 10/10 逐位一致、sha256 10/10 复核过。
- 产物：`result/checkpoints/mild/latent_flow_symw/`（10 文件，133,548,970 B）+ `result/checkpoints/MANIFEST.md` + `result/README.md` 当前状态记录。
- 图表：无。

### 结论

- 支持（筛选结论确凿）：modma·0705 训练 5/5 但 fold0 EC 在途（活动任务 10-29-modma-ec-te-corr，2026-10-30 22:38 起跑）→ 不满足双全、未注册，跑完后补注册（届时含仓库外 VAE ×5：`/root/autodl-tmp/latent_ode/checkpoints/fold{0..4}_oc_latent_flow_vae_best.pt`）；modma·symw 1/5 折；mild 其余四线均单折。
- 下一步 / 派生决策项：modma·0705 fold0 链（train→EC）跑完后的补注册时点，随其活动任务收尾联动。

## Run 002 — 五折 EC_pert 结果 + EC×GC 相关性注册（2026-10-30，用户追加指令）

### 目的与假设

- 要回答：同款双全模型（mild·latent-flow symw）五折的 EC_pert 结果与逐被试/组平均 EC×GC 相关性注册入 result/；fold0/1/2 按用户指令**仅取 recompute_v4 版本**。

### 配置

- 代码：无代码改动（`cp -p` / `cmp` / `sha256sum` / 程序断言）。
- 版本口径：fold0/1/2 ← `recompute_v4/`（v4 公式被试 json + gccorr 双档汇总，v1 原版/curves/summary 剔除）；fold3/4 ← 10-28 v4 直出全套（ec_main + summary + gccorr）；`crossfold_summary.json` 混编 v1/v2 不注册。
- 数据来源：`outputs/10-25-symw-ec-unbiased/`、`outputs/10-25-symw-ec-f12/fold{1,2}/`、`outputs/10-28-symw-ec-f34/fold{3,4}/`。
- 命令：`cp -p <src> result/ec_pert/mild/latent_flow_symw/fold{0..4}/{ec_main,summary,ec_gc_corr}/` + `cmp` 全量 + sha256.json 生成。

### 结果

- 指标：66 文件（123,934,584 B）`cmp` 66/66 逐位一致；`connectivity_formula` 全 v4 断言过；五折逐被试 EC×GC Spearman +0.1344±0.1587 / +0.1590±0.1164 / +0.1757±0.1269 / +0.2124±0.1073（8/8 全正）/ +0.1822±0.0928（8/8 全正），组平均矩阵 Spearman +0.1806/+0.3805/+0.2675/+0.3503/+0.2541。
- 产物：`result/ec_pert/mild/latent_flow_symw/`（fold{0..4}/ + crossfold/ + MANIFEST.md + sha256.json）+ `result/README.md` 二次注册记录。
- 图表：无。

### 结论

- 支持：三类产物（EC_pert 逐被试、逐被试相关、组平均相关）五折全注册且版本口径统一 v4；首版曾将 f012 原始 ec_main（v1 公式）一并复制的 123 文件方案被用户指令纠正，剔除非 recompute_v4 件后终版 66 文件。
- 下一步 / 派生决策项：任务暂不收尾（用户指令），后续轮次待命；modma·0705 补注册仍联动其活动任务。

## Run 003 — 五折训练损失历史注册（2026-10-30，用户三次指令）

### 目的与假设

- 要回答：同款双全模型五折的「训练时损失数据文件 + 测试集损失数据文件」注册入 result/；核查训练管线中测试集损失的真实落盘形态。

### 配置

- 代码：无代码改动（`cp -p` / `cmp` / `sha256sum` + 代码走读确认划分语义）。
- 注册物：5 折 `stage2_history.json` + fold1–4 本折 `stage1_history.json` + fold0 借用 w500 `stage1_history.json`（共 10 文件，599,646 B）。
- 语义核查（`train_mild_depress.py` 走读）：`trainer.fit(train_loader, val_loader)` ⇒ history `val` = **4 例验证划分**（28/4/8 三划分中的 val）；训练后 `_smoke_eval_mild(model, test_loader, n_batches=3)` 仅 3-batch smoke、控制台打印不落数据文件。

### 结果

- 指标：10/10 `cmp` 逐位一致；stage2 五折 best val 0.8054@E144（150 顶格）/0.8264@E116/0.8433@E113/0.8358@E113/0.8769@E144（150 顶格）均值 0.8376、stage1 五折 2.0992（w500 基座）/2.0358/2.1210/2.0326/2.0580——与看板/提交记录全吻合；五折 smoke test MAE 0.8974/0.8156/0.9087/1.0280/0.7925（3-batch、最终权重）抄录进 MANIFEST。
- 产物：`result/loss_data/mild/latent_flow_symw/`（fold{0..4}/ + MANIFEST.md + sha256.json）+ `result/README.md` 三次注册记录。
- 图表：无。

### 结论

- 支持且澄清口径：训练侧存在的损失数据文件 = 逐 epoch train（28 例）+ val（4 例验证划分）曲线；**8 例测试折无训练侧损失数据文件**（仅 smoke 行；测试折正式产物 = 已注册 EC_pert）。best-ckpt 全量测试折损失如需数值文件须现算——待用户指令，不在本轮注册范围。
- 下一步 / 派生决策项：任务继续暂不收尾；测试折损失现算与否待拍板。

## Run 004 — modma·0705 五折补注册（2026-10-30，用户四次指令）

### 目的与假设

- 要回答：10-29-modma-ec-te-corr 完成后 modma·0705 达成「双全」，执行首轮挂起的补注册：ckpt（stage-1/stage-2）+ 损失曲线 + EC_pert（σ\*/τ/EC×GC/EC×TE 两级）按 stage × fold 分层入 result/。

### 配置

- 代码：无代码改动（`cp -p` / `cmp` / `sha256sum` / 走读确认口径）。
- 注册物：ckpt ×10（stage1 外部旧库 VAE + stage2 仓库 flow）、stage2_history ×5、EC 58（被试 json 10/10/11/11/11 + GC 相关 ×4 + TE 相关 ×1）；目录 `checkpoints|loss_data|ec_pert /modma/latent_flow_0705/`。
- 口径核验：`_build_loaders`（train 43 内扣 5 val、test 10 独立）⇒ history val=验证被试；EC v4 公式前缀断言；TE=refs/symbolic_transfer_entropy modma5fold 10 例/折；GC=AIC refs（fold2–4 各跳 1 例无 ref）。

### 结果

- 指标：73/73 `cmp` 逐位一致（10+5+58）；stage2 best val 1.0929@E44/1.0014@E48/1.1253@E49/1.1151@E47/1.2039@E48（均 50ep 顶格）；tau 每折范围 [5.4e-5, 2.6e-4]；EC×TE 逐被试 +0.20~+0.36、组平均 +0.41~+0.61（p<1e-300）；EC×GC（fold1–4）逐被试 +0.21~+0.38、组平均 +0.37~+0.61。
- 产物：`result/{checkpoints,loss_data,ec_pert}/modma/latent_flow_0705/`（各含 MANIFEST.md + sha256.json）+ README 四次注册记录。
- 图表：无。

### 结论

- 支持且登记缺口：**fold0 EC×GC 相关无文件**（te-corr 任务只做 TE 五折）——MANIFEST 明确缺口并注明 GC refs fold0 档存在、可现算补齐待令；stage1 无训练曲线（旧库预训）、test 无损失数据文件（smoke 行五折抄录）均如实随档。
- 下一步 / 派生决策项：fold0 GC 相关现算补齐与否、测试折损失现算与否——均待拍板；任务继续暂不收尾。
