# result/ 五折双全模型 ckpt 注册（mild·latent-flow symw 五折）

## Goal

在 mild/modma 两数据集全量筛选『五折训练 + 五折 EC 双全』模型：仅 mild·latent-flow symw 合格（fold0-4 训练 + EC 全完成）。将其 10 个 ckpt（5 flow + 5 VAE，fold0 VAE 借用 w500 基座）复制注册入 `result/checkpoints/`，随档 MANIFEST 登记（sha256 / 来源任务 / 原始路径 / 口径摘要），并报告不合格模型（modma·0705 fold0 EC 在途、modma·symw 单折、mild 其余单折线）。modma·0705 待 fold0 EC 完成后另行补注册。用户 2026-10-30 指令点名 result 目录 + AskUserQuestion 拍板为同意凭据。

## Requirements

1. **筛选（核查已完成，结论固化）**：mild + modma 全量检查 outputs/ 下所有模型族；唯一合格 = mild·latent-flow symw（变体 `10-25-md-lf-anti0.2-gnnexit-symw`）——五折训练 ckpt 齐（fold0 `outputs/10-25-latent-flow-exit-gnn-symw/run_symw_fold0/`、fold1/2 `outputs/10-25-latent-flow-symw-f12/run_symw_fold{1,2}/`、fold3/4 `outputs/10-28-latent-flow-symw-f34/run_symw_fold{3,4}/`）+ 五折 EC 齐（各 8 例：`10-25-symw-ec-unbiased` / `10-25-symw-ec-f12` / `10-28-symw-ec-f34`）。
2. **注册物（10 文件，保留原文件名）**：
   - fold0：flow `run_symw_fold0/checkpoints/fold0_mild_flow_best_graph_mexhat.pt` + VAE `outputs/10-24-latent-flow-mild-w500/run_w500_fold0/checkpoints/fold0_mild_vae_best.pt`（stage1 借用 w500 基座，config `cli/vae_ckpt` 指证，manifest 注记）；
   - fold1–4：各折 flow + 本折 VAE（各 run 目录 checkpoints/ 下）。
3. **目标布局**：`result/checkpoints/mild/latent_flow_symw/fold{0..4}/<原文件名>`。
4. **随档登记**：`result/checkpoints/MANIFEST.md`——逐文件 sha256 / size / mtime、来源任务（`.trellis/tasks/...`）与原始路径、模型口径摘要（变体名、训练口径、五折 val）、EC 口径（conn v4 公式、8 例/折、GC refs v1 1–45 Hz fixed8、五折组 Spearman）；含不合格模型清单及原因。
5. **result/README.md**「当前状态」段追加本次注册记录一行。
6. **约束**：写入许可 = 用户指令点名 result（result-dir-policy 情形 1，凭据入 consent.md）；只读复制，不改任何 outputs/ 原件；不干扰运行中的 modma fold0 训练进程（PID 18222，属活动任务 10-29-modma-ec-te-corr）；存入后 result/ 内不改（更新 = 新版本并排 + 登记）。

## Acceptance Criteria

- [x] AC1：`result/checkpoints/mild/latent_flow_symw/` 下 10 个 ckpt 与源文件 `cmp` 逐位一致。
- [x] AC2：MANIFEST.md 覆盖全部 10 文件，sha256 与实存文件一致（可复核），含来源任务 / 原始路径 / 口径摘要 / fold0 VAE 借用注记。
- [x] AC3：不合格模型报告齐备（modma·0705 fold0 EC 在途待补、modma·symw 仅 1/5 折、mild 单折线 w500 / exit-gnn / w500-no-reg / 10-21 旧线）。
- [x] AC4：result/README.md 当前状态含本次注册记录；未改动 result/ 以外任何 outputs/ 原件（时间戳/内容不变）。
- [x] AC5：modma·0705 未注册（fold0 EC 未完成，不满足双全标准；待其活动任务收尾后补注册）。

### 追加范围（用户 2026-10-30 二次指令：五折 EC_pert 结果 + EC×GC 相关性注册）

**Requirements（追加）**

7. 注册 mild·latent-flow symw 五折 EC_pert 结果（逐被试 json）、逐被试 EC×GC 相关性、组平均 EC×GC 相关性入 `result/ec_pert/mild/latent_flow_symw/`。
8. **版本口径（用户三次指令逐步收紧）**：fold0/1/2 **仅注册 recompute_v4 版本**（被试 json 取各任务 `recompute_v4/`，v4 公式；原始 v1 版 ec_main / curves / summary 不注册）；fold3/4 注册 v4 直出原版全套（ec_main + summary + gccorr）；混编 v1/v2 的 `crossfold_summary.json` 不注册。

**Acceptance Criteria（追加）**

- [x] AC6：`result/ec_pert/mild/latent_flow_symw/` 66 文件与源 `cmp` 逐位一致；`sha256.json` 全量随档。
- [x] AC7：conn 公式全 v4 断言（逐被试 json `connectivity_formula` 字段 == `-(log sigma* + 0.5*log(2*pi*e)) + 5/7`）；fold0/1/2 树内无任何非 recompute_v4 来源文件（原始 ec_main v1 json/curves/summary 已剔除）。
- [x] AC8：EC MANIFEST 含五折逐被试/组平均 EC×GC 关键数字表（+0.1344/+0.1590/+0.1757/+0.2124/+0.1822 等）、版本口径说明与明确未注册清单；result/README.md 当前状态含二次注册记录。

### 追加范围（用户 2026-10-30 三次指令：五折训练/测试损失数据注册）

**Requirements（追加）**

9. 注册 mild·latent-flow symw 五折训练的逐 epoch 损失历史文件入 `result/loss_data/mild/latent_flow_symw/`：5 折 stage2_history + fold1–4 本折 stage1_history + fold0 借用 w500 基座 stage1_history（与 ckpt 清单 #2 同源，来源注记）。
10. 语义辨析随档：history `val` = 4 例验证划分（模型选择/早停）曲线，**非 8 例测试折**；测试折训练侧无损失数据文件（仅 3-batch smoke MAE 控制台行，五折数值抄录进 MANIFEST）；如需 best-ckpt 全量测试折损失须现算（超出注册范围，待令）。

**Acceptance Criteria（追加）**

- [x] AC9：`result/loss_data/mild/latent_flow_symw/` 10 文件与源 `cmp` 逐位一致；`sha256.json` 全量随档；stage2 五折 best val（0.8054@E144/0.8264@E116/0.8433@E113/0.8358@E113/0.8769@E144，均值 0.8376）与看板一致。
- [x] AC10：MANIFEST 含 train/val 划分语义辨析、五折 smoke test MAE 抄录表、fold0 stage1 借用注记；result/README.md 当前状态含三次注册记录。

### 追加范围（用户 2026-10-30 四次指令：modma·0705 五折补注册，按 stage-1/stage-2 × fold 分层）

**Requirements（追加）**

11. 前提成立核验：10-29-modma-ec-te-corr 已完成归档，fold0 训练+EC 落盘（best val 1.0929@E44、EC 10/10）——modma·0705 达成首轮「双全」标准，执行挂起的补注册。
12. 注册三类产物，按 stage-1/stage-2 × fold0–4 分层：①ckpt ×10（stage1 = 旧库外部 VAE `/root/autodl-tmp/latent_ode/checkpoints/fold{N}_oc_latent_flow_vae_best.pt` 随档入库注记、stage2 = 仓库内 flow ckpt）；②训练损失历史 ×5（仅 stage2 有曲线；train/val/test 语义随档——val = 训练侧扣留 5 例、test 无数据文件仅 smoke 行抄录）；③EC_pert ×58（σ\* 矩阵与 tau 在逐被试 json；EC×GC 逐被试+组平均 fold1–4；EC×TE 两级五折）。
13. 缺口如实登记：**fold0 EC×GC 相关无文件**（te-corr 任务只做 TE）；cross_fold_summary 等混编/非范围件不注册。

**Acceptance Criteria（追加）**

- [x] AC11：`checkpoints/modma/latent_flow_0705/` 10 文件 + `loss_data/.../` 5 文件 + `ec_pert/.../` 58 文件全部 `cmp` 逐位一致，三份 `sha256.json` 随档；conn 公式 v4 前缀断言过。
- [x] AC12：MANIFEST 层级清晰（stage-1/stage-2 × fold），含 σ\*/tau 摘要表、逐被试/组平均 × GC/TE 四象限数字表、fold0 GC 缺口与未注册清单。
- [x] AC13：注册后 `result/` 内 modma 无任何未声明来源文件；stage1 外部件来源注记在案；README 四次注册记录同步。

## Notes

- Lightweight 任务，PRD-only。
- modma·0705 补注册（含仓库外 5 个 VAE：`/root/autodl-tmp/latent_ode/checkpoints/fold{0..4}_oc_latent_flow_vae_best.pt`）不在本任务范围。

## 子研究路由（1.0b / 1.0c）

<!-- trellis-research-routing -->

- **分类结果**：`待确认`（非科研任务——结果注册/管理操作，不涉研究迭代；用户已给全规格指令，无需向用户另行确认歧义）
- **依据**：任务本体为正式结果目录的 ckpt 登记操作，非 Study I/II 研究内容。
- **新路径/方法检测（1.0c）**：`不涉及`（纯文件复制 + 登记，无新方法路径）
- **research 产物**：`无`
