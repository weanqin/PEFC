# summary — 10-29-result-ckpt-reg

> 任务级汇总；Run 细节见 experiment.md（Run 001），同意凭据见 consent.md，需求与 AC 见 prd.md。

## 交付物

- **筛选结论**：mild + modma 全量核查（outputs/ 41 个任务目录中 mild/modma 相关全部模型族），「五折训练 + 五折 EC 双全」唯一合格 = **mild · latent-flow symw**（`10-25-md-lf-anti0.2-gnnexit-symw`）。
- **ckpt 注册**：`result/checkpoints/mild/latent_flow_symw/fold{0..4}/` ×10 文件（5×flow 5,219,587 B + 5×VAE 21,495,207 B，合计 133,548,970 B）——fold0 flow 源自 `10-25-latent-flow-exit-gnn-symw`、fold0 VAE 借用 w500 基座（`10-24-latent-flow-mild-w500`，config `cli/vae_ckpt` 指证）、fold1/2 源自 `10-25-latent-flow-symw-f12`、fold3/4 源自 `10-28-latent-flow-symw-f34`。
- **随档登记**：`result/checkpoints/MANIFEST.md`（逐文件 sha256/size/mtime/来源任务/原始路径/口径摘要/不合格清单）+ `result/README.md` 当前状态注册记录。
- **校验**：`cp -p` 只读复制后 `cmp` 10/10 逐位一致（AC1）；MANIFEST sha256+size 对实存文件 10/10 复核过（AC2）；源文件 mtime 未动（AC4）；`*.pt` 按仓库既有 `.gitignore` 规约不入 git（仓库 0 个 tracked .pt），ckpt 为盘上正式交付物、清单与登记入库。

## 关键数字

| 项 | 结果 |
|---|---|
| 五折训练 stage2 best val | fold0 **0.8054@E144**（150 顶格）/ fold1 **0.8264@E116** / fold2 **0.8433@E113** / fold3 **0.8358@E113** / fold4 **0.8769@E144**（150 顶格）；均值 **0.8376**（从各 run logs/stage2_history.json 现场重提，与看板一致） |
| 五折 EC | 8 例/折 ×5 全完成（conn v4 公式；fold0 `10-25-symw-ec-unbiased`、fold1/2 `10-25-symw-ec-f12`、fold3/4 `10-28-symw-ec-f34`） |
| EC×GC 对照（v1 refs 1–45 Hz fixed8） | 五折组 Spearman +0.134 / +0.159 / +0.176 / +0.2124 / +0.1822 |

## 未注册清单（不满足双全，2026-10-30 时点）

| 数据集 | 模型 | 缺口 |
|---|---|---|
| modma | 0705 std graph-mexhat | 训练 5/5 ckpt 齐，**EC 4/5**——fold0 训练 2026-10-30 22:38 起在途（活动任务 10-29-modma-ec-te-corr，链式 train→EC），跑完即合格；补注册时须含仓库外 VAE ×5（`/root/autodl-tmp/latent_ode/checkpoints/fold{0..4}_oc_latent_flow_vae_best.pt`，已确认在位） |
| modma | symw（10-28-modma-symw-f0） | 仅 fold0（1/5 折） |
| mild | w500 / exit-gnn / w500-no-reg-flow-only / 10-21 三窗口旧线 | 均单折 |

## 边界

- 注册 = 正式结果登记（result-dir-policy 情形 1：用户指令点名 result）；不构成 symw θ 候选/代理脑资格判定（该项仍待拍板⑩）。
- modma·0705 补注册不在本任务范围，随其活动任务收尾联动。

## Spec 更新判断（3.3）

无 spec 变更：本次实践（sha256 随档 + cmp 校验 + 跨任务借用件注记 + 不合格清单）被既有 `.trellis/spec/research/result-dir-policy.md`「随档登记」条款完全覆盖，未产生新契约或新坑；`*.pt` 不入 git 为 `.gitignore` 既有规约（零 tracked .pt 先例），无需新增条目。
