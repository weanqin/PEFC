# design — 10-24-latent-flow-mild-w500

## 0. 输入与引用

| 输入 | 位置 |
|---|---|
| 上轮方案全文（架构/超参/口径论证，用户审阅通过） | 本任务主会话（2026-10-24），要点收录于 prd.md |
| 上轮负证据链（Run 001–006、coverage/medoid、patience 教训） | `.trellis/tasks/archive/2026-10/10-21-latent-flow-mild-train/{prd,experiment,summary}.md` |
| 数据契约（40 例/30 trial/1500 点/manifest SHA256/E 序假设） | `.trellis/spec/data/mild-depression.md` |
| 质量抽查口径（coverage/medoid 双口径） | `.trellis/spec/research/evaluation-calibers.md` |
| 现行代码基线 | `models/latent-flow-v1/train_mild_depress.py`（wrapper）、`viz_mild.py`（v2.1）、`latent_flow/models/oc_latent_flow.py`（模型，零改动）、`latent_flow/scripts/train_oc_latent_flow.py`（包内训练器，零改动） |

## 1. 总体结构：三处变体层改动，包体零 diff

```
models/latent-flow-v1/
  train_mild_depress.py      # 改：--window-len / --anti-uniform-lambda / --unfreeze-decoder
                             #     + 归一化范围联动 + wrapper 本地 stage2 runner
  viz_mild.py                # 改：--window-len 口径透传（数据切片与归一化同训练侧实现）
  quality_check_mild.py      # 新：数值抽查持久化脚本（coverage/medoid 双口径）
  latent_flow/               # 零改动（硬约束 AC5）
```

## 2. 数据流设计（新口径）

```
trial (30,128,1500)  [concat x_p‖x_f，契约保证 trial 内连续]
  → slice [:, :, :500]                     # 前缀窗口；数据文件 x_f 字段不参与
  → per-channel z-score                    # 训练折 28 例 · 全部 trial · 前 500 点统计
                                            # (wrapper float64 sum/sumsq, std==0→1.0)
  → 每 trial (500,128) float32
      x_obs    = window[:375]              # (375,128) 编码器唯一输入
      x_target = window                    # (500,128) 重建目标（全窗，obs+fc）
      x_fc     = window[375:]              # (125,128) stage2 潜空间目标与 fc 损失真值
```

- `t_obs = arange(375)/250`、`t_target = arange(500)/250`；模型 `obs_len=375, forecast_len=125,
  t_full=500`——与包内 MODMA 原生口径完全同构（23.4 样本/潜步）。
- **归一化范围联动**：`compute_mild_train_norm(train_files, window_len)` 累加域 = `x[:, :, :window_len]`；
  `window_len=1500` 时与旧行为逐位一致（默认兼容）。norm stats 的 mean/std SHA256 与 window_len
  一并落 config（AC6）。
- 断言链（失败即拒跑）：`0 < window_len <= 1500`、`obs_len + forecast_len == window_len`、
  manifest 全量断言重放（40 例逐文件 shape/fs/label/subject_id/起点样本 + 折结构 + 计数
  840/120/240）——全部沿用现行实现，仅断言对象从定长 1500 换为 `window_len`。

## 3. train_mild_depress.py 改动设计

### 3.1 新 CLI（默认值 = 旧行为，bit 兼容）

| 参数 | 默认 | 新口径取值 | 说明 |
|---|---|---|---|
| `--window-len` | 1500 | 500 | 前缀窗口长度；数据集切片与归一化范围联动 |
| `--anti-uniform-lambda` | 0.0 | 0.1 | stage2 mexhat mixer A_eff 反均匀正则权重（经 `Stage2Step` 生效，loss 增 `uni_reg` 项） |
| `--unfreeze-decoder` | 关 | 开 | stage2 decoder 联合训练开关（见 3.2） |

沿用既有参数：`--obs-len 375 --forecast-len 125 --batch-size 64 --lr 5e-4 --epochs 150
--patience 30 --seed 42 --beta-kl 1e-4`（epochs/patience 默认值不动，正式 Run 显式传参，
config 全量落盘）。`--run-dir` 显式指向本轮新目录（wrapper 默认 OUT_BASE 仍指上轮任务目录，
保持旧口径 replay 语义）。

**口径守卫（review B 建议 3 采纳）**：wrapper 启动断言——当 `window_len != 1500` 或
`unfreeze_decoder` 或 `anti_uniform_lambda != 0`（本轮新口径特征）而 `run_dir` 仍解析为
OUT_BASE 默认值时**拒跑**（防漏传 `--run-dir` 误写上轮受保护目录树；旧口径 + 旧默认目录
组合不受影响，bit 兼容保持）。另断言：`--dense-fallback` 与 `anti_uniform_lambda > 0`
互斥（dense 流无 graph_mixer，正则会静默失效）。

### 3.2 stage2 冻结面：wrapper 本地 runner（不能复用包内 `_run_stage2` 的原因）

包内 `_run_stage2` 冻结逻辑只有两态：`unfreeze_vae=False` → 冻结 encoder+proj+decoder；
`True` → 全解冻（含 encoder，且语义为 08-01 joint ablation）。均不满足"仅冻结 encoder"。
故 wrapper 本地 `_run_stage2_mild(model, train_loader, val_loader, args, save_path, vae_ckpt)`：

- **复刻**包内流程：加载 stage1 ckpt（format v1 解包，`strict=False`，非 flow 键不匹配即报错，
  flow 键重初始化预期）；`Stage2Step`（设 `anti_uniform_lambda = args.anti_uniform_lambda`）；
  `BaseTrainer(use_amp=True, max_grad_norm=1.0, patience=args.patience)`；
  `Adam([p for p in parameters if requires_grad], lr=args.lr)`。
- **差异 1（冻结集）**：冻结 `encoder` + `proj`；decoder + flow 可训练。proj 不在
  `forward_stage2` 计算图内（无梯度路径），冻结仅为显式声明，行为与"仅冻结 encoder"无差。
  打印 trainable 参数量（AC3 核对项：总 1,243,234 − encoder − proj）。
- **差异 2（KL）**：不挂 `_KLAnnealCallback`。encoder 冻结 ⇒ `_kl_zobs(mu,logvar)` 对可训练
  参数无梯度，β 恒 0（`__init__` 默认），与包内 frozen 分支行为一致；val loss 不含 KL 贡献。
- **差异 3（history 落盘）**：wrapper 自写 `run_dir/logs/stage2_history.json`（本地 runner 不触
  包内 `logs/mild/` 目录，避免跨任务污染；stage1 沿用现行 wrapper 自写路径）。
- checkpoint 经 BaseTrainer → `model.save_checkpoint()`（自描述 format v1；viz 加载器已兼容）。

### 3.3 config json 增记

`windows` 增 `window_len`；`norm_stats.source` 明示 `train fold first-{window_len} samples`；
新增 `stage2` 块：`unfreeze_decoder`、`frozen_modules: ["encoder","proj"]`、
`anti_uniform_lambda`、trainable 参数量。复现凭据其余字段（manifest SHA256 / norm hash /
git head / cli 全量）沿用。

## 4. viz_mild.py 改动设计

- 新增 `--window-len`（默认 1500），透传至：`compute_mild_train_norm(train_files, window_len)`、
  trial 切片（与训练侧同一实现路径 import），断言 `obs+fc == window_len`。
- 渲染口径 v2.1 不变（real-future + gen medoid + Q05–Q95 带）；时间轴随 `obs_len/fs` 自适应。
- `--out-dir` 默认值不动（指上轮目录），本轮显式传新目录。

## 5. quality_check_mild.py（新增，持久化上轮内联抽查）

- 输入：`--fold 0 --ckpt <stage2 best> --window-len 500 --obs-len 375 --forecast-len 125
  --subjects 1,7,21 --trials all --k 32 --seed 42 --out-dir <run>/quality`。
- 入口 BLAS 线程护栏与同目录脚本同式（numpy 导入前设上限，environment.md 契约）。
- 采样管线复用 `viz_mild.sample_forecast`（import，同一实现；确定性采样 ⇒ K 条逐位相同，
  coverage 结构性 0 如实输出并注明）。subjects 默认 fold test 首 MDD+HC（sid 1/7/21，与上轮
  Run 004–006 同 triad，保持判读连续性）。
- 指标（双口径：全程 125 点 / 去末端 25 点——**定点沿用登记口径的绝对 25 点**（占 fc 窗比例
  由上轮 6.7% 变为 20%）；数值与上轮不可比已由窗口变更覆盖，不再另立口径变更）：
  1. 逐时间点 Q05–Q95 coverage：真实 future 各时刻 ∈ 32 样本分位带的比例（按例/按 trial 聚合）；
  2. medoid/real std 比：medoid 样本（`viz_mild.medoid_index` 实样本）逐通道 std ÷ 真实
     future 逐通道 std，按 trial 均值报告 mean/min/max。
- 输出：`quality_meta.json`（口径 + 全部数字 + ckpt/norm/seed 凭据）+ stdout 摘要表；
  数字回填 experiment.md。

## 6. 超参数总表（用户拍板最终版）

| 项 | 值 | 来源 |
|---|---|---|
| window / obs / forecast | 500 / 375 / 125（前缀单窗） | 用户 R1 |
| fold | 0（manifest 只读复用，train/val/test = 28/4/8 例） | 用户 R1 |
| batch size | 64 | 用户 R2 |
| lr（两阶段，Adam，常值） | 5e-4 | 用户 R2 |
| epochs / patience | 150 / 30（两阶段） | 方案默认（上轮收敛证据） |
| stage2 冻结面 | encoder+proj 冻结，decoder+flow 训练 | 用户 R2 |
| anti-uniform λ | 0.1（仅 stage2，mexhat mixer） | 用户 R2 |
| β_KL（stage1 退火终值，前 20% epochs 线性） | 1e-4 | 方案默认 |
| seed | 42 | 方案默认 |
| 流 | mexhat-graph，K=2，γ_W=0.1，γ_M=1.0；sparse λ=0 | 沿用 |
| decoder 范式 | 全窗 ConvDecoder（use_fc_decoder=False） | 沿用 |
| 采样 | 确定性（Euler 10 步，无后验采样） | 用户 R2 |
| 归一化 | 训练折 trial 前 500 点逐通道 mean/std | 用户 R2 |

损失函数（模型内置，零改动）：stage1 = recon + 0.1·td + 0.05·bd + β·KL + corr + cons；
stage2 = recon2 + corr2 + cons2 + 0.1·uni_reg（新增项经 Stage2Step）。

## 7. 风险与对策

| # | 风险 | 依据 | 对策/观察 |
|---|---|---|---|
| 1 | decoder 联合训练可能强化"绕过 z_fc"：decoder 靠 z_obs+skip 外推 fc，z_fc_hat 被边缘化，生成幅度/相关性不升反降 | 08-01 任务 oracle 证据（判决笔记 `latent_flow/docs/experiments-notes/08-02-cross-subject-pearson-bottleneck-diagnosis.md`）：全窗 decoder 喂真 z_fc 的 fc Pearson 仅 0.032 | 用户已知并拍板；观察 stage2 val corr2 走向 + 抽查 medoid 欠幅是否改善，如实登记（阴性也是本口径的有效证据）。备择解释备查：08-02 诊断在 MODMA 同 375/125 口径上将瓶颈定位到 encoder 时序压缩（125→16 潜步），decoder 侧迭代若阴性，encoder 侧是证据链下一待验假设 |
| 2 | anti-uniform 0.1 在建议带（0.01–0.1）上沿，可能过强 | 包内 CLI 建议 | 观察 uni_reg 曲线与 val loss 健康；仅训练不读出 EC，无下游污染 |
| 3 | lr 5e-4 × batch 64 组合未经本数据验证（上轮 1e-3×32） | 配置变更 | stage1 val 曲线平台化过早/不收敛 → 中止并回报用户拍板回退（不自行改口径） |
| 4 | coverage 恒 0（结构性） | `sample()` 全确定性；用户拍板沿用 | 抽查脚本注明结构性；判读以 medoid/real std + 波形为主 |
| 5 | AMP 下非逐位确定 | 上轮已登记 | config 已记 use_amp；复现以同 seed + 同硬件为口径 |
| 6 | 训练时长：上轮实测 150ep 单阶段 stage1 286 s / stage2 638 s（Run 005，4080S）；本口径窗更短（500 vs 1500 点）只会更快，两阶段串行预计 **<30 min 量级** | 上轮 experiment.md Run 005/006 wall time 实测外推 | nohup 后台 + 日志轮询；patience 30 可能提前停 |

## 8. 输出路径声明（path-isolation）

```
outputs/10-24-latent-flow-mild-w500/
  run_w500_fold0/{configs,checkpoints,logs,viz,quality}/
  smoke/                                    # Run 000（--save-path 隔离，不落正式 ckpt 名）
```

- `outputs/10-21-latent-flow-mild-train/`（上轮五目录）零触碰。
- `experiment.md` Run 编号：Run 000 冒烟 / Run 001 stage1 / Run 002 stage2 / Run 003 质量抽查。

## 9. 验证方案（每步门禁）

1. **S1 改造后静态验证**：`--help` 两套口径参数齐；旧默认路径 1-epoch 冒烟
   （`--limit-subjects 2`，断言链走通、产物落 smoke 目录）。
2. **Run 000 新口径冒烟**：`--window-len 500 --obs-len 375 --forecast-len 125 --unfreeze-decoder
   --anti-uniform-lambda 0.1 --epochs 2 --patience 1 --limit-subjects 2 --save-path smoke/...`
   （patience 1 而非 0：BaseTrainer 的 ckpt 落盘与早停整体包在 `patience > 0` 分支内，patience 0
   不产 ckpt，stage2 冒烟会因 `--vae-ckpt` 缺失而拒跑——review B 阻塞项修正）：
   核对 trainable 打印（= 总 − encoder − proj）、loss 含 `uni_reg`、Dataset 打印
   `window=500`、norm 打印前 500 点统计、口径守卫生效（不传 --run-dir 时拒跑）。
3. **Run 001/002 正式**：experiment.md 先登记再跑；收 stage1 后核 val 曲线（风险 3 门禁），
   再启 stage2。
4. **Run 003 抽查**：脚本数字 + viz PNG 交用户判读；experiment.md 回填。

## 10. 可比性登记

- 窗口（1500→500 前缀）与归一化（全长→前 500）均变 ⇒ 与上轮 Run 001–006 **数值不可直接比**
  （口径变更，evaluation-calibers 纪律）；与 MODMA/EDRA 线恢复时间压缩比层面的结构同构
  （23.4 样本/潜步），仍非数值可比。
- 上轮判读伪影（末端 Euler 发散 × 数据尾段衰减）在本口径失效：forecast 段 1.5–2.0 s 距尾段
  ~5.8 s 远；去末 25 点双口径仍保留（对 Euler 末步发散本身保持敏感）。
