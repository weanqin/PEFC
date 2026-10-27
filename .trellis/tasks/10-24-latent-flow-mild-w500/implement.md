# implement — 10-24-latent-flow-mild-w500

> 执行前提：design.md 经 1.4 review + 用户明确同意、`task.py start` 已跑（status=in_progress）。
> 运行环境：`.trellis/research/environment.md` 所载 npi 环境（Python 3.12 / torch 2.8 / 4080S）；
> 数据根 `/root/autodl-tmp/Mild Depression/`（目录名含空格，引用加引号）。
> Run 纪律：每个 Run 先在 `experiment.md` 登记目的+配置，跑完同会话回填结果+结论。

## S1 — train_mild_depress.py 口径参数化（改动面：变体层 wrapper）

- [x] S1.1 `--window-len`（默认 1500）：`MildTrialDataset` 增 `window_len` 参数（trial 存储切片
      `[:window_len]`、`t_target=arange(window_len)*dt`、断言 `obs+fc==window_len` 且
      `0<window<=1500`）；`compute_mild_train_norm(train_files, window_len)` 累加域切片；
      `build_model_mild`/`_smoke_eval_mild` 透传 obs/fc（现状已参数化，核对即可）。
- [x] S1.2 `--anti-uniform-lambda`（默认 0.0）：进入钉死 train_args 与 config。
- [x] S1.3 `--unfreeze-decoder`（默认关）+ wrapper 本地 `_run_stage2_mild`（design §3.2：
      freeze = encoder+proj；Stage2Step 挂 anti_uniform；BaseTrainer AMP+clip+patience；
      history wrapper 自写（文件名沿用 `_history_name` 约定）；不再复制包内 logs/mild/）。
      `--unfreeze-decoder` 关闭时走原包内 `_run_stage2` 路径不动（默认兼容）。
- [x] S1.6 口径守卫（review B 采纳）：新口径特征（`window_len != 1500` 或 `unfreeze_decoder`
      或 `anti_uniform_lambda != 0`）而 run_dir 解析为 OUT_BASE 默认值时断言拒跑；
      `--dense-fallback` × `anti_uniform_lambda > 0` 断言互斥。
- [x] S1.4 config json 增记：`window_len`、norm source 串、stage2 冻结面/anti-uniform/trainable 数。
- [x] S1.5 模块 docstring 数据口径段更新（前缀窗口说明）。

**验证**：
```bash
python3 models/latent-flow-v1/train_mild_depress.py --help   # 新参数在列
# 旧默认路径不炸（1 epoch 冒烟，产物隔离）：
python3 models/latent-flow-v1/train_mild_depress.py --fold 0 --stage 1 --epochs 1 --patience 0 \
  --limit-subjects 2 --run-dir /root/npi/outputs/10-24-latent-flow-mild-w500/smoke/compat_s1
```

## S2 — viz_mild.py 口径透传

- [x] S2.1 `--window-len` 参数（默认 1500）→ `compute_mild_train_norm` 与 trial 切片透传；
      断言 `obs+fc==window_len`；时间轴随 obs_len/fs 自适应核对（v2.1 渲染不变）。

**验证**：`--help` 在列；待 Run 003 实跑验证。

## S3 — quality_check_mild.py 新增（数值抽查持久化）

- [x] S3.1 按 design §5 实现：import `viz_mild.sample_forecast`/`medoid_index` 与训练侧
      `load_manifest`/`compute_mild_train_norm`/`MildTrialDataset`；coverage（全程/去末 25 双
      口径，定点沿用）+ medoid/real std（双口径）；`quality_meta.json` 落盘（含确定性采样
      结构性注明）。入口 BLAS 线程护栏与同目录脚本同式（numpy 导入前）。
- [x] S3.2 默认 subjects = fold test 首 MDD+HC（sid 1/7/21）、k=32、seed=42。

## S4 — Run 000 冒烟（两阶段，新口径全链路；`--patience 1` 保 ckpt 落盘——patience 0 时
BaseTrainer 不存 ckpt，stage2 会因 `--vae-ckpt` 缺失拒跑，review B 阻塞项修正）

```bash
python3 models/latent-flow-v1/train_mild_depress.py --fold 0 --stage 1 --epochs 2 --patience 1 \
  --window-len 500 --obs-len 375 --forecast-len 125 --limit-subjects 2 \
  --run-dir .../smoke/r000 --save-path .../smoke/r000/ckpt_smoke_s1.pt
python3 models/latent-flow-v1/train_mild_depress.py --fold 0 --stage 2 --epochs 2 --patience 1 \
  --window-len 500 --obs-len 375 --forecast-len 125 --limit-subjects 2 \
  --unfreeze-decoder --anti-uniform-lambda 0.1 \
  --vae-ckpt .../smoke/r000/ckpt_smoke_s1.pt \
  --run-dir .../smoke/r000 --save-path .../smoke/r000/ckpt_smoke_s2.pt
```
- [x] 核对：Dataset 打印 `window=500`、norm 前 500 点、trainable = 总−encoder−proj、
      stage2 loss 字典含 `uni_reg`、产物全部落 smoke 目录、口径守卫生效
      （不传 `--run-dir` 时新口径拒跑）。

## S5 — Run 001 stage1 正式（先 experiment.md 登记；mkdir 前置——重定向目录须先存在）

```bash
RUN=/root/npi/outputs/10-24-latent-flow-mild-w500/run_w500_fold0 && mkdir -p $RUN/logs && \
nohup python3 models/latent-flow-v1/train_mild_depress.py --fold 0 --stage 1 \
  --window-len 500 --obs-len 375 --forecast-len 125 \
  --batch-size 64 --lr 5e-4 --epochs 150 --patience 30 --seed 42 \
  --run-dir $RUN > $RUN/logs/stage1.log 2>&1 &
```
- [x] 门禁：val 曲线健康收敛（风险 3：平台化过早/发散 → 中止回报用户，不自行改参）。

## S6 — Run 002 stage2 正式（先登记；依赖 Run 001 ckpt）

```bash
RUN=/root/npi/outputs/10-24-latent-flow-mild-w500/run_w500_fold0 && \
nohup python3 models/latent-flow-v1/train_mild_depress.py --fold 0 --stage 2 \
  --window-len 500 --obs-len 375 --forecast-len 125 \
  --batch-size 64 --lr 5e-4 --epochs 150 --patience 30 --seed 42 \
  --unfreeze-decoder --anti-uniform-lambda 0.1 \
  --run-dir $RUN > $RUN/logs/stage2.log 2>&1 &
```
- [x] 门禁：trainable 打印复核 + uni_reg 出现 + val 收敛。

## S7 — Run 003 质量抽查 + viz（先登记）

```bash
python3 models/latent-flow-v1/quality_check_mild.py --fold 0 \
  --ckpt .../run_w500_fold0/checkpoints/fold0_mild_flow_best_graph_mexhat.pt \
  --window-len 500 --obs-len 375 --forecast-len 125 \
  --out-dir .../run_w500_fold0/quality
python3 models/latent-flow-v1/viz_mild.py --fold 0 \
  --ckpt <同上> --window-len 500 --obs-len 375 --forecast-len 125 \
  --subjects 1,7,21 \
  --out-dir .../run_w500_fold0/viz
```
- [x] 数字/PNG 交用户判读；experiment.md 回填（coverage 注明结构性；medoid/real 双口径；
      viz subjects 1/7/21 与数值抽查及上轮 Run 004–006 判读 triad 对齐）。

## S8 — 质量检查与收尾

- [x] lint/typecheck（变体层脚本）；`git diff` 核对 `latent_flow/` 零改动（AC5）。
- [x] trellis-check（全范围：AC1–AC6 逐项）。
- [ ] 数据契约回填：`spec/data/mild-depression.md` §5 消费规则增记前缀 500 口径（口径变更
      告知用户后生效）。
- [ ] 评估口径回填（review A 采纳）：`spec/research/evaluation-calibers.md` mild 抽查条目增记
      两条——①确定性采样口径下 coverage 恒 0 属结构性、不携带质量信息（判读以 medoid/real
      std + 波形为主）；②本轮 125 点 forecast 沿用定点去末 25 点（占 fc 窗比例 6.7%→20%）。
- [ ] 3.3 spec update；3.4 commit；`/trellis:finish-work`。

## 回滚点

- S1–S3 代码面：git checkout 单文件回退；旧口径默认行为有 S1 验证步骤兜底。
- S5/S6 训练失败：保留 log/history 定位；**任何超参/口径回退（lr、batch、冻结面、λ）须用户
  拍板后另立 Run，不得静默改**。
- 产物误写上轮目录：立即停止写入，报告用户处置（不自行删除）。
