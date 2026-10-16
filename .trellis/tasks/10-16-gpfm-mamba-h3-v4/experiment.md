# 实验日志 — H3 v4-B 通道独立骨干 + cMLP 组件式读出变体（10-16-gpfm-mamba-h3-v4）

> 每次「跑批/训练/评估/统计」追加一个 Run 块：跑前先写「目的+配置」，跑后同会话补「结果+结论」，不留空块。
> 格式契约见 `.trellis/spec/research/experiment-log.md`；产物只落本任务声明的变体目录（`.trellis/spec/research/path-isolation.md`）。
> v4-B 输出根（已拍板）：`/root/npi/outputs/gpfm-mamba-f0/v4b-*/`（沿用 f0 根，run_id 前缀隔离）。

## Run 001 — v4b 冒烟（smoke，全链路限批）（2026-10-16）

### 目的与假设

- 验证 v4-B 变体（CI 骨干 + cMLP head）经 config+model 双遮蔽包装接入 v1 训练主流程的全链路机械正确性：数据加载/划分校验、GPR 先验、新结构前向反向、四项损失、R9 监控与 checkpoint 落盘、config_hash 写入。

### 配置

- 代码：`models/gpfm-mamba-v4/`（config.py + model.py + train.py/evaluate.py 双遮蔽包装 + tests/test_v4b.py；mamba/data/gpr_prior/losses/train 循环只读复用 v1）；git commit 29a9ab4（v4 目录未提交，工作树新增）
- 参数：`--smoke`（每类 2 例受试者共 4 例，epochs=1，seed=1234）；结构 d_ch(=d_model)=1、d_z=16、head_act=silu、CI 共享、不加 group lasso（用户 2026-10-16 拍板，design §3.1）；其余与 v2 逐字段一致；config_hash 251c7be664ea
- 环境：base conda @ AutoDL 云服务器（本地即远程机），torch 2.8.0+cu128 / mamba_ssm 2.3.1 + causal_conv1d 1.7.0（锁定），1× RTX 4080 SUPER 32GB，16 核 cgroup（BLAS 上限 8 线程，包装内置）
- 数据：`/root/autodl-tmp/edra_process_v4/`，subject_filter=status_ok（46 例），fold0、split_seed=42（校验 v1 固化 manifest，只读）
- 命令：`python3 models/gpfm-mamba-v4/train.py --smoke --run-id v4b-smoke-a`

### 结果

- 指标：全链路跑通（exit 0）。train 4 例 5,284 窗 × 1 epoch（20 step）| train total 1.3341（mae 0.6950 末步）| 内部验证（EMA）total 1.16759（新 best）| 4.2 it/s | epoch 4.75 s | peak 显存 12.697 GiB | `[model] params 64056`（与单测/design §3.2 精确一致）| 损失全程有限无 NaN
- 产物：`/root/npi/outputs/gpfm-mamba-f0/v4b-smoke-a/`（config.yaml、env_snapshot.txt、channel_mean/std.npy、logs/{train.log, metrics.jsonl, summary.json}、checkpoints/gpfm-mamba-v4b-{best,last}.pt）
- 图表：无（冒烟不出图）

### 结论

- 支持：CI 骨干 + cMLP head 新结构经双遮蔽接入 v1 训练主流程机械正确（数据划分校验/GPR 先验/前向反向/R9 监控/checkpoint/config_hash=91e44df5e142 全就位）；peak 显存 ~12.7 GiB（vs v2 smoke 同口径 1.9 GiB 量级——CI 形态 ×62 scan 批的显存代价，32 GB 卡内安全）
- 下一步：Run 002 全量 1-epoch 基准（硬门禁：显存/墙钟/it/s/四项损失走势对照 v2 + 100-epoch 外推）

## Run 002 — v4b 全量 1-epoch 基准（bench，R9 硬门禁）（2026-10-16）

### 目的与假设

- R9 硬门禁：v4-B 计算形态改变（scan 等效序列 ×62 = 15,872 条 vs v2 的 256 条、head 参数 63.5k、d_ch=1 的 kernel 效率不确定 design §9.11）——实测全量 fold0 训练 1 epoch 的 peak 显存 / epoch 墙钟 / it/s / 四项损失走势（对照 v2d62-bench-1ep 同期）/ t 分桶 sanity / 100-epoch 外推，作为 100-epoch 主 Run 启动前的用户确认门禁材料；同时按 design §9.10 判读 d_ch=1 欠拟合嫌疑（train loss 高原 vs v2 同期）。

### 配置

- 代码：同 Run 001（`models/gpfm-mamba-v4/`，commit 29a9ab4 工作树）
- 参数：`--epochs 1 --bench`（全量 fold0：train 34 例 44,914 窗，batch 256，seed=1234，AdamW lr 1e-3 wd 1e-4 cosine T_max=1，EMA 0.999，损失 {1,1,0.1,0.1}）；结构 d_ch=1/d_z=16/silu；resolved config_hash 见 run 目录 config.yaml（epochs=1 覆写口径，与 v2 bench 同款）
- 环境：同 Run 001（base @ AutoDL，RTX 4080 SUPER 32GB）
- 数据：同 Run 001（status_ok 46 例，fold0，v1 splits manifest 只读校验）
- 命令：`python3 models/gpfm-mamba-v4/train.py --epochs 1 --bench --run-id v4b-bench-1ep`

### 结果

- 指标（对照 v2d62-bench-1ep 同期，同 seed/数据/batch）：

| 项 | **v4b-bench-1ep** | v2d62-bench-1ep | 备注 |
|---|---|---|---|
| peak 显存 | **12.697 GiB** | 1.942 GiB | CI 形态 ×62 scan 批的代价；32 GB 卡余量充足 |
| epoch 墙钟 | **32.33 s**（全周期 ~36 s 含内部验证+ckpt） | 9.45 s | 3.4× 慢 |
| it/s | **5.41** | 18.53 | d_ch=1 kernel 效率（design §9.11）实测 |
| 100-epoch 外推 | **0.90 h**（train 循环口径；含验证全周期 ≈1.0 h） | 0.26 h | |
| train total | **1.34579**（mae 0.70931/cons 0.55954/tv 0.20948/corr 0.55996） | 1.22138（mae 0.65246） | v4b 高 ~10% |
| val total（内部验证，EMA） | **1.06179**（mae 0.54549） | 1.10751（mae 0.5718） | **v4b 反而优 4.1%** |
| 损失走势（step 1→171） | 1.3783→1.3247 | 1.4894→1.1103 | v4b 起点低、降得慢 |
| t 分桶 mae | 0.979→0.093 单调降 | 0.881→0.094 单调降 | sanity 通过，形状一致 |
| v_norm_mean / x̂1_std | 11.04 / 0.676 | 52.52 / 0.651 | v4b 速度范数小一个量级（head 输出尺度） |

- 产物：`/root/npi/outputs/gpfm-mamba-f0/v4b-bench-1ep/`（config.yaml、env_snapshot.txt、logs/{train.log, metrics.jsonl, summary.json}、checkpoints/gpfm-mamba-v4b-{best,last}.pt）
- 图表：无（bench 不出图）

### 结论

- 支持（可进入主 Run，待用户确认门禁）：① 显存 12.7 GiB 无 OOM 风险（余量 ~19 GiB）；② 时长 100 epoch ≈ 0.9–1.0 h 可接受（v2 的 3.4×）；③ t 分桶 sanity 通过；④ **d_ch=1 欠拟合判读：1-epoch 证据不支持欠拟合红旗**——train total 仅高 v2 约 10%（非「高原远差于」），val total 反优 4.1%（1.0618 vs 1.1075），损失走势正常下降；v_norm 偏小与 head 尺度有关，非异常。风险登记（design §9.10）：1 epoch 不足以排除后续欠拟合，主 Run 期间持续对照 v2 train loss 曲线（v2 第 45 epoch best_val 0.74167 为参照带）。
- 下一步：Run 003 eval 前向探针（评估路径显存/耗时）后，摘要汇报用户确认门禁 → 启动 §3 主 Run

## Run 003 — v4b eval 前向探针（sampler 路径单批，R9 门禁补充）（2026-10-16）

### 目的与假设

- R9 门禁评估侧：训练 bench 不覆盖评估路径——v4-B 评估元素量 ~16×（EulerSampler 每窗 K=16 条样本按 (B·K) 展平积分，CI 再 ×62 通道折批）。装载 v4 model（bench best ckpt，EMA 权重）按 v1 sample.py 的 EulerSampler/iterate_test_batches 正典路径跑一个评估 batch（B=64、K=16、euler 16 步、终点 40 Hz 低通），量峰值显存与耗时；不跑完整评估。

### 配置

- 代码：同 Run 001/002；探针脚本 = 内联 heredoc（复用 v4 train.py 包装的 `_shadow_v4()` 双遮蔽后 import v1 sample.py 路径），全文见下方命令
- 参数：batch 64、K=16（cfg.n_samples）、chunk_bk=1024（单批 B·K=1024 恰一块）、euler_steps=16、gen_lowpass_hz=40.0（正式口径）、采样 seed=cfg.seed=1234；ckpt = v4b-bench-1ep 的 best（epoch 0，config_hash 91e44df5e142 复原 epochs=1 覆写口径装载）
- 环境：同 Run 001
- 数据：测试折 iterate_test_batches（run 目录 z-score 统计重算 allclose 断言，只取首个 batch=64 窗）
- 命令：
  ```bash
  python3 - <<'EOF'   # 全文即探针（可复现）
  import os
  for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"): os.environ.setdefault(v,"8")
  import sys, time
  sys.path.insert(0, "/root/npi/models/gpfm-mamba-v4")
  import train as v4w                      # v4 包装（仅用其遮蔽函数，不跑 main）
  v4w._shadow_v4(); v4w._assert_model_config_consistency()
  import torch, config as C, sample as S
  cfg = C.build_config(epochs=1)           # bench ckpt 口径（hash 91e44df5e142）
  dev = "cuda"
  model, ema, ck = S.load_eval_model(
      "/root/npi/outputs/gpfm-mamba-f0/v4b-bench-1ep/checkpoints/gpfm-mamba-v4b-best.pt", cfg, dev)
  prior = S.build_eval_prior(cfg, dev)
  loader, test_ds = S.iterate_test_batches(cfg, f"{cfg.outputs_root}/v4b-bench-1ep", batch_size=64)
  sampler = S.EulerSampler(model, prior, euler_steps=cfg.euler_steps, sigma_min=cfg.sigma_min,
                           chunk_bk=1024, gen_lowpass_hz=cfg.gen_lowpass_hz)
  g = torch.Generator(device=dev); g.manual_seed(cfg.seed)
  torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0 = time.time()
  with ema.swap(model):
    for xp, _ in loader:                   # 只取首个 batch（64 窗 × K=16）
      s = sampler.sample(xp.to(dev), cfg.n_samples, generator=g); break
  torch.cuda.synchronize(); el = time.time() - t0
  print(f"[probe] B={s.shape[0]} K={s.shape[1]} out {tuple(s.shape)} finite={bool(torch.isfinite(s).all())}"
        f" | {el:.2f}s/batch | peak {torch.cuda.max_memory_allocated()/2**30:.3f} GiB"
        f" | test 折 {len(test_ds)} 窗 -> 全量 {(len(test_ds)+63)//64} 批")
  EOF
  ```

### 结果

- 指标：首个 batch（含 kernel 预热）3.70 s；3 批稳态 **3.31 s/批**（B=64、K=16、euler 16 步、终点 40 Hz 低通）；输出 (64,16,62,128) 全有限；**峰值显存 12.362 GiB**；test 折 13,210 窗 = 207 批 → **全量 Pass A 采样外推 ~11 min**（稳态口径；不含 GC/PSD/图）
- 产物：无落盘（只读探针，experiment-log.md「设计可行性探针 Run」口径；结论数字记本块）
- 图表：无

### 结论

- 支持（可进入主 Run，待用户确认门禁）：评估路径无 OOM 风险（12.4 GiB / 32 GB，batch 64 无需降档；如与训练并行再议）；单批 3.3 s 可接受（v1/v2 评估同批约 0.5 s 量级，v4-B 评估元素量 ~16×+CI×62 的代价）；全量评估采样段 ~11 min 可接受。
- 下一步：R9 门禁摘要（Run 002+003）汇报用户确认 → 确认后启动 §3 主 Run `v4b-main-01`（100-epoch）；预案（均口径变更须确认）：训练侧 batch 256→128 / d_z 16→8 / d_ch 1→8/16（欠拟合第一预案）；评估侧 --batch-size 64→32/16







## R9 门禁摘要（Run 001–003 汇总 · 待用户确认）

**状态：未确认——100-epoch 主 Run（v4b-main-01）在用户确认前不得启动（R9 硬门禁）。**

- 单测：`tests/test_v4b.py` 七项全 PASS（CI 隔离/因果/head 结构/前向梯度/参数量 64,056 精确/config↔model 一致+hash 稳定/双遮蔽+前缀守卫）；v1 既有 7 测回归全 PASS（v1/v2/v3 零改动，config_hash 零漂移：v1 c6ff80ac1c43 / v2 fc2a9ce14948）。
- 冒烟（Run 001）：全链路跑通，params 64,056 与 design §3.2 精确一致。
- 训练基准（Run 002）：peak **12.70 GiB**（v2 1.94）/ epoch **32.3 s**（v2 9.45，3.4×）/ 5.41 it/s / **100-epoch 外推 0.90–1.0 h** / train total 1.3458（v2 1.2214，+10%）/ **val total 1.0618（v2 1.1075，反优 4.1%）** / t 分桶单调 sanity 通过。
- 评估探针（Run 003）：B=64·K=16 稳态 **3.31 s/批**、峰值 **12.36 GiB**、全量采样段外推 **~11 min**；无需 --batch-size 降档。
- **d_ch=1 欠拟合判读**：1-epoch 证据不支持欠拟合红旗（train loss 未高原远差于 v2、val 反优）；风险登记 design §9.10，主 Run 期间持续对照 v2 曲线（v2 best_val 0.74167 参照带）。
- 预案（均属口径变更，须用户确认后方可用于主 Run）：训练侧 batch 256→128、d_z 16→8、d_ch 1→8/16（欠拟合第一预案，config+model 双改+重跑单测）；评估侧 --batch-size 64→32/16。

## 口径变更登记（2026-10-16）

**用户拍板（2026-10-16）：长期数据划分口径 train/val/test = 7/1/2；本任务整数分配 32/4/10**（取代此前 v1/v2/v3 实际执行、v4-B 曾拟的 train 34/val 2/test 10）：

- **不变**：fold0 外层 5-fold 划分（split_seed=42）与 **test 10 例**逐 ID 不动——v1/v2/v4-B 的 test 指标保持可比；受试者过滤 status_ok 46 例不变。
- **变更**：训练折 36 例内部由 train 34 / val 2 改为 **train 32 / val 4**（分层 2 HC + 2 MDD，val_subject_seed=42 不变）；整体 32/4/10 ≈ 6.96/0.87/2.17 ≈ 7/1/2。
- **实现机制**：v1 `data.py::split_manifest_path` 增 **default-off 钩子** `split_manifest_override`（getattr 透传，先例 `prior_bandlimit_hz`）——v4 config `n_val_subjects=4` + `split_manifest_override` 指向 v4 自备固化 manifest `models/gpfm-mamba-v4/splits/fold0_status_ok_seed42_val4.json`（首建写入、之后校验一致）；v1/v2/v3 config 无此字段 ⇒ manifest 路径与 config_hash **零漂移**（v1 c6ff80ac1c43 / v2 fc2a9ce14948 / v3lp 0c2de30a4868 守护断言入单测 ⑥/⑧）。
- **v4 config_hash 更新**：默认 f088186dd386（旧 251c7be664ea 作废）；epochs=1 覆写口径 323c6d49e6bf（旧 91e44df5e142 作废）。
- **不可比注记（对 v1/v2 对比时必随附）**：① v4-B 训练数据 32 例 vs v1/v2 的 34 例（−6%）；② v4 best 判据基于 val4（v1/v2 为 val2）；③ test 折相同 ⇒ test 指标可比；④ Run 002/003 的 val 数值为 val2 口径，主 Run val 轨迹以 Run 004（val4 口径）为对照基线，不与 Run 002 直接比。
- 「以后」= 长期划分口径：3.3 时回填 `.trellis/spec/research/data-pipeline.md`。

## Run 004 — v4b 1-epoch 基准更新（val4 口径 32/4/10）（2026-10-16）

### 目的与假设

- 划分口径变更（32/4/10）后的 1-epoch 基准更新：train 32 例（~42k 窗）全量 1 epoch 的显存/墙钟/it/s，以及 **val4 口径的内部验证总 loss（EMA）**——后者是主 Run（v4b-main-01，待 R9 用户确认后启动）val 轨迹的对照基线；与 Run 002（val2 口径，train 34 例）数值并列展示但**不直接可比**（数据量与 best 判据均不同）。

### 配置

- 代码：`models/gpfm-mamba-v4/`（本任务口径变更增量：v1 data.py default-off 钩子 + v4 config n_val_subjects=4/split_manifest_override + 单测 ⑥⑧ 扩展）；git commit 84e1dc3 工作树增量
- 参数：`--epochs 1 --bench`（train 32 例 / val 4 例（2HC+2MDD）/ test 10 不入训练管道；batch 256，seed=1234，AdamW lr 1e-3 wd 1e-4 cosine T_max=1，EMA 0.999，损失 {1,1,0.1,0.1}）；结构 d_ch=1/d_z=16/silu（不变）；resolved config_hash 323c6d49e6bf（epochs=1 覆写口径）
- 环境：同 Run 001（base @ AutoDL，RTX 4080 SUPER 32GB，16 核 cgroup BLAS 上限 8 线程）
- 数据：`/root/autodl-tmp/edra_process_v4/`，status_ok 46 例，fold0 split_seed=42 + val_subject_seed=42（**校验 v4 自备固化 manifest** `models/gpfm-mamba-v4/splits/fold0_status_ok_seed42_val4.json`；test 与 v1 逐 ID 一致断言见单测 ⑧）
- 命令：`python3 models/gpfm-mamba-v4/train.py --epochs 1 --bench --run-id v4b-bench-1ep-val4`

### 结果

- 指标（对照 Run 002——**val 列口径不同不直接可比**：Run 002 为 val2（train 34 例/171 step），本 Run 为 val4（train 32 例/165 step）；epoch 墙钟差异主要来自窗数 44,914→42,272）：

| 项 | **Run 004 v4b-bench-1ep-val4**（32/4/10） | Run 002 v4b-bench-1ep（34/2/10，val2 口径） | 备注 |
|---|---|---|---|
| train 受试者/窗 | **32 例 42,272 窗** | 34 例 44,914 窗 | 训练数据 −6%（不可比注记①） |
| val 受试者/窗 | **4 例（2HC+2MDD）5,284 窗** | 2 例 | **主 Run val 轨迹对照基线** |
| peak 显存 | **12.697 GiB** | 12.697 GiB | 同（batch 形态不变） |
| epoch 墙钟 | **30.59 s** | 32.33 s | 窗数减少所致 |
| it/s | **5.39** | 5.41 | 持平 |
| 100-epoch 外推 | **0.85 h**（train 循环） | 0.90 h | |
| train total | **1.34453**（mae 0.70863/cons 0.55924/tv 0.20942/corr 0.55721） | 1.34579（mae 0.70931） | 逐位近同——d_ch=1 无欠拟合红旗 |
| **val total（EMA）** | **1.10607**（mae 0.55871/cons 0.49036/tv 0.18368/corr 0.38621；x̂1 std 0.7152 vs x1 std 0.8533） | 1.06179（mae 0.54549） | **不直接可比**（val 集不同：4 例含新增 37/49） |
| 损失走势（trend） | 1.3831→1.2902（step 1→161） | 1.3783→1.3247（step 1→171） | 正常下降 |
| t 分桶 mae | 0.966→0.131 单调降 | 0.979→0.093 单调降 | sanity 通过 |
| v_norm_mean / x̂1_std | 13.52 / 0.6728 | 11.04 / 0.676 | 正常量级 |

- 损失全程有限无 NaN；best（=唯一 epoch，EMA val total 1.10607）/last 就位
- 产物：`/root/npi/outputs/gpfm-mamba-f0/v4b-bench-1ep-val4/`（config.yaml（resolved，hash 323c6d49e6bf）、env_snapshot.txt、channel_mean/std.npy（train-32 口径）、logs/{train.log, metrics.jsonl, summary.json}、checkpoints/gpfm-mamba-v4b-{best,last}.pt）
- 图表：无（bench 不出图）

### 结论

- 支持（口径切换机械正确 + 主 Run 基线就位）：① 32/4/10 划分经训练主流程实跑验证（log 载「train 32 例 42272 窗 | val 4 例 5284 窗」，v4 自备 manifest 校验通过）；② 显存/时长与 Run 002 持平（12.7 GiB / 30.6 s / 0.85 h 外推），资源结论不变；③ **val4 口径 1-epoch val 总 loss（EMA）= 1.10607 登记为主 Run（v4b-main-01）val 轨迹的对照基线**——与 Run 002 的 1.06179（val2 口径）不直接可比；④ d_ch=1 欠拟合判读：train total 1.34453 与 Run 002 的 1.34579 逐位近同，无欠拟合红旗。
- 下一步：R9 硬门禁仍未确认——**100-epoch 主 Run（v4b-main-01）继续等待用户确认后方可启动**（Run 002/003/004 为门禁材料全集）。

## 早停口径变更登记（2026-10-16）

**用户拍板（2026-10-16）：v4-B 训练加入早停，patience=10 epoch**（v1/v2 为固定 100 epoch 无早停——本变更为口径变更，非 bug 修正）：

- **判定基准 = 既有 best 判据**：val4 总 loss（固定 t 网格 {0.1,0.3,0.5,0.7,0.9} + 固定 ε 确定性复算、EMA 权重）**连续 10 个 epoch 无新 best 即停止训练**（恰好第 10 轮无改善触发）。
- **实现机制**：v1 `train.py` **default-off 钩子**（先例 `prior_bandlimit_hz` / `split_manifest_override`）——模块级纯函数 `should_early_stop`（patience<=0 恒 False=禁用语义）+ epoch 循环每 epoch 末 `getattr(cfg, "early_stop_patience", 0)`；新 best 计数清零否则 +1；触发时 break **位于每 epoch 收尾动作（best/last checkpoint + metrics.jsonl 落盘）之后**（best/last checkpoint 语义不变），触发行附 `"early_stop": true`；resume 兼容 = `replay_epochs_since_best` 从截齐后的 metrics.jsonl 回放计数（checkpoint 不扩字段，旧 ckpt 兼容，断点续传后计数连续）。v1/v2/v3 config 无此字段 ⇒ patience=0 ⇒ 恒不触发，**行为零漂移**（每 epoch 仅多一次 getattr 与整数比较）。
- **v4 config**：`early_stop_patience: int = 10`；**config_hash 更新**：默认 `26cd2bb2e43f`（f088186dd386 作废）、epochs=1 覆写口径 `f42570df2704`（323c6d49e6bf 作废——Run 004 已落盘产物按其记录 hash 不受影响，此后 bench 复跑用新值）。单测 pin 同步（test_v4b ⑥/⑨）。
- **不可比注记（对 v1/v2 对比时必随附）**：① v4-B 可能提前结束（best 仍在停止前最优点，best/last checkpoint 语义不变）；② cosine T_max=100 不随早停调整——提前停止时 lr 未退火到底，属早停+cosine 固有交互，**登记不修正**；③ epochs_run < 100 时与 v1/v2 的「训练长度」不同构，epoch 数轴对比须注明；④ 早停不改变每 epoch 成本，只缩短总时长——Run 004 的 0.85 h 外推变为**上界**，无需重跑 bench。
- **主 Run 门禁不变**：R9 硬门禁仍待用户确认，100-epoch 主 Run（v4b-main-01）未启动。

## Run 005 — v4b 早停钩子集成冒烟（无触发/触发/resume 回放三路径）（2026-10-16）

### 目的与假设

- 验证早停口径变更（上块）的 v1 train.py default-off 钩子在 v4 双遮蔽真实训练主流程中的三条路径：① 无触发（计数不足时零副作用）；② 触发（连续 patience 轮无新 best → break 前产物完整落盘 + metrics 行附 `early_stop` 字段）；③ resume 回放（断点续传后早停计数连续）。patience 采用运行时临时覆盖（2）以缩短触发窗口——**v4 config.py 文件不动（拍板值仍 10）**。

### 配置

- 代码：`models/gpfm-mamba-v1/train.py`（早停钩子增量）+ `models/gpfm-mamba-v4/config.py`（early_stop_patience=10）+ tests/test_v4b.py ⑨；git commit ed80e3e 工作树增量
- 参数：三子跑均临时 patience=2（驱动脚本运行时覆盖 `C.build_config`，机制为「构建后 setattr」，v4 config.py 文件值 10 不动）——① `--smoke --epochs 13 --run-id v4b-smoke-es`（train 4 例）；② `--epochs 15 --limit-train-subjects 1 --run-id v4b-smoke-es2`（train 2 例，构造易过拟合小集）；③ resume 确定性夹具 `--resume .../v4b-smoke-es3/checkpoints/gpfm-mamba-v4b-last.pt --epochs 15 --limit-train-subjects 1`（es2 run 目录副本：ckpt 回拨 epoch 10 + best_val 抬高 0.5 不可逾越 + metrics epoch-10 行 best 翻 false → 回放计数应为 1；等价构造「kill 于无改善断点」的真实场景）
- 环境：同 Run 001（base @ AutoDL，RTX 4080 SUPER 32GB）
- 数据：同 Run 001（status_ok 46 例 fold0；val 均为 val4 固定 4 例 5,284 窗确定性复算）

### 结果

- ① es（13 epoch 上限）：每 epoch 均新 best（EMA 平滑下 val 单调改善 1.1676→1.1970 量级）→ **未触发**，13/13 跑满、正常收尾；全部 metrics 行**无** `early_stop` 字段（无触发=零副作用，v1/v2/v3 同构路径零漂移佐证）；config_hash 5f90cced0284（临时口径）。
- ② es2（15 epoch 上限，train 2 例）：仍未触发（val 1.2261→1.2203 单调）——小集 15 epoch 内 EMA 仍持续改善，有机触发不可得，转 ③ 确定性夹具；两跑共同佐证无触发路径 metrics/checkpoint/summary 与旧代码行为一致。
- ③ es3（resume 夹具）：`[resume] epoch 10 -> 11 | best_val 0.5000 | metrics.jsonl 截齐至 epoch 10` → 首个续跑 epoch（显示 epoch 12/15）val 1.2199 > 0.5 非新 best → 计数 1→2 达 patience → **触发**：`[early-stop] epoch 12/15 触发早停：连续 2 个 epoch 无新 best（patience 2）| best epoch 10 | best val total 0.5000 | metrics.jsonl 本 epoch 行已附 "early_stop": true`；**break 前产物核验**：last.pt 已写（epoch 11/best_val 0.5，mtime 晚于 metrics 行）、非新 best 故 best.pt 未覆写、metrics.jsonl 末行（epoch 11）恰含 `"early_stop": true` 且此前各行无该字段。
- 产物：`/root/npi/outputs/gpfm-mamba-f0/v4b-smoke-{es,es2,es3}/`（各含 config.yaml/logs/checkpoints；es3 的 metrics.jsonl 12 行=截齐 11 行+触发行）
- 图表：无（冒烟不出图）

### 结论

- 支持：早停钩子三路径（无触发零副作用/触发时产物先落盘后 break/resume 回放计数连续）在 v4 真实训练主流程验证通过；配合单测 ⑨（纯函数边界 0/负恒 False、恰第 patience 轮触发、循环计数模拟、v1/v2/v3 无字段 getattr→0）与 v1 既有 7 测回归 + config_hash pin（v1 c6ff80ac1c43 / v2 fc2a9ce14948 / v3lp 0c2de30a4868 零漂移），口径变更落定。
- 下一步：R9 硬门禁确认后启动 `v4b-main-01`（100-epoch 上限 + early_stop_patience=10；提前停止时 0.85 h 为上界）。

## lr 调度口径变更登记（2026-10-16）

**用户拍板（2026-10-16）：v4-B 取消 cosine 退火，lr 恒定 1e-3**（v1/v2 为 cosine T_max=epochs——口径变更，非 bug 修正；取代早停块中「cosine T_max=100 不随早停调整」注记，见下）：

- **变更内容**：v4-B 优化器 lr 全程恒定 `lr=1e-3` 不退火（AdamW/wd=1e-4/batch/clip 等其余训练超参不变）；v1/v2 维持 cosine（T_max=epochs）。
- **实现机制**：v1 `train.py` **default-off 钩子**（先例 `prior_bandlimit_hz` / `split_manifest_override` / `early_stop_patience`）——模块级纯函数 `build_lr_scheduler(opt, cfg)`（`getattr(cfg, "lr_schedule", "cosine")` 透传）："cosine" = CosineAnnealingLR(T_max=cfg.epochs)，与原内联构建逐位同构；"constant" = LambdaLR 恒 1.0（lr 恒 = cfg.lr；**scheduler 对象保留**——checkpoint 存 scheduler 状态的既有契约不变，LambdaLR 对普通函数 lambda 的 state_dict 存 None、load_state_dict round-trip 正常，`--resume` 全量恢复语义不变）；未知值 ValueError（构建期暴露拼写错误）。v1/v2/v3 config 无此字段 ⇒ 默认 cosine ⇒ 构建行为**零漂移**；日志 = `[model]` 行 cosine 口径逐字节不变（constant 口径下该行片段改打 `lr_schedule constant`，不出现失实的 cosine 字样），**仅非 cosine 时多打印一行 `[sched] lr_schedule=constant`**（v1/v2/v3 日志零漂移）。训练循环 lr 记录来源 = `sched.get_last_lr()`（每 step 进度行 + metrics.jsonl `lr` 字段）——constant 口径下恒为 1e-3（单测 ⑩ 双检 get_last_lr 与 opt.param_groups 同值）。
- **v4 config**：`lr_schedule: str = "constant"`（lr=1e-3 不变）；**config_hash 更新**：默认 **`a965f24e580a`**（26cd2bb2e43f 作废）、epochs=1 覆写口径 `275f6c8fe93b`（f42570df2704 作废——此前已落盘产物按各自记录 hash 不受影响，此后 bench 复跑用新值）。单测 pin 同步（test_v4b ⑥），v1/v2/v3 hash 逐位不变（c6ff80ac1c43 / fc2a9ce14948 / 0c2de30a4868，⑥ 子进程断言）。
- **早停交互注记作废**：早停登记块的「cosine T_max=100 不随早停调整——提前停止时 lr 未退火到底，属早停+cosine 固有交互，登记不修正」**随本变更作废**：lr 全程恒定，早停不再有 lr 未退火交互。
- **不可比注记（对 v1/v2 对比时必随附）**：v4-B 优化轨迹与 v1/v2 不同构（恒定 lr vs cosine 退火到 ~0——后期步长更大，收敛动力学不同），与 32/4 划分、早停 patience=10 一并列入 H3 判读的口径差异清单（prd R3 三条口径变更登记）。
- **验证**：test_v4b 十项全 PASS（⑩ 新增：constant 10 步 lr 仍精确 1e-3 + state_dict round-trip + cosine T_max=epochs 严格下降 + 未知值 ValueError + v1/v2/v3 无字段 getattr→cosine）；v1 既有 7 测回归全 PASS。资源口径不变（调度钩子零计算开销，Run 004 的 0.85 h 外推仍为上界）。
- **主 Run**：本变更落地后由主会话启动 `v4b-main-01`（100-epoch 上限 + early_stop_patience=10 + **lr 恒定 1e-3**）；R9 门禁材料（Run 002/003/004）不受影响，无需重跑 bench（调度钩子不改每 epoch 计算形态）。

## Run 006 — v4b 主 Run：fold0 完整训练（100ep 上限 + 早停 patience=10 + 恒定 lr）（2026-10-16）

### 目的与配置（跑前登记）

- 目的：v4-B（CI 骨干 d_ch=1 + cMLP head）fold0 单 seed 完整训练，产出 H3 判读与三检验/5-fold 候选评估对象。
- 代码：commit `896fbb6`（models/gpfm-mamba-v4 + v1 三处 default-off 钩子）。
- 配置：d_ch=1/d_z=16/SiLU/CI 共享/无 lasso；划分 **32/4/10**（val4=25,37HC+40,49MDD，test=v1 fold0 同 10 例）；seed 1234（划分 42/val ε 2026/评估 1235）；batch 256；**lr 恒定 1e-3（无 cosine，用户拍板）**；EMA 0.999；损失 {1,1,0.1,0.1}；best=val4 总 loss（固定 t 网格 EMA 复算）；**早停 patience=10**；epochs 上限 100；config_hash `a965f24e580a`。
- 口径差异清单（vs v1/v2）：CI+cMLP 结构 / 32 例训练（vs 34）/ val4 判据（vs val2）/ 早停+恒定 lr（vs 固定 100ep+cosine）——判读时并列陈述。
- 命令：`mkdir -p outputs/gpfm-mamba-f0/v4b-main-01/logs && python3 models/gpfm-mamba-v4/train.py --run-id v4b-main-01`（后台，日志 logs/train.log）。

### 结果（2026-10-16 训练完成）

- **早停触发**：epoch 27/100 停止（连续 10 epoch 无新 best，patience=10；触发行附 `"early_stop": true`，先落盘后 break）——**best epoch 17**（metrics 0 起始 epoch 16）| **best val total 1.04405**（val4 口径，EMA，固定 t 网格确定性复算）。
- 曲线形态：val total 自 ep16 后缓慢单调爬升（1.04405 → 1.04808，+0.4%），train total 同期 1.3076 → 1.3066 持平——**欠拟合无红旗亦无急速过拟合**（d_ch=1 风险 §9.10 未兑现为 train loss 高原）；train/val gap（总 loss）best 处 0.264、末期 0.259，全程稳定。
- 资源：**peak 12.697 GiB** | **29.77 s/ep 均值**（27 epoch）| 5.54 it/s | 全程无 NaN | lr 恒 1e-3（metrics `lr` 字段逐 epoch 核验）。
- 产物：`/root/npi/outputs/gpfm-mamba-f0/v4b-main-01/`（config.yaml（resolved，hash `a965f24e580a`）、env_snapshot.txt、channel_mean/std.npy、logs/{train.log, metrics.jsonl（27 行）, summary.json}、checkpoints/gpfm-mamba-v4b-{best,last}.pt（best=epoch 16/0 起始，best_val 1.04405））。
- 图表：无（训练不出图；训练曲线图随 Run 007 evaluate 产出）。

### 结论

- 支持（主 Run 完成口径内目标）：v4-B fold0 单 seed 训练完整跑通且早停钩子按拍板口径首次实弹触发（27 epoch 即停 vs v1/v2 固定 100 epoch——训练长度不同构，不可比注记③）；best val 1.04405（val4 口径，与 v1/v2 的 val2 best 0.7417/0.7741 带不可直接比——判据集合与数据量均不同）。
- 观察项登记：val 曲线 27 epoch 内近乎走平（1.044 量级平台）+ train 同步走平 ⇒ d_ch=1 容量疑虑（design §9.10）呈「温和欠拟合」迹象而非红旗——判读材料随附；正式口径评估（Run 007）的五维度实测为最终证据。
- 下一步：Run 007 正式口径评估（Run 017 同口径）+ 指标 1 v2.1 + readout 诊断副产品。

## Run 007 — v4b-main-01 正式口径评估（Run 017 同口径）+ readout 诊断（2026-10-16）

### 目的与配置（跑前登记）

- 目的：v4b-main-01（best ckpt，EMA 权重）按 **Run 017 完全同口径**正式评估——七维度（test-MAE/CRPS/逐通道 r/R1/指标 1 FC/GC 一致性/GC 幅值 + 空间/低频 PSD）与 v1/v2（Run 017 正式口径数值）并列；三检验核算材料（指标 1 v2.1 / 指标 3 分频段 |Δlog10| / 指标 2 轨迹图备好交用户目视）；R1 附分子分母；std_t 幅度照登；**readout 诊断副产品 M[j,i]=‖W^{(i),1}_{:,j}‖_F 热图 + 与外部 GC 矩阵 Spearman（仅诊断）**。
- 代码：同 Run 006（commit `896fbb6` 工作树）；评估链只读复用 v1（evaluate/gc_psd/fc_metric/sample/visualizer 经双遮蔽）。
- 配置/命令（默认参数即正式口径，与 Run 017 对 v2 的重评逐项一致）：
  - `python3 models/gpfm-mamba-v4/evaluate.py --run-id v4b-main-01`（Pass A K=16 seed=1234、Pass B K=1 seed=1235、GC p_max=10 bootstrap 1000 受试者级 seed=2026、PSD nperseg=512、空间 σ=0.080、**gen_lowpass_hz=40.0 正式口径**（config 显式，无命令行覆写）；评估侧 log/eval_metrics.jsonl 留存）
  - `python3 models/gpfm-mamba-v1/fc_metric.py --run-id v4b-main-01`（指标 1 v2.1）
- ckpt：`checkpoints/gpfm-mamba-v4b-best.pt`（epoch 16/0 起始，best_val 1.04405，config_hash `a965f24e580a` 校验）。
- 环境/数据：同 Run 001（base @ AutoDL，RTX 4080 SUPER 32GB；test 折 = v1 fold0 同 10 例——**test 指标与 v1/v2 可比**）。

### 结果（2026-10-16 评估完成）

- 评估跑通（evaluate ~12 min：Pass A 207 批 + Pass B + GC/PSD/空间 + bootstrap 1000；fc_metric 3.5 s）；无命令行参数偏离（默认参数即正式口径：Pass A K=16 seed=1234、GC 单样本 seed=1235、BIC p\*=4（真实组选定两组共用）、bootstrap 受试者级 1000 seed=2026、gen_lowpass 40 Hz）。

**七维度并列（v1/v2 = Run 017 正式口径；v4b = 本 Run；test 折三方相同）**

| 量 | v1 main-01 | v2 v2d62-main-01 | **v4b v4b-main-01** | 备注 |
|---|---|---|---|---|
| test-MAE（主指标） | 0.70834 | **0.70096** | 0.71663 | v4b +1.1%/+2.2% vs v1/v2；GPR 均值基线 0.69297（三方均未超，v1/v2 同定位） |
| CRPS | 0.52629 | **0.52496** | 0.53461 | v4b +1.6% vs v1 |
| 逐通道 r | 0.14371 | **0.15053** | 0.11479 | **v4b 塌缩 −20%/−24%** vs v1/v2 |
| R1 触发 | 未触发（1.2432） | 触发（1.5261，口径效应） | **未触发（1.0386）** | 分子分母见下 |
| 指标 1 FC r（v2.1） | 0.95856 | 0.97625 | **0.98510（三方最优）** | ρ 0.9844；t=181.9 p=2.3e-17（df=9） |
| GC r（补充参考） | **0.27549** | 0.12272 | 0.13343 | ρ 0.0620；**未回升至 v1 带**（0.2+），≈ v2 水平 +0.011；bootstrap r CI [0.027, 0.202] |
| GC 幅值 gen/real（max 比） | 92% | 89% | **3.9%**（0.00119/0.03051） | **生成侧有向动态绝对量级塌缩 ~25×**；显著边 gen 3705/3782、Jaccard 0.980 |
| 空间 R̂~R / Fro | 0.953 / 10.79 | 0.970 / **6.71** | **0.984** / 9.89 | r 三方最优、Fro 劣于 v2；R̂~K_s 0.893 |
| 低频 δ（指标 3） | 0.091 | 0.122 | 0.131 | θ 0.296/0.313/**0.244**；α 0.147/0.137/**0.391**；β 0.344/0.381/**0.202**；δ–β(1–30) 整段 **0.233** |

- 指标 3 高频照登：30–60 Hz **2.295**（v1/v2 2.36/2.38）；60–250 Hz **9.594**（9.74/9.76）——量级与 v1/v2 相同（真实侧数值地板主导）；分频段口径 = 全通道平均谱 |Δlog10|（bin [lo,hi)），本会话代码对 v1/v2 复算逐段复现 Run 017 登记值（±0.04 内）。**gen 峰频率 1.95 Hz vs real 0.98 Hz**（v1/v2 均为 0.98=0.98）——v4b 特有偏移。
- **R1 分子分母（附 v2 对照）**：分子 = 采样终点 test-MAE **0.71663**；分母 = 训练末期（ep 22–26/0 起始）5-epoch 均值 x̂1-MAE **0.690**；比值 **1.0386 < 1.5 未触发**（高/中 t 桶比 0.3396 亦未触发）。v2 对照：0.70096/0.45931=1.5261 触发——按 R1 解读规则（evaluation-calibers.md），v2 触发属「训练折拟合加深使分母收缩」的口径效应；v4b 分母 0.690 未收缩（train 不深拟合），未触发主因在分母侧，非终点劣化。
- **std_t 幅度**：生成 **0.73684** vs 真实 **0.79459**（本 run z-score 口径）→ 比值 **92.7%（轻微欠冲 −7.3%）**——与 v1/v2 的**过冲**（0.97183/0.78808=+23%、1.01275/0.78808=+29%）方向相反。注记：真实侧 std 逐 run 略异（0.7946 vs 0.7881）因 z-score 统计随训练折（32 vs 34 例）不同，比值（生成/各自真实）为可比口径。样本间展宽 0.81372（v1 0.93065/v2 0.96344）。
- **readout 诊断副产品（仅诊断，R6）**：从 best ckpt EMA shadow 的 W1(62 i,16 d_z,62 j·d_ch=1) 按 (i,d_z,j,d_ch) 分块 Frobenius 范数（d_ch=1 双路径 allclose 断言通过）得 **M[j,i]（行=源、列=靶）**：mean 0.311，对角（自源）0.460 > 非对角 0.308。与外部 GC 矩阵展平相关（**主口径 = 去对角 3782 有向边**，与 GC 一致性口径同构；全 62×62 辅口径含 M 对角自项、GC 对角恒 0）：
  - M ~ GC(gen)：Spearman **0.23981** / Pearson 0.38797（全 62×62 Spearman 0.20543）
  - M ~ GC(real)：Spearman **0.04927** / Pearson 0.11165（全 62×62 0.02180）
  - 解读：head 内部有向读出与**模型自身生成动态**的 GC 结构中等一致（0.24），与**真实 GC** 几乎不相关（0.05）——显式混合层确实学到了*某种*有向结构，但该结构与真实有向动态不对应。
- 产物（全部落 `/root/npi/outputs/gpfm-mamba-f0/v4b-main-01/`）：eval/{summary.json, gc_psd_results.json, fc_metric.json, fc_matrices.npz, gc_input.npz, gc_real.npy, gc_gen.npy, gc_F_p_*.npz, psd.npz, spatial_corr.npz, readout_M.npy, readout_diag.json}；figs/{eval_trajectories, eval_training_curves, eval_gc_matrix, eval_psd, eval_spatial_corr, eval_readout_matrix}_v4b-main-01.png；logs/{eval.log, eval_metrics.jsonl（8 stage 齐全）}。readout 计算脚本：`models/gpfm-mamba-v4/readout_diag.py`（本任务新增，仅诊断、不改 v1/common）。
- 图表：指标 2 素材 = `figs/eval_trajectories_v4b-main-01.png`（medoid + 90% PI、K=16、lowpass 40 Hz 注记）——**交用户目视判定**。

### 结论

- **七维度初判（vs v1/v2，正式口径同 test 折）**：v4b 两项最优（指标 1 FC r 0.9851、空间 R̂~R 0.984）+ R1 安静；三项轻微劣化（MAE +1.1% vs v1、CRPS +1.6%、）；GC r 0.1334 **未回升**（≈ v2 水平 +0.011，未达 0.18 部分支持线，更未达 v1 带 0.2+）；**逐通道 r 0.1148 塌缩 −20~24%** 与 **GC 幅值 3.9%（vs 92%/89%）**两项显著劣化。
- **H3 三态判读建议（判定权在用户）**：按预注册阈值，GC r 落「**不支持**」侧（≈ v2 水平 ⇒ GC 偏弱主因不在结构侧）——但**不对称守护生效**：不回升 ≠ 结构无关被证伪，且 v4b 存在强竞争解释——**d_ch=1 trunk 容量塌缩**（design §9.10）：证据链 = GC 幅值崩塌 25× + gen 峰频移位（1.95 vs 0.98 Hz）+ α 段 |Δlog10| 0.391（v1/v2 ~0.14）+ 逐通道 r 塌缩 + std_t 欠冲方向反转 + 训练曲线 27 ep 走平（d_ch=1 表达容量上限）。readout 诊断（M~GCgen 0.24 vs M~GCreal 0.05）佐证：显式混合层结构上可表达且可辨识，但 trunk 供给的逐通道表征（每通道 1 标量/时间步）不足以承载真实有向动态。
- **过拟合竞争解释检查（design §1 预登记）**：不成立——best epoch 17 ≪ 100 但 val 曲线走平非恶化（1.04405→1.04808，+0.4%）、train/val gap 稳定（~0.26）、参数/样本比 1.4:1 低；「五维度不塌缩」条件的失败更可能由 trunk 容量（欠拟合侧）驱动而非过拟合。
- **口径差异清单（对 v1/v2 对比必随附）**：① CI trunk + cMLP head 两处变化（非单变量，归因局限——分解变体「CI trunk+原 head」「共享 trunk+cMLP head」为条件后续）；② 训练数据 32 例 vs 34 例（−6%）；③ best 判据 val4 vs val2；④ 早停 27 ep + 恒定 lr 1e-3 vs 固定 100 ep + cosine（训练长度与优化轨迹不同构）。
- 三检验核算材料齐备（指标 1 数值过线 / 指标 3 低频全段 <0.5 + 高频照登 / 指标 2 图备好）——**判定留用户**；H3 三态判读建议如上，**判定留用户**。
- 下一步：§5 H3 判读收尾（用户判定留痕）→ trellis-check 全量质量检查 → 3.3 spec 更新与 progress.md 同步。

## d_ch 口径变更登记（2026-10-16 方案 B）

**用户拍板（2026-10-16）：d_ch 1→8（方案 B），batch 256 保持**

- **处置背景（Run 006/007 结论）**：d_ch=1 口径下 H3 不可判定——评估显示 GC 幅值塌缩（gen/real max 比 **3.9%** vs v1/v2 92%/89%，绝对量级 ~25×）、gen 峰频偏移（1.95 vs 0.98 Hz）、逐通道 r 塌缩（0.1148，−20~24%）、α 段 |Δlog10| 0.391（v1/v2 ~0.14）——证据链指向 design §9.10 预登记的「**d_ch=1 trunk 容量塌缩**」强竞争解释，H3 结构假设无法在此口径下裁决。按 R9 预案路径（design §3.3：d_ch 升档 1→8/16 为欠拟合第一预案）走口径变更流程。
- **变更内容**：d_ch(=config 字段 `d_model`) 1→8；batch 256、d_z=16、SiLU、CI 共享、无 group lasso 及全部训练/评估/划分/早停/lr 口径**不变**。
- **显存探针（主会话实测，2026-10-16）**：d_ch=8 @ batch 256 训练侧单步 fwd+bwd 峰值 **24.01 GiB**（32 GB 卡可行，余量 ~8 GiB）；v4 单测 ④ GPU 冒烟同口径复测 24.88 GiB（测试进程含先前断言驻留张量，量级一致）。
- **参数量**：64,056 → **499,456**（7.8×；明细 seg 2 + in_proj 152 + trunk 4×1,304=5,216 + head 前 RMSNorm(8) 8 + head 62×7,969=494,078；head 占 98.9%）；参数/训练窗（42,272）≈ **11.8:1**——过拟合重新列为观察项（design §1 竞争解释），判读材料随附 train/val gap 与 best epoch。
- **实现流程（design §4.3 双处一致性防线明文流程）**：`config.py` 与 `model.py` 类默认**双处同改** + `test_v4b.py` ⑤ 参数断言（499,456 ±1% 明细逐项）与 ⑥ hash pin 更新 + 全部单测重跑（十项 PASS）+ v1 既有 7 测回归（全 PASS，v1/v2/v3 config_hash 逐位不变：c6ff80ac1c43 / fc2a9ce14948 / 0c2de30a4868）。
- **config_hash 更新**：默认 **`91dfd211dbc3`**（a965f24e580a 作废）；epochs=1 覆写口径 **`02f0fb271e0f`**（275f6c8fe93b 作废）。此前已落盘 run（Run 001–007）按各自记录 hash 不受影响；**Run 006/007（d_ch=1）产物原样保留，作为 d_ch=1 口径的 H3 判读材料存档**。
- **readout 诊断兼容**：readout_diag.py 的 d_ch 由 ckpt 形状推断（k_head//62），d_ch=8 主路径（(i,d_z,j,d_ch) 分块 Frobenius 范数）自动生效；d_ch=1 专用双路径 allclose 断言不触发，无碍。
- **不可比注记**：① d_ch=8 的 Run 008+ 与 Run 004–007（d_ch=1）不直接可比；② 对 v1/v2 对比的口径差异清单在原四条（CI+cMLP 结构 / 32 例训练 / val4 判据 / 早停+恒定 lr）上增加第五条——**d_ch=8 仍低于 v1 的 d_model=16**（若 GC 仍不回升，容量解释未穷尽，判读时随附）。

## Run 008 — v4b d_ch=8 口径 1-epoch 基准（bench，R9 硬门禁复跑）（2026-10-16）

### 目的与配置（跑前登记）

- 目的：d_ch 1→8 口径变更（上块）后的 1-epoch 基准复跑——peak 显存（预期 ≈24 GiB，对照探针；**若 OOM 立即停止并汇报**）/ epoch 墙钟 / it/s / 四项损失走势 / 100-epoch 外推，作为新口径主 Run 启动前（待用户确认）的门禁材料；对照 d_ch=1 的 Run 004（12.70 GiB / 30.6 s/ep / 5.39 it/s / 外推 0.85 h）。
- 代码：`models/gpfm-mamba-v4/`（d_ch=8 双处同改增量：config.py + model.py + tests/test_v4b.py）；git commit ad1bb64 工作树增量。
- 参数：`--epochs 1 --bench`（train 32 / val 4 / test 10，batch 256，seed=1234，AdamW lr 1e-3 wd 1e-4 **恒定**（无 cosine），EMA 0.999，损失 {1,1,0.1,0.1}，早停 patience=10（1 epoch 不触发））；结构 **d_ch=8**/d_z=16/silu；resolved config_hash `02f0fb271e0f`（epochs=1 覆写口径）。
- 环境：同 Run 001（base @ AutoDL，RTX 4080 SUPER 32 GB，16 核 cgroup BLAS 上限 8 线程）。
- 数据：同 Run 004（status_ok 46 例 fold0 + v4 自备 val4 固化 manifest 只读校验）。
- 命令：`python3 models/gpfm-mamba-v4/train.py --epochs 1 --bench --run-id v4b-d8-bench-1ep`

### 结果（2026-10-16 跑完，exit 0 无 OOM）

- 指标（对照 Run 004 = d_ch=1 同口径 bench（32/4/10、batch 256、恒定 lr、EMA、val4 判据），同 seed/数据）：

| 项 | **Run 008 d_ch=8** | Run 004 d_ch=1 | 备注 |
|---|---|---|---|
| peak 显存 | **24.042 GiB** | 12.697 GiB | 与主会话探针 24.01 GiB 一致；32 GB 卡余量 ~8.6 GiB，无 OOM |
| epoch 墙钟 | **119.8 s**（train 循环） | 30.59 s | 3.9× 慢（×62 scan 批 × d_ch=8 激活） |
| it/s | **1.38** | 5.39 | |
| 100-epoch 外推 | **3.33 h**（train 循环；早停口径下为上界） | 0.85 h | |
| train total（epoch 均值） | **1.27398**（mae 0.65948/cons 0.54338/tv 0.18615/corr 0.525） | 1.34453（mae 0.70863） | **d_ch=8 同期拟合优 5.2%**（容量提升方向正确） |
| val total（EMA，val4） | **1.09372**（mae 0.55154/cons 0.48610/tv 0.17848/corr 0.38235） | 1.10607（mae 0.55871） | 1 epoch 即优 1.1% |
| 损失走势 | 1.3912→1.2270（step 1→165） | 1.3831→1.2902（step 1→161） | 下降更快且更深 |
| t 分桶 mae | 0.86853→0.12937 单调降 | 0.966→0.131 单调降 | sanity 通过，形状一致 |
| v_norm_mean / x̂1_std(EMA) | 42.19 / 0.70775 | 13.52 / 0.7152 | v_norm 量级回到 v2（52.52）邻域——trunk 供给恢复 |

- `[model] params 499456`（与单测/口径变更登记精确一致）；全程有限无 NaN；best（=唯一 epoch，EMA val total 1.09372）/last 就位；lr 恒 1e-3 核验。
- 产物：`/root/npi/outputs/gpfm-mamba-f0/v4b-d8-bench-1ep/`（config.yaml（resolved，hash `02f0fb271e0f`，d_model: 8）、env_snapshot.txt、channel_mean/std.npy、logs/{train.log, metrics.jsonl, summary.json}、checkpoints/gpfm-mamba-v4b-{best,last}.pt）。
- 图表：无（bench 不出图）。

### 结论

- 支持（d_ch=8 口径机械正确 + 资源可行，可进入主 Run，**待用户确认门禁**）：① 显存 24.04 GiB 与探针一致，32 GB 卡内稳定运行无 OOM；② 时长上界 3.33 h（100 epoch 满跑口径；早停 patience=10 下为上界，参照 d_ch=1 主 Run 27 epoch 即停则可能显著缩短）；③ t 分桶 sanity 通过；④ 1-epoch 拟合能力证据积极——train/val 同期均优于 d_ch=1（−5.2%/−1.1%），v_norm 量级恢复，与「d_ch=1 容量不足」的 Run 006/007 竞争解释处置方向一致。
- 观察项预登记：参数/训练窗 11.8:1（过拟合观察项重启）——主 Run 监控 train/val gap 与 best epoch（design §1）；d_ch=8 仍低于 v1 的 d_model=16，判读时随附。
- 下一步：摘要汇报用户确认 → 确认后启动 d_ch=8 主 Run（run_id `v4b-d8-main-01`，100-epoch 上限 + 早停 patience=10 + 恒定 lr；**主会话接续，本实现会话不启动**）；预案（均口径变更须确认）：显存侧 batch 256→128（若主 Run 期间与评估并行触发紧张）、时长侧接受或降 d_z。

## Run 009 — v4b-d8 主 Run：d_ch=8 fold0 完整训练（100ep 上限 + 早停 10 + 恒定 lr）（2026-10-16）

### 目的与配置（跑前登记）

- 目的：d_ch=8 重训 v4-B（方案 B，用户拍板），检验容量恢复后 CI+cMLP 结构下 GC 一致性是否回升（H3 主判据）；产出三检验与 5-fold 候选评估对象。
- 代码：commit（本块上方最近一次 feat(task-10-16)）；config_hash `91dfd211dbc3`。
- 配置：与 Run 006 唯一差异 = d_model 1→8（config+model 双处同改，参数 499,456）；batch 256 保持；其余全部同 Run 006（32/4/10、seed 1234、lr 恒定 1e-3、EMA 0.999、早停 patience=10、损失 {1,1,0.1,0.1}、best=val4 总 loss）。
- 显存预期：24.0 GiB（Run 008 实测）；100ep 上界 3.33 h（早停可能大幅缩短）。
- 命令：`mkdir -p outputs/gpfm-mamba-f0/v4b-d8-main-01/logs && python3 models/gpfm-mamba-v4/train.py --run-id v4b-d8-main-01`（后台）。

### Run 009 中断与续跑登记（2026-10-16）

- 中断：epoch 28（1 基）step 136/165 处进程被杀（主会话监控等待取消连带终止未 detach 的后台任务；无 NaN/无 early-stop，属外部中断非训练故障）。中断点状态：metrics.jsonl 0–26（0 基）27 行完整，best=epoch 27（1 基，val 0.76912，连续 27 轮新 best），last/best ckpt 完好。
- 续跑：`nohup python3 models/gpfm-mamba-v4/train.py --resume …/v4b-d8-main-01/checkpoints/gpfm-mamba-v4b-last.pt`（同 Run 延续口径，artifacts-management；RNG/optimizer/scheduler/EMA 全量恢复，早停计数经 metrics 回放连续=0）。日志续写 logs/train_resume.log（train.log 保留中断现场，尾部有取消时的重复行，无害）。

### 结果（2026-10-16 训练完成）

- **早停触发**：epoch 55/100 停止（连续 10 epoch 无新 best，patience=10；末行附 `"early_stop": true`，先落盘后 break）——**best epoch 45**（metrics 0 基 44）| **best val total 0.75638**（val4 口径，EMA，固定 t 网格确定性复算）。
- 曲线形态：val total 1.09372（ep1）→ 0.75638（ep45 best）→ 0.75913（ep55，+0.36%）；train total 同期 0.887→0.872 仍缓降——**过拟合观察项（11.8:1）轻微兑现**（best 后 val 微升而 train 继续降，分离幅度温和，非急速恶化）；全程 val<train 属口径不同构（train=随机 t 批内损失、val=EMA 固定 t 网格，同 Run 006 现象），不作品质判读输入。
- 资源：**peak 24.04 GiB**（summary.json，与 Run 008 bench 一致）| **119.06 s/ep 均值** | 1.39 it/s | 355.1 samples/s | **全程无 NaN**（55 行 metrics 数值域扫描 0 NaN/inf）| lr 恒 1e-3 逐 epoch 核验。
- 中断续跑时间线（上块中断登记的收口）：run 目录 13:03:54 建立 → 原进程 ep28 step 136 被外部终止（train.log 现场核实：末行恰为 `[ep 028/100 step 0136/165]`，其后取消重复行 2 行 + resume 输出追加入 train.log 并同步 train_resume.log——即「尾部重复行无害」的具体形态）→ 14:19:39 `--resume`（last ckpt ep27 1 基 / best_val 0.7691；RNG/optimizer/scheduler/EMA 全量恢复；metrics 截齐至 0 基 26 后自 ep28 重跑）→ 15:25:02 早停完成（resume 段 28 epoch ≈ 65 min）。总墙钟 ~2 h 22 m（含 ~15 min 中断间隙）；**metrics.jsonl 终态 55 行（0–54）无缺口无重复**（resume 段确定性重跑 ep28，val 0.7671 与中断前日志行一致）。
- 产物：`/root/npi/outputs/gpfm-mamba-f0/v4b-d8-main-01/`（config.yaml（resolved，hash `91dfd211dbc3`）、env_snapshot.txt、channel_mean/std.npy、logs/{train.log, train_resume.log, metrics.jsonl（55 行）, summary.json}、checkpoints/gpfm-mamba-v4b-{best,last}.pt（best=epoch 45 1 基，best_val 0.75638））。注记：summary.json 的 `epochs_run: 28` 为 resume 会话口径（非累计），累计 epoch 数以 metrics.jsonl 55 行为准。
- 图表：无（训练不出图；训练曲线图随 Run 010 evaluate 产出）。

### 结论

- 支持（主 Run 完成口径内目标）：d_ch=8 主 Run 完整跑通且早停 ep55 触发（vs d_ch=1 的 ep27——容量提升伴随更长有效训练）；**best val 0.75638 显著优于 d_ch=1 的 1.04405（同 val4 口径，−27.5%）**——d_ch 1→8 容量增益在内部验证上大幅兑现，与 Run 008 bench 1-epoch 证据方向一致，d_ch=1 容量塌缩竞争解释（design §9.10）在训练侧进一步坐实。
- 观察项登记：① 过拟合轻微（best 后 val +0.36% vs train −1.7%，参数/窗 11.8:1）——正式口径五维度为最终证据；② 早停 55 ep 仍未跑满 100，训练长度与 v1/v2 不同构注记照登；③ best val 0.7564 与 v1/v2 的 val2 best（0.7417/0.7741）不可直接比（判据集合与数据量不同，口径差异清单随附）。
- 下一步：Run 010 正式口径评估（Run 017 同口径）+ 指标 1 v2.1 + readout 诊断（d_ch 自适应）。

## Run 010 — v4b-d8-main-01 正式口径评估（Run 017 同口径）+ readout 诊断（2026-10-16）

### 目的与配置（跑前登记）

- 目的：v4b-d8-main-01（best ckpt，EMA 权重，**d_ch=8**）按 **Run 017 完全同口径**正式评估——七维度**四列**对比（v1/v2 = Run 017、v4b-d1 = Run 007、**v4b-d8 = 本 Run**）+ 三检验核算材料（指标 1 v2.1 / 指标 3 分频段 |Δlog10| / 指标 2 轨迹图备好交用户目视）+ R1 附分子分母 + std_t 幅度照登 + **readout 诊断副产品 M[j,i]（d_ch 由 ckpt 形状自适应，d8 主路径）**；产出 **H3 三态判读建议**（GC r 是否回升）与 **d1→d8 剂量对照**（GC 幅值是否自 3.9% 恢复、峰频偏移是否消失）。
- 代码：同 Run 009（commit `acfc899` 工作树）；评估链只读复用 v1（evaluate/gc_psd/fc_metric/sample/visualizer 经双遮蔽）。
- 配置/命令（默认参数即正式口径，与 Run 007 对 d1 的评估逐项一致）：
  - `python3 models/gpfm-mamba-v4/evaluate.py --run-id v4b-d8-main-01`（Pass A K=16 seed=1234、Pass B K=1 seed=1235、GC p_max=10 bootstrap 1000 受试者级 seed=2026、PSD nperseg=512、空间 σ=0.080、**gen_lowpass_hz=40.0 正式口径**（config 显式，无命令行覆写）；**显存预警：d8 评估路径显存显著高于 d1 探针（当时 12.36 GiB）——若 OOM 立即 `--batch-size 32`（或 16）重试并登记降档口径**；Pass A 采样预计 ~30–40 min）
  - `python3 models/gpfm-mamba-v1/fc_metric.py --run-id v4b-d8-main-01`（指标 1 v2.1）
  - `python3 models/gpfm-mamba-v4/readout_diag.py --run-id v4b-d8-main-01`（M[j,i]=‖W^{(i),1}_{:,j}‖_F 热图 + M~GC(gen)/M~GC(real) Spearman；对照 d1 的 0.240/0.049）
- ckpt：`checkpoints/gpfm-mamba-v4b-best.pt`（epoch 45 1 基，best_val 0.75638，config_hash `91dfd211dbc3` 校验）。
- 环境/数据：同 Run 001（base @ AutoDL，RTX 4080 SUPER 32GB；test 折 = v1 fold0 同 10 例——**test 指标与 v1/v2/d1 三方可比**）。

### 结果（2026-10-16 评估完成）

- 评估跑通（evaluate 全程 ~49 min：Pass A 207 批 2633 s 主导 + Pass B + GC/PSD/空间 + bootstrap 1000 + 图；fc_metric 3.6 s；readout_diag ~0.5 min）；**无 OOM、无 --batch-size 降档**（评估期 GPU 峰值 ~28.1 GiB/32.8 GiB 稳定——高于 d1 探针 12.36 GiB，与预警一致但卡内安全）；无命令行参数偏离（默认参数即正式口径：Pass A K=16 seed=1234、GC 单样本 seed=1235、BIC p\*=4（真实组选定两组共用）、bootstrap 受试者级 1000 seed=2026、gen_lowpass 40 Hz）。

**七维度四列并列（v1/v2 = Run 017 正式口径；v4b-d1 = Run 007；v4b-d8 = 本 Run；test 折四方相同）**

| 量 | v1 main-01 | v2 v2d62-main-01 | v4b-d1 v4b-main-01 | **v4b-d8 v4b-d8-main-01** | 备注 |
|---|---|---|---|---|---|
| test-MAE（主指标） | 0.70834 | **0.70096** | 0.71663 | 0.72563 | d8 四方最差（+2.4%/+3.5%/+1.3% vs v1/v2/d1）；GPR 均值基线 0.69297 四方均未超 |
| CRPS | 0.52629 | **0.52496** | 0.53461 | 0.56966 | **d8 显著劣化**（+8.2%/+8.5% vs v1/v2） |
| 逐通道 r | 0.14371 | **0.15053** | 0.11479 | 0.14643 | **d1 塌缩恢复**至 v1/v2 带（+1.9% vs v1、−2.7% vs v2） |
| R1 触发 | 未触发（1.2432） | 触发（1.5261，口径效应） | 未触发（1.0386） | **未触发（1.3899）** | 分子分母见下 |
| 指标 1 FC r（v2.1） | 0.95856 | 0.97625 | **0.98510** | 0.96662 | **≥0.95 过线**；ρ 0.97205；t=106.8 p=2.8e-15（df=9） |
| GC r（补充参考） | **0.27549** | 0.12272 | 0.13343 | 0.16300 | ρ 0.20762；bootstrap r CI [0.093, 0.195]、ρ CI [0.137, 0.240]；**落预注册 0.15–0.22 邻域** |
| GC 幅值 gen/real（max 比） | 92% | 89% | 3.9% | **109%**（0.03334/0.03051） | **d1 塌缩完全恢复**；显著边 gen 3782/3782、Jaccard 1.000（d1 0.980） |
| 空间 R̂~R / Fro | 0.953 / 10.79 | 0.970 / 6.71 | 0.984 / 9.89 | 0.968 / **6.50** | **Fro 四方最优**；R̂~K_s 0.803（先验注入相关低于 d1 的 0.893） |
| 低频 δ（指标 3） | 0.091 | 0.122 | 0.131 | 0.124 | θ 0.497/α 0.249/**β 0.674**；**δ–β(1–30) 整段 0.519 > 0.5 超线**（v1/v2/d1 0.091/0.122/0.233） |

- 指标 3 高频照登：30–60 Hz **2.618**（v1/v2/d1 2.36/2.38/2.295）；60–250 Hz **9.97**（9.74/9.76/9.594）——量级与三方相同（真实侧数值地板主导）；分频段口径 = 全通道平均谱 |Δlog10|（bin [lo,hi)，方法对 d1 复算逐段复现 Run 007 登记值）。**gen 峰频率 0.98 Hz = real 0.98 Hz——d1 特有的 1.95 Hz 偏移消失**。偏差方向登记：θ/α/β 全为**正向**（gen 功率偏高），最差 bin 集中 **15–19 Hz**（|Δlog10| 至 0.82）——与 std_t 过冲同源的过量功率特征。
- **R1 分子分母（附三方对照）**：分子 = 采样终点 test-MAE **0.72563**；分母 = 训练末期（ep51–55 1 基）5-epoch 均值 x̂1-MAE **0.52206**；比值 **1.3899 < 1.5 未触发**（高/中 t 桶比 0.3677 亦未触发）。对照：v1 0.70834/0.5699=1.2432、v2 0.70096/0.45931=1.5261（触发，按 R1 规则定性为分母收缩口径效应）、d1 0.71663/0.690=1.0386——d8 分母较 d1 收缩（0.690→0.522，拟合更深）但未及 v2 深度，且分子为四方最差；未触发属分子轻微劣化与分母收缩共同作用，**非 v2 型纯口径效应**，按规则照登两值。
- **std_t 幅度**：生成 **1.38492** vs 真实 **0.79459**（本 run z-score 口径，与 d1 同 32 例训练折）→ 比值 **174.2%（过冲 +74.2%）**——四方最差（v1 +23%/v2 +29%/d1 欠冲 −7.3%）；**d1 的欠冲方向随容量提升反转为大幅过冲**。样本间展宽 1.32873（v1 0.93065/v2 0.96344/d1 0.81372）。与 CRPS +8.2%、β 段过量功率互洽：**d8 生成分布过散**。
- **readout 诊断副产品（仅诊断，R6；d_ch 由 ckpt W1 (62,16,496) 形状自适应推断 = 8，主路径生效）**：M[j,i]（行=源、列=靶）mean 0.75146，**对角（自源）2.36846 vs 非对角 0.72495（占优比 3.27×，d1 为 0.460/0.308=1.49×）**；与外部 GC 展平相关（主口径 = 去对角 3782 有向边）：
  - M ~ GC(gen)：Spearman **0.49322** / Pearson 0.58619（d1 0.23981/0.38797；全 62×62 0.42249）
  - M ~ GC(real)：Spearman **0.17388** / Pearson 0.25285（d1 0.04927/0.11165；全 62×62 0.11826）
  - 解读：d8 的 head 内部有向读出与**模型自身生成动态** GC 高度一致（0.49），与**真实 GC** 从 d1 的几乎不相关（0.05）升至中等（0.17）——显式混合层学到的有向结构与真实有向动态的对应随 trunk 容量恢复而建立，但仍未达 gen 侧水平（0.17 vs 0.49）。
- 产物（全部落 `/root/npi/outputs/gpfm-mamba-f0/v4b-d8-main-01/`）：eval/{summary.json, gc_psd_results.json, fc_metric.json, fc_matrices.npz, gc_input.npz, gc_real.npy, gc_gen.npy, gc_F_p_gen.npz, gc_F_p_real.npz, psd.npz, spatial_corr.npz, readout_M.npy, readout_diag.json}；figs/{eval_trajectories, eval_training_curves, eval_gc_matrix, eval_psd, eval_spatial_corr, eval_readout_matrix}_v4b-d8-main-01.png；logs/{eval.log, eval_metrics.jsonl（8 stage 齐全）}。
- 图表：指标 2 素材 = `figs/eval_trajectories_v4b-d8-main-01.png`（medoid + 90% PI、K=16、lowpass 40 Hz 注记）——**交用户目视判定**。

### 结论

- **七维度初判（vs v1/v2/d1，正式口径同 test 折）**：d8 三项积极（空间 Fro 6.50 四方最优、GC 幅值 109% 恢复至 v1/v2 带、逐通道 r 0.1464 回带）+ 指标 1 FC r 0.9666 过线；两项劣化（MAE 0.7256 四方最差、**CRPS 0.5697 显著劣化 +8.2%**）+ std_t 过冲 +74.2% + **指标 3 低频 0.519 超线**。
- **H3 三态判读建议（判定权在用户）**：GC r（Pearson 主口径）**0.16300**——未达「部分支持」线 0.18，明显高于 v2 水平（0.1227，+0.040），更未达 v1 带（0.2+）；**落在预注册的 0.15–0.22 邻域（design §1 预设动作：提请用户加 seed 再判）**。Spearman ρ 0.20762 已入 v1 带——两口径分歧登记（Pearson 幅值加权线性 vs Spearman 边序；gen 幅值恢复后大值边贡献真实化，ρ 侧更乐观）。**建议：此单 seed 落点不定论，按预注册规则提请加 seed（≥2 个补充 seed）后再判**；若以当前单 seed 强判，位置在「不支持」与「部分支持」之间、偏部分支持侧。
- **d1→d8 剂量对照（同结构容量加倍，单变量）**：① GC 幅值 3.9%→**109%**（25× 塌缩完全恢复）；② gen 峰频偏移**消失**（1.95→0.98=real）；③ 逐通道 r 0.1148→0.1464（塌缩恢复）；④ readout M~GC(real) 0.049→**0.174**、M~GC(gen) 0.240→0.493；⑤ best val 1.044→0.756（训练侧 −27.5%）。——**d1 口径的「不支持」判读主要由 trunk 容量塌缩驱动（design §9.10 竞争解释在 d1 成立）**；但容量恢复后 GC r 仅 +0.030（0.133→0.163），未随幅值同步回到 v1 带——提示容量是 GC 一致性的必要条件而非充分条件，剩余缺口指向损失/训练目标侧（无跨通道时序监督）或 d_ch=8 仍低于 v1 的 16（未穷尽）。
- **五维度守护检查（H3 判据的守护条件）**：逐通道 r/FC/空间三项不塌缩（r 回带、FC 过线高于 v1、Fro 四方最优）；MAE 轻微劣化（+2.4%/+3.5%）；**CRPS 显著劣化（+8.2%/+8.5%）**——配合 std_t +74% 过冲与 15–19 Hz 过量功率，**d8 生成分布过散是明确的质量代价**；若 H3 判「支持/部分支持」须随附此代价声明。
- **过拟合竞争解释检查（design §1 预登记）**：轻微兑现——best epoch 45/55（中段）、best 后 val +0.36% vs train −1.7%、参数/窗 11.8:1、R1 分母收缩至 0.522；「CRPS 劣化 + 过散」可能部分来自训练分布拟合加深而非纯结构效应，列为竞争解释之一（非唯一归因；欠拟合侧已排除）。
- **口径差异清单（对 v1/v2 对比必随附）**：① CI trunk + cMLP head 两处变化（非单变量，归因局限——分解变体为条件后续）；② 训练数据 32 例 vs 34 例（−6%）；③ best 判据 val4 vs val2；④ 早停 55 ep + 恒定 lr vs 固定 100 ep + cosine；⑤ d_ch=8 仍低于 v1 的 d_model=16。对 d1 对比为单变量剂量差（仅 d_ch 1→8）。
- 三检验核算：**指标 1 通过**（0.96662 ≥ 0.95；t=106.8 p=2.8e-15 df=9）/ **指标 3 低频数值侧未达标**（δ–β 0.519 > 0.5，超线 +3.8%；主因 β 0.674 + θ 0.497；高频照登 2.618/9.97；PSD 形态目视交用户）/ 指标 2 轨迹图备好——**判定留用户**；H3 三态判读建议如上（含加 seed 决策），**判定留用户**。
- 下一步：§5 H3 判读收尾（用户判定留痕：三态选择 + 是否加 seed + 指标 2/3 目视）→ trellis-check 全量质量检查 → 3.3 spec 更新与 progress.md 同步。

## 方向变更登记（2026-10-16）：v4-B 放弃存档，v4-A 复活

**用户拍板（2026-10-16）：放弃 v4-B（CI 骨干 + cMLP 读出）方向，回到本任务最初设计 v4-A（低秩输入瓶颈，第一轮 trellis-review 修正版）**：

- **v4b 结论摘要（放弃依据）**：d_ch 1→8 剂量对照（Run 006/007/009/010）证「容量必要而不充分」——容量恢复后 GC 幅值（3.9%→109%）、gen 峰频偏移消失（1.95→0.98 Hz）、逐通道 r（0.1148→0.1464）、readout M~GC(real)（0.049→0.174）俱恢复，但 **GC r 仅 0.133→0.163**（未达部分支持线 0.18，落预注册 0.15–0.22 邻域）；**指标 3 未达标**（δ–β 0.519>0.5）、std_t 过冲 +74%、CRPS +8.2%（生成过散代价）。代码与 Run 记录**存档不删**（`models/gpfm-mamba-v4/`、`outputs/gpfm-mamba-f0/v4b-*`）；不再加 seed、不做 d_ch=16。
- **v4-A 复活（2×2 缺失格恢复原框架，design §1）**：v2 基础（共享 trunk d_model=62）单变量——in_proj `Linear(79→62)` 满秩分解为 `Linear(79→16)→Linear(16→62)` **纯线性低秩**（两层间无激活，rank≤16；PyTorch 默认初始化——两层复合谱窄为预登记风险项）；trunk/head/其余与 v2 逐字段一致（**容量不变：参数 130,056 = v2 132,682 − 2,626 = 0.9802× v2**；in_proj 4,960→2,334）。检验 H3「接口强制混合 × 容量保持」格；与 v4-B「trunk 零混合 + 顶层显式读出」互为对照臂——若 v4-A GC 回升而 v4-B 未回升，指向接口混合优于顶层读出；若同不回升，缺口共同指向损失侧。别名 v4a（仓库 v4 已被存档的 v4-B 占用）。
- **口径继承（当前累计口径，均用户 2026-10-16 拍板，v4-A 全量继承）**：划分 **32/4/10**（`n_val_subjects=4` + v4a 自备 manifest `models/gpfm-mamba-v4a/splits/fold0_status_ok_seed42_val4.json`，内容与 v4b 的 val4 manifest**逐字节一致**由单测 ⑧ 断言）、**早停 patience=10**、**lr 恒定 1e-3**、gen_lowpass_hz=40.0 显式入 config。对 v1/v2 对比的口径差异四项清单（32 例训练 / val4 判据 / 早停+恒定 lr / 低秩结构）随判读登记；test 折与 v1 逐 ID 相同 ⇒ test 指标可比。
- **实现（v1 default-off 钩子 + v4a 薄变体目录）**：v1 `model.py::VelocityNetwork` 增 `bottleneck_rank=None`（None 分支结构与 state_dict 键逐位不变——单测 ① 与**改动前 git ref `0c8d0ee`** 同进程同 seed 前向逐位一致断言；r 分支 `nn.Sequential(Linear(79→r), Linear(r→62))`，键 `in_proj.{0,1}.*`）+ v1 `train.py`/`sample.py::load_eval_model` getattr 透传（先例 prior_bandlimit_hz）；新目录 `models/gpfm-mamba-v4a/{config.py, train.py, evaluate.py, tests/}`（v2 式**单遮蔽** wrapper——v4a 无 model.py，模型经 v1 钩子构建；`RUN_PREFIX="v4a-"`；`--rank {8,16,32}` 运行时改写被遮蔽 config 实例 + run_id `rXX` 标识校验 + resume hash 校验防 r 串用；config.py 文件默认唯一 16）。**v1/v2/v3 零漂移守护**：三变体 config_hash 逐位不变（c6ff80ac1c43 / fc2a9ce14948 / 0c2de30a4868）+ v1 七测回归全 PASS（单测 ⑤⑥）。v4a config_hash 新 pin：默认 r16 `2a456201dc90`、`--rank 8` `a1960979c695`、`--rank 32` `f949f6c63531`、epochs=1 bench 覆写 `eacb763a176a`。

## Run 011 — v4a 冒烟 + 1-epoch 基准（val4 口径 32/4/10）（2026-10-16）

### 目的与假设（跑前登记）

- 目的：① 冒烟（`--smoke`，4 例受试者 1 epoch）验证 v4a 低秩结构经 v1 单遮蔽包装接入训练主流程的全链路机械正确性（数据加载/val4 划分校验、GPR 先验、低秩 in_proj 前向反向、四项损失、R9 监控/checkpoint/config_hash 落盘）；② bench（`--epochs 1 --bench`，全量 train 32 例）实测 peak 显存 / epoch 墙钟 / it/s / 四项损失走势 / 100-epoch 外推，作为主 Run（v4a-r16-main-01）启动门禁材料——**用户已授权方向（「尝试 v4a」）：bench 无异常（无 NaN、显存安全、损失正常下降）即直接进主 Run，不再等待确认**；异常则停下汇报。
- 预期（design §3）：计算/显存 ≈ v2 量级（v2 实测 9.45 s/ep@34 例、1.94 GiB；v4a@32 例 ~9 s/ep、~2 GiB、100ep 上界 ~15 min，早停 10 可能缩短）；对照 v4b-d8 bench 119.8 s/ep / 24.0 GiB。

### 配置

- 代码：`models/gpfm-mamba-v4a/`（本块新增目录：config.py + train.py/evaluate.py 单遮蔽包装 + tests/test_v4a.py 八项）+ v1 三文件 default-off 钩子增量（model.py bottleneck_rank / train.py / sample.py getattr 透传）；git commit 0c8d0ee 工作树增量
- 参数：结构 in_proj 79→16→62 纯线性低秩（`bottleneck_rank=16`）、d_model=62 其余与 v2 逐字段一致（参数 130,056）；训练 32/4/10、batch 256、seed=1234、AdamW lr 1e-3 wd 1e-4 **恒定**（无 cosine）、EMA 0.999、损失 {1,1,0.1,0.1}、早停 patience=10（1 epoch 不触发）、best=val4 总 loss；resolved config_hash：smoke/bench 均默认口径 2a456201dc90、bench epochs=1 覆写 eacb763a176a
- 环境：base @ AutoDL（RTX 4080 SUPER 32GB，16 核 cgroup BLAS 上限 8 线程）
- 数据：`/root/autodl-tmp/edra_process_v4/`，status_ok 46 例，fold0 split_seed=42 + val_subject_seed=42（校验 v4a 自备 val4 固化 manifest，内容与 v4b 逐字节一致）
- 命令：
  - `python3 models/gpfm-mamba-v4a/train.py --smoke --run-id v4a-smoke-a`
  - `python3 models/gpfm-mamba-v4a/train.py --epochs 1 --bench --run-id v4a-r16-bench-1ep`

### 结果（2026-10-16 跑完，smoke 与 bench 均 exit 0 无 NaN）

- 单测前置：`tests/test_v4a.py` 八项全 PASS（①None 分支与改动前 git ref `0c8d0ee` 同 seed 前向逐位一致 + state_dict 键集/逐值不变 ②r=16 两层纯线性/键形状/复合 rank=16≤16 构造 ③前向 (4,62,128)/return_all/因果三段/49 参数张量梯度有限/EMA/GPU B=256 冒烟 34 ms ④参数 130,056 精确（seg 2 + in_proj 2,334（1,280+1,054）+ trunk 123,752 + head_norm 62 + head 3,906 = v2 132,682 − 2,626）⑤三变体 hash 零漂移 + v4a 四 pin ⑥v1 七测全 PASS ⑦单遮蔽/前缀守卫/--rank 正反例 ⑧32/4/10 + val=[25,37HC,40,49MDD] + test 逐 ID 同 v1 + v4a manifest 与 v4b 逐字节一致）。
- **冒烟（v4a-smoke-a）**：全链路跑通——train 4 例 5,284 窗 × 1 epoch（20 step）| train total 1.34738（mae 0.71262）| val4 总 loss（EMA）1.27179（新 best）| 7.48 it/s | epoch 2.67 s | peak **1.95 GiB** | `[model] params 130056` | 损失全程有限。注：smoke 落盘 resolved hash = **eacb763a176a**（`--smoke` 强制 epochs=1 ⇒ 与 bench 覆写同口径；预登记「smoke 默认口径 2a456201dc90」表述以此更正）。
- **基准（v4a-r16-bench-1ep，32/4/10 全量）**：

| 项 | **v4a-r16-bench-1ep** | v2d62-bench-1ep（34/2/10） | v4b-d8-bench-1ep（32/4/10） | 备注 |
|---|---|---|---|---|
| train 受试者/窗 | **32 例 42,272 窗**（165 step） | 34 例 44,914 窗（171） | 32 例 42,272 窗 | v4a 与 v4b 同口径；v2 多 2 例 |
| peak 显存 | **1.95 GiB** | 1.942 GiB | 24.042 GiB | ≈ v2 量级（design §3 预期兑现）；32 GB 卡余量充足 |
| epoch 墙钟 | **8.5 s** | 9.45 s | 119.8 s | 窗数归一后 v4a ≈ 9.0 s 等效——与 v2 持平；12.3× 快于 v4b-d8 |
| it/s | **19.41** | 18.53 | 1.38 | |
| 100-epoch 外推 | **0.236 h（~14 min）** | 0.26 h | 3.33 h | train 循环口径；早停 10 下为上界 |
| train total（epoch 均值） | **1.25357**（mae 0.66781） | 1.22138（mae 0.65246） | 1.27398（mae 0.65948） | 同口径对比 v4b-d8：v4a 优 1.6%（1 epoch 拟合）；vs v2 有窗数/口径差仅作参照 |
| val total（EMA，val4） | **1.15632**（mae 0.58216/cons 0.51604/tv 0.18755/corr 0.39444） | 1.10751（**val2 口径，不可直接比**） | 1.09372 | v4a 主 Run val 轨迹对照基线（v4a 口径首登） |
| 损失走势（step 1→165） | **1.4982→1.1484** | 1.4894→1.1103 | 1.3912→1.2270 | 正常下降、无高原（两层默认初始化谱窄风险 §6.2 无 1-epoch 红旗） |
| t 分桶 mae | 0.90064→0.13057 单调降 | 0.881→0.094 单调降 | 0.86853→0.12937 单调降 | sanity 通过，形状一致 |
| v_norm_mean / x̂1_std(EMA) | **45.87 / 0.7309**（val 口径 0.7309 vs x1 std 0.8533） | 52.52 / 0.651 | 42.19 / 0.70775 | v2 量级邻域（速度范数正常） |

- lr 恒 1e-3 逐 step 核验（进度行）；best（=唯一 epoch，EMA val4 总 loss 1.15632）/last 就位。
- 产物：`/root/npi/outputs/gpfm-mamba-f0/v4a-{smoke-a,r16-bench-1ep}/`（config.yaml（resolved：bottleneck_rank 16 / n_val_subjects 4 / lr_schedule "constant" / early_stop_patience 10）、env_snapshot.txt、channel_mean/std.npy（train-32 口径）、logs/{train.log, metrics.jsonl, summary.json}、checkpoints/gpfm-mamba-v4a-{best,last}.pt）；bench resolved hash **eacb763a176a**（epochs=1 覆写，与 pin 一致）。
- 图表：无（冒烟/bench 不出图）。

### 结论（跑后补）

- **支持（具备进主 Run 条件，用户已授权方向——按 implement.md §2 直接进主 Run，不再等待确认）**：① 显存 1.95 GiB 与 v2 同量级（vs v4b-d8 的 24 GiB——低秩接口不改变计算形态，32 GB 卡内余量 ~30 GiB）；② 时长 100 epoch 上界 ~14 min（早停 10 可能缩短）；③ 损失正常下降（1.4982→1.1484，无 NaN、无高原）+ t 分桶单调 sanity 通过 + v_norm/x̂1_std 均处 v2 邻域——**两层默认初始化谱窄（design §6.2 第一怀疑项）1-epoch 无红旗**（风险登记保留，主 Run 期间持续对照 v2 train 曲线）；④ 低秩结构经 v1 单遮蔽包装机械正确（八项单测 + val4 划分实跑校验）。
- 观察项登记：v4a 1-epoch train total 同口径优于 v4b-d8（1.2536 vs 1.2740）、val4 略高（1.1563 vs 1.0937）——共享 trunk 容量下接口瓶颈的早期拟合形态，非判读输入（1 epoch 证据弱，主 Run 曲线为准）。
- 下一步：**Run 012 主 Run `v4a-r16-main-01`**（100ep 上限 + 早停 10 + 恒定 lr；主会话 nohup 脱离式启动，本实现会话不启动）→ Run 013 正式评估（Run 017 同口径）→ H3 判读（含与 v4b 剂量对照联合陈述）。
