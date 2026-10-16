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
