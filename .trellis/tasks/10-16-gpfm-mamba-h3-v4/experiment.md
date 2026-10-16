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
