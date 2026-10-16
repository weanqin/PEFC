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
