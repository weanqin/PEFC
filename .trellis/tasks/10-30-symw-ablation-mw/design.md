# Design — MILD symw 五折 M/W 消融

## 1. 消融机制（零训练代码改动）

### 1.1 A_eff 与 γ 的身份

`LearnableMexHatMixer.effective_adjacency()`（`latent_flow/models/oc_latent_flow.py:426-441`）：

```
A_eff = gamma_m · M(d; log_sigma) + gamma_buf · W
```

- `gamma_m` = `--mexhat-gamma-m`（buffer persistent，基线 1.0）；`gamma_buf` = `--mexhat-gamma`（buffer，基线 0.1）。二者随 ckpt 持久化，装载侧无感。
- M = ricker 核 `(1−d²/σ²)·exp(−d²/2σ²)`，`pairwise_d` 为 ckpt 内 persistent buffer（geometry_mild.pt GSN-HydroCel-128），唯一可学参数 `log_sigma`。
- W = 128×128 `nn.Parameter` 零初始化；`--sym-w` 下 SymWAdam 每步投影 `W ← (W+Wᵀ)/2`。
- A_eff 两个应用位共享单份：入口 `graph_mixer.forward` 与出口 `FlowVelocityNetExitGNN._forward_graph`（`flow_exit_gnn.py:143`）——消融两处同时生效（stage-2 GNN 模块的整体 M/W 消融语义）。

### 1.2 γ 置零的数学后果（自动失活，无需冻结代码）

- γ_W=0 ⇒ `∂A_eff/∂W ≡ 0`（精确浮点零）⇒ Adam 对 W 更新量精确为零（无 weight decay；exp_avg 恒 0）⇒ W 逐位保持零初始化；SymWAdam 投影 (0+0)/2 no-op。训后守卫：`W.abs().max() == 0` 逐位。
- γ_M=0 ⇒ `∂A_eff/∂log_sigma ≡ 0` ⇒ log_sigma 逐位保持初值 `log(median(pairwise_d[offdiag]))`。训后守卫：`|log_sigma − log(median_d)| < 1e-12`。
- A3（双零）：A_eff ≡ 0；消息轮退化为 `x += GELU(Linear(0)) = GELU(bias)` 常数残差（无跨通道混合，参数量恒等）；anti-uniform `1/(1+var(0)+1e-6) ≈ 1.0` 常数 ⇒ loss 恒加 λ·1.0 ≈ 0.2、梯度为零（argmin 选择不受影响，但原始 loss 跨臂不可直比 ⇒ M1 用 datafit 口径）。
- A1（no-M）：`uni_reg` 作用于 W-only，与基线同机制。

### 1.3 为什么不是随机初始化/恒等映射（已否决，记录理由）

- 随机 W：混淆「先验缺失」与「随机扰动」，破坏 W 零初始化起步语义（exit-gnn/零初始化门控依赖 W=0 起步，A_eff 初值 = γ_M·M 纯先验）。
- 恒等 A：M 为距离核语义，恒等阵无几何定义；需代码注入，破坏「除 γ 外逐键相同」的对拍面。
- scale-to-zero 是仓库登记口径（`effective_adjacency` docstring: "set 0 to ablate the band-pass prior, task 07-15"）。

## 2. 运行拓扑

### 2.1 目录布局（path-isolation）

```
outputs/10-30-symw-ablation-mw/
  run_{arm}_fold{k}/                  # arm ∈ {noM, noW, noMW}，k ∈ 0..4
    checkpoints/fold{k}_mild_flow_best_graph_mexhat.pt
    configs/stage2_config.json        # wrapper 自落（cli/pinned/windows/manifest/norm 全档）
    logs/stage2_history.json          # train/val 全 epoch 全字段（R4）
    quality/quality_meta.json         # M2 抽查（quality_check_mild.py）
  fc/{arm}_fold{k}/fc_metric.json     # M3（fc_metric_mild_ablation.py）
  aeff_gc/{arm}_fold{k}.json          # M4（aeff_gc_corr.py）
  guards/guards_{arm}_fold{k}.json    # 训后守卫结果
  smoke/                              # S1 冒烟（独立子目录，epochs 2）
  summary/                            # S5 汇总：summary_5fold.{json,md} + 损失曲线并置数据/图
  run_{arm}_fold{k}_stage2.log        # 完整 stdout（R4，nohup 级）
```

- 训练日志全量保留（用户要求 6）：stdout log 重定向到 run 级 `*_stage2.log`（wrapper 自身逐 epoch 打印 loss），history JSON 双保险。

### 2.2 训练命令（S2 正式批，15 跑串行；argv 从基线 cli 逐键重建，不手写模板）

**构造规则（Mandate B blocker 1 修正）**：argparse 默认值与基线不同（默认 epochs 50/batch 32/patience 10/window 1500/obs 1125/fc 375），禁止"其余旗标全默认"手写模板。驱动脚本从基线同折 `stage2_config.json.cli` **逐键重建 argv**（每个非 None/False 的键都显式传参），仅替换 `run_dir`（→ 本任务目录）与 γ 两键（按臂置 0），`vae_ckpt` 一律 `Path(...).resolve()` 绝对化（fold3/4 基线 cli 为相对路径，子进程 cwd 敏感——Mandate B 发现 4）。

**跑前守卫（每次 subprocess 启动前执行，不是批后）**：`parse_args(重建的 argv)`（train_mild_depress.py:134 可 import）→ 结果 dict 与基线 cli 对拍，diff 键 ⊆ {run_dir, mexhat_gamma, mexhat_gamma_m}，不等即拒跑该单跑（fail-fast，不浪费 GPU）。

- 驱动脚本 `models/latent-flow-v1/run_ablation_mw.py`：逐臂逐折串行调用（subprocess，stdout/stderr 转 run 级 log），单跑失败即停。
- 环境约束（environment.md）：单卡 4080S；入口自带 BLAS 守卫（train_mild_depress.py 已有）；驱动脚本自身 import torch 前同款 BLAS 守卫；无 tmux，用 nohup 跑批；batch.log 目录先建。

### 2.3 与基线的确定性关系

- 基线不重训（复用五折既有 history/quality/fc summary，M4 的 A0 A_eff 从已注册 ckpt 现提取——`result/checkpoints/mild/latent_flow_symw/fold{k}/` 有 SHA 注册，且与 outputs 同源；A0 的 M4 直接用 outputs 侧基线 ckpt 提取并与 result 侧 SHA 对拍一致后使用）。
- 消融臂 seed 42 与基线相同；同折 stage-1 相同 ⇒ 唯一差异 = γ 两键（对拍守卫钉死）。

## 3. 新增脚本（3 个，全部只读基线/refs，产物落本任务 outputs）

### 3.1 `run_ablation_mw.py` — S1/S2 驱动 + CLI 对拍守卫

- 读基线五折 `stage2_config.json`，生成每臂每折的命令；`--smoke` 模式 = fold0 × A3 × `--epochs 2 --limit-subjects 2`（验旗标链路 + 守卫逻辑，不产正式数）。
- 跑前守卫（对拍）+ 跑后钩子：校验 history/best ckpt 存在。

### 3.2 `ablation_guards.py` — AC2 训后守卫

每跑装载 best ckpt（`--flow-exit-gnn` 架构、strict load）后断言：
- A2/A3：`W` 全零（`int(W.abs().sum())==0` 精确）；
- A1/A3：`|log_sigma − log(median_d)| < 1e-12`（median_d 由 ckpt 内 `pairwise_d` offdiag 现算，与构造口径一致）；
- A1/A2：`torch.equal(W, Wᵀ)`（A1 必须严格对称；A2 W=0 平凡对称）；
- 全臂：γ buffer 值与臂定义一致（gamma_m ∈ {0,1}、gamma_buf ∈ {0,0.1} 按臂）；
- A0 复验：γ=(1.0, 0.1) 且 W 对称（symw 线身份）。
- 结果落 `guards/guards_{arm}_fold{k}.json`；任一断言失败非零退出。

### 3.3 `fc_metric_mild_ablation.py` — M3 变体（镜像 `fc_metric_mild_symw.py` 口径）

- 差异仅三处：① ckpt = `--ckpt` 显式路径（不再锚 result 白名单）；② 保留 manifest SHA 与 norm SHA 守卫（数据侧不变）；③ ckpt SHA256 随档登记进输出 JSON（替代白名单，供复跑对拍）。
- 口径逐字镜像：trial 前 500 点的 [375,500) 段、30 trial 拼接 → 128×128 corrcoef、组平均上三角 k=1 Pearson/Spearman、逐被试 t 检验、z 域计算 + 仿射不变断言、确定性采样 K=1。
- 装载旗标 `--flow-exit-gnn` 恒开（四臂 ckpt 同架构）。

**装载路径硬约束（Mandate B 发现 6）**：`OCLatentFlow.load_checkpoint` 的 `_build_from_variant` 不识别 exit-gnn，v1 路径 strict=False 会**静默丢弃 exit_\* 键**——guards/M3/M4 一律走 `build_model_mild(_model_args(flow_exit_gnn=True))` + `viz_mild.load_flow_ckpt`（strict=True；quality/fc 先例路径，viz_mild.py:116-134），不得用 OCLatentFlow.load_checkpoint。

### 3.4 `aeff_gc_corr.py` — M4 新指标

- 每折：装载 best ckpt（装载路径同 §3.3 硬约束）→ `model.flow.graph_mixer.effective_adjacency()` 得 A_eff（float64 化）。
- GC 参考：`refs/.../mild40-trial500-zscore-fs250-fixed8-f1-45-favg/refs_s{id}.npz`（40 例；ec[a,b]=a→b 行=源/列=靶；refs 只读 + SHA256SUMS 全验；方向语义以 10-24/10-28 gccorr 实际代码用法为最终准绳）。
- 组平均：主口径 = 该折 test 8 例 sid 的 ec 逐元素平均；副口径 = 全 40 例平均。
- 相关（offdiag 16256 对）：Spearman + Pearson × {raw A_eff, |A_eff|} × {aligned = vs ec_group, flipped = vs ec_group.T}——**命名沿用 EC×GC 先例**（ec_mild_unbiased.py:813-817：aligned 对 ec、flipped 对 ec.T；Mandate B 发现 3 修正），与 10-25/10-28 EC 线并置不错位。A_eff 对称 ⇒ 两个方向只差参考矩阵转置；A_eff 对称性本身有 ~4.8e-7 cdist 地板（登记一句，对相关无实质影响）。
- **距离偏相关副口径（Mandate A 发现 1 采纳）**：A2 臂 A_eff = 纯距离核，raw 相关与 GC 的共同距离梯度混杂（evaluation-calibers Run 008/009 教训）⇒ 增加 partial Spearman：A_eff/GC/通道对 3D 距离三者秩变换后，距离作偏出变量（coords 由 ckpt 内 pairwise_d 提供），报告 residual 相关。README 判读：A2 臂以 partial 口径为主。
- A3 守卫：A_eff 全零 ⇒ 全部相关字段 NaN + `"by_design": true`（Spearman 常数向量无定义）。
- 输出 `aeff_gc/{arm}_fold{k}.json`：每折点估计（raw/abs × aligned/flipped × plain/partial）+ 参考侧 SHA + A_eff 的 SHA/范数摘要。

### 3.5 `ablation_summary.py` — S5 汇总

- 收集：15+5 跑 history（含基线五折）→ M1 表（datafit/loss/recon2/corr2/cons2 @best-epoch，五折 mean±sd + 逐折 Δ vs A0）；15+5 quality → M2；15+5 fc → M3；15+5 aeff_gc → M4。
- **逐臂逐折登记 best_epoch + 停止状态（CAP=贴 150 上限 / early=早停；Mandate A 发现 2 采纳）**：基线 2/5 折贴顶格，消融臂贴限状态可能系统性不同，贴限折的 Δ 判读加 caveat。
- **敏感性副列 datafit@argmin(datafit)（Mandate A 发现 3 采纳）**：主报 datafit@trainer 所选 epoch（与基线同规则保可比），副列报各臂 history 中 datafit 最小 epoch 的值，README 结论段写一句 best-epoch 选择目标跨臂不同构的 caveat。
- 损失曲线并置（R4/要求 7）：四臂五折 val loss + 三分量逐 epoch 曲线数据落 `summary/loss_curves.npz`，并出汇总图 `summary/loss_curves.png`（五折均值曲线，四臂同图；viz 规范走统一 visualizer 风格，nature-figure 美学）。
- M2 口径交叉核验：各 quality_meta 的 window_len==500（Mandate B 发现 2 采纳）。
- 产物：`summary/summary_5fold.json` + `summary/README.md`（表 1 主表 + 表 2 逐折 + 结论段；结论措辞按 Mandate A 备注：贡献归因表述为「GNN 图混合（入口+出口联合）」、A3 量化的是跨通道混合贡献而非整个 GNN 模块、单 seed/五折配对比较）。

## 4. 口径登记（evaluation-calibers）

- M1 best-epoch 由 trainer 按 val loss（含 λ·uni_reg）选择；四臂报 datafit @同 epoch。A3 中 argmin(loss) ≡ argmin(datafit)（常数差不改 argmin）。
- M2 std_ratio full_mean（3 被试 × 30 trial × K=32；确定性采样 ⇒ 32 条逐位相同，口径沿用基线不改 K）。
- M3 z 域、K=1、seed 42 登记；与基线 fc summary 可直接并置。
- M4 描述性并置登记（先例协议）：A_eff=模型内部量 vs GC=观测数据量，实现口径隔离，不构成代理脑资格判定；不与 EC_pert↔GC 数值直接对比（不同量纲/语义，仅各自并置）。

## 5. 风险与回滚

- **风险 1**：γ 置零链路上有未预期守卫/断言（如 sym-w 起跑对称断言只查 W 零初始化，不受影响；但 wrapper 可能存在 γ 相关校验）→ S1 冒烟先验（fold0 × A3 × 2 epoch），失败即修设计再进 S2。
- **风险 2**：A3 臂 loss 曲线可能显著更高/不收敛 → 这正是消融结论的一部分，不是失败；逐 epoch 日志保留供解读。
- **风险 3**：M3/M4 脚本装载任意 ckpt 的 strict-load 键面（exit_* 键）→ 沿用 quality_check_mild/viz_mild 的 `--flow-exit-gnn` 装载路径，冒烟覆盖。
- **风险 4**：共享 GPU 逐位复现性（environment.md 2026-10-29 登记）不适用于本任务——训练不做逐位对拍，评估侧采样本就确定性（单进程串行跑批）。
- 回滚：全部新产物在独立 outputs 子目录 + 新脚本文件，删目录即回滚；不触既有代码与基线产物。

## 6. 执行顺序与检查点

S0 立项（本文件）→ S1 冒烟（fold0×A3×2ep + 守卫演练）→ S2 正式 15 跑（fail-fast 串行）→ S3 M2 ×15 → S4 M3 ×15 + M4 ×15 → S5 汇总 → S6 check/spec/commit。每步完成在 implement.md 勾选；S2/S3/S4 为长任务（后台 nohup，日志轮询）。
