# design — gpfm-mamba-v5：v2 基线 × 轴分解交替架构（GNN 先验矩阵融合）

> 任务：`10-19-gpfm-mamba-v4b-gnn-f0`（study1）。
> **2026-10-19 二次转向（用户拍板）**：弃用 v4b 基座路线（S1/S2 产物 `models/gpfm-mamba-v4b-gnn/` 停用留档，experiment.md Run 001/002 为其记录）；改为**以 v2 为基线的新变体 v5**——不复用 v4b 的架构/参数/实现细节。研究内核不变：融合 latent-flow 的 GNN 先验矩阵学习（`A_eff = γ_m·M(d;σ) + γ·W`）。
> 修订链：v4bg 双 mandate 评审处置（方向约定/判定读出/运行时契约）中仍适用于 v5 的部分随文保留。

## 0. 假设与判读框架（H4，随转向更新）

- H3 已判（10-16）：v2 自由稠密跨通道（d_model=62 瓶颈消融）GC r 0.2755 档不回升（v2 主 run 0.231 档）；v4b 稠密 cMLP head 亦不回升。10-17 定位 θ* 缺口 = 跨通道传导弥散无特异性。
- **v5 的 H4'**：显式几何先验约束的空间块（带通距离核 + 可学习有向 W）与时间块**逐块交替**（相对 v2 的"通道仅经 Mamba 内部投影隐式混合"），能否成为 GC 载体 / 给传导以结构。
- 三态判读与 δ=0.02 噪声框、判定读出（GC(real) 距离偏 Spearman 判定轨 + GC(gen) 描述轨）**原样沿用**（v4bg design 版本，用户已采纳）；对照列改为 **v1 / v2 / v5**（v4a/v4b 列不进本变体对照；用户 10-19 拍板不进 d8）。
- 归因口径：v5 vs v2 的差异 = 交替结构整体（空间块 + 逐块 LN），非单变量——对照 v2 干净陈述，不再主张 trunk 逐位同构。

## 1. v2 基线事实（改造出发点）

v2 = 薄变体（`models/gpfm-mamba-v2/`：config.py + train/evaluate 包装，config 遮蔽 sys.modules["config"]，模型本体只读复用 v1 `VelocityNetwork` 以 d_model=62 实例化）：

```
输入 (B, L=512, 79)  # 62 通道值 + seg 1 + temb 16（future 位）
→ in_proj Linear(79→62)                # (B, 512, 62=d_model=通道轴)
→ 4× MambaBlock(62)：x + Mamba(RMSNorm(x))   # 时间轴 L 扫描；通道混合仅经 Mamba 内部投影（隐式）
→ head_norm RMSNorm(62) + head Linear(62→62) # (B, 512, 62)
→ 截取 future 128 → v_θ (B, 62, 128)
```

用户四项要求（2026-10-19）：① 不用 v4b、通道不折叠进 batch、通道不只经 Mamba 处理；② v2 接口/config 方式/训练推理调用流兼容；③ **d_model = C = 62**，不以折叠/合并/投影改变该对应（v2 自身的 in_proj 79→62 与 head 62→62 为基线固有，保留）；④ 时间块/空间块交替。

## 2. v5 结构（轴分解交替）

新目录 `models/gpfm-mamba-v5/`：`config.py` + `model.py` + `graph.py` + `train.py`/`evaluate.py` **双遮蔽**包装（遮蔽 config + model 两模块，v4 惯例；run_id 前缀 `v5-`，checkpoint 名 `gpfm-mamba-v5-{best,last}.pt`）。v1 的 data/gpr_prior/losses/train 循环只读复用。

### 2.1 前向（维度链）

```
输入构造 = v2 逐字（seq+seg+temb → (B,512,79)）
→ in_proj Linear(79→62)                          # v2 原样
→ 4 × [ 时间块 → 空间块 → LayerNorm ]：           # 交替主体（n_layers=4，逐块）
     时间块  = MambaBlock(62)：x + Mamba(RMSNorm(x))     # (B,512,62)，v1 mamba.py 只读复用，时间轴 L
     空间块  = x + GELU(Linear_i(A_eff @_axis x))        # (B,512,62)；A_eff 通道轴显式聚合
                                                             agg[b,l,i] = Σ_j A_eff[i,j]·x[b,l,j]
     LayerNorm = nn.LayerNorm(62)（逐块实例，post-norm，无残差包裹）
→ head_norm RMSNorm(62) + head Linear(62→62)      # v2 原样
→ 截取 future 128 → v_θ (B, 62, 128)（return_all=True 全 512，因果性单测口径）
```

- **维度不变量**：全程 (B, 512, 62)，d_model = 通道轴 = 62，零折叠/合并（要求③满足）。
- **交替与重复**：复合块 ×4；每时间块后恰一空间块 + 一 LN（要求④满足；顺序 = 用户指定的 Mamba→GNN→LN）。
- **残差**：时间块内残差（v2 原样）；空间块残差 `x + GELU(Linear(A_eff x))`；LN 为裸 post-norm。
- **归一化清单**：4×（MambaBlock 内 pre-RMSNorm + 块后 LayerNorm(62)）+ head RMSNorm(62)（v2 原样）。norm_type 钩子仍控 MambaBlock 内 norm（默认 rmsnorm）；块后 LN 固定 LayerNorm（用户指定，不走钩子）。

### 2.2 A_eff 空间算子（latent-flow LearnableMexHatMixer 机制，v5 目录内自含实现）

- `A_eff = γ_m·M(d;σ) + γ·W`：(62,62)，源语义 **A[i,j] = 通道 j 驱动 i（行=靶/列=源）**；对外读出统一转置为行=源/列=靶（转置在 graph_readout 单点，A-1 处置沿用）。
- ricker（默认，σ 可学习 log_sigma 参数化）/ dog（4 参数重参数化）双核；γ/γ_m/pairwise_d 为 buffer；W (62,62) 零初始化、不对称化（有向），支持 w_init 注入。
- **跨块共享单实例**（唯一可解释连接矩阵——R5 读出物；每块仅 msg Linear_i 独立：4× Linear(62,62)）。
- `uniformity_penalty` / `sparsity_penalty` / `dog_params` 保留源语义；λ_u=λ_s=0 默认不进损失（口径沿用）。
- 几何单一 choke point：构造内加载 `geometry_edra_fmt.pt` + 全 46 例通道名顺序全等断言（lru_cache；A-7 沿用）。**与 v4bg 的 graph_head.py 无代码复用关系**（按用户要求不复用 v4b 线实现；公式源 = latent-flow 源码，等价测对源断言）。

### 2.3 接口与运行时契约（要求②）

- `VelocityNetwork` 签名 = v2 全集（n_channels/past_len/future_len/d_model/n_layers/d_state/d_conv/expand/time_emb_dim/use_fast_path + bottleneck_rank/norm_type/cond_adaln + graph_* 新参在类默认）；**钩子三参接受并断言默认值**（None/"rmsnorm"/"off"，非默认报错——B-2 处置沿用；v1 train.py/sample.py 无条件透传可直跑）。
- d_model 与 n_channels 一致性断言（d_model=62=C；不一致即报错——要求③的机器可读防线）。
- EMA/time_embedding 复制自 v1（勿单独演化惯例）。
- config.py = v2 副本 + `variant_name="gpfm-mamba-v5"` + graph 字段（graph_kernel/graph_gamma/graph_gamma_m/graph_w_init/graph_lambda_uniform/graph_lambda_sparse/geometry_path）+ d_model=62；config_hash 稳定性要求同 v2。

### 2.4 注册（要求⑤）

- 代码：`models/gpfm-mamba-v5/`（config.variant_name + checkpoint 主名 + run 前缀三处一致）。
- 配置：`build_config(**overrides)` 惯例，`python3 models/gpfm-mamba-v5/train.py --run-id v5-xxx` 经双遮蔽以 v5 config + v5 model 实例化。
- registry：`.trellis/research/context.md` 技术路线表 gpfm-mamba 行变体注记回填 v5（本转向落档时同步，不等 3.3）。

### 2.5 参数量预算（实现后精确登记）

v2 全量（d_model=62）+ 增量：4×(msg Linear 62×62+62 = 3,906 + LN 62×2 = 124) = 16,120；A_eff 侧 W 3,844 + σ 1。合计 ≈ **v2 + 19,965**。显存：交替结构保持 (B,L,62) 激活（无 CI 折叠的 B·C 膨胀），batch 256 量级沿用 v2 预算，bench 实测为准。

## 3. 训练/评估（后续 run 阶段适用，本轮不做）

fold0 32/4/10（v1 固化 manifest 只读，经 v2 同款校验路径）、batch 256、seed 1234、EMA 0.999、早停 10、恒定 lr 1e-3、gen_lowpass 40 Hz 正式口径；R9 链路（smoke→bench→用户确认→主 Run）；评估 Run 017 同口径，对照列 **v1 / v2 / v5**；A_eff/W 读出（EMA shadow、转置、距离偏 Spearman 判定轨）沿用 v4bg 版处置。

## 4. 单测清单（`tests/test_v5.py`）

1. 构建 + 前向 shape：(B,62,128) / return_all (B,62,512)；dtype/设备；
2. **交替结构断言**：trunk 子模块序 = [时间块, 空间块 msg_i, LN] ×4（named 结构 + 前向钩子序验证"每时间块后恰一空间块一 LN"）；
3. 因果性：return_all 口径下扰动 future 位不改 past 位输出（时间因果）；空间块位置逐点（扰动 l 位只影响 l 位——空间轴不串时间）；
4. **方向单测**：γ_m=0 + w_init 单边注入（W[i,j]=1），扰动通道 j ⇒ 仅通道 i 出现跨通道响应、反向注入不响应（B-3 沿用）；
5. 参数量精确断言（±0，对照 §2.5 预算）；
6. config↔model 一致 + config_hash 稳定 + **d_model==n_channels 断言**正反例；
7. **A_eff 等价测**：同 coords/σ/γ/W 下与 latent-flow 源 `LearnableMexHatMixer.effective_adjacency` allclose rtol=1e-6（ricker+dog；源经 sys.path 只读导入）；
8. 双遮蔽 + run_id 前缀守卫（v5-）+ 钩子三参默认接受/非默认报错；
9. 几何门禁 46 例通道名断言正反例；λ 旋钮（λ=0 损失逐位不变——组件级）；
10. 回归：v1 七测 / v4 test_v4b / v4a 测全绿 + v1 `c6ff80ac1c43` / v2 `fc2a9ce14948` 零漂移。

## 5. 与 v2 的架构差异陈述（要求⑥交付，汇报口径）

| 维度 | v2 | v5 |
|---|---|---|
| 通道处理 | 仅 Mamba 内部投影隐式混合（d_model=62 全程） | 逐块显式空间块：A_eff 通道轴聚合（几何先验 + 有向 W） |
| 块结构 | 4×[pre-RMSNorm→Mamba→残差] 连续 | 4×[Mamba→GNN 空间块→LayerNorm] 交替复合块 |
| 归一化 | pre-RMSNorm ×4 + head RMSNorm | 前者全保留 + 逐块 post-LayerNorm(62) ×4 |
| 可解释连接矩阵 | 无（混合弥散于投影权重） | 单一 A_eff（跨块共享）：M(d;σ) 先验 + W 残差，有向 |
| 接口/config/调用流 | — | 逐项兼容（签名/双遮蔽包装/v1 训练循环只读复用） |

## 6. 边界与零改动

- 本轮交付 = 结构 + 单测 + 最小验证（构建/前向维度/差异陈述）；**不启动任何训练 run**（smoke/bench/主 Run 属后续阶段，主 Run 仍有 R9 门）。
- 零改动：v1*/v2/v3/v4*/v4a*/latent-flow-v1（含 3 处悬置未提交文件）/common/；v4bg 目录停用留档不删不改。
- 产物（后续训练阶段）：`outputs/gpfm-mamba-f0/v5-*/`。

## 7. 回滚

删 `models/gpfm-mamba-v5/`（+ 后续 outputs v5-* 目录）即净；v4bg 停用分支同理独立可删。
