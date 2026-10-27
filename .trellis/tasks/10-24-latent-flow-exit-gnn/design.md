# design — 10-24-latent-flow-exit-gnn（flow 出口 GNN 图混合变体 · 方案 v1 供审）

> 状态：**方案设计，未实施**。用户确认通过后进入实施（实施依赖 w500 任务提交先落地）。
> 代码基线：`latent_flow/models/oc_latent_flow.py`（包体零改动纪律）+
> `train_mild_depress.py`（w500 任务改造后工作区）。

## 1. 修改方案（含维度对接）

### 1.1 修改位点（唯一结构性修改）

`FlowVelocityNet._forward_graph` 尾段（现行 `oc_latent_flow.py:677-682`）：

```
现行：  x = out_norm(x)                        # (B·C, H=128, T_lat=16)
        x = reshape → (B, C=128, T_lat, H)
        x = graph_out(x)                       # 逐通道 Linear(128→64)，零初始化
        return x                               # v (B,128,16,64)

变体：  x = out_norm(x)                        # 不变
        x = reshape → (B, C, T_lat, H)         # 不变
        ── 出口图混合块（新增）─────────────────────
        A = graph_mixer.effective_adjacency()   # (128,128) 有向，与入口共享（默认方案）
        for layer in exit_msg_layers (K=2):    # Linear(128→128) 逐通道
            agg = A @ X （通道轴消息传递）      # 同入口混合器 503-516 行同式
            x = x + GELU(layer(agg))           # 残差式，同入口
        ────────────────────────────────────────
        v = exit_out(x)                        # 逐通道 Linear(128→64)，零初始化（恒等起步保证）
        return v                               # (B,128,16,64)
```

维度对接全链：`out_norm` 出 (B·128, 128, 16) → reshape (B,128,16,128) → 消息传递只在
通道轴（C=128）上做矩阵乘、hidden=128 与 T_lat=16 逐元素保留 → 零初始化投影 (128→64) →
v (B,128,16,64) 喂给单步公式/Euler——**与下游接口零变化**。时间条件不二次注入（hidden
已在入口携带 t_emb 信息；保持最小改动）。

### 1.2 模块一致性（R2）

出口块与入口 `LearnableMexHatMixer` 的消息传递核心**同构**（`oc_latent_flow.py:503-516`）：
同 `agg = A_eff @ X` + `x += GELU(Linear(agg))` 残差形式、同 A_eff 参数化
`γ_M·MexHat(d;σ)+γ_W·W`（`effective_adjacency()` `426-441`）。差异仅两点（均有明确理由）：
①无 in_proj/t_proj——出口输入已是 hidden=128 无需投影，且 t 已在入口注入；②无 K 轮前的
cat/z_obs 条件——出口处理的是 flow 内部表示，无外部条件输入。

### 1.3 实施形态（R4：包体零改动）

- 变体层新文件 `models/latent-flow-v1/flow_exit_gnn.py`：
  `FlowVelocityNetExitGNN(FlowVelocityNet)`——`super().__init__(同参)` 后 `del self.graph_out`
  （避免死参数/死键），新建 `exit_msg_layers`（K×Linear(128,128)，默认 PyTorch 初始化，
  同入口 msg_layers）与 `exit_out`（Linear(128,64)，weight/bias 零初始化）；覆写
  `_forward_graph`（复用 super 构建的 graph_mixer/blocks/out_norm，尾段替换为出口块，
  ~30 行，先例：wrapper 复刻 smoke eval）。
- wrapper `train_mild_depress.py`：`--flow-exit-gnn`（默认关）。开启时 stage2 构建后
  `model.flow = FlowVelocityNetExitGNN(<同参>)`；本地 `_run_stage2_mild` 的 flow 前缀白名单
  补 `"flow.exit_"`（VAE ckpt 装载面：stage1 dense 流 ckpt 无 exit 键，须按"flow 键重初始化"
  容忍，`flow.graph_out` 已在白名单）。
- `viz_mild.py` / `quality_check_mild.py`：`--flow-exit-gnn` 旗标透传至
  `build_model_mild`（strict load 架构匹配）。
- 已知限制（登记）：包内自描述 ckpt 头 `_variant_tag` 不感知本旗标，变体身份凭据 =
  run config json + 装载侧显式旗标（任务内闭环够用）。

## 2. 完整参数配置（出口块）

| 项 | 取值 | 依据 |
|---|---|---|
| K（消息传递层数） | **2** | = 入口 graph_k（一致性；R2） |
| 隐藏维度 | **128**（= flow hidden_ch，无投影直入） | 输入即 hidden，无需变维 |
| 消息层 | `Linear(128→128)` ×2，PyTorch 默认初始化 | 同入口 msg_layers（588-596 构造同式） |
| A_eff | **与入口共享**（默认方案，R3）：σ/W/γ_M/γ_W 单份，前向取 `graph_mixer.effective_adjacency()`，梯度双路累计 | 参数增量最小 + 单一连接假设 + gpfm-v5 跨块共享先例；备选独立 A_eff（σ_exit 中位距初始化、W_exit 零初始化）+16,385 参数 |
| 激活 | **GELU**（消息层后，残差内） | 同入口消息循环 |
| 归一化 | **不新增**（出口前已有 out_norm GroupNorm(8)；入口消息循环内亦无 norm） | 一致性 + 最小改动 |
| 出口投影 | `Linear(128→64)`，**weight/bias 零初始化** | 保 v≡0 恒等起步（见 §4） |
| 残差连接 | **保留**（`x = x + GELU(layer(agg))` 逐轮残差） | 同入口形式；梯度通路上不破坏 out_norm 后的表示 |
| 抗均匀正则 | 沿用 stage2 现行 λ=0.1（作用于共享 A_eff，入口/出口一并约束） | 共享方案下自然覆盖，无需新 knob |

**参数增量（默认共享方案）**：`exit_msg_layers` 2×(128×128+128)=**33,024**；
`exit_out` 8,256 **替换**原 `graph_out` 8,256（净 0，且原键删除不残留）；A_eff 共享 +0。
**净增 +33,024**：flow 488,897 → **521,921**（+6.8%）；全模型 1,243,234 → **1,276,258**
（+2.7%）；stage2 可训练面 903,458 → **936,482**（w500 冻结面口径下）。
备选（独立 A_eff）：+49,409（flow +10.1%）。

## 3. 与原架构差异对比

| # | 变更点 | 原架构 | 变体 | 理由 |
|---|---|---|---|---|
| 1 | flow 出口 | `graph_out` 零初始化逐通道 Linear(128→64)，**无跨通道混合**（注释明言防出口再平均） | 同位置先 K=2 轮 A_eff 消息传递，再零初始化 Linear(128→64) | 用户指定方向：出口引入邻居混合，检验其对生成/连接结构的作用 |
| 2 | A_eff 应用位 | 仅入口 1 处 | 入口 + 出口 2 处（默认共享同一 A_eff） | 单一连接假设下测试双侧应用；备选独立 |
| 3 | state_dict | 含 `flow.graph_out.*` | 删 `graph_out`，增 `exit_msg_layers.*` / `exit_out.*` | 架构即凭据；装载侧旗标匹配 |
| 4 | wrapper/装载 | — | `--flow-exit-gnn`（train/viz/quality 三处）+ runner 白名单补 `flow.exit_` | 变体开关 + ckpt 装载面 |
| 5 | 其余全部 | — | **零变化**：入口混合器、6 膨胀卷积块、时间嵌入、encoder/proj/decoder、两阶段损失、确定性采样、mild w500 数据口径 | 轻量约束（R6） |

## 4. 潜在风险评估

1. **初始行为 = 恒等映射：保留**。零初始化 `exit_out` 使 v≡0 在初始化时严格成立
   （单步公式 `z_fc_hat = z_t + (1−t)·v` 退化为 z_fc_hat = z_t；Euler 采样退化为不动点），
   与原架构初始语义**逐位同型**。出口消息层在首步被零投影门控（梯度先打开 exit_out 再
   传导至消息层——与现行 graph_out/ProjHead 零初始化门控同机制，代码库内已验证的模式）。
   **训练稳定性风险：低**。
2. **出口再平均风险（本变体的核心科学风险）**：原注释明言出口纯逐通道是为防入口带通结构
   被"re-averaged at the exit"。变体重新引入出口邻居混合，若 A_eff 侧演化偏各向同性，
   出口可能抹平入口学到的通道结构。缓解：默认**共享** A_eff（同一带通算子、非 Gaussian
   低通）+ anti-uniform λ=0.1 沿用（直接约束 A_eff 行和均匀化）+ 零初始化门控让模型自主
   决定用多少出口混合。观察指标：val corr2 / uni_reg 走向、|A_eff| 读出、Run 003 同口径
   medoid/real std。
3. **共享 A_eff 的梯度耦合**：σ/W 由入口/出口两路损失共同更新，W 读出解释变为"双侧联合"；
   若需出口独立连接读出 → 备选独立 A_eff（+16,385），属拍板项。
4. **收敛预算**：w500 口径 stage2 在 E150 仍未收敛；本变体可训练面 +3.7%，收敛速度预期
   同量级——若实施并训练，建议与"延长预算"拍板一并考虑（组合实验设计属后续拍板）。
5. **装载面风险（工程）**：旗标漏传 = strict load 失败（viz/quality 拒跑，安全失败模式）；
   变体身份不入 ckpt 自描述头（§1.3 已知限制，config 凭据补位）。

## 5. 实施清单（已获批，2026-10-24 用户三项拍板随附）

**用户拍板（2026-10-24，方案 v1 通过）**：①本变体 stage2 **lr=1e-3**（w500 为 5e-4，
口径差异登记）；②**stage1 沿用** w500 Run 001 VAE ckpt
（`outputs/10-24-latent-flow-mild-w500/run_w500_fold0/checkpoints/fold0_mild_vae_best.pt`，
best 2.0992@E57，--vae-ckpt 只读复用，不重训）；③**A_eff 共享**拍板（弃独立备选）。
其余沿用 w500 口径：batch 64 / epochs 150 / patience 30 / seed 42 / anti-uniform 0.1 /
decoder+flow 联合（仅冻结 encoder）/ 确定性采样。

S1 `flow_exit_gnn.py` 子类（构造 + 覆写 `_forward_graph` + 初始 v≡0 自验）→ S2 wrapper
旗标 + runner 白名单（`flow.exit_` 前缀）+ 口径守卫纳入 `--flow-exit-gnn` → S3 viz/quality
透传 → S4 冒烟（trainable 打印 936,482、graph_out 键消失/exit 键在、初始 v≡0、默认关 bit
兼容、守卫拒跑）→ S5 训练 Run（登记 experiment.md）：

```bash
RUN=/root/npi/outputs/10-24-latent-flow-exit-gnn/run_exitgnn_fold0 && mkdir -p $RUN/logs && \
python3 models/latent-flow-v1/train_mild_depress.py --fold 0 --stage 2 \
  --window-len 500 --obs-len 375 --forecast-len 125 \
  --batch-size 64 --lr 1e-3 --epochs 150 --patience 30 --seed 42 \
  --unfreeze-decoder --anti-uniform-lambda 0.1 --flow-exit-gnn \
  --vae-ckpt /root/npi/outputs/10-24-latent-flow-mild-w500/run_w500_fold0/checkpoints/fold0_mild_vae_best.pt \
  --run-dir $RUN > $RUN/logs/stage2.log 2>&1
```

S6 同口径质量抽查 + viz（`--flow-exit-gnn` 透传，triad 1/7/21）→ S7 check + 收尾。
产物目录：`outputs/10-24-latent-flow-exit-gnn/`（声明隔离；w500 ckpt 只读复用，不写其目录）。
