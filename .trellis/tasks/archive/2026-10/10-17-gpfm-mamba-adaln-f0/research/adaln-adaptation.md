# Research: adaLN（流时间 t 条件调制归一化）适配 gpfm-mamba v4a-fmt 速度网络

- Query: 把 v4a-fmt 速度网络 5 个 LayerNorm 位（4×MambaBlock pre-norm + head_norm）替换为 adaLN——emb(t) 经 MLP 生成 scale/shift 调制归一化后特征
- Scope: mixed（内部代码勘察 + 外部 adaLN/DiT 文献核对）
- Date: 2026-10-17
- 触发：1.0c 新路径检测（adaLN 不在 context.md 技术路线表登记变体内；表内变体 v1/v2/v4b/v4a/v1-fmt/v4a-fmt 均无 t 条件 norm 机制）

---

## 0. 适配结论（推荐方案 + 理由）

**推荐：A 案（纯 scale/shift）+ 逐 site 独立 SiLU→Linear 调制 MLP + 全序列（含 past 段）per-sample 全局调制 + 复用 time_embedding(t,16) 原始正弦特征 + 零初始化（初始严格等价 v4a-fmt LayerNorm 基线）。**

| 设计维度 | 推荐 | 理由摘要 |
|---|---|---|
| adaLN 形式（PRD Q1） | A：`(1+s)⊙LN(x)+b`，残差结构不动 | 初始前向与基线**逐位一致**（PRD 验收②字面可满足）；对 v4a-fmt 是**真单变量**（唯一差异 = norm 条件化 + 登记参数增量）；130k 小网络 + LN 已稳定收敛（基线 55ep 早停），无 DiT 百层深网所需的 zero-gate 稳定性收益 |
| 调制信号 | 复用 `time_embedding(t,16)` 原始 (B,16) 向量，不加共享投影 | 零新嵌入参数；16→124 已是升维，逐 site MLP 内的 SiLU 提供非线性；DiT 式共享投影变体 +40,114 参数（+30.8%）违背单变量最小容量原则 |
| 调制 MLP 形态 | 逐 site 独立（DiT 逐 block 先例），键落位各 norm 旁 | 与共享单 Linear(16→620) **参数量与函数类完全等价**（均为 R^16→R^620 线性映射，10,540 参数），差异仅在 state_dict 键布局与代码局部性；逐 site 与 blocks.{i} 键共位、后续可单独归因 |
| 调制范围 | per-sample 全局 (B,1,62) 广播到 L=512 **全序列（含 past 段）** | 采样语义自洽（见 §1）；DiT/UNet-FiLM 先例均为全 token 调制；仅 future 段调制会在 past/future 边界引入调制跳变（因果 SSM 扫描瞬态）且无任何计算节省 |
| default-off 钩子 | config 字段 `cond_adaln: str = "off"`（"off"\|"scale_shift"\|"zero"），非 `norm_type="adaln"` | adaLN 需 forward 传入调制参数，非纯 nn.Module 替换——与 build_norm 签名正交；枚举字符串先例 = lr_schedule/kernel_t；两案并存期枚举免二次改 config |
| 变体目录 | `models/gpfm-mamba-v4a-fmt-adaln/`（薄变体三件套，无 --rank） | PRD R2 指定；r=16 固定 → wrapper 比 v4a-fmt 更简 |

**B 案（adaLN-Zero）结论**：技术上成立且有 DiT 大规模证据，但 (a) 初始前向 ≠ 基线（mamba 支路零门控致死，PRD 验收②须改写为逐块恒等断言）；(b) 改残差结构，对比解释复杂化（PRD Q1 已自认）；(c) 零门控引入梯度耦合——g=0 时 mamba 权重梯度恒 0，须先 g 逃零才轮到 trunk 学习，叠加 EMA 0.999 滞后在小网络短程训练不利。若 A 案出现不稳定（本规模概率低）再回退 B。

---

## 1. 调制信号设计（研究问题 1）

### 1.1 emb(t) 来源：复用 time_embedding，不加投影

- `models/gpfm-mamba-v1/model.py:24-33`：`time_embedding(t, dim)` = sin/cos 各 dim/2 频率（1→1e-4，×1000 尺度），输出 (B, 16)。当前唯一消费点在 `model.py:94-95`（future 128 位置拼接进输入投影，past 段置零）。
- adaLN 直接复用该 (B,16) 原始特征：`temb = time_embedding(t, self.time_emb_dim)`（forward 内再调一次，代价可忽略；default-off 分支代码保持不动）。逐 site MLP = `Sequential(SiLU(), Linear(16, 2×62))`——SiLU 即投影非线性，无需 DiT 的共享 cond MLP（DiT 的 `y` 还承载 class label，本项目只有 t 一个条件源）。
- 平滑性红利：sin/cos 特征有界 [-1,1] 且对 t 光滑 ⇒ (s,b)(t) = W·SiLU(temb)+b 对 t Lipschitz（界 ∝ ‖W‖）——Euler 16 步网格相邻 t 间调制跳变有界，无轨迹稳定性风险。
- 语义注意（登记）：t 的输入侧注入（仅 future token）与调制侧注入（全 token）**并存且不对称**——这是 DiT 同款结构（DiT timestep 既拼 token 也驱动 adaLN），不是缺陷，但写入 design 差异说明。

### 1.2 全序列调制与采样语义自洽性（结论：自洽）

- 现状因果性事实：temb 只加在 future 位置输入（`model.py:94-95`），Mamba 因果单向 ⇒ **past 段（0..383）表征当前严格 t 无关**（future 位置看不到，past 自己不带）。
- adaLN 全局调制后 past 表征变 t 依赖。采样侧无任何矛盾：`sample.py:128-132` Euler 每步对完整 512 token 重新前向（无跨步缓存/状态复用），v_θ 是 (x_past, x_t, t) 的纯函数——"past 编码随查询时刻 t 变化" 是良定义的全局条件机制，等价于 DiT 对全部 token 做 timestep 调制。训练（train.py:357 逐样本 t）、内部验证（train.py:162-166 固定 t 网格标量广播）、采样（sample.py:130 每步标量广播）三条链路的 t 形状均兼容 (B,)→(B,1,d) 广播。
- 替代案「仅 future 段调制」及其代价：(a) 需 (1,L,1) 掩码乘法，position 384 处调制系数从 1+s 跳回 1（或 0），因果 SSM 扫描在边界吸入一个瞬态——序列对位置的平稳性被人为破坏；(b) 无计算节省（全前向照跑，past 表征反正每步重算）；(c) 非标准（DiT/ADM/FiLM 均全位置调制）。**不推荐**；若用户坚持可作为后续消融，但属于非单变量加项。

---

## 2. 调制 MLP 形态（研究问题 2）

**逐 site vs 共享单 MLP 的精确关系**：5 个独立 Linear(16→124) 与一个 Linear(16→620) 在参数量（10,540）、函数类（R^16→R^620 全线性映射，前者堆叠即后者）、输入相同激活相同下的梯度动力学上**完全等价**——差别只有 state_dict 键布局与代码局部性。

| 方案 | 参数增量 | 总参数 | 评价 |
|---|---|---|---|
| 逐 site SiLU→Linear(16→124) ×5（**推荐**） | +10,540 | 140,906（+8.08%） | DiT 逐 block `adaLN_modulation` 先例；键 `blocks.{i}.mod.*` 与各 norm 共位；head 位独立 `head_mod` |
| 共享单 Linear(16→620) | +10,540 | 140,906 | 等价物；但需中心化切分再逐 site 下传，签名管线更丑，无任何收益 |
| DiT 两段式（共享 SiLU→Linear(16→62) + 逐 site Linear(62→124)） | +40,114 | 170,480（+30.77%） | DiT 的 cond MLP 承载 label+t 两源且 hidden=1152；本项目单一 16 维 t 源，纯增容量，违背单变量最小容量登记惯例（先例：+310 bias 都逐位登记）——排除 |

repo 风格判据（`.trellis/spec/research/reproducibility.md:10` 单变量钩子 + test_v4afmt.py:54-55 容量逐位登记）：选逐 site。

---

## 3. 两种 adaLN 形式对比（研究问题 3，PRD Q1 两案）

DiT 官方实现核对（facebookresearch/DiT models.py）：`modulate(x, shift, scale) = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)`；block 残差 `x = x + gate.unsqueeze(1) * sublayer(modulate(LN(x), shift, scale))`；adaLN_modulation = `Sequential(SiLU(), Linear(h, 6h))` **最后线性层权重+偏置全零初始化**；FinalLayer 只出 2h（shift/scale，**无 gate**——与本项目 head_norm 位同构）。

### A 案：纯 scale/shift（推荐）

- 接入：4×MambaBlock 内 `x + self.mamba( modulate(self.norm(x), s, b) )`；head 位 `self.head( modulate(self.head_norm(h), s_h, b_h) )`。残差结构、mamba 支路、head 全部不动。
- 初始等价性：W=0,b=0 ⇒ s=b=0 ⇒ `(1+0)⊙LN(x)+0`。浮点事实：乘 1.0 与加 0.0 对有限浮点值逐位不变 ⇒ **初始前向与 v4a-fmt LayerNorm 基线逐位一致**，PRD 验收②字面成立（epoch-0 损失 = 基线 epoch-0 损失，归因干净）。
- 稳定性预期：130k 小网络、LN 已归一、调制是有界乘性扰动；恒 lr 1e-3 + grad_clip 0.5 + EMA 0.999 的现有纪律已把基线带到 55ep 稳定早停——A 案不引入新的失稳自由度。DiT 消融中 adaLN（无 gate）已优于 in-context/cross-attention，adaLN-Zero 的额外收益在极深网络才显著（DiT 观察：zero-init 残差帮助深层训练；本项目 4 块）。
- 解释纯净度：对 v4a-fmt 的唯一差异 = "5 个 norm 位从常量仿射（LN 可学习 γβ）变为 t 条件仿射"，参数增量一行登记——**真单变量**。

### B 案：adaLN-Zero（额外残差门控）

- 接入：4×MambaBlock `x + g⊙self.mamba( modulate(self.norm(x), s, b) )`（g 零初始化）；head 位与 A 案相同（DiT FinalLayer 先例：无残差支路处无 gate；head 前加 gate 与 head Linear 权重代数退化重参数化，无意义）。
- 初始等价性：g=0 ⇒ 每 MambaBlock 初始 = 恒等 ⇒ 整网初始 = `head(head_norm(in_proj(x)))`——**mamba trunk 初始死亡，初始前向 ≠ 基线**。PRD 验收②须改写为"恒等初始化下每个 MambaBlock 输出=输入（逐位）"。
- 梯度耦合（关键风险）：g=0 处 ∂L/∂(mamba 权重) ∝ g = 0——trunk 权重初始无梯度，须先 g 逃零（∂L/∂g ∝ mamba(norm(x)) ≠ 0 可逃）才级联解锁。DiT 在大网络+长程训练下成立；本项目 EMA 0.999（时间常数 ~1000 step）叠加早停 10，评估用的 EMA 影子权重中调制/门控参数滞后原始权重，短程训练的 best 判据（train.py:146 EMA swap 后复算）看到的近恒等网络期更长。
- 解释纯净度：残差结构改变 ⇒ 与基线对比是两个变量（norm 条件化 + 门控残差），PRD Q1 自认"对比解释复杂化"。
- 参数：+14,756（145,122，+11.32%），其中 gate 列 4×62×17=4,216。

### head_norm 位两案共性

head 无残差支路 ⇒ 两案 head 位相同（scale/shift only，DiT FinalLayer 同构）；差异全部集中在 4 个 MambaBlock。

---

## 4. default-off 钩子形态（研究问题 4）

### 4.1 config 字段：`cond_adaln: str = "off"`（推荐）

- 取值 `"off" | "scale_shift" | "zero"`：枚举字符串先例 = `lr_schedule`（train.py:103-108）与 `kernel_t`（config.py:68）；非法值构建期 ValueError 先例 = build_norm（mamba.py:25）/build_lr_scheduler（train.py:108）。
- 透传：构造点 `cond_adaln=str(getattr(cfg, "cond_adaln", "off"))`——旧六变体 config 无此字段 ⇒ "off" ⇒ 零漂移（先例 norm_type：train.py:279 / sample.py:53）。
- **否决 `norm_type="adaln"`**：`build_norm(d_model, norm_type)`（mamba.py:19-25）返回纯 nn.Module（零参构造）；adaLN 调制需要 forward 期传入 (B,16) 条件张量并携带一个参数化 MLP——塞进 build_norm 会让返回类型带隐藏状态/签名耦合，且 adaLN 变体的**底层 norm 仍是 LayerNorm**（norm_type="layernorm" 保持不变），两维度正交就该两个字段。

### 4.2 forward 签名扩展（默认分支零漂移）

- `MambaBlock.forward(self, x, mod=None)`（mamba.py:48-49 扩展）：`mod is None` 走原式 `x + self.mamba(self.norm(x))`（逐位不变）；非 None 走 A/B 分支。s,b（,g）由 mod 经 `self.mod` 在块内自产（下传原始 temb 而非成品 s/b——保持 DiT 逐块自产形态且签名最瘦）。
- **backbone 不可用 Sequential 直传额外参数**：`build_mamba_backbone`（mamba.py:52-66）返回 nn.Sequential，`model.py:97` 一把 `self.blocks(h)`。改 ModuleList 虽键名不变（Sequential/ModuleList 子项同名注册 "blocks.0..."），但 default-off 分支必须保留原 `self.blocks(h)` 单行——故 adaLN 分支改为显式循环：`for blk in self.blocks: h = blk(h, temb)`。两分支条件切换，默认分支零改动。
- `VelocityNetwork.forward`（model.py:82-101）签名**不变**（t 已在参里）：adaLN 分支开头多算一次 `temb = time_embedding(t, self.time_emb_dim)`，块循环 + head 位 `modulate(self.head_norm(h), ...)`；default-off 分支逐字节保持现行 4 行。
- 构造签名：`MambaBlock/__init__` 与 `build_mamba_backbone` 各加 `cond_adaln: str = "off", mod_dim: int | None = None`（mod_dim=time_emb_dim，仅启用时使用）；`VelocityNetwork.__init__` 加 `cond_adaln: str = "off"`（排在 norm_type 后），启用时建 `self.head_mod = Sequential(SiLU(), Linear(time_emb_dim, 2*d_model))`，关闭时**不创建任何新属性/参数**（结构零漂移）。

### 4.3 state_dict 键设计

| 键 | 形状 | 说明 |
|---|---|---|
| `blocks.{i}.mod.1.weight` | (124, 16) | Sequential(SiLU, Linear) ⇒ Linear 落索引 1（SiLU 无参占 0）；i=0..3 |
| `blocks.{i}.mod.1.bias` | (124,) | B 案为 (186,)（s/b/g） |
| `head_mod.1.weight` / `head_mod.1.bias` | (124,16) / (124,) | head 位调制（两案同形） |

- 命名先例：与 `blocks.{i}.norm.*`、`head_norm.*`（test_v4afmt.py:229-235 断言式）平行对仗。
- EMA/ckpt：EMA 对 state_dict 键全泛型（model.py:112-123 逐键遍历）——新键自动纳入 shadow/update/swap；save_ckpt（train.py:186-206）存整 state_dict——自动覆盖。**零改动**。
- 初始化：`nn.init.constant_(mod[-1].weight, 0)` + `bias 0`（DiT initialize_weights 同款；A/B 两案同法——A 案零初始化即 LayerNorm 等价，B 案零初始化即块恒等）。

### 4.4 评估/采样链路改动面（结论：构造点透传各一行，其余零改动）

- `train.py:273-280` 与 `sample.py:47-54` 两个构造点各加一行 getattr 透传（先例 norm_type 同位）。
- `evaluate.py` 经 `from sample import ... load_eval_model`（v1 evaluate.py:40,274）复用装载——零改动；`evaluate_internal_val`（train.py:166 `model(xp, x_t, t)`）与 `EulerSampler`（sample.py:131 `self.model(xp, x, t)`）调用签名不变——零改动；`fc_metric.py` 不构建模型（消费 evaluate 产物，v4a-fmt/fc_metric.py:1-16）——零改动。
- config_hash 严格校验（sample.py:42-46）天然把 adaLN 变体 ckpt 与 v4a-fmt 隔离（变体 config 多 cond_adaln 字段 ⇒ hash 不同 ⇒ 跨变体 resume/评估结构性拒绝）。

### 4.5 确定性/CUDA 链路

- cudnn_deterministic（config.py:116）：调制 = 逐元素广播乘加（确定性内核），mamba_ssm 核完全不动——无影响。
- EMA swap（model.py:136-145）全量 load_state_dict——键匹配由构造期结构决定，无破坏点。

---

## 5. CUDA fast path 交互（研究问题 5：无风险）

- 调制作用点在 `self.norm(x)` 输出之后、`self.mamba(...)` 输入之前（mamba.py:48-49 现调用式），以及 head 之前（model.py:98）——`mamba_ssm.Mamba`（d_model=62, expand=2, use_fast_path=True，mamba.py:40-46）的输入形状/ dtype/ 连续性均不变（广播算术结果天然 contiguous）。
- fast path（mamba_ssm 2.3.1 + causal_conv1d 1.7.0 fused selective-scan autograd Function）封装在 Mamba.forward 内部，对外就是普通 (B,L,62)→(B,L,62) 模块调用——调制不触碰内核。**确认零兼容风险**。

---

## 6. 参数量核算（研究问题 6；d_model=62, time_emb_dim=16, 5 sites，基线 130,366 = test_v4afmt.py:55 pin）

| 方案 | 调制参数 | 增量 | 总参数 | 相对基线 |
|---|---|---|---|---|
| A 案逐 site（Linear(16→124)×5，推荐） | 每 site 16·124+124=2,108 | +10,540 | **140,906** | +8.08% |
| A 案共享单 Linear(16→620) | 16·620+620 | +10,540 | 140,906 | +8.08%（与上行等价） |
| B 案（4×Linear(16→186) + head Linear(16→124)） | 块 3,162×4 + 头 2,108 | +14,756 | **145,122** | +11.32% |
| （参考）DiT 两段式含共享投影 | 1,054+7,812×5 | +40,114 | 170,480 | +30.77%（排除） |

B−A 差 = 4,216 = 4 个 gate 列（4×62×17）。登记口径沿用 test_v4afmt.py:54-55 惯例（V4AFMT_LN_PARAMS 精确整数 pin）。

---

## 7. 风险清单（研究问题 7）

| # | 风险 | 评估 | 缓解/登记 |
|---|---|---|---|
| 1 | 零初始化调制 MLP 学不动 | 低。∂L/∂W 在 W=0 处 ∝ SiLU(temb)·上游梯度 ≠ 0（temb sin/cos 非零；GPT-2/DiT 零初始化先例均可逃零） | 监控既有 v_norm/t_bucket_mae（train.py:375-383 无需新增代码） |
| 2 | t_loc=-0.8 logit-normal 偏态：训练 t 集中 ≈[0.15,0.47]（P(t>0.7)≈3%），而评估 t 网格含 0.9、Euler 网格均匀铺满 [0,1]——(s,b)(t) 在稀有 t 区外推 | 中低。(s,b) 是有界 sin 特征的线性映射 ⇒ 外推 Lipschitz 有界、平滑无爆炸；且输入侧 temb 注入已带完全相同的外推性质（同一嵌入），非新增风险类 | design 登记；判读时留意 Euler 末端（t→1）步行为 |
| 3 | EMA 0.999 + 零初始化：影子权重从 0 起步，时间常数 ~1000 step，评估/best 判据用 EMA（use_ema_for_val=True, train.py:146）⇒ 有效调制滞后原始权重，早期 epoch val 近似基线 | 中（既有机制，非破坏）。短程训练（早停 10、~50ep 量级）下 best 可能落在调制未充分生效的 epoch | 无需代码动作；experiment.md Run 块登记该机制；若 val total 迟迟不动可查 metrics.jsonl 的 lr/v_norm 走势 |
| 4 | B 案专属：g=0 ⇒ mamba trunk 权重零梯度，级联解锁慢；叠加风险 3 滞后 | 中高（相对 A） | 选 A 案即规避；B 案须接受并登记 |
| 5 | weight_decay 1e-4（AdamW 解耦）持续把调制 W 拉向 0 ⇒ 轻微向恒等回归的先验 | 低（与所有权重同等对待） | 登记；不做特殊分组（避免复杂化） |
| 6 | grad_clip 0.5 全局范数分母多了调制参数 | 低 | 语义不变，登记即可 |
| 7 | 评估侧遗漏 | 低：仅 train.py/sample.py 两构造点各一行；evaluate/fc_metric/EulerSampler/内部验证零改动（§4.4 已逐点核尽，全仓库 VelocityNetwork 构造点仅此两处 + 各 tests） | 单测断言 adaLN 分支构造键齐全 + load_eval_model round-trip |
| 8 | 单测回归面：v1 model.py/mamba.py 改动波及 v1/v2/v3/v4a/v1-fmt/v4a-fmt 六变体 hash pin 与结构测试 | 按惯例可控：default-off 分支逐位一致 + 旧变体 config 无新字段 ⇒ hash pin（如 V4AFMT_HASH_DEFAULT=05d9bb875a2c）不变 | 沿用 test_fmt_hooks.py ① 跨变体零漂移断言式，新增 adaLN 三断言（默认零漂移/结构与键/恒等初始化等价） |

---

## 8. 可执行方案要点（design 阶段输入）

1. **v1 共享库**（`models/gpfm-mamba-v1/`，default-off 钩子 + getattr 透传，零漂移断言必测）：
   - `model.py`：`VelocityNetwork.__init__` 加 `cond_adaln: str = "off"`；启用时 `self.head_mod = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, 2*d_model))`（零初始化）并把 `cond_adaln/time_emb_dim` 透传 `build_mamba_backbone`；forward 加 adaLN 分支（`temb = time_embedding(t, ...)` → 逐块 `h = blk(h, temb)` → head 位 modulate），默认分支不动。
   - `mamba.py`：`MambaBlock` 加 `cond_adaln="off"` + `mod_dim=None`；启用时 `self.mod = nn.Sequential(nn.SiLU(), nn.Linear(mod_dim, 2或3×d_model))`（零初始化）；`forward(x, mod=None)`；`build_mamba_backbone` 透传（返回仍 nn.Sequential，键名不变）。
   - `train.py:279` / `sample.py:53` 后各加一行 `cond_adaln=str(getattr(cfg, "cond_adaln", "off"))`。
2. **变体目录** `models/gpfm-mamba-v4a-fmt-adaln/`：config 完整副本 + 唯差异 `cond_adaln="scale_shift"`（若 Q1 拍 A）+ `variant_name="gpfm-mamba-v4a-fmt-adaln"` + `outputs_root="/root/npi/outputs/gpfm-mamba-adaln-f0"` + `split_manifest_override` 指本目录 splits/（与 v4a-fmt manifest **逐字节一致**断言，先例 test_v4afmt.py:201-204）；train/evaluate/fc_metric 包装三要素（RUN_PREFIX 建议 `v4afmt-adaln-`，无 --rank——r=16 固定，wrapper 比 v4a-fmt 更简）。
3. **单测**（tests/test_v4afmt_adaln.py，对仗 test_v4afmt.py 四段式）：①config 字段 + hash pin + 旧变体零漂移；②遮蔽/守卫；③manifest 逐字节一致；④结构面——5 调制位键、参数 pin（A 案 140,906）、**恒等初始化下前向输出与 norm_type="layernorm" 基线逐位一致**（A 案；同 seed 同输入 allclose+逐位相等断言，先例 test_fmt_hooks.py:170-175 同 seed 双构建对比式）、mod=None 默认分支与旧 MambaBlock 逐位一致。
4. **初始化**：全部调制 MLP 的 Linear `weight=0, bias=0`（DiT initialize_weights 同款）。
5. **登记**：experiment.md 差异声明 = "norm LayerNorm→adaLN(t)（5 位，A 案 scale/shift）+ 参数 130,366→140,906（+10,540）"，结论标注单次运行。

---

## 9. 引用锚点（内部）与外部参考

### 内部代码

| 锚点 | 事实 |
|---|---|
| `models/gpfm-mamba-v1/model.py:24-33` | time_embedding 正弦嵌入 (B,16) |
| `models/gpfm-mamba-v1/model.py:65-79` | in_proj 低秩钩子 / build_mamba_backbone / head_norm / head 构造（norm_type 钩子先例式注释） |
| `models/gpfm-mamba-v1/model.py:82-101` | forward：temb 仅 future 位（94-95）、`self.blocks(h)` 一把过（97）、head(head_norm)（98） |
| `models/gpfm-mamba-v1/model.py:107-145` | EMA 全键泛型遍历 + swap |
| `models/gpfm-mamba-v1/mamba.py:19-25` | build_norm（rmsnorm/layernorm，其他 ValueError） |
| `models/gpfm-mamba-v1/mamba.py:28-49` | MambaBlock pre-norm 残差（forward 48-49） |
| `models/gpfm-mamba-v1/mamba.py:52-66` | build_mamba_backbone 返回 nn.Sequential（额外参不可直传 ⇒ 块循环） |
| `models/gpfm-mamba-v1/train.py:273-280` | 训练构造点（bottleneck_rank/norm_type getattr 先例） |
| `models/gpfm-mamba-v1/train.py:80-83,353-357` | t 采样 logit-normal 与 v_θ 调用 |
| `models/gpfm-mamba-v1/train.py:137-183` | 内部验证（固定 t 网格 + EMA swap） |
| `models/gpfm-mamba-v1/sample.py:37-60` | load_eval_model（构造点 + hash 严格校验） |
| `models/gpfm-mamba-v1/sample.py:96-137` | EulerSampler 逐步全前向（无缓存 ⇒ 全序列调制自洽） |
| `models/gpfm-mamba-v4a-fmt/config.py:77-118` | d_model=62/bottleneck_rank=16/time_emb_dim=16/norm_type="layernorm"/训练纪律 |
| `models/gpfm-mamba-v4a-fmt/train.py:41-57` | wrapper 三要素先例（RUN_PREFIX/单遮蔽/importlib） |
| `models/gpfm-mamba-v4a-fmt/tests/test_v4afmt.py:48-55,213-246` | hash pin / 5 norm 位断言式 / 参数 pin 130,366 |
| `.trellis/spec/research/reproducibility.md:10` | default-off + getattr 钩子总纲 |
| `.trellis/spec/research/path-isolation.md:25-33` | 薄变体协议（config 副本 + 包装三要素 + 只读共享） |
| `.trellis/research/environment.md:12-13` | torch 2.8.0 / mamba_ssm 2.3.1 + causal_conv1d 1.7.0（fast path 事实） |

### 外部参考

- [Peebles & Xie, Scalable Diffusion Models with Transformers (DiT), ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Peebles_Scalable_Diffusion_Models_with_Transformers_ICCV_2023_paper.pdf)（[arXiv:2212.09748](https://arxiv.org/abs/2212.09748)）——adaLN/adaLN-Zero 条件机制消融（adaLN-Zero 最优）；modulate = x·(1+scale)+shift；FinalLayer 仅 shift/scale 先例
- [DiT 官方代码 models.py](https://github.com/facebookresearch/DiT/blob/main/models.py)——`adaLN_modulation = Sequential(SiLU, Linear(h, 6h))` 逐 block、最后线性层零初始化、`x + gate.unsqueeze(1) * sublayer(modulate(LN(x)))`（本文件 §3 公式的直接出处）
- [Unveiling the Secret of AdaLN-Zero in Diffusion Transformer (OpenReview)](https://openreview.net/forum?id=E4roJSM9RM)——adaLN-Zero 优于纯 adaLN 的机制后续研究（B 案证据侧）
- Dhariwal & Nichol, Diffusion Models Beat GANs (2021)——adaptive group norm 起源；Perez et al., FiLM (2018)——scale/shift 条件通用范式（背景）

## Caveats / Not Found

- 训练窗数/每 epoch step 数未从本任务材料精确取得（风险 3 的 EMA 滞后绝对步数量级只能给机制性判断）；不影响方案成立。
- 仓库内无任何 adaLN/FiLM 先例代码（1.0c 触发判断正确）；全部外部公式以 DiT 官方代码为锚核对，非二手转述。
- PRD 验收②措辞（"恒等初始化下前向输出与 LayerNorm 基线逐位一致"）仅 A 案字面成立——若 Q1 拍 B 案须同步改写该验收项（§3 B 案）。

---

## 审查回填（2026-10-17，trellis-review 双 Mandate 后主会话合成）

1. **[Major 修正] §3/§8 的「epoch-0 损失 = 基线」按 §8.3 原始构建顺序不成立**：`nn.Linear` 构建期 reset_parameters 无条件消耗全局 RNG（零初始化不归还随机数）——mod 内联各 MambaBlock 构建会使同 seed 下共享权重相对基线整体后移。**采纳修复 = fork_rng RNG 中性构造**（全部调制 MLP 包 `torch.random.fork_rng(devices=[])` 构建）：同 seed 共享权重逐位相同 ⇒ §3「初始前向逐位一致」与 run 级 epoch-0 等价声明成立（design §1.1 已固化为契约）。
2. **[Minor 定量化] §7 风险 3 的 Caveat「训练窗数/每 epoch step 数未取得」不成立**：edra-fmt.md §3（930 窗/例）× fold0 34 例 = 31,620 窗；batch 256 + drop_last ⇒ **123 step/epoch**，EMA τ≈1000 step ≈ **8.1 epoch**。阴性预案（design §2.3）：无改善且 best ≲16 ep 时追加非 EMA last.pt 补充评估。
3. **[Minor 补充] 共享钩子影响面含 v4/v4b**（自有 model.py 但骨干经 v1 mamba.py 构建）——由 A5 test_v4b 回归覆盖。
4. **[Suggestion] 等价断言用 `torch.equal`**（值相等语义）：`+0.0` 会把 LN 输出中的 −0.0 映射为 +0.0（值等字节异），tobytes digest 式有假红风险。
