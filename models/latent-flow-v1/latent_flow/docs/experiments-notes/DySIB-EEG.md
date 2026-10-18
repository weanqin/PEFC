# DySIB-EEG

## v2.2 修订（2026-08-20）— paper-faithful · 单 seed 版

> 本节为 DYSIB-01-v2.2 的现行设计条款，**逐条替换 v2.1 及 v1 中与之冲突的条款**；v2.1/v1
> 正文保留为历史设计存档。任务：`08-19-dysib-01-v22`。依据 DySIB 原论文
> （arXiv:2604.24662v2，Martini et al.；全文抽取 `experiments/_dysib_paper.txt`）。
> **v2.1 为样点偏移变体（n_s/n_F 以样点计），与本版（以窗计）语义不可通约**；v2.1 的
> DEBUG-FACE 判决适用范围照旧（见勘误小节）。本阶段只做训练+表征评估，不做任何扰动实验。

### v2.2-1 样本构造（论文 Eq.6；替换 v2.1-3 的样点偏移语义）

- **帧（window）** F = 128 通道 × 128 样点 ≈ 0.5s，窗间时间不重叠；窗索引 n 的样点起点
  t_n = n·128。
- X={F_n..F_{n+nF-1}}，Y={F_{n+ns}..F_{n+ns+nF-1}}；默认 **(nF=2, ns=1)**：过去集
  {F_n,F_{n+1}}，未来集 {F_{n+1},F_{n+2}}，集间重叠 max(0, nF−ns) 个窗。
- 每对编码 **nF+ns=3 个窗**（并集）——共享窗 F_{n+1} **只编码一次**，在两次拼接中以
  **不同位置**复用（X 拼接位 1 / Y 拼接位 0；论文非平凡性机制，App.C2：共享窗经同一
  编码器但在拼接中位置不同，预测仍非平凡）。
- n_s 语义 = 集合平移量（窗单位），与训练采样密度无关；训练数据 = 原始 recording 在线
  滑窗，anchor 在**窗级**枚举。

### v2.2-2 帧编码器 Φ（无 softmax attention；替换 v2.1-1 的 MHSA 联合编码器）

a. 逐通道共享 temporal conv stem（stride-2 ×3，T 128→16，feature 256；C fold 进 batch）；
b. 时间池化（mean over T_lat）→ 每通道向量 h_c∈R^256；
c. 通道混合 = 图消息传递 **K=2 轮**：`x ← x + gelu(MLP(A_eff^T x))`，
   **A_eff = gamma_m·M(d) + gamma·W**（M=mexhat 距离核，`load_geometry("modma")` 128 通道
   坐标，sigma 可学习初始化为中位距离；W 零初始化可学习残差，有向；gamma=0.1、gamma_m=1.0
   对齐 oc_latent_flow 惯例；复用其核与消息传递结构，轻量适配层，不改动原模块）；
d. 通道 mean-pool → LN + Linear(256→**dF=128**) → 帧嵌入 e∈R^{dF}。
- **禁用 softmax/全局 attention**。备选 `--phi-mixer linear_attn`（默认关）：时间池化后
  elu+1 核 linear attention over 128 通道 token，帧头不变。
- 集合统计头 Ψ（论文 Fig.1 阴影块，X/Y 共享）：concat [Φ(F_n),...,Φ(F_{n+nF-1})]∈
  R^{nF·dF} → Linear(nF·dF→2d) → (μ,ℓ)∈R^d，ℓ clamp[−8,4]；z=μ+σ⊙ε；评估一律用 μ。
- **护栏诊断（预注册）**：训练日志每 5 epoch 记录 encoder A_eff 行和 CV
  （std/|mean|）；**<0.02 视为各向同性化预警 → 启用 uniformity_penalty（λ=0.1，
  1/(1+var(rowsum))，唯一预设处置）**，如实入笔记。G4（subject-ID）照常监控指纹化。

### v2.2-3 δ-predictor（论文 Eq.8–10；作用于 z_x 本身，替换 v2.1-4 的多步头）

小 MLP（3 层，hidden 64）：z_x∈R^d → (μδ,ℓδ)∈R^d；
r(z_y|z_x)=N(z_x+μδ(z_x), diag exp ℓδ(z_x))。**μδ 零初始化**（初始预测=恒等）、
**ℓδ bias=−2**。单一输出（无每步头——v2.2 每对只有一步集合平移预测）。

### v2.2-4 InfoNCE critic（论文 Eq.11–12；替换 v2.1-6 的 /d 归一）

- s(z_x,i,z_y,j)=log r(z_y,j|z_x,i)：高斯对数似然**逐维求和、不除 d**（Eq.11 逐字，
  含 ln2π 常数）。ĨNCE = ln(M) − CE（M=候选数；Eq.12 字面分母 1/B 按 InfoNCE 惯例读作
  对候选求和，与论文"ceiling≈log2 B"自述一致），**以 bits 报告**（nats/ln2）。
- 负样本（**EEG 适配偏差，记录**）：**15 same + 15 cross** 平衡，候选集 1+15+15=31；
  same 负样本的样本并集跨度 [t_n, t_n+(nF+ns)·128) 与 anchor 不重叠且**间隔 ≥2 窗**
  （256 样点，守卫单测钉死）；cross 只要求跨被试。
- **天花板 log2(31)≈4.95 bits**（论文 B=1024 为 ~10 bits；差异写入笔记；若 MI 贴顶
  （≥4.8 bits）需加候选数）。

### v2.2-5 损失（论文 Eq.13–14；替换 v2.1-6 的 NCE+KL/β 形式）

$$\mathcal L = \mathrm{KL}(q(z_x|x)\Vert\mathcal N(0,I)) +
\mathrm{KL}(q(z_y|y)\Vert\mathcal N(0,I)) - \beta\,\tilde I_{\mathrm{NCE}}$$

- KL 为**每样本总 KL**（逐维求和，**不归一**；批内取均值）；ĨNCE 以 nats 进损失。
- **β=100 默认**（可选 {30,300} 微调）；ℓ clamp[−8,4]，饱和率监控（**<5% 健康线**）。
- 训练 ε_x⊥ε_y（独立重参数化噪声，v2.1-3 规则维持）。

### v2.2-6 训练协议（frozen）

AdamW lr 1e-4 / wd 1e-4 / grad-clip 1.0 / bf16；epochs≤200 patience 20；
batch=8 被试×16 对（B=128 对/step）；epoch=100 随机步；train-only 归一化复用
`results/fm_gate/_norm_cache`（只读），被试表 seed42 + val carve5（与 08-15/08-16/08-17
同源）。被试内 anchor 两两间隔 ≥ (nF+ns)·128+256 样点（批内 same 负候选恰好 15）。
**每臂单 seed=42**（共 2 次训练：d=32 与 d=128；梯度检查点 enc_chunk=64）。

### v2.2-7 单 seed 仲裁规则（预注册）

若任一臂 G1 的 CI_low ∈ (−0.005,+0.005) 或 CI 跨 0 → 补跑该臂 **seed=43 一次**作仲裁
（两 seed 分歧则**如实并排报告、不合并**）；否则不跑第二 seed。诚实警示段写明
"单 seed，训练方差未量化；G1 的 CI 为评估侧 bootstrap"。

### v2.2-8 模型选择（论文自洽协议的 val-only 版）

MI 误差棒 = **被试层 bootstrap**（非 seed 方差）；MI–d 饱和判读为**描述性**（仅两点）；
可选 nF∈{1,2} 小扫描判时间阶数（EEG 单窗已含时间结构，nF=1 可能已足——由饱和说话）；
**拒绝论文的 test-MI 选择**（val-only，v1 §44 协议维持）。

### v2.2-9 预检（训练前、CPU、原始窗；不阻断）

窗级 horizon h∈{1,2,4}（窗单位）的 dist(F_t,F_{t+h}) **同被试 vs 跨被试**
AUC/Wasserstein（慢态捷径残留量）；写 `results/dysib/v22_precheck/` 并入笔记。

### v2.2-10 DYSIB-01-v2.2 gate 表（预注册，frozen before training — 逐字）

```
G0 泄漏：shuffle R² |值|<0.01 或 CI 含 0；
G1 主判（cross-subject）：Gain_b = R²(x_{新窗,b}←z_x) − R²(x_{新窗,b}←x_obs,b)
  （shuffle 地板 + 被试层 bootstrap 1000）：CI_low>0，实用目标 ≥+0.02；
  目标时点契约：z_x 概括 X，预测 X 之后第一个新窗 F_{n+nF} 的内容
  （X 锚在归档 anchor 窗，x_obs 基线 = 与 z_x 同信息的 nF 窗逐通道原始内容；
  目标窗越界行剔除并计数）；
G2 压缩预测无损：Δ_b = R²_b(z_x) − R²_b(PCA-256 raw 参照探针)：CI_low>−0.02；
  α 网格 logspace(-2,5,14)；16384 维 raw 探针降为描述项；
G3（描述项，不进通过线）：论文 Eq.15 随机 rollout z^(t+1)~r(·|z^(t)) 的
  latent RMSE–步数曲线（每步=0.5s；均值版并列）；
G4 反捷径：subject-ID probe（冻结 μ，被试层 split）≤ VAE 基线+5pp，
  G1 通过时不得显著上升；
G5 数值：clamp 饱和 <5%、后验健康、d_eff↓ 时 MI/R² 必须升。
通过线 G1∧G2∧G0∧G4∧G5。
```

**预注册分支（逐字）**：G1 过线 → 进 DYSIB-02（Oracle decoder）与扰动机件排期；
G1 离零（>0）但不过线 → 与 Exp 5（band×horizon）联读定 v2.3；G1 仍 ≈0（CI_high<0 或
贴零）→ 任务级/数据级不足坐实，DySIB 线终止归档，转诊断叙事决策。
**并列读数（不进 gate）**：MI–d 两点曲线 = EEG 预测状态本征维数的初步估计。

### v2.2-11 "不变表"更新与 v2.1 勘误

| 条目 | 状态 |
|---|---|
| 窗 128 样点、C=128 通道 | 不变 |
| 评估用 μ、训练独立 ε | 不变（v2.1-3 规则维持） |
| val-only 选择（拒绝 test-MI）、subject-held-out 评估 | 不变 |
| 归档 fold0_pairs.npz 同窗 8881 评估 + 08-16 对齐硬校验 | 不变（v2.2 目标窗越界行剔除并计数） |
| near 阈值 ±0.005 纪律、点估计-vs-margin 手工警示 | 不变 |
| ~~n_s/n_F 样点偏移语义（v2.1-3）~~ | **替换**为窗集合语义（v2.2-1） |
| ~~MHSA 联合编码器（v2.1-1）~~ | **替换**为帧编码器 Φ + 集合统计头（v2.2-2） |
| ~~每步独立 δ 头（v2.1-4）~~ | **替换**为单输出 δ-predictor（v2.2-3） |
| ~~NCE /d、KL/d 全窗平均、NCE+KL/β（v2.1-6）~~ | **替换**为 Eq.13–14 形式（v2.2-4/-5） |
| ~~β=10（v2.1-6）~~ | **替换**为 β=100（v2.2-5） |

**v2.1 勘误（适用范围澄清）**：v2.1 的 n_s=1/n_F=2 实为**样点偏移**（未来窗 = 观测窗
平移 1/2 样点、共享 127/128 样本），非论文 Eq.6 的窗平移构造；其"近微分目标过于平凡"
的读法与 DEBUG-FACE 判决在**该样点偏移任务**内有效，不可外推到窗集合任务（v2.2）。
G0–G5 gate 数值定义在 v2.2 目标时点契约（F_{n+nF}）下沿用，但 v2.1 与 v2.2 的
R²(x_fc←z) 数值不可直接比较（目标窗相差 128 样点）。

### v2.2 与 v2.1 条款的替换映射

| v2.1 条款 | v2.2 处置 |
|---|---|
| v2.1-1（MHSA 联合编码器 + h=256 容量旋钮） | 由 v2.2-2 取代（帧编码器 Φ + Ψ 统计头；h 旋钮概念删除） |
| v2.1-3（样点偏移任务 + §15 反转） | 由 v2.2-1 取代（窗集合；App.C2 非平凡性机制） |
| v2.1-4（加强 δ-predictor 多步头） | 由 v2.2-3 取代 |
| v2.1-5（same 负样本间隔，样点单位） | 由 v2.2-4 取代（窗单位、≥2 窗间隔） |
| v2.1-6（NCE /d、KL/d、NCE+KL/β、β=10） | 由 v2.2-4/-5 取代 |
| v2.1-2（d sweep {16,32,64,128} 先两臂） | 维持两臂 {32,128}，单 seed=42 + 仲裁规则（v2.2-6/-7） |
| v2.1-7（DE 扰动位点预置条款） | 维持预置、过门后执行（本阶段仍禁止实现/运行） |
| v2.1-8（v2.1 gate 表） | 由 v2.2-10 取代（G3 降为描述项、G2 参照改 PCA-256、G5 加 MI 条款、目标时点契约入表） |
| v2.1-9（不变表） | 由 v2.2-11 取代 |

---

## v2.1 修订（2026-08-18）

> 本节为 DYSIB-01-v2.1 的现行设计条款，**逐条替换下方 v1 中与之冲突的条款**；v1 正文保留为
> 历史设计存档。任务：`08-17-dysib-01-v21`。本阶段**只做训练与表征可行性评估（G0–G5），
> 不做任何扰动实验**；DE 扰动全套见条款 7（仅设计文档，禁止实现/执行）。

### v2.1-1 两级瓶颈（替换 v1 §3–§6、§53 的 per-channel latent 结构）

$$
X_{(128\times128)} \rightarrow \text{backbone} \rightarrow h\in\mathbb R^{256}
\rightarrow \text{统计头 } \mathrm{Linear}(256\to 2d) \rightarrow
(\mu,\ \log\sigma^2)\in\mathbb R^{d}
$$

- $h\in\mathbb R^{256}$ 是**确定性中间特征**：容量旋钮，**不受 KL 约束，禁止用于任何评估
  通路**（一切评估只取 $\mu$）。
- $z=\mu+\sigma\odot\epsilon$ 是 $d$ 个**学习到的预测充分统计量**（窗口级单向量，非
  per-channel 张量）；对角高斯，$\ell=\log\sigma^2$ clamp $[-8,4]$。
- v1 的 $T_{\mathrm{lat}}=16$ / $D_{\mathrm{lat}}$ / channel-aligned latent 概念整体作废。

### v2.1-2 d sweep 与饱和判读（替换 v1 §6）

$$
d \in \{16,\,32,\,64,\,128\},\qquad \text{先跑 } \{32,\,128\} \text{ 两臂。}
$$

饱和判读正式化：**R²–d 曲线的饱和点 = 窗口预测状态的本征统计量维数**（若 $d$ 增大而
R² 不再升，则当前 $d$ 已覆盖该维数）。

### v2.1-3 任务定义（替换 v1 §2 的 obs/fc 邻接关系）

- 窗 128 时间点、$n_s=1$、$n_F=2$：targets 为 $z_{t+1}, z_{t+2}$（窗口起点平移 1/2 样本），
  **每步独立头**，NCE 对 $k$ 平均；负样本 = 负样本窗的**第 $k$ 步未来 latent**。
- 训练 $\epsilon_{\mathrm{obs}}\perp\epsilon_{\mathrm{fc}}$（独立重参数化噪声），评估一律
  用 $\mu$。
- **§15 告诫反转**：v1 §15 警告"residual 不可解释为微分动力学"的前提（未来窗远离观测窗）
  在 $n_s=1$ 下不再成立——**近微分 residual = 微分结构**，$\Delta\mu_\delta$ 重新获得
  近微分（数值上近似 $z$ 对时间的导数 × 步长）的解释地位。

### v2.1-4 δ-predictor 加强（替换 v1 §9–§13 的架构草案）

- residual MLP $d\to4d\to d$ × 4 block，作用于 $z\in\mathbb R^d$，后接**每步独立头**
  （step $k=1,2$ 各一套）：$\Delta\mu$ head **零初始化**（初始 $\mu_{\mathrm{pred}}=z$）、
  $\ell_\delta$ head **bias $=-2$**。
- 监测 NCE **分负样本类型的准确率 / margin**（same vs cross），作为 "强 predictor 弱表征"
  守卫：若 cross 类型准确率远高于 same 类型，说明 critic 在做被试判别而非动力学判别。

### v2.1-5 负样本（细化 v1 §21）

- $15$ same $+$ $15$ cross 平衡（v1 §23 不变）。
- **same 负样本窗与 anchor 跨度 $[t,\ t+127+n_F]$ 不重叠且间隔 ≥256 样本**（该守卫以单测
  钉死；对 $k\in\{1,2\}$ 的全部 future 窗成立）。

### v2.1-6 损失归一与 β（替换 v1 §17 的 $/CTD$、§25 的分窗 KL）

- NCE 分数归一 $/d$（跨 $d$ 可比）；KL 对**批内全部窗**（obs + 各步 future）平均、$/d$。
- $\beta=10$ 起步（$\mathcal L = \mathcal L_{\mathrm{NCE}} + \mathcal L_{\mathrm{KL}}/\beta$）；
  失败按 v1 §27 规则调；**val-only 选择，拒绝 test-MI 选择**（v1 §44 末段告诫维持）。

### v2.1-7 DE 扰动位点迁移条款【仅设计文档，本阶段禁止实现/执行】

> **预置设计，DYSIB-01 过门后执行。** 本阶段（DYSIB-01-v2.1）不实现、不运行任何扰动代码。

- **主位点 = 输入空间虚拟刺激**：$\delta x_a = r\cdot L_a\xi$，其中 $L_a$ 为训练集通道 $a$
  原始窗协方差 Cholesky、$r_{\max}$ 按 Mahalanobis $q=0.99$ 校准（$\xi\sim\mathcal N(0,I)$）。
- 响应链：$x\pm\delta x_a \rightarrow E \rightarrow$ 动力学（rollout）$\rightarrow$ decoder
  $\rightarrow \hat x_b$；$R_{a,b}/\mathrm{MAD}_b$ 语义不变；Null = 被试层 bootstrap。
- encoder 传输审计与 decoder-Jacobian 位点 B 为**过门后第一批实验**。
- v1 §50/§52 的 latent 空间统计量重冻结条款相应迁移到新表示（$d$ 维单向量上的协方差 /
  白化 / OOD），在 02（Oracle decoder）之后重新建立。

### v2.1-8 DYSIB-01-v2.1 gate 表（预注册，frozen before training — 逐字）

```
G0 泄漏：shuffle R² |值|<0.01 或 CI 含 0；
G1 主判（cross-subject）：Gain_b = R²(x_fc,b←z) − R²(x_fc,b←x_obs,b)
  （shuffle 地板 + 被试层 bootstrap 1000）：CI_low>0，实用目标 ≥+0.02；
G2 压缩预测无损：Δ_b = R²_b(z) − R²_b(x_full 16384维 ridge 探针)：CI_low>−0.02；
G3 长程（latent 域，无 decoder）：δ-rollout horizon {1,2,16,64,128} 的
  z-空间 R²/Pearson 曲线，gate 在 128：CI_low>0，实用目标 ≥0.2，标注
  "latent 域替代，decoded 版待 02"；
G4 反捷径：subject-ID probe（冻结 μ，被试层 split）≤ VAE 基线+5pp，
  G1 通过时不得显著上升；
G5 数值：clamp 饱和 <5%、后验健康、d_eff↓ 时 R² 必须升。
通过线 G1∧G2∧G0∧G4∧G5。
```

**预注册分支（逐字）**：G1 败而 G2 过 → β/负样本/d 调试面；双败 → 近微分目标与 DE horizon
脱节，Exp 5（band×horizon）升为决定性实验。

### v2.1-9 "不变表"更新（替换 v1 §53 对应行）

| 条目 | 状态 |
|---|---|
| $L=128$（窗长） | 不变（公平对照） |
| $C=128$ 通道 | 不变 |
| 评估用 $\mu$ | 不变 |
| 训练独立 $\epsilon$ | 不变 |
| val-only 选择（拒绝 test-MI） | 不变 |
| subject-held-out 评估 | 不变 |
| ~~$T_{\mathrm{lat}}=16$~~ | **删除**（latent 为 $d$ 维单向量） |
| ~~$D_{\mathrm{lat}}=8$ / per-channel $d=128$~~ | **删除**（由 v2.1-2 的 $d$ sweep 取代） |
| ~~channel-aligned latent~~ | **删除**（联合编码器整体混通道；DE 语义迁至输入空间位点，见 v2.1-7） |
| ~~无 skip decoder~~ | **删除**（decoder 本阶段不存在，02 再立） |

### v2.1 与 v1 条款的替换映射

| v1 条款 | v2.1 处置 |
|---|---|
| §3–§6（per-channel encoder / latent 形状 / $D_{\mathrm{lat}}$） | 由 v2.1-1、v2.1-2 取代 |
| §2（$Y$ 窗邻接关系） | 由 v2.1-3 取代（$n_s=1$、$n_F=2$） |
| §9–§13（δ-predictor 架构） | 由 v2.1-4 取代 |
| §15（近微分告诫） | 由 v2.1-3 反转（近微分 residual=微分结构） |
| §17（NCE $/CTD$ 归一）、§25（KL） | 由 v2.1-6 取代（$/d$、全窗平均） |
| §21（same 负样本间隔） | 由 v2.1-5 细化并单测钉死 |
| §32/§33/§35/§45（01 阶段阈值/gate） | 由 v2.1-8 取代 |
| §44（第一轮固定配置表） | 以 v2.1-1…-6、-9 为准 |
| §50/§52（latent 统计量重冻结） | 迁移至输入空间位点（v2.1-7），02 后重建 |
| §53（不变表） | 由 v2.1-9 取代 |
| §26/§27（β 定义与 sweep）、§28（无 recon）、§30（随机/确定切换）、§42（subject-ID probe）、§43（val-only 选择） | 维持有效 |

---

## 一、DySIB-EEG 总体架构

整体数据流定义为：

$$
X_{s,t} \overset{E_\theta}{\longrightarrow} q_\theta(Z_{\mathrm{obs}}\mid X_{s,t})
$$

$$
Y_{s,t} \overset{E_\theta}{\longrightarrow} q_\theta(Z_{\mathrm{fc}}\mid Y_{s,t})
$$

其中 **同一个 $E_\theta$** 同时编码过去和未来。

训练时：

$$
Z_{\mathrm{obs}} = \mu_{\mathrm{obs}} + \sigma_{\mathrm{obs}}\odot \epsilon_{\mathrm{obs}},
$$

$$
Z_{\mathrm{fc}} = \mu_{\mathrm{fc}} + \sigma_{\mathrm{fc}}\odot \epsilon_{\mathrm{fc}}.
$$

然后：

$$
Z_{\mathrm{obs}} \overset{g_\delta}{\longrightarrow} \left( \Delta\mu_\delta, \log\sigma^2_\delta \right)
$$

$$
\mu_{\mathrm{pred}} = Z_{\mathrm{obs}} + \Delta\mu_\delta,
$$

最终定义：

$$
r_\delta\left( Z_{\mathrm{fc}}\mid Z_{\mathrm{obs}} \right) = \mathcal N\left( \mu_{\mathrm{pred}}, \operatorname{diag}(\sigma_\delta^2) \right).
$$

这个 conditional Gaussian 就是 InfoNCE critic 的基础。原 DySIB 也是用

$$
r(z_y\mid z_x) = \mathcal N\left( z_x+\mu_\delta(z_x), \operatorname{diag}(\exp \ell_\delta(z_x)) \right)
$$

作为 predictive distribution，再把它的 log-likelihood 作为 InfoNCE score。

---

## 二、输入与未来窗口

### 2.1 第一轮严格沿用当前任务

对 subject $s$ 的 EEG：

$$
x_s\in\mathbb R^{C\times T}, \qquad C=128.
$$

构造：

$$
X_{s,t} = x_s[:,\,t:t+127] \in\mathbb R^{128\times128},
$$

$$
Y_{s,t} = x_s[:,\,t+128:t+255] \in\mathbb R^{128\times128}.
$$

所以：

$$
\boxed{ L_{\mathrm{obs}}=128,\qquad L_{\mathrm{fc}}=128 }
$$

且二者第一轮 **不重叠**。

必须保证一对 $(X,Y)$：

- 来自同一 subject；
- 来自同一连续 recording/trial；
- 不跨 trial/session 边界；
- subject-level train/val/test split 先完成，再构造窗口；
- normalization 参数只由 training subjects 估计。

这样可以直接和当前 VAE latent 做公平比较。

---

## 三、Shared EEG Encoder

这里有一个很重要的结构原则：

$$
\boxed{ E_\theta \text{ 本身不做跨 EEG channel mixing} }
$$

而只学习：

$$
x^{(c)} \rightarrow z^{(c)}.
$$

原因是后续 DE 中你需要保持：

$$
\text{EEG channel }c \leftrightarrow Z^{(c)}
$$

这层语义。

如果 Encoder 一开始就把 128 个通道完全混合，那么后面"扰动 source channel $a$"在 latent 中会失去明确含义。

跨通道动力学应该交给 δ-predictor 和后续 FM 去学习。

---

### 3.1 输入 reshape

输入：

$$
X\in\mathbb R^{B\times128\times128}.
$$

将通道 fold 到 batch：

$$
X' \in \mathbb R^{(B\cdot128)\times1\times128}.
$$

所有 EEG channel 使用同一个 temporal encoder。

这也意味着 Encoder 参数量不会随 electrode 数量成倍增长。

---

## 四、推荐 Encoder backbone

**最严格的第一轮对照实验，其实最好直接复用当前 VAE Encoder backbone，只改变训练目标。**

这是优先级最高的建议。

因为这样：

$$
\text{VAE vs DySIB}
$$

唯一核心变量就是：

$$
\text{reconstruction objective} \quad\rightarrow\quad \text{predictive-information objective}.
$$

如果当前 backbone 不方便直接复用，那么推荐下面这个 reference implementation。

---

### 4.1 Stem

输入：

$$
(B\cdot C,\,1,\,128)
$$

执行：

```text
Conv1d(
    in=1,
    out=32,
    kernel=7,
    stride=1,
    padding=3
)
GroupNorm(8,32)
GELU
```

输出：

$$
(B\cdot C,\,32,\,128).
$$

---

### 4.2 Residual Down Block 1

主分支：

```text
Conv1d 32→64, k=5, stride=2, padding=2
GroupNorm(8,64)
GELU

Conv1d 64→64, k=3, stride=1, padding=1
GroupNorm(8,64)
```

shortcut：

```text
Conv1d 32→64, k=1, stride=2
```

然后：

```text
main + shortcut
GELU
```

输出：

$$
(B\cdot C,\,64,\,64).
$$

---

### 4.3 Residual Down Block 2

```text
64 → 128
kernel = 5
stride = 2
```

输出：

$$
(B\cdot C,\,128,\,32).
$$

---

### 4.4 Residual Down Block 3

```text
128 → 256
kernel = 5
stride = 2
```

输出：

$$
(B\cdot C,\,256,\,16).
$$

因此：

$$
\boxed{ T_{\mathrm{lat}}=16 }
$$

是结构自然得到的，不需要 AdaptivePool。

---

### 4.5 Bottleneck temporal block

增加两个不降采样 residual block：

#### Block A

```text
Conv1d 256→256, k=3, dilation=1
GroupNorm
GELU
Conv1d 256→256, k=1
GroupNorm
+ residual
```

#### Block B

```text
Conv1d 256→256, k=3, dilation=2, padding=2
GroupNorm
GELU
Conv1d 256→256, k=1
GroupNorm
+ residual
```

这样每个 latent token 能覆盖较大的原始 EEG 时间范围。

---

## 五、Gaussian latent head

最后：

```text
Conv1d(
    256 → 2*D_lat,
    kernel=1
)
```

得到：

$$
(B\cdot C,\,2D_{\mathrm{lat}},\,16).
$$

reshape 为：

$$
(B,\,C,\,16,\,2D_{\mathrm{lat}}).
$$

切分：

$$
\mu \in \mathbb R^{B\times C\times16\times D_{\mathrm{lat}}},
$$

$$
\ell := \log\sigma^2 \in \mathbb R^{B\times C\times16\times D_{\mathrm{lat}}}.
$$

DySIB 原论文同样采用 Gaussian variational posterior，并由网络输出 mean 和 log-variance。

定义：

$$
q_\theta(Z\mid X) = \mathcal N\left( \mu_\theta(X), \operatorname{diag}(\exp\ell_\theta(X)) \right).
$$

---

## 六、D_lat 设置

第一轮：

$$
\boxed{D_{\mathrm{lat}}=8}
$$

因此：

$$
Z \in \mathbb R^{B\times128\times16\times8}.
$$

每个 EEG channel 对应：

$$
d = T_{\mathrm{lat}}\cdot D_{\mathrm{lat}} = 16\times8 = 128.
$$

这与当前 D8 表示保持一致。

随后 sweep：

$$
D_{\mathrm{lat}} \in \{8,16,32\}.
$$

对应：

$$
d \in \{128,256,512\}.
$$

这样可以判断之前 D8 的失败究竟是：

$$
\text{representation objective}
$$

还是：

$$
\text{latent capacity}
$$

造成的。

---

## 七、log-variance 数值处理

推荐：

$$
\ell = \operatorname{clamp}(\ell,\,-8,\,4).
$$

然后：

$$
\sigma = \exp(0.5\,\ell).
$$

训练期间必须监控 clamp saturation。

如果大量 latent：

$$
\ell=-8
$$

或者：

$$
\ell=4
$$

说明 posterior variance 发生数值退化。

建议 gate：

$$
<5\%
$$

latent element 长期触及 clamp boundary。

---

## 八、一个极其重要的区别：DySIB 训练不要使用共享 ε

你当前 Stage-2 FM 已经确定使用：

$$
z_{\mathrm{obs}} = \mu_{\mathrm{obs}} + \sigma_{\mathrm{obs}}\epsilon,
$$

$$
z_{\mathrm{fc}} = \mu_{\mathrm{fc}} + \sigma_{\mathrm{fc}}\epsilon,
$$

即 shared reparameterization noise。

但是：

### DySIB training 这里必须改成独立噪声

$$
\boxed{ \epsilon_{\mathrm{obs}} \perp \epsilon_{\mathrm{fc}} }
$$

即：

$$
Z_{\mathrm{obs}} = \mu_{\mathrm{obs}} + \sigma_{\mathrm{obs}} \odot \epsilon_{\mathrm{obs}},
$$

$$
Z_{\mathrm{fc}} = \mu_{\mathrm{fc}} + \sigma_{\mathrm{fc}} \odot \epsilon_{\mathrm{fc}}.
$$

原因非常重要：

如果使用同一个 $\epsilon$，那么：

$$
I(Z_{\mathrm{obs}};\,Z_{\mathrm{fc}})
$$

里会人为包含：

$$
I(\epsilon;\,\epsilon)
$$

带来的共享随机信息。

InfoNCE 可能因此获得虚假的 past-future MI。

也就是说：

> **FM 中 shared ε 是为了降低 transition target 方差；DySIB 中 shared ε 反而会构成 MI shortcut。**

所以这两个阶段必须区别处理。

---

## 九、δ-predictor

Encoder 输出 sampled：

$$
Z_{\mathrm{obs}} \in \mathbb R^{B\times C\times16\times D}.
$$

δ-predictor：

$$
g_\delta:\; Z_{\mathrm{obs}} \rightarrow \left( \Delta\mu_\delta,\, \ell_\delta \right).
$$

输出形状与 latent 完全相同。

---

## 十、δ-predictor 推荐架构

它应该能够跨通道建模，否则 DySIB objective 很难主动奖励跨通道动力学信息。

但它不应该像最终 FM 那么复杂。

否则你可能得到：

> 强 predictor + 弱 representation

的训练集过拟合。

建议：

$$
H=128.
$$

---

### 10.1 Input projection

逐 latent token：

```text
Linear(D_lat → 128)
```

得到：

$$
H \in \mathbb R^{B\times128\times16\times128}.
$$

---

### 10.2 Temporal residual block ×2

把：

$$
B\times C
$$

fold：

$$
(B\cdot C,\,128,\,16).
$$

第一层：

```text
Conv1d 128→128
kernel=3
dilation=1
padding=1
GroupNorm(8,128)
GELU

Conv1d 128→128
kernel=1
GroupNorm

+ residual
```

第二层：

```text
kernel=3
dilation=2
padding=2
```

其它相同。

---

## 十一、Channel mixer

之后 reshape：

$$
(B\cdot16,\,C,\,H).
$$

第一版建议使用一个很小的：

```text
MultiHeadSelfAttention(
    embed_dim=128,
    num_heads=4
)
LayerNorm
Residual

FFN:
128 → 256 → 128
GELU
Residual
LayerNorm
```

即：

$$
\boxed{ 1\text{ 个 channel-attention block} }
$$

就足够。

不建议第一轮直接放：

- 6 层 Transformer；
- 完整 MexHat FM mixer；
- learnable dense $(128\times128)$ dynamics network。

因为那会让 DySIB probe 本身成为一个大型动力学模型。

我们的目标不是让 δ-predictor 获得最强预测能力，而是：

> 迫使 Encoder 把 future-relevant information 放进 latent。

---

## 十二、为什么 Encoder 不混 channel，而 δ-predictor 混 channel

最终结构实际上是：

$$
x_{\mathrm{obs}}^{(c)} \rightarrow z_{\mathrm{obs}}^{(c)}
$$

但：

$$
\Delta z_{\mathrm{pred}}^{(c)} = g_\delta\left( z_{\mathrm{obs}}^{(1)},\,\dots,\,z_{\mathrm{obs}}^{(128)} \right).
$$

所以：

- channel alignment 保留；
- channel interaction 仍然可以参与 predictive objective；
- 后续可以测：

$$
R^2_{\mathrm{self}}
$$

与

$$
R^2_{\mathrm{full}}
$$

的差异。

这是特别适合你当前 DE 目标的结构。

---

## 十三、δ 输出 head

Channel mixer 后再经过一个 temporal residual block，然后：

### Δμ head

```text
Linear(128 → D_lat)
```

得到：

$$
\Delta\mu_\delta \in \mathbb R^{B\times C\times16\times D}.
$$

### variance head

```text
Linear(128 → D_lat)
```

得到：

$$
\ell_\delta = \log\sigma_\delta^2.
$$

初始化建议：

### Δμ head

$$
W=0,\qquad b=0.
$$

因此初始时：

$$
\mu_{\mathrm{pred}} = Z_{\mathrm{obs}}.
$$

### logvar

bias：

$$
b_{\ell_\delta}\approx-2.
$$

---

## 十四、Residual prediction

最终：

$$
\boxed{ \mu_{\mathrm{pred}} = Z_{\mathrm{obs}} + \Delta\mu_\delta(Z_{\mathrm{obs}}) }
$$

以及：

$$
r_\delta\left( Z_{\mathrm{fc}}\mid Z_{\mathrm{obs}} \right) = \mathcal N\left( \mu_{\mathrm{pred}},\, \operatorname{diag}(\exp\ell_\delta) \right).
$$

这是 DySIB 原论文最关键的 inductive bias 之一。

---

## 十五、这里与原 DySIB 有一个必须说明的差异

原论文中 δ-predictor 强调的是：

$$
z_{t+\Delta t} \approx z_t+\delta z
$$

其中过去和未来非常接近，因此是"近微分"的 dynamics。论文甚至使用 $n_s=1$ 的短 stride 来强化这个性质。

而你这里是：

$$
128\text{ samples obs} \rightarrow 128\text{ samples future}
$$

即未来窗口从 $t+128$ 开始。

所以：

$$
\Delta Z
$$

**未必很小。**

因此在 DySIB-EEG v1 中：

> 保留 residual parameterization，但不要加入任何
> $|\Delta Z|$ 很小的额外正则。

也就是说：

$$
Z_{\mathrm{pred}} = Z_{\mathrm{obs}} + \Delta Z
$$

只是优化上的 residual parameterization，

**不能把它解释成严格 differential dynamics。**

这一点在论文撰写时尤其要区分。

---

## 十六、InfoNCE critic

DySIB 原论文使用：

$$
s(z_{x,i},\,z_{y,j}) = \log r(z_{y,j}\mid z_{x,i})
$$

作为 critic。

我们保持这一点。

定义：

$$
M_i = Z_{\mathrm{obs},i} + \Delta\mu_{\delta,i}.
$$

对于未来 sample $j$：

$$
s_{ij} = -\frac12 \sum_k \left[ \ell_{\delta,i,k} + \frac{(Z_{\mathrm{fc},j,k}-M_{i,k})^2}{\exp(\ell_{\delta,i,k})} \right].
$$

常数项可以省略。

---

## 十七、必须做 latent-dim normalization

因为：

$$
D=8,\,16,\,32
$$

时总 latent scalar 数量不同。

如果直接 sum：

$$
|s_{ij}| \propto C\,T_{\mathrm{lat}}\,D.
$$

会导致 D sweep 无法公平比较。

因此推荐：

$$
\boxed{
\bar s_{ij} = -\frac{1}{2\,C\,T_{\mathrm{lat}}\,D} \sum_{c,t,d} \left[ \ell_{i,c,t,d} + e^{-\ell_{i,c,t,d}}\,(Z_{j,c,t,d}-M_{i,c,t,d})^2 \right]
}
$$

然后：

$$
\tilde s_{ij} = \frac{\bar s_{ij}}{\tau_{\mathrm{NCE}}}.
$$

第一轮：

$$
\boxed{ \tau_{\mathrm{NCE}}=1 }
$$

先不要再引入 temperature sweep。

如果 logits 后续过平，可以再试：

$$
0.5,\;1,\;2.
$$

---

## 十八、正负样本构造：这是 EEG 版本最重要的改动之一

不能简单：

```text
一个 subject 一个样本
不同 subject 全作为 negatives
```

否则模型非常可能学习：

$$
\text{subject identity}
$$

而不是：

$$
\text{brain state dynamics}.
$$

---

## 十九、推荐 batch sampler

理想第一版：

$$
S=8
$$

个 subjects，每个 subject：

$$
K=16
$$

个 window pairs。

因此：

$$
B=S\times K=128.
$$

即：

```text
Subject 1:
  pair1 pair2 ... pair16

Subject 2:
  pair1 pair2 ... pair16

...

Subject 8:
  pair1 pair2 ... pair16
```

---

## 二十、每个 anchor 的 positive

对于：

$$
X_{s,t_i}
$$

唯一 positive：

$$
Y_{s,t_i}.
$$

即：

$$
(Z_{\mathrm{obs},i},\,Z_{\mathrm{fc},i}).
$$

---

## 二十一、same-subject negatives

从：

$$
s_j=s_i,\qquad j\neq i
$$

中取：

$$
M_{\mathrm{same}}=15.
$$

最好要求其原始窗口与 anchor pair：

$$
[t,\,t+255]
$$

不发生 overlap。

如果记录长度允许，建议 temporal separation：

$$
\ge256
$$

samples。

这样减少"几乎同一个脑状态被错误作为 negative"的概率。

---

## 二十二、cross-subject negatives

从：

$$
s_j\neq s_i
$$

中随机采样：

$$
M_{\mathrm{cross}}=15.
$$

于是每个 anchor candidate set 为：

$$
1\text{ positive} + 15\text{ same-subject negatives} + 15\text{ cross-subject negatives}.
$$

---

## 二十三、为什么不要使用 batch 中全部 cross-subject negatives

如果：

$$
15\text{ same-subject}
$$

对比：

$$
112\text{ cross-subject}
$$

那么梯度会被容易的 cross-subject discrimination 主导。

模型会发现：

> "只要学 subject identity 就能解决大量 negatives。"

因此建议：

$$
\boxed{ M_{\mathrm{same}} = M_{\mathrm{cross}} }
$$

进行 group-balanced negative sampling。

这是 EEG DySIB 相对论文的一个重要适配。

---

## 二十四、InfoNCE loss

对 anchor $i$，定义 candidate 集：

$$
\mathcal C_i = \{i\} \cup \mathcal N_i^{\mathrm{same}} \cup \mathcal N_i^{\mathrm{cross}}.
$$

则：

$$
\mathcal L_{\mathrm{NCE},i} = -\log \frac{ \exp(\tilde s_{ii}) }{ \sum_{j\in\mathcal C_i} \exp(\tilde s_{ij}) }.
$$

batch：

$$
\boxed{ \mathcal L_{\mathrm{NCE}} = \frac1B \sum_i \mathcal L_{\mathrm{NCE},i} }
$$

---

## 二十五、完整 KL

DySIB 原论文用 posterior 与 standard normal prior 的 KL 作为 compression term。

对于 obs：

$$
\mathcal L_{\mathrm{KL}}^{\mathrm{obs}} = \frac1{BCTD} \sum \frac12 \left[ \mu_{\mathrm{obs}}^2 + e^{\ell_{\mathrm{obs}}} - 1 - \ell_{\mathrm{obs}} \right].
$$

future：

$$
\mathcal L_{\mathrm{KL}}^{\mathrm{fc}} = \frac1{BCTD} \sum \frac12 \left[ \mu_{\mathrm{fc}}^2 + e^{\ell_{\mathrm{fc}}} - 1 - \ell_{\mathrm{fc}} \right].
$$

定义：

$$
\boxed{ \mathcal L_{\mathrm{KL}} = \frac12 \left( \mathcal L_{\mathrm{KL}}^{\mathrm{obs}} + \mathcal L_{\mathrm{KL}}^{\mathrm{fc}} \right) }
$$

这样 D=8/16/32 时 KL 尺度基本可比。

---

## 二十六、最终训练目标

实现版本推荐写成：

$$
\boxed{ \mathcal L_{\mathrm{DySIB-EEG}} = \lambda_{\mathrm{KL}}\,\mathcal L_{\mathrm{KL}} + \lambda_{\mathrm{dyn}}\,\mathcal L_{\mathrm{NCE}} }
$$

第一轮：

$$
\lambda_{\mathrm{dyn}}=1.
$$

定义概念上的：

$$
\beta = \frac{\lambda_{\mathrm{dyn}}}{\lambda_{\mathrm{KL}}}.
$$

所以：

$$
\lambda_{\mathrm{KL}} = \frac1\beta.
$$

原 DySIB 把 predictive term 权重设得很大，论文使用 $\beta=100$，让 predictive information 主导，而 KL 主要作为弱正则。

但这一数值不能直接搬到 EEG。

---

## 二十七、β sweep

建议：

$$
\boxed{ \beta \in \{1,\,3,\,10,\,30,\,100\} }
$$

等价于：

| $\beta$ | $\lambda_{\mathrm{dyn}}$ | $\lambda_{\mathrm{KL}}$ |
|--:|--:|--:|
| 1 | 1 | 1 |
| 3 | 1 | 0.333 |
| 10 | 1 | 0.1 |
| 30 | 1 | 0.033 |
| 100 | 1 | 0.01 |

第一轮推荐从：

$$
\boxed{\beta=10}
$$

开始 sanity check。

如果：

- posterior collapse；
- predictive R² 很低；

则向：

$$
30,\;100
$$

增加 predictive dominance。

如果：

- latent 极不稳定；
- subject identity 增强；
- Oracle 几乎完全丢失；

则向：

$$
3,\;1
$$

增加 compression regularization。

---

## 二十八、第一版不要加入 reconstruction loss

核心主实验应是：

$$
\boxed{ L = L_{\mathrm{KL}} + L_{\mathrm{predictive}} }
$$

不训练 decoder。

因为 DySIB 的核心思想就是：

> 不要求 representation 保留所有可重建信息，只要求保留 future-relevant information。

原论文明确整个 DySIB objective 都位于 latent space，不执行 observation reconstruction。

---

## 二十九、如果 Oracle 失败，再进入 Hybrid DySIB

如果出现：

$$
R^2_{\mathrm{direct}}\uparrow
$$

但：

$$
\mathrm{Oracle}_{\mathrm{EEG}}\downarrow\downarrow,
$$

第二轮才考虑：

$$
\mathcal L = \lambda_{\mathrm{KL}}\,L_{\mathrm{KL}} + L_{\mathrm{NCE}} + \lambda_{\mathrm{rec}}\,L_{\mathrm{rec}}.
$$

例如：

$$
\lambda_{\mathrm{rec}} \in \{0.01,\,0.03,\,0.1\}.
$$

这里 reconstruction 是：

> guardrail

而不是：

> representation 主目标。

第一轮不要混进去，否则无法回答 DySIB 是否真正改变了表示。

---

## 三十、训练与 inference 的 stochastic / deterministic 切换

### Training

必须：

$$
Z_{\mathrm{obs}} = \mu_{\mathrm{obs}} + \sigma_{\mathrm{obs}}\,\epsilon_{\mathrm{obs}},
$$

$$
Z_{\mathrm{fc}} = \mu_{\mathrm{fc}} + \sigma_{\mathrm{fc}}\,\epsilon_{\mathrm{fc}},
$$

其中：

$$
\epsilon_{\mathrm{obs}},\,\epsilon_{\mathrm{fc}} \overset{iid}{\sim} \mathcal N(0,I).
$$

InfoNCE 使用 sampled latent。

---

### Evaluation / downstream

全部使用：

$$
\boxed{ Z=\mu }
$$

包括：

- direct future probe；
- self-history probe；
- cross-channel probe；
- Oracle decoder；
- Stage-2 FM condition；
- FM endpoints；
- latent covariance；
- DE perturbation geometry。

DySIB 论文自己的 downstream probe 也使用 posterior mean $\mu(x)$ 作为 representation。

---

## 三十一、第一优先级评估：direct future R²

冻结 Encoder。

定义：

$$
X_{\mathrm{probe}} = \mu_{\mathrm{obs}}
$$

target：

$$
Y_{\mathrm{probe}} = \mu_{\mathrm{fc}}.
$$

注意：

$$
\boxed{ \text{绝对不要预测 } \Delta z = z_{\mathrm{fc}}-z_{\mathrm{obs}} }
$$

继续直接预测：

$$
z_{\mathrm{fc}}.
$$

保持与目前 gate 完全一致的 probe architecture、regularization 和 split。

---

## 三十二、成功阈值：direct R²

你当前 baseline：

$$
R^2_{\mathrm{direct}} \approx 0.103.
$$

DySIB 第一层成功标准推荐定义成：

$$
\boxed{ \Delta R^2_{\mathrm{direct}} := R^2_{\mathrm{DySIB}} - R^2_{\mathrm{VAE}} > 0 }
$$

且 subject-level bootstrap：

$$
95\%\;CI(\Delta R^2)>0.
$$

实际 practical target：

$$
\boxed{ \Delta R^2\ge0.05 }
$$

即大约：

$$
R^2_{\mathrm{direct}} \gtrsim 0.15
$$

才值得认为是有实际意义的 improvement。

如果能达到：

$$
R^2\ge0.20
$$

可以认为 improvement 已经比较强。

---

## 三十三、Shuffle R²

随机破坏：

$$
(z_{\mathrm{obs},i},\,z_{\mathrm{fc},i})
$$

配对。

期望：

$$
R^2_{\mathrm{shuffle}} \approx 0.
$$

推荐 gate：

$$
|R^2_{\mathrm{shuffle}}| < 0.01
$$

或者至少：

$$
95\%\;CI
$$

必须包含 0，且不能显著为正。

否则说明 probe / split 有 leakage。

---

## 三十四、最重要：cross-channel Gain

训练两个 probe。

### Self

$$
\mu_{\mathrm{obs}}^{(c)} \rightarrow \mu_{\mathrm{fc}}^{(c)}.
$$

得到：

$$
R^2_{\mathrm{self},c}.
$$

### Full-history

$$
\mu_{\mathrm{obs}}^{(1:C)} \rightarrow \mu_{\mathrm{fc}}^{(c)}.
$$

得到：

$$
R^2_{\mathrm{full},c}.
$$

定义：

$$
\boxed{ \mathrm{Gain}_c = R^2_{\mathrm{full},c} - R^2_{\mathrm{self},c} }
$$

以及：

$$
\mathrm{Gain}_{\mathrm{cross}} = \operatorname{aggregate}_c\, \mathrm{Gain}_c.
$$

---

## 三十五、cross-channel 成功标准

最低 gate：

$$
\boxed{ \mathrm{Gain}_{\mathrm{cross}}>0 }
$$

且：

$$
\boxed{ 95\%\;CI \text{ 不跨 }0 }
$$

建议 practical target：

$$
\mathrm{Gain}_{\mathrm{cross}} \ge 0.02.
$$

如果达到：

$$
\ge0.05
$$

则已经是相当明确的跨通道动力学增益。

如果：

$$
R^2_{\mathrm{direct}} \uparrow
$$

但：

$$
\mathrm{Gain}_{\mathrm{cross}}\approx0,
$$

结论应是：

> DySIB 改善了单通道 temporal state representation，但仍没有证据表明 latent 保留了对 EC 所需的额外跨通道动力学结构。

---

## 三十六、第二优先级：Oracle Decoder

Pure DySIB 不训练 decoder。

因此 DYSIB-02 要重新训练一个。

关键是：

$$
\boxed{ \text{Encoder 完全冻结} }
$$

然后：

$$
\mu(X) \rightarrow D_\psi \rightarrow \hat X.
$$

Decoder 不允许反向改变 Encoder。

---

## 三十七、Oracle decoder reference architecture

输入：

$$
(B,\,C,\,16,\,D).
$$

fold：

$$
(B\cdot C,\,D,\,16).
$$

首先：

```text
Conv1d D→256, kernel=1
```

然后三级 upsample：

```text
16 → 32
256 → 128

32 → 64
128 → 64

64 → 128
64 → 32
```

每级：

```text
Linear interpolation ×2
Conv1d k=5
GroupNorm
GELU

Conv1d k=3
GroupNorm
GELU
Residual
```

最后：

```text
Conv1d 32→1, k=7, padding=3
```

得到：

$$
(B,\,C,\,128).
$$

**无 skip。**

这与目前已经确认"不允许 decoder 通过 skip 绕过 latent"的原则一致。

---

## 三十八、Oracle 成功标准

优先保持与你当前 VAE Oracle 完全相同的：

- MAE；
- Pearson；
- PSD / spectral error；
- temporal derivative error。

我建议把：

$$
\boxed{ \text{Oracle primary metric 相对下降}\le5\% }
$$

作为 PASS。

5–10%：

> borderline。

超过 10%：

> 说明 pure DySIB 丢弃了太多 downstream EEG synthesis 所需信息，进入 Hybrid DySIB 实验。

---

## 三十九、Latent effective dimension

建议至少计算三个指标。

### Active units

例如：

$$
\mathrm{AU}_j = \mathbf 1\left[ \operatorname{Var}_x(\mu_j)>10^{-2} \right].
$$

---

### Covariance effective rank

设：

$$
\lambda_1,\,\dots,\,\lambda_d
$$

为 covariance eigenvalues。

使用 participation ratio：

$$
\boxed{ d_{\mathrm{eff}} = \frac{ \left(\sum_i\lambda_i\right)^2 }{ \sum_i\lambda_i^2 } }
$$

---

### Predictive saturation

比较：

$$
D=8,\;16,\;32.
$$

如果：

$$
D=8\rightarrow16
$$

predictive R² 明显提升，而：

$$
16\rightarrow32
$$

几乎不提升，

则：

$$
D=16
$$

可能更接近实际 predictive dimensionality。

DySIB 原论文也将 latent dimensionality 本身视为一个重要 bottleneck regularizer。

---

## 四十、effective dimension 不建议设死阈值

例如：

$$
d_{\mathrm{eff}}=20
$$

对于：

$$
d=128
$$

不一定意味着 collapse。

因为 DySIB 本来就是想寻找低维 predictive state。

所以真正的问题应该是：

$$
d_{\mathrm{eff}}\downarrow
$$

的同时：

$$
R^2_{\mathrm{direct}}\uparrow
$$

还是：

$$
R^2_{\mathrm{direct}}\downarrow.
$$

前者可能是成功压缩。

后者才是真 collapse。

---

## 四十一、posterior variance diagnostics

记录：

$$
\overline{\sigma},\quad \operatorname{median}\sigma,\quad P(\ell=-8),\quad P(\ell=4).
$$

同时记录 per-dimension KL。

重点防止两个极端：

### posterior collapse

$$
\mu\approx0,\qquad \sigma\approx1.
$$

### deterministic collapse

$$
\sigma\rightarrow0.
$$

建议 clamp saturation：

$$
<5\%
$$

作为基础数值健康 gate。

---

## 四十二、额外强烈建议：subject-ID probe

虽然不属于核心 metric，但 EEG DySIB 非常值得加。

冻结 $\mu$，训练一个简单 classifier：

$$
\mu \rightarrow \text{subject ID}.
$$

比较：

$$
\mathrm{Acc}_{\mathrm{VAE}}
$$

与：

$$
\mathrm{Acc}_{\mathrm{DySIB}}.
$$

如果：

$$
R^2_{\mathrm{direct}}\uparrow
$$

同时：

$$
\mathrm{Acc}_{\mathrm{subject}} \uparrow\uparrow
$$

则必须怀疑 DySIB 仍在利用 subject identity。

这也是为什么 same-subject negatives 是必要的。

---

## 四十三、β / D 选择标准

不要按：

$$
\text{train loss}
$$

选。

也不要按最终 test 结果选。

推荐 validation criterion：

第一层先满足：

$$
R^2_{\mathrm{direct}}>\text{VAE}
$$

且：

$$
\mathrm{Gain}_{\mathrm{cross}}>0.
$$

然后要求：

$$
\mathrm{Oracle}
$$

不发生明显退化。

最后在多个满足条件的配置中选择：

$$
\boxed{ \text{最小的 }D + \text{最大的可接受压缩强度} }
$$

例如：

若：

| $D$ | $\beta$ | direct $R^2$ | Gain | Oracle |
|---|--:|--:|--:|--:|
| 8 | 10 | .16 | .03 | PASS |
| 8 | 30 | .18 | .04 | PASS |
| 8 | 100 | .18 | .04 | FAIL |
| 16 | 30 | .19 | .04 | PASS |
| 32 | 30 | .191 | .041 | PASS |

那么优先考虑：

$$
D=8,\;\beta=30
$$

而不是直接选 D32。

---

## 四十四、第一轮固定配置

我建议 DYSIB-01 第一版只跑：

| 参数 | 固定值 |
|---|---|
| EEG channels | 128 |
| Obs length | 128 |
| Future length | 128 |
| overlap | 0 |
| $T_{\mathrm{lat}}$ | 16 |
| $D_{\mathrm{lat}}$ | **8** |
| per-channel $d$ | 128 |
| Encoder channel mixing | **No** |
| δ channel mixing | 1× MHSA |
| δ hidden | 128 |
| attention heads | 4 |
| stochastic training | Yes |
| obs/fc ε | **independent** |
| inference | deterministic $\mu$ |
| β | **10** |
| batch | 128 |
| batch composition | 8 subjects × 16 pairs |
| same negatives | 15 |
| cross negatives | 15 |
| NCE score normalization | mean over $CTD$ |
| optimizer | AdamW |
| LR | $10^{-4}$ |
| weight decay | $10^{-4}$ |
| gradient clip | 1.0 |
| max epochs | 200 |
| early-stop patience | 20 |
| seeds | 10 |

原 DySIB 实验也是使用 Adam、$10^{-4}$ learning rate、大 batch

尤其值得指出：论文报告会按 test mutual information 选择 run；**你的实验不要沿用这个做法**，因为你当前协议要求严格 held-out。模型和超参数必须在 validation/calibration subjects 上选定，然后 test 只报告冻结配置。你现有数据协议也已经明确将 training、calibration 和 test 的职责分离。

---

## 四十五、完整实验顺序

### DYSIB-01 — Representation feasibility

只训练：

$$
\text{Encoder}+\delta\text{-predictor}.
$$

不训练：

- decoder；
- FM；
- DE。

评估：

$$
R^2_{\mathrm{direct}}
$$

$$
R^2_{\mathrm{shuffle}}
$$

$$
R^2_{\mathrm{self}}
$$

$$
R^2_{\mathrm{full}}
$$

$$
\mathrm{Gain}_{\mathrm{cross}}
$$

以及：

- held-out NCE；
- train–val gap；
- posterior variance；
- effective dimension；
- active units；
- subject-ID probe。

#### Gate

必须首先看到：

$$
R^2_{\mathrm{direct}}>0.103
$$

且最好：

$$
\mathrm{Gain}_{\mathrm{cross}}>0.
$$

否则：

$$
\boxed{ \text{不进入 FM} }
$$

---

## 四十六、DYSIB-02 — Oracle sufficiency

冻结：

$$
E_\theta.
$$

训练：

$$
D_\psi.
$$

评估：

$$
x_{\mathrm{fc}} \rightarrow \mu_{\mathrm{fc}} \rightarrow D_\psi \rightarrow \hat x_{\mathrm{fc}}.
$$

比较：

$$
\mathrm{Oracle}_{\mathrm{DySIB}}
$$

vs.

$$
\mathrm{Oracle}_{\mathrm{VAE}}.
$$

如果 predictive representation 提升，但 Oracle 严重下降：

> 进入 Hybrid DySIB。

---

## 四十七、DYSIB-03 — FM feasibility gate

DySIB Encoder 和新 Decoder 都冻结。

然后使用与你当前 Stage-2 完全相同的 FM。

这里：

$$
\boxed{ \text{不要因为换了 Encoder 顺便改 FM architecture} }
$$

否则实验无法归因。

继续使用已经确认的显式 FM loss：

$$
z_t = (1-t)\,z_{\mathrm{obs}} + t\,z_{\mathrm{fc}}
$$

和：

$$
u_t = z_{\mathrm{fc}}-z_{\mathrm{obs}}.
$$

当前方案已经确定显式 FM loss 是 Stage-2 主监督。

这里又可以恢复 Stage-2 的 shared posterior ε protocol，因为这里的问题已经从"估计 mutual information"变回"降低 transition target sampling variance"。

---

## 四十八、DYSIB-04 — Full forecast generalization

最后比较：

$$
\text{VAE latent + FM}
$$

vs.

$$
\text{DySIB latent + same FM}.
$$

评估：

- test Pearson；
- MAE；
- spectral error；
- endpoint error；
- ODE rollout；
- train/test gap；
- condition utilization。

只有到这里才能回答：

$$
\boxed{ \text{DySIB representation 是否真的改善了 FM 泛化} }
$$

---

## 四十九、完整 PASS gate

我建议最终用这个顺序判断：

$$
\boxed{
\begin{aligned}
&1.\ R^2_{\mathrm{direct}} \text{ 显著超过 }0.103\\
&2.\ R^2_{\mathrm{shuffle}}\approx0\\
&3.\ \mathrm{Gain}_{\mathrm{cross}}>0,\ \quad CI>0\\
&4.\ \mathrm{Oracle\ degradation}\le5\%\text{–}10\%\\
&5.\ \mathrm{Posterior\ numerically\ healthy}\\
&6.\ \mathrm{FM\ test\ generalization}\uparrow
\end{aligned}
}
$$

只有第 1–4 层先成立，第 6 层的 FM improvement 才有很清晰的因果解释。

---

## 五十、更换 Encoder 后必须重新计算哪些东西

这一部分非常重要。

### 一定重新训练 / 计算

#### 1. Decoder

旧 decoder：

$$
D_{\mathrm{old}}
$$

不能用于：

$$
Z_{\mathrm{DySIB}}.
$$

必须：

$$
\boxed{ D_{\mathrm{new}} }
$$

重新训练。

---

#### 2. Latent mean

训练集：

$$
\bar\mu.
$$

重新估计。

---

#### 3. Latent covariance

$$
\Sigma_{\mathrm{latent}}
$$

全部重新计算。

---

#### 4. 每 channel covariance

如果 DE 使用 channel-level geometry：

$$
\Sigma_c
$$

也重新估计。

---

#### 5. $S_{\mathrm{ref}}$

必须：

$$
\boxed{ \text{重新估计} }
$$

因为 latent geometry 已改变。

---

#### 6. whitening / eigendecomposition

例如：

$$
S_{\mathrm{ref}} = U\Lambda U^\top
$$

对应：

$$
S_{\mathrm{ref}}^{-1/2}
$$

全部重新计算。

---

#### 7. latent support / OOD

包括：

- Mahalanobis distribution；
- support quantiles；
- OOD cutoff；
- norm distribution。

都重新确定。

---

#### 8. $r_{\max}$

必须重新校准。

因为旧 latent 上：

$$
r
$$

的几何意义已经不存在。

---

#### 9. perturbation grid

$$
\mathcal G_r
$$

至少重新验证。

---

#### 10. DE threshold $\tau$

必须重新校准。

因为：

$$
R_{a,b}(r)
$$

的整体响应分布会随着 Encoder 和 Decoder 改变。

当前方案本来就要求 $\tau$ 在 test 前通过合成数据、训练 subject 内部交叉校准等方式冻结。

---

## 五十一、MAD 有一点需要特别区分

如果你的：

$$
\mathrm{MAD}_b
$$

严格定义为：

> 原始 training EEG target channel $b$ 的 MAD

那么数据完全没变时：

$$
\mathrm{MAD}_b
$$

**理论上不会因为 Encoder 改变。**

所以它不属于 latent statistic。

不过实践中我仍建议：

> 在新 experiment manifest 中重新计算并保存一次。

这样避免 preprocessing version 不一致。

如果你的 MAD 是基于：

- decoded response；
- reconstruction residual；
- generated baseline；

那么必须重新估计。

---

## 五十二、DE 相关需要重新冻结的完整集合

新 Encoder 下：

$$
\Theta_{\mathrm{DE}}' = \left\{ S_{\mathrm{ref}},\, W_{\mathrm{white}},\, r_{\min},\, r_{\max},\, \mathcal G_r,\, \tau,\, \text{support cutoff},\, \text{curve rules} \right\}.
$$

全部重新建立。

当前协议已经要求这些统计量和规则在测试前冻结，测试只运行最终 DE inference。

---

## 五十三、哪些东西相对原 VAE 保持不变

| 模块 | 状态 | 原因 |
|---|---|---|
| $L_{\mathrm{obs}}=128$ | 不变 | 公平对照 |
| $L_{\mathrm{fc}}=128$ | 不变 | 与 FM 任务一致 |
| $C=128$ | 不变 | EEG topology |
| $T_{\mathrm{lat}}=16$ | 不变 | 保持 FM/DE interface |
| $D_{\mathrm{lat}}=8$ main | 不变 | 和当前 D8 直接比较 |
| latent shape | 不变 | $(B,\,C,\,16,\,D)$ |
| channel-aligned latent | 不变 | 保留 DE source 语义 |
| no decoder skip | 不变 | latent 必须承担信息瓶颈 |
| downstream FM architecture | 不变 | 隔离 Encoder 贡献 |
| FM explicit loss | 不变 | 已确认 Stage-2 主目标 |
| FM shared ε | 不变 | Stage-2 sampling variance control |
| deterministic μ condition | 不变 | downstream 一致 |
| subject-held-out evaluation | 不变 | 防止 leakage |
| Oracle gate | 不变 | 检查 latent→EEG 上限 |
| DE calibration protocol | 不变 | 保持统计有效性 |

---

## 五十四、相对 VAE 真正新增 / 修改的东西

| 设计 | 新增/修改 | 动机 | 预期效果 |
|---|---|---|---|
| Past/Future shared encoder | 新增训练方式 | 强制同一 latent coordinate system | obs/fc 可直接比较 |
| Future window 参与 Stage-1 | 新增 | 给 Encoder 动力学监督 | future information ↑ |
| reconstruction loss 移除 | 修改 | 不再奖励动态无关信息 | representation 更聚焦 dynamics |
| Gaussian bottleneck 保留 | 保留但重新使用 | 控制复杂度 | 防 trivial memorization |
| InfoNCE | **新增** | 最大化 past–future predictive MI | $R^2_{\mathrm{direct}}\uparrow$ |
| δ-predictor | **新增** | 参数化 latent transition | 建立动力学 critic |
| cross-channel δ mixer | **新增** | 允许跨 electrode dynamics 进入 objective | $\mathrm{Gain}_{\mathrm{cross}}$ ↑ |
| same-subject negatives | **新增** | 阻断 subject identity shortcut | 学 brain state 而非 subject |
| balanced negative sampling | **新增** | 避免 easy cross-subject negatives 主导 | harder dynamics discrimination |
| independent training ε | **修改** | 防 stochastic MI leakage | InfoNCE 更可信 |
| deterministic μ evaluation | 保持 | 去掉 sampling noise | 稳定 probe/FM |
| new Oracle decoder | 必须新增 | latent coordinate 已变化 | 检查 EEG sufficiency |
| latent statistics recalibration | 必须新增 | geometry 已变化 | DE 有效性 |

---

## 五十五、最终推荐的 DySIB-EEG v1

如果把整个设计压缩成最核心的技术定义，就是：

$$
\boxed{ X_{128} \overset{E_\theta}{\longrightarrow} q_\theta( Z_X \mid X ) }
$$

$$
\boxed{ Y_{128} \overset{\text{same }E_\theta}{\longrightarrow} q_\theta( Z_Y \mid Y ) }
$$

其中：

$$
Z_X,\,Z_Y \in \mathbb R^{128\times16\times8}.
$$

训练阶段：

$$
\epsilon_X \perp \epsilon_Y.
$$

δ-predictor：

$$
\boxed{ r_\delta(Z_Y\mid Z_X) = \mathcal N\left( Z_X+\Delta\mu_\delta(Z_X),\, \operatorname{diag}\,e^{\ell_\delta(Z_X)} \right) }
$$

critic：

$$
\boxed{ s_{ij} = \frac{1}{CTD} \log r_\delta( Z_{Y,j} \mid Z_{X,i} ) }
$$

negative：

$$
\boxed{ 15\text{ same-subject} + 15\text{ cross-subject} }
$$

loss：

$$
\boxed{ \mathcal L = \frac1\beta \mathcal L_{\mathrm{KL}} + \mathcal L_{\mathrm{NCE}} }
$$

第一轮：

$$
\boxed{ D=8,\qquad \beta=10 }
$$

然后：

$$
\boxed{ \mathrm{DYSIB}\text{-}01 \rightarrow \mathrm{DYSIB}\text{-}02 \rightarrow \mathrm{DYSIB}\text{-}03 \rightarrow \mathrm{DYSIB}\text{-}04 }
$$

其真正的核心成功链应该是：

$$
\boxed{ \text{DySIB objective} }
$$

$$
\Downarrow
$$

$$
\boxed{ R^2_{\mathrm{direct}}:\; 0.103 \rightarrow \text{显著更高} }
$$

$$
\Downarrow
$$

$$
\boxed{ \mathrm{Gain}_{\mathrm{cross}}>0 }
$$

$$
\Downarrow
$$

$$
\boxed{ \mathrm{Oracle\ sufficiency\ maintained} }
$$

$$
\Downarrow
$$

$$
\boxed{ \mathrm{FM\ held\text{-}out\ generalization}\,\uparrow }
$$

这条链比直接"训练 DySIB → FM Pearson 上升"重要得多，因为它能够真正回答你目前的科学问题：

> **当前 FM 的失败究竟是不是因为 VAE latent 不是一个好的 dynamical state representation。**

如果 DYSIB-01 就无法把 direct $R^2$ 和 cross-channel Gain 拉起来，那么无需再投入大量算力训练 FM；如果 01 成功而 02 失败，则问题变成"predictive sufficiency 与 generative sufficiency 冲突"；如果 01、02 都成功而 03/04 仍失败，那么问题才真正重新指向 FM 本身。
