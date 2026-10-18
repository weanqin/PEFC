# OC-LatentFlow 与 DE 推断实验方案：已确认决策与当前状态汇总

> **汇总范围**：严格基于本轮历史讨论与当前实现说明 `oc_latent_flow_architecture.md`。  
> **状态口径**：  
> - **已确认**：用户明确采纳、确认或随后要求按该方案继续推进。  
> - **条件性确认**：原则已确认，但具体数值、主次位置或实现细节仍待实验决定。  
> - **待讨论 / 已暂停**：尚未形成最终规则，不应写入正式方法定稿。

---

## 0. 讨论轮次索引

为便于追溯，下文使用以下上下文编号：

| 编号 | 讨论上下文 |
|---|---|
| **C0** | 读取当前模型与 DE 实现文档：两阶段 VAE/Flow、`z_obs → z_fc`、现有 signal/latent probe、原始 `σ*` 与 DE 公式 |
| **C1** | 第一轮系统审查：显式 FM loss、条件忽略、确定性映射、obs/fc 长度、skip 路径、单调性与量纲问题 |
| **C2** | 用户逐条回应并采纳核心重构方向 |
| **C3** | posterior 采样、共享噪声耦合、潜空间加噪可行性、无量纲响应、多维熵初步讨论 |
| **C4** | Encoder–Decoder 表示上限、Stage 1 能力边界、共享重参数噪声正式确认 |
| **C5** | 潜空间加噪方案正式展开：paired mean-shift、同步扰动初始状态与 condition、共享 `S_ref` |
| **C6** | 正负连接、完整多维熵维度、训练/校准/测试统计量边界 |
| **C7** | 无独立 Calibration set：单被试自校准风险、留一被试交叉校准与时间块 cross-fitting |
| **C8** | “训练集固定 + 测试集推断”的 held-out 协议确认 |
| **C9** | `r` 网格、`r_max`、单调性诊断、越阈与删失处理；提出候选 `τ` 规则 |
| **C10** | 对训练集来源 `τ` 的风险复审；决定暂不锁定最终 `τ`，改由合成数据与训练内部交叉实验选择 |

---

# 1. 已确认决策清单

## 1.1 生成模型与两阶段训练

### 决策 1：Stage 2 加入显式 Flow Matching loss，并将其作为主损失

- **状态**：已确认。
- **来源**：C1 → C2。
- Stage 2 不再只依靠解码后的 EEG 重建损失间接学习速度场。
- 对线性条件路径：

\[
z_t=(1-t)z_{\mathrm{obs}}+tz_{\mathrm{fc}},
\]

显式速度目标为：

\[
u_t=z_{\mathrm{fc}}-z_{\mathrm{obs}}.
\]

- 主损失包含：

\[
\mathcal L_{\mathrm{FM}}
=
\mathbb E
\left[
\left\|
v_\theta(z_t,t,c_{\mathrm{obs}})
-
\operatorname{sg}(z_{\mathrm{fc}}-z_{\mathrm{obs}})
\right\|_2^2
\right].
\]

- EEG 域重建、相关性或 endpoint/rollout 类损失可作为辅助项，但不能替代显式 FM 监督。
- 该决策用于把模型从“解码终点导向的隐式预测器”约束为真正学习预定义概率路径上的速度场。

---

### 决策 2：Stage 2 采用共享重参数噪声耦合

- **状态**：已确认。
- **来源**：C3 → C4。
- 配对的 obs 与 fc posterior 使用同一随机变量：

\[
z_{\mathrm{obs}}
=
\mu_{\mathrm{obs}}
+
s_{\mathrm{obs}}\odot\epsilon,
\]

\[
z_{\mathrm{fc}}
=
\mu_{\mathrm{fc}}
+
s_{\mathrm{fc}}\odot\epsilon,
\]

其中：

\[
\epsilon\sim\mathcal N(0,I).
\]

- 对应速度目标：

\[
z_{\mathrm{fc}}-z_{\mathrm{obs}}
=
(\mu_{\mathrm{fc}}-\mu_{\mathrm{obs}})
+
(s_{\mathrm{fc}}-s_{\mathrm{obs}})\odot\epsilon.
\]

- 共享噪声用于减少独立重参数噪声给速度目标引入的额外方差，并形成更平滑的端点耦合。
- Flow 的条件信息与随机初始状态应区分；讨论中倾向使用确定性条件（至少包含 \(\mu_{\mathrm{obs}}\)）与随机状态分离。

---

### 决策 3：确认 Encoder–Decoder 构成 Flow 生成能力的上限

- **状态**：已确认的理论结论。
- **来源**：C4。
- 即使速度场完美回归并得到：

\[
\hat z_{\mathrm{fc}}=z_{\mathrm{fc}},
\]

最终输出仍受限于：

\[
\hat x_{\mathrm{fc}}^{\mathrm{oracle}}
=
D_\psi(z_{\mathrm{fc}}).
\]

- 因此需分别验证：
  1. **目标端表示充分性**：\(z_{\mathrm{fc}}\) 是否足以重建 \(x_{\mathrm{fc}}\)；
  2. **条件端预测充分性**：\(z_{\mathrm{obs}}\) 是否保留未来相关信息。
- 复杂 latent prior 不能自动恢复 Encoder 已经丢失的信息。
- 必须建立 VAE Oracle 路径：

\[
x_{\mathrm{fc}}
\rightarrow
z_{\mathrm{fc}}
\rightarrow
D_\psi(z_{\mathrm{fc}})
\]

作为 Flow 预测上限诊断。

---

### 决策 4：obs 与 fc 使用相同时间长度

- **状态**：已确认原则；具体长度待实验。
- **来源**：C2、C3、C4。
- 统一长度用于消除同一 Encoder 对不同时间长度输入进行 Adaptive Pooling 时潜时间语义不一致的问题。
- 当前候选：

\[
L\in\{128,256\}.
\]

- 最终选 128 或 256 由实验结果决定。
- 选择不能只依据 forecast Pearson，还需综合：
  - VAE Oracle 重建；
  - Flow forecast；
  - 条件利用度；
  - 合成有效连接恢复；
  - DE 稳定性；
  - 长预测窗引入间接路径混淆的风险。
- 在比较 128 与 256 时，保持：

\[
d=T_{\mathrm{lat}}\times D_{\mathrm{lat}}
\]

一致。

---

### 决策 5：取消 skip 路径，Stage 1 简化为固定片段 VAE 重建

- **状态**：已确认。
- **来源**：C2、C3。
- Stage 1 不再承担 obs→fc 预测任务，仅学习：

\[
x
\rightarrow
q_\phi(z\mid x)
\rightarrow
\hat x.
\]

- 输入为固定长度 EEG 片段。
- Decoder 不再通过 skip、直接 \(z_{\mathrm{obs}}\) 路径或 full-window 重建绕过 Flow。
- 目的：
  - 让 Encoder 只负责 EEG 压缩与表示；
  - 让 Flow 成为最终预测链路中的主要跨时段传输模块；
  - 避免 DE 响应混入 skip 直接路径。
- Stage 1 预测辅助目标暂不加入，见“待讨论断点”。

---

### 决策 6：Stage 1 预测辅助目标暂缓

- **状态**：已暂停 / 待实验后决定。
- **来源**：C4 → C5。
- 当前先训练纯固定片段 VAE。
- 待完成 Oracle 重建、条件利用度和 Flow 实验后，再判断是否加入：

\[
\mathcal L_{\mathrm{pred}}
=
\left\|
g_\omega(\mu_{\mathrm{obs}})
-
\operatorname{sg}(\mu_{\mathrm{fc}})
\right\|.
\]

- 该辅助目标不属于当前已执行主方案。

---

## 1.2 条件依赖与模型资格验证

### 决策 7：必须进行条件利用度验证

- **状态**：已确认。
- **来源**：C1 → C2，后续多轮重复确认。
- 认可 Flow Matching 训练路径中使用真实未来端点插值是标准做法，但必须证明显式条件 \(c_{\mathrm{obs}}\) 未被忽略。
- 条件利用度测试要求固定相同的 \(z_t\) 与 \(t\)，只替换 condition：
  - 正确 \(c_{\mathrm{obs}}\)；
  - batch 内打乱；
  - 置零；
  - 跨被试替换。
- 比较：
  - \(\mathcal L_{\mathrm{FM}}\)；
  - forecast MAE；
  - Pearson；
  - 频谱误差；
  - ODE endpoint/rollout 误差。
- 若错误条件对性能影响很小，则模型不能被视为学习了可信的条件代理脑动力学。

---

### 决策 8：DE 运行前需要模型资格门控

- **状态**：原则已确认；数值阈值待定。
- **来源**：C10。
- 至少包括：
  1. **Oracle 表示能力**：Decoder 对 \(z_{\mathrm{fc}}\) 的重建必须达到可接受水平；
  2. **条件利用度**：正确条件相对错误条件必须带来显著收益；
  3. **Null–扰动可分性**：扰动响应需能与模型随机生成变异区分。
- 若模型不使用 obs 条件，则调节 `τ` 不能修复 DE 的因果解释。

---

## 1.3 潜空间虚拟扰动方案

### 决策 9：DE 主扰动采用潜空间 paired mean-shift，而不是修改 posterior logvar

- **状态**：已确认。
- **来源**：C5。
- 不直接把虚拟扰动解释为增加 Encoder posterior 不确定性。
- 对单个 source channel \(a\) 的潜变量：

\[
z_a\in\mathbb R^d,
\qquad
d=T_{\mathrm{lat}}\times D_{\mathrm{lat}}.
\]

- 使用位置扰动：

\[
\delta_{a,k}(r)
=
rL_{\mathrm{ref}}\xi_k,
\qquad
\xi_k\sim\mathcal N(0,I_d).
\]

- 同一扰动方向 \(\xi_k\) 在整个 \(r\) 网格上复用，确保输入扰动尺度与 \(r\) 严格对应。

---

### 决策 10：使用正负成对、对称的 mean-shift 扰动估计无符号强度

- **状态**：已确认用于当前无符号 DE。
- **来源**：C5、C6，随后用户决定暂不推断符号。
- 构造：

\[
z_a^{+}(r)
=
\mu_a+\delta_{a,k}(r)+s_a\odot\epsilon,
\]

\[
z_a^{-}(r)
=
\mu_a-\delta_{a,k}(r)+s_a\odot\epsilon.
\]

- baseline、正扰动、负扰动共享同一 posterior 噪声 \(\epsilon\)。
- 正负扰动平均用于降低局部方向偏差、抵消部分奇数/偶数阶非线性影响并提高强度曲线稳定性。
- 当前只计算无符号强度，不从该步骤推断连接正负性。

---

### 决策 11：同步扰动 Flow 初始状态与 condition

- **状态**：已确认。
- **来源**：C5。
- 同一个 \(\delta_{a,k}(r)\) 同时作用于：
  - ODE 初始状态 \(z_{\mathrm{obs},a}\)；
  - Flow 条件 \(c_{\mathrm{obs},a}\)。
- 原因：只扰动其中一个会造成“状态已改变、条件未改变”或相反的不一致输入，使 Flow 把干预当作噪声修正，或产生无法解释的条件冲突。
- 其他通道保持不变。

---

### 决策 12：主实验使用全通道共享参考协方差 `S_ref`

- **状态**：已确认原则；具体估计器仍需实现定稿。
- **来源**：C5、C6、C8。
- 所有 source 通道使用同一：

\[
S_a=S_{\mathrm{ref}}.
\]

- 目的：
  - 保持不同 source 的扰动几何一致；
  - 避免 source-specific 方差体积混入连接强度；
  - 支持跨 source、跨 subject 比较。
- `S_ref` 仅由训练数据的 latent 统计估计并冻结。
- source-specific \(S_a\) 不进入主实验，只可作为消融。
- 已讨论过 pooled shrinkage、白化坐标或 \(S_{\mathrm{ref}}=I_d\) 等实现，但最终估计器尚未完全锁定。

---

### 决策 13：共享随机变量规则

- **状态**：已确认。
- **来源**：C5。
- baseline、\(+\delta\)、\(-\delta\) 共享 posterior 噪声：

\[
\epsilon_{n,p}.
\]

- 同一扰动方向 \(\xi_k\) 在所有 \(r\) 上复用。
- 建议同一组 \(\xi_k\) 跨 source 复用，以减少 source 间 Monte Carlo 比较方差。
- 不允许在不同 \(r\) 下重新采样其他随机变量。
- ODE 步数、数值精度和模型状态必须保持一致。

---

## 1.4 响应定义与 DE 公式

### 决策 14：使用目标通道 MAD 归一化的无量纲响应

- **状态**：已确认。
- **来源**：C2、C3、C5、C6。
- 对 target channel \(b\)，使用训练集稳健尺度：

\[
MAD_b.
\]

- 无符号响应基于正负扰动平均：

\[
R_{a,b}(r)
=
\operatorname{RobustAggregate}
\left[
\frac{
\operatorname{MSE}_t(\Delta_{a,b}^{+}(r))
+
\operatorname{MSE}_t(\Delta_{a,b}^{-}(r))
}{
2(MAD_b^2+\varepsilon)
}
\right].
\]

- `MAD_b` 使响应无量纲，并降低 EEG 单位、设备增益与通道幅值差异对阈值的直接影响。
- 聚合需优先以 subject 为统计单位，避免片段较多的被试主导群体结果。

---

### 决策 15：连接强度与连接符号分离

- **状态**：已确认。
- **来源**：C6，随后用户进一步限制为“当前只做无符号强度”。
- 当前论文主流程只输出无符号：

\[
DE^{(d)}_{a,b}.
\]

- “正负连接”“兴奋性/抑制性连接”和 signed influence 均进入待讨论队列。
- 合成数据验证完成前：
  - 不给 DE 增加符号；
  - 不将输出解释为兴奋性/抑制性连接；
  - 不作正式正负性结论。
- 若未来加入符号，需独立定义为可验证的：

> signed directed perturbational influence

而不能直接等同于神经突触兴奋/抑制。

---

### 决策 16：采用完整多维熵 `DE^(d)`

- **状态**：已确认。
- **来源**：C3、C4、C6。
- 对扰动分布：

\[
\delta_a(r)\sim\mathcal N(0,r^2S_{\mathrm{ref}}),
\]

完整多维熵连接强度为：

\[
DE^{(d)}_{a,b}
=
-
\left[
\frac d2\log(2\pi e)
+
d\log r^*_{a,b}
+
\frac12\log|S_{\mathrm{ref}}|
\right].
\]

- 当前研究保持：

\[
d=T_{\mathrm{lat}}\times D_{\mathrm{lat}}
\]

不变。
- EEG 通道数 \(C\) 的变化只改变连接矩阵尺寸，不改变单 source 扰动维度 \(d\)。
- 因而在当前研究设定内可直接比较完整多维熵。
- 若未来跨不同 \(d\) 比较，再引入每维 DE 或相对每维 DE；当前不进入主方案。

---

## 1.5 数据划分与统计量边界

### 决策 17：训练/校准/测试的角色分离

- **状态**：已确认。
- **来源**：C6、C7、C8。
- 理想协议：

#### Training

用于：
- 训练 Encoder、Flow、Decoder；
- 估计 EEG 标准化统计量；
- 估计 \(MAD_b\)；
- 估计 \(S_{\mathrm{ref}}\)；
- 构造 latent 白化/扰动几何；
- 进行训练内部收敛和稳定性分析。

#### Calibration / Validation

用于：
- 选择 `τ` 的规则；
- 确定 \(r_{\min}\)、\(r_{\max}\) 与 `r` 网格；
- 确定扰动方向数 \(K\)、posterior 样本数 \(P\)；
- 固定单调性诊断、曲线处理和越阈规则；
- 在合成数据或训练内部 held-out subject 上比较候选方案。

#### Test

只执行冻结方法下的：
- \(\mu_{\mathrm{obs}},s_{\mathrm{obs}}\) 计算；
- baseline 与 perturbation inference；
- \(R_{a,b}(r)\)；
- \(r^*_{a,b}\)；
- \(DE^{(d)}_{a,b}\)；
- 置信区间、稳定性和最终验证指标。

---

### 决策 18：“训练集固定 + 测试集推断”属于严格 held-out

- **状态**：已确认。
- **来源**：C8。
- 若模型和完整 DE 配置在测试前冻结：

\[
\Theta_{\mathrm{DE}}
=
\{
MAD_b,\,
S_{\mathrm{ref}},\,
\tau,\,
r_{\min},r_{\max},\,
\mathcal G_r,\,
K,P,\,
\text{aggregation},\,
\text{curve rule},\,
\text{crossing rule}
\},
\]

测试阶段仅运行：

\[
R_{a,b}(r)
\rightarrow
r^*_{a,b}
\rightarrow
DE^{(d)}_{a,b},
\]

则属于严格 held-out 评估。
- 测试集不能用于重新调整 `τ`、`r_max`、网格、越阈规则或协方差。
- 测试集可做只读式分布迁移审计，但不能根据审计结果修改主方案。

---

### 决策 19：无独立 Calibration set 时优先采用留一被试交叉校准

- **状态**：已确认的首选替代协议。
- **来源**：C7。
- 对外层测试 subject \(s\)，所有 DE 辅助统计量由其他 subject 估计：

\[
\Theta_{-s}
=
\{
MAD_{b,-s},
S_{\mathrm{ref},-s},
\tau_{-s},
r_{\min},r_{\max},K,P,\ldots
\}.
\]

- 然后：

\[
DE_s
=
\mathcal A(\mathcal D_s;\Theta_{-s}).
\]

- 该协议避免 subject 同时参与自身参数估计与最终 DE 计算。
- 若只有一个 subject，可采用连续时间块 cross-fitting，但结论只能限定为 subject-level relative DE。

---

### 决策 20：测试 subject 自适应只允许作为附加协议

- **状态**：已确认。
- **来源**：C7、C8。
- 主实验不在测试 subject 上重新估计：
  - \(S_{\mathrm{ref}}\)；
  - `τ`；
  - \(r_{\max}\)。
- subject-specific `MAD_b` 或单一增益 \(g_s\) 可作为 target-adaptive 消融，但会改变跨 subject 绝对可比性。
- 自适应结果必须与 strict held-out 主结果分开报告。

---

## 1.6 `r` 搜索、响应曲线与临界尺度

### 决策 21：使用对数 `r` 网格

- **状态**：已确认。
- **来源**：C9，用户随后明确认可除 `τ` 外的参数方案。
- 网格：

\[
\mathcal G_r=\{0,r_1,\ldots,r_M\}.
\]

- 正值点按 log-space 等距分布。
- 初始建议为 24 个正值点，加 \(r=0\) baseline。
- 最终网格密度需通过训练/合成数据上的稳定性比较确认。

---

### 决策 22：`r_max` 由潜空间支持域约束，而不是强迫所有边越阈

- **状态**：已确认原则；具体数值待定。
- **来源**：C9。
- `r_max` 必须保证扰动 latent 大部分仍处于训练 latent 支持域内。
- 不允许无限增大 `r_max` 直到所有边越阈。
- 需要检查：
  - latent Mahalanobis/OOD 距离；
  - 解码振幅；
  - 频谱异常；
  - NaN/Inf；
  - 饱和行为。
- 在有效 `r_max` 内未越阈的边应作为右删失，而不是继续扩大扰动。

---

### 决策 23：不直接假设 `R_{a,b}(r)` 严格单调

- **状态**：已确认。
- **来源**：C1、C3、C5、C9。
- 固定扰动方向只保证输入扰动范数随 \(r\) 增大，不保证非线性 Flow–Decoder 的输出 MSE 单调。
- 必须计算至少：
  - Spearman 单调相关；
  - 相邻下降比例；
  - 下降质量；
  - 多次阈值穿越次数。
- 若大量曲线严重非单调，应返回加噪方案或模型稳定性检查，不应单纯依靠阈值修补。

---

### 决策 24：`r*` 定义为首次持续越阈

- **状态**：已确认。
- **来源**：C9。
- 临界位置定义为第一次连续多个网格点达到或超过阈值。
- 当前候选主规则为连续 2 点越阈；2 或 3 点的最终选择需在训练/合成实验中确认。
- 插值在 \(\log r\) 空间进行，因为完整多维熵直接依赖 \(\log r^*\)。

---

### 决策 25：边界情况采用删失标记，而不是把 clamping 当作精确值

- **状态**：已确认。
- **来源**：C9。
- 若：

\[
R(r_{\min})\ge\tau,
\]

则：

\[
r^*\le r_{\min},
\]

标记为 left-censored。

- 若：

\[
R(r_{\max})<\tau,
\]

则：

\[
r^*>r_{\max},
\]

标记为 right-censored。

- 可为绘图保存 display value，但统计分析必须保留 censor flag。
- 严重非单调曲线标记为 unreliable，不应强制生成精确 DE。

---

### 决策 26：非单调曲线可使用条件式单调拟合，但主次位置尚未最终锁定

- **状态**：条件性确认。
- **来源**：C9，最新汇总请求要求将主次方案列为断点。
- 已确认原则：
  - 不无条件强制单调；
  - 轻度非单调可进行单调拟合；
  - 严重非单调应标记 unreliable。
- 候选处理为 weighted isotonic regression。
- 仍待确定：
  - isotonic 是否作为主流程；
  - raw first-crossing 是否作为主方案或消融；
  - 轻度/严重非单调的最终阈值。

---

## 1.7 `τ` 的当前状态

### 决策 27：暂不锁定最终 `τ` 来源或公式

- **状态**：明确未决。
- **来源**：C9 → C10。
- 已否定“未经验证直接采用训练集 Null 的 \(Q_{0.99}\)”作为最终定稿。
- 训练集完全固定 `τ` 仍是 held-out 主实验的必要边界，但 **`τ` 的生成规则** 需要先通过：
  - 合成数据；
  - 训练 subject 内部交叉迁移实验；
  - Null–扰动分离度；
  - 分布偏移场景；
  - FPR、TPR、AUPRC 等指标
  进行选择。
- 测试集可以审计训练阈值是否迁移，但不能据此重新选择主 `τ`。
- 当前候选队列包括：
  - 训练 Null 分位数；
  - 训练最大响应比例；
  - 合成无边响应的 FPR 控制；
  - 混合保守阈值。
- 候选不等于定稿。

---

# 2. 实验方案执行顺序

## 2.1 总体流程链

```text
A. 数据划分与固定预处理
→ B. 128/256 固定长度候选设置
→ C. Stage 1 固定片段 VAE 重建
→ D. VAE Oracle 表示能力评估
→ E. Stage 2 显式 Flow Matching 训练
→ F. Flow 预测质量与 train/inference 一致性评估
→ G. 条件利用度验证
→ H. 模型资格门控
→ I. 用训练数据估计 MAD_b 与 S_ref
→ J. 确定 paired latent mean-shift 扰动实现
→ K. 确定 log-r 网格与训练支持域约束的 r_max
→ L. 构造 Null 分布与扰动响应曲线
→ M. 单调性诊断与曲线质量筛查
→ N. 合成数据上的 τ 候选比较与 EC 恢复实验
→ O. 训练 subject 内部交叉校准并冻结 τ 规则
→ P. 测试集冻结参数下计算 R_{a,b}(r)
→ Q. 提取 r^*、删失标记与 DE^(d)
→ R. 跨 seed / fold / subject 稳定性与真实数据泛化验证
→ S. 后续再决定是否讨论 signed influence
```

---

## 2.2 分步执行说明

### A. 数据划分与预处理

1. 按 subject 划分训练集与测试集。
2. EEG 标准化统计量仅由训练数据估计。
3. 若存在独立 validation/calibration，则在测试前完成所有参数选择。
4. 无独立 calibration 时，在训练 subject 内进行交叉校准。

### B. 窗口长度候选实验

1. 比较 128 与 256。
2. 保持相同：
   - \(T_{\mathrm{lat}}\)；
   - \(D_{\mathrm{lat}}\)；
   - \(d\)；
   - 尽可能相同的模型容量与训练协议。

### C. Stage 1 VAE

1. 固定长度 EEG 片段重建。
2. 无 skip。
3. 不加入预测辅助目标。
4. 监控 posterior collapse、active units、KL 与重建质量。

### D. Oracle 表示能力

直接使用：

\[
x_{\mathrm{fc}}
\rightarrow
q_\phi(z_{\mathrm{fc}}\mid x_{\mathrm{fc}})
\rightarrow
D_\psi(z_{\mathrm{fc}})
\]

评估目标端潜空间上限。

### E. Stage 2 Flow Matching

1. 冻结 Stage 1 编解码器作为当前主设置。
2. obs/fc 使用共享重参数噪声。
3. 显式 \(\mathcal L_{\mathrm{FM}}\) 为主损失。
4. EEG/endpoint/rollout 损失作为辅助。

### F. Flow 预测评估

比较：
- Oracle Decoder；
- Flow endpoint；
- 完整 ODE rollout；
- forecast MAE、Pearson、频谱与 endpoint 误差。

### G. 条件利用度

固定 \(z_t,t\)，分别使用正确、打乱、置零、跨被试 condition。

### H. 模型资格门控

只有 Oracle、条件利用度和 Null–扰动可分性达到要求后，才进入正式 DE 推断。

### I. 训练参考统计量

由训练数据计算并保存：
- \(MAD_b\)；
- \(S_{\mathrm{ref}}\)；
- 对应分解或白化矩阵；
- latent 支持域统计量。

### J. 潜空间扰动

对每个 source：
- baseline、\(+\delta\)、\(-\delta\) 共享 posterior 噪声；
- 初始状态和 condition 同步 mean-shift；
- 所有 source 使用共享 \(S_{\mathrm{ref}}\)。

### K. `r` 网格与 `r_max`

1. 使用 log-space 网格。
2. \(r_{\max}\) 受训练 latent 支持域约束。
3. 网格密度与数值稳定性在训练/合成数据上确认。

### L. Null 与响应曲线

1. 构造无扰动随机变异 Null。
2. 构造 \(R_{a,b}(r)\)。
3. 使用训练 `MAD_b` 做无量纲化。

### M. 单调性诊断

检查每条曲线的单调相关、下降质量和穿越次数。

### N. 合成数据阈值与 EC 恢复

在已知真实连接矩阵下比较 `τ` 候选规则，并评估：
- AUPRC；
- AUROC；
- FPR；
- TPR；
- specificity；
- 连接强度排序；
- 不同分布偏移下的稳定性。

### O. 冻结 `τ` 规则

结合：
- 合成数据；
- 训练 subject 内部交叉校准；
- Null–扰动分离度；
决定最终 `τ` 规则。

### P–Q. 测试 DE

测试集只计算：
- \(R_{a,b}(r)\)；
- 单调性质量；
- \(r^*_{a,b}\)；
- censor/unreliable flags；
- \(DE^{(d)}_{a,b}\)。

### R. 泛化与稳定性

报告：
- 跨 seed；
- 跨 fold；
- 跨 subject；
- 跨数据集；
- latent OOD；
- 左/右删失比例；
- 测试 Null–response 分布迁移。

### S. 符号扩展

只有无符号 DE 与合成验证完成后，才重新讨论 signed directed perturbational influence。

---

# 3. 依赖关系

| 步骤 / 决策 | 前置条件 | 直接依赖 |
|---|---|---|
| Stage 1 VAE | 固定长度方案、无 skip 架构 | 数据划分、预处理 |
| VAE Oracle | Stage 1 收敛 | Encoder–Decoder 表示能力 |
| Stage 2 FM | 可用 VAE latent | Stage 1、共享重参数噪声 |
| 条件利用度验证 | 已训练 Flow | 固定 \(z_t,t\)，condition 替换 |
| 正式 DE 运行 | 模型通过资格门控 | Oracle、条件利用度、Null–扰动可分性 |
| `MAD_b` | 训练 EEG 可用 | 训练数据与目标通道定义 |
| `S_ref` | 冻结 Encoder 可用 | 训练 latent、共享扰动几何 |
| paired mean-shift | `S_ref` 可用 | posterior pairing、初始状态/condition 同步 |
| `r_max` | 训练 latent 支持域可估计 | `S_ref`、OOD 判据 |
| `R_{a,b}(r)` | 扰动实现与 `MAD_b` 已固定 | paired perturbation、log-r 网格 |
| 单调性诊断 | 已获得响应曲线 | \(R_{a,b}(r)\) |
| `τ` | Null、合成数据、交叉校准 | 模型资格、响应分布、无边 ground truth |
| \(r^*_{a,b}\) | `τ` 与曲线处理规则已冻结 | \(R_{a,b}(r)\)、单调性诊断 |
| `DE^(d)` | 精确或删失的 \(r^*\)、固定 \(S_ref\)、固定 \(d\) | 完整多维熵公式 |
| 测试 held-out DE | 所有参数测试前冻结 | Training/Calibration 规则 |
| 正负/符号推断 | 无符号 DE 与合成验证完成 | 已知正负 ground truth、独立操作性定义 |

---

# 4. 断点标注

## 断点 1：`τ` 的最终选择规则

- **状态**：待讨论 / 当前最主要断点。
- **已确认**：
  - 测试集不能用于选择主 `τ`；
  - 最终 `τ` 在测试前冻结；
  - 需通过合成数据与训练内部交叉实验选择。
- **未确认**：
  - Null 分位数；
  - 最大响应比例；
  - 合成无边 FPR 控制；
  - 混合阈值；
  - 单一全局 `τ` 是否足够。
- **依赖卡点**：
  - 条件利用度；
  - Null–扰动分离度；
  - 合成数据 ground truth；
  - 训练/测试分布偏移模拟。

---

## 断点 2：`r_max` 的具体数值

- **状态**：原则已确认，数值未定。
- **已确认**：
  - 由训练 latent 支持域约束；
  - 不能为让全部边越阈而无限增大。
- **未确认**：
  - Mahalanobis 分位点；
  - 允许 OOD 比例；
  - 不同窗口长度下是否使用相同数值。
- **依赖卡点**：
  - Stage 1 latent 分布；
  - `S_ref` 估计；
  - 扰动后解码信号的合理性分析。

---

## 断点 3：`S_ref` 的具体估计器

- **状态**：共享原则已确认，实现细节未完全锁定。
- **已确认**：
  - 训练集估计；
  - 全通道共享；
  - source-specific 只做消融。
- **未确认**：
  - pooled sample covariance + shrinkage；
  - 统一白化坐标；
  - 对角/低秩近似；
  - 是否直接使用 \(I_d\) 基线。
- **依赖卡点**：
  - 训练样本有效量；
  - 协方差条件数；
  - `log|S_ref|` 数值稳定性。

---

## 断点 4：非单调曲线处理的主次方案

- **状态**：原则确认，主次方案待定。
- **已确认**：
  - 不默认严格单调；
  - 轻度违背可拟合；
  - 严重违背标记 unreliable；
  - 不能用“最后一个稳定点”作为唯一规则。
- **未确认**：
  - weighted isotonic 是否作为主方案；
  - raw first-crossing 是否作为主方案或消融；
  - 单调质量阈值；
  - 连续 2 点还是 3 点越阈。
- **依赖卡点**：
  - 训练与合成响应曲线的实际形态；
  - 网格密度；
  - Monte Carlo 采样稳定性。

---

## 断点 5：Stage 1 预测辅助目标

- **状态**：已暂停。
- **当前方案**：先执行纯固定片段 VAE。
- **何时重启**：
  - Oracle 重建良好；
  - 但 \(z_{\mathrm{obs}}\) 条件预测充分性不足；
  - Flow 条件利用度弱。
- **依赖卡点**：
  - Stage 1/Stage 2 首轮实验结果。

---

## 断点 6：正负连接与兴奋/抑制语义

- **状态**：已暂停。
- **当前方案**：只输出无符号 `DE^(d)`。
- **未决内容**：
  - 正向扰动的生理语义；
  - signal-space 或语义锚定 latent 方向；
  - signed influence 统计量；
  - 正负连接与兴奋/抑制的边界。
- **依赖卡点**：
  - 已知正负耦合的合成数据；
  - 无符号 DE 先通过；
  - 可复核的符号准确率。

---

## 断点 7：窗口长度 128 或 256

- **状态**：待实验决定。
- **比较依据**：
  - VAE Oracle；
  - Flow forecast；
  - 条件利用度；
  - 合成 EC 恢复；
  - DE 稳定性；
  - 间接路径混淆。
- **依赖卡点**：
  - 两套完整训练与验证结果。

---

## 断点 8：删失边的群体统计方式

- **状态**：未讨论完成。
- **已确认**：
  - left/right censor 必须保留标记；
  - clamping 只能用于显示。
- **未确认**：
  - 群体平均、组间比较或网络统计中如何处理删失值。
- **依赖卡点**：
  - 最终 `τ`；
  - \(r_{\max}\)；
  - 实际删失比例。

---

# 5. 当前状态定位

**当前讨论已完成生成框架重构、潜空间扰动定义、训练/测试边界、`r` 网格与临界尺度原则设计；距离形成可直接执行的完整实验方案，仍缺少 `τ` 的最终选择规则、`S_ref` 与 `r_max` 的具体实现参数、非单调曲线处理主方案，以及相应的合成数据验证协议定稿。**
