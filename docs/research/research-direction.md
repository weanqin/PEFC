# 基于 EEG 数据的虚拟扰动推断研究框架

## 一、研究主题

### 基于 EEG 代理脑虚拟扰动的节点连接推断

研究目标是利用 EEG 数据训练一个能够模拟脑电时序动态的代理模型（surrogate brain），然后在代理模型内部进行虚拟扰动实验，观察扰动从某一源节点向其他节点传播所引起的响应变化，据此构建节点间有方向的连接强度。

整体研究可以抽象为：

\[
\text{EEG dynamics learning}
\rightarrow
\text{surrogate brain}
\rightarrow
\text{virtual perturbation}
\rightarrow
\text{response measurement}
\rightarrow
\text{connectivity inference}
\]

---

## 二、研究背景

传统 EEG 连接分析通常从观测信号之间的统计关系出发，但本研究希望采用一种更接近“扰动—响应”实验的思路：

\[
A\text{发生变化}
\quad\Longrightarrow\quad
B\text{是否随之发生变化？}
\]

直接参考文献为：

> **Mapping effective connectivity by virtually perturbing a surrogate brain**  
> arXiv:2301.00148v3

参考文献中的 NPI（Neural Perturbational Inference）采用“代理脑 + 虚拟扰动”的思路：先训练人工神经网络学习神经动力学，使其成为真实脑动态的计算代理；随后固定模型，在不同节点上施加虚拟扰动，并通过比较扰动与未扰动状态下其他节点的输出变化推断节点间连接。

本研究在这一思想基础上，将研究对象转移至 EEG，并将代理脑的基础任务描述为：

\[
\text{过去 EEG}
\rightarrow
\text{未来 EEG}
\]

因此，本研究可以理解为：

> **利用 EEG 历史段学习未来 EEG 的生成规律，构建 EEG 代理脑；随后在代理脑中施加虚拟扰动，并依据扰动传播响应推断节点间连接。**

需要特别说明的是：

- “代理脑 + perturb-and-record”来自参考文献的基本框架；
- “过去 EEG → 未来 EEG”的条件生成形式，是本研究对代理脑建模任务的具体定义；
- “逐步增加扰动直到响应超过阈值，并以临界扰动强度度量连接”属于本研究进一步设计的推断方法，不是参考论文已经给出的结论。

---

## 三、核心科学任务

整个研究可划分为两个前后依赖的核心任务。

### 任务一：EEG 代理脑建模

给定一段历史 EEG：

\[
X_{\mathrm{past}}
=
[x_{t-T_p+1},\ldots,x_t]
\]

模型生成未来 EEG：

\[
\hat X_{\mathrm{future}}
=
[\hat x_{t+1},\ldots,\hat x_{t+T_f}]
\]

即学习：

\[
\boxed{
X_{\mathrm{past}}
\rightarrow
X_{\mathrm{future}}
}
\]

也可以表示为条件生成问题：

\[
p(
X_{\mathrm{future}}
\mid
X_{\mathrm{past}}
)
\]

该任务可以从以下三个视角理解：

1. **时序预测**

   \[
   \text{Past EEG}\rightarrow\text{Future EEG}
   \]

2. **条件生成**

   \[
   p(X_{\mathrm{future}}\mid X_{\mathrm{past}})
   \]

3. **时间序列修补 / 条件补全**

   \[
   [\text{observed past},\text{masked future}]
   \rightarrow
   [\text{observed past},\text{generated future}]
   \]

这一部分的目标不是直接推断连接，而是先获得一个能够较好描述 EEG 动态演化规律的**代理脑模型**。

---

### 任务二：虚拟扰动连接推断

代理脑训练完成后，对模型输入或中间层的源节点施加虚拟扰动或噪声，并观察其他节点输出变化。

研究目标是建立：

\[
\text{Perturbation}
\rightarrow
\text{Response}
\rightarrow
\text{Critical Intensity}
\rightarrow
\text{Connectivity}
\]

即：

1. 对源节点 \(A\) 施加扰动；
2. 测量目标节点 \(B\) 的输出变化；
3. 逐渐增加扰动强度；
4. 找到目标节点响应达到预设阈值时的最小扰动强度；
5. 将其定义为临界扰动强度；
6. 基于临界扰动强度构建节点连接度量。

---

# 四、总体方法框架

整个方法分为两个阶段。

## Stage I：EEG Surrogate Brain Learning

输入：

\[
X_{\mathrm{past}}\in
\mathbb{R}^{C\times T_p}
\]

其中：

- \(C\)：EEG 节点或通道数量；
- \(T_p\)：历史 EEG 时间长度。

模型学习：

\[
F_\theta:
X_{\mathrm{past}}
\rightarrow
X_{\mathrm{future}}
\]

输出：

\[
\hat X_{\mathrm{future}}
\in
\mathbb{R}^{C\times T_f}
\]

其中 \(T_f\) 为需要生成的未来 EEG 时间长度。

训练完成后固定模型参数：

\[
\theta=\theta^\*
\]

得到 EEG 代理脑：

\[
F_{\theta^\*}
\]

这一阶段的核心问题是：

\[
\boxed{
\text{怎样设计模型，使 }
X_{\mathrm{past}}
\rightarrow
X_{\mathrm{future}}
\text{ 的生成足够可靠？}
}
\]

---

## Stage II：Virtual Perturbation

对于一个未扰动历史 EEG：

\[
X_{\mathrm{past}}
\]

首先获得基线未来 EEG：

\[
\hat X^{(0)}
=
F_{\theta^\*}(X_{\mathrm{past}})
\]

然后选择源节点 \(A\)，施加强度为 \(r\) 的虚拟扰动：

\[
X_{\mathrm{past}}^{(A,r)}
=
\mathcal P
(X_{\mathrm{past}},A,r)
\]

其中：

- \(A\)：被扰动源节点；
- \(r\)：扰动强度；
- \(\mathcal P\)：扰动算子。

扰动后得到：

\[
\hat X^{(A,r)}
=
F_{\theta^\*}
(X_{\mathrm{past}}^{(A,r)})
\]

对于目标节点 \(B\)，定义输出变化：

\[
\Delta_{A\rightarrow B}(r)
=
D
\left(
\hat X_B^{(A,r)},
\hat X_B^{(0)}
\right)
\]

其中 \(D(\cdot,\cdot)\) 为输出变化度量函数。

于是得到：

\[
\boxed{
r
\longrightarrow
\Delta_{A\rightarrow B}(r)
}
\]

即：

> **源节点扰动强度—目标节点响应强度关系。**

---

# 五、临界扰动强度

本研究进一步引入响应阈值：

\[
\tau
\]

寻找最小扰动强度：

\[
\boxed{
r_{A\rightarrow B}^{*}
=
\inf
\left\{
r:
\Delta_{A\rightarrow B}(r)
\ge \tau
\right\}
}
\]

定义：

> **Critical Perturbation Intensity，临界扰动强度**

其含义为：

> 从节点 \(A\) 出发，需要多大的虚拟扰动，才能使节点 \(B\) 的未来输出变化达到预设响应标准。

遍历所有节点后，可以得到临界扰动矩阵：

\[
R=
\begin{bmatrix}
r^*_{1\rightarrow1}&\cdots&r^*_{1\rightarrow C}\\
\vdots&\ddots&\vdots\\
r^*_{C\rightarrow1}&\cdots&r^*_{C\rightarrow C}
\end{bmatrix}
\]

该矩阵可以作为后续连接强度定义的基础。

如何从 \(r^*\) 映射到最终“连接强度”，属于算法研究中需要进一步正式定义的问题，目前不预设具体结论。

---

# 六、完整研究流程

整体流程可表示为：

\[
\boxed{
\text{EEG}
\rightarrow
\text{Past/Future pairs}
\rightarrow
\text{Surrogate Brain}
\rightarrow
\text{Perturb Source}
\rightarrow
\text{Generate Future}
\rightarrow
\text{Measure Target Response}
\rightarrow
\text{Search Critical Intensity}
\rightarrow
\text{Connectivity Matrix}
}
\]

具体步骤如下：

1. EEG 数据预处理；
2. 构造 Past/Future EEG 样本；
3. 训练 EEG 未来生成模型；
4. 验证代理脑能否完成未来 EEG 生成；
5. 固定代理脑参数；
6. 选择源节点 \(A\)；
7. 对 \(A\) 施加强度 \(r\) 的虚拟扰动；
8. 分别生成 baseline future 与 perturbed future；
9. 测量目标节点 \(B\) 的响应变化；
10. 增大或优化扰动强度 \(r\)；
11. 找到首次达到响应阈值的 \(r^*\)；
12. 遍历所有 \(A\rightarrow B\)；
13. 得到节点间临界扰动矩阵；
14. 基于该矩阵定义最终连接强度。

---

# 七、子研究一：EEG 代理脑模型设计

## 研究问题

> 怎样设计一个模型，使其能够利用过去 EEG 生成未来 EEG，并具备作为虚拟扰动代理脑的基础能力？

整体映射为：

\[
\boxed{
X_{\mathrm{past}}
\xrightarrow{\ F_\theta\ }
\hat X_{\mathrm{future}}
}
\]

这一部分主要包括以下三个层次。

---

## 7.1 EEG 表征

研究：

\[
X_{\mathrm{past}}
\rightarrow
Z_{\mathrm{past}}
\]

即模型如何表示 EEG 中的：

- 时间信息；
- 节点信息；
- 跨节点交互信息。

目标是从历史 EEG 中形成可用于未来生成的有效表征。

---

## 7.2 未来 EEG 生成

研究：

\[
Z_{\mathrm{past}}
\rightarrow
\hat X_{\mathrm{future}}
\]

核心任务是：

> 根据历史 EEG 中包含的信息生成后续 EEG 时间段。

该部分属于模型设计的核心。

---

## 7.3 代理脑有效性

需要进一步区分两个概念：

### 未来预测 / 生成质量

即：

\[
\hat X_{\mathrm{future}}
\approx
X_{\mathrm{future}}
\]

### Surrogate Brain Validity

即：

> 该模型是否适合作为虚拟扰动实验的代理系统？

因此：

\[
\text{Future Prediction Quality}
\]

和

\[
\text{Surrogate Brain Validity}
\]

需要作为两个相关但不同的问题进行评价。

---

# 八、子研究二：虚拟扰动算法设计

第二部分主要解决连接推断算法的定义问题。

核心流程为：

\[
\boxed{
\text{Perturb}
\rightarrow
\text{Response}
\rightarrow
\text{Threshold}
\rightarrow
\text{Critical Intensity}
\rightarrow
\text{Connectivity}
}
\]

该部分至少需要明确定义以下四个核心要素。

---

## 8.1 扰动形式

定义扰动算子：

\[
\mathcal P(X,A,r)
\]

需要回答：

> 对 EEG 节点 \(A\) 应如何施加虚拟扰动？

需要明确的内容包括：

- 扰动是加性扰动还是其他形式；
- 扰动作用在模型输入还是中间表示；
- 扰动作用于单个时刻还是一段时间；
- 扰动具有正负方向还是仅考虑幅值；
- 扰动强度 \(r\) 如何定义。

参考文献中采用的是对指定节点信号进行选择性增加，其他节点保持不变。本研究需要针对 EEG 信号重新定义具体扰动协议。

---

## 8.2 输出变化度量

定义：

\[
D
(
\hat X_B^{(A,r)},
\hat X_B^{(0)}
)
\]

需要回答：

> 当目标节点 \(B\) 的未来 EEG 发生多大变化时，可以认为其对源节点扰动作出了响应？

由于未来 EEG 是一整个时间段：

\[
\hat X_B
=
[\hat x_{B,t+1},\ldots,\hat x_{B,t+T_f}]
\]

因此需要定义：

\[
\text{future waveform difference}
\rightarrow
\text{response magnitude}
\]

即将两段未来 EEG 的差异转换成一个可以比较的响应强度。

具体度量标准属于后续算法研究内容，目前不预设具体方案。

---

## 8.3 响应阈值

定义：

\[
\tau
\]

并建立：

\[
D_{A\rightarrow B}(r)
<
\tau
\Rightarrow
\text{未达到有效响应}
\]

\[
D_{A\rightarrow B}(r)
\ge \tau
\Rightarrow
\text{达到有效响应}
\]

这里的核心问题是：

> 如何设置响应阈值，使其能够区分模型本身的自然输出波动与由虚拟扰动真正引起的变化？

因此，阈值设计应作为独立算法模块，而不是简单指定一个固定常数。

---

## 8.4 临界强度搜索

目标是求：

\[
r^*
=
\inf
\{
r:
D(r)\ge\tau
\}
\]

即寻找：

\[
D(r^*)\approx\tau
\]

因此该问题本质上可以表示为：

\[
r
\rightarrow
D(r)
\]

上的临界点搜索问题。

具体采用：

- 逐步增加；
- 区间搜索；
- 数值优化；

或其他方法，需要在后续算法研究中确定。

---

# 九、正式子研究划分

## Study I：EEG Conditional Future Generation

### 研究问题

> 能否训练一个 EEG surrogate brain，根据历史 EEG 有效生成未来 EEG？

对应任务：

\[
X_{\mathrm{past}}
\rightarrow
X_{\mathrm{future}}
\]

主要研究内容包括：

- EEG 动态表征；
- 条件未来生成；
- 模型结构设计；
- 模型训练；
- 未来 EEG 预测 / 生成质量；
- surrogate brain 有效性。

---

## Study II：Critical Perturbation Connectivity Inference

### 研究问题

> 在训练好的 EEG surrogate brain 上，能否通过虚拟扰动产生的临界响应定义节点间连接？

核心对象：

\[
r_{A\rightarrow B}^{*}
\]

主要研究内容包括：

- 扰动形式；
- 响应度量；
- 响应阈值；
- 临界强度搜索；
- 临界强度到连接强度的映射。

两个子研究之间的逻辑关系为：

\[
\boxed{
\text{Study I：建立可信代理脑}
\Longrightarrow
\text{Study II：利用代理脑进行虚拟扰动}
}
\]

因此，“未来 EEG 生成”和“连接推断”应被视为两个前后衔接的研究问题，而不是一个单一任务。

---

# 十、关键术语说明

| 术语 | 本研究中的含义 |
|---|---|
| **EEG History / Past EEG** | 用于条件预测的历史 EEG 时间段 |
| **Future EEG** | 模型需要生成的随后 EEG 时间段 |
| **Conditional Generation** | 在 Past EEG 条件下生成 Future EEG |
| **Temporal Inpainting** | 将未来 EEG 看作待补全区域，根据已观测历史段完成条件生成 |
| **Surrogate Brain** | 学习 EEG 动态、用于替代真实脑进行虚拟扰动实验的模型 |
| **Source Node** | 接受虚拟扰动的 EEG 节点 \(A\) |
| **Target Node** | 测量扰动响应的节点 \(B\) |
| **Virtual Perturbation** | 在代理脑输入或内部表示中人为施加的可控变化 |
| **Perturbation Intensity \(r\)** | 虚拟扰动的大小 |
| **Baseline Response** | 无扰动条件下代理脑生成的未来 EEG |
| **Perturbed Response** | 加入扰动后代理脑生成的未来 EEG |
| **Response Metric \(D\)** | 衡量扰动与未扰动情况下未来 EEG 差异大小的指标 |
| **Response Threshold \(\tau\)** | 判断扰动是否产生有效响应的标准 |
| **Critical Intensity \(r^*\)** | 使目标节点响应首次达到阈值的最小扰动强度 |
| **Connectivity Strength** | 最终基于临界扰动信息定义出的 \(A\rightarrow B\) 连接量 |

---

# 十一、整个课题的核心逻辑

整个研究问题可以压缩为四个层次：

\[
\boxed{
\begin{array}{c}
\textbf{Q1：能否学习 EEG 的未来动态？}\\
\downarrow\\
X_{\mathrm{past}}\rightarrow X_{\mathrm{future}}
\\[6pt]
\textbf{Q2：该模型能否作为 EEG surrogate brain？}\\
\downarrow\\
F_{\theta^\*}
\\[6pt]
\textbf{Q3：扰动 A 后，B 的未来 EEG 是否发生变化？}\\
\downarrow\\
D_{A\rightarrow B}(r)
\\[6pt]
\textbf{Q4：需要多强的扰动才能使 B 达到预设响应？}\\
\downarrow\\
r_{A\rightarrow B}^{*}
\rightarrow
\text{Connectivity}
\end{array}
}
\]

---

# 十二、研究方案总结

本研究首先利用“过去 EEG → 未来 EEG”的条件生成任务学习一个 EEG 代理脑，使模型能够根据历史 EEG 生成后续 EEG 时间段。

代理脑训练完成后，固定模型参数，并对模型输入或中间表示中的源节点施加可控虚拟扰动。通过比较扰动条件与未扰动条件下其他节点未来 EEG 输出的变化，构建扰动强度与节点响应之间的关系。

随后逐步增加虚拟扰动强度，并寻找使目标节点响应首次达到预设阈值的最小扰动强度，将其定义为临界扰动强度：

\[
r_{A\rightarrow B}^{*}
=
\inf
\left\{
r:
\Delta_{A\rightarrow B}(r)\ge\tau
\right\}
\]

最后，对所有节点对重复上述过程，构建临界扰动矩阵，并以此作为后续定义节点间有向连接强度的基础。

整体研究由两个核心子研究构成：

1. **EEG Conditional Future Generation**：研究如何建立可靠的 EEG 代理脑；
2. **Critical Perturbation Connectivity Inference**：研究如何通过虚拟扰动、响应阈值和临界强度推断节点连接。

整个研究的核心思想可以概括为：

> **先学习 EEG 动态，再在代理脑中进行虚拟干预；通过“扰动多少才能使其他节点发生显著变化”这一临界响应问题，构建节点间连接度量。**

---

## 直接参考文献

Luo, Z., Peng, K., Liang, Z., et al.  
**Mapping effective connectivity by virtually perturbing a surrogate brain.**  
arXiv:2301.00148v3, 2024.
