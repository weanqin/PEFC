# Research: GP 条件先验流匹配（TSFlow）× Mamba 骨干 —— edra fold0 适配研究

- Query: 新路径 `gpfm-mamba`（1.0c 触发）适配研究：TSFlow GPR 条件先验 + Mamba 骨干（d_model=16）+ 复合损失（mae+cons+tv+corr），edra fold0 单 seed
- Scope: mixed（内部契约/规范 + arXiv 论文 + 官方代码 + PyPI/GitHub 元数据 + 本机只读数值诊断）
- Date: 2026-10-12
- 论文：arXiv:2410.03024（TSFlow，Kollovieh et al.）。**注**：官方 repo README 标注发表 venue 为 **ICLR 2025**（非任务背景所写 NeurIPS 2024），引用时以 ICLR 2025 为准。
- 代码：https://github.com/MarcelKollovieh/TSFlow （`tsflow/utils/gaussian_process.py`、`tsflow/model/tsflow_cond.py` 已逐行核验）

---

## ① 方法摘要与本项目适配结论

**TSFlow 条件模型做法**（论文 §3 + 代码核验）：
- 先验因子分解 `q0(x0|y^p) = δ(x0^p − y^p) · N(x0^f; μ_{f|p}, Σ_{f|p})`，其中 `μ_{f|p}=Σ_pf Σ_pp⁻¹ y^p`、`Σ_{f|p}=Σ_ff−Σ_pf Σ_pp⁻¹ Σ_fp`（GP 回归后验）。
- 核：SE `exp(−γd²)` / OU `exp(−γ|d|)` / Periodic `exp(−γ sin²d)`，γ=1/ℓ²，**论文/代码默认 γ=1**（对应论文 ℓ_SE=√(1/2)、ℓ_OU=1、ℓ_PE=√2）；时间轴归一化 `τ_i = i·π/p`（p=周期样本数）；白噪 `+iso·I`（`Q0Dist` 默认 iso=1e-4，**`tsflow_cond.py` 实际传 iso=1e-1**）+ Σ_pp 求逆前额外 `+1e-4·I` jitter；固定超参不拟合数据（论文 A.6 明示动机是数值稳定）。
- 路径 `x_t = t·x1+(1−t)·x0`（+σ_min=1e-4 噪声），速度目标 `u=x1−x0`（MSE，本任务不用）；t~U[0,1]；条件信息 y^p（+观测 mask）作网络额外输入；采样 Euler 16/32 步，起点 `x0+1e-4·ε`；EMA 0.9999、lr 1e-3、grad clip 0.5、batch 64。
- 骨干：DiffWave+S4（~176k 参数），多变量经特征维 transformer 混合；**GP 先验本身逐通道独立**（`rearrange(b l c -> (b c) l)` 后做一维 GP，无跨通道协方差）。
- 论文结论：GPR 先验全面优于各向同性先验、大幅降低 NFE 需求；**预测（conditional forecasting）场景三种核结果相近，OU 略优**（Table 3）；无条件生成场景 PE 占优（依赖数据周期性）；协方差矩阵一次性预计算、每样本只需一次矩阵向量乘（论文明确陈述 + 代码 `register_buffer` 于 `__init__` 完成，实测开销可忽略）。

**适配结论（总评）**：
1. GPR 条件先验移植到 edra 是**低风险、高置信度**的：past 全观测（mask 全 1）正是官方代码唯一支持的固定路径；预计算论断被官方代码直接证实；62 通道共享时间轴算子、通道独立 = 论文多变量口径原样照搬。
2. Mamba 替换 S4 属常规骨干替换，风险集中在**实现路线**（见 Q3）与 **d_model=16 的信息瓶颈**（62 通道压到 16 维，见②）。
3. 去速度损失的复合损失是**本任务最大方法学风险点**（见②风险 R1）——可行但必须带监控与预案。
4. 论文里 GP 先验收益最大的场景是强周期/季节数据；edra 静息 EEG 线性可预测性低（本机诊断：R²_ridgeAR≈0.12–0.23，GPR 均值 R²≈0–0.05），**GP 先验在此的价值预期是"尺度匹配+接缝连续的结构化基分布"而非预测增益**，预期收益小于论文的季节数据，需在 fold0 前置验证中如实陈述。

---

## ② 关键设计决策与风险

### Q1 GPR 核选择（待用户拍板）

**τ 网格口径（对齐论文）**：`τ_i = i·π/p`，p 取 50（500 Hz 下 α 节律 10 Hz 一个周期）。γ=1 ⇒ OU/SE 相关长度 ≈ p/π ≈ 15.9 样本 ≈ 32 ms；PE 周期恰为 50 样本（sin² 的 π 周期 ÷ π/50 步长）。

**本机只读数值诊断**（HC13/MDD10 各 302 窗、逐受试者 z-score 代理、float64；仅指示性）：GT future 粗度 |Δx|=0.12–0.18、std≈1.0；R²_persist≈−0.9、R²_ridgeAR=0.12–0.23（线性上限参考）。

| 核（γ=1, p=50） | R²(μ) | post_std | 先验样本粗度/GT | 条件数(Σ_pp, iso) | 评注 |
|---|---|---|---|---|---|
| OU | **0.01–0.03** | **0.97–1.02** | 1.5–2.2（iso=1e-4）/ 2.2–3.8（iso=1e-1） | 2.4e2（iso=1e-1），最稳 | 粗度略过但量级正确；数值最稳（近似三对角精度阵） |
| Matérn-3/2 | −0.03–0.02 | 0.93–1.00 | 0.5–0.7（iso=1e-4）/ ~3（iso=1e-1） | 3.6e2 | 粗度最贴近 GT；论文/代码外新增核（实现代价小） |
| SE | iso=1e-4 时 **数值崩溃（R²=−3.7~−12.9）**；iso=1e-1 时 ≈0 | 0.90–1.00 | 0.35–0.54 / ~3 | 需 iso≥1e-2 才稳 | 光滑样本（C∞）欠拟合 EEG 瞬态；必须抬高白噪 |
| Periodic | −0.13~−0.19 | **0.010（塌缩）** | 0.06–0.10 | 稳 | 严格周期 ⇒ 后验近确定性外推；EEG 频率漂移下失配 |
| SE×PE | −0.03~0.05（跨被试不稳） | 0.94–1.02 | 0.54–0.82 | 2.2e2 | 准周期核，最"EEG 形"，但收益在噪声范围内 |
| SE/M32 p=512 | R²=−16~−337 | — | — | 病态 | 整窗归一口径在本数据不可用，弃 |

**推荐起点：OU（γ=1, p=50, iso=1e-2）**。理由：① 论文预测场景结论"OU 略优"，口径最对齐；② 数值最稳健（γ=1 时精度阵近似三对角，iso=1e-2 下 cond~1e2–1e3，float32 buffer 也安全）；③ 后验展宽 ≈1.0 与 z-score 数据尺度匹配（基分布尺度正确 ⇒ 流的输运距离短）；④ 粗度比 GT 高 1.5–2.5× 属常规（各向同性先验是 ∞）。**变体 B：Matérn-3/2（同 γ/iso）**——粗度最贴近 GT、同样数值稳定，作为 `-v2` 备选。**不推荐**：纯 Periodic（后验塌缩+严格周期失配，α 周期先验会向无 α 被试/通道注入伪节律，污染 PSD 指标）；SE 仅在 iso≥1e-2 时可用且无优势；组合核（SE×PE）留作 PSD 检验显示节律缺失时的后续变体。iso 选择：1e-4（Q0Dist 默认，先验最干净但 SE 不可用）vs 1e-1（tsflow_cond 实际值，先验掺 ~0.32 std 白噪）——取 1e-2 折中（cond≈1e2–1e3、白噪贡献 std≈0.1）。

### Q2 GPR 预计算可行性（对照官方代码已验证）

- **预计算论断成立**：`gaussian_process.py` 在 `__init__` 内一次性构造 512×512 核矩阵并 `register_buffer` 存 `K_inv_x_K_star`（=Σ_pp⁻¹Σ_pf，384×128）与 `cov_reg`（=Σ_{f|p}，128×128）；训练时仅 `μ = y^p @ A`（一次矩阵向量乘）+ 从 MVN 采样。论文与代码双重确认"与数据无关、算一次"。
- **mask 依赖**：官方 `Q0Dist` 的观测位置由 `context_length` 硬编码（`t_obs = ones(context)`，连续全观测）——本任务正是该情形，算子固定；任意 mask 的填补场景走另一条 Langevin 代码路径，与本任务无关。
- **白噪进入哪个矩阵**：`K = kernel + iso·I` 加在**整窗**（past+future）对角上 ⇒ iso 同时进入 Σ_pp 与 Σ_ff 对角、不进 Σ_fp；随后仅对 Σ_pp 额外 `+1e-4·I` 再求逆。注意官方条件模型 iso=1e-1（非签名默认 1e-4）。
- **数值稳定性**：官方 float64 构造 → `torch.linalg.inv`（显式求逆，非 Cholesky solve）→ buffer 转 float32；采样用 `MVN(mean, cov_reg)`（torch 内部对 128×128 做 Cholesky，每批一次）。**本项目改进**：float64 中 `A = Σ_pf (Σ_pp)⁻¹`（用 `cholesky_solve` 而非 inv）、`chol(Σ_{f|p})` 一次算好，二者转 float32 作 buffer；训练时 `x0^f = μ + Lε`——严格更快更稳，且 PRD 验收标准（与 numpy 直接解一致、经验矩逼近解析值、PSD）可直接单测。Σ_{f|p} 的 PSD 数学上由 Schur 补保证，数值上靠 iso+jitter 兜底（诊断显示 OU/M32/PE 全部通过 Cholesky；SE 在 iso=1e-4 下即使 float64 也失效——选核时已规避）。
- **62 通道共享算子**：官方多变量即 `rearrange(b l c -> (b c) l)` 逐通道一维 GP、共享同一组矩阵，无跨通道协方差——照搬即与论文一致。合理性：62×128 维联合 GP 不可辨识也不必要；跨通道结构（含容积传导零滞后相关）全部交给网络学。局限：先验不含跨通道信息 ⇒ GC 一致性完全依赖速度场学到跨通道时序结构，是 Q2 代理脑有效性的直接风险载体（监控指标 1 及早暴露）。
- **口径偏离登记**：论文在 GPR 前做周期段均值中心化（`loc_freq`），PE 核下置零；本项目 z-score 已在 dataloader 完成且 EEG 无稳定周期 ⇒ **跳过 loc_freq**（SE/OU 下它只是去均值偏置，影响小），在 design.md 登记为有意偏离。

### Q3 Mamba 实现路线（待用户拍板）

本机事实（本次实测）：torch 2.8.0+cu128、`_GLIBCXX_USE_CXX11_ABI=True`、cp312、triton 3.4.0（torch 自带）、**einops 未装**、scipy 未装；`/usr/local/cuda-12.8/bin/nvcc` V12.8.93 存在但不在 PATH（注意：`.trellis/research/environment.md` 写"系统无独立 nvcc toolkit"已过时，需主 agent 回填）。

| 路线 | 事实核查结论 | 风险 |
|---|---|---|
| (a) 官方 mamba-ssm 预编译 wheel | **精确匹配 wheel 存在**：`mamba_ssm-2.3.2.post1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl`（GitHub release v2.3.2.post1；PyPI 本身全系列无 wheel 只有 sdist）。causal-conv1d `1.7.0+cu12torch2.8cxx11abiTRUE-cp312` 同样存在，且对 mamba_simple 是可选依赖（try/except 回退 `F.conv1d`） | ① 2.3.2.post1 元数据强依赖 `tilelang==0.1.8 / apache-tvm-ffi / quack-kernels>=0.3.4 / triton>=3.5.0`（本机 triton 3.4.0 会被迫升级，重编译栈体积大且可能干扰 torch）；② `mamba_ssm/__init__.py` 顶层 import Mamba3 → `mamba3.py` 顶层（无 try/except）import triton 版 mamba3 kernel ⇒ `--no-deps` 安装后 `import mamba_ssm` 仍可能直接 ImportError；③ 2.2.x 旧版依赖干净（仅 einops）但 GitHub wheel 只到 torch2.7，跨 torch 小版本用 CUDA 扩展易未定义符号。**源码编译**可行（nvcc 12.8 + `PATH/CUDA_HOME` 导出 + `TORCH_CUDA_ARCH_LIST=8.9` + 需装 ninja/einops，预计 5–20 min 编译），但需安装授权 |
| (b) 纯 PyTorch 自实现 | selective scan 顺序循环（逐步 `h=a·h+b·x`）实现量 ~200–300 行，精确可微、无依赖；d_model=16（d_inner=32）、d_state=16、L≤512 下计算量极小，瓶颈仅 kernel launch：估 4 层 × 512 步 × fwd/bwd ≈ 数万次小 kernel ≈ 0.1–0.3 s/iter（B=64–256），fold0 全程 ~2–8 h，可接受。log-space 并行 scan（a=exp(ΔA)>0 可精确 cumsum 化）需分块防 O(L²) 显存（B64·D32·N16·L² ≈ 34 GB → 64 点分块后 ~0.5 GB）与下溢，作为二期优化 | 吞吐上限低（~5–10× 慢于 CUDA kernel）；需自写单测（恰好是 PRD 验收项：形状/因果性/梯度有限） |
| (c) 第三方纯 torch 包（mambapy 1.2.0） | PyPI 通用 wheel（py3-none-any），**唯一依赖 torch**，MIT，Mamba-1 实现（2024-07 后未更新；内部 scan 为朴素实现，未逐行核验） | 仍需 pip 安装授权（违背"零新增依赖"默认）；代码冻结两年；单测可控性不如自实现 |

**推荐 (b) 纯 PyTorch 自实现**：与 R8"默认零新增依赖"及 PRD Mamba 单测验收天然契合，d_model=16 规模下性能损失无关紧要，完全可控可审。加速升级路径（若日后需要）：优先评估 (a) 的 causal-conv1d 1.7.0 精确匹配 wheel（依赖干净）+ mamba-ssm 源码编译，避开 2.3.2.post1 的 tilelang 依赖链；mambapy 作为"想用现成代码"时的折中。

### Q4 损失函数风险（口径评估）

口径本身可行，但存在结构性风险，逐项：

- **R1（top 风险）t→1 梯度消失 + 高 t 段欠约束**：`x̂1 = x_t+(1−t)v_θ` ⇒ ∂x̂1/∂v_θ=(1−t)→0；且 t→1 时 x_t→x1，x̂1 误差 =(1−t)(v_θ−v*) 自动变小——高 t 区 loss 天然"宽松"，v 在该区基本不受约束，而 **ODE 采样积分到 t=1 时仍使用 v**（终点增量 = ∫v dt，v 不乘 (1−t)）。缓解：① 训练保持 t~U[0,1]，但**监控分箱 loss-vs-t 与采样终点误差**；② 预案 A：x̂1 中以 `(1−t+ε)`（ε=0.01–0.05）设下限（仅训练侧，采样不变，微小偏置）；③ 预案 B：t 改 logit-normal/Beta 集中中段；④ 预案 C（需用户另行同意）：恢复小权重速度项。
- **R2 回归均值/多样性塌缩**：mae+corr 都是条件中心趋向，多样本采样方差易系统性偏低；cons（std 匹配）正是对冲项——保持 w_cons=1 并**监控多样本展宽 vs GT 展宽**。
- **R3 tv 纯平滑压幅度**：L1 一阶差分无条件惩罚 ⇒ 偏好低幅低频输出，z-score 下 0.1 权重可能仍过度平滑（高斯噪声先验情形更甚）；**监控 PSD 高频段与幅度直方图**；预案：降 0.01–0.05 或改为对照 GT 差分（口径变更须登记）。
- **R4 corr 数值稳定**：T=128 逐通道 Pearson，方差→0 即 NaN。实现：方差 eps=1e-6–1e-4 进 sqrt、r 截断 [−1,1]、`torch.where(var<eps, 0, ...)` 屏蔽近平坦通道；另注意 H0 下 E[1−r]≈1 的损失地板（无害常数，日志中体现）。
- **R5 cons 量纲**：沿时间窗内 std/mean，z-score 下 O(1)，量纲一致无需再归一化；mean 项信号弱（窗均值 SE≈0.09）属预期。两项同样加 eps。
- **与 GPR 先验的相互作用**：x0^f 与 y^p 经 μ 相关 ⇒ 输运距离缩短（论文动机），损失集中在远离接缝处；past 段 x_t≡y^p 恒定（t·y^p+(1−t)·y^p）、速度目标 past 部分为 0 ⇒ **四项损失只作用于 future 128 列**，模型对 past 段输出可在设计上屏蔽或忽略（design.md 定）。
- 通用：逐项 loss 值 + 梯度范数落日志、全项 NaN guard、grad clip 0.5（沿用论文）。

### Q5 评估实现（无 scipy，numpy-only）

- **GC（指标 1）**：VAR(p) OLS——堆叠测试窗（past+future 512 点/窗，z-score 已完成），设计阵 X∈R^{N×(62p+1)}，BIC 在真实组上选 p∈{1..10} 后两组共用（口径一致，experiment.md 登记）；G=Xᵀ X 一次算出（p=10 时 621²，N≈1.3 万行，BLAS 秒级）。逐方程 OLS 得残差协方差 Σ(62×62)；**条件 GC 的成对 F 统计量**用"从 G 的块结构做列删除 Schur 补/小规模重解"避免 3782 次全量重拟合（每次 ~0.5–0.6 k 阶 solve，总计 ~1–2 min 级）；F 的 p 值用 incomplete beta（Lentz 连分式，~30 行 numpy）——或按规范本质需求改为**矩阵向量化后的 Pearson/Spearman + 置换/自助 CI**（对窗/被试重采样，纯 numpy），显著性路径更简洁。多重比较：BH-FDR（numpy 排序实现）附 raw p。对角线剔除、有向边 62×61=3782，口径显式登记。
- **PSD（指标 3）**：Welch——Hann（`np.hanning`）+ `np.fft.rfft` 分段周期图平均；**两组用完全相同的分段**（建议 512 点窗 past+future，分辨率 0.98 Hz；若只看 future 128 点则为 3.91 Hz）；85000×62 规模秒级完成；log-log 叠加对比图 + 可选 62 通道 topo（`chan_coords_62.npy` 在数据集根目录）。
- **可视化**：全部经统一 visualizer（nature-figure 契约），图落本变体目录、路径登记 Run 块；包含多样本轨迹图（真实 vs 生成，指标 2 人工检验素材）。
- **可行性结论**：85000×62 上 GC+PSD 全流程 CPU 分钟级，128 核内存 503 GB 充裕，无需 scipy。

### ⑥ 之外的横切风险

- **R6 d_model=16 信息瓶颈**：62 通道经 62→16 线性投影压入 Mamba（时间轴序列、每步 62→16→…→62），16 维隐态携带 62 通道动力学对 GC 类跨通道时序结构可能不足——GC 一致性（指标 1）是首要观测点；缓解预案（变体级，不动 d_model）：Mamba 块间插入 62×62 通道混合线性层、或加深层数（4→6–8）。
- **R7 前置验证通过标准占位**：指标 1/3 通过标准待用户拍板（规范明示），本任务只落数值与图，结论由用户判。

---

## ③ 可执行方案

### 模块划分（代码区 `models/gpfm-mamba-v1/`，一变体一目录，含独立 config）

| 文件 | 职责 | 关键点 |
|---|---|---|
| `config.yaml` | 本变体唯一配置 | 核=OU/γ=1/p=50/iso=1e-2；σ_min=1e-4；损失权重 {1,1,0.1,0.1}；Euler 16 步；seed；路径 |
| `data.py` | edra fold0 dataloader | 受试者级 5-fold 分层、fold0 固定；z-score 仅训练折；窗 384/128；筛选口径（qc_pass）登记；stride 待 design.md 定（~64 → 每被试 ~1320 窗） |
| `prior.py` | GPR 条件先验 | float64 预计算 A(128×384)、L=chol(Σ_{f|p})(128×128) → float32 buffer；`x0=concat([y^p, μ+Lε])`；单测：与 numpy 直解一致、经验矩、PSD/chol 成功 |
| `model.py` | Mamba 骨干（纯 torch） | 每时间步 62→16 投影、n_layers≈4（expand2, d_state=16, d_conv=4）、正弦时间嵌入+MLP、速度头 16→62；scan 顺序循环实现；单测：形状/因果性/有限梯度 |
| `loss.py` | 复合损失 | 仅 future 列；mae/cons/tv/corr 各带 eps 与 NaN guard；逐项日志 |
| `train.py` | 训练 | t~U[0,1]；Adam lr1e-3、clip0.5、EMA0.9999（论文值起步）；ckpt best/last 命名 `变体+用途+标识`；分箱 loss-vs-t 监控 |
| `sample.py` | Euler ODE 采样 | x0=GPR 先验+1e-4ε；16 步默认；多条独立样本 |
| `eval.py` | GC + PSD（numpy-only） | Q5 口径；FDR/betainc 或置换 CI |
| `visualizer.py` | 统一产图 | nature-figure 契约；轨迹/PSD/热图类型 |

### 训练与评估流程

1. 单测（PRD 验收：dataloader 泄漏/分层、GPR 算子、Mamba 因果性、四项损失手算对齐）→ 2. fold0 单 seed 训练（≈36–37 被试 ~4–5 万窗、B=64–256、~100–200 epoch，纯 torch scan 预计 2–8 h，nohup 后台化）→ 3. best ckpt 采样（测试集窗、多样本）→ 4. 前置验证三指标落盘（GC 数值+图、PSD 对比图、人工检验样本图）→ 5. Run 块登记 experiment.md（seed/口径/commit hash）→ 6. **停止，5-fold 须用户同意**。

### 路径清单（path-isolation 合规）

- 新建代码：`/root/npi/models/gpfm-mamba-v1/`（仓库现无 models/，本变体为首建）
- 输出：`/root/autodl-tmp/outputs/gpfm-mamba-f0/{checkpoints,logs,samples,eval,figs}/`（**outputs 根落数据盘待用户确认**，PRD 待拍板 #3）
- 只读依赖：`/root/autodl-tmp/edra_process_v4/`（含 manifest、chan_coords_62.npy）；共享规范文件
- 不安装任何依赖（推荐路线 b）；若用户改选 (a)/(c)，安装清单与授权单独立项

---

## ④ 依据

- 论文（在线全文）：Kollovieh et al., *Flow Matching with Gaussian Process Priors for Probabilistic Time Series Forecasting*, arXiv:2410.03024, https://arxiv.org/html/2410.03024 （ICLR 2025；核定义/ℓ 值/τ·π/p 归一化/白噪动机 App. A.6、"矩阵只算一次"陈述、预测场景 OU 略优 Table 3、GP 不拟合数据的声明）
- 官方代码：https://github.com/MarcelKollovieh/TSFlow
  - `tsflow/utils/gaussian_process.py` — 核实现（γ 参数化、默认 1.0）、τ=arange·π/freq、iso 白噪+1e-4 jitter、float64→float32 buffer、`linalg.inv`、`__init__` 预计算
  - `tsflow/model/tsflow_cond.py` — iso=1e-1、`(b c)` 逐通道 GP、x0=concat([context, GP future])、t~U[0,1]、Euler 16 步、采样起点 +sigmax·ε、EMA/guidance 默认值
- Mamba wheel 事实：https://github.com/state-spaces/mamba/releases/tag/v2.3.2.post1 （`cu12torch2.8cxx11abiTRUE-cp312` wheel 存在；`__init__.py`/`modules/mamba_simple.py`/`modules/mamba3.py` 的 import 链已核验）；https://github.com/Dao-AILab/causal-conv1d/releases/tag/v1.7.0 （同组合 wheel 存在）；PyPI 元数据：mamba-ssm 2.3.2.post1（仅 sdist；依赖 tilelang/tvm/quack/triton≥3.5）、causal-conv1d 1.7.0（仅 sdist）、mambapy 1.2.0（通用 wheel，仅依赖 torch，https://github.com/alxndrTL/mamba.py）
- 内部契约：`.trellis/spec/data/edra.md`、`.trellis/spec/research/{data-pipeline,surrogate-brain-qualification,path-isolation,visualization}.md`、`.trellis/research/environment.md`
- 本机只读诊断：2026-10-12，numpy 2.3.2，HC/13+MDD/10 各 302 窗（数据只读、无写入、无安装）；torch ABI/einops/triton/nvcc 存在性实测

## Caveats / Not Found

- 核诊断仅 2 被试、逐被试者 z-score 代理（正式管道为训练折统计），数值为指示性；正式 fold0 训练前建议在 fold0 训练折上复跑一次（~20 s）作为 design.md 附件。
- mamba-ssm 2.3.2.post1 在 `--no-deps` 下 `import mamba_ssm` 是否成功未实测（禁止安装）；已核验 import 链证据指向高风险。
- mambapy 内部 scan 实现未逐行核验；纯 torch 吞吐为 launch-overhead 推算值，非实测。
- 论文附录 B.2 消融以 OU 为主，SE/PE 为全量结果对比，无逐数据集核消融表——本文引用口径已按此收敛。
- **环境契约勘误待回填**：environment.md "系统无独立 nvcc toolkit" 与实测 `/usr/local/cuda-12.8/bin/nvcc V12.8.93` 不符（本 agent 无权修改该文件）。
