# 实验日志 — 生成器 past 依赖审计（v1-fmt θ\*·fold0）

> 每次「跑批/训练/评估/统计」追加一个 Run 块：跑前先写「目的+配置」，跑后同会话补「结果+结论」，不留空块。

## Run 001 — E1–E4 四件套主 Run（2026-10-18）

### 目的与假设

- 问题：future 生成对 past 的依赖是 H1 结构性短记忆（OU 先验主导、解析半衰预测 ≈5.5 样本/22 ms）、H2 敏感性≠信息依赖、还是 H3 几乎不依赖？用解析对拍 + 位置扫描 + 置换检验 + 边界条件四证据裁决。
- 源头：ec-nmtime-f0 波形图人工检验（r=2 时 future 响应峰 @t=0、半衰 12–24 ms、后 64 样本仅占 14–24%）。

### 配置

- 代码：`models/pastdep-audit-v1/`；θ\* 只读同 v1-fmt pin（hash 0dc0d5249c68 + gen_lowpass=None 断言）。
- 窗集/CRN：同 perturb-search-f0（fold0 test 前 3 例 {10,11,12} × 前 8 窗 = 24 窗，win_idx 0..23；K=16；SEED=2026 逐窗 manual_seed(SEED+win_idx)；E3 各臂独立 seed = SEED+win_idx+arm·1000）。
- E1b/E2/E4 批：行 0 baseline + 62 源 × (3 剂量 {0.5,1.0,2.0} 全 past N + 4 子窗掩码 N@1.0) = **435 行/窗**，一次 sample()/窗；存全 128 点 δ̄ 与 |δ̄|（62 靶）。
- E2 口径（钉死）：boxcar 子窗 {[0,96),[96,192),[192,288),[288,384)}；注入 RMS 子窗内重归一到 r·σ_A；σ_A = 全 past std。
- E3 四臂：①噪声地板（同 past 换 seed）②同人置换（w+1 例内循环）③跨人置换（下一受试者同例内窗）④消融（past≡0）；逐 (窗,通道) 128 维能量距离，ρ = d(臂)/d(地板)，窗中位+IQR。
- 判据（预注册）：T50 对拍 |sample/prior −1|<20% ⇒ H1；ρ<1.2 无信息依赖 / ≥2 强依赖；ρ₂≈1 ⇒ H3、ρ₂>1.2 ⇒ H2。
- 环境：base @ 本地（RTX 4080S、BLAS 护栏）；命令 `cd models/pastdep-audit-v1 && nohup python3 run.py smoke|main > ../../outputs/pastdep-audit-f0/logs/….log 2>&1 &`。
- 产物：`outputs/pastdep-audit-f0/run_01/`（A_prior.npz / curves.npz / e3.npz / summary.md / figs/）。

### 结果

- **执行史**：主 Run 2 次调用达成（Run#1 崩于三处小 bug：E3 映射用截断窗表 / sample() 位置参数误传 generator / 单窗返回带 batch 维——均修复后 smoke 过、Run#2 干净完成）。wall 1.3 min（summary.md 为准；trellis-check 纠正）（24 窗 × 批 435 行 + 每窗 5 次单窗采样）；B0 六项单测全绿。
- **E1 双通路对拍**：解析 env 峰@f=0，**T50_prior = 6.0 样本（24 ms）**（OU 零参数预测 5.52，吻合）；采样自响应 T50 = **6.0（全剂量 {0.5,1,2} 一致，对拍偏差 0.0% < 判据 ±20%）**；跨响应 T50 = 9.5–10.0（略长）；φ（前 32 样本占比）= 0.73–0.79。
- **E2 注入位置**：解析子窗质量 **[0, 0, 0, 1]**（先验通路只看最后 96 样本）；采样靶中位响应 [0.0105, 0.013, 0.0171, **0.0756**]——早/中子窗仍有晚段的 14–23% 响应，而该处先验贡献解析为 0 ⇒ **长程利用全部来自速度网络通路**。
- **E3 置换/消融**：噪声地板 d₀ 中位 0.0217；**ρ(same) = 1.96、ρ(cross) = 4.64、ρ(ablate) = 6.92**（中位排序 ablate > cross > same；IQR 14–34 窗间异质大，照登：逐窗比值对地板敏感，中位排序稳健）。
- 图：`figs/pastdep_audit.{pdf,png,…}` 三面板（E1 对拍半对数 / E2 成对棒 / E3 ρ 棒+IQR）；QA 链全过（碰撞 0 FAIL、visual-judge pass）。

### 结论

- **三假说裁决：H1 支持（敏感性面）+ H2 支持（信息面）+ H3 不支持**。
- **「future 几乎不依赖 past」不成立**。依赖为双层结构：①**边界敏感性强但短程**——OU 先验通路主导，半衰 6 样本/24 ms（解析与采样逐位对拍），用户波形图观察即此层；②**速度网络通路携带长程与个体级信息**——早段 past 注入仍有 ~17% 晚段响应（该处先验贡献为 0），past 置换/跨人置换/消融使 future 分布位移 2.0/4.6/6.9× 噪声地板（cross > same ⇒ 个体级统计被使用）。
- 对 EC 推断的含义：扰动-响应读出主要测到短记忆先验通路（时间局部、自跨比特异来源），而携带长程/个体信息的通路对单通道加性扰动弱敏感（弥散）——**结构性地解释了 N×M-time EC「稳定但 Gate A 弱」的组合**，并把改造入口指向先验核/条件化结构（Study I 侧），而非继续换度量/范式。
- 口径边界照登：E3 逐窗 ρ 的 IQR 大（中位排序为判读基准）；boxcar 掩码边缘瞬态；消融臂 past≡0 属分布外段。


## Run 002 — μ/前缀解耦交叉注入（2026-10-18）

### 目的与假设

- 裁决 H\*：(前缀 B, μ_A) 是否 ≈ (前缀 A, μ_A)——past 对 future 的贡献是否仅经 OU 先验 μ。

### 配置

- 代码：`models/pastdep-audit-v1/decouple.py`（θ\* 与 CRN 契约同 Run 001；`sample_decoupled` 逐位复刻主循环、μ 注入、前缀独立传入；R 臂与原 sampler.sample 逐位相等断言）。
- 窗集：同 Run 001（24 窗）；B_same/B_cross 映射同 E3；M_rand 通道置换固定 seed=99；seed 偏置：R1/R2/XS/XC/MB/MR/Z/配对 = +5000/5050/5200/5300/5400/5500/5600/5700。
- 六臂与判据：见 design §5（ρ 带 1.2/2.0；Wilcoxon+Holm；bootstrap CI；配对 |δ̄| 谱与 t50）。
- 命令：`cd models/pastdep-audit-v1 && nohup python3 decouple.py > ../../outputs/pastdep-audit-f0/logs/decouple.log 2>&1 &`；产物 `run_01/decouple.npz + decouple_summary.md`。

### 结果

- 执行 2 次调用（Run#1 统计段双重 median 笔误，修后 Run#2 干净完成，采样确定）；wall 0.3 min；**解耦实现与原 sampler.sample 逐位相等断言 24/24 过**（μ 注入复刻的有效性证明）。
- **口径修正（如实登记）**：预注册 ρ=d(臂)/d(地板) 失效——地板能量距离无偏估计逐格围绕 0（均值 −0.032、正占比 40%），中位为负致比值符号翻转；改报绝对量级 + 配对面分解（decouple_summary.md 修正版）。
- 六臂结果：X_same=(B,μ_A) 能量距离 1.02（88% 格正，Wilcoxon p=1.2e-7，Holm 6.0e-7）｜配对 |δ̄| 0.133；X_cross 1.14｜0.194；M_B=(B,μ_B) 1.36｜0.142；M_rand=(A,μ_rand) 1.66｜0.060；Z=(0,μ_A) 2.83｜0.225；地板 ≈0（−0.032±0.267）。

### 结论

- **H\*（A 的 future 仅依赖 OU 先验 μ、不直接依赖 past 段）被否定**：(B, μ_A) 与 (A, μ_A) 分布远非不可区分（位移 ~1.0 ≫ 地板 0，全臂 Holm p ≤ 6e-7）。
- **定量分解（均值面）**：换前缀保 μ = 完整换 past 的 **93%**；前缀已换后 μ 的边际效应仅 **7%**——past→future 的均值轨迹影响主要经速度网络前缀直接通路；μ 的独立作用主要在分布形状层（μ 置换臂分布面 1.66 ≥ M_B 但均值面仅 0.060——μ 塑造 x₀ 空间协方差而非均值轨迹）。
- 前缀置零（保 μ）位移 = 完整换 past 的 1.6×（分布外更强扰动）。
- 与 Run 001 闭环的双层图景最终版：**局部高频敏感短记忆（24 ms）经 OU 先验；全局身份/长程依赖（~93% 均值面）经速度网络前缀**。对 EC 推断：换 past 段不换 μ 绝非无害操作；扰动-响应式 EC 读出的「自跨比」信号主体来自前缀通路的跨通道弥散性质。

- **补充图（2026-10-18 用户请求）**：`figs/pastdep_waveforms_3curves.{pdf,png,…}`——future 三曲线（Real 数据真值 / Generated K=16 均值±SD / OU 先验 μ=A·x_past），win0 sid=10，ch36+ch2（与 ec-nmtime 波形图同对象）；QA 链全过（碰撞 PASS、visual-judge pass）；数据快照 `figs/wave3_win0_ch36_ch2.npz`。读数：μ 的 RMS 前 32 样本 ≫ 后 64（短记忆的直观呈现），gen 均值与 real 的相关显著高于 μ 与 real。

- **补充图 2（2026-10-18 用户请求 + 口径更正）**：`figs/pastdep_inference_window512.{pdf,png,…}`——**512-token 推理窗**视图（用户更正：模型每步 Euler 前向消费 384 数据前缀 + 128 流变量位，128 由 x₀=μ+Lε 初始化后 16 步 Euler 计算；real 不参与推理）：past 灰 + future 计算位演化（x₀ 点线 / 第 4/8/12 步淡线 / 第 16 步 gen 实线±SD）+ real 黑对照；win0 ch36/ch2；384 前缀逐位见证断言。读数：gen 均值→real 的 RMSE 在 Euler 步间近平稳（ch36 0.77→0.75→0.75→0.76→0.80）——**128 位的均值轨迹基本由 x₀（μ 主导）定形，Euler 流主要精修涨落/分布细节**，与 Run 001/002 双通路图景一致。QA：碰撞/文字审计 PASS、visual-judge pass；快照 `figs/inference_window512_win0.npz`。
