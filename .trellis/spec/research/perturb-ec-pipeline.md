# Study II 扰动→EC 推断流程规范（首版）

**适用范围**：路由为 `study2`（框架 Study II：Critical Perturbation Connectivity Inference）的任务——凡在冻结代理脑 θ\* 上做虚拟扰动、响应度量、临界强度 readout 或 EC 对比的实现，必须遵循本规范的可执行契约。

**首版来源**：任务 `10-17-perturb-ec-f0`（2026-10-17，v4a-fmt fold0 θ\*，Run 001–006）；变更按 evaluation-calibers 口径变更纪律执行，旧 Run 原口径不动。

## 1. CRN 确定性契约（D 要素基石）

- **签名**：`EulerSampler.sample(x_past(B,C,P), n_samples, generator=None, eps=(B,K,C,F), eta=(B,K,C,F))`（`models/gpfm-mamba-v1/sample.py`）——显式 eps/eta 时完全确定。
- **契约**：baseline 与全部扰动变体 (A,r) **并入同一 `sample()` 调用同批**（B=62×|R⁺|+1），逐窗一组 eps/eta 跨全变体共享（种子 = 固定 seed + 窗全局序号）。⇒ D(r=0)=0 **逐位恒等**；跨 Run 同窗逐位复现（已验证 bench≡main）。
- **为什么**：①跨批形 cuBLAS/mamba 核不保证逐位相等——恒等断言必须同批内；②跨 (A,r) 共享噪声使剂量-响应曲线与跨源比较无蒙特卡洛抖动；③「自然波动 vs 真实响应」的区分由契约结构性完成（D 确定性），τ 只承担材质性标尺。
- **错误用法**：逐变体单独调用 sample()（批形漂移+噪声不配对）；用 generator 而非显式 eps/eta。
- **测试锚点**：`models/perturb-ec-v1/tests/`（r=0 同批恒等 / 仅改源通道 / 同 eps-eta 两次调用恒等 / `sampler.gen_lowpass_hz is None`——EulerSampler 构造默认 40.0 + brickwall fs=500 是已登记陷阱，fmt 数据有效截止 ~20 Hz 静默削波）。

## 2. 扰动算子（P 要素，A 案正典）

- **签名**：`apply_gain(x_past, ch, r)`：x_A(t) ← (1+r)·x_A(t)，past 全段，r≥0 无量纲相对增益；`build_variant_batch(x_past)` → (1+62×|R⁺|, C, P)，index 0 = baseline。
- **选型依据**：谱域留在数据流形（带限数据不引入 DC/带外）、r=0 严格回基线、神经科学标准干预；参考文献「选择性增加」的带限加性版为文档级预留（启用须走口径变更+实现）。
- **幅度域警示**：r=1.28 ⇒ 源通道 z 域 ×2.28，超出训练分布——高剂量段（r≥0.32）属网络外推，AUC/r\* 判读随附 caveat。
- **传播事实（gpfm 路线）**：GPR 先验 `posterior_mean` 逐通道时间投影（空间核只作用于噪声 ε）⇒ CRN 下跨通道响应 **仅经速度网络单通路**（含 v4a rank-16 瓶颈）；先验只贡献自响应（实测占自响应 ≈40%，近 r 常数）。换代理脑结构时此分解须重核。

## 3. 响应度量与阈值（D/τ 要素）

- **D**：D_{A→B}(r;w) = RMS_{k,t}(δ) / s_{w,B}，δ=CRN 配对差，s=baseline 生成分布自身尺度；受试者级聚合 = 逐例 N_w 窗 **median**。
- **τ 双口径（v2，2026-10-17 用户拍板）**：
  - **操作性 = 相对自响应 τ̃**：D̃_{A→B}(r)=D̄_{A→B}(r)/D̄_{B→B}(r)，阈值 0.25（敏感 0.50）。判读纪律：D̃ 分母为自响应、靶通道自响应偏小会放大比值（实测 max ~6.6），**D̃ 数值不作跨靶比较**，仅用于 r\*/路径集判定。
  - **绝对口径 = 材质性观察照登**：null=同窗 baseline K 条两两配对距离（同构归一），逐被试分池 q90/95/99。实测（v4a-fmt θ\*）：q95≈1.82σ 而全边最大响应 1.14σ ⇒ **100% 删失**——绝对口径回答「响应是否达条件分布量级」，不作为路径判据。
- **已知行为（当前 θ\*）**：D̃ 近 r 无关（中位 ~0.77）⇒ r\* 近二元（94% 首档 0.01）；「临界强度」剂量语义在弥散传导的代理脑上弱承载，如实登记不强行解读。

## 4. readout 与比较接口

- **三件套**（沿 evaluation-calibers 预登记）：主=低剂量斜率（OLS 段 {0,0.01,0.02,0.04,0.08}，天然大=强）；次=AUC（全网格梯形/r_max）；降级=r\*（τ̃ 判定、右删失编码 R_NEXT=R_GRID 推导禁字面量、进比较器前翻转 strength=R_NEXT−r\*）。方向约定：**EC 矩阵行=源、列=靶**，offdiag 3782 边。
- **契约 npz**：`ec_pert_{readout}_s{sid}.npz` = {ec (62,62), readout∈{slope,auc,rstar}, subject_ids, orientation='larger_is_stronger'}——`models/baseline-ec-v1/compare.py::load_ec_pert` 直接消费。
- **主比较**：slope × {GC, gPDC} 逐被试距离偏相关 Spearman ρ（控制=通道对欧氏距离）→ 单样本 t × Holm 跨 2 法 + 1000×bootstrap（seed 2026）+ ρ̄/split-half-ceiling 比值。参考侧 = 真实数据 125 Hz 降采样重算（ridge λ_rel=1e-2、p=8 定标沿用；**子进程隔离**调 baseline-ec-v1——其模块级 `import config as C` 在进程内必被遮蔽击穿）。检验力注记随附；阴性=证据不足非证伪。

## 5. 隔离与装载惯例（防坑矩阵）

| 坑 | 契约 |
|---|---|
| 三变体目录各有 config.py | 扰动侧参数独立常量模块（不进 gpfm Config）；跨目录 config 一律 importlib 唯一名装载，**sys.modules 先注册后 exec**（@dataclass 按cls.__module__ 查 sys.modules，实测不注册即崩） |
| ckpt config_hash 校验 | cfg 对象必须来自被装载变体 config 实例；扰动侧任何字段不得进 gpfm Config |
| baseline-ec-v1 进程内 import | 一律子进程（fresh python、sys.path 只含 baseline 目录 + 防御剔除、npz 文件面交换），worker 断言 `config.__file__` 归属 |
| `fixate_split` 不存在则写入 | 调 `iterate_test_batches` 前断言 `split_manifest_override` 存在且等于登记路径 |
| 入口 BLAS | 全部入口/worker/tests 在 numpy 导入前 setdefault 五变量（`_blas_guard` 惯例） |

## 6. 首版实测基线（v4a-fmt θ\*，fold0 10 例，判读参照非门槛）

- Dose-response 100% 单调、低剂量 R²≈1.0（确定性信号特征）；slope 池 median 0.176（min 0.028/max 1.22）；τ̃0.25 删失 3.7%、路径集交集 3201/3782 边十例稳定；空间衰减平（ρ=−0.05，弥散）。
- **AC6 首验判定：未获结构收敛初步证据**（slope×GC ρ̄=−0.007 p=0.581；slope×gPDC ρ̄=+0.022 Holm p=0.112，bootstrap CI 不含 0 但预注册门为 t×Holm）；缺口四重证据定位 **θ\* 跨通道传导缺乏通道特异性**（非 readout/非参考侧，ceiling 0.96、实际可检 ρ̄≈0.028 而 power 足够）。
- **推论**：流程可复用；结构收敛的瓶颈在代理脑侧（跨通道特异性），为 Study I 迭代（损失侧跨通道监督等）提供直接判据。

## 7. NI-MSE-50-q0.90 路径首版实测基线（v5-fmt-01 θ\*，fold0 10 例，2026-10-20）

**来源**：任务 10-20-ni-mse-v5fmt-f0（`models/ni-mse-v1/`）；设计权威 = 归档任务目录 ni-mse-design.md **v1.5**（五要素钉定，引用以该文档为准）。本节只记可执行契约与教训。

### 7.1 与 §1 CRN 契约的关键差异（本路径的确定性语义）

- **null/响应侧刻意独立 fresh ε**（可交换性）：H₀「源 a 无模型内传导」时响应 R 与 null N 同分布的构造前提；恒等断言只在 R2（c=0、同 ε/η、**同批形**）意义上使用——不是全变体 CRN。
- **种子树四支 + 窗子集**：`SeedSequence(entropy=2026, spawn_key=(branch, sid[, w, a, k]))`，分支整数 ε_ref=10/ε_null=11/ξ=12/η=13/win-null=20/win-pert=21（**spawn_key 必须全 int**——字符串形式 numpy 拒绝）；ξ/ε/η 键**不含 c** ⇒ 跨 c 臂 CRN 配对（剂量-响应无蒙特卡洛抖动）。
- **η 必须显式**：`sample()` 的 η 缺省走全局 CUDA RNG（跨跑不可复现）——v1.5 §2 伪代码只传 eps 的写法不得照抄。
- **R2 断言组织**：c=0 变体与干净基线**并入同一 sample() 调用同批**（行 0/1 共享 eps/eta，`torch.equal` 逐位比对）；断言先于产物落盘、`r2_checked` 随产物登记 = per-例断点续跑依据（产物含完整标记即该例硬门已过）。
- **装载**：thostar 须**先预占 `sys.modules[config/model/graph]`（先注册后 exec）再 `import sample`**（v1 sample.py 导入时会把 v1 目录插回 sys.path[0]，顺序敏感）；EulerSampler 构造显式 `gen_lowpass_hz=None`（fmt 口径；默认 40.0 brickwall 静默削波——§5 已登记陷阱，本路径测试锚点含 `is None` 断言）。

### 7.2 实测基线（判读参照非门槛）

- split-half 逐例 0.914–2.000%（max_b ≤5.74%，R3 线 10% 未触发）；R1/R6 干净；R4 轻微（null 期 5 窗、perturb 期 s61 249 (a,窗) 项）。
- **G 门控 10 例全 0/3,782**（c≤0.5 全臂，Q95 带亦 0%）；S offdiag 中位 0.450±0.015；offdiag 剂量曲线 c∈[0.05,0.5] 全平（raw MSE 贡献 <1%）；自响应 S 对角 0.447→0.476（+7.3–7.5%）。
- refs（原库 GC fmt 48 例）相关：逐被试 Pearson direct 0.059±0.043 / Spearman 0.049±0.044；组级 0.097/0.075；**flip≈direct（差<0.001）方向信息≈零**；距离偏 Spearman 0.049（不降——微弱相关非几何承载）。

### 7.3 Common Mistakes（本路径实测教训）

1. **可交换 null 的灵敏度代价（设计层）**：R̄ 为 (w,k) 双中位统计量，门控 `R̄ > τ_b(Q90)` 在 H₀ 下通过概率≈0（中位数越过同分布池 Q90 几乎不可能）——「Q90 名义 10% 假阳性」只属**单样本 vs Q90** 的门。触发门需 Δμ² > null Q90−Q50（本次 ≈0.59 MSE，即 Δμ≳0.77 z 单位）。**引用 ni-mse-design.md §4.2 的「~10% 名义」预期线时须按此勘误口径**（设计文档 §4.2 与 §1.5-F 不相容，已在任务 10-20 experiment.md Run 003 登记）。
2. **四元组 RNG 契约的成本（实现层）**：逐 (sid,w,a,k) 构造 PCG64 流（每例 ~29.8 万流 + 16.1 亿高斯 CPU 物化）≈ 预算 ×1.36（GPU 空转等喂）。优化选项：**D 预取重叠**（生产者线程预生成下一源通道的噪声，流与值零变化，可直接用）；**A 向量化派生 / B torch GPU 生成 / C (源×c臂) 并批**（均改变噪声数值或批形 ⇒ 与既有 run 非逐位可比，**须走口径变更纪律**）。
3. **env_snapshot 时序**：必须在 thostar 装载**之后**捕获（thostar 按 cfg 设 `cudnn_deterministic`；装载前快照得到进程缺省 False 伪象——实测 manifest 记 False 而执行态 True，勘误随档）。
4. **chunk_bk=1024 内部分块**：sample() 的 Euler 前向按 1024 行分块 ⇒ 调用批再大不省前向时间；`B·K≤8192` 上限限的是 eps/η/x0 预物化显存（~780 MB 峰值），不是吞吐参数。

## 8. mild 线无偏 EC 噪声构造变体契约（hist-block，任务 10-27-ec-histblock-noise 2026-10-27 沉淀）

适用 `models/latent-flow-v1/ec_mild_unbiased.py` 系（global-τ 无偏口径）。可复用纪律：

1. **噪声变体开关 = 默认逐位等价 + 公式不变量对拍**：新噪声构造一律旗标化
   （`--noise-mode {iid,<variant>}` 默认 iid）；等价对拍锚点选 `sigma_matrix` /
   `raw_response_matrix` / `mse_curve(npz)` / `tau_global`，**不选 conn**（公式平移
   会污染锚点）；iid 分支 payload 键面不变（hist 键仅在变体分支注入）。
2. **共享/分组噪声的实现模式 = provider 注入**：子类只重写 `set_source` 的 ξ 来源
   （`HistBlockProbeSite(xi_provider=...)`），forecast/measure/CRN 全继承——扰动范式
   替换不改 engine/measure 路径（bisect_sigma「新 ProbeSite 不是新 engine」契约的
   线内体现）。全共享构造在被试循环外生成一次、各被试**切片同一存储**（强于种子
   共享：结构性一致，无形状漂移风险）。
3. **新随机流种子偏移须正交并登记**：hist-block ξ = `seed+9375`（=375×25），与
   `seed+source∈[42,169]`、`seed*10000+sid` 无碰撞；偏移值随 payload `hparams` 落档
   （`hist_xi_seed_offset`）。独立 `torch.Generator`（不受全局 RNG/调用顺序影响）。
4. **不落盘噪声数组时的跨任务复现四前提**（用户拍板不落盘 ⇒ 配方重建是唯一通道）：
   同 seed / 同 `(n_trials, n_groups)` 形状（CUDA 流按形状生成，`randn(20,G)[:k] ≠
   `randn(k,G)` 无保证）/ 同设备类型（CPU↔CUDA 流不同）/ 同 torch 版本（RNG 实现
   无跨版本契约）。四前提写入 payload `noise_model` 重建配方串，跨任务对拍前逐项核对。

## 9. symw/mild 线跨折扩展配方（任务 10-28-latent-flow-symw-f34 2026-10-28 沉淀）

新折（fold3/4 型）接入 `ec_mild_unbiased.py` 跑 EC 时的两道守卫扩展，缺一会被
assert 拒跑（这是设计行为——出处/归一化双保险，勿绕过）：

1. **symw ckpt 出处白名单**（`_caliber_asserts`）：按 f12/f34 先例为新任务 run_dir
   增 `elif "<task-dir-tag>" in a.ckpt` 分支，断言 `fold ∈ {新折集}` +
   `fold{k}_` 文件名前缀（防折-权重错配）；改动后必做负例回归（错折 × 错前缀
   双拦截实测），旧分支行为不变。
2. **norm SHA 注册表**（`EXPECTED_NORM_SHA_BY_FOLD`）：从训练侧
   `configs/stage{1,2}_config.json` 的 `norm_stats.mean_std_sha256` 登记（两 stage
   须逐字一致；登记值将在 EC 运行时被数据侧重算对拍——训练与 EC 归一化同源
   由此闭环）。
3. **包内 viz 折锚定坑**：`viz_ec_pert.py`（组均值锚点断言 +0.1934）与
   `viz_symw_ec_pairs.py`（fold0 被试表 + 旧版 dual-summary 结构）深度绑定
   fold0，非 fold0 复用前先验守卫；不可复用时走任务本地出图脚本（落本任务
   outputs，同款三图契约 + 图面 ρ 与 gccorr json 漂移对拍断言 1e-9——
   `outputs/10-28-symw-ec-f34/make_ec_viz_f34.py` 为模板）。

## 10. 关键节点排序（节点级 readout）契约（任务 10-30-critical-node-ranking 2026-10-30 沉淀）

EC_pert 的节点级对应：固定 σ 扰动源通道历史段 → I_c(σ)=Σ_{b≠c} MSE_b（排除自响应）→
全 128 通道降序排序 = 关键节点排序。实现 `models/latent-flow-v1/critical_node_ranking.py`
（--line {modma,mild}；装载链/守卫镜像双线 EC 主跑，ckpt 取 result/ 注册件 SHA 钉死对拍）。

可复用契约：

1. **CRN 配对**：同源 `set_source(c)` 一次生成 ξ、跨 σ 档复用（SignalProbeSite 原生）⇒
   多剂量档天然配对；σ=0 恒等断言（mse 全 0，须包 bf16 autocast 域）逐被试作免费检查点。
2. **σ² 缩放退化判读（防平凡稳健性）**：线性近似 I_c(σ)≈σ²·G_c(ξ)，σ² 因子在排序中约去 ⇒
   跨 σ 排序高一致部分是构造性必然。稳健性结论必须配**比例性诊断**：I(σ_hi)/I(σ_lo) 分布
   （对照 (σ_hi/σ_lo)²）+ 组平均 log-log 通道级斜率（线性⇒2）；斜率≈2+ρ≈1=平凡缩放稳定，
   斜率偏离 2 且 ρ 高=非线性一致（非平凡），ρ 低=实质重排。实测（双线 93 例）：比值 2.61/1.99
   ≪5.44、斜率 1.46/1.44 ≪2 而排序 ρ 0.87–0.98 ⇒ 非平凡稳健。
3. **聚合双口径**：组级排序主口径 = 原始影响向量均值（EC 组平均先例一致），副口径 = 逐被试
   within-subject 秩均值（防高幅度被试支配）——跑前钉死主从、两口径并列交付。
4. **top-K 交叠读数附机会水平** K/128（K=10/20 ⇒ 7.8%/15.6%），防「高交叠」过度主张。
5. **坑**：summary 侧跨被试堆叠统计一律 `np.concatenate(axis=0)`（列=σ 档）——`np.stack`
   后取 `[:, j]` 会静默取到「第 j 个被试整张矩阵」（本任务实测翻车，修复后与独立复算对拍）。
