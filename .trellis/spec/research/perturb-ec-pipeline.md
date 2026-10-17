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
