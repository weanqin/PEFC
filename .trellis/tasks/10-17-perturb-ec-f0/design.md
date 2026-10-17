# design.md — fold0 虚拟扰动→EC 有效路径推断流程（v4a-fmt 冻结代理脑）

> 口径总源：prd.md + 本文件 + evaluation-calibers.md「Study II 扰动 EC 对比口径」（2026-10-13 预登记）。方向约定全仓库统一：**EC 矩阵行 = 源 source，列 = 靶 target**；有向边展平非三角（3782）。
> 本版已回填 trellis-review 双 mandate 发现（A-M1 先验通路事实更正、B-Major1/2 实现阻断项及全部 Minor）。

## 0. 范围与边界

- **θ\*（只读冻结）**：`/root/npi/outputs/gpfm-mamba-fmt-f0/v4afmt-main-01/checkpoints/gpfm-mamba-v4a-fmt-best.pt`（best ep45/0-based 44，config_hash 05d9bb875a2c，EMA 权重，`load_eval_model` 严格校验）。前置验证级（prd Constraints），一切结论标注「流程验证·单模型·单折」。
- **框架定位**：本任务 = 框架 14 步流程的第 6–13 步首次实例化（第 5 步固定 θ\* 由引用 ckpt 完成）；第 14 步（r\*→最终连接强度映射）只交付候选 readout 三件套，不拍板唯一定义。
- **口径隔离**：本任务 GC/gPDC 参考在 **125 Hz 真实数据**侧计算（fmt 同源降采样），与 fmt 评估侧被阻塞的「250 Hz genGC vs real GC 一致性」为不同对象；experiment.md 显式登记该区分，产物中不出现 genGC 一致性内容。

## 1. 模块布局（path-isolation 薄变体）

```
models/perturb-ec-v1/
├── config.py          # 扰动侧全部口径常量（ckpt 路径/r 网格/K/N_w/τ 分位/seeds）——不碰 gpfm Config
├── loader.py          # importlib 装载 v4a-fmt config（唯一名）+ load_eval_model + test 窗子采样
├── perturb.py         # P(X,A,r) 扰动算子（A 案增益——用户 2026-10-17 拍板；B 案文档级预留不实现）
├── response.py        # CRN 批采样：baseline / perturbed / null 配对 → 窗级 D 曲线落盘
├── readout.py         # slope/AUC/r* 三 readout + 受试者级 EC 矩阵 + 有效路径集 + load_ec_pert 契约 npz
├── refs_fmt.py        # fmt→125 Hz 重采样+z-score → 子进程隔离调 baseline-ec-v1 refs_for_subject
├── compare_wrap.py    # 子进程隔离调 baseline-ec-v1 compare.py（npz 文件面交换；无进程内调用路径）
├── run.py             # CLI：--stage {smoke,bench,main,readout,refs,compare,figs} --subjects N
└── tests/             # 独立脚本单测（①②③断言惯例，rng 2026）
```

- **只读依赖（不改写）**：`models/gpfm-mamba-v1/`（sample/data/model/gpr_prior——经 sys.path）、`models/gpfm-mamba-v4a-fmt/`（config 与 splits manifest——经 importlib）、`models/baseline-ec-v1/`（ec_refs/compare——**一律子进程隔离**：fresh python 进程，sys.path 只含 baseline 目录，npz 文件面交换；进程内 import 不可行——ec_refs/compare 模块级 `import config as C` 并在 def 期求值 `C.BOOTSTRAP_N`，进程内必被遮蔽击穿）、`/root/autodl-tmp/edra_processed_fmt/`。
- **`common/visualizer.py` 纯增量扩展**（visualization.md 正典路线：新增诊断图类型，不动既有类型签名与行为）——这是对共享文件的**已知写入点**，design 显式声明，check 阶段路径扫描以此为准（非违例）。
- **模块名纪律（消遮蔽）**：本目录 config 一律经唯一名装载（如 `perturb_config`），**不占据 `config` 模块名**（本任务无 v4a-fmt 式「给 train.py 喂 config」的消费者，遮蔽无功能收益）；v4a-fmt config 经 importlib 唯一名 `v4afmt_config` 装载。**装载配方（@dataclass 坑，注册必须先于 exec）**：

```python
spec = importlib.util.spec_from_file_location("v4afmt_config", path)
mod = importlib.util.module_from_spec(spec)
sys.modules["v4afmt_config"] = mod      # 必须先注册（dataclasses 按 cls.__module__ 查 sys.modules）
spec.loader.exec_module(mod)
```

- **config_hash 契约**：扰动侧参数**一律放本目录 config.py**，绝不进 gpfm Config（v4a-fmt config_hash 逐位不可变；ckpt 校验需原 hash 05d9bb875a2c）。cfg 对象一律来自 `v4afmt_config` 实例（`iterate_test_batches` / `load_eval_model` / `build_eval_prior` 全 cfg 驱动，已核不受遮蔽影响）。
- **装载前置断言**：调用 `iterate_test_batches` 前断言 `Path(cfg.split_manifest_override).exists()` 且等于已登记路径——防 `fixate_split`「不存在则写入」分支把新 manifest 写进只读变体目录。
- **BLAS 护栏**：run.py / tests 启动器 / refs/compare 子进程启动器统一复制 `_blas_guard` 五变量 setdefault 块（numpy/torch 导入前；子进程经环境继承）。

## 2. 四要素设计（本任务核心交付；1.4 用户审定，主 Run 后定版）

### 2.1 扰动算子 P(X, A, r) —— A 案：增益调制（默认）

$$x_A(t) \leftarrow (1+r)\cdot x_A(t),\quad t\in\text{past 全段}(384\ \text{样本}),\ r\ge 0$$

- **注入层**：模型输入层（X_past 通道 A）——框架 Stage II 正典位置；r 保持 EEG 幅度单位（相对增益，无量纲），可解释、跨通道/受试者可比。
- **选型理由**：①谱域保持在数据流形上（fmt 数据 1–40 Hz 带限 + z-score，增益不引入 DC/带外分量；加性 DC 偏移 off-manifold）；②r=0 严格回到 baseline（CRN 下 D(0)=0 逐位）；③剂量-响应连续单调可期；④神经科学中增益调制是标准干预范式。
- **幅度域外推警示（判读纪律）**：谱域在流形上 ≠ 幅度域在流形上——r=1.28 使源通道 z 域幅度 ×2.28，超出训练分布支持域，**高剂量段（r≥0.32）属网络外推行为**。主 readout（slope，r≤0.08）受保护；AUC（全网格）与 r\*（搜索至 1.28）的判读须随附此 caveat；若启用「网格上界扩至 2.56」档位调整权，外推进一步加剧，须重登记。
- **备选 B 案（文档级预留，不实现——用户 2026-10-17 拍板 A 案）**：加性带限固定波形 x_A + r·σ_A·g(t)，g = 1–40 Hz 带限、固定 seed 的归一化波形（参考文献「选择性增加」的带限版）。仅当后续任务需要时按口径变更纪律另行实现。
- **方向**：主流程仅正方向（r≥0）；负方向（r∈(−1,0]，去增益）留作扩展观察项不进主流程。
- **扰动传播通路（review A-M1 更正后的事实口径）**：GPR 条件先验的 `posterior_mean` 为**逐通道**时间投影（`einsum("bcp,fp->bcf")`，与空间核 K_s 无关）；空间核 L_s 只作用于先验噪声 ε。CRN 契约下 baseline/perturbed 共享同一 ε/η ⇒ **先验随机项对 δ 的贡献恒为零**。因此：扰动源 A 后先验均值只在 A 通道变化，**一切跨通道响应（B≠A）仅经速度网络单通路传播**（含 rank-16 输入瓶颈——恰是 H3 已知弱项，见 §7 Risk#1）；先验通路仅贡献自响应（A=B）。`posterior_mean` 诊断相应登记为：①恒等校验（预期 B≠A 的先验差严格为零，作实现正确性断言）+ ②自响应的先验分量量化（解析、零采样成本）。

### 2.2 响应度量 D —— CRN 配对归一化 RMS

逐窗 w、源 A、靶 B、强度 r，K 条 CRN 样本：

- baseline：$\hat x^{(0)}_{w,k} = F_{\theta^*}(X_w)$；perturbed：$\hat x^{(A,r)}_{w,k} = F_{\theta^*}(\mathcal P(X_w,A,r))$；**同一组显式 eps/eta**（CRN 契约，`EulerSampler.sample(eps=, eta=)`）。
- **eps/eta 共享范围（预注册钉死）**：逐窗一组 eps/eta，**跨全部 62 源 × 8 档 r 与 baseline 全共享**——同窗剂量-响应曲线与跨源比较不受蒙特卡洛抖动污染，且单次调用复用张量省显存。
- **采样器构造口径（B-Major1 登记）**：`EulerSampler(..., gen_lowpass_hz=getattr(cfg, "gen_lowpass_hz", 40.0))`——v4a-fmt cfg → **None**（fmt 产物口径）；构造器默认 40.0 + `brickwall_lowpass` 默认 fs=500 是已登记陷阱（对 250 Hz 数据有效截止 ~20 Hz，静默削波 D），B0 单测断言 `sampler.gen_lowpass_hz is None`。
- **批采样纪律（B-Minor4/6）**：每窗**一次** `sample()` 调用，baseline 与全部 (A,r) 变体并入同一 batch（B = 62×8+1 = 497，chunk_bk=1024 单 chunk）——既保 GPU 利用率（外推 ~0.4–2 h 量级成立的前提），又使 r=0 恒等断言在同批形内成立（跨批形逐位相等无保证，cuBLAS/mamba 核随批形选择）。
- 逐样本配对差：$\delta_{w,k}(t) = \hat x^{(A,r)}_{w,k,B}(t) - \hat x^{(0)}_{w,k,B}(t)$
- 归一化尺度：$s_{w,B} = \mathrm{RMS}_{k,t}\big(\hat x^{(0)}_{w,k,B} - \overline{k}\big)$（baseline 生成分布在靶通道的自身波动尺度，纯模型侧、不依赖真实未来段）
- **响应强度**：$D_{A\to B}(r;w) = \mathrm{RMS}_{k,t}(\delta_{w,k}) \,/\, s_{w,B}$（无量纲；= 靶通道未来段变化幅度占生成分布自身尺度的比例）
- 受试者级聚合（预注册主口径）：$\bar D_{A\to B}(r) = \mathrm{median}_w\big[D_{A\to B}(r;w)\big]$（窗数 N_w，稳健）。窗级曲线一并落盘供诊断。

### 2.3 响应阈值 τ —— 数据驱动独立模块（框架 §8.3 要求）

- **null 分布（自然波动）**：同一窗 baseline 的 K 条样本两两配对（k_i,k_j），$d^{\text{null}}_{w,B}(i,j) = \mathrm{RMS}_{t}(\hat x^{(0)}_{w,i,B} - \hat x^{(0)}_{w,j,B}) / s_{w,B}$——与 D 同构、扰动为零，复用主 Run 的 baseline 样本（零额外采样成本；K=8 → 28 对/窗/通道）。
- **τ 池化域（预注册钉死，review A 建议）**：**逐被试分池**（该被试全部 w×B×对，≈ 32×62×28 ≈ 5.6 万样本/例）——与 D̄ 逐被试 median 聚合一致；τ 主口径 = 该池 **95% 分位**，敏感性登记 q90 / q99。含义：响应须超过该被试生成过程自身波动幅度方计为有效响应。
- 逐通道 τ_B（按 B 通道分池）落盘作敏感性口径，不进主判定。

### 2.4 临界强度搜索与 readout 三件套

- **r 网格（低剂量加密，9 点含 0）**：$R = \{0,\ 0.01,\ 0.02,\ 0.04,\ 0.08,\ 0.16,\ 0.32,\ 0.64,\ 1.28\}$（对数等比 ×2，上界 = 全幅增益 ~128%；0 点由 CRN 恒等按构造取 D=0，不采样）。
- **r\***：$r^*_{A\to B} = \min\{r\in R: \bar D_{A\to B}(r)\ge\tau\}$；网格内未达 → **右删失**，编码 $r^* = r_{\max}^{+}$（下一档，由 R_GRID 推导 = 2.56，**禁字面量**——档位调整权启用时自动一致），敏感性随 q90/q99 τ 重算登记。
- **主 readout = 低剂量斜率**：OLS 于低剂量段 $\{0, 0.01, 0.02, 0.04, 0.08\}$（5 点含原点）；敏感性段 $\{0,0.01,0.02,0.04\}$。天然大=强。
- **次 readout = AUC**：全网格梯形面积 / r_max 归一。天然大=强；**判读随附 §2.1 高剂量外推 caveat**。
- **rstar 翻转（进比较器前）**：$\text{strength} = r_{\max}^{+} - r^*$（删失 → 0 最弱；越大越强；由 R_GRID 推导），契约 orientation='larger_is_stronger'。
- **有效路径集**：$\{(A,B): A\ne B,\ r^*_{A\to B} \le r_{\max}\}$（τ 主口径判定），随附 slope 排序 top-100 路径清单与全矩阵落盘。

## 3. 数据与运行规模

- **数据**：fold0 test 10 例（IDs [10,11,12,17,24,28,30,44,61,69]），`EdraWindowsDataset` 只读（past 384 / future 128 / stride 64 @250 Hz，930 窗/例，评估侧 z-score = run 目录 channel_{mean,std}）。
- **窗子采样**：N_w = 32 窗/例（确定性，seed 2026，逐例独立；具体抽样式实现时固定并登记 experiment.md）。
- **K** = 8 条 CRN 样本/窗（显式 eps/eta 由固定种子张量生成，seed 2026；注册与评估口径 K=16 的差异——本任务为扰动配对推断，K 需求由 null 对数与 RMS 稳定性决定）。
- **计算量（B-Minor6 口径修正）**：主 Run ≈ 10 例 × 32 窗 × (62×8+1) 变体 × K=8 ≈ **1.27M 条采样轨迹**，每条 16 Euler 步前向 ≈ 20.3M 步前向；对照 v4a-fmt 训练 6.27 s/epoch（31.6k 窗 fwd+bwd）外推 **~0.4–2 h**——前提 = §2.2 批采样纪律（每窗一次调用、变体并批）；bench 实测校准。nohup 前先 `mkdir -p outputs/perturb-ec-f0/main-01/logs/`。
- **档位纪律**：smoke + bench 后 N_w/K/网格**最多调整一次**（evaluation-calibers 变更纪律），调整前后 Run 不可比部分显式登记（含 §2.1 外推警示重登记）。

## 4. 参考侧（observational references，fmt 同源重算）

- **数据口径**：fmt npz (62, 60001) @250 Hz → `resample_poly(down=2)` → (62, 30000) @125 Hz → 逐通道 z-score（该例全记录统计，ddof=0）——与 baseline-ec-v1 resample_io 同构、仅数据源与降采样比不同（登记：fmt 1–40 Hz 带限无 ICA·48 例 vs v4 bp1-40+notch·46 例）。
- **EC 参考**：`refs_for_subject(x, p=8)` 复用（ridge λ_rel=1e-2、p=8、GC=条件 ln(SSR 比)、gPDC=|gPDC|² 宽带 1–40 Hz——定标证据见 10-13 任务 experiment.md Run 001；同为 125 Hz 物理率，λ/p 口径直接适用）。
- **split-half 天花板（必做，review A 对齐预登记口径）**：fmt 侧 10 例按 baseline-ec-v1 split_half 同口径重算（奇偶 2 s 段 @125 Hz=250 样本），登记 ρ̄/ceiling 比值（成本 ≈ refs×2，分钟级）。
- **术语**：一律 observational directed-connectivity references；目标是结构收敛性验证。

## 5. 比较器（复用 baseline-ec-v1 compare.py）

- 输入：refs = {gc, gpdc}（10 例）+ EC_pert 逐 readout npz（load_ec_pert 契约：`ec (62,62)`、`readout∈{slope,auc,rstar}`、`subject_ids`、`orientation='larger_is_stronger'`）。
- **主比较（预注册唯一）**：主 readout（slope）× 两参考法（GC, gPDC）→ 逐被试距离偏相关 Spearman ρ → 单样本 t（n=10）× **Holm 跨 2 法** + 1000×受试者 bootstrap（seed 2026）。auc/rstar 同流程跑作上下文，不进 Holm 族。
- 未偏相关 ρ、天花板比值为上下文登记。
- **检验力注记（review A）**：df=9、Holm 校正后 α≈0.025、逐被试 ρ 典型 sd 0.25–0.35 下，可检效应量约 ρ̄≥0.55–0.65。**阴性结果 = 证据不足，走 AC6 缺口诊断路径，不表述为「证伪收敛」**。

## 6. 诊断与可视化（流程验证证据链）

1. **CRN 契约**（单测）：r=0 逐位恒等（**同批内**相邻元素断言，B-Minor4）；扰动仅改源通道行；同 eps/eta 两次调用逐位相同；`sampler.gen_lowpass_hz is None`。
2. **剂量-响应形态**：随机抽样 20 组 (A→B) 的 D̄(r) 曲线图 + 全边单调性分布（Spearman(D̄, r)）+ 低剂量段 OLS R² 分布。
3. **噪声地板分离**：null 分布 vs 各 r 档 D 分布重叠图；τ 线标注；跨通道边（A≠B）在 r_max 达 τ 的比例（删失率）。
4. **空间结构 sanity**：D̄(r_max) vs 通道对欧氏距离散点/衰减曲线（定性 sanity，不作判据）。
5. **对角 vs 跨通道**：A=B 自响应（含先验分量解析量化）与 A≠B 响应量级对比（自响应应显著更强——代理脑扰动传导 sanity）。
6. **EC 矩阵热图**（三 readout × 例均）+ top-100 有效路径清单表。
7. 图全走 `common/visualizer.py` **纯增量扩展**的新绘制类型（300 dpi PNG，figs/ 落盘）。

## 7. 已知风险与对策

| # | 风险 | 对策 |
|---|---|---|
| 1 | **跨通道响应仅速度网络单通路**（§2.1 更正：先验逐通道、CRN 零贡献），叠加 v4a rank-16 输入瓶颈已知弱项（H3 旧范式 GC r 0.092）→ D 贴地板、r\* 大面积右删失 | B1/B2 首个诊断节点如实登记；预置一次档位调整权（网格上界扩档，附外推重登记；B 案算子为文档级预留、启用须另走口径变更+实现）；「流程可跑通但该 θ\* 跨通道传播弱」本身是合法任务结论（AC6 缺口诊断路径） |
| 2 | **模块名遮蔽**（三目录各有 config.py；ec_refs/compare 模块级 `import config as C` + def 期求值） | 本目录与 v4a-fmt config 均唯一名装载（§1 配方，sys.modules 先注册后 exec）；baseline-ec-v1 链一律子进程隔离；B0 断言 `config` 未被本任务占据、sample/data/gpr_prior `__file__` 归属 v1 目录 |
| 3 | **gen_lowpass_hz=40 默认陷阱**（构造器默认 + brickwall fs=500 → 250 Hz 数据有效截止 ~20 Hz 静默削波） | 构造显式 `getattr(cfg,...)` 透传（→None）+ B0 断言 + experiment.md 实现口径声明（B-Major1） |
| 4 | config_hash 漂移破坏 ckpt 校验 | 扰动参数隔离在本目录 config，gpfm Config 零改动；单测断言装载 hash = 05d9bb875a2c |
| 5 | 跨批形逐位不相等（cuBLAS/mamba 核随批形选择） | 恒等断言一律同批内；主 Run 每窗单次调用变体并批（§2.2） |
| 6 | readout/翻转常数字面量失配（档位调整权启用时） | R_NEXT/翻转常数一律由 R_GRID 推导；合成数据 readout 单测（D̄=2r→slope 2、删失→strength 0、npz 往返） |
| 7 | 930 窗全集 vs N_w=32 子采样代表性 | 窗级分布诊断 + N_w=32 vs 全集 1 例对照抽查（bench 阶段顺带） |
| 8 | K=8 的 null 分位数估计噪声 | 28 对/窗/通道 × 32 窗 ≈ 5.6 万/例分池，q95 稳定；q90/99 敏感性兜底 |

## 8. 执行序与回滚

- 顺序：脚手架+单测 → smoke（1 例 2 窗全链）→ bench（1 例 N_w=32 计时外推）→ 主 Run（10 例 nohup）→ readout+诊断 → 参考侧+比较 → check/spec/收尾。experiment.md 每 Run 块先登记后跑。
- 回滚点：①CRN/低通/hash 单测红 → 停在装载/采样层修复，不进主 Run；②smoke 曲线形态异常（非单调/NaN）→ 停，诊断后重设计网格或算子（登记）；③主 Run 中断 → run 目录保留换 run_id 续跑（nohup 前先建 logs/）；④比较器契约断言失败 → 只修包装层与子进程胶水，不碰 baseline-ec-v1。
