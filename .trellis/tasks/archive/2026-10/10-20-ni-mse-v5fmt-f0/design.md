# 设计 — NI-MSE-50-q0.90 执行实例化（θ\* = v5-fmt-01）

> 权威设计源：`.trellis/tasks/10-19-gpfm-mamba-v4b-gnn-f0/ni-mse-design.md` **v1.5**（五要素/伪代码/预算/风险全文在该文档，本设计只记**实例化落定、勘误、代码架构**）。规范约束：`.trellis/spec/research/perturb-ec-pipeline.md`（装载防坑矩阵/BLAS/契约惯例——本路径算子与度量不同，但装载与工程契约全部适用）。
> 立项：2026-10-20，用户指令「v5-fmt 已经实现完毕，进行 NI-MSE」（主实例化 θ\* 已由归档任务 10-20-gpfm-mamba-v5-fmt-f0 产出）。

## 1. v1.5 → 本任务实例化落定

| 项 | v1.5 原文 | 本任务落定 | 备注 |
|---|---|---|---|
| θ\* | v5-fmt-01（计划中） | `outputs/gpfm-mamba-f0/v5-fmt-01/checkpoints/gpfm-mamba-v5-fmt-best.pt` EMA shadow；config_hash pin `a64c0d416756`；**epoch 28 / best_val 0.70242**（ckpt 字段；experiment.md「ep29」为 1 基显示差异，以 ckpt 为准） | 只读 |
| 数据 | edra_processed_fmt fold0 test 10 例 | manifest `models/gpfm-mamba-v1-fmt/splits/fold0_status_ok_seed42_val4.json` 只读（34/4/10）；**test = [10,11,12,17,24,28,30,44,61,69]** | **勘误**：设计 §1.1 名单 [12,17,18,24,30,38,44,55,58,61] 为 v4 过渡口径 |
| n₀ | fmt@250 = 25 | `n0 = ceil(0.1 × 250) = 25` | future 128 样本的前 25 |
| c 臂 | 主 0.5；敏感性 {0.1, 0.3} | perturb 期跑 **{0.05, 0.1, 0.3, 0.5}** 四臂：0.5 主读出 / 0.05 阴性对照（§4.2）/ {0.1,0.3} 敏感性 | null 与 c 无关（同批复用） |
| run-id / 产物根 | `ni-mse-v5-01`（产物根立项时定） | **`ni-mse-v5fmt-01`**，根 `outputs/10-20-ni-mse-v5fmt-f0/ni-mse-v5fmt-01/`（`null/ perturb/ report/ manifest.json logs/`） | run-id 显式标 θ\*，与未来可能的 v5-main 过渡实例化隔离 |
| 判读主口径 | §1.1（v1.4a/v1.5）逐被试配对 | **逐被试配对主口径**（n=10）；组级单次副口径——§1.5 步骤表 E 行「主口径组级单次」为 v1.0 旧文本，以 §1.1 为准 | refs 按被试 ID 命名 s{sid}，10/10 在位（已实测核对） |
| 代码位置 | （未定） | 新路径目录 **`models/ni-mse-v1/`** | registry Study II 行新路径，3.3 回填 |
| 种子树 | 三支（ε_ref/ε_null/ξ_pert） | **四支：+ η（σ_min 噪声）独立派生支** | **审查补钉（Mandate B 阻断①）**：`sample()` 的 η 缺省走全局 CUDA RNG 破坏逐位复现；v1.5 §2 伪代码只传 eps 的写法不得照抄 |

其余执行参数（算子语义/窗子集/R1–R6/聚合链）**逐字按 v1.5，不再复述**。**τ_b 口径澄清（Mandate A 建议①）**：τ_b 对池化 (W_null×K_null) 样本逐通道取 Q90；「k 中位→w 中位」链只作用于响应 R̄ 与汇报层。**判读锚点（Mandate A 建议②）**：实现锚定本设计 §4；v1.5 §2 伪代码 E 段（组级均值主读出/bootstrap seed）为 v1.4a 前旧口径残留，不得照抄。

### 1b. 审查补钉清单（2026-10-20，trellis-review 双 mandate 后钉定）

1. **η 独立种子支**（阻断①，见上表）；η 与 ξ 同构四元组 (sid,w,a,k) 派生。
2. **SeedSequence 合法形式**（建议③）：字符串 spawn_key 本机非法——统一 `np.random.SeedSequence(entropy=2026, spawn_key=(<branch_int>, …))`，分支整数映射固定：ε_ref=10 / ε_null=11 / ξ_pert=12 / η=13 / win-null=20 / win-pert=21（被试级再加 sid 末位）；映射表落 manifest。
3. **每次 `sample()` 调用行上限 B·K ≤ 8,192**（建议④）：eps/eta/x0 在 chunk 分块前按 (B,K) 全量物化，perturb 期按 per-(a,c) 窗循环组织（50×32=1,600 行/调用，天然安全）；禁止整例单调用。
4. **R2 同批形前提**（建议⑤）：c=0 变体与干净基线**并入同一 `sample()` 调用同批**（同 (B,K) 形 + 同 eps/eta）——跨批形 cuBLAS/mamba 核不保证逐位相等（spec §1 契约）。
5. **Ŷ_ref 随 null 落盘**（建议⑥）：`null/{sid}.npz` 含 W_pert 50 窗的 Ŷ_ref（≈1.6 MB），perturb 期只读复用，消除跨期重算批形微漂移。
6. **R3 加密不覆盖**（建议⑦）：K_null 加倍重跑落独立子目录 `null-k128/`，双登记。
7. **续跑工程契约**（建议⑧）：manifest 原子写（tmp + os.replace）；report 期先断言上游 10×4 perturb + 10 null 产物齐全有限；R2 失败 = 非零退出码 + 失败标记文件（后续 stage 拒跑），不得 catch-all 吞异常。
8. **thostar 装载顺序钉死**（建议⑨，Run 007 实测教训）：**先预占 `sys.modules[config/model/graph]`（exec v5-fmt 文件）再 `import sample`**——v1 `sample.py` 导入时会把 v1 目录插回 sys.path[0]，顺序敏感；`__file__` 断言面覆盖 config/model/graph/mamba/sample/data/gpr_prior 七项。

## 2. 代码架构（`models/ni-mse-v1/`）

```
models/ni-mse-v1/
├── _blas_guard.py        # 入口 BLAS 五变量 setdefault（numpy 导入前，spec 惯例）
├── thostar.py            # θ* 装载：先预占 sys.modules[config/model/graph]（v5-fmt 文件 exec）再 import sample
│                         #   （v1 sample.py 复用；其 import model 命中 v5-fmt 缓存；mamba 命中 v1）
│                         #   __file__ 断言七项（config/model/graph/mamba/sample/data/gpr_prior）
│                         #   ckpt EMA shadow 回灌 + config_hash 断言 + channel_mean/std 只读
│                         #   EulerSampler(…, gen_lowpass_hz=None)  ← fmt 无低通，默认 40.0 为已登记陷阱
├── windows.py            # fmt test loader（shuffle=False 确定性顺序）+ 窗子集
│                         #   win-null: SeedSequence(2026, spawn_key=(20, sid)) 无放回抽 200 按原序
│                         #   win-pert: SeedSequence(2026, spawn_key=(21, sid)) 从 W_null 内抽 50 按原序（⊂ 断言）
├── noise.py              # P(X,a,c) 加性零均值高斯（标准化空间）+ ξ/η 种子树
│                         #   ξ: spawn_key=(12, sid, w, a, k)；η: spawn_key=(13, sid, w, a, k)（同构四元组派生）
├── stages.py             # 三期核心：null（B_ref 网格 spawn_key=(10,…) + K_null 独立 ε (11,…) → N + τ_b/split-half/Q95
│                         #   + Ŷ_ref(W_pert 50 窗) 随 null/{sid}.npz 落盘）
│                         #   perturb（per-(a,c) 窗循环，B·K≤8192 → R → R̄=median_k median_w）
│                         #   R2 断言内嵌：每例首窗 c=0 变体与干净基线并入同一 sample() 调用同批
│                         #   （同 (B,K) 形 + 同 eps/eta）逐位比对，失败=非零退出+标记文件，后续 stage 拒跑
├── report.py             # S=R̄/τ_b、G=R̄>τ_b、Q95 旗标带；refs 配对（s{sid}.npz 只读）
│                         #   Pearson/Spearman direct 主 + flip 副 + 距离偏 Spearman（geometry pairwise_d）
│                         #   逐被试 n=10 mean±std/median/range + 组级副口径 + 48 例敏感性登记
├── run.py                # CLI: --stage {null|perturb|report} --run-id ni-mse-v5fmt-01
│                         #   分期断点续跑；nohup + 显式重定向；manifest.json 每期增量写
└── tests/                # test_ni_mse.py：
                          #   ① P(X,a,0)≡X 逐位恒等（纯张量）
                          #   ② 窗子集确定性（同种子两次抽取逐位一致；⊂ 关系；原序保持；spawn_key 合法形式）
                          #   ③ τ_b 分位数单点（合成 N 池化数组 vs 手算 Q90/split-half——池化口径，非中位链）
                          #   ④ refs 映射（10 test sid 全命中 s{sid}.npz；ec 形状 (62,62)；行=源）
                          #   ⑤ R2 GPU 前向强形式（1 窗最小规模：P(X,a,0) 扰动路径输出 vs 干净基线路径输出，
                          #     并入同一 sample() 调用同批同 eps/eta，逐位恒等）
                          #   ⑥ loader 遮蔽契约（thostar 装载 config/model/graph 解析到 v5-fmt 目录，七项 __file__ 断言）
                          #   ⑦ sampler.gen_lowpass_hz is None 断言（已登记陷阱锚点：默认 40.0 静默削波）
```

**装载契约（防坑矩阵移植）**：
- `sys.path` 插入顺序 = [v5-fmt 目录, v1 目录]；`import sample`（v1）→ 其 `from model import …` 命中 v5-fmt `model.py`（sys.modules 缓存）；`import config` 解析到 v5-fmt `config.py`；逐一 `__file__` 断言（对齐 v5-fmt `evaluate.py` 包装的守卫样式）。
- cfg 对象来自被装载变体 config 实例；扰动侧参数（c 臂/窗数/K/种子）为 **本目录独立常量**，不进 gpfm Config。
- 模型构建/prior 构建逻辑从 v1 `evaluate.py` 的同型路径复用（kernel_t=ou、p=25、σ_s/jitter、euler_steps=16、sigma_min=1e-4 等全部来自 θ\* config.yaml，禁止字面量重抄）。
- 采样接口：`EulerSampler.sample(x_past(B,C,P), n_samples, generator=None, eps=(B,K,C,F), eta=(B,K,C,F))`——**eps 与 eta 均显式传入**（禁 generator；η 缺省走全局 CUDA RNG 为逐位复现破坏点——种子树第四支即为此钉）；单次调用行上限 **B·K ≤ 8,192**（eps/eta/x0 在 chunk 分块前全量物化）；**注意**：本路径 null/响应侧刻意用**独立 fresh ε**（可交换性，v1.5 §3 与 perturb-ec-v1 全 CRN 的口径差异），恒等断言只在 R2（c=0、同 ε/η、同批形）意义上使用。

## 3. 数据流与产物

```
  fmt npz(10 例) ─loader(z-score, shuffle=False)→ 窗网格(~930/例)
  └─ win-null(200/例) ──┬─ B_ref=64 固定 ε 网格 → Ŷ_ref(w)         [null/]
                      │    （W_pert 50 窗的 Ŷ_ref 随档落盘，perturb 期只读复用）
                      └─ K_null=64 独立 ε → N(62,200,64) → τ_b(62)（池化 W×K 逐通道 Q90）
                          + split-half + Q95                          → null/{sid}.npz
     win-pert(50/例 ⊂ win-null) × 62 源 × c∈{4 臂} × K=32
                        → R[a,b,w,k] → R̄=median_k median_w           [perturb/]
                                                                     → perturb/{sid}_c{c}.npz
  τ_b × R̄ → S=R̄/τ_b, G=R̄>τ_b（逐被试）→ 组级 median
  × refs_s{sid}.npz（只读）→ 相关四件套 + 距离偏 + flip + 阴性对照    [report/]
                                                                     → report/report.json + summary 表
  manifest.json：种子树全指纹（含分支整数映射表）/ 窗子集索引 / 参数 / θ\* config_hash+ckpt SHA / env 快照
               （原子写 tmp+os.replace；R3 触发时 K_null 加密重跑落 null-k128/ 独立子目录双登记，不覆盖）
```

- 聚合链：N →（k 中位 → w 中位）→ 被试级 → 组级中位 + 逐被试表；R 同构。
- 相关计算纯 CPU 矩阵运算（scipy.stats.pearsonr/spearmanr + 距离偏残差化 OLS），入口 BLAS guard。
- 交付无图（v1.5 未要求图；数值 json/npz/表）。

## 4. 判读口径（规范化，以 v1.5 §1.1 为准）

1. **主口径**：逐被试 EC_pert_s（S 矩阵）× ec_refs_s，offdiag 3,782 对直接配对（行=源，无转置）→ Pearson + Spearman；n=10 汇报 mean±std / median / range。
2. **副口径**：组级单次（10 例 S 逐元素中位 vs 同 10 例 refs 均值）；48 例 refs 全均值敏感性登记。
3. **flip 检查**：同式对 ec.T（direct ≈ flipped ⇒ 相关由对称分量承载，方向信息弱）。
4. **距离偏判定轨**：双方对 pairwise_d[off] 残差化（OLS）后再 Spearman（组级 + 逐被试随附）。
5. **阴性对照**：c=0.05 臂 G 通过率（名义 ~10%=Q90）与 S 分布位置（→1 附近）。
6. **结构预测登记**（§4.4，判读非门槛）：v5-fmt-01 A_eff 几何主导更强（W 占比 0.85% vs v5-main 1.7%、|A_eff|×距离 Pearson −0.698）⇒ 预期 EC_pert 强距离衰减、flip≈direct；若方向性显著超出 W 信息量 → 归因排查（LN/head Linear 通路）。

## 5. 风险与边界（继承 v1.5 §5 + 实例化新增）

- 全部 v1.5 §5 条目适用（null 可交换性近似/逐通道 τ 合理性/0.1s 窗漏检慢传导已拍板接受/Q90 多重比较——Q95 旗标带承担稳健参考）。
- **新增①（跨范式 caveat 状态）**：v5-fmt-01 与 refs 同处 fmt 管线 ⇒ v1.5 预告的 caveat 消除成立；剩余限制 = 样本量 n=10 + EC_pert 自 50 窗（每例 930 窗的确定性子集）× refs 自全长 60,001 样本的窗数支撑差异（解释性登记）。
- **新增②（θ\* 训练纪律）**：v5-fmt-01 早停 ep39/best ep28（vs v5-main-01 跑满 100）——θ\* 质量差异随档登记，不影响本路径流程效度。
- **新增③（预算）**：四 c 臂约 4×17 min + null ~5 min + 装载/IO ≈ **1.5 h 单卡**；R3 触发（split-half >10%）则 null 侧 K_null 加倍（~+5 min）重跑。
- **边界重申**：θ\* 为「候选」非「合格代理脑」（三项检验另立）；本任务产物 = EC_pert 路径首执行 + 判读登记，不构成 r\*→连接强度最终映射（框架第 14 步，待拍板）。
