# 实验日志 — 虚拟扰动范式×响应度量低成本搜索（gpfm-mamba-v1-fmt θ*·fold0）

> 每次「跑批/训练/评估/统计」追加一个 Run 块：跑前先写「目的+配置」，跑后同会话补「结果+结论」，不留空块。
> 格式契约见 `.trellis/spec/research/experiment-log.md`；产物只落本任务声明的变体目录（`.trellis/spec/research/path-isolation.md`）。

## Run 001 — 阶段一：固定双探针 {G, N} 扫 6 度量候选 → 选 M\*（2026-10-17）

### 目的与假设

- 问题：冻结 v1-fmt θ\* 上，6 个输出端响应度量候选（M-RMS/M-PSD/M-band/M-env/M-FC/M-time）哪个能对输入扰动给出**敏感（离开采样噪声地板）、特异（自响应≥2×跨响应）、稳定（跨受试者 split-half ρ≥0.5）、单调（仅改变 r 下 |Δ| 随 r 单调不减，Gate D）**的读出？选出 M\* 供阶段二使用。
- 假设（预注册 gates，design §5；**Gate D 为 2026-10-18 用户拍板增补**）：前次失败模式 = 跨通道传导弥散（自跨比 1.27×）；本阶段以 Gate B 为主序检验度量能否放大自/跨差异。
- 含 B1 smoke（1 窗阶段一批机械正确性）先行。

### 配置

- 代码：`models/perturb-search-v1/`（config.py 唯一配置源 = design §2–§5 字面转录；candidates/metrics/scorecard/run_search + tests/）；git 快照见 meta.json。
- θ\*（只读冻结）：`outputs/gpfm-mamba-fmt-f0/v1fmt-main-01/checkpoints/gpfm-mamba-v1-fmt-best.pt`（EMA，epoch 81，config_hash `0dc0d5249c68` 严格校验 + `sampler.gen_lowpass_hz is None` 断言；装载沿 perturb-ec-v1/loader.py 配方换 v1-fmt config）。
- 固定输入探针（design §6 预注册，用户 2026-10-18 拍板双探针）：G（乘性增益 r0=0.25）+ N（加性带限噪声 r0=0.5，g = 1–40 Hz 带限固定 seed=2026 FFT 掩码 RMS=1 全局共享波形）。
- **r 网格（2026-10-18 用户拍板增补，Gate D 用）**：5 点倍率梯 {0.25, 0.5, 1, 2, 4}×r0 + 逐算子上限钳制（G≤0.32 外推警示界、L≤1.0 语义界），钳制后升序去重 → 本阶段 **G 4 格 {0.0625, 0.125, 0.25, 0.32}、N 5 格 {0.125, 0.25, 0.5, 1, 2}**（r0 本点并入网格不重复行；字面网格 = config.R_GRID 预注册钉死）。
- 度量候选 6 个 + 实现参数（钉死）：Welch fs=250/nperseg=128/Hann/50% overlap/bin 中心归属/ε=1e-12；五带 δ1–4/θ4–8/α8–13/β13–30/γ30–40；M-env α 带 Butterworth(3) sos+filtfilt 零相位 + Hilbert 包络；M-FC 逐 k 相关阵后均摊；M-time 前 32 样本 |δ̄| 均值（全段峰值/早期全段比照登）；null = baseline K=16 拆 8+8 两半同构距离（h1 pert 侧/h2 base 侧）。
- 窗集与枚举：fold0 test manifest 顺序前 3 例 = **受试者 {10, 11, 12}**（MDD）× 每例数据集顺序前 8 窗 = 24 窗；**win_idx = 受试者序×8 + 例内窗序（例#1→0–7、#2→8–15、#3→16–23）**；通道顺序 = 数据集 npz 列序（与 chan_coords_62.npy / channel_{mean,std}.npy / splits manifest 同源对齐）。
- CRN：K=16 样本/窗，seed=2026（逐窗 `manual_seed(2026+win_idx)` 先 eps 后 eta），每窗**一次** `sample()` 调用、baseline 与 558 变体行并批（**批形状 559 = 1 + 62 源 × 9 格点（G4+N5）**，行 0 = baseline，行 1+a·9+j——j = (范式, r) 平铺格点序）；**跨 r 的 CRN 逐位配对 = Gate D「仅改变 r」**。
- Gates（design §5 预注册 + Gate D 增补）：A 敏感性 median_offdiag(z)≥2（z = D_edge(24 窗 median)/null_std(24 值 std，逐探针判，至少一探针过）；B 特异性 median 自响应/median 跨响应 ≥2（两探针合并边池；Spearman(D, 通道距离) 照登）；C 稳定性 例#1+#2 vs #3 offdiag 秩相关 ρ≥0.5（合并池）；**D 单调性（2026-10-18 增补，用户拍板）**：逐 (探针, 度量) 对 |Δ| 聚合曲线（D_edge 24 窗 median → offdiag 边池 median）随 r 单调不减、**曲线相邻步 100% 不减才过**（「不减」含等号，rtol=1e-9 浮点护栏；**逐边通过比例记录并报告**——csv `gateD_edge_mono_frac` 列 + summary「D 边单调占比」列；违反步明细——步序/两端 r/曲线值/相对降幅——落盘）；A/B/C/D 均评：A 与 D 逐探针判（各至少一探针过线）、B/C 合并池、A/B/C 一律 @ **r_2 = 网格中位格点（= r0 本点，全部范式格点序 2，用户 2026-10-18 口径确认）** 切片。选择规则 = 四门全过入场，Gate B 主排序 → Gate C → 可解释性 → 成本；全不过线 = 如实阴性 + 逐 gate 缺口。
- 环境：base @ 本地服务器（RTX 4080S 32GB、16 核 cgroup、BLAS 护栏 8 线程、torch 2.8.0+cu128、scipy 1.18.1；environment.md 契约）。
- 数据：`/root/autodl-tmp/edra_processed_fmt`（fmt 范式，1–40 Hz@250 Hz 带限，z-score 用 v1fmt-main-01 训练折统计）；splits manifest `models/gpfm-mamba-v1-fmt/splits/fold0_status_ok_seed42_val4.json`（只读，装载前置断言）。
- 命令：`cd models/perturb-search-v1 && nohup python3 run_search.py smoke|phase1 > ../../outputs/perturb-search-f0/logs/….log 2>&1 &`
- 产物：`outputs/perturb-search-f0/{smoke/run_01, phase1-metric-search/run_01}/`（逐格点 per_edge_D.npz 键 `D_{pr}_r{r}_{m}` + `gateD_curve_{pr}_{m}` + `r_grid_{pr}` + scorecard.csv + summary.md 含 Gate D 违反明细段 + meta.json）。

### 结果

- **执行史（如实登记）**：主 Run 共 4 次调用。前 3 次分别崩于 B0 起潜伏、但首次被本 Run 走到的三处 scorecard 路径 bug，均已修复 + 单测 ⑨ 回归锁定：①Gate B Spearman 合并边池 dist 未随探针平铺（7564 vs 3782 维度错）；②阴性诊断访问 cr 形行缺失的逐门 pass 键（KeyError）；③「最佳可得」方向反（min → 修正为 maximin 达成率最大）。**Run#1（崩溃前 npz 已落盘）与 Run#3 的 per_edge_D.npz 94 键逐位一致（CRN 跨 Run 复现 ✓）**；Run#4 = 最终产物。
- **Run#4**：wall **7.4 min**（sample 累计 74.8 s；AC6 30 min 预算内，r 网格化 ×4.7 行数下仍宽裕）；批 559 行 × 24 窗 × K=16。
- **scorecard（A/B/C @ r_2 = r0 切片；D 全网格；G 4 格 / N 5 格）**：

| 度量 | A_z（G/N，门 2） | B 自跨比（门 2） | C ρ（门 0.5） | D 违反步 G/N（门 0） | D 逐边占比 G/N | 四门 |
|---|---|---|---|---|---|---|
| M-RMS | 0.79 / 0.99 | 1.50 | 0.72 | 0/3、0/4 | 1、1 | 未过（A、B） |
| M-PSD | **1.23** / 0.99 | 1.13 | 0.74 | 0/3、0/4 | 1、1 | 未过（A、B） |
| M-band | 0.24 / 0.20 | 1.15 | 0.61 | 0/3、0/4 | 1、0.998 | 未过（A、B） |
| M-env | 0.21 / 0.18 | 1.17 | 0.48 | 0/3、0/4 | 0.999、0.988 | 未过（A、B、C） |
| M-FC | 0.63 / 0.63 | 1.10 | 0.64 | 0/3、0/4 | 1、1 | 未过（A、B） |
| M-time | 0.23 / 0.63 | **2.05**（过） | **0.81** | 0/3、0/4 | 1、0.998 | 未过（A） |

- **Gate 汇总**：A 敏感性 12/12 未过（最高 M-PSD×G z=1.23）；B 特异性仅 M-time 2.047 过线（其余 1.10–1.50）；C 稳定性 11/12 过（M-env 0.476 差 0.024）；**D 单调性 12/12 全过（0 违反步，逐边占比 0.988–1.0）**。
- Spearman(D, 通道距离) 全部 ≈ −0.06～−0.08（弱负，照登）。

### 结论

- **如实阴性（预注册规则）**：全候选未过四门，**无 M\***；最佳可得 = **M-PSD（maximin 达成率 0.56：A 62%、B 56%、C 147%）**。按 implement.md 回滚点纪律 **B3 不启动**（无 M\* 不空转）。
- **Gate D 首个阳性事实**：12 对全部 |Δ| 聚合曲线 100% 单调不减（逐边占比 ≥0.988）——「仅改变 r」的 CRN 配对机制成立、度量读出对 r 良态（r\* 可定义性在度量读出层面成立）。
- 缺口定位：主缺口 **Gate A**（响应中位未过 2× null 地板；null split-half 口径已知偏严 ~2× 属保守方向照登——z 系统性低估约 2×）+ **Gate B 弥散复现**（自跨比 1.10–1.50，与 10-17-perturb-ec-f0 的 θ\* 跨通道弥散 1.27× 方向一致）；M-time 为 B/C 双最高（2.05 / 0.81）但 A 仅 32%。
- 后续路径待用户拍板：①接受阴性走 B4 收尾；②动用预注册允许的一次「档位级」调整（K 16→32 或窗数加倍，针对 Gate A 地板）后重跑；③其他。

## Run 002 — 阶段二：{M-RMS, M-PSD, M-time} 扫 6 范式候选 → 逐度量 P\*（2026-10-17；度量集 2026-10-18 用户拍板修订）

### 目的与假设

- 问题：三度量集下，6 个输入端虚拟扰动范式候选（G/N/O/P/L/B，各自 r 网格）哪个的响应读出最敏感/特异/稳定/单调？逐度量选出 P\*（结论限定「各自 r0 下」）。
- **度量集拍板（2026-10-18 用户）**：阶段一如实阴性无单 M\*；取 Gate A 最强 M-PSD（z 1.23）/ Gate B 唯一过线 M-time（自跨比 2.05）/ 现行基准 M-RMS 三度量继续——「单 M\*」预注册规则的显式偏离（design §6 登记）。
- 硬断言：同窗 eps/eta 张量跨阶段逐位相同（sha256 逐窗比对）；baseline 输出跨阶段 allclose rtol=2e-6 对照（逐位相等如实照登，不作门）。

### 配置

- 代码：同 Run 001（`models/perturb-search-v1/`）；θ\* 同 Run 001（同 checkpoint 同校验）。
- 范式候选 6 个（design §3 预注册）：G r0=0.25 / N r0=0.5 / O r0=1.0（DC 平移，外推警示照登）/ P r0=3.0（20 样本汉宁脉冲，中心 past 末端 −40）/ L r0=0.5（钳位损毁，代数形式实现）/ B r0=1.0（α 8–13 Hz Butterworth(3) sos+filtfilt 零相位分量增益）。**r 网格（Gate D 用，同 Run 001 口径）**：倍率梯 {0.25,0.5,1,2,4}×r0 钳制（G≤0.32、L≤1.0）去重 → G {0.0625,0.125,0.25,0.32}、N {0.125,0.25,0.5,1,2}、O {0.25,0.5,1,2,4}、P {0.75,1.5,3,6,12}、L {0.125,0.25,0.5,1}、B {0.25,0.5,1,2,4} = **28 格点/源**。
- 窗集/K/seed/CRN：同 Run 001（**同 win_idx 固定枚举 ⇒ 同窗 eps/eta 跨阶段逐位相同**）；批形状 **1737 = 1 + 62 × 28**（行 1+a·28+j），每窗一次 sample()。
- 度量：**{M-RMS, M-PSD, M-time}**（用户 2026-10-18 拍板，见目的段；config.PHASE2_METRICS）；每 (范式, 度量) 独立四门 A/B/C/D（A/B/C @ r0 切片、D 全网格）；**P\* 逐度量独立判定**，联合 {P×M} 终选以 3×6 矩阵呈报留用户拍板。
- 环境与数据：同 Run 001。
- 命令：`cd models/perturb-search-v1 && nohup python3 run_search.py phase2 > ../../outputs/perturb-search-f0/logs/phase2.log 2>&1 &`（度量集缺省取 config.PHASE2_METRICS）
- 产物：`outputs/perturb-search-f0/phase2-paradigm-search/run_01/`（逐格点 D/gateD_curve/r_grid + scorecard 3×6 + summary 含逐度量结论 + meta）。

### 结果

- **Run#1（2026-10-18）一次通过**：wall **9.9 min**（sample 251.0 s；度量 M-RMS 29.1 s / M-PSD 233.9 s / M-time 29.3 s；算子施加 B 最贵 4.8 s）；批 1737 行 × 24 窗 × K=16，一次 sample()/窗。
- **跨阶段硬断言全过**：24/24 窗 eps/eta sha256 与 phase1 逐窗相同；**baseline 输出 24/24 窗逐位相等（max_rel=0）**——跨批形（559→1737 行）实测 cuBLAS/mamba 核逐位一致，超出 rtol=2e-6 的门，如实照登为强复现事实。
- **3×6 scorecard（A/B/C @ r0 切片；D 全网格）核心数字**：

| 度量 | Gate A 过线（z） | Gate B 过线（自跨比） | Gate C | Gate D | 最佳可得（maximin） |
|---|---|---|---|---|---|
| M-RMS | O 5.68 | 无（N 最高 1.76） | 6/6 过 | 6/6 过（0 步） | L（0.58：A 1.55、B 1.17） |
| M-PSD | O 6.64、L 2.35 | 无（最高 1.09） | 6/6 过 | 6/6 过（0 步） | L（0.54：A 过、B 1.09） |
| M-time | O 2.06 | **N 2.166（唯一）** | 6/6 过 | 6/6 过（0 步） | O（0.60：A 过、B 1.20） |

- C 全 18 组合 ρ 0.56–0.80 过线；D 全 18 组合 0 违反步（O 逐边占比 0.93–0.98 略低、其余 ≥0.98，照登）；Spearman(D, 距离) 全部 −0.06～−0.11（弱负，照登）。

### 结论

- **三度量×六范式（30 组合含阶段一）无一四门全过，无 P\***；三度量均如实阴性（逐度量最佳可得 L/L/O，maximin 0.54–0.60），联合 {P×M} 终选不存在过线候选，如实呈报。
- **结构性发现一（幅度-特异性解耦）**：O（DC 偏置）把 Gate A 拉到 5.7–6.6（门 2 的 3 倍）而 Gate B 恰为 1.01–1.20——响应可以任意做强，但强度**均匀铺满全脑**，自/跨比不随幅度改善；Gate A 与 Gate B 的缺口不是同一件事。
- **结构性发现二（唯一特异性组合稳定复现）**：全矩阵唯一 Gate B 过线 = **N×M-time（2.166；阶段一同组合 2.047）**——特异性由「注入方差 × 时间形态读出」组合携带，换探针（G 1.61）或换度量（M-RMS 1.76）即掉线。
- **结构性发现三（Gate B 为绑定约束）**：两阶段 30 组合中除 N×M-time 外自跨比全部 0.98–1.76——θ\* 跨通道弥散**不因换范式、换度量、加强度而改变**，与 10-17-perturb-ec-f0 判据（弥散缺口在 θ\*）一致，进一步指向 Study I 损失侧迭代。
- Gate D 两阶段 30/30 全绿：|Δ| 单调性对全部范式×度量成立（r\* 可定义性无虞）。
- **任务级结论**：低成本搜索的收获 = ①{N×M-time} 为唯一特异性组合（后续 θ\* 迭代后的首选复检对象）；②O 的幅度-特异性解耦证据（「加强度」不是出路）；③全组合单调性成立。四门框架下本 θ\* 上不存在可用 {P\*×M\*}——EC 推断优化回到 Study I（θ\* 跨通道监督迭代）。
- 后续：B4 收尾（trellis-check + progress/context 同步 + 提交）待用户启动。
