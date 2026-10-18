# 实验日志 — N×M-time 完整 EC 推断（v1-fmt θ\*·fold0 test；只出矩阵版）

> 每次「跑批/训练/评估/统计」追加一个 Run 块：跑前先写「目的+配置」，跑后同会话补「结果+结论」，不留空块。
> 格式契约见 `.trellis/spec/research/experiment-log.md`；产物只落本任务声明的变体目录（`.trellis/spec/research/path-isolation.md`）。

## Run 001 — N×M-time 剂量-响应 80 窗主 Run + 三 readout EC 矩阵（2026-10-18）

### 目的与假设

- 问题：用搜索任务唯一 Gate B 过线组合 N×M-time 在冻结 v1-fmt θ\* 上生产完整 EC_pert 矩阵（slope 主 / AUC 次 / r\* 删失降级），并给出六项描述统计。**不做 GC 对照与收敛判定**（用户 2026-10-18 拍板「只出 EC 矩阵」）。
- 已知前情（登记）：N×M-time Gate A z=0.63（弱信噪，slope 读出随附此警示）；Gate D 全网格单调（r\* 可定义）；自跨比 2.17（置换 p<0.0002）。

### 配置

- 代码：`models/ec-nmtime-v1/`（config 唯一配置源 = design §2–§5 字面转录；loader 自 perturb-search-v1 复制适配）；git 快照见 meta。
- θ\*（只读冻结）：`outputs/gpfm-mamba-fmt-f0/v1fmt-main-01/checkpoints/gpfm-mamba-v1-fmt-best.pt`（EMA，epoch 81，config_hash `0dc0d5249c68` 严格校验 + `sampler.gen_lowpass_hz is None` 断言）。
- 输入端：N 单范式（加性带限噪声，波形常量与 perturb-search-v1 登记逐位一致：seed 2026 / 1–40 Hz / RMS=1 / 全局共享）；**R_GRID = (0, 0.03125, 0.0625, 0.125, 0.25, 0.5, 0.75, 1.0, 1.5)**，索引 0 ≡ r=0 ≡ D̄=0 恒等构造不采样，采样点 8 个，批 **497 = 1 + 62×8** 行/窗，行 1 + a·8 + j。
- 输出端：M-time 主标量（|δ̄| 前 32 样本均值；full/peak 随附）。
- 窗集：fold0 test manifest 顺序全 10 例 **(10, 11, 12, 17, 24, 28, 30, 44, 61, 69)** × 每例前 8 窗 = **80 窗**；win_idx = 受试者序×8 + 例内窗序（0..79）。
- CRN：K=16，seed=2026（逐窗 `manual_seed(2026+win_idx)` 先 eps 后 eta），每窗一次 sample()、baseline 与 496 变体行同批。
- 聚合：逐 subject 8 窗 median（10 份）+ global 80 窗 median；dbar (62, 9, 62) 索引 [A, r, B]。
- Readout：slope = SLOPE_SEG {0, 0.03125, 0.0625, 0.125, 0.25} OLS（主）；AUC = 梯形/R_MAX；r\* = min{r: D̃ ≥ 0.25}（D̃ = D̄_{A→B}/D̄_{B→B} 靶自响应分母；敏感性 0.50 照登），右删失编码 R_NEXT=2.0（线性外推推导），存盘前 flip（大=强）。
- 描述统计六项：slope 自跨比 / 删失率（0.25 主+0.50）/ 有效路径数 / Spearman(slope, 距离) / 跨受试者 split-half（例 #1–#5 vs #6–#10）/ dbar 单调占比。
- 环境：base @ 本地服务器（RTX 4080S、BLAS 护栏 8 线程；environment.md 契约）。
- 数据：`/root/autodl-tmp/edra_processed_fmt`；splits manifest `models/gpfm-mamba-v1-fmt/splits/fold0_status_ok_seed42_val4.json`（只读前置断言）。
- 命令：`cd models/ec-nmtime-v1 && nohup python3 run.py smoke|main > ../../outputs/ec-nmtime-f0/logs/….log 2>&1 &`
- 产物：`outputs/ec-nmtime-f0/run_01/`（dbar_curves.npz + ec_{slope|auc|rstar}_{s{id}|global}.npz ×33 + summary.md）。

### 结果

- **执行史（如实登记）**：主 Run 2 次调用。Run#1 采样完成后崩于 readout 轴序 bug——dbar9 原生轴序 [A, r, B] 未转 r 末维即喂 slope/AUC（perturb-ec-v1 契约 r 末维），被 `save_ec_pert_npz` 形状断言拦截；修复 = 调用点 `transpose(0, 2, 1)`（`dtil_curves` 输入原生 [A, r, B] 不动）。Run#2 完成。
- **Run#2**：wall **4.9 min**（sample 累计 221.4 s）；80 窗 × 497 行 × K=16；dbar (80, 62, 9, 62)——0 槽逐位 0（恒等构造）、80 个唯一 eps sha256、10 受试者 × 8 窗。
- **产物**（`outputs/ec-nmtime-f0/run_01/`，36 文件）：`dbar_curves.npz`（80 窗 dbar + full 随附 + sha + 窗登记）；**33 个契约 npz**（slope/AUC/r\* × 10 subject + global；字段 = ec (62,62) / readout / subject_ids / orientation='larger_is_stronger'，rstar 已 flip）；summary.md；meta.json。
- **图表（2026-10-18 用户请求补出；`models/ec-nmtime-v1/visualize.py`，nature-figure 契约链）**：
  - `figs/ec_nmtime_ec.{pdf,png,svg,tiff}`——(a) slope EC global 热图；(b) r\*-strength EC 热图（τ̃=0.25，删失 535 格浅灰=0）；(c) 跨受试者 split-half 散点（ρ=0.904，含恒等线）。
  - `figs/ec_nmtime_waveforms.{pdf,png,svg,tiff}`——(a) past 段源通道注入足迹（win0 sid=10, A=ch36, r=0.5；同 CRN 整批复现 ⇒ 与主 Run 逐位一致）；(b) future 段自响应（K=16 均值±SD）；(c) |δ̄(t)| 读出 + M-time 128 ms 窗；(d) future 段跨响应（B=ch2，A 的最强出边靶）。数据快照 `figs/waveforms_win0_a36_b2_r0.5.npz`。
  - `figs/ec_nmtime_waveforms_r2.{pdf,png,svg,tiff}`（**2026-10-18 用户请求**；r=2.0 = 网格外外推剂量 4×r0，2 行批新增采样——行独立性由 r=0 同批恒等 + 跨批形逐位两证担保）：同四面板结构，panel a 标注 extrapolated；快照 `waveforms_win0_a36_b2_r2.npz`。单窗读出：A 前 32 样本 |δ̄| 0.332→1.104（×3.3）、B 0.218→0.575（×2.6），自/跨比 1.5×→1.9×（次线性饱和 + 高剂量特异性略升，与搜索任务 B(r) 扫描 2.17→2.23 方向一致）。QA 同链：0 FAIL / 2 条图例白框 WARN（受控覆盖留档）/ visual-judge pass。
  - QA：静态预检 20 过/1 WARN（受试者拆分切片的启发式误报，预注册指派留档）；面板对齐 strict 双过；PDF 文字审计双过；碰撞审计 0 FAIL（1 条图例白框边缘 WARN=受控覆盖，留档）；visual-judge 双 pass。
- **描述统计六项（global 为主）**：

| # | 指标 | 数值 |
|---|---|---|
| 1 | slope EC 自跨比 | **2.109**（subject 范围 1.694–2.507；搜索任务 @r0 D 值 2.17） |
| 2 | r\* 删失率（offdiag） | τ̃=0.25 → **14.15%**；τ̃=0.50 → 49.79% |
| 3 | 有效路径数 | 3247 / 3782（85.9%） |
| 4 | Spearman(slope EC, 通道距离) | −0.1057（弱距离衰减，与前任务 −0.05~−0.11 同款） |
| 5 | 跨受试者 split-half（#1–#5 vs #6–#10） | **ρ = 0.9040** |
| 6 | dbar 单调占比（offdiag 逐边非减） | 1.0000（9 点细网格） |

### 结论

- **完整 EC_pert 矩阵产出成功**：N×M-time × v1-fmt θ\* × fold0 test 全 10 例 + global，三 readout（slope 主 / AUC 次 / r\* flip 删失标注），契约接口就位——本任务交付面（用户拍板「只出矩阵」）完成。
- **特异性在 slope 读出上保持**：自跨比 2.109 ≈ 搜索任务的 2.17——N×M-time 的特异性不是 r0 单点巧合，逐 subject 均过 2 的边缘（1.694–2.507，10 例中 8 例 ≥2）。
- **稳定性突出**：split-half ρ 0.9040，与双变量 GC 观测参考的天花板（0.9243±0.0447，10-17-bivar 任务）同量级——矩阵边间排序跨受试者高度可复现。
- 85.9% 边在 τ̃=0.25 下可达阈；剂量-响应 100% 单调（细网格复证 Gate D）。
- **边界照登**：弱信噪（Gate A z=0.63）下 slope 对低信噪敏感；r ≥ 1.0 外推；无 GC 对照——「N×M-time 是否改变 10-17『EC 推断不可用』定论」仍开放，契约 npz 已备好比较器接口，后续任务可直接接入判定。
