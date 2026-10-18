# design — N×M-time 完整 EC 推断（v1-fmt θ\*·fold0 test；只出矩阵版）

> 预注册文档：R_GRID / slope 段 / τ̃ / 聚合口径 / 窗集一经本文件 + experiment.md 首个 Run 块登记即钉死。三档拍板（2026-10-18 用户）：10 例全 test、9 点含原点网格、只出 EC 矩阵。

## 1. 总览与「不做」清单

**流程一句话**：冻结 θ\* → 逐窗 N 单范式批采样（497 行 = 1 + 62 源 × 8 采样格点，一次 sample()/窗，CRN）→ M-time 度量 → dbar 曲线（索引 0 ≡ 0）→ 逐 subject / global 聚合 → slope/AUC/r\* 三 readout → 契约 npz + 描述统计。

**不做**：GC/gPDC 对照与 AC6 收敛判定（用户拍板；契约 npz 为后续对照任务预留）；gates 复评；多种子/多 K；null 地板（无门无需）；~~诊断图 > 1 张~~ → **2026-10-18 用户请求指定交付两图**（EC 三面板 + 扰动波形四面板，experiment.md 图表行登记）。

## 2. 资产与装载（只读复用）

- **θ\***：同 perturb-search-f0（`outputs/gpfm-mamba-fmt-f0/v1fmt-main-01/checkpoints/gpfm-mamba-v1-fmt-best.pt`，EMA + hash 0dc0d5249c68 + gen_lowpass_hz=None 断言）。loader.py 从 perturb-search-v1 **复制适配**（窗枚举扩全 10 例；config 引用换本任务 `ec_config`——不复用 search_config，二者窗集/R_GRID 不同）。
- **N 算子常量逐位一致**（登记）：波形 g(t) = 固定 seed 2026、FFT 1–40 Hz 掩码、RMS 重归一 1、全局共享单波形、σ_A = 该窗源通道 past std——与 perturb-search-v1/config.py 登记值相同 ⇒ 同一波形实例（跨任务可比性根基）。
- **CRN 契约**：K=16、SEED=2026、逐窗 `manual_seed(SEED + win_idx)` 先 eps 后 eta；每窗一次 sample()，baseline 与 496 变体行同批共享（跨 r、跨源 CRN 逐位配对）；r=0 不在采样网格（索引 0 槽由恒等构造 D̄=0）。
- **窗集**：fold0 test manifest 顺序全 10 例 (10, 11, 12, 17, 24, 28, 30, 44, 61, 69) × 每例数据集顺序前 8 窗 = 80 窗；win_idx = 受试者序×8 + 例内窗序（0..79）；通道序 = npz 列序（同源对齐不变）。

## 3. 输入端：N 单范式 + 9 点网格 + 批构造

- **N**：x_A ← x_A + r·σ_A·g(t)（本任务自实现，读自身 config；算子契约同前：只动源通道 / r=0 短路 clone / 新张量）。
- **R_GRID（9 点含原点，预注册钉死）**：`(0, 0.03125, 0.0625, 0.125, 0.25, 0.5, 0.75, 1.0, 1.5)`——低段 1/32–1/4 加密（slope 段覆盖），高段至 1.5（≈3×r0，外推警示照登）；**采样点 = R_GRID[1:]（8 点）**；R_MAX = 1.5；**R_NEXT = R_GRID[-1] + (R_GRID[-1] − R_GRID[-2]) = 2.0**（线性外推推导，禁字面量——删失编码档）。
- **slope 段**：SLOPE_SEG = R_GRID[:5] = {0, 0.03125, 0.0625, 0.125, 0.25}（5 点含原点锚）；敏感性段 SLOPE_SEG_SENS = R_GRID[:4] 照登。
- **批构造**：行 0 = baseline（逐位复制）；行 1 + a·8 + j = 源 a × 采样格点 j（源主序、r 次序）；批 497 行，每窗一次 sample() 消费。

## 4. 输出端：M-time + dbar 曲线 + 聚合

- **M-time（主标量同搜索任务钉死口径）**：δ̄(t) = mean_k δ_k(t)（CRN 配对差分集群均值）→ a(t) = |δ̄(t)| → D = 前 32 样本（128 ms）均值；mtime_full / mtime_peak 作随附登记（不入 readout）。
- **dbar 曲线**：逐窗 D → dbar_w (62 源, 9 格点含 0 槽, 62 靶)，索引 0 槽 = 0（恒等构造）。
- **聚合（受试者级 = 窗 median 惯例）**：逐 subject dbar_s = 该例 8 窗逐元素 median（10 份）；global dbar_g = 全 80 窗逐元素 median。

## 5. Readout 三件套 + τ̃ + 描述统计

沿 perturb-ec-v1/readout.py 结构**自实现读自身 config**（evaluation-calibers Study II 节：主 = 低剂量斜率，天然大=强免翻转）：

- **slope（主）**：SLOPE_SEG 上 OLS（含截距），dbar (…,9) → (…)。
- **AUC（次）**：全网格梯形面积 / R_MAX。
- **D̃ 曲线**：D̃_{A→B}(r) = D̄_{A→B}(r) / D̄_{B→B}(r)（**分母 = 靶通道自响应**，τ v2 相对自响应口径，用户 10-17 拍板沿用）；r=0 槽置 0；A=B 处 D̃≡1（r>0 槽，仅约定不进判定）；分母零断言。判读纪律：D̃ 数值不作跨靶比较。
- **r\***（降级）：r\*_{A→B} = min{r ∈ R_GRID : D̃_{A→B}(r) ≥ τ̃}，τ̃ = **0.25**（主）+ 0.50（敏感性照登）；网格内未达 → 右删失编码 R_NEXT；**存盘前 flip**（strength = R_NEXT − r\*，删失 → 0 最弱）。
- **有效路径集**：{(A,B): A≠B, r\* ≤ R_MAX}。
- **契约 npz**：{ec (62,62), readout, subject_ids（逐 subject id 或 'global'）, orientation='larger_is_stronger'}——33 个矩阵文件（3 readout × (10 subject + global)）。
- **描述统计（无对照，六项）**：① slope EC 自跨比（median 对角 / median offdiag）② r\* 删失率（τ̃=0.25 主 + 0.5）③ 有效路径数 ④ Spearman(slope EC, 通道距离)（offdiag）⑤ 跨受试者 split-half（预注册指派：例 #1–#5 vs #6–#10 的 slope EC offdiag 秩相关）⑥ dbar 单调占比（offdiag 逐边曲线非减占比，Gate D 同构 sanity）。

## 6. 执行序

B0 骨架 + 单测（N 算子契约/批构造 497/CRN/d_time 解析值/readout 合成数据全套/loader 口径 + 窗枚举 0..79/输出级 r=0 恒等）→ B1 smoke（1 窗）→ B2 主 Run（80 窗 + readout + 落盘 + summary）→ B3 收尾（trellis-check / 提交，待后续）。

## 7. 产物布局

`outputs/ec-nmtime-f0/run_01/`：`dbar_curves.npz`（10+global dbar + 逐窗 eps/eta sha256 + 窗登记 + meta）、`ec_{readout}_s{sid}.npz` ×10 ×3 readout + `ec_{readout}_global.npz` ×3、`summary.md`（描述统计六项表）、`logs/`（nohup）。代码 `models/ec-nmtime-v1/`（config/loader/candidates（N 单算子+批构造+crn）/metrics（d_time）/readout/run + tests/）。

## 8. 风险与已知边界

- **响应地板（已知）**：N×M-time 的 Gate A z = 0.63（未过 2×地板）⇒ 逐窗 D 信噪弱——靠 subject 级 8 窗 median + CRN 配对压噪声；slope 读出对低信噪敏感，结果判读随附此警示。
- **r=0 槽外推**：索引 0 的 D̄=0 是恒等构造（非采样），slope 含原点锚依赖该构造成立（r=0 逐位恒等已由单测锁定）。
- **高格点外推**：r ≥ 1.0（2×r0）超出 N 的标定区，AUC/r\* 读出随附外推警示。
- **删失预期高**：θ\* 弥散 + τ̃ 相对口径下多数边或不到 0.25×靶自响应——删失率如实登记，不作为失败判据（本任务无门）。
- **前置验证级 θ\***：产物表述「v1-fmt θ\* 上的 EC_pert」。
