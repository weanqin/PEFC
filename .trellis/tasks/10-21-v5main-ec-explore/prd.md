# v5-main 双方法 EC 推断探索（噪声→熵失稳 σ* + 常量偏移→均值响应）

## Goal

用户 2026-10-21 直给规格的探索性 run：在 gpfm-mamba-v5-main-01（v4 数据 fold0 直训模型，best 0.73155@ep64）的 **test 集（fold0 固化划分 10 例：12/17/18/24/30/38/44/55/58/61）**上执行两种扰动→EC 读出，并与 v4 GC refs（`refs/granger_bivariate_spectral/edrav4-46-resample250-cont-zscore-fs250-fixed8-f1-125-nf65/`，任务 10-21-oldlib-gc-edrav4）逐被试计算相关性。**同数据源配对（v4 模型 × v4 观测参考）**。

## Requirements

### R1 方法1 — 噪声注入 → 靶通道熵失稳 → σ* → EC1（用户给定规格；**已宣告失败并取消，2026-10-21 用户**）

> **状态：FAILED**。实测（s12/s17/s18/s24 四例）offdiag 删失率均 **1.000**——全部 3782 边在 σ 网格上限 0.48 仍未触发 20% 熵涨幅，EC1 恒定 → 与 refs 相关不可定义。结构原因 = 靶通道预测方差对源噪声不敏感（σ=0.48 时 ln-方差膨胀中位 ~0.014 / p95 ~0.086 / max ~0.24，远低于阈值 ln(1.44)=0.365）。用户拍板取消后续计算并宣告方法失败；已算 4 例 npz（含 `logvar_infl` 62源×24σ×62靶 诊断量）作为失败证据随档，不再扩展。

- 原始规格（存档）：注入位置 = 模型输入窗（z 空间）源通道全部 384 past 采样点；σ·ε 无偏噪声 CRN（seed=2026+sid）；σ 网格 logspace(0.05, 0.48, 24)；K=1；50 窗/例等间隔；熵 = 高斯拟合微分熵（用户确认），「涨幅超 20%」⇔ mean_w ln Var 增幅 > ln(1.44)；EC1[s,t] = -(ln σ* + ln 2πe) + 1。

### R2 方法2 — 常量偏移 → 均值响应 → EC2（用户给定规格）

- 同一位置源通道加常量 c = 0.5×该通道全部输入窗实际 std（z 空间 ≈0.5，逐被试逐源经验值）；K=1；**全量窗**（1321/例）。
- **EC2[s,t] = 靶通道预测均值变化量**（全窗×128 时间点均值差，带符号=方向；eps/eta 与 baseline 共享 CRN，seed=2026+100000+sid）。

### R3 采样与实现口径（沿 evaluate 正典 + 已向用户简报确认）

- EMA shadow 权重（Run 006 口径）、Euler ODE 16 步、GPR 条件先验、终点 40 Hz 砖墙低通（gpfm 产物正典口径）；显式 eps/eta 全确定化；cudnn deterministic。
- 扁平批分行分块（cap=1000 行，避开 brickwall reflect-pad 核 65535 行 grid 上限——evaluate 标准批 256 不触及，本探索大扁批必须分块，实测 1200×62 崩）。
- 逐被试 npz 可续跑（`m1/subject_s{sid}.npz`、`m2/subject_s{sid}.npz`）。

### R4 对照口径

- EC1/EC2 均为**行=源/列=靶**，与 ec[a,b]=a→b **同向**：aligned = 直接对照 ec（非 A_eff 的 ec.T 转置口径）；flipped = 对照 ec.T（方向性检查）。
- 逐被试（10 例）：**Spearman 主读出**（方法1 有 24 级格点 + 删失 ties）+ Pearson 次要；组均值（10 例平均 EC vs 平均 GC）单次相关；方法1 删失率随档登记。
- 范围边界：探索性 run，不做 split-half 天花板、不做与其它 EC 路径的横向对比交付；test 10 例 ⊂ refs 46 例，无跨数据源 caveat。

### R5 运行纪律

- 产物只落 `outputs/10-21-v5main-ec-explore/`（脚本 + README 口径登记 + 逐被试 npz + ec_explore_stats.json）；ckpt/refs/v4 数据集只读；不写入 `outputs/gpfm-mamba-f0/`（他任务产物根）。
- experiment.md Run 块：跑前目的+配置、跑后同会话结果+结论。

## Acceptance Criteria

- [ ] AC1 smoke（s12）双方法跑通：setup 断言过（config_hash/划分/窗数 1321），npz schema 齐全（m1: ec1/sigma_star/censored/logvar_infl/sigma_grid；m2: ec2）
- [ ] AC2 方法2 全量 10 例 npz 落盘（断点续跑验证）；方法1 按 4 例证据在档封存（用户宣告失败，不再扩展）
- [ ] AC3 `ec_explore_stats.json`：方法2 逐被试 aligned/flipped × Spearman/Pearson + 组级 mean±sd + 组均值单次；方法1 失败宣告段（状态 + 逐例删失率证据）
- [ ] AC4 README.md 口径登记与 experiment.md Run 块完整（含方法1 失败的如实登记与解读）；结果回填研究记忆（视结论）
- [ ] AC5 零写入他任务目录；`gen_ec_explore.py` 双遮蔽断言（config/model 命中 v5）在位

## 子研究路由（1.0b / 1.0c）

<!-- trellis-research-routing -->

- **分类结果**：`study2` —— 扰动→响应 EC 推断探索（perturb-ec-pipeline 规范域内）
- **依据**：用户直给方法规格的探索读出（熵失稳阈值 σ* / 均值偏移响应），服务 EC 推断算法线
- **新路径/方法检测（1.0c）**：`已有路径：Study II 扰动→EC 推断（spec/research/perturb-ec-pipeline.md 规范域）` —— 本任务为同域探索读出变体（用户直给规格，无需适配调研）
- **research 产物**：`无`
