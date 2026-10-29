# 实验日志 — <任务名>

> 每次「跑批/训练/评估/统计」追加一个 Run 块：跑前先写「目的+配置」，跑后同会话补「结果+结论」，不留空块。
> 格式契约见 `.trellis/spec/research/experiment-log.md`；产物只落本任务声明的变体目录（`.trellis/spec/research/path-isolation.md`）。

## Run 001 — <一句话名称>（YYYY-MM-DD）

### 目的与假设

- 要回答什么问题 / 验证什么假设？

### 配置

- 代码：`<变体目录/文件 + 版本快照>`
- 参数：<与默认不同的超参 / 阈值 / seed>
- 环境：<env 名 @ 机器（本地/远程），对照 .trellis/research/environment.md>
- 数据：<输入路径 + 筛选口径（qc_pass、分组等）>
- 命令：<可复现启动命令>

### 结果

- 指标：<数值 / 小表>
- 产物：`<本变体 outputs/... 路径>`
- 图表：<路径 + 一句话说明>

### 结论

- 支持 / 证伪 / 不确定 + 一句话依据
- 下一步 / 派生决策项：<同步到 .trellis/research/progress.md>

## Run 001 — fold0 同口径补全（2026-10-30）

### 配置

- `train_modma_0705.py --fold 0`（choices 放宽含 0，向后兼容）+ `ec_modma_symw_v4.py --arch std-mexhat --sigma-max 0.48 --n-grid 24`；参数与 fold1-4 逐字段同轨；产物 `outputs/10-29-modma-ec-te-corr/fold0/{run, ec_main}`

### 结果

- 训练 50ep 跑满，best val 1.0929@E44（fold 族 1.00–1.20 区间内）；EC 10/10、源失败 0、P(resp>τ)=0.500 逐例成立。

## Run 002 — 5 折 EC_pert × TE-EC 两级相关（2026-10-30）

### 配置

- EC_pert：fold0 = 本任务产物；fold1-4 = `outputs/10-29-modma-0705-f1234`（同口径）；TE = `refs/symbolic_transfer_entropy/modma5fold-...`（10 例/折，TE∩EC 全取 10）
- 协议：offdiag 16256；逐被试 Pearson/Spearman + p；组平均矩阵 Pearson/Spearman + p

### 结果

| fold | 逐被试 ρ (Spearman) | 逐被试 r (Pearson) | 组平均 ρ (p) | 组平均 r (p) |
|---|---|---|---|---|
| 0 | +0.3318±0.1544 | +0.3368 | +0.5656 (<1e-300) | +0.5766 (<1e-300) |
| 1 | +0.2691±0.2000 | +0.2542 | +0.6118 (<1e-300) | +0.6153 (<1e-300) |
| 2 | +0.3564±0.0840 | +0.3292 | +0.6070 (<1e-300) | +0.6148 (<1e-300) |
| 3 | +0.2365±0.1037 | +0.2379 | +0.4493 (<1e-300) | +0.4741 (<1e-300) |
| 4 | +0.2001±0.1779 | +0.1920 | +0.4060 (<1e-300) | +0.4108 (<1e-300) |

- 逐被试明细（50 行，含 p）落 `outputs/10-29-modma-ec-te-corr/ec_te_correlation.json`；p=0.00e+00 = 双精度下溢（p<1e-300）
- 不显著例（Spearman p>0.05）仅 2/50：fold0 s98（ρ=+0.015, p=0.051）、fold1 s96-15（ρ≈0.000, p=0.985）；显著负相关 2/50（fold1 s96-15 小幅、fold4 s92 小幅）

### 结论

- 5/5 折两级相关全部为正且组平均强（+0.41~+0.61）；TE 对照的组平均读数整体高于 GC 对照（+0.37~+0.61）同折水平——EC_pert 与符号化 TE 的一致性不弱于与谱 GC 的一致性。中间产物与相关 json 均在本任务 outputs，未触 result/（写入门遵守）。
