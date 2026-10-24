# 实验日志 — 10-24-exitgnn-ec-viz

> 每次「跑批/训练/评估/统计」追加一个 Run 块：跑前先写「目的+配置」，跑后同会话补「结果+结论」，不留空块。
> 格式契约见 `.trellis/spec/research/experiment-log.md`；产物只落本任务声明的变体目录（`.trellis/spec/research/path-isolation.md`）。

## Run 001 — EC_pert v4 三图（nature-figure 契约 + QA 门）（2026-10-24）

### 目的与假设

- 用户指令「可视化EC_pert」：把归档任务 10-24-exitgnn-ec-unbiased 的 EC_pert v4
  （8 例 128×128）与双 GC 参考做成可判读三图（个体结构 / GC 结构对照 / 组间形态）。

### 配置

- 代码：`models/latent-flow-v1/viz_ec_pert.py`（变体层单文件，Python matplotlib 已存
  偏好后端独占；rcParams 按 nature-figure quick-start：7pt 基准、Arial/DejaVu、
  pdf.fonttype 42）
- 数据只读：`outputs/10-24-exitgnn-ec-unbiased/recompute_v4/subject_{0..7}.json` +
  refs 双档（v1 fixed8 / v2 AIC4-25）各 8 例折叠；组均值 ρ 图面注记与 Run 003 在档值
  assert 对拍（±5e-3）
- 色标契约：热图共享稳健色标（offdiag P1–P99，对角自响应钳 σ_min 饱和注记）；三联图
  量纲不同各自标度显式注明；差异面板 RdBu_r 0 居中（±P99|diff|）
- QA：validate_figure.py（源码 preflight）+ require_matplotlib_panel_alignment
  （render-time 门，strict 1.5pt）+ audit_figure_collisions.py（PDF 逐图）+ documents
  visual-judge 单轮视觉验收
- 命令：`/root/miniconda3/bin/python models/latent-flow-v1/viz_ec_pert.py`

### 结果

- 三图落 `outputs/10-24-exitgnn-ec-viz/`（PNG 300dpi + PDF + alignment.json/svg +
  collision-audit.json 伴档）：
  - `ec_subjects_v4.png` — 2×4 逐被试（行 1 MDD s1,5,7,10 / 行 2 HC s21,25,27,30），
    共享色标 offdiag P1–P99 = 0.03–3.90；对角亮带（自响应钳边）与列条纹（靶主效应）可辨；
    visual-judge **pass**
  - `ec_group_vs_gc.png` — a EC_pert 组均值 × b GC v1（ρ=+0.19）× c GC v2（ρ=+0.11）
    三联各自标度；EC 列条纹 vs GC 空间结构差异直观；初版 2 collision FAIL（面板字母
    text 与长标题 bbox 重叠）→ 字母并入标题（mathtext 粗体）修复后 0 fail；
    visual-judge **pass**
  - `ec_mdd_hc.png` — a/b MDD/HC 均值共享标度（形态高度相似）+ c MDD−HC 发散色
    （±1.90）；visual-judge 初判 fail（c 呈强列带结构，与「接近零色」预期不符）
- **图 3 裁定（数据核实，非渲染缺陷）**：offdiag |diff| 中位 0.251（offdiag conn 中位
  ~0.49 的一半）但**跨组配对 Spearman 0.657 ≈ 组内 0.652**（MDD/HC 个体结构可互换）；
  逐靶列均值差 |z|（vs 组内 SE）中位 0.52 / P90 1.40 / max 3.82@E115——列带结构 =
  n=4/组靶灵敏度抽样噪声（与方差分解靶主效应 0.907 同源），非组间信号；诚实判读注记
  已入图 3 suptitle，重生成后 alignment + collision 双过
- 全部 QA：preflight 18 pass/0 fail；alignment 三图 strict PASS；collision 三图
  0 fail/0 warn。

### 结论

- 三图支撑归档任务阴性读数的视觉判读：EC_pert 结构 = 靶主效应主导的列灵敏度（个体差异
  在强度非拓扑，跨组无分离），与两套 GC 参考结构均不对应——图面结论与 Run 003 数字
  一致，无新事实（研究记忆不回写）。
- 派生登记：n=4/组差异面板的判读陷阱（强列带 ≠ 组间信号，须 z/配对 ρ 裁定）随本 Run
  在档；EC provisional 状态随 exit-gnn θ 判断拍板联动（不变）。
