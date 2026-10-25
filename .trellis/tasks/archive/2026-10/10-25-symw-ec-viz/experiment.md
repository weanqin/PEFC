# 实验日志 — symw EC_pert × GC-EC 并排可视化（10-25-symw-ec-viz）

> 每次「跑批/训练/评估/统计」追加一个 Run 块：跑前先写「目的+配置」，跑后同会话补「结果+结论」，不留空块。
> 格式契约见 `.trellis/spec/research/experiment-log.md`；产物只落本任务声明的变体目录（`.trellis/spec/research/path-isolation.md`）。

## Run 000 — 脚本 + 单被试冒烟（2026-10-25）

### 目的与假设

- 写出 `viz_symw_ec_pairs.py` 并以 `--subjects 1` 冒烟：验证数据装载、秩归一、对角
  掩蔽、ρ 注记对拍、1×2 等宽布局 alignment gate、PNG(300dpi)+PDF 落盘全链路可用。

### 配置

- 代码：`models/latent-flow-v1/viz_symw_ec_pairs.py`（新文件，首版；后端 python
  matplotlib Agg，纯 CPU）
- 参数：`--subjects 1`（子集冒烟，组级两图仅全量渲染）；秩归一 = offdiag
  rankdata/(N−1)；cmap viridis；个体/组 rank 图共享色标 [0,1]，组 abs 图各测
  vmax = pooled offdiag P99.5
- 环境：npi 本机 CPU（OMP_NUM_THREADS=8 前缀 + 脚本 BLAS 守卫；GPU 禁占——并行
  训练会话在跑）
- 数据（只读）：`outputs/10-25-symw-ec-unbiased/recompute_v4/subject_{0..7}.json`
  connectivity_matrix（conn v4）+ `refs/granger_bivariate_spectral/
  mild40-trial500-zscore-fs250-fixed8-f1-45-favg/refs_s{sid}.npz` ec（GC v1）+
  同目录 `gccorr_dual_summary.json` per-subject spearman_raw_aligned + 
  `/root/autodl-tmp/Mild Depression/geometry_mild.pt` channel_names（E1–E128）
- 命令：`cd /root/npi && OMP_NUM_THREADS=8 /root/miniconda3/bin/python
  models/latent-flow-v1/viz_symw_ec_pairs.py --subjects 1`

### 结果

- py_compile 过；`--subjects 1` 冒烟 3.7s 完成：
  `subject_pairs/subj_s1_MDD.{png(300dpi,260KB),pdf}` +
  `qa/subj_s1_MDD.alignment.json`（verdict **PASS**，1 组行比较，fail 0 / warn 0；
  面板 a/b 各 246.7×246.7pt 等宽等高）。
- 图面核对（快速 sanity，非正式 visual-judge）：两等大方形面板、对角白+细纹掩蔽、
  E1–E113 稀疏刻度、target/source 轴标、共享 colorbar、ρ 角注均在，无破损渲染。
- 修复回环 1 次：首版 qa/ JSON 误落 subject_pairs/qa/（_save 以图目录为根）——改为
  显式 qa_dir 参数，QA 统一落 `outputs/10-25-symw-ec-viz/qa/`；误产物已删重渲。
- ρ 对拍：s1 json 值 +0.009945 与本图两矩阵 offdiag Spearman 逐位一致（assert <1e-9 过）。

### 结论

- 脚本链路（装载→秩归一→掩蔽→对拍→alignment gate→PNG+PDF）可用；进入全量渲染
  Run 001。

## Run 001 — 全量渲染（8 个体 + 2 组级）（2026-10-25）

### 目的与假设

- 全量渲染 10 组图（8 张个体并排 + group_means_rank + group_means_abs），
  PNG(300dpi)+PDF 齐落 `outputs/10-25-symw-ec-viz/`，alignment gate 逐图过。

### 配置

- 代码：同 Run 000（qa_dir 修复版）
- 参数：全量默认（无 --subjects）；其余同 Run 000
- 环境：同 Run 000（npi 本机 CPU）
- 数据：同 Run 000（全 8 例 + 全部组级）
- 命令：`cd /root/npi && OMP_NUM_THREADS=8 /root/miniconda3/bin/python
  models/latent-flow-v1/viz_symw_ec_pairs.py`

### 结果

- 19.0s 渲染 10 组图全部落档（PNG 300dpi ≈260–290KB + PDF ≈84–94KB 各一份）：
  - 个体 ×8：`subject_pairs/subj_s{1,5,7,10}_{MDD}.{png,pdf}` +
    `subj_s{21,25,27,30}_{HC}.{png,pdf}`
  - 组级 ×2：`group_means_rank.{png,pdf}` + `group_means_abs.{png,pdf}`
- 逐图 alignment gate（render-time，容差 1.5pt）：**10/10 PASS**（fail=0 warn=0，
  JSON 随档 `qa/*.alignment.json`）。
- ρ 注记对拍 8/8 过（json 值 vs 本图矩阵重算，<1e-9）。
- 数据侧备忘（图面真实值域）：
  - EC_pert(v4) pooled offdiag [0.029, 3.901]，P99.5=3.901（=σ-clamp 上界；8 例
    min/max 逐位相同，均来自 clamp 两端）；组均值矩阵 [0.029, 3.901]
  - GC-EC(v1) pooled offdiag [0.00017, 0.322]，P99.5=0.0999；组均值矩阵 [0.000, 0.135]
  - 个体 GC 值域：s1 [0,0.288] s5 [0,0.184] s7 [0,0.283] s10 [0,0.199] s21 [0,0.215]
    s25 [0,0.187] s27 [0,0.261] s30 [0,0.322]
  - 个体 ρ(aligned)：s1 +0.010 / s5 +0.305 / s7 +0.066 / s10 +0.259 / s21 +0.127 /
    s25 **−0.135（最负）** / s27 +0.330 / s30 +0.112；组 ρ̄ +0.134 ± 0.159

### 结论

- 全量交付面齐备；进入 Run 002 三门 QA（validate / alignment 汇总 / collision）。

## Run 002 — 自动化 QA 三门（2026-10-25）

### 目的与假设

- 三门 QA 全绿交付前门：①`validate_figure.py` 源码 preflight；②面板对齐门
  （Run 001 已逐图 render-time PASS，JSON 随档）；③逐 PDF `audit_figure_collisions`
  ——FIX BEFORE DELIVERY 必须清零；不过即修复回环重渲。

### 配置

- 代码：同 Run 001（qa_dir 修复版）
- 命令：
  - `python /root/npi/.zcode/skills/nature-figure/scripts/validate_figure.py
    models/latent-flow-v1/viz_symw_ec_pairs.py`
  - `python /root/npi/.zcode/skills/nature-figure/scripts/audit_figure_collisions.py
    <fig>.pdf --json-out qa/<fig>.collision-audit.json --overlay-pdf
    qa/<fig>.collision-audit.pdf`（10 PDF 逐图）
- 环境：同前（npi 本机 CPU）

### 结果

- **门 1 validate_figure（源码 preflight）**：17 pass / 0 fail / 4 warn，verdict READY
  FOR VISUAL QA。4 个 WARN 逐条理由在档：
  1. EXPORT-RASTER（无 TIFF）：任务契约交付面 = PNG+PDF（PRD R2/R3），TIFF 为期刊
     投稿件格式，本研究内部交付不需要；
  2. RASTER-DPI 300<600 默认：PRD 明确 PNG 300dpi；
  3. FINAL-WIDTH 218mm：内部研究对比图非期刊版面件，8.6in 宽为 5.5pt 刻度可读性所需；
  4. ANNOTATION-WORKAROUND（ρ 角注白 bbox）：bbox 覆盖对象是热图栅格（无曲线/误差
     棒可切），collision 审计判定 contained（informational），非隐藏通路。
- **门 2 面板对齐（render-time，1.5pt）**：修复版重渲后 **10/10 PASS**（fail=0
  warn=0；`qa/*.alignment.json` + `.alignment.svg` 随档）。
- **门 3 collision（逐 PDF）**：首轮 **10/10 FIX BEFORE DELIVERY**（text-stroke
  FAIL：suptitle/刻度/面板题被多条 stroke 穿越判定）→ 根因定位：对角掩蔽初版用
  patch+hatch（128 白矩形/面板）——hatch 线以未裁剪平铺几何进 PDF（PyMuPDF 读到
  跨页面 bbox），且批量矩形 fill 合并为整对角带 bbox（/tmp 双盲实验证实：有 hatch
  即 FAIL、无 hatch 即 PASS）。修复 = 改用单 imshow 栅格纹理叠加（白+细纹，RGBA
  1024²，fill-only 零 stroke path），视觉等价。修复版重渲重审：**10/10 PASS**
  （0 fail 0 warn；个体/rank 图各 1 个 contained fill overlay = ρ 角注白 bbox，
  intentional；overlay PDF 仅在有 findings 时生成，全绿即无）。
- 修复回环 2 处累计：Run 000 qa 落点（qa_dir 参数）+ Run 002 hatch→raster overlay。
- 渲染核对（快速 sanity，正式 visual-judge 由主会话执行）：三类图（个体 rank 对、
  组 rank、组 abs）面板规格/色标/注记/掩蔽均确认无破损。

### 结论

- 三门全绿（validate 0 fail、alignment 10/10 PASS、collision 10/10 PASS），修复
  过程与 WARN 理由在档；交付面就绪，待主会话 visual-judge 终审。

