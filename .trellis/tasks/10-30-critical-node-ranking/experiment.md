# 实验日志 — 10-30-critical-node-ranking

> 每次「跑批/训练/评估/统计分析」追加一个 Run 块：跑前先写「目的+配置」，跑后同会话补「结果+结论」，不留空块。
> 格式契约见 `.trellis/spec/research/experiment-log.md`；产物只落本任务声明的变体目录（`.trellis/spec/research/path-isolation.md`）。

## Run 001 — S2 冒烟：守卫负例 + 双线 fold0×1 被试全 128 源（2026-10-30）

### 目的与假设

- 验证三件事：① ckpt SHA/fold 锁定守卫对「fold1 ckpt 冒充 fold0」负例拒跑（PRD R4/AC2）；
  ② main 模式全链路可跑通——σ=0 结构性恒等断言过（CRN 契约免费检查点，AC6 前提）、
  单调性违例数如实可读（预期 0）、逐被试 json 字段齐（design §4）；
  ③ mild 线守卫链（ckpt SHA + manifest SHA + norm SHA 按 fold）同过。
- 假设：独占 GPU 下单被试（128 源 × 3σ + σ=0 检查 ≈ 385 次前向）耗时 ~1 min 量级，
  可外推五折全量预算 ≤1.5 h（design §5）。

### 配置

- 代码：`models/latent-flow-v1/critical_node_ranking.py`（本任务唯一新增文件，git e607716 工作区）
- 参数：`--mode main --fold 0 --subjects 1`（其余默认：σ={0.3,0.5,0.7}、seed 42、
  n_steps 10、batch 32、bf16 autocast 每源单域含 σ=0；modma 32 trial 子采样、mild 30 trial 全量窗 500）
- 环境：conda base python3 @ 本地云服务器（RTX 4080S 独占，对照 environment.md）
- 数据：modma = `/root/autodl-tmp/modma`（env `MODMA_DATA_ROOT`）；mild = `/root/autodl-tmp/Mild Depression`；
  ckpt 只读 = `result/checkpoints/` 注册件（modma 0705 fold0 stage2 / mild symw fold0 stage2，SHA 对拍钉死常量）
- 命令（按执行序）：
  1. 负例（预期拒跑）：
     `MODMA_DATA_ROOT=/root/autodl-tmp/modma python3 /root/npi/models/latent-flow-v1/critical_node_ranking.py --line modma --mode main --fold 0 --subjects 1 --ckpt /root/npi/result/checkpoints/modma/latent_flow_0705/fold1/stage2/fold1_modma_0705_graph_mexhat.pt --out-dir /root/npi/outputs/10-30-critical-node-ranking/smoke/negative`
  2. modma 冒烟：
     `MODMA_DATA_ROOT=/root/autodl-tmp/modma python3 /root/npi/models/latent-flow-v1/critical_node_ranking.py --line modma --mode main --fold 0 --subjects 1 --out-dir /root/npi/outputs/10-30-critical-node-ranking`
  3. mild 冒烟：
     `python3 /root/npi/models/latent-flow-v1/critical_node_ranking.py --line mild --mode main --fold 0 --subjects 1 --out-dir /root/npi/outputs/10-30-critical-node-ranking`

### 结果

- 守卫负例（拒跑，exit=1，先于任何 GPU 工作，报错行照录）：
  `AssertionError: ckpt 文件名 fold 前缀与 --fold 锁定不符（折-权重错配防护）: fold1_modma_0705_graph_mexhat.pt 不以 fold0_ 开头（--fold 0）`
- modma 冒烟（fold0 × subject_0 = 02010013EOG.set_59_.mat，169 窗子采样 32 trial）：
  **0.79 min**，status=ok，**σ=0 恒等断言 PASS（max|x|=0.0e+00）**，**单调性违例 0**，
  逐源失败 0/128，top5@σ0.5=[87,38,88,25,31]；json 字段全（impact/ranking/
  impact_rank_channelwise/self_response 全 128×3，hparams 含 CRN/按线剂量外推/
  nan_to_num 注记，ckpt_sha256=5c6fbf31f1087447…=注册值，git e607716）；
  rank 一致性抽查过（argmax impact@0.5=ch87 ⇔ rank_ch[87,0.5]=1）。
- mild 冒烟（fold0 × subject_0 = s1(MDD)，30 trial 全量窗 500）：**0.78 min**，
  status=ok，σ=0 PASS（max|x|=0.0e+00），单调性违例 0，逐源失败 0/128，
  top5@σ0.5=[6,10,4,5,88]；norm SHA 对拍 fold0 注册值 d184e64772b36841… 过，
  manifest SHA 对拍过，exit-gnn 架构 strict 装载（FlowVelocityNetExitGNN，
  1,276,258 参数）。
- 离线模式代码路径检查（单被试弃用树 smoke/codepath_check/，不污染正式产物树）：
  summary/figs 双线全跑通（robustness json + top_nodes.md + 4×2 张 png，dpi150）；
  修正 boxplot `labels`→`tick_labels`（matplotlib 3.9+ 弃用告警清零）。
  单被试参考读数（不外推）：跨 σ Spearman ρ 0.93–0.99；I(0.7)/I(0.3) 中位
  modma 2.81 / mild 1.88（< 线性参照 5.444）；log-log 斜率 0.3→0.7 modma 1.35 /
  mild 1.49（<2）——比例性诊断在单例上已呈现「非 σ² 平方缩放」信号，待全量判读。
- 产物：`outputs/10-30-critical-node-ranking/modma/fold0/subject_0.json`、
  `outputs/10-30-critical-node-ranking/mild/fold0/subject_0.json`；
  负例与代码路径检查产物在 `outputs/10-30-critical-node-ranking/smoke/`。

### 结论

- 支持——守卫（fold 锁定/SHA/manifest/norm）全过且负例被拦；双线 main 链路通，
  σ=0 结构性恒等断言与单调性诊断（0 违例）符合 CRN 契约预期。
- 预算外推：0.79 min/例 × 53 例 ≈ 42 min（modma）+ 0.78 × 40 ≈ 31 min（mild）
  ⇒ 合计 ≈ 73 min ≤ 1.5 h（design §5 复核成立，S3/S4 可直接开跑）。
- 派生决策项：S5 全量后按「三层判读」读比例性诊断（单例斜率 <2 已提示非线性，
  需组级确认）；无需回改设计。

## Run 002 — S2 冒烟复跑逐位对拍（modma smoke 例）（2026-10-30）

### 目的与假设

- 断点续跑/复现契约验证：删除 Run 001 的 modma 冒烟 json 后同命令重跑，impact_matrix
  逐位比对（np.allclose rtol=0/atol=0，GPU 独占口径）；若漂移 → 复核独占性（nvidia-smi）
  后再跑一次，仍漂移则降 CPU 单线程（--device cpu + BLAS 线程=1）统计级对拍并如实登记
  （10-29 教训，AC6）。
- 假设：GPU 独占下同 seed 同代码逐位一致。

### 配置

- 代码：同 Run 001
- 参数：同 Run 001 命令 2（完全一致，仅删除 json 后重跑）
- 环境：同 Run 001（GPU 独占复核：nvidia-smi 0 MiB 起跑）
- 数据：同 Run 001
- 命令：
  `cp <modma subject_0.json> smoke/rerun_check/subject_0_run1.json`（留档对照）→ 删原 json →
  `MODMA_DATA_ROOT=/root/autodl-tmp/modma python3 /root/npi/models/latent-flow-v1/critical_node_ranking.py --line modma --mode main --fold 0 --subjects 1 --out-dir /root/npi/outputs/10-30-critical-node-ranking`
  → python np.allclose(rtol=0, atol=0) 对拍两份 impact_matrix

### 结果

- 跑前 GPU 独占复核：nvidia-smi 0 MiB；run1 json 留档
  `smoke/rerun_check/subject_0_run1.json` 后删原件重跑（同命令、同 seed、同代码）。
- run2：0.79 min，status=ok，σ=0 PASS，mono_viol=0，top5@σ0.5=[87,38,88,25,31]
  （与 run1 逐项一致）。
- **逐位对拍（GPU 独占口径）：impact_matrix np.allclose(rtol=0, atol=0) = True，
  max|Δ| = 0.0**；self_response 逐位一致；ranking 矩阵全等。
- 产物：`smoke/rerun_check/subject_0_run1.json`（run1 留档）+
  `modma/fold0/subject_0.json`（run2 正件）。

### 结论

- 支持——GPU 独占口径下同 seed 同代码**逐位复现**（无需 10-29 降 CPU 单线程路径）；
  断点续跑（json 已存在即跳过）行为正常。S3/S4 主跑沿用独占串行 + nohup 日志规约即可。

## Run 003 — modma 五折主跑（2026-10-30）

### 目的与假设

- 五折全量 53 例关键节点影响值/排序（σ=0.3/0.5/0.7），AC1 modma 侧。

### 配置

- 代码：`models/latent-flow-v1/critical_node_ranking.py`（本任务新增，Run 001 已冒烟）
- 参数：--line modma --mode main --fold {0..4} --out-dir /root/npi/outputs/10-30-critical-node-ranking；默认 σ=(0.3,0.5,0.7)/seed 42/n_steps 10/batch 32/bf16；ckpt = result/checkpoints/modma/latent_flow_0705/fold{k}/stage2/ 注册件（SHA 守卫）
- 环境：本机 conda base torch 2.8.0+cu128 @ RTX 4080S 独占（跑前 0 MiB）；`MODMA_DATA_ROOT=/root/autodl-tmp/modma`
- 数据：MODMA 128ch，fold test 10/10/11/11/11，每例 32 trial 子采样（seed 42）
- 命令：nohup 串行五折，日志 `outputs/10-30-critical-node-ranking/run/logs/modma_fold{k}.log`（与 Run 004 同链后台执行）

### 结果

- 指标：五折 json 计数 10/10/11/11/11（含 fold0 冒烟件复用）全对表；0 FAILED、0 源失败、
  σ=0 恒等 53/53、单调性违例 0；耗时 ≈0.79 min/例 × 53 ≈ 42 min（GPU 独占串行）。
- 产物：`outputs/10-30-critical-node-ranking/modma/fold{0..4}/subject_*.json`（53 件）
- 图表：见 Run 005。

### 结论

- 支持——modma 线五折全量执行契约全兑现（AC1 modma 侧 + AC6）；断点续跑消费冒烟件正常。

## Run 004 — mild 五折主跑（2026-10-30）

### 目的与假设

- 五折全量 40 例（8/折）关键节点影响值/排序，AC1 mild 侧；norm SHA 守卫按 fold 对拍。

### 配置

- 参数：--line mild --mode main --fold {0..4}；ckpt = result/checkpoints/mild/latent_flow_symw/fold{k}/ 注册件（SHA + norm SHA + manifest SHA 守卫）
- 环境：同 Run 003（GPU 独占串行；无特殊 env）
- 数据：Mild Depression 40 例 128ch，窗 500（obs 375/fc 125），每例 30 trial 全量
- 命令：nohup 串行五折，日志 `run/logs/mild_fold{k}.log`

### 结果

- 指标：五折 json 计数 8×5 全对表；0 FAILED、0 源失败、σ=0 恒等 40/40、单调性违例 1 行
  （fold1 s22(HC)，40×128 行中 0.02%，容差级如实登记）；耗时 ≈0.78 min/例 × 40 ≈ 31 min。
- 产物：`outputs/10-30-critical-node-ranking/mild/fold{0..4}/subject_*.json`（40 件）

### 结论

- 支持——mild 线五折全量执行契约全兑现（AC1 mild 侧）；norm SHA×5 守卫全过。

## Run 005 — 稳健性 summary + 图件 + REPORT（2026-10-30）

### 目的与假设

- AC3/AC4：两级三对 σ 一致性 + 比例性诊断 + 图件 + study 2 实验文档四段。

### 配置

- 命令：`critical_node_ranking.py --line {modma,mild} --mode {summary,figs} --out-dir …`（离线，无 GPU）

### 结果

- **关键数字**：modma（n=53）逐被试 ρ 0.3↔0.5 0.928±0.028 / 0.3↔0.7 0.867±0.053 /
  0.5↔0.7 0.961±0.018，组平均（primary）0.995/0.989/0.997，top-10 交叠 0.72/0.58/0.78
  （机会 0.078）；mild（n=40）逐被试 0.973/0.929/0.980，组 0.995/0.982/0.994。比例性：
  I(0.7)/I(0.3) 逐被试比值中位数的均值 modma 2.61 / mild 1.99（线性参照 5.44）、log-log
  斜率 0.3→0.7 1.463/1.440（线性⇒2）⇒ 亚二次压缩但排序稳 = 非平凡稳健。头部节点跨 σ
  锁定（modma 107/56/38、mild 88）。
- **执行期 bug 修复（如实登记）**：summary `diagnostics.impact_distribution` 原实现
  `imp_all=np.stack(...)` 后取 `imp_all[:, j]` 实取「第 j 个被试整张 128×3 矩阵」（三 σ 混合、
  σ 键实为被试号）；主判读口径（ρ/τ/top-K/组向量/比值/斜率）构建路径不受影响。修复为
  `np.concatenate(axis=0)` 后重生成双线 summary/figs，修复后分布与独立复算逐位一致
  （modma 中位 0.0154/0.0275/0.0408 随 σ 递增，与单调性诊断自洽）。
- 产物：`robustness/{modma,mild}_robustness.json` + `{line}_top_nodes.md`、`figs/` ×8 PNG、
  `REPORT.md`（四段，数字全部自 json 抄录）。

### 结论

- 支持——AC3/AC4 达成：关键节点排序对 σ∈{0.3,0.5,0.7} 稳健（逐被试 ρ≥0.69 全体、组平均
  ≥0.98、top-K 4–11× 机会水平），且比例性诊断排除 σ² 缩放平凡解释；判读边界（CRN 单实现/
  剂量外推/mild θ\* 前提）随档登记于 REPORT §4.4。

## 收尾登记

- **本轮零写入 result/**（trellis-check 事实核验：result/ 当日变更均为 00:03–00:37 的
  10-29-result-ckpt-reg 注册件，早于本任务首个产物；用户拍板归档 = 仅 outputs）。
- trellis-check（2026-10-30）：AC1–AC3/AC5/AC6 过，AC4 有条件过（两个 Kendall 单元格
  0.872→0.838 / 0.900→0.888 + 比值 2.62→2.61 + fold0→fold1 折标已改，独立复算确认科学
  结论不受影响）；路径隔离/复现登记/无空块均核过。
